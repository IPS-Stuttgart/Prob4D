"""Crash-safe, resumable orchestration for the low-level MotionCrafter adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .data import PredictionWindow
from .lineage import motioncrafter_temporal_lineage_manifest
from .motioncrafter import (
    MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
    MotionCrafterAdapter,
    MotionCrafterRunConfig,
    motioncrafter_seed_for_call,
    validate_motioncrafter_seed_schedule,
)
from .motioncrafter_integrity import (
    MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA,
    MOTIONCRAFTER_MANIFEST_FILENAME,
    MOTIONCRAFTER_PROGRESS_FILENAME,
    _artifact_descriptor,
    _atomic_write_json,
    _atomic_write_npz,
    _git_provenance,
    _json_config,
    _load_progress,
    _new_progress,
    _prediction_window_payload,
    _run_spec,
    _sha256_file,
    _sha256_json,
    _validate_run_spec,
    _video_descriptor,
    resolve_motioncrafter_member,
    verify_motioncrafter_prediction_manifest,
)


class _AdapterProtocol(Protocol):
    config: MotionCrafterRunConfig

    def read_video(self, video_path: Path | None = None) -> Any: ...

    def infer(
        self,
        frames: Any,
        *,
        window_size: int,
        overlap: int,
        seed: int,
    ) -> tuple: ...

    @staticmethod
    def _arrays(results: tuple) -> dict[str, np.ndarray]: ...


AdapterFactory = Callable[[MotionCrafterRunConfig], _AdapterProtocol]
ProvenanceProvider = Callable[[Path], Mapping[str, object]]


def _runner_descriptor() -> dict[str, object]:
    path = Path(__file__).resolve()
    return {
        "module": "prob4d.motioncrafter_runner",
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


class SafeMotionCrafterRunner:
    """Produce an atomic prediction bundle and resume only validated products."""

    def __init__(
        self,
        config: MotionCrafterRunConfig,
        *,
        adapter_factory: AdapterFactory = MotionCrafterAdapter,
        provenance_provider: ProvenanceProvider = _git_provenance,
        allow_dirty_upstream: bool = False,
    ) -> None:
        self.config = config
        self.adapter_factory = adapter_factory
        self.provenance_provider = provenance_provider
        self.allow_dirty_upstream = bool(allow_dirty_upstream)

    @staticmethod
    def _artifact_complete(progress: Mapping[str, Any], relative_path: str) -> bool:
        artifacts = progress.get("artifacts")
        return isinstance(artifacts, Mapping) and relative_path in artifacts

    @staticmethod
    def _record_artifact(
        progress: dict[str, Any],
        progress_path: Path,
        path: Path,
        *,
        output_root: Path,
        kind: str,
    ) -> None:
        descriptor = _artifact_descriptor(path, root=output_root, kind=kind)
        progress["artifacts"][str(descriptor["path"])] = descriptor
        _atomic_write_json(progress_path, progress)

    def run(self, *, resume: bool = False) -> Path:
        config = self.config
        actual_video_path = config.video_path.resolve()
        output = config.output_directory.resolve()
        manifest_path = output / MOTIONCRAFTER_MANIFEST_FILENAME
        progress_path = output / MOTIONCRAFTER_PROGRESS_FILENAME

        provenance = dict(self.provenance_provider(config.upstream_root.resolve()))
        clean = provenance.get("clean")
        if not isinstance(clean, bool):
            raise ValueError("MotionCrafter provenance clean flag must be Boolean")
        if not clean and not self.allow_dirty_upstream:
            raise ValueError("MotionCrafter checkout is dirty; refuse claim-bearing inference")

        run_spec = _run_spec(
            config,
            video_descriptor=_video_descriptor(actual_video_path),
            upstream_provenance=provenance,
            runner_descriptor=_runner_descriptor(),
        )
        _validate_run_spec(run_spec)
        run_spec_sha256 = _sha256_json(run_spec)

        if output.exists() and not output.is_dir():
            raise ValueError("MotionCrafter output path exists and is not a directory")
        if output.exists() and any(output.iterdir()) and not resume:
            raise ValueError(
                "output directory is non-empty; use --resume only for the identical recorded run"
            )
        output.mkdir(parents=True, exist_ok=True)

        if resume and manifest_path.is_file():
            verify_motioncrafter_prediction_manifest(
                manifest_path,
                verify_hashes=True,
                expected_run_spec_sha256=run_spec_sha256,
            )
            return manifest_path
        if resume:
            if not progress_path.is_file():
                raise ValueError("resume requested but no MotionCrafter progress journal exists")
            progress = _load_progress(
                progress_path,
                expected_run_spec_sha256=run_spec_sha256,
                verify_hashes=True,
            )
        else:
            progress = _new_progress(run_spec, run_spec_sha256)
            _atomic_write_json(progress_path, progress)

        adapter = self.adapter_factory(config)
        frames = adapter.read_video(actual_video_path)
        num_frames = int(frames.shape[0])
        source_stop = (
            config.frame_start + config.frame_stride * num_frames
            if config.frame_stop is None
            else config.frame_stop
        )
        frame_indices = np.arange(
            config.frame_start,
            source_stop,
            config.frame_stride,
            dtype=np.int64,
        )[:num_frames]
        if frame_indices.size != num_frames:
            raise ValueError("selected source-frame metadata does not match decoded frame count")

        stride = config.window_size - config.overlap
        starts = list(range(0, max(1, num_frames - config.window_size + 1), stride))
        final_start = max(0, num_frames - config.window_size)
        if final_start not in starts:
            starts.append(final_start)

        window_plan: list[dict[str, object]] = []
        for index, start in enumerate(sorted(set(starts))):
            stop = min(start + config.window_size, num_frames)
            window_id = f"window_{index:04d}"
            start_frame = int(frame_indices[start])
            stop_frame = int(frame_indices[stop - 1]) + config.frame_stride
            window_plan.append(
                {
                    "window_id": window_id,
                    "start": start,
                    "stop": stop,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                    "path": f"windows/{window_id}.npz",
                    "call_id": (
                        f"overlap-window:{window_id}:{start_frame}:{stop_frame}"
                    ),
                }
            )

        expected_paths = {
            "baseline_disjoint.npz",
            "baseline_latent_linear.npz",
            *(str(item["path"]) for item in window_plan),
        }
        unknown_progress = set(progress["artifacts"]) - expected_paths
        if unknown_progress:
            raise ValueError(
                f"progress journal contains stale artifacts: {sorted(unknown_progress)}"
            )
        allowed_files = {
            *expected_paths,
            MOTIONCRAFTER_PROGRESS_FILENAME,
            MOTIONCRAFTER_MANIFEST_FILENAME,
        }
        existing_files = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        unexpected_files = existing_files - allowed_files
        if unexpected_files:
            raise ValueError(
                "MotionCrafter output contains unbound files: "
                f"{sorted(unexpected_files)}"
            )

        seed_schedule: list[dict[str, object]] = []

        def seed_record(
            call_id: str,
            *,
            product: str,
            window_id: str | None = None,
            source_frame_start: int | None = None,
            source_frame_stop_exclusive: int | None = None,
        ) -> int:
            effective_seed = motioncrafter_seed_for_call(
                config.seed,
                policy=config.seed_policy,
                call_id=call_id,
            )
            record: dict[str, object] = {
                "call_id": call_id,
                "product": product,
                "effective_seed": effective_seed,
            }
            if window_id is not None:
                record["window_id"] = window_id
            if source_frame_start is not None:
                record["source_frame_start"] = source_frame_start
            if source_frame_stop_exclusive is not None:
                record["source_frame_stop_exclusive"] = source_frame_stop_exclusive
            seed_schedule.append(record)
            return effective_seed

        disjoint_relative = "baseline_disjoint.npz"
        disjoint_seed = seed_record("baseline-disjoint", product="disjoint_baseline")
        disjoint_path = output / disjoint_relative
        if not self._artifact_complete(progress, disjoint_relative):
            arrays = adapter._arrays(
                adapter.infer(
                    frames,
                    window_size=config.window_size,
                    overlap=0,
                    seed=disjoint_seed,
                )
            )
            arrays["frame_indices"] = frame_indices
            _atomic_write_npz(disjoint_path, arrays)
            self._record_artifact(
                progress,
                progress_path,
                disjoint_path,
                output_root=output,
                kind="disjoint_baseline",
            )

        latent_relative = "baseline_latent_linear.npz"
        latent_seed = seed_record(
            "baseline-latent-linear",
            product="latent_linear_baseline",
        )
        latent_path = output / latent_relative
        if not self._artifact_complete(progress, latent_relative):
            arrays = adapter._arrays(
                adapter.infer(
                    frames,
                    window_size=config.window_size,
                    overlap=config.overlap,
                    seed=latent_seed,
                )
            )
            arrays["frame_indices"] = frame_indices
            _atomic_write_npz(latent_path, arrays)
            self._record_artifact(
                progress,
                progress_path,
                latent_path,
                output_root=output,
                kind="latent_linear_baseline",
            )

        manifest_windows: list[dict[str, object]] = []
        for item in window_plan:
            window_id = str(item["window_id"])
            start = int(item["start"])
            stop = int(item["stop"])
            start_frame = int(item["start_frame"])
            stop_frame = int(item["stop_frame"])
            relative_path = str(item["path"])
            effective_seed = seed_record(
                str(item["call_id"]),
                product="independently_decoded_overlap_window",
                window_id=window_id,
                source_frame_start=start_frame,
                source_frame_stop_exclusive=stop_frame,
            )
            path = resolve_motioncrafter_member(
                output,
                relative_path,
                name=f"window {window_id!r} path",
            )
            if not self._artifact_complete(progress, relative_path):
                arrays = adapter._arrays(
                    adapter.infer(
                        frames[start:stop],
                        window_size=stop - start,
                        overlap=0,
                        seed=effective_seed,
                    )
                )
                window = PredictionWindow(
                    window_id=window_id,
                    frame_indices=frame_indices[start:stop],
                    point_map=arrays["point_map"],
                    valid_mask=arrays["valid_mask"],
                    scene_flow=arrays.get("scene_flow"),
                    deform_mask=arrays.get("deform_mask"),
                )
                _atomic_write_npz(path, _prediction_window_payload(window))
                self._record_artifact(
                    progress,
                    progress_path,
                    path,
                    output_root=output,
                    kind="independently_decoded_overlap_window",
                )
            manifest_windows.append(
                {
                    "window_id": window_id,
                    "path": relative_path,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                }
            )

        ordered_paths = [
            disjoint_relative,
            latent_relative,
            *(str(item["path"]) for item in window_plan),
        ]
        progress = _load_progress(
            progress_path,
            expected_run_spec_sha256=run_spec_sha256,
            verify_hashes=True,
        )
        manifest: dict[str, Any] = {
            "format_version": 1,
            "video_path": str(actual_video_path),
            "motioncrafter_commit": provenance["commit"],
            "config": _json_config(config),
            "stochastic_seed_schedule": {
                "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
                "policy": config.seed_policy,
                "root_seed": config.seed,
                "calls": seed_schedule,
                "interpretation": (
                    "legacy-common exactly preserves the historical common seed; "
                    "derived-per-call deterministically assigns a source-bound seed to "
                    "each inference call without claiming statistical independence"
                ),
            },
            "temporal_lineage": motioncrafter_temporal_lineage_manifest(
                window_size=config.window_size,
                overlap=config.overlap,
            ),
            "overlap_windows": manifest_windows,
            "disjoint_baseline": disjoint_relative,
            "latent_linear_baseline": latent_relative,
            "artifact_integrity": {
                "schema": MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA,
                "run_spec": run_spec,
                "run_spec_sha256": run_spec_sha256,
                "members": [progress["artifacts"][path] for path in ordered_paths],
            },
        }
        manifest["config"]["video_path"] = str(actual_video_path)
        manifest["config"]["output_directory"] = str(output)
        validate_motioncrafter_seed_schedule(manifest)
        _atomic_write_json(manifest_path, manifest)
        verification = verify_motioncrafter_prediction_manifest(
            manifest_path,
            verify_hashes=True,
            expected_run_spec_sha256=run_spec_sha256,
        )
        progress["status"] = "complete"
        progress["manifest"] = _artifact_descriptor(
            manifest_path,
            root=output,
            kind="prediction_manifest",
        )
        progress["verification"] = verification
        _atomic_write_json(progress_path, progress)
        return manifest_path
