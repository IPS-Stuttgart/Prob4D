#!/usr/bin/env python3
"""Run and compare the frozen issue-50 real-bundle storage profile.

The benchmark is deliberately an engineering evidence tool.  It uses one
pre-registered real source bundle and executes the same estimator in separate
fresh processes for verified eager-NPZ and verified mmap-NPY loading.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

LOCK_SCHEMA = "prob4d.issue50-real-bundle-profile-lock"
LOCK_VERSION = 1
ARM_SCHEMA = "prob4d.issue50-real-bundle-profile-arm"
ARM_VERSION = 1
COMPARISON_SCHEMA = "prob4d.issue50-real-bundle-profile-comparison"
COMPARISON_VERSION = 1
_MIB = 1024 * 1024


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value!r} in {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_lock(path: Path) -> dict[str, Any]:
    value = _strict_json(path)
    if value.get("schema") != LOCK_SCHEMA or value.get("version") != LOCK_VERSION:
        raise ValueError("unsupported issue-50 profile lock")
    supplied = str(value.get("lock_id", ""))
    portable = dict(value)
    portable.pop("lock_id", None)
    expected = _sha256_bytes(_canonical_json(portable))
    if supplied != expected:
        raise ValueError(
            f"issue-50 lock ID mismatch: supplied={supplied!r}, expected={expected!r}"
        )

    inference = value.get("inference")
    if not isinstance(inference, Mapping):
        raise ValueError("lock inference must be an object")
    start = int(inference["frame_start"])
    stop = int(inference["frame_stop_exclusive"])
    window_size = int(inference["window_size"])
    overlap = int(inference["overlap"])
    if start < 0 or stop <= start or not 0 <= overlap < window_size:
        raise ValueError("lock frame/window configuration is invalid")
    plan = inference.get("window_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("lock window_plan must be a nonempty array")
    expected_plan: list[dict[str, Any]] = []
    stride = window_size - overlap
    starts = list(range(start, max(start + 1, stop - window_size + 1), stride))
    final_start = max(start, stop - window_size)
    if final_start not in starts:
        starts.append(final_start)
    for index, window_start in enumerate(sorted(set(starts))):
        window_stop = min(window_start + window_size, stop)
        expected_plan.append(
            {
                "window_id": f"window_{index:04d}",
                "source_frame_start": window_start,
                "source_frame_stop_exclusive": window_stop,
            }
        )
    if plan != expected_plan:
        raise ValueError(
            f"lock window_plan changed: declared={plan!r}, expected={expected_plan!r}"
        )
    return value


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("repository HEAD is not a full Git revision")
    return revision


def _current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class _RssSampler:
    def __init__(self, *, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        initial = _current_rss_bytes()
        self._phase_peak = initial
        self._total_peak = initial
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def reset_phase(self) -> int:
        value = _current_rss_bytes()
        with self._lock:
            self._phase_peak = value
            self._total_peak = max(self._total_peak, value)
        return value

    def phase_peak(self) -> int:
        value = _current_rss_bytes()
        with self._lock:
            self._phase_peak = max(self._phase_peak, value)
            self._total_peak = max(self._total_peak, value)
            return self._phase_peak

    def total_peak(self) -> int:
        with self._lock:
            return max(self._total_peak, _current_rss_bytes())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            value = _current_rss_bytes()
            with self._lock:
                self._phase_peak = max(self._phase_peak, value)
                self._total_peak = max(self._total_peak, value)


@contextmanager
def _phase(
    name: str,
    *,
    sampler: _RssSampler,
    records: dict[str, dict[str, Any]],
) -> Iterator[None]:
    if name in records:
        raise RuntimeError(f"phase {name!r} was measured twice")
    gc.collect()
    start_rss = sampler.reset_phase()
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    end_rss = _current_rss_bytes()
    records[name] = {
        "wall_seconds": elapsed,
        "rss_start_bytes": start_rss,
        "rss_end_bytes": end_rss,
        "peak_rss_bytes": sampler.phase_peak(),
    }


def _directory_bytes(path: Path) -> int:
    total = 0
    for member in path.rglob("*"):
        if member.is_file() and not member.is_symlink():
            total += member.stat().st_size
    return total


def _update_array_digest(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(_canonical_json(list(array.shape)) + b"\0")
    if array.flags.c_contiguous:
        view = memoryview(array).cast("B")
        for start in range(0, len(view), 4 * 1024 * 1024):
            digest.update(view[start : start + 4 * 1024 * 1024])
        return
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
        return
    for index in range(array.shape[0]):
        digest.update(np.ascontiguousarray(array[index]).tobytes())


def _array_record(label: str, value: np.ndarray) -> dict[str, Any]:
    digest = hashlib.sha256()
    _update_array_digest(digest, label, np.asarray(value))
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
        "sha256": digest.hexdigest(),
    }


def _window_record(window: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "frame_indices": _array_record("frame_indices", window.frame_indices),
        "point_map": _array_record("point_map", window.point_map),
        "valid_mask": _array_record("valid_mask", window.valid_mask),
    }
    if window.scene_flow is not None:
        fields["scene_flow"] = _array_record("scene_flow", window.scene_flow)
        fields["deform_mask"] = _array_record("deform_mask", window.deform_mask)
    if window.ray_directions is not None:
        fields["ray_directions"] = _array_record(
            "ray_directions", window.ray_directions
        )
    digest = hashlib.sha256()
    digest.update(_canonical_json(fields))
    return {
        "window_id": window.window_id,
        "start_frame": int(window.start_frame),
        "stop_frame_exclusive": int(window.stop_frame),
        "shape": list(window.shape),
        "dense_storage_dtype": window.dense_storage_dtype,
        "fields": fields,
        "content_sha256": digest.hexdigest(),
    }


def _bundle_record(bundle: Any) -> dict[str, Any]:
    overlap = [_window_record(value) for value in bundle.overlap_windows]
    disjoint = _window_record(bundle.disjoint_baseline)
    latent = _window_record(bundle.latent_linear_baseline)
    record = {
        "overlap_windows": overlap,
        "disjoint_baseline": disjoint,
        "latent_linear_baseline": latent,
    }
    record["content_sha256"] = _sha256_bytes(_canonical_json(record))
    return record


def _alignment_record(value: Any) -> dict[str, Any]:
    result = value.result
    return {
        "reference_id": value.reference_id,
        "moving_id": value.moving_id,
        "common_frames": [int(item) for item in value.common_frames],
        "transform": {
            "scale": float(result.transform.scale),
            "rotation": np.asarray(result.transform.rotation).tolist(),
            "translation": np.asarray(result.transform.translation).tolist(),
        },
        "covariance": np.asarray(result.covariance).tolist(),
        "residual_rms": float(result.residual_rms),
        "inlier_fraction": float(result.inlier_fraction),
        "num_correspondences": int(result.num_correspondences),
        "covariance_method": result.covariance_method,
        "num_covariance_clusters": int(result.num_covariance_clusters),
        "information_rank": int(result.information_rank),
        "information_condition": float(result.information_condition),
        "covariance_calibration_id": result.covariance_calibration_id,
        "covariance_fallback": result.covariance_fallback,
    }


def _gauge_record(estimates: Mapping[str, Any]) -> dict[str, Any]:
    return {
        window_id: {
            "transform": {
                "scale": float(estimate.global_from_local.scale),
                "rotation": np.asarray(estimate.global_from_local.rotation).tolist(),
                "translation": np.asarray(
                    estimate.global_from_local.translation
                ).tolist(),
            },
            "covariance": np.asarray(estimate.covariance).tolist(),
        }
        for window_id, estimate in sorted(estimates.items())
    }


def _disagreement_record(evidence: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for window_id, value in sorted(evidence.items()):
        record[window_id] = {
            "parallel_sum": _array_record(
                f"{window_id}:parallel_sum", value.parallel_sum
            ),
            "lateral_sum": _array_record(
                f"{window_id}:lateral_sum", value.lateral_sum
            ),
            "count": _array_record(f"{window_id}:count", value.count),
        }
    return {
        "windows": record,
        "content_sha256": _sha256_bytes(_canonical_json(record)),
    }


def _uncertainty_record(values: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for window_id, value in sorted(values.items()):
        record[window_id] = {
            "ray_directions": _array_record(
                f"{window_id}:ray_directions", value.ray_directions
            ),
            "parallel_variance": _array_record(
                f"{window_id}:parallel_variance", value.parallel_variance
            ),
            "lateral_variance": _array_record(
                f"{window_id}:lateral_variance", value.lateral_variance
            ),
        }
    return {
        "windows": record,
        "content_sha256": _sha256_bytes(_canonical_json(record)),
    }


def _fused_record(value: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "frame_indices": _array_record("frame_indices", value.frame_indices),
        "point_map": _array_record("point_map", value.point_map),
        "valid_mask": _array_record("valid_mask", value.valid_mask),
        "point_covariance": _array_record(
            "point_covariance", value.point_covariance
        ),
        "contributors": _array_record("contributors", value.contributors),
    }
    if value.scene_flow is not None:
        fields["scene_flow"] = _array_record("scene_flow", value.scene_flow)
        fields["deform_mask"] = _array_record("deform_mask", value.deform_mask)
        fields["flow_covariance"] = _array_record(
            "flow_covariance", value.flow_covariance
        )
    return {
        "fields": fields,
        "content_sha256": _sha256_bytes(_canonical_json(fields)),
    }


def _validate_bundle_against_lock(
    bundle: Any,
    lock: Mapping[str, Any],
    *,
    backend: str,
    input_path: Path,
) -> dict[str, Any]:
    inference = lock["inference"]
    plan = inference["window_plan"]
    actual_plan = [
        {
            "window_id": value.window_id,
            "source_frame_start": int(value.start_frame),
            "source_frame_stop_exclusive": int(value.stop_frame),
        }
        for value in bundle.overlap_windows
    ]
    if actual_plan != plan:
        raise ValueError(f"{backend} window plan differs from lock: {actual_plan!r}")
    expected_shape = [
        int(inference["window_size"]),
        int(inference["height"]),
        int(inference["width"]),
    ]
    for value in bundle.overlap_windows:
        if list(value.shape) != expected_shape:
            raise ValueError(
                f"{backend} overlap window shape {value.shape!r} differs from lock"
            )
        if value.dense_storage_dtype != lock["storage"]["dense_storage_dtype"]:
            raise ValueError(f"{backend} storage dtype differs from lock")
    baseline_shape = [
        int(inference["frame_stop_exclusive"]) - int(inference["frame_start"]),
        int(inference["height"]),
        int(inference["width"]),
    ]
    for value in (bundle.disjoint_baseline, bundle.latent_linear_baseline):
        if list(value.shape) != baseline_shape:
            raise ValueError(f"{backend} baseline shape {value.shape!r} differs from lock")
        if value.dense_storage_dtype != lock["storage"]["dense_storage_dtype"]:
            raise ValueError(f"{backend} baseline dtype differs from lock")

    metadata = bundle.metadata
    if str(metadata.get("motioncrafter_commit")) != lock["motioncrafter_source"][
        "revision"
    ]:
        raise ValueError("MotionCrafter source revision differs from lock")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prediction metadata lacks config")
    expected_config = {
        "model_type": inference["model_type"],
        "height": inference["height"],
        "width": inference["width"],
        "window_size": inference["window_size"],
        "overlap": inference["overlap"],
        "decode_chunk_size": inference["decode_chunk_size"],
        "seed": inference["seed"],
        "seed_policy": inference["seed_policy"],
        "low_memory_usage": inference["low_memory_usage"],
        "frame_start": inference["frame_start"],
        "frame_stop": inference["frame_stop_exclusive"],
        "frame_stride": inference["frame_stride"],
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"prediction config {key!r} differs from lock: "
                f"{config.get(key)!r} != {expected!r}"
            )
    source_sha = (
        _sha256_file(input_path)
        if backend == "eager_npz"
        else str(metadata["prediction_execution_store"]["source_manifest_sha256"])
    )
    if backend == "mmap_npy":
        execution_store = metadata.get("prediction_execution_store")
        if not isinstance(execution_store, Mapping):
            raise ValueError("mmap backend lacks execution-store metadata")
        if execution_store.get("verify_hashes") is not True:
            raise ValueError("mmap backend did not perform full hash verification")
        if execution_store.get("dense_storage_dtype") != lock["storage"][
            "dense_storage_dtype"
        ]:
            raise ValueError("mmap store dtype differs from lock")
    return {
        "source_manifest_sha256": source_sha,
        "input_path": str(input_path),
        "input_persistent_bytes": _directory_bytes(
            input_path.parent if backend == "eager_npz" else input_path
        ),
        "model_source_set_sha256": config.get("model_source_set_sha256"),
        "artifact_run_spec_sha256": (
            metadata.get("artifact_integrity", {}).get("run_spec_sha256")
            if isinstance(metadata.get("artifact_integrity"), Mapping)
            else None
        ),
    }


def _host_record() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "pid": os.getpid(),
    }


def _run_arm(arguments: argparse.Namespace) -> int:
    lock = _load_lock(arguments.lock)
    repository_root = Path(__file__).resolve().parents[2]
    execution_revision = _git_revision(repository_root)
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"arm output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    phases: dict[str, dict[str, Any]] = {}
    sampler = _RssSampler()
    sampler.start()
    process_started = time.perf_counter()
    try:
        with _phase("source_loading", sampler=sampler, records=phases):
            if arguments.backend == "eager_npz":
                from prob4d.io import load_prediction_bundle

                bundle = load_prediction_bundle(
                    arguments.input,
                    dense_storage_dtype=lock["storage"]["dense_storage_dtype"],
                )
            else:
                from prob4d.prediction_store import load_prediction_bundle_store

                bundle = load_prediction_bundle_store(
                    arguments.input,
                    verify_hashes=True,
                )
            input_identity = _validate_bundle_against_lock(
                bundle,
                lock,
                backend=arguments.backend,
                input_path=arguments.input.resolve(),
            )

        digest_started = time.perf_counter()
        source_record = _bundle_record(bundle)
        source_digest_seconds = time.perf_counter() - digest_started

        from prob4d.alignment import align_windows
        from prob4d.gauge import (
            FixedLagGaugeSmoother,
            RelativeGaugeConstraint,
            SequentialGaugeEstimator,
        )

        with _phase("gauge_estimation", sampler=sampler, records=phases):
            alignments = []
            for moving_index, moving in enumerate(bundle.overlap_windows):
                for reference in bundle.overlap_windows[:moving_index]:
                    if reference.common_frames(moving).size:
                        alignments.append(
                            align_windows(
                                reference,
                                moving,
                                max_correspondences=int(
                                    lock["estimator"]["alignment"][
                                        "maximum_correspondences"
                                    ]
                                ),
                                seed=moving_index,
                                covariance_cluster_size=int(
                                    lock["estimator"]["alignment"][
                                        "covariance_cluster_size"
                                    ]
                                ),
                            )
                        )
            constraints = [
                RelativeGaugeConstraint.from_window_alignment(value)
                for value in alignments
            ]
            ordered_ids = [value.window_id for value in bundle.overlap_windows]
            sequential = SequentialGaugeEstimator(
                covariance_intersection_grid_size=int(
                    lock["estimator"]["gauge"][
                        "covariance_intersection_grid_size"
                    ]
                )
            ).estimate(ordered_ids, constraints)
            smoothed = FixedLagGaugeSmoother(
                lag=int(lock["estimator"]["gauge"]["fixed_lag"])
            ).smooth(ordered_ids, sequential, constraints)

        alignment_records = [_alignment_record(value) for value in alignments]
        gauge_records = {
            "sequential": _gauge_record(sequential),
            "fixed_lag": _gauge_record(smoothed),
        }
        gauge_digest = _sha256_bytes(
            _canonical_json(
                {
                    "alignments": alignment_records,
                    "gauges": gauge_records,
                }
            )
        )

        from prob4d.uncertainty import DepthDisagreementModel, accumulate_disagreement

        windows = {value.window_id: value for value in bundle.overlap_windows}
        model_configuration = dict(
            lock["estimator"]["uncertainty"]["depth_disagreement_model"]
        )
        with _phase(
            "overlap_disagreement",
            sampler=sampler,
            records=phases,
        ):
            disagreement = accumulate_disagreement(windows, alignments)
            model = DepthDisagreementModel(**model_configuration)
            point_uncertainties = {
                window_id: model.predict(window, disagreement[window_id])
                for window_id, window in windows.items()
            }

        digest_started = time.perf_counter()
        disagreement_record = _disagreement_record(disagreement)
        uncertainty_record = _uncertainty_record(point_uncertainties)
        disagreement_digest_seconds = time.perf_counter() - digest_started
        del disagreement
        gc.collect()

        from prob4d.fusion import fuse_windows

        gauges = {
            window_id: value.global_from_local for window_id, value in smoothed.items()
        }
        gauge_covariances = {
            window_id: value.covariance for window_id, value in smoothed.items()
        }
        with _phase("dense_fusion", sampler=sampler, records=phases):
            fused = fuse_windows(
                bundle.overlap_windows,
                gauges,
                point_uncertainties,
                method=lock["estimator"]["fusion"]["method"],
                flow_uncertainties=None,
                gauge_covariances=gauge_covariances,
                fusion_tile_size=int(lock["estimator"]["fusion"]["tile_size"]),
            )

        digest_started = time.perf_counter()
        fused_record = _fused_record(fused)
        fused_digest_seconds = time.perf_counter() - digest_started

        from prob4d.io import save_fused_prediction

        export_path = output / "provider-export.npz"
        with _phase("provider_export", sampler=sampler, records=phases):
            export_metadata = save_fused_prediction(
                export_path,
                fused,
                method_id=lock["provider_export"]["method_id"],
                fusion_method=lock["provider_export"]["fusion_method"],
                include_covariance=bool(lock["provider_export"]["include_covariance"]),
                metadata={
                    "issue": 50,
                    "profile_lock_id": lock["lock_id"],
                    "estimator_base_revision": lock["estimator_base_revision"],
                    "execution_revision": execution_revision,
                    "uncertainty_calibration": lock["estimator"]["uncertainty"][
                        "calibration_mode"
                    ],
                    "evaluation_role": "engineering-memory-and-parity-evidence",
                },
                compressed=bool(lock["provider_export"]["compressed"]),
            )
        export_sha256 = _sha256_file(export_path)

        from prob4d.metrics import TruthSequence, evaluate_sequence

        baseline = bundle.disjoint_baseline
        with _phase(
            "provider_evaluation",
            sampler=sampler,
            records=phases,
        ):
            truth = TruthSequence(
                frame_indices=baseline.frame_indices,
                point_map=baseline.point_map,
                valid_mask=baseline.valid_mask,
                scene_flow=baseline.scene_flow,
                deform_mask=baseline.deform_mask,
            )
            metrics = evaluate_sequence(
                fused,
                truth,
                boundary_frames=[
                    int(value) for value in lock["evaluation"]["boundary_frames"]
                ],
                align_scale_translation=bool(
                    lock["evaluation"]["align_scale_translation"]
                ),
                evaluation_chunk_size=int(lock["evaluation"]["chunk_size"]),
            )
            metrics_record = metrics.to_dict()
            del truth
        metrics_digest = _sha256_bytes(_canonical_json(metrics_record))

        elapsed = time.perf_counter() - process_started
        report = {
            "schema": ARM_SCHEMA,
            "version": ARM_VERSION,
            "backend": arguments.backend,
            "profile_lock_id": lock["lock_id"],
            "estimator_base_revision": lock["estimator_base_revision"],
            "repository_execution_revision": execution_revision,
            "input_identity": input_identity,
            "host": _host_record(),
            "configuration": {
                "dense_storage_dtype": lock["storage"]["dense_storage_dtype"],
                "alignment": lock["estimator"]["alignment"],
                "gauge": lock["estimator"]["gauge"],
                "uncertainty": lock["estimator"]["uncertainty"],
                "fusion": lock["estimator"]["fusion"],
                "provider_export": lock["provider_export"],
                "evaluation": lock["evaluation"],
            },
            "phases": phases,
            "memory": {
                "maximum_process_rss_bytes": max(
                    sampler.total_peak(), _maximum_rss_bytes()
                ),
                "retained_dense_vector_bytes": int(
                    bundle.dense_storage_summary()["retained_bytes"]
                ),
                "float64_equivalent_dense_vector_bytes": int(
                    bundle.dense_storage_summary()["float64_equivalent_bytes"]
                ),
            },
            "timing": {
                "total_wall_seconds": elapsed,
                "source_digest_seconds": source_digest_seconds,
                "disagreement_and_uncertainty_digest_seconds": (
                    disagreement_digest_seconds
                ),
                "fused_digest_seconds": fused_digest_seconds,
            },
            "semantic_outputs": {
                "source_bundle": source_record,
                "alignments": alignment_records,
                "gauges": gauge_records,
                "gauge_content_sha256": gauge_digest,
                "disagreement": disagreement_record,
                "uncertainty": uncertainty_record,
                "fused": fused_record,
                "calibration": {
                    "mode": lock["estimator"]["uncertainty"]["calibration_mode"],
                    "model_parameters": model_configuration,
                    "fitted_artifact": None,
                },
                "provider_export": {
                    "path": str(export_path),
                    "bytes": export_path.stat().st_size,
                    "file_sha256": export_sha256,
                    "metadata": export_metadata.to_dict(),
                    "content_sha256": fused_record["content_sha256"],
                },
                "provider_evaluation": {
                    "reference": lock["evaluation"]["reference"],
                    "metrics": metrics_record,
                    "content_sha256": metrics_digest,
                },
            },
            "claim_boundary": (
                "Hardware-specific engineering memory and numerical-parity "
                "evidence on one frozen calibration video. The disjoint baseline "
                "is used only as an internal common evaluation reference. This "
                "report does not establish reconstruction accuracy, uncertainty "
                "calibration, provider competence, BayesianPhysTwin benefit, "
                "Causal4D benefit, deployment safety, or state of the art."
            ),
        }
        report_path = output / "arm-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(report_path)
        return 0
    finally:
        sampler.close()


def _require_arm(value: Mapping[str, Any], *, backend: str) -> None:
    if (
        value.get("schema") != ARM_SCHEMA
        or value.get("version") != ARM_VERSION
        or value.get("backend") != backend
    ):
        raise ValueError(f"invalid {backend} arm report")


def _iter_numeric_pairs(
    left: Any,
    right: Any,
    *,
    path: str = "$",
) -> Iterator[tuple[str, float, float]]:
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            raise ValueError(f"Boolean value differs at {path}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        yield path, float(left), float(right)
        return
    if isinstance(left, str) or left is None:
        if left != right:
            raise ValueError(f"value differs at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(f"mapping fields differ at {path}")
        for key in sorted(left):
            yield from _iter_numeric_pairs(left[key], right[key], path=f"{path}.{key}")
        return
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            raise ValueError(f"sequence length differs at {path}")
        for index, (left_value, right_value) in enumerate(
            zip(left, right, strict=True)
        ):
            yield from _iter_numeric_pairs(
                left_value, right_value, path=f"{path}[{index}]"
            )
        return
    if left != right:
        raise ValueError(f"value differs at {path}: {left!r} != {right!r}")


def _maximum_numeric_difference(left: Any, right: Any) -> float:
    maximum = 0.0
    for path, left_value, right_value in _iter_numeric_pairs(left, right):
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            if left_value != right_value:
                raise ValueError(f"non-finite numeric value differs at {path}")
            continue
        maximum = max(maximum, abs(left_value - right_value))
    return maximum


def _compare_export_payloads(
    left_path: Path,
    right_path: Path,
    *,
    chunk_elements: int,
) -> dict[str, Any]:
    categories: dict[str, float] = {
        "point": 0.0,
        "flow": 0.0,
        "covariance": 0.0,
    }
    exact_fields: dict[str, bool] = {}
    category_for_field = {
        "point_map": "point",
        "scene_flow": "flow",
        "point_covariance_packed": "covariance",
        "flow_covariance_packed": "covariance",
    }
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            raise ValueError("provider export fields differ between backends")
        for field in sorted(left.files):
            left_value = left[field]
            right_value = right[field]
            if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
                raise ValueError(f"provider export field {field!r} contract differs")
            if np.issubdtype(left_value.dtype, np.number):
                left_flat = left_value.reshape(-1)
                right_flat = right_value.reshape(-1)
                maximum = 0.0
                exact = True
                for start in range(0, left_flat.size, chunk_elements):
                    stop = min(start + chunk_elements, left_flat.size)
                    left_chunk = left_flat[start:stop]
                    right_chunk = right_flat[start:stop]
                    if not np.array_equal(left_chunk, right_chunk):
                        exact = False
                        maximum = max(
                            maximum,
                            float(
                                np.max(
                                    np.abs(
                                        left_chunk.astype(np.float64)
                                        - right_chunk.astype(np.float64)
                                    )
                                )
                            ),
                        )
                exact_fields[field] = exact
                category = category_for_field.get(field)
                if category is not None:
                    categories[category] = max(categories[category], maximum)
            else:
                equal = np.array_equal(left_value, right_value)
                exact_fields[field] = bool(equal)
                if not equal:
                    raise ValueError(
                        f"non-numeric provider export field {field!r} differs"
                    )
            del left_value, right_value
            gc.collect()
    return {
        "maximum_absolute_difference": categories,
        "exact_field_equality": exact_fields,
    }


def _markdown_summary(comparison: Mapping[str, Any]) -> str:
    memory = comparison["memory_comparison"]
    phases = comparison["phase_comparison"]
    parity = comparison["numerical_parity"]
    lines = [
        "# Frozen real-bundle eager versus mmap profile",
        "",
        f"Completion result: **{comparison['completion_result']}**.",
        "",
        "| Quantity | Eager NPZ | mmap NPY | mmap minus eager |",
        "| --- | ---: | ---: | ---: |",
        (
            "| Maximum process RSS | "
            f"{memory['eager_maximum_process_rss_bytes'] / _MIB:.2f} MiB | "
            f"{memory['mmap_maximum_process_rss_bytes'] / _MIB:.2f} MiB | "
            f"{memory['mmap_minus_eager_maximum_process_rss_bytes'] / _MIB:.2f} MiB |"
        ),
    ]
    for name in (
        "source_loading",
        "gauge_estimation",
        "overlap_disagreement",
        "dense_fusion",
        "provider_export",
        "provider_evaluation",
    ):
        record = phases[name]
        lines.append(
            f"| {name.replace('_', ' ').title()} peak RSS | "
            f"{record['eager_peak_rss_bytes'] / _MIB:.2f} MiB | "
            f"{record['mmap_peak_rss_bytes'] / _MIB:.2f} MiB | "
            f"{record['mmap_minus_eager_peak_rss_bytes'] / _MIB:.2f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Numerical parity",
            "",
            f"- Point maximum absolute difference: `{parity['point_max_abs_difference']:.17g}`",
            f"- Flow maximum absolute difference: `{parity['flow_max_abs_difference']:.17g}`",
            f"- Covariance maximum absolute difference: `{parity['covariance_max_abs_difference']:.17g}`",
            f"- Gauge maximum absolute difference: `{parity['gauge_max_abs_difference']:.17g}`",
            f"- Seam metric maximum absolute difference: `{parity['seam_max_abs_difference']:.17g}`",
            f"- Calibration-state maximum absolute difference: `{parity['calibration_max_abs_difference']:.17g}`",
            f"- Provider-evaluation maximum absolute difference: `{parity['provider_evaluation_max_abs_difference']:.17g}`",
            "",
            comparison["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _run_compare(arguments: argparse.Namespace) -> int:
    lock = _load_lock(arguments.lock)
    eager = _strict_json(arguments.eager_report)
    mmap = _strict_json(arguments.mmap_report)
    _require_arm(eager, backend="eager_npz")
    _require_arm(mmap, backend="mmap_npy")
    for value in (eager, mmap):
        if value.get("profile_lock_id") != lock["lock_id"]:
            raise ValueError("arm report lock ID mismatch")
        if value.get("estimator_base_revision") != lock["estimator_base_revision"]:
            raise ValueError("arm report estimator revision mismatch")
    if eager["repository_execution_revision"] != mmap["repository_execution_revision"]:
        raise ValueError("backend arms used different execution revisions")
    if eager["configuration"] != mmap["configuration"]:
        raise ValueError("backend arm estimator configurations differ")

    source_exact = (
        eager["semantic_outputs"]["source_bundle"]["content_sha256"]
        == mmap["semantic_outputs"]["source_bundle"]["content_sha256"]
    )
    disagreement_exact = (
        eager["semantic_outputs"]["disagreement"]["content_sha256"]
        == mmap["semantic_outputs"]["disagreement"]["content_sha256"]
    )
    uncertainty_exact = (
        eager["semantic_outputs"]["uncertainty"]["content_sha256"]
        == mmap["semantic_outputs"]["uncertainty"]["content_sha256"]
    )
    if not source_exact:
        raise ValueError("eager and mmap source arrays are not byte-identical")
    if not disagreement_exact:
        raise ValueError("eager and mmap disagreement fields are not byte-identical")
    if not uncertainty_exact:
        raise ValueError("eager and mmap uncertainty fields are not byte-identical")

    gauge_difference = _maximum_numeric_difference(
        {
            "alignments": eager["semantic_outputs"]["alignments"],
            "gauges": eager["semantic_outputs"]["gauges"],
        },
        {
            "alignments": mmap["semantic_outputs"]["alignments"],
            "gauges": mmap["semantic_outputs"]["gauges"],
        },
    )
    calibration_difference = _maximum_numeric_difference(
        eager["semantic_outputs"]["calibration"],
        mmap["semantic_outputs"]["calibration"],
    )
    evaluation_difference = _maximum_numeric_difference(
        eager["semantic_outputs"]["provider_evaluation"]["metrics"],
        mmap["semantic_outputs"]["provider_evaluation"]["metrics"],
    )
    seam_difference = abs(
        float(
            eager["semantic_outputs"]["provider_evaluation"]["metrics"][
                "seam_rmse"
            ]
        )
        - float(
            mmap["semantic_outputs"]["provider_evaluation"]["metrics"]["seam_rmse"]
        )
    )
    export_comparison = _compare_export_payloads(
        Path(eager["semantic_outputs"]["provider_export"]["path"]),
        Path(mmap["semantic_outputs"]["provider_export"]["path"]),
        chunk_elements=int(lock["completion_gate"]["comparison_chunk_elements"]),
    )
    export_differences = export_comparison["maximum_absolute_difference"]
    tolerance = lock["completion_gate"]["numerical_tolerance"]
    parity = {
        "source_array_content_exact": source_exact,
        "disagreement_content_exact": disagreement_exact,
        "uncertainty_content_exact": uncertainty_exact,
        "point_max_abs_difference": float(export_differences["point"]),
        "flow_max_abs_difference": float(export_differences["flow"]),
        "covariance_max_abs_difference": float(export_differences["covariance"]),
        "gauge_max_abs_difference": gauge_difference,
        "seam_max_abs_difference": seam_difference,
        "calibration_max_abs_difference": calibration_difference,
        "provider_evaluation_max_abs_difference": evaluation_difference,
        "provider_export_exact_field_equality": export_comparison[
            "exact_field_equality"
        ],
    }
    thresholds = {
        "point_max_abs_difference": float(tolerance["artifact_float32_atol"]),
        "flow_max_abs_difference": float(tolerance["artifact_float32_atol"]),
        "covariance_max_abs_difference": float(tolerance["artifact_float32_atol"]),
        "gauge_max_abs_difference": float(tolerance["float64_atol"]),
        "seam_max_abs_difference": float(tolerance["float64_atol"]),
        "calibration_max_abs_difference": float(tolerance["float64_atol"]),
        "provider_evaluation_max_abs_difference": float(tolerance["float64_atol"]),
    }
    failures = {
        name: value
        for name, value in parity.items()
        if name in thresholds and float(value) > thresholds[name]
    }
    if failures:
        raise ValueError(f"numerical parity gate failed: {failures}")

    phase_names = (
        "source_loading",
        "gauge_estimation",
        "overlap_disagreement",
        "dense_fusion",
        "provider_export",
        "provider_evaluation",
    )
    phase_comparison: dict[str, Any] = {}
    for name in phase_names:
        eager_phase = eager["phases"][name]
        mmap_phase = mmap["phases"][name]
        phase_comparison[name] = {
            "eager_peak_rss_bytes": int(eager_phase["peak_rss_bytes"]),
            "mmap_peak_rss_bytes": int(mmap_phase["peak_rss_bytes"]),
            "mmap_minus_eager_peak_rss_bytes": (
                int(mmap_phase["peak_rss_bytes"])
                - int(eager_phase["peak_rss_bytes"])
            ),
            "eager_wall_seconds": float(eager_phase["wall_seconds"]),
            "mmap_wall_seconds": float(mmap_phase["wall_seconds"]),
            "mmap_minus_eager_wall_seconds": (
                float(mmap_phase["wall_seconds"])
                - float(eager_phase["wall_seconds"])
            ),
        }

    total_wall_comparison = {
        "eager_total_wall_seconds": float(eager["timing"]["total_wall_seconds"]),
        "mmap_total_wall_seconds": float(mmap["timing"]["total_wall_seconds"]),
        "mmap_minus_eager_total_wall_seconds": (
            float(mmap["timing"]["total_wall_seconds"])
            - float(eager["timing"]["total_wall_seconds"])
        ),
    }

    eager_peak = int(eager["memory"]["maximum_process_rss_bytes"])
    mmap_peak = int(mmap["memory"]["maximum_process_rss_bytes"])
    saved = eager_peak - mmap_peak
    saved_fraction = saved / eager_peak if eager_peak else 0.0
    loading_saved = (
        int(eager["phases"]["source_loading"]["peak_rss_bytes"])
        - int(mmap["phases"]["source_loading"]["peak_rss_bytes"])
    )
    loading_saved_fraction = (
        loading_saved / int(eager["phases"]["source_loading"]["peak_rss_bytes"])
        if int(eager["phases"]["source_loading"]["peak_rss_bytes"])
        else 0.0
    )
    gate = lock["completion_gate"]["material_benefit"]
    material_peak = (
        saved >= int(gate["minimum_absolute_peak_rss_reduction_bytes"])
        and saved_fraction >= float(gate["minimum_relative_peak_rss_reduction"])
    )
    material_loading = (
        loading_saved >= int(gate["minimum_absolute_loading_peak_rss_reduction_bytes"])
        and loading_saved_fraction
        >= float(gate["minimum_relative_loading_peak_rss_reduction"])
    )
    completion_result = (
        "material_memory_benefit"
        if material_peak or material_loading
        else "valid_negative_result_no_practical_memory_benefit"
    )
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "version": COMPARISON_VERSION,
        "profile_lock_id": lock["lock_id"],
        "estimator_base_revision": lock["estimator_base_revision"],
        "repository_execution_revision": eager["repository_execution_revision"],
        "completion_result": completion_result,
        "completion_criteria_satisfied": True,
        "numerical_parity": parity,
        "parity_thresholds": thresholds,
        "memory_comparison": {
            "eager_maximum_process_rss_bytes": eager_peak,
            "mmap_maximum_process_rss_bytes": mmap_peak,
            "mmap_minus_eager_maximum_process_rss_bytes": mmap_peak - eager_peak,
            "eager_minus_mmap_maximum_process_rss_bytes": saved,
            "relative_maximum_process_rss_reduction": saved_fraction,
            "eager_retained_dense_vector_bytes": int(
                eager["memory"]["retained_dense_vector_bytes"]
            ),
            "mmap_retained_dense_vector_bytes": int(
                mmap["memory"]["retained_dense_vector_bytes"]
            ),
            "eager_input_persistent_bytes": int(
                eager["input_identity"]["input_persistent_bytes"]
            ),
            "mmap_input_persistent_bytes": int(
                mmap["input_identity"]["input_persistent_bytes"]
            ),
            "material_peak_rss_gate_passed": material_peak,
            "material_loading_peak_rss_gate_passed": material_loading,
        },
        "phase_comparison": phase_comparison,
        "total_wall_comparison": total_wall_comparison,
        "content_digests": {
            "eager_source_bundle": eager["semantic_outputs"]["source_bundle"][
                "content_sha256"
            ],
            "mmap_source_bundle": mmap["semantic_outputs"]["source_bundle"][
                "content_sha256"
            ],
            "eager_fused": eager["semantic_outputs"]["fused"]["content_sha256"],
            "mmap_fused": mmap["semantic_outputs"]["fused"]["content_sha256"],
            "eager_provider_evaluation": eager["semantic_outputs"][
                "provider_evaluation"
            ]["content_sha256"],
            "mmap_provider_evaluation": mmap["semantic_outputs"][
                "provider_evaluation"
            ]["content_sha256"],
        },
        "provider_export": {
            "eager_bytes": int(eager["semantic_outputs"]["provider_export"]["bytes"]),
            "mmap_bytes": int(mmap["semantic_outputs"]["provider_export"]["bytes"]),
            "semantic_content_equal": (
                eager["semantic_outputs"]["provider_export"]["content_sha256"]
                == mmap["semantic_outputs"]["provider_export"]["content_sha256"]
            ),
            "payload_field_comparison": export_comparison,
        },
        "provider_evaluation": {
            "eager_metrics": eager["semantic_outputs"]["provider_evaluation"][
                "metrics"
            ],
            "mmap_metrics": mmap["semantic_outputs"]["provider_evaluation"][
                "metrics"
            ],
        },
        "claim_boundary": (
            "This comparison establishes only backend engineering behavior and "
            "numerical parity on one frozen real calibration video. The disjoint "
            "baseline is an internal common reference, not ground truth for an "
            "accuracy claim. The result does not establish uncertainty "
            "calibration, provider competence, BayesianPhysTwin benefit, "
            "Causal4D benefit, deployment safety, or state of the art."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = _markdown_summary(comparison)
    arguments.markdown.write_text(markdown, encoding="utf-8")
    print(arguments.output)
    return 0


def _run_verify_lock(arguments: argparse.Namespace) -> int:
    value = _load_lock(arguments.lock)
    print(value["lock_id"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-lock")
    verify.add_argument("--lock", type=Path, required=True)
    verify.set_defaults(function=_run_verify_lock)

    arm = subparsers.add_parser("arm")
    arm.add_argument(
        "--backend",
        choices=("eager_npz", "mmap_npy"),
        required=True,
    )
    arm.add_argument("--input", type=Path, required=True)
    arm.add_argument("--lock", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.set_defaults(function=_run_arm)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--lock", type=Path, required=True)
    compare.add_argument("--eager-report", type=Path, required=True)
    compare.add_argument("--mmap-report", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--markdown", type=Path, required=True)
    compare.set_defaults(function=_run_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
