"""GPU adapter for producing all MotionCrafter inputs to the Prob4D ablation.

This module deliberately imports MotionCrafter, torch, diffusers, and decord
only after the CLI starts. The NumPy estimator remains installable without the
large upstream inference environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from .data import PredictionWindow
from .lineage import motioncrafter_temporal_lineage_manifest

MotionCrafterSeedPolicy = Literal["legacy-common", "derived-per-call"]
MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON: Final[MotionCrafterSeedPolicy] = (
    "legacy-common"
)
MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL: Final[MotionCrafterSeedPolicy] = (
    "derived-per-call"
)
MOTIONCRAFTER_SEED_POLICIES: Final[tuple[MotionCrafterSeedPolicy, ...]] = (
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
)
MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA: Final = "prob4d.motioncrafter-seed-schedule.v1"
_NUMPY_SEED_MODULUS: Final = 2**32


def motioncrafter_seed_for_call(
    root_seed: int,
    *,
    policy: MotionCrafterSeedPolicy,
    call_id: str,
) -> int:
    """Return the deterministic effective seed for one inference call.

    ``legacy-common`` preserves the historical behavior in which every baseline
    and independently decoded window receives the same seed. ``derived-per-call``
    hashes the root seed and stable call identity into NumPy's accepted 32-bit
    seed range. Distinct seeds avoid an implicit common-random-number coupling;
    they do not by themselves establish statistical independence.
    """

    seed = int(root_seed)
    if not 0 <= seed < _NUMPY_SEED_MODULUS:
        raise ValueError("seed must lie in [0, 2**32)")
    normalized_call_id = str(call_id).strip()
    if not normalized_call_id:
        raise ValueError("call_id must be non-empty")
    if policy == MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON:
        return seed
    if policy != MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL:
        raise ValueError(f"unsupported MotionCrafter seed policy {policy!r}")
    descriptor = json.dumps(
        {
            "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
            "root_seed": seed,
            "call_id": normalized_call_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(descriptor).digest()[:4], "big")


def validate_motioncrafter_seed_schedule(
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Validate a prediction manifest's recorded inference-seed schedule.

    Manifests created before the schedule existed are accepted only under the
    historical ``legacy-common`` policy. New ``derived-per-call`` runs must carry
    the complete source-bound call list, and every effective seed is recomputed.
    """

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prediction manifest config must be a mapping")
    root_seed = int(config.get("seed", -1))
    policy = str(
        config.get("seed_policy", MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON)
    )
    if policy not in MOTIONCRAFTER_SEED_POLICIES:
        raise ValueError(f"unsupported MotionCrafter seed policy {policy!r}")
    motioncrafter_seed_for_call(root_seed, policy=policy, call_id="validation")

    schedule = manifest.get("stochastic_seed_schedule")
    if schedule is None:
        if policy != MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON:
            raise ValueError("derived-per-call prediction manifest lacks a seed schedule")
        return {
            "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
            "policy": policy,
            "root_seed": root_seed,
            "call_count": 0,
            "source": "implicit-legacy-common",
        }
    if not isinstance(schedule, Mapping):
        raise ValueError("stochastic_seed_schedule must be a mapping")
    if schedule.get("schema") != MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA:
        raise ValueError("unsupported stochastic seed-schedule schema")
    if schedule.get("policy") != policy:
        raise ValueError("seed schedule policy differs from prediction config")
    if int(schedule.get("root_seed", -1)) != root_seed:
        raise ValueError("seed schedule root_seed differs from prediction config")
    calls = schedule.get("calls")
    if not isinstance(calls, list):
        raise ValueError("seed schedule calls must be a list")
    windows = manifest.get("overlap_windows")
    if not isinstance(windows, list):
        raise ValueError("seed-scheduled manifest must contain overlap_windows")

    expected: list[dict[str, object]] = [
        {"call_id": "baseline-disjoint", "product": "disjoint_baseline"},
        {
            "call_id": "baseline-latent-linear",
            "product": "latent_linear_baseline",
        },
    ]
    for item in windows:
        if not isinstance(item, Mapping):
            raise ValueError("overlap window records must be mappings")
        window_id = str(item.get("window_id", "")).strip()
        if not window_id:
            raise ValueError("overlap window_id must be non-empty")
        start = int(item.get("start_frame", -1))
        stop = int(item.get("stop_frame", -1))
        if start < 0 or stop <= start:
            raise ValueError("overlap window frame bounds are invalid")
        expected.append(
            {
                "call_id": f"overlap-window:{window_id}:{start}:{stop}",
                "product": "independently_decoded_overlap_window",
                "window_id": window_id,
                "source_frame_start": start,
                "source_frame_stop_exclusive": stop,
            }
        )
    if len(calls) != len(expected):
        raise ValueError("seed schedule call count does not match prediction products")

    call_ids: list[str] = []
    for index, (actual, required) in enumerate(zip(calls, expected, strict=True)):
        if not isinstance(actual, Mapping):
            raise ValueError(f"seed schedule call {index} must be a mapping")
        for key, value in required.items():
            if actual.get(key) != value:
                raise ValueError(f"seed schedule call {index} has inconsistent {key}")
        call_id = str(required["call_id"])
        effective_seed = int(actual.get("effective_seed", -1))
        expected_seed = motioncrafter_seed_for_call(
            root_seed,
            policy=policy,
            call_id=call_id,
        )
        if effective_seed != expected_seed:
            raise ValueError(f"seed schedule call {call_id!r} has an invalid effective seed")
        call_ids.append(call_id)
    if len(set(call_ids)) != len(call_ids):
        raise ValueError("seed schedule call IDs must be unique")
    if (
        policy == MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL
        and len({int(item["effective_seed"]) for item in calls}) != len(calls)
    ):
        raise ValueError("derived-per-call seed schedule contains a seed collision")
    return {
        "schema": MOTIONCRAFTER_SEED_SCHEDULE_SCHEMA,
        "policy": policy,
        "root_seed": root_seed,
        "call_count": len(calls),
        "source": "manifest",
    }


@dataclass(frozen=True)
class MotionCrafterRunConfig:
    upstream_root: Path
    video_path: Path
    output_directory: Path
    model_type: str = "determ"
    unet_path: str = "TencentARC/MotionCrafter"
    vae_path: str = "TencentARC/MotionCrafter"
    cache_directory: str = "workspace/pretrained_models"
    height: int = 320
    width: int = 640
    window_size: int = 25
    overlap: int = 8
    num_inference_steps: int = 5
    guidance_scale: float = 1.0
    decode_chunk_size: int = 25
    seed: int = 42
    seed_policy: MotionCrafterSeedPolicy = MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON
    low_memory_usage: bool = False
    frame_start: int = 0
    frame_stop: int | None = None
    frame_stride: int = 1

    def __post_init__(self) -> None:
        if self.model_type not in {"determ", "diff"}:
            raise ValueError("model_type must be 'determ' or 'diff'")
        if self.height % 64 or self.width % 64:
            raise ValueError("MotionCrafter height and width must be divisible by 64")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError("overlap must be non-negative and smaller than window_size")
        if not 0 <= int(self.seed) < _NUMPY_SEED_MODULUS:
            raise ValueError("seed must lie in [0, 2**32)")
        if self.seed_policy not in MOTIONCRAFTER_SEED_POLICIES:
            raise ValueError(
                "seed_policy must be one of " + ", ".join(MOTIONCRAFTER_SEED_POLICIES)
            )
        if self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_stop is not None and self.frame_stop <= self.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")


class MotionCrafterAdapter:
    """Load upstream once and emit exact baselines plus independent windows."""

    def __init__(self, config: MotionCrafterRunConfig) -> None:
        self.config = config
        upstream_root = config.upstream_root.resolve()
        if not (upstream_root / "motioncrafter").is_dir():
            raise ValueError(f"{upstream_root} is not a MotionCrafter checkout")
        sys.path.insert(0, str(upstream_root))

        try:
            import torch
            import torch.nn.functional as functional
            from decord import VideoReader, cpu
            from diffusers import AutoencoderKL
            from diffusers.training_utils import set_seed
            from motioncrafter import (
                MotionCrafterDetermPipeline,
                MotionCrafterDiffPipeline,
                UNetSpatioTemporalConditionModelVid2vid,
                UnifyAutoencoderKL,
            )
        except ImportError as error:
            raise RuntimeError(
                "Run prob4d-motioncrafter inside the upstream MotionCrafter environment"
            ) from error

        self.torch = torch
        self.functional = functional
        self.VideoReader = VideoReader
        self.cpu = cpu
        self.set_seed = set_seed
        self.geometry_motion_vae = (
            UnifyAutoencoderKL.from_pretrained(
                config.vae_path,
                subfolder="geometry_motion_vae",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                cache_dir=config.cache_directory,
            )
            .requires_grad_(False)
            .to("cuda", dtype=torch.float32)
        )
        unet = (
            UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
                config.unet_path,
                subfolder="unet_diff" if config.model_type == "diff" else "unet_determ",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                cache_dir=config.cache_directory,
            )
            .requires_grad_(False)
            .to("cuda", dtype=torch.float16)
        )
        pipeline_class = (
            MotionCrafterDiffPipeline
            if config.model_type == "diff"
            else MotionCrafterDetermPipeline
        )
        self.pipeline = pipeline_class.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=config.cache_directory,
        ).to("cuda")
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipeline.enable_attention_slicing()
        self.video_vae_class = AutoencoderKL

    def read_video(self, video_path: Path | None = None) -> Any:
        """Read, cover-resize, and center-crop the entire RGB sequence."""

        config = self.config
        reader = self.VideoReader(str(video_path or config.video_path), ctx=self.cpu(0))
        stop = len(reader) if config.frame_stop is None else min(config.frame_stop, len(reader))
        frame_indices = list(range(config.frame_start, stop, config.frame_stride))
        if not frame_indices:
            raise ValueError("selected source-frame interval is empty")
        frames = reader.get_batch(frame_indices).asnumpy().astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        resize_scale = max(config.height / tensor.shape[-2], config.width / tensor.shape[-1])
        resized = (
            int(round(tensor.shape[-2] * resize_scale)),
            int(round(tensor.shape[-1] * resize_scale)),
        )
        tensor = self.functional.interpolate(tensor, resized, mode="bicubic", antialias=True).clamp(
            0, 1
        )
        row = (tensor.shape[-2] - config.height) // 2
        column = (tensor.shape[-1] - config.width) // 2
        return tensor[:, :, row : row + config.height, column : column + config.width]

    def infer(self, frames: Any, *, window_size: int, overlap: int, seed: int) -> tuple:
        self.set_seed(seed)
        kwargs = {
            "height": self.config.height,
            "width": self.config.width,
            "window_size": min(window_size, int(frames.shape[0])),
            "overlap": overlap if int(frames.shape[0]) > window_size else 0,
            "decode_chunk_size": self.config.decode_chunk_size,
            "force_projection": True,
            "force_fixed_focal": True,
            "track_time": False,
            "low_memory_usage": self.config.low_memory_usage,
        }
        if self.config.model_type == "diff":
            kwargs.update(
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
            )
        with self.torch.inference_mode():
            return self.pipeline(frames.to("cuda"), self.geometry_motion_vae, None, **kwargs)

    @staticmethod
    def _arrays(results: tuple) -> dict[str, np.ndarray]:
        if len(results) == 4:
            points, valid, flow, deform = results
            return {
                "point_map": points.detach().cpu().numpy().astype(np.float32),
                "valid_mask": valid.detach().cpu().numpy().astype(bool),
                "scene_flow": flow.detach().cpu().numpy().astype(np.float32),
                "deform_mask": deform.detach().cpu().numpy().astype(bool),
            }
        points, valid = results
        return {
            "point_map": points.detach().cpu().numpy().astype(np.float32),
            "valid_mask": valid.detach().cpu().numpy().astype(bool),
        }

    def _write_baseline(
        self,
        path: Path,
        results: tuple,
        frame_indices: np.ndarray,
    ) -> None:
        arrays = self._arrays(results)
        arrays["frame_indices"] = frame_indices
        np.savez_compressed(path, **arrays)

    def run(
        self,
        *,
        video_path: Path | None = None,
        output_directory: Path | None = None,
    ) -> Path:
        config = self.config
        actual_video_path = (video_path or config.video_path).resolve()
        output = output_directory or config.output_directory
        windows_directory = output / "windows"
        windows_directory.mkdir(parents=True, exist_ok=True)
        frames = self.read_video(actual_video_path)
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
        seed_schedule: list[dict[str, object]] = []

        def effective_seed(
            call_id: str,
            *,
            product: str,
            window_id: str | None = None,
            source_frame_start: int | None = None,
            source_frame_stop_exclusive: int | None = None,
        ) -> int:
            seed = motioncrafter_seed_for_call(
                config.seed,
                policy=config.seed_policy,
                call_id=call_id,
            )
            record: dict[str, object] = {
                "call_id": call_id,
                "product": product,
                "effective_seed": seed,
            }
            if window_id is not None:
                record["window_id"] = window_id
            if source_frame_start is not None:
                record["source_frame_start"] = source_frame_start
            if source_frame_stop_exclusive is not None:
                record["source_frame_stop_exclusive"] = source_frame_stop_exclusive
            seed_schedule.append(record)
            return seed

        disjoint_path = output / "baseline_disjoint.npz"
        disjoint = self.infer(
            frames,
            window_size=config.window_size,
            overlap=0,
            seed=effective_seed(
                "baseline-disjoint",
                product="disjoint_baseline",
            ),
        )
        self._write_baseline(disjoint_path, disjoint, frame_indices)

        latent_path = output / "baseline_latent_linear.npz"
        latent = self.infer(
            frames,
            window_size=config.window_size,
            overlap=config.overlap,
            seed=effective_seed(
                "baseline-latent-linear",
                product="latent_linear_baseline",
            ),
        )
        self._write_baseline(latent_path, latent, frame_indices)

        stride = config.window_size - config.overlap
        starts = list(range(0, max(1, num_frames - config.window_size + 1), stride))
        final_start = max(0, num_frames - config.window_size)
        if final_start not in starts:
            starts.append(final_start)
        starts = sorted(set(starts))
        manifest_windows: list[dict[str, Any]] = []
        for index, start in enumerate(starts):
            stop = min(start + config.window_size, num_frames)
            window_id = f"window_{index:04d}"
            start_frame = int(frame_indices[start])
            stop_frame = int(frame_indices[stop - 1]) + config.frame_stride
            results = self.infer(
                frames[start:stop],
                window_size=stop - start,
                overlap=0,
                seed=effective_seed(
                    f"overlap-window:{window_id}:{start_frame}:{stop_frame}",
                    product="independently_decoded_overlap_window",
                    window_id=window_id,
                    source_frame_start=start_frame,
                    source_frame_stop_exclusive=stop_frame,
                ),
            )
            arrays = self._arrays(results)
            window = PredictionWindow(
                window_id=window_id,
                frame_indices=frame_indices[start:stop],
                point_map=arrays["point_map"],
                valid_mask=arrays["valid_mask"],
                scene_flow=arrays.get("scene_flow"),
                deform_mask=arrays.get("deform_mask"),
                dense_storage_dtype="float32",
            )
            relative_path = Path("windows") / f"{window_id}.npz"
            window.to_npz(output / relative_path)
            manifest_windows.append(
                {
                    "window_id": window_id,
                    "path": relative_path.as_posix(),
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                }
            )

        manifest = {
            "format_version": 1,
            "video_path": str(actual_video_path),
            "motioncrafter_commit": self._upstream_commit(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
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
            "disjoint_baseline": disjoint_path.name,
            "latent_linear_baseline": latent_path.name,
        }
        manifest["config"]["video_path"] = str(actual_video_path)
        manifest["config"]["output_directory"] = str(output.resolve())
        validate_motioncrafter_seed_schedule(manifest)
        manifest_path = output / "predictions.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    def _upstream_commit(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.config.upstream_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["determ", "diff"], default="determ")
    parser.add_argument("--unet-path", default="TencentARC/MotionCrafter")
    parser.add_argument("--vae-path", default="TencentARC/MotionCrafter")
    parser.add_argument("--cache-dir", default="workspace/pretrained_models")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--decode-chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-policy",
        choices=MOTIONCRAFTER_SEED_POLICIES,
        default=MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
        help=(
            "legacy-common reuses one seed for exact historical behavior; "
            "derived-per-call records deterministic source-bound seeds"
        ),
    )
    parser.add_argument("--low-memory-usage", action="store_true")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    arguments = parser.parse_args(argv)
    config = MotionCrafterRunConfig(
        upstream_root=arguments.upstream_root,
        video_path=arguments.video_path,
        output_directory=arguments.output_dir,
        model_type=arguments.model_type,
        unet_path=arguments.unet_path,
        vae_path=arguments.vae_path,
        cache_directory=arguments.cache_dir,
        height=arguments.height,
        width=arguments.width,
        window_size=arguments.window_size,
        overlap=arguments.overlap,
        num_inference_steps=arguments.num_inference_steps,
        guidance_scale=arguments.guidance_scale,
        decode_chunk_size=arguments.decode_chunk_size,
        seed=arguments.seed,
        seed_policy=arguments.seed_policy,
        low_memory_usage=arguments.low_memory_usage,
        frame_start=arguments.frame_start,
        frame_stop=arguments.frame_stop,
        frame_stride=arguments.frame_stride,
    )
    manifest = MotionCrafterAdapter(config).run()
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
