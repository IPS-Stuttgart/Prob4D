"""Batch MotionCrafter inference and Prob4D export for upstream evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .alignment import WindowAlignment, align_windows
from .fusion import FusedSequence, fuse_windows
from .gauge import FixedLagGaugeSmoother, RelativeGaugeConstraint, SequentialGaugeEstimator
from .io import PredictionBundle, load_prediction_bundle
from .motioncrafter import MotionCrafterAdapter, MotionCrafterRunConfig
from .uncertainty import DepthDisagreementModel, accumulate_disagreement


@dataclass(frozen=True)
class BenchmarkExportConfig:
    dataset_directory: Path
    output_directory: Path
    upstream_root: Path
    cache_directory: str
    metadata_filename: str = "filename_list.txt"
    model_type: str = "determ"
    height: int = 320
    width: int = 640
    window_size: int = 25
    overlap: int = 8
    seed: int = 42
    max_sequences: int | None = None
    skip_existing: bool = False


def _read_video_paths(dataset_directory: Path, metadata_filename: str) -> list[Path]:
    metadata_path = dataset_directory / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    paths: list[Path] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line.split()[0]))
    if not paths:
        raise ValueError(f"{metadata_path} contains no samples")
    return paths


def _build_alignments(bundle: PredictionBundle) -> list[WindowAlignment]:
    alignments: list[WindowAlignment] = []
    for moving_index, moving in enumerate(bundle.overlap_windows):
        for reference in bundle.overlap_windows[:moving_index]:
            if reference.common_frames(moving).size:
                alignments.append(align_windows(reference, moving, seed=moving_index))
    return alignments


def fuse_prediction_bundle(
    bundle: PredictionBundle,
) -> tuple[FusedSequence, FusedSequence]:
    """Produce decoded uniform and uncalibrated smoothed-CI predictions."""

    alignments = _build_alignments(bundle)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment) for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in bundle.overlap_windows]
    estimates = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(ordered_ids, estimates, constraints)
    windows = {window.window_id: window for window in bundle.overlap_windows}
    evidence = accumulate_disagreement(windows, alignments)
    model = DepthDisagreementModel()
    uncertainties = {
        window_id: model.predict(window, evidence[window_id])
        for window_id, window in windows.items()
    }
    sequential_gauges = {
        window_id: estimate.global_from_local for window_id, estimate in estimates.items()
    }
    smoothed_gauges = {
        window_id: estimate.global_from_local for window_id, estimate in smoothed.items()
    }
    uniform = fuse_windows(
        bundle.overlap_windows,
        sequential_gauges,
        uncertainties,
        method="uniform",
    )
    covariance_intersection = fuse_windows(
        bundle.overlap_windows,
        smoothed_gauges,
        uncertainties,
        method="covariance_intersection",
    )
    return uniform, covariance_intersection


def _write_fused_prediction(path: Path, sequence: FusedSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "point_map": sequence.point_map.astype(np.float16),
        "valid_mask": sequence.valid_mask,
        "frame_indices": sequence.frame_indices,
    }
    if sequence.scene_flow is not None:
        payload["scene_flow"] = sequence.scene_flow.astype(np.float16)
        payload["deform_mask"] = sequence.deform_mask
    np.savez(path, **payload)


def _copy_upstream_prediction(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_benchmark_export(config: BenchmarkExportConfig) -> Path:
    video_paths = _read_video_paths(config.dataset_directory, config.metadata_filename)
    if config.max_sequences is not None:
        video_paths = video_paths[: config.max_sequences]
    first_video = config.dataset_directory / video_paths[0]
    adapter = MotionCrafterAdapter(
        MotionCrafterRunConfig(
            upstream_root=config.upstream_root,
            video_path=first_video,
            output_directory=config.output_directory / "artifacts" / video_paths[0].with_suffix(""),
            model_type=config.model_type,
            cache_directory=config.cache_directory,
            height=config.height,
            width=config.width,
            window_size=config.window_size,
            overlap=config.overlap,
            seed=config.seed,
        )
    )
    methods = {
        "motioncrafter_disjoint": config.output_directory / "motioncrafter_disjoint",
        "motioncrafter_latent_linear": config.output_directory / "motioncrafter_latent_linear",
        "prob4d_uniform": config.output_directory / "prob4d_uniform",
        "prob4d_ci_smoothed_uncalibrated": (
            config.output_directory / "prob4d_ci_smoothed_uncalibrated"
        ),
    }
    samples: list[dict[str, object]] = []
    for sample_index, relative_video_path in enumerate(video_paths):
        relative_prediction_path = relative_video_path.with_suffix(".npz")
        destinations = {method: root / relative_prediction_path for method, root in methods.items()}
        if config.skip_existing and all(path.exists() for path in destinations.values()):
            samples.append(
                {
                    "video": relative_video_path.as_posix(),
                    "status": "existing",
                }
            )
            continue

        started = time.perf_counter()
        artifact_directory = (
            config.output_directory / "artifacts" / relative_video_path.with_suffix("")
        )
        manifest_path = adapter.run(
            video_path=config.dataset_directory / relative_video_path,
            output_directory=artifact_directory,
        )
        bundle = load_prediction_bundle(manifest_path)
        uniform, covariance_intersection = fuse_prediction_bundle(bundle)
        _copy_upstream_prediction(
            artifact_directory / "baseline_disjoint.npz",
            destinations["motioncrafter_disjoint"],
        )
        _copy_upstream_prediction(
            artifact_directory / "baseline_latent_linear.npz",
            destinations["motioncrafter_latent_linear"],
        )
        _write_fused_prediction(destinations["prob4d_uniform"], uniform)
        _write_fused_prediction(
            destinations["prob4d_ci_smoothed_uncalibrated"], covariance_intersection
        )
        samples.append(
            {
                "video": relative_video_path.as_posix(),
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - started,
                "frames": int(covariance_intersection.frame_indices.size),
                "index": sample_index,
            }
        )
        del bundle, uniform, covariance_intersection
        gc.collect()
        adapter.torch.cuda.empty_cache()

    manifest = {
        "format_version": 1,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "prob4d_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "motioncrafter_commit": _git_commit(config.upstream_root),
        "methods": {method: str(path.resolve()) for method, path in methods.items()},
        "samples": samples,
        "warning": (
            "The CI row uses the fixed depth/disagreement model without held-out "
            "uncertainty calibration. Treat it as a preliminary accuracy result."
        ),
    }
    manifest_path = config.output_directory / "benchmark_export.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--metadata-filename", default="filename_list.txt")
    parser.add_argument("--model-type", choices=["determ", "diff"], default="determ")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = run_benchmark_export(
        BenchmarkExportConfig(
            dataset_directory=arguments.dataset_dir,
            output_directory=arguments.output_dir,
            upstream_root=arguments.upstream_root,
            cache_directory=arguments.cache_dir,
            metadata_filename=arguments.metadata_filename,
            model_type=arguments.model_type,
            height=arguments.height,
            width=arguments.width,
            window_size=arguments.window_size,
            overlap=arguments.overlap,
            seed=arguments.seed,
            max_sequences=arguments.max_sequences,
            skip_existing=arguments.skip_existing,
        )
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
