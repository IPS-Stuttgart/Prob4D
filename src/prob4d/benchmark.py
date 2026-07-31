"""Batch MotionCrafter inference and Prob4D export for upstream evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .alignment import WindowAlignment, align_windows
from .fusion import FusedSequence, fuse_windows
from .gauge import FixedLagGaugeSmoother, RelativeGaugeConstraint, SequentialGaugeEstimator
from .io import (
    PredictionBundle,
    fusion_covariance_semantics,
    load_prediction_bundle,
    save_fused_prediction,
)
from .motioncrafter import (
    MOTIONCRAFTER_SEED_POLICIES,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MotionCrafterAdapter,
    MotionCrafterRunConfig,
    MotionCrafterSeedPolicy,
)
from .uncertainty import DepthDisagreementModel, accumulate_disagreement

FUSION_METHOD_NAMES = (
    "prob4d_uniform",
    "prob4d_uniform_smoothed",
    "prob4d_precision",
    "prob4d_ci",
    "prob4d_ci_smoothed_uncalibrated",
)

_BENCHMARK_METHOD_SPECS: dict[str, dict[str, str]] = {
    "prob4d_uniform": {
        "fusion_method": "uniform",
        "gauge_estimator": "sequential",
        "uncertainty_calibration": "uncalibrated_fixed_model",
    },
    "prob4d_uniform_smoothed": {
        "fusion_method": "uniform",
        "gauge_estimator": "fixed_lag",
        "uncertainty_calibration": "uncalibrated_fixed_model",
    },
    "prob4d_precision": {
        "fusion_method": "precision",
        "gauge_estimator": "sequential",
        "uncertainty_calibration": "uncalibrated_fixed_model",
    },
    "prob4d_ci": {
        "fusion_method": "covariance_intersection",
        "gauge_estimator": "sequential",
        "uncertainty_calibration": "uncalibrated_fixed_model",
    },
    "prob4d_ci_smoothed_uncalibrated": {
        "fusion_method": "covariance_intersection",
        "gauge_estimator": "fixed_lag",
        "uncertainty_calibration": "uncalibrated_fixed_model",
    },
}


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
    seed_policy: MotionCrafterSeedPolicy = MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON
    max_sequences: int | None = None
    skip_existing: bool = False
    include_covariance: bool = False
    fusion_methods: tuple[str, ...] = FUSION_METHOD_NAMES


def benchmark_method_semantics(method_id: str) -> dict[str, str]:
    """Return the explicit estimator, covariance, and dependence semantics."""

    try:
        spec = dict(_BENCHMARK_METHOD_SPECS[method_id])
    except KeyError as error:
        raise ValueError(f"unknown fusion method {method_id!r}") from error
    covariance_semantics, correlation_assumption = fusion_covariance_semantics(
        spec["fusion_method"]
    )
    return {
        **spec,
        "covariance_semantics": covariance_semantics,
        "correlation_assumption": correlation_assumption,
    }


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

    methods = fuse_prediction_bundle_methods(bundle)
    return methods["prob4d_uniform"], methods["prob4d_ci_smoothed_uncalibrated"]


def fuse_prediction_bundle_methods(
    bundle: PredictionBundle,
    *,
    model: DepthDisagreementModel | None = None,
    method_names: Collection[str] | None = None,
) -> dict[str, FusedSequence]:
    """Produce all decoded fusion variants with propagated gauge covariance."""

    requested = set(FUSION_METHOD_NAMES if method_names is None else method_names)
    unknown = requested.difference(FUSION_METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown fusion methods: {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one fusion method must be requested")

    alignments = _build_alignments(bundle)
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment) for alignment in alignments
    ]
    ordered_ids = [window.window_id for window in bundle.overlap_windows]
    estimates = SequentialGaugeEstimator().estimate(ordered_ids, constraints)
    smoothed = FixedLagGaugeSmoother(lag=4).smooth(ordered_ids, estimates, constraints)
    windows = {window.window_id: window for window in bundle.overlap_windows}
    evidence = accumulate_disagreement(windows, alignments)
    model = model or DepthDisagreementModel()
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
    sequential_covariances = {
        window_id: estimate.covariance for window_id, estimate in estimates.items()
    }
    smoothed_covariances = {
        window_id: estimate.covariance for window_id, estimate in smoothed.items()
    }
    methods: dict[str, FusedSequence] = {}
    if "prob4d_uniform" in requested:
        methods["prob4d_uniform"] = fuse_windows(
            bundle.overlap_windows,
            sequential_gauges,
            uncertainties,
            method="uniform",
            gauge_covariances=sequential_covariances,
        )
    if "prob4d_uniform_smoothed" in requested:
        methods["prob4d_uniform_smoothed"] = fuse_windows(
            bundle.overlap_windows,
            smoothed_gauges,
            uncertainties,
            method="uniform",
            gauge_covariances=smoothed_covariances,
        )
    if "prob4d_precision" in requested:
        methods["prob4d_precision"] = fuse_windows(
            bundle.overlap_windows,
            sequential_gauges,
            uncertainties,
            method="precision",
            gauge_covariances=sequential_covariances,
        )
    if "prob4d_ci" in requested:
        methods["prob4d_ci"] = fuse_windows(
            bundle.overlap_windows,
            sequential_gauges,
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=sequential_covariances,
        )
    if "prob4d_ci_smoothed_uncalibrated" in requested:
        methods["prob4d_ci_smoothed_uncalibrated"] = fuse_windows(
            bundle.overlap_windows,
            smoothed_gauges,
            uncertainties,
            method="covariance_intersection",
            gauge_covariances=smoothed_covariances,
        )
    return methods


def _write_fused_prediction(
    path: Path,
    sequence: FusedSequence,
    *,
    method_id: str = "prob4d_uniform",
    include_covariance: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist one benchmark row with explicit covariance semantics."""

    semantics = benchmark_method_semantics(method_id)
    extra = {} if metadata is None else dict(metadata)
    save_fused_prediction(
        path,
        sequence,
        method_id=method_id,
        fusion_method=semantics["fusion_method"],
        include_covariance=include_covariance,
        metadata={
            "producer": "prob4d-benchmark",
            "gauge_estimator": semantics["gauge_estimator"],
            "uncertainty_calibration": semantics["uncertainty_calibration"],
            **extra,
        },
    )


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
            seed_policy=config.seed_policy,
        )
    )
    fusion_methods = tuple(dict.fromkeys(config.fusion_methods))
    unknown = set(fusion_methods).difference(FUSION_METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown fusion methods: {sorted(unknown)}")
    if not fusion_methods:
        raise ValueError("at least one fusion method must be requested")
    methods = {
        "motioncrafter_disjoint": config.output_directory / "motioncrafter_disjoint",
        "motioncrafter_latent_linear": config.output_directory / "motioncrafter_latent_linear",
    }
    methods.update({method: config.output_directory / method for method in fusion_methods})
    prob4d_commit = _git_commit(Path(__file__).resolve().parents[2])
    motioncrafter_commit = _git_commit(config.upstream_root)
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

        total_started = time.perf_counter()
        artifact_directory = (
            config.output_directory / "artifacts" / relative_video_path.with_suffix("")
        )
        inference_started = time.perf_counter()
        manifest_path = adapter.run(
            video_path=config.dataset_directory / relative_video_path,
            output_directory=artifact_directory,
        )
        inference_seconds = time.perf_counter() - inference_started

        loading_started = time.perf_counter()
        bundle = load_prediction_bundle(manifest_path)
        loading_seconds = time.perf_counter() - loading_started

        fusion_started = time.perf_counter()
        fused_methods = fuse_prediction_bundle_methods(bundle, method_names=fusion_methods)
        fusion_seconds = time.perf_counter() - fusion_started

        export_started = time.perf_counter()
        _copy_upstream_prediction(
            artifact_directory / "baseline_disjoint.npz",
            destinations["motioncrafter_disjoint"],
        )
        _copy_upstream_prediction(
            artifact_directory / "baseline_latent_linear.npz",
            destinations["motioncrafter_latent_linear"],
        )
        for method, sequence in fused_methods.items():
            _write_fused_prediction(
                destinations[method],
                sequence,
                method_id=method,
                include_covariance=config.include_covariance,
                metadata={
                    "prob4d_revision": prob4d_commit,
                    "motioncrafter_revision": motioncrafter_commit,
                    "motioncrafter_seed_policy": config.seed_policy,
                },
            )
        export_seconds = time.perf_counter() - export_started
        total_seconds = time.perf_counter() - total_started
        samples.append(
            {
                "video": relative_video_path.as_posix(),
                "status": "completed",
                "elapsed_seconds": total_seconds,
                "stage_seconds": {
                    "motioncrafter_inference": inference_seconds,
                    "prediction_loading": loading_seconds,
                    "prob4d_fusion": fusion_seconds,
                    "artifact_export": export_seconds,
                },
                "frames": int(next(iter(fused_methods.values())).frame_indices.size),
                "index": sample_index,
            }
        )
        del bundle, fused_methods
        gc.collect()
        adapter.torch.cuda.empty_cache()

    manifest = {
        "format_version": 1,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "prob4d_commit": prob4d_commit,
        "motioncrafter_commit": motioncrafter_commit,
        "methods": {method: str(path.resolve()) for method, path in methods.items()},
        "method_semantics": {
            method: benchmark_method_semantics(method) for method in fusion_methods
        },
        "samples": samples,
        "warning": (
            "The exported Prob4D rows use the fixed depth/disagreement model without "
            "held-out uncertainty calibration. Their covariance semantics are explicit, "
            "but calibration and downstream benefit remain empirical gates."
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
    parser.add_argument(
        "--seed-policy",
        choices=MOTIONCRAFTER_SEED_POLICIES,
        default=MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
        help=(
            "legacy-common preserves historical common random numbers; "
            "derived-per-call records deterministic source-bound seeds"
        ),
    )
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-covariance", action="store_true")
    parser.add_argument(
        "--fusion-method",
        dest="fusion_methods",
        action="append",
        choices=FUSION_METHOD_NAMES,
        help="Fusion output to export; repeat for multiple methods (default: all).",
    )
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
            seed_policy=arguments.seed_policy,
            max_sequences=arguments.max_sequences,
            skip_existing=arguments.skip_existing,
            include_covariance=arguments.include_covariance,
            fusion_methods=tuple(arguments.fusion_methods or FUSION_METHOD_NAMES),
        )
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
