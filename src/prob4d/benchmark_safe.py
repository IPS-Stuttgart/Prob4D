"""Pinned, fail-closed batch MotionCrafter inference and Prob4D export."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .benchmark import (
    FUSION_METHOD_NAMES,
    _copy_upstream_prediction,
    _git_commit,
    _read_video_paths,
    _write_fused_prediction,
    benchmark_method_semantics,
    fuse_prediction_bundle_methods,
)
from .io import (
    load_fused_prediction_artifact,
    load_fused_prediction_metadata,
    load_prediction_bundle,
)
from .motioncrafter import (
    MOTIONCRAFTER_SEED_POLICIES,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MotionCrafterSeedPolicy,
)
from .motioncrafter_models import (
    DEFAULT_BASE_PIPELINE,
    PinnedMotionCrafterModelSet,
)


@dataclass(frozen=True)
class BenchmarkExportConfig:
    """Complete model, dataset, and export configuration for one benchmark run."""

    dataset_directory: Path
    output_directory: Path
    upstream_root: Path
    cache_directory: str
    unet_path: str = "TencentARC/MotionCrafter"
    unet_revision: str | None = None
    vae_path: str = "TencentARC/MotionCrafter"
    vae_revision: str | None = None
    base_pipeline_path: str = DEFAULT_BASE_PIPELINE
    base_pipeline_revision: str | None = None
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read benchmark artifact {path}") from error
    return digest.hexdigest()


def _model_set(config: BenchmarkExportConfig) -> PinnedMotionCrafterModelSet:
    return PinnedMotionCrafterModelSet.inspect(
        model_type=config.model_type,
        unet_reference=config.unet_path,
        unet_revision=config.unet_revision,
        vae_reference=config.vae_path,
        vae_revision=config.vae_revision,
        base_pipeline_reference=config.base_pipeline_path,
        base_pipeline_revision=config.base_pipeline_revision,
    )


def _existing_prediction_manifest(
    artifact_directory: Path,
    *,
    motioncrafter_commit: str,
    seed_policy: str,
    model_set_sha256: str,
) -> tuple[Path, int]:
    manifest_path = artifact_directory / "predictions.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot validate existing benchmark prediction bundle {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("existing benchmark prediction manifest must be an object")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise ValueError("existing benchmark prediction manifest lacks config metadata")
    expected = {
        "motioncrafter_commit": motioncrafter_commit,
        "seed_policy": seed_policy,
        "model_source_set_sha256": model_set_sha256,
    }
    actual = {
        "motioncrafter_commit": manifest.get("motioncrafter_commit"),
        "seed_policy": manifest_config.get("seed_policy"),
        "model_source_set_sha256": manifest_config.get(
            "model_source_set_sha256"
        ),
    }
    if actual != expected:
        raise ValueError(
            "existing benchmark prediction bundle belongs to another MotionCrafter "
            f"configuration: expected={expected}, actual={actual}"
        )
    bundle = load_prediction_bundle(manifest_path)
    frame_count = int(bundle.disjoint_baseline.frame_indices.size)
    return manifest_path, frame_count


def _validate_existing_outputs(
    *,
    artifact_directory: Path,
    destinations: dict[str, Path],
    fusion_methods: tuple[str, ...],
    prob4d_commit: str,
    motioncrafter_commit: str,
    seed_policy: str,
    model_set_sha256: str,
    include_covariance: bool,
) -> tuple[Path, int]:
    """Validate every skipped output against the current exact producer identity."""

    manifest_path, frame_count = _existing_prediction_manifest(
        artifact_directory,
        motioncrafter_commit=motioncrafter_commit,
        seed_policy=seed_policy,
        model_set_sha256=model_set_sha256,
    )
    prediction_manifest_sha256 = _sha256_file(manifest_path)
    baseline_pairs = (
        (
            artifact_directory / "baseline_disjoint.npz",
            destinations["motioncrafter_disjoint"],
        ),
        (
            artifact_directory / "baseline_latent_linear.npz",
            destinations["motioncrafter_latent_linear"],
        ),
    )
    for source, destination in baseline_pairs:
        if not destination.is_file() or _sha256_file(source) != _sha256_file(
            destination
        ):
            raise ValueError(
                f"existing benchmark baseline {destination} differs from its bound source"
            )

    required_metadata: dict[str, object] = {
        "producer": "prob4d-benchmark",
        "prob4d_revision": prob4d_commit,
        "motioncrafter_revision": motioncrafter_commit,
        "motioncrafter_seed_policy": seed_policy,
        "motioncrafter_model_set_sha256": model_set_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "includes_covariance": include_covariance,
    }
    for method in fusion_methods:
        path = destinations[method]
        metadata = load_fused_prediction_metadata(path)
        if metadata.legacy_unspecified or metadata.method_id != method:
            raise ValueError(
                f"existing benchmark output {path} lacks the registered method identity"
            )
        details = metadata.metadata
        mismatches = {
            key: {"expected": value, "actual": details.get(key)}
            for key, value in required_metadata.items()
            if details.get(key) != value
        }
        semantics = benchmark_method_semantics(method)
        for key in ("gauge_estimator", "uncertainty_calibration"):
            expected_value = semantics[key]
            if details.get(key) != expected_value:
                mismatches[key] = {
                    "expected": expected_value,
                    "actual": details.get(key),
                }
        if mismatches:
            raise ValueError(
                f"existing benchmark output {path} has incompatible metadata: "
                f"{mismatches}"
            )
        if include_covariance:
            load_fused_prediction_artifact(path)
    return manifest_path, frame_count


def _validate_existing_benchmark_manifest(
    path: Path,
    *,
    prob4d_commit: str,
    motioncrafter_commit: str,
    model_set_sha256: str,
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot validate existing benchmark manifest {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("existing benchmark manifest must be an object")
    actual = {
        "prob4d_commit": payload.get("prob4d_commit"),
        "motioncrafter_commit": payload.get("motioncrafter_commit"),
        "motioncrafter_model_set_sha256": payload.get(
            "motioncrafter_model_set_sha256"
        ),
    }
    expected = {
        "prob4d_commit": prob4d_commit,
        "motioncrafter_commit": motioncrafter_commit,
        "motioncrafter_model_set_sha256": model_set_sha256,
    }
    if actual != expected:
        raise ValueError(
            f"existing benchmark manifest belongs to another run: {actual}"
        )


def run_benchmark_export(config: BenchmarkExportConfig) -> Path:
    """Run one pinned model-loading session across all registered videos."""

    model_set = _model_set(config)
    video_paths = _read_video_paths(config.dataset_directory, config.metadata_filename)
    if config.max_sequences is not None:
        video_paths = video_paths[: config.max_sequences]
    fusion_methods = tuple(dict.fromkeys(config.fusion_methods))
    unknown = set(fusion_methods).difference(FUSION_METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown fusion methods: {sorted(unknown)}")
    if not fusion_methods:
        raise ValueError("at least one fusion method must be requested")

    methods = {
        "motioncrafter_disjoint": config.output_directory / "motioncrafter_disjoint",
        "motioncrafter_latent_linear": config.output_directory
        / "motioncrafter_latent_linear",
    }
    methods.update({method: config.output_directory / method for method in fusion_methods})
    prob4d_commit = _git_commit(Path(__file__).resolve().parents[2])
    motioncrafter_commit = _git_commit(config.upstream_root)
    samples: list[dict[str, object]] = []
    adapter: Any | None = None

    for sample_index, relative_video_path in enumerate(video_paths):
        relative_prediction_path = relative_video_path.with_suffix(".npz")
        destinations = {
            method: root / relative_prediction_path for method, root in methods.items()
        }
        artifact_directory = (
            config.output_directory / "artifacts" / relative_video_path.with_suffix("")
        )
        existing_destinations = [
            path for path in destinations.values() if path.exists()
        ]
        if config.skip_existing and len(existing_destinations) == len(destinations):
            _, frame_count = _validate_existing_outputs(
                artifact_directory=artifact_directory,
                destinations=destinations,
                fusion_methods=fusion_methods,
                prob4d_commit=prob4d_commit,
                motioncrafter_commit=motioncrafter_commit,
                seed_policy=config.seed_policy,
                model_set_sha256=model_set.set_sha256,
                include_covariance=config.include_covariance,
            )
            samples.append(
                {
                    "video": relative_video_path.as_posix(),
                    "status": "existing_validated",
                    "frames": frame_count,
                    "index": sample_index,
                }
            )
            continue
        if existing_destinations:
            raise ValueError(
                "benchmark output is partial or replacement was not authorized: "
                + ", ".join(str(path) for path in existing_destinations)
            )
        if artifact_directory.exists() and any(artifact_directory.iterdir()):
            raise ValueError(
                f"benchmark prediction directory is nonempty: {artifact_directory}; "
                "remove the incomplete directory before rerunning"
            )
        if adapter is None:
            adapter_config = model_set.build_config(
                upstream_root=config.upstream_root,
                video_path=config.dataset_directory / relative_video_path,
                output_directory=artifact_directory,
                cache_directory=config.cache_directory,
                height=config.height,
                width=config.width,
                window_size=config.window_size,
                overlap=config.overlap,
                seed=config.seed,
                seed_policy=config.seed_policy,
            )
            adapter = model_set.adapter_factory()(adapter_config)

        total_started = time.perf_counter()
        inference_started = time.perf_counter()
        manifest_path = adapter.run(
            video_path=config.dataset_directory / relative_video_path,
            output_directory=artifact_directory,
        )
        inference_seconds = time.perf_counter() - inference_started
        prediction_manifest_sha256 = _sha256_file(manifest_path)

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
        shared_metadata = {
            "prob4d_revision": prob4d_commit,
            "motioncrafter_revision": motioncrafter_commit,
            "motioncrafter_seed_policy": config.seed_policy,
            "motioncrafter_model_set_sha256": model_set.set_sha256,
            "prediction_manifest_sha256": prediction_manifest_sha256,
            "includes_covariance": config.include_covariance,
        }
        for method, sequence in fused_methods.items():
            _write_fused_prediction(
                destinations[method],
                sequence,
                method_id=method,
                include_covariance=config.include_covariance,
                metadata=shared_metadata,
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
                "prediction_manifest_sha256": prediction_manifest_sha256,
            }
        )
        del bundle, fused_methods
        gc.collect()
        adapter.torch.cuda.empty_cache()

    manifest = {
        "format_version": 2,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "prob4d_commit": prob4d_commit,
        "motioncrafter_commit": motioncrafter_commit,
        "motioncrafter_model_set_sha256": model_set.set_sha256,
        "motioncrafter_model_source_manifest": json.loads(model_set.manifest_json),
        "methods": {method: str(path.resolve()) for method, path in methods.items()},
        "method_semantics": {
            method: benchmark_method_semantics(method) for method in fusion_methods
        },
        "samples": samples,
        "warning": (
            "Model and artifact semantics are provenance-bound. The exported Prob4D "
            "rows still use the fixed depth/disagreement model without held-out "
            "uncertainty calibration, so calibration and downstream benefit remain "
            "empirical gates."
        ),
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_directory / "benchmark_export.json"
    if manifest_path.exists():
        if not config.skip_existing:
            raise ValueError(f"benchmark manifest already exists: {manifest_path}")
        _validate_existing_benchmark_manifest(
            manifest_path,
            prob4d_commit=prob4d_commit,
            motioncrafter_commit=motioncrafter_commit,
            model_set_sha256=model_set.set_sha256,
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--unet-path", default="TencentARC/MotionCrafter")
    parser.add_argument("--unet-revision")
    parser.add_argument("--vae-path", default="TencentARC/MotionCrafter")
    parser.add_argument("--vae-revision")
    parser.add_argument("--base-pipeline-path", default=DEFAULT_BASE_PIPELINE)
    parser.add_argument("--base-pipeline-revision")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = BenchmarkExportConfig(
        dataset_directory=arguments.dataset_dir,
        output_directory=arguments.output_dir,
        upstream_root=arguments.upstream_root,
        cache_directory=arguments.cache_dir,
        unet_path=arguments.unet_path,
        unet_revision=arguments.unet_revision,
        vae_path=arguments.vae_path,
        vae_revision=arguments.vae_revision,
        base_pipeline_path=arguments.base_pipeline_path,
        base_pipeline_revision=arguments.base_pipeline_revision,
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
    try:
        manifest = run_benchmark_export(config)
    except ValueError as error:
        parser.error(str(error))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkExportConfig",
    "main",
    "run_benchmark_export",
]
