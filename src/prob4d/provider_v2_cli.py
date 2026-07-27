"""Explicit provider-v2 CLIs for exploratory and claim-bearing observation export."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .covariance_root import COVARIANCE_ROOT_MODES
from .observation_export import SAMPLING_MODES
from .provider_v2 import (
    export_calibrated_observation_belief,
    export_exploratory_observation_belief,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_observation_belief_export,
    load_point_uncertainty_calibration,
    save_observation_belief_export,
)


def _temporary_path(parent: Path, *, prefix: str, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=parent,
        prefix=prefix,
        suffix=suffix,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(
        path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_observation(path: Path, artifact: Any) -> None:
    temporary = _temporary_path(
        path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
    )
    try:
        save_observation_belief_export(temporary, artifact)
        restored = load_observation_belief_export(temporary)
        if restored.artifact_id != artifact.artifact_id:
            raise RuntimeError("observation artifact changed during serialization")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_source_revision: bool,
) -> None:
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--causal-frame-stop", type=int, required=True)
    parser.add_argument("--metric-gauge-anchor", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default="fixed_grid",
    )
    parser.add_argument("--effective-samples-per-group", type=float, default=64.0)
    parser.add_argument("--minimum-prior-reliability", type=float, default=0.05)
    parser.add_argument("--max-gauge-rank", type=int, default=64)
    parser.add_argument(
        "--minimum-retained-gauge-trace",
        type=float,
        default=0.999,
    )
    parser.add_argument("--view-name", default="camera0")
    parser.add_argument(
        "--source-revision",
        required=require_source_revision,
        help=(
            "exact executing Prob4D commit; calibrated export independently verifies "
            "installed VCS metadata or a clean source checkout"
        ),
    )
    parser.add_argument("--summary-json", type=Path)


def build_calibrated_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a claim-bearing provider-v2 observation belief after runtime "
            "revision and prediction/calibration compatibility validation."
        )
    )
    _add_common_arguments(parser, require_source_revision=True)
    parser.add_argument(
        "--gauge-covariance-calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--point-uncertainty-calibration",
        type=Path,
        required=True,
    )
    return parser


def build_exploratory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export an explicitly exploratory provider-v2 observation belief. "
            "Uncalibrated covariance and labelled reconstruction controls are allowed."
        )
    )
    _add_common_arguments(parser, require_source_revision=False)
    parser.add_argument("--gauge-covariance-calibration", type=Path)
    parser.add_argument("--point-uncertainty-calibration", type=Path)
    parser.add_argument(
        "--gauge-mode",
        choices=("sequential", "fixed_lag"),
        default="sequential",
    )
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument(
        "--allow-approximate-fixed-lag-covariance",
        action="store_true",
        help=(
            "acknowledge that portable fixed-lag output retains historical marginal "
            "blocks rather than the complete all-window cross-covariance"
        ),
    )
    parser.add_argument(
        "--gauge-root-mode",
        choices=COVARIANCE_ROOT_MODES,
        default="canonical_eigenspaces",
    )
    parser.add_argument(
        "--allow-pointwise-covariance-fallback",
        action="store_true",
    )
    return parser


def _publish_artifact(args: argparse.Namespace, artifact: Any) -> int:
    _atomic_write_observation(args.output_npz, artifact)
    summary = {
        **artifact.summary(),
        "output": str(args.output_npz.resolve()),
        "provider_attestation": artifact.metadata["prob4d_provider_attestation"],
        "covariance_calibration": artifact.metadata.get("covariance_calibration"),
    }
    if args.summary_json is not None:
        _atomic_write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main_calibrated(argv: Sequence[str] | None = None) -> int:
    args = build_calibrated_parser().parse_args(argv)
    artifact = export_calibrated_observation_belief(
        args.predictions_manifest,
        case_id=args.case_id,
        causal_frame_stop=args.causal_frame_stop,
        metric_anchor=load_metric_gauge_anchor(args.metric_gauge_anchor),
        gauge_covariance_calibration=load_gauge_covariance_calibration(
            args.gauge_covariance_calibration
        ),
        point_uncertainty_calibration=load_point_uncertainty_calibration(
            args.point_uncertainty_calibration
        ),
        source_revision=args.source_revision,
        pixel_stride=args.pixel_stride,
        sampling_mode=args.sampling_mode,
        effective_samples_per_group=args.effective_samples_per_group,
        minimum_prior_reliability=args.minimum_prior_reliability,
        max_gauge_rank=args.max_gauge_rank,
        minimum_retained_gauge_trace=args.minimum_retained_gauge_trace,
        view_name=args.view_name,
    )
    return _publish_artifact(args, artifact)


def main_exploratory(argv: Sequence[str] | None = None) -> int:
    args = build_exploratory_parser().parse_args(argv)
    gauge_calibration = (
        None
        if args.gauge_covariance_calibration is None
        else load_gauge_covariance_calibration(args.gauge_covariance_calibration)
    )
    point_calibration = (
        None
        if args.point_uncertainty_calibration is None
        else load_point_uncertainty_calibration(args.point_uncertainty_calibration)
    )
    artifact = export_exploratory_observation_belief(
        args.predictions_manifest,
        case_id=args.case_id,
        causal_frame_stop=args.causal_frame_stop,
        metric_anchor=load_metric_gauge_anchor(args.metric_gauge_anchor),
        pixel_stride=args.pixel_stride,
        sampling_mode=args.sampling_mode,
        effective_samples_per_group=args.effective_samples_per_group,
        minimum_prior_reliability=args.minimum_prior_reliability,
        gauge_mode=args.gauge_mode,
        fixed_lag=args.fixed_lag,
        allow_approximate_fixed_lag_covariance=(
            args.allow_approximate_fixed_lag_covariance
        ),
        max_gauge_rank=args.max_gauge_rank,
        gauge_root_mode=args.gauge_root_mode,
        minimum_retained_gauge_trace=args.minimum_retained_gauge_trace,
        view_name=args.view_name,
        source_revision=args.source_revision,
        gauge_covariance_calibration=gauge_calibration,
        point_uncertainty_calibration=point_calibration,
        allow_pointwise_covariance_fallback=(
            args.allow_pointwise_covariance_fallback
        ),
    )
    return _publish_artifact(args, artifact)


if __name__ == "__main__":
    raise SystemExit(main_calibrated())


__all__ = [
    "build_calibrated_parser",
    "build_exploratory_parser",
    "main_calibrated",
    "main_exploratory",
]
