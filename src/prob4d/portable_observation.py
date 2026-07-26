"""Stable portable-observation boundary for Prob4D Bayesian consumers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import observation_export as _core
from ._metric_gauge_anchor import MetricGaugeAnchor, load_metric_gauge_anchor
from .observation_contract import ObservationBeliefExportV1
from .observation_validation import load_observation_belief_export
from .uncertainty import DepthDisagreementModel

PROB4D_OBSERVATION_CONTRACT_VERSION = 2
JOINT_GAUGE_COVARIANCE_LAYOUT = "joint_sim3_tree_root_v1"
APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT = (
    "approximate_fixed_lag_block_diagonal_sim3_root_v1"
)
JOINT_GAUGE_FACTOR_GROUP_SEMANTICS = "single_shared_standard_normal_latent"


def enrich_prob4d_observation_belief(
    artifact: ObservationBeliefExportV1,
    *,
    metric_anchor: MetricGaugeAnchor,
) -> ObservationBeliefExportV1:
    """Attach the machine-readable provider contract required by consumers."""

    if artifact.source_repository != "FlorianPfaff/Prob4D":
        raise ValueError("portable Prob4D metadata cannot be added to another producer")
    if artifact.stream_id != "prob4d:causal-overlap-window-points":
        raise ValueError("portable Prob4D metadata requires the causal overlap stream")
    if not artifact.window_names or artifact.window_names[0] != metric_anchor.window_id:
        raise ValueError("metric gauge anchor does not identify the first window")
    if metric_anchor.case_id is not None and metric_anchor.case_id != artifact.case_id:
        raise ValueError("metric gauge-anchor case differs from observation case")

    metadata = dict(artifact.metadata)
    if metadata.get("coordinate_frame") != metric_anchor.world_frame_id:
        raise ValueError("metric gauge-anchor frame differs from observation frame")
    lineage = metadata.get("causal_source_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("Prob4D artifact has no causal source lineage")
    selected_windows = lineage.get("selected_windows")
    if not isinstance(selected_windows, list) or not selected_windows:
        raise ValueError("Prob4D artifact has no selected source windows")
    first_window = selected_windows[0]
    if not isinstance(first_window, Mapping):
        raise ValueError("Prob4D first selected source window is invalid")
    if first_window.get("window_id") != metric_anchor.window_id:
        raise ValueError("metric gauge anchor does not identify the first source window")
    if first_window.get("payload_sha256") != metric_anchor.source_artifact_sha256:
        raise ValueError("metric gauge anchor is not bound to the first source payload")
    gauge_posterior = metadata.get("gauge_posterior")
    if not isinstance(gauge_posterior, Mapping):
        raise ValueError("Prob4D artifact has no joint gauge-posterior metadata")
    factor_rank = len(artifact.factor_names)
    expected_names = tuple(
        f"joint_gauge_latent_{index:04d}" for index in range(factor_rank)
    )
    if artifact.factor_names != expected_names:
        raise ValueError("Prob4D joint gauge factor names changed unexpectedly")
    if int(gauge_posterior.get("exported_factor_rank", -1)) != factor_rank:
        raise ValueError("joint gauge metadata and exported factor rank differ")
    if int(gauge_posterior.get("window_count", -1)) != len(artifact.window_names):
        raise ValueError("joint gauge metadata and observation windows differ")
    cross_window_covariance_preserved = (
        gauge_posterior.get("cross_window_covariance_preserved") is True
    )
    if cross_window_covariance_preserved:
        covariance_layout = JOINT_GAUGE_COVARIANCE_LAYOUT
    else:
        if gauge_posterior.get("model") != (
            "fixed_lag_block_diagonal_approximation_v1"
        ):
            raise ValueError("unsupported non-joint gauge covariance layout")
        if metric_anchor.covariance_treatment != "fixed_external_calibration":
            raise ValueError(
                "approximate fixed-lag export requires a fixed metric anchor"
            )
        covariance_layout = APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT
    if artifact.factor_group_ids.size and not (
        artifact.factor_group_ids == artifact.factor_group_ids[0]
    ).all():
        raise ValueError("joint gauge factors must use one shared factor group")

    metadata.update(
        {
            "prob4d_observation_contract_version": (
                PROB4D_OBSERVATION_CONTRACT_VERSION
            ),
            "covariance_layout": covariance_layout,
            "factor_group_semantics": JOINT_GAUGE_FACTOR_GROUP_SEMANTICS,
            "factor_group_semantics_description": (
                "all rows are driven by one standard-normal latent vector; each "
                "window contributes its block of the same joint Sim(3) root"
            ),
            "metric_gauge_anchor": metric_anchor.contract_metadata(),
            "metric_anchor_covariance_included_in_joint_factor": True,
        }
    )
    return replace(artifact, metadata=metadata)


def build_prob4d_observation_belief(
    manifest_path: str | Path,
    *,
    case_id: str,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
    pixel_stride: int = 4,
    effective_samples_per_group: float = 64.0,
    minimum_prior_reliability: float = 0.05,
    gauge_mode: str = "sequential",
    fixed_lag: int = 4,
    allow_approximate_fixed_lag_covariance: bool = False,
    max_gauge_rank: int | None = 64,
    minimum_retained_gauge_trace: float = 0.999,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Build a causally sealed joint-gauge artifact with contract-v2 metadata."""

    metric_anchor.require_portable()
    artifact = _core.build_prob4d_observation_belief(
        manifest_path,
        case_id=case_id,
        causal_frame_stop=causal_frame_stop,
        metric_anchor=metric_anchor,
        pixel_stride=pixel_stride,
        effective_samples_per_group=effective_samples_per_group,
        minimum_prior_reliability=minimum_prior_reliability,
        gauge_mode=gauge_mode,
        fixed_lag=fixed_lag,
        allow_approximate_fixed_lag_covariance=(
            allow_approximate_fixed_lag_covariance
        ),
        max_gauge_rank=max_gauge_rank,
        minimum_retained_gauge_trace=minimum_retained_gauge_trace,
        view_name=view_name,
        source_revision=source_revision,
        uncertainty_model=uncertainty_model,
    )
    return enrich_prob4d_observation_belief(
        artifact,
        metric_anchor=metric_anchor,
    )


def save_observation_belief_export(
    path: str | Path,
    artifact: ObservationBeliefExportV1,
) -> None:
    """Atomically write, reload, and content-validate an observation artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".npz",
        delete=False,
    )
    temporary = Path(descriptor.name)
    descriptor.close()
    try:
        _core.save_observation_belief_export(temporary, artifact)
        restored = load_observation_belief_export(temporary)
        if restored.artifact_id != artifact.artifact_id:
            raise RuntimeError("observation artifact changed during serialization")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(descriptor.name)
    try:
        with descriptor:
            descriptor.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            descriptor.flush()
            os.fsync(descriptor.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--causal-frame-stop", type=int, required=True)
    parser.add_argument("--metric-gauge-anchor", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--effective-samples-per-group", type=float, default=64.0)
    parser.add_argument("--minimum-prior-reliability", type=float, default=0.05)
    parser.add_argument(
        "--gauge-mode",
        choices=("sequential", "fixed_lag"),
        default="sequential",
        help=(
            "sequential exports a full joint spanning-tree covariance; fixed_lag "
            "is an explicit approximate reconstruction ablation"
        ),
    )
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument(
        "--allow-approximate-fixed-lag-covariance",
        action="store_true",
        help=(
            "acknowledge that legacy fixed-lag covariance treats marginalized "
            "boundary gauges as exact"
        ),
    )
    parser.add_argument("--max-gauge-rank", type=int, default=64)
    parser.add_argument(
        "--minimum-retained-gauge-trace",
        type=float,
        default=0.999,
    )
    parser.add_argument("--view-name", default="camera0")
    parser.add_argument("--source-revision")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)

    anchor = load_metric_gauge_anchor(args.metric_gauge_anchor)
    anchor.require_portable()
    selection = _core.select_causal_overlap_windows(
        args.predictions_manifest,
        causal_frame_stop=args.causal_frame_stop,
        metric_anchor=anchor,
    )
    artifact = _core._build_prob4d_observation_belief(
        selection,
        case_id=args.case_id,
        causal_frame_stop=args.causal_frame_stop,
        metric_anchor=anchor,
        pixel_stride=args.pixel_stride,
        effective_samples_per_group=args.effective_samples_per_group,
        minimum_prior_reliability=args.minimum_prior_reliability,
        gauge_mode=args.gauge_mode,
        fixed_lag=args.fixed_lag,
        allow_approximate_fixed_lag_covariance=(
            args.allow_approximate_fixed_lag_covariance
        ),
        max_gauge_rank=args.max_gauge_rank,
        minimum_retained_gauge_trace=args.minimum_retained_gauge_trace,
        view_name=args.view_name,
        source_revision=args.source_revision,
    )
    artifact = enrich_prob4d_observation_belief(
        artifact,
        metric_anchor=anchor,
    )
    save_observation_belief_export(args.output_npz, artifact)
    summary = {
        **selection.run_summary(causal_frame_stop=args.causal_frame_stop),
        **artifact.summary(),
        "metric_gauge_anchor_id": anchor.artifact_id,
        "metric_anchor_covariance_treatment": anchor.covariance_treatment,
        "covariance_layout": artifact.metadata["covariance_layout"],
        "gauge_posterior": artifact.metadata["gauge_posterior"],
        "output": str(args.output_npz.resolve()),
    }
    if args.summary_json is not None:
        _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT",
    "JOINT_GAUGE_COVARIANCE_LAYOUT",
    "JOINT_GAUGE_FACTOR_GROUP_SEMANTICS",
    "PROB4D_OBSERVATION_CONTRACT_VERSION",
    "build_prob4d_observation_belief",
    "enrich_prob4d_observation_belief",
    "main",
    "save_observation_belief_export",
]
