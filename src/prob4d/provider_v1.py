"""Stable Prob4D provider API for Bayesian consumers.

Downstream projects should import this module rather than underscore-prefixed
source-selection or metric-anchor modules. Version 1 exposes the causal source
selector, the joint-gauge portable observation path, strict artifact loading,
and the richer unfused factor-bundle contract.
"""

from __future__ import annotations

from pathlib import Path

from ._causal_observation_source import (
    CausalOverlapSelection,
    SelectedOverlapWindow,
    select_causal_overlap_windows,
)
from ._metric_gauge_anchor import (
    METRIC_GAUGE_ANCHOR_SCHEMA,
    METRIC_GAUGE_ANCHOR_VERSION,
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
)
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_export import (
    JointGaugePosterior,
    build_prob4d_observation_belief,
    deterministic_covariance_root,
    estimate_joint_gauge_tree,
)
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
    load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from .observation_validation import load_observation_belief_export
from .provider_manifest import prob4d_provider_manifest
from .uncertainty import DepthDisagreementModel

PROVIDER_API_VERSION = 1


def select_causal_source(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
) -> CausalOverlapSelection:
    """Select complete independently decoded windows before the causal cutoff."""

    return select_causal_overlap_windows(
        manifest_path,
        causal_frame_stop=causal_frame_stop,
        metric_anchor=metric_anchor,
    )


def export_observation_belief(
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
    """Export a causally sealed, uncertainty-bearing observation belief.

    The production default propagates a fixed metric anchor and selected causal
    relative-gauge constraints into one joint cross-window covariance. The
    fixed-lag mode remains an explicitly acknowledged approximate reconstruction
    control.
    """

    return build_prob4d_observation_belief(
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


__all__ = [
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "PROVIDER_API_VERSION",
    "CausalOverlapSelection",
    "DepthDisagreementModel",
    "JointGaugePosterior",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "SelectedOverlapWindow",
    "deterministic_covariance_root",
    "estimate_joint_gauge_tree",
    "export_observation_belief",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "prob4d_provider_manifest",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "select_causal_source",
    "write_observation_factor_bundle",
]
