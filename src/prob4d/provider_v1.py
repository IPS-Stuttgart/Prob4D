"""Stable Prob4D provider API for Bayesian-PhysTwin and Causal4D.

Downstream repositories should import this module rather than experiment helpers
or underscore-prefixed implementation modules. Breaking provider changes require
a new versioned module instead of silently changing this surface.
"""

from __future__ import annotations

from pathlib import Path

from ._causal_observation_source import (
    CausalOverlapSelection,
    SelectedOverlapWindow,
    select_causal_overlap_windows,
)
from ._metric_gauge_anchor import (
    FIXED_EXTERNAL_CALIBRATION,
    METRIC_GAUGE_ANCHOR_SCHEMA,
    METRIC_GAUGE_ANCHOR_VERSION,
    PROPAGATED_JOINT_GAUGE_COVARIANCE,
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    prediction_window_sha256,
    save_metric_gauge_anchor,
)
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
)
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
    load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from .observation_validation import load_observation_belief_export
from .portable_observation import (
    APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT,
    JOINT_GAUGE_COVARIANCE_LAYOUT,
    JOINT_GAUGE_FACTOR_GROUP_SEMANTICS,
    PROB4D_OBSERVATION_CONTRACT_VERSION,
    build_prob4d_observation_belief,
    save_observation_belief_export,
)
from .provider_manifest import (
    PROB4D_PROVIDER_API_VERSION,
    prob4d_provider_manifest,
)
from .uncertainty import DepthDisagreementModel

PROVIDER_API_VERSION = PROB4D_PROVIDER_API_VERSION


def select_causal_source(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
) -> CausalOverlapSelection:
    """Select the complete independently decoded source prefix for prediction."""

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
    """Export a causally sealed portable observation belief.

    The production sequential mode carries the joint cross-window gauge
    covariance induced by the metric anchor and selected causal gauge tree through
    one shared low-rank latent factor. The fixed-lag mode is available only as an
    explicitly acknowledged approximate reconstruction control.
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
    "APPROXIMATE_FIXED_LAG_COVARIANCE_LAYOUT",
    "FIXED_EXTERNAL_CALIBRATION",
    "JOINT_GAUGE_COVARIANCE_LAYOUT",
    "JOINT_GAUGE_FACTOR_GROUP_SEMANTICS",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "PROB4D_OBSERVATION_CONTRACT_VERSION",
    "PROB4D_PROVIDER_API_VERSION",
    "PROPAGATED_JOINT_GAUGE_COVARIANCE",
    "PROVIDER_API_VERSION",
    "CausalOverlapSelection",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "SelectedOverlapWindow",
    "export_observation_belief",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "prediction_window_sha256",
    "prob4d_provider_manifest",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "select_causal_source",
    "write_observation_factor_bundle",
]
