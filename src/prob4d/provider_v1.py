"""Stable Prob4D provider API for Bayesian-PhysTwin and Causal4D.

Downstream repositories should import this module rather than experiment helpers
or underscore-prefixed implementation modules. Version 1 exposes the causal
source selector, the fixed metric-anchor contract, the portable observation
belief writer, and the richer unfused factor-bundle contract.
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
from .observation_export import build_prob4d_observation_belief
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
    load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from .uncertainty import DepthDisagreementModel

PROVIDER_API_VERSION = 1


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
    gauge_mode: str = "fixed_lag",
    fixed_lag: int = 4,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
) -> ObservationBeliefExportV1:
    """Export a causally sealed portable observation belief.

    Version 1 is conditional on a fixed metric anchor and carries per-window
    gauge marginals as explicit low-rank nuisance factors. It intentionally does
    not claim to encode the full joint cross-window gauge posterior.
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
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "SelectedOverlapWindow",
    "export_observation_belief",
    "load_metric_gauge_anchor",
    "load_observation_factor_bundle",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "select_causal_source",
    "write_observation_factor_bundle",
]
