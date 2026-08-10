"""Stable ecosystem-facing Prob4D API version 1.

BayesianPhysTwin, Causal4D, and other downstream consumers should import this
module instead of the broad package root or experiment-specific implementation
modules. Breaking changes require a new ``prob4d.api.vN`` module.
"""

from __future__ import annotations

from .._version import __version__
from ..provider_v1 import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION,
    METRIC_GAUGE_ANCHOR_SCHEMA,
    METRIC_GAUGE_ANCHOR_VERSION,
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    PROVIDER_API_VERSION,
    CausalOverlapSelection,
    GaugeCovarianceCalibrationV1,
    MetricGaugeAnchor,
    ObservationBeliefExportV1,
    ObservationFactorBundle,
    PointUncertaintyCalibrationV1,
    SamplingMode,
    SelectedOverlapWindow,
    export_calibrated_observation_belief,
    export_observation_belief,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_observation_belief_export,
    load_observation_factor_bundle,
    load_point_uncertainty_calibration,
    prob4d_provider_manifest,
    save_gauge_covariance_calibration,
    save_metric_gauge_anchor,
    save_observation_belief_export,
    save_point_uncertainty_calibration,
    select_causal_source,
    write_observation_factor_bundle,
)

API_VERSION = 1

__all__ = [
    "API_VERSION",
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROVIDER_API_VERSION",
    "CausalOverlapSelection",
    "GaugeCovarianceCalibrationV1",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "PointUncertaintyCalibrationV1",
    "SamplingMode",
    "SelectedOverlapWindow",
    "__version__",
    "export_calibrated_observation_belief",
    "export_observation_belief",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_point_uncertainty_calibration",
    "prob4d_provider_manifest",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "select_causal_source",
    "write_observation_factor_bundle",
]
