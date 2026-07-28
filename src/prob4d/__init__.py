"""Probabilistic long-horizon fusion for MotionCrafter predictions."""

from importlib.metadata import PackageNotFoundError, version

from .calibration import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    load_gauge_covariance_calibration,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_point_uncertainty_calibration,
)
from .cross_fitted_disagreement import (
    CrossFittedDisagreementReport,
    accumulate_cross_fitted_disagreement,
)
from .data import PredictionWindow
from .evaluation_modes import (
    EvaluationModeResult,
    EvaluationModes,
    evaluate_sequence_modes,
)
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_export import (
    JointGaugePosterior,
    MetricGaugeAnchor,
    build_prob4d_observation_belief,
    deterministic_covariance_root,
    estimate_joint_gauge_tree,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
)
from .observation_factors import (
    LinearizedObservationFactor,
    ObservationFactor,
    ObservationFactorBundle,
    StackedObservationFactors,
    load_observation_factor_bundle,
    stack_observation_factors,
    write_observation_factor_bundle,
)
from .observation_validation import load_observation_belief_export
from .sim3 import Sim3

try:
    __version__ = version("prob4d")
except PackageNotFoundError:  # Source tree without installed distribution metadata.
    __version__ = "0.2.0"

__all__ = [
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "CrossFittedDisagreementReport",
    "EvaluationModeResult",
    "EvaluationModes",
    "GaugeCovarianceCalibrationV1",
    "JointGaugePosterior",
    "LinearizedObservationFactor",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactor",
    "ObservationFactorBundle",
    "PointUncertaintyCalibrationV1",
    "PredictionWindow",
    "Sim3",
    "StackedObservationFactors",
    "accumulate_cross_fitted_disagreement",
    "build_prob4d_observation_belief",
    "deterministic_covariance_root",
    "estimate_joint_gauge_tree",
    "evaluate_sequence_modes",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_point_uncertainty_calibration",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "stack_observation_factors",
    "write_observation_factor_bundle",
]
