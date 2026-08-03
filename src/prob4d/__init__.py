"""Probabilistic long-horizon fusion for MotionCrafter predictions."""

from importlib.metadata import PackageNotFoundError, version

from .alignment_cycles import (
    AlignmentCycleAudit,
    AlignmentCycleResidual,
    alignment_edge_id,
    audit_alignment_cycles,
)
from .calibration import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    fit_group_balanced_point_uncertainty_calibration,
    group_balanced_point_calibration_metadata,
    load_gauge_covariance_calibration,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_point_uncertainty_calibration,
)
from .causal_tracklets import (
    CausalTrackletReport,
    CausalTrackletSet,
    build_causal_scene_flow_tracklets,
    tracklets_to_observation_factors,
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
from .fusion import FusedSequence
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
from .observation_factor_stream import (
    OBSERVATION_FACTOR_STREAM_SCHEMA,
    OBSERVATION_FACTOR_STREAM_VERSION,
    ObservationFactorStreamUpdateV1,
    ObservationFactorStreamV1,
    append_observation_factor_bundle,
    load_observation_factor_stream,
    write_observation_factor_stream,
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
from .project_identity import (
    PROB4D_CANONICAL_REPOSITORY,
    PROB4D_FROZEN_ARTIFACT_REPOSITORY,
    PROB4D_GITHUB_REPOSITORY_ID,
    PROB4D_PROJECT_ID,
    PROB4D_REPOSITORY_ALIASES,
    canonical_prob4d_repository,
    is_prob4d_repository,
    prob4d_project_identity,
    validate_prob4d_project_identity,
)
from .sim3 import Sim3
from .source_reliability import (
    SOURCE_RELIABILITY_SCHEMA,
    SOURCE_RELIABILITY_VERSION,
    SourceReliabilityCalibrationReport,
    SourceReliabilityFeatures,
    SourceReliabilityModelV1,
    build_source_reliability_features,
    fit_group_balanced_source_reliability,
    load_source_reliability_model,
    save_source_reliability_model,
)
from .uncertainty import GroupBalancedCalibrationReport

try:
    __version__ = version("prob4d")
except PackageNotFoundError:  # Source tree without installed distribution metadata.
    __version__ = "0.3.1"

__all__ = [
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_STREAM_SCHEMA",
    "OBSERVATION_FACTOR_STREAM_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PROB4D_CANONICAL_REPOSITORY",
    "PROB4D_FROZEN_ARTIFACT_REPOSITORY",
    "PROB4D_GITHUB_REPOSITORY_ID",
    "PROB4D_PROJECT_ID",
    "PROB4D_REPOSITORY_ALIASES",
    "SOURCE_RELIABILITY_SCHEMA",
    "SOURCE_RELIABILITY_VERSION",
    "AlignmentCycleAudit",
    "AlignmentCycleResidual",
    "CausalTrackletReport",
    "CausalTrackletSet",
    "CrossFittedDisagreementReport",
    "EvaluationModeResult",
    "EvaluationModes",
    "FusedSequence",
    "GaugeCovarianceCalibrationV1",
    "GroupBalancedCalibrationReport",
    "JointGaugePosterior",
    "LinearizedObservationFactor",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactor",
    "ObservationFactorBundle",
    "ObservationFactorStreamUpdateV1",
    "ObservationFactorStreamV1",
    "PointUncertaintyCalibrationV1",
    "PredictionWindow",
    "Sim3",
    "SourceReliabilityCalibrationReport",
    "SourceReliabilityFeatures",
    "SourceReliabilityModelV1",
    "StackedObservationFactors",
    "accumulate_cross_fitted_disagreement",
    "alignment_edge_id",
    "append_observation_factor_bundle",
    "audit_alignment_cycles",
    "build_causal_scene_flow_tracklets",
    "build_prob4d_observation_belief",
    "build_source_reliability_features",
    "canonical_prob4d_repository",
    "deterministic_covariance_root",
    "estimate_joint_gauge_tree",
    "evaluate_sequence_modes",
    "fit_group_balanced_point_uncertainty_calibration",
    "fit_group_balanced_source_reliability",
    "group_balanced_point_calibration_metadata",
    "is_prob4d_repository",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_observation_factor_stream",
    "load_point_uncertainty_calibration",
    "load_source_reliability_model",
    "prob4d_project_identity",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "save_source_reliability_model",
    "stack_observation_factors",
    "tracklets_to_observation_factors",
    "validate_prob4d_project_identity",
    "write_observation_factor_bundle",
    "write_observation_factor_stream",
]
