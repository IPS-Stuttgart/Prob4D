"""Public Prob4D exports with compatibility-preserving lazy loading.
Importing the package root defines the historical export inventory without
eagerly importing calibration, fusion, provider, or experiment modules. Each
public attribute is loaded from its owning module on first access and then
cached in this module. New downstream code should prefer ``prob4d.api.v1`` or
``prob4d.api.v2`` for explicit compatibility promises.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

from ._version import __version__

_LAZY_EXPORT_GROUPS: Final = (
    (
        "prob4d.alignment_cycles",
        (
            "AlignmentCycleAudit",
            "AlignmentCycleResidual",
            "alignment_edge_id",
            "audit_alignment_cycles",
        ),
    ),
    (
        "prob4d.calibration",
        (
            "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
            "GAUGE_COVARIANCE_CALIBRATION_VERSION",
            "GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2",
            "LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1",
            "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
            "POINT_UNCERTAINTY_CALIBRATION_VERSION",
            "UPPER_WINSORIZED_MEAN_V1",
            "GaugeCovarianceCalibrationV1",
            "PointUncertaintyCalibrationV1",
            "fit_group_balanced_point_uncertainty_calibration",
            "group_balanced_point_calibration_metadata",
            "load_gauge_covariance_calibration",
            "load_point_uncertainty_calibration",
            "save_gauge_covariance_calibration",
            "save_point_uncertainty_calibration",
            "upper_winsorized_mean",
        ),
    ),
    (
        "prob4d.causal_gauge_graph",
        (
            "CAUSAL_GAUGE_GRAPH_DEPENDENCE",
            "CAUSAL_GAUGE_GRAPH_MODE",
            "CausalGaugeGraphReport",
            "CausalGaugeGraphStep",
            "estimate_causal_multi_edge_gauge_graph",
        ),
    ),
    (
        "prob4d.causal_tracklets",
        (
            "CausalTrackletReport",
            "CausalTrackletSet",
            "build_causal_scene_flow_tracklets",
            "tracklets_to_observation_factors",
        ),
    ),
    (
        "prob4d.cross_fitted_disagreement",
        (
            "CrossFittedDisagreementReport",
            "accumulate_cross_fitted_disagreement",
        ),
    ),
    (
        "prob4d.data",
        ("PredictionWindow",),
    ),
    (
        "prob4d.evaluation_modes",
        (
            "EvaluationModeResult",
            "EvaluationModes",
            "evaluate_sequence_modes",
        ),
    ),
    (
        "prob4d.finite_sample_threshold",
        (
            "FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS",
            "FiniteSampleUpperThreshold",
            "fit_finite_sample_upper_threshold",
        ),
    ),
    (
        "prob4d.fusion",
        ("FusedSequence",),
    ),
    (
        "prob4d.gauge_tree_prior",
        (
            "GAUGE_TREE_PRIOR_SCHEMA",
            "GAUGE_TREE_PRIOR_SEMANTICS",
            "GAUGE_TREE_PRIOR_VERSION",
            "GaugeTreeSquareRootPriorV1",
        ),
    ),
    (
        "prob4d.gauge_tree_prior_io",
        (
            "GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY",
            "GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA",
            "GAUGE_TREE_PRIOR_ARTIFACT_VERSION",
            "gauge_tree_prior_artifact_id",
            "load_gauge_tree_prior",
            "write_gauge_tree_prior",
        ),
    ),
    (
        "prob4d.guarded_causal_gauge_graph",
        (
            "GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE",
            "GUARDED_CAUSAL_GAUGE_GRAPH_MODE",
            "GuardedCausalGaugeGraphReport",
            "estimate_guarded_causal_multi_edge_gauge_graph",
        ),
    ),
    (
        "prob4d.observation_contract",
        (
            "OBSERVATION_BELIEF_SCHEMA",
            "OBSERVATION_BELIEF_VERSION",
            "ObservationBeliefExportV1",
            "save_observation_belief_export",
        ),
    ),
    (
        "prob4d.observation_export",
        (
            "JointGaugePosterior",
            "MetricGaugeAnchor",
            "build_prob4d_observation_belief",
            "deterministic_covariance_root",
            "estimate_joint_gauge_tree",
            "load_metric_gauge_anchor",
            "save_metric_gauge_anchor",
        ),
    ),
    (
        "prob4d.observation_factor_stream",
        (
            "OBSERVATION_FACTOR_STREAM_SCHEMA",
            "OBSERVATION_FACTOR_STREAM_VERSION",
            "ObservationFactorStreamUpdateV1",
            "ObservationFactorStreamV1",
            "append_observation_factor_bundle",
            "load_observation_factor_stream",
            "write_observation_factor_stream",
        ),
    ),
    (
        "prob4d.observation_factors",
        (
            "LinearizedObservationFactor",
            "ObservationFactor",
            "ObservationFactorBundle",
            "StackedObservationFactors",
            "load_observation_factor_bundle",
            "stack_observation_factors",
            "write_observation_factor_bundle",
        ),
    ),
    (
        "prob4d.observation_validation",
        ("load_observation_belief_export",),
    ),
    (
        "prob4d.prediction_store",
        (
            "PREDICTION_BUNDLE_STORE_SCHEMA",
            "PREDICTION_BUNDLE_STORE_VERSION",
            "PREDICTION_WINDOW_STORE_SCHEMA",
            "PREDICTION_WINDOW_STORE_VERSION",
            "MMapPredictionWindow",
            "load_prediction_bundle_store",
            "load_prediction_window_store",
            "materialize_prediction_bundle_store",
            "prediction_bundle_store_summary",
            "write_prediction_window_store",
        ),
    ),
    (
        "prob4d.project_identity",
        (
            "PROB4D_CANONICAL_REPOSITORY",
            "PROB4D_FROZEN_ARTIFACT_REPOSITORY",
            "PROB4D_GITHUB_REPOSITORY_ID",
            "PROB4D_PROJECT_ID",
            "PROB4D_REPOSITORY_ALIASES",
            "canonical_prob4d_repository",
            "is_prob4d_repository",
            "prob4d_project_identity",
            "validate_prob4d_project_identity",
        ),
    ),
    (
        "prob4d.sim3",
        ("Sim3",),
    ),
    (
        "prob4d.source_diagnostics",
        (
            "CommonModeFailureAudit",
            "SourceOnlyDiagnosticGrid",
            "audit_common_mode_failures",
            "augment_source_reliability_features",
            "build_common_gauge_seed_dispersion_diagnostic",
            "build_flow_point_consistency_diagnostic",
        ),
    ),
    (
        "prob4d.source_reliability",
        (
            "SOURCE_RELIABILITY_SCHEMA",
            "SOURCE_RELIABILITY_VERSION",
            "SourceReliabilityCalibrationReport",
            "SourceReliabilityFeatures",
            "SourceReliabilityModelV1",
            "build_source_reliability_features",
            "fit_group_balanced_source_reliability",
            "load_source_reliability_model",
            "save_source_reliability_model",
        ),
    ),
    (
        "prob4d.uncertainty",
        ("GroupBalancedCalibrationReport",),
    ),
)

_LAZY_EXPORTS: Final = {
    name: module_name for module_name, names in _LAZY_EXPORT_GROUPS for name in names
}

__all__ = [
    "CAUSAL_GAUGE_GRAPH_DEPENDENCE",
    "CAUSAL_GAUGE_GRAPH_MODE",
    "GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE",
    "GUARDED_CAUSAL_GAUGE_GRAPH_MODE",
    "GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY",
    "GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA",
    "GAUGE_TREE_PRIOR_ARTIFACT_VERSION",
    "GAUGE_TREE_PRIOR_SCHEMA",
    "GAUGE_TREE_PRIOR_SEMANTICS",
    "GAUGE_TREE_PRIOR_VERSION",
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2",
    "LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_STREAM_SCHEMA",
    "OBSERVATION_FACTOR_STREAM_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "UPPER_WINSORIZED_MEAN_V1",
    "PREDICTION_BUNDLE_STORE_SCHEMA",
    "PREDICTION_BUNDLE_STORE_VERSION",
    "PREDICTION_WINDOW_STORE_SCHEMA",
    "PREDICTION_WINDOW_STORE_VERSION",
    "PROB4D_CANONICAL_REPOSITORY",
    "PROB4D_FROZEN_ARTIFACT_REPOSITORY",
    "PROB4D_GITHUB_REPOSITORY_ID",
    "PROB4D_PROJECT_ID",
    "PROB4D_REPOSITORY_ALIASES",
    "SOURCE_RELIABILITY_SCHEMA",
    "SOURCE_RELIABILITY_VERSION",
    "AlignmentCycleAudit",
    "AlignmentCycleResidual",
    "CausalGaugeGraphReport",
    "CausalGaugeGraphStep",
    "GuardedCausalGaugeGraphReport",
    "CausalTrackletReport",
    "CausalTrackletSet",
    "CommonModeFailureAudit",
    "CrossFittedDisagreementReport",
    "EvaluationModeResult",
    "EvaluationModes",
    "FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS",
    "FiniteSampleUpperThreshold",
    "FusedSequence",
    "GaugeCovarianceCalibrationV1",
    "GaugeTreeSquareRootPriorV1",
    "GroupBalancedCalibrationReport",
    "JointGaugePosterior",
    "LinearizedObservationFactor",
    "MMapPredictionWindow",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactor",
    "ObservationFactorBundle",
    "ObservationFactorStreamUpdateV1",
    "ObservationFactorStreamV1",
    "PointUncertaintyCalibrationV1",
    "PredictionWindow",
    "Sim3",
    "SourceOnlyDiagnosticGrid",
    "SourceReliabilityCalibrationReport",
    "SourceReliabilityFeatures",
    "SourceReliabilityModelV1",
    "StackedObservationFactors",
    "__version__",
    "accumulate_cross_fitted_disagreement",
    "alignment_edge_id",
    "append_observation_factor_bundle",
    "audit_alignment_cycles",
    "audit_common_mode_failures",
    "augment_source_reliability_features",
    "build_causal_scene_flow_tracklets",
    "build_common_gauge_seed_dispersion_diagnostic",
    "build_flow_point_consistency_diagnostic",
    "build_prob4d_observation_belief",
    "build_source_reliability_features",
    "canonical_prob4d_repository",
    "deterministic_covariance_root",
    "estimate_causal_multi_edge_gauge_graph",
    "estimate_guarded_causal_multi_edge_gauge_graph",
    "estimate_joint_gauge_tree",
    "evaluate_sequence_modes",
    "fit_finite_sample_upper_threshold",
    "fit_group_balanced_point_uncertainty_calibration",
    "fit_group_balanced_source_reliability",
    "gauge_tree_prior_artifact_id",
    "group_balanced_point_calibration_metadata",
    "is_prob4d_repository",
    "load_gauge_covariance_calibration",
    "load_gauge_tree_prior",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_observation_factor_stream",
    "load_point_uncertainty_calibration",
    "load_prediction_bundle_store",
    "load_prediction_window_store",
    "load_source_reliability_model",
    "materialize_prediction_bundle_store",
    "prediction_bundle_store_summary",
    "prob4d_project_identity",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "save_source_reliability_model",
    "stack_observation_factors",
    "tracklets_to_observation_factors",
    "upper_winsorized_mean",
    "validate_prob4d_project_identity",
    "write_gauge_tree_prior",
    "write_observation_factor_bundle",
    "write_observation_factor_stream",
    "write_prediction_window_store",
]

if len(_LAZY_EXPORTS) != sum(len(names) for _, names in _LAZY_EXPORT_GROUPS):
    raise RuntimeError("duplicate Prob4D lazy export")
if set(__all__) != set(_LAZY_EXPORTS) | {"__version__"}:
    raise RuntimeError("Prob4D lazy exports and __all__ differ")


def __getattr__(name: str) -> object:
    """Load one historical top-level export from its owning module."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in module introspection."""

    return sorted(set(globals()) | set(__all__))
