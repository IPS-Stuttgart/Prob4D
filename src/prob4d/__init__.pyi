"""Typing surface for the compatibility-preserving package-root exports."""

from prob4d.alignment_cycles import (
    AlignmentCycleAudit as AlignmentCycleAudit,
    AlignmentCycleResidual as AlignmentCycleResidual,
    alignment_edge_id as alignment_edge_id,
    audit_alignment_cycles as audit_alignment_cycles,
)
from prob4d.calibration import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA as GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION as GAUGE_COVARIANCE_CALIBRATION_VERSION,
    GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2 as GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
    LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1 as LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA as POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION as POINT_UNCERTAINTY_CALIBRATION_VERSION,
    UPPER_WINSORIZED_MEAN_V1 as UPPER_WINSORIZED_MEAN_V1,
    GaugeCovarianceCalibrationV1 as GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1 as PointUncertaintyCalibrationV1,
    fit_group_balanced_point_uncertainty_calibration as
    fit_group_balanced_point_uncertainty_calibration,
    group_balanced_point_calibration_metadata as group_balanced_point_calibration_metadata,
    load_gauge_covariance_calibration as load_gauge_covariance_calibration,
    load_point_uncertainty_calibration as load_point_uncertainty_calibration,
    save_gauge_covariance_calibration as save_gauge_covariance_calibration,
    save_point_uncertainty_calibration as save_point_uncertainty_calibration,
    upper_winsorized_mean as upper_winsorized_mean,
)
from prob4d.causal_gauge_graph import (
    CAUSAL_GAUGE_GRAPH_DEPENDENCE as CAUSAL_GAUGE_GRAPH_DEPENDENCE,
    CAUSAL_GAUGE_GRAPH_MODE as CAUSAL_GAUGE_GRAPH_MODE,
    CausalGaugeGraphReport as CausalGaugeGraphReport,
    CausalGaugeGraphStep as CausalGaugeGraphStep,
    estimate_causal_multi_edge_gauge_graph as estimate_causal_multi_edge_gauge_graph,
)
from prob4d.causal_tracklets import (
    CausalTrackletReport as CausalTrackletReport,
    CausalTrackletSet as CausalTrackletSet,
    build_causal_scene_flow_tracklets as build_causal_scene_flow_tracklets,
    tracklets_to_observation_factors as tracklets_to_observation_factors,
)
from prob4d.cross_fitted_disagreement import (
    CrossFittedDisagreementReport as CrossFittedDisagreementReport,
    accumulate_cross_fitted_disagreement as accumulate_cross_fitted_disagreement,
)
from prob4d.data import (
    PredictionWindow as PredictionWindow,
)
from prob4d.evaluation_modes import (
    EvaluationModeResult as EvaluationModeResult,
    EvaluationModes as EvaluationModes,
    evaluate_sequence_modes as evaluate_sequence_modes,
)
from prob4d.finite_sample_threshold import (
    FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS as FINITE_SAMPLE_UPPER_THRESHOLD_SEMANTICS,
    FiniteSampleUpperThreshold as FiniteSampleUpperThreshold,
    fit_finite_sample_upper_threshold as fit_finite_sample_upper_threshold,
)
from prob4d.fusion import (
    FusedSequence as FusedSequence,
)
from prob4d.gauge_tree_prior import (
    GAUGE_TREE_PRIOR_SCHEMA as GAUGE_TREE_PRIOR_SCHEMA,
    GAUGE_TREE_PRIOR_SEMANTICS as GAUGE_TREE_PRIOR_SEMANTICS,
    GAUGE_TREE_PRIOR_VERSION as GAUGE_TREE_PRIOR_VERSION,
    GaugeTreeSquareRootPriorV1 as GaugeTreeSquareRootPriorV1,
)
from prob4d.gauge_tree_prior_io import (
    GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY as GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY,
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA as GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    GAUGE_TREE_PRIOR_ARTIFACT_VERSION as GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
    gauge_tree_prior_artifact_id as gauge_tree_prior_artifact_id,
    load_gauge_tree_prior as load_gauge_tree_prior,
    write_gauge_tree_prior as write_gauge_tree_prior,
)
from prob4d.guarded_causal_gauge_graph import (
    GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE as GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE,
    GUARDED_CAUSAL_GAUGE_GRAPH_MODE as GUARDED_CAUSAL_GAUGE_GRAPH_MODE,
    GuardedCausalGaugeGraphReport as GuardedCausalGaugeGraphReport,
    estimate_guarded_causal_multi_edge_gauge_graph as
    estimate_guarded_causal_multi_edge_gauge_graph,
)
from prob4d.observation_contract import (
    OBSERVATION_BELIEF_SCHEMA as OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION as OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1 as ObservationBeliefExportV1,
    save_observation_belief_export as save_observation_belief_export,
)
from prob4d.observation_export import (
    JointGaugePosterior as JointGaugePosterior,
    MetricGaugeAnchor as MetricGaugeAnchor,
    build_prob4d_observation_belief as build_prob4d_observation_belief,
    deterministic_covariance_root as deterministic_covariance_root,
    estimate_joint_gauge_tree as estimate_joint_gauge_tree,
    load_metric_gauge_anchor as load_metric_gauge_anchor,
    save_metric_gauge_anchor as save_metric_gauge_anchor,
)
from prob4d.observation_factor_stream import (
    OBSERVATION_FACTOR_STREAM_SCHEMA as OBSERVATION_FACTOR_STREAM_SCHEMA,
    OBSERVATION_FACTOR_STREAM_VERSION as OBSERVATION_FACTOR_STREAM_VERSION,
    ObservationFactorStreamUpdateV1 as ObservationFactorStreamUpdateV1,
    ObservationFactorStreamV1 as ObservationFactorStreamV1,
    append_observation_factor_bundle as append_observation_factor_bundle,
    load_observation_factor_stream as load_observation_factor_stream,
    write_observation_factor_stream as write_observation_factor_stream,
)
from prob4d.observation_factors import (
    LinearizedObservationFactor as LinearizedObservationFactor,
    ObservationFactor as ObservationFactor,
    ObservationFactorBundle as ObservationFactorBundle,
    StackedObservationFactors as StackedObservationFactors,
    load_observation_factor_bundle as load_observation_factor_bundle,
    stack_observation_factors as stack_observation_factors,
    write_observation_factor_bundle as write_observation_factor_bundle,
)
from prob4d.observation_validation import (
    load_observation_belief_export as load_observation_belief_export,
)
from prob4d.prediction_store import (
    PREDICTION_BUNDLE_STORE_SCHEMA as PREDICTION_BUNDLE_STORE_SCHEMA,
    PREDICTION_BUNDLE_STORE_VERSION as PREDICTION_BUNDLE_STORE_VERSION,
    PREDICTION_WINDOW_STORE_SCHEMA as PREDICTION_WINDOW_STORE_SCHEMA,
    PREDICTION_WINDOW_STORE_VERSION as PREDICTION_WINDOW_STORE_VERSION,
    MMapPredictionWindow as MMapPredictionWindow,
    load_prediction_bundle_store as load_prediction_bundle_store,
    load_prediction_window_store as load_prediction_window_store,
    materialize_prediction_bundle_store as materialize_prediction_bundle_store,
    prediction_bundle_store_summary as prediction_bundle_store_summary,
    write_prediction_window_store as write_prediction_window_store,
)
from prob4d.project_identity import (
    PROB4D_CANONICAL_REPOSITORY as PROB4D_CANONICAL_REPOSITORY,
    PROB4D_FROZEN_ARTIFACT_REPOSITORY as PROB4D_FROZEN_ARTIFACT_REPOSITORY,
    PROB4D_GITHUB_REPOSITORY_ID as PROB4D_GITHUB_REPOSITORY_ID,
    PROB4D_PROJECT_ID as PROB4D_PROJECT_ID,
    PROB4D_REPOSITORY_ALIASES as PROB4D_REPOSITORY_ALIASES,
    canonical_prob4d_repository as canonical_prob4d_repository,
    is_prob4d_repository as is_prob4d_repository,
    prob4d_project_identity as prob4d_project_identity,
    validate_prob4d_project_identity as validate_prob4d_project_identity,
)
from prob4d.sim3 import (
    Sim3 as Sim3,
)
from prob4d.source_diagnostics import (
    CommonModeFailureAudit as CommonModeFailureAudit,
    SourceOnlyDiagnosticGrid as SourceOnlyDiagnosticGrid,
    audit_common_mode_failures as audit_common_mode_failures,
    augment_source_reliability_features as augment_source_reliability_features,
    build_common_gauge_seed_dispersion_diagnostic as build_common_gauge_seed_dispersion_diagnostic,
    build_flow_point_consistency_diagnostic as build_flow_point_consistency_diagnostic,
)
from prob4d.source_reliability import (
    SOURCE_RELIABILITY_SCHEMA as SOURCE_RELIABILITY_SCHEMA,
    SOURCE_RELIABILITY_VERSION as SOURCE_RELIABILITY_VERSION,
    SourceReliabilityCalibrationReport as SourceReliabilityCalibrationReport,
    SourceReliabilityFeatures as SourceReliabilityFeatures,
    SourceReliabilityModelV1 as SourceReliabilityModelV1,
    build_source_reliability_features as build_source_reliability_features,
    fit_group_balanced_source_reliability as fit_group_balanced_source_reliability,
    load_source_reliability_model as load_source_reliability_model,
    save_source_reliability_model as save_source_reliability_model,
)
from prob4d.uncertainty import (
    GroupBalancedCalibrationReport as GroupBalancedCalibrationReport,
)

__version__: str
