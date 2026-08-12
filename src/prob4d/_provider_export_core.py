"""Private observation-export implementation shared by provider API version 2.

Prob4D 0.5 no longer publishes provider API version 1.  This private module keeps
the established export implementation and historical wire semantics available to
the current provider-v2 façade without retaining an importable public v1 surface.
Exact provider-v1 reproduction remains available from the Prob4D 0.4.1 release.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

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
from ._observation_factor_io import (
    load_observation_factor_bundle_v3,
    write_observation_factor_bundle_v3,
)
from .alignment import CovarianceFallbackPolicy, alignment_covariance_context
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
from .causal_stream_contract import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    bind_causal_stream_contract_v2,
)
from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_export import SamplingMode, build_prob4d_observation_belief
from .observation_factors import (
    OBSERVATION_FACTOR_SCHEMA,
    PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorBundle,
)
from .observation_validation import load_observation_belief_export
from .provider_manifest import (
    PROB4D_PROVIDER_API_VERSION,
    prob4d_provider_manifest,
)
from .uncertainty import DepthDisagreementModel

# Internal compatibility constants.  They are not re-exported by prob4d.api.v2.
PROVIDER_API_VERSION = PROB4D_PROVIDER_API_VERSION
OBSERVATION_FACTOR_SCHEMA_VERSION = PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION
load_observation_factor_bundle = load_observation_factor_bundle_v3
write_observation_factor_bundle = write_observation_factor_bundle_v3


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
    sampling_mode: SamplingMode = "fixed_grid",
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
    gauge_covariance_calibration: GaugeCovarianceCalibrationV1 | None = None,
    point_uncertainty_calibration: PointUncertaintyCalibrationV1 | None = None,
    allow_uncalibrated_exploratory_covariance: bool = True,
    allow_pointwise_covariance_fallback: bool = False,
) -> ObservationBeliefExportV1:
    """Build a causally sealed portable observation belief.

    Provider v2 selects the covariance-root and composition-Jacobian modes before
    entering this implementation.  Missing claim-bearing calibration fails closed;
    pointwise covariance fallback remains an explicit exploratory control.
    """

    if uncertainty_model is not None and point_uncertainty_calibration is not None:
        raise ValueError(
            "uncertainty_model and point_uncertainty_calibration are mutually exclusive"
        )
    if not allow_uncalibrated_exploratory_covariance and (
        gauge_covariance_calibration is None
        or point_uncertainty_calibration is None
    ):
        raise ValueError(
            "claim-bearing exports require both gauge and point covariance "
            "calibration artifacts"
        )

    resolved_uncertainty_model = (
        point_uncertainty_calibration.model
        if point_uncertainty_calibration is not None
        else uncertainty_model
    )
    fallback_policy: CovarianceFallbackPolicy = (
        "pointwise" if allow_pointwise_covariance_fallback else "error"
    )
    with alignment_covariance_context(
        calibration=gauge_covariance_calibration,
        fallback_policy=fallback_policy,
    ) as alignment_diagnostics:
        artifact = build_prob4d_observation_belief(
            manifest_path,
            case_id=case_id,
            causal_frame_stop=causal_frame_stop,
            metric_anchor=metric_anchor,
            pixel_stride=pixel_stride,
            sampling_mode=sampling_mode,
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
            uncertainty_model=resolved_uncertainty_model,
        )

    if isinstance(artifact, ObservationBeliefExportV1):
        if (
            gauge_covariance_calibration is not None
            and point_uncertainty_calibration is not None
        ):
            calibration_status = "calibrated"
        elif (
            gauge_covariance_calibration is not None
            or point_uncertainty_calibration is not None
        ):
            calibration_status = "partially_calibrated"
        else:
            calibration_status = "uncalibrated_exploratory"
        metadata = dict(artifact.metadata)
        metadata["covariance_calibration"] = {
            "status": calibration_status,
            "gauge_artifact_id": (
                None
                if gauge_covariance_calibration is None
                else gauge_covariance_calibration.artifact_id
            ),
            "point_artifact_id": (
                None
                if point_uncertainty_calibration is None
                else point_uncertainty_calibration.artifact_id
            ),
            "uncalibrated_exploratory_covariance_allowed": bool(
                allow_uncalibrated_exploratory_covariance
            ),
            "pointwise_covariance_fallback_allowed": bool(
                allow_pointwise_covariance_fallback
            ),
            "alignment_count": alignment_diagnostics.alignment_count,
            "gauge_calibrated_alignment_count": (
                alignment_diagnostics.calibrated_alignment_count
            ),
            "covariance_fallback_counts": alignment_diagnostics.fallback_counts,
        }
        artifact = replace(artifact, metadata=metadata)

    if gauge_mode != "sequential":
        return artifact
    return bind_causal_stream_contract_v2(
        artifact,
        metric_anchor=metric_anchor,
    )


def export_calibrated_observation_belief(
    manifest_path: str | Path,
    *,
    case_id: str,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
    gauge_covariance_calibration: GaugeCovarianceCalibrationV1,
    point_uncertainty_calibration: PointUncertaintyCalibrationV1,
    **kwargs: Any,
) -> ObservationBeliefExportV1:
    """Build a claim-bearing observation belief requiring both calibrations."""

    if "uncertainty_model" in kwargs:
        raise ValueError(
            "export_calibrated_observation_belief derives its uncertainty model "
            "from point_uncertainty_calibration"
        )
    if "allow_uncalibrated_exploratory_covariance" in kwargs:
        raise ValueError(
            "export_calibrated_observation_belief always fails closed on missing "
            "calibration"
        )
    return export_observation_belief(
        manifest_path,
        case_id=case_id,
        causal_frame_stop=causal_frame_stop,
        metric_anchor=metric_anchor,
        gauge_covariance_calibration=gauge_covariance_calibration,
        point_uncertainty_calibration=point_uncertainty_calibration,
        allow_uncalibrated_exploratory_covariance=False,
        **kwargs,
    )


__all__ = [
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "METRIC_GAUGE_ANCHOR_SCHEMA",
    "METRIC_GAUGE_ANCHOR_VERSION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "CausalOverlapSelection",
    "GaugeCovarianceCalibrationV1",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "PointUncertaintyCalibrationV1",
    "SamplingMode",
    "SelectedOverlapWindow",
    "bind_causal_stream_contract_v2",
    "export_calibrated_observation_belief",
    "export_observation_belief",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_point_uncertainty_calibration",
    "prob4d_provider_manifest",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "select_causal_source",
]
