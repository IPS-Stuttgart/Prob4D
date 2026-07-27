"""Safe-by-default Prob4D provider API for claim-bearing development.

Version 1 remains frozen for exact reproduction. Version 2 separates exploratory
and calibrated export entry points and validates calibration compatibility against
prediction-manifest metadata before opening any decoded prediction payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from . import provider_v1 as _v1
from .alignment import (
    DEFAULT_COVARIANCE_CLUSTER_SIZE,
    DENSE_ALIGNMENT_COVARIANCE_METHOD,
)
from .calibration_compatibility import (
    POINT_UNCERTAINTY_COVARIANCE_METHOD,
    CalibrationCompatibilityError,
    PredictionCalibrationTargetV1,
    assert_calibration_pair_compatible,
    load_prediction_calibration_target,
    motioncrafter_model_identifier,
)
from .provider_v1 import (
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
    CausalOverlapSelection,
    GaugeCovarianceCalibrationV1,
    MetricGaugeAnchor,
    ObservationBeliefExportV1,
    ObservationFactorBundle,
    PointUncertaintyCalibrationV1,
    SamplingMode,
    SelectedOverlapWindow,
    bind_causal_stream_contract_v2,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_observation_belief_export,
    load_observation_factor_bundle,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_metric_gauge_anchor,
    save_observation_belief_export,
    save_point_uncertainty_calibration,
    select_causal_source,
    write_observation_factor_bundle,
)
from .uncertainty import DepthDisagreementModel

PROVIDER_API_VERSION = 2
PROB4D_PROVIDER_API_VERSION = PROVIDER_API_VERSION


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def prob4d_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the version-2 capability descriptor without altering v1."""

    inherited = dict(
        _v1.prob4d_provider_manifest(provider_revision=provider_revision)
    )
    inherited.pop("manifest_id", None)
    capabilities = list(cast(list[str], inherited["capabilities"]))
    for capability in (
        "explicit_exploratory_and_claim_bearing_exports",
        "strict_prediction_calibration_compatibility",
    ):
        if capability not in capabilities:
            capabilities.append(capability)
    metadata = dict(cast(dict[str, object], inherited["metadata"]))
    metadata.update(
        {
            "python_import_boundary": "prob4d.provider_v2",
            "calibration_compatibility_semantics": (
                "claim-bearing export validates source repository, MotionCrafter "
                "revision, canonical model configuration, image resolution, window "
                "geometry, covariance cluster size, and gauge/point covariance methods "
                "before opening prediction payloads"
            ),
            "export_mode_semantics": (
                "exploratory and calibrated entry points are distinct; calibrated "
                "export fixes sequential gauge propagation and forbids pointwise "
                "covariance fallback"
            ),
        }
    )
    limitations = dict(cast(dict[str, object], inherited["limitations"]))
    limitations["uncalibrated_export_is_default"] = False
    descriptor: dict[str, object] = {
        **inherited,
        "provider_api_version": PROVIDER_API_VERSION,
        "capabilities": capabilities,
        "limitations": limitations,
        "metadata": metadata,
    }
    manifest_id = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    return {"manifest_id": manifest_id, **descriptor}


def export_exploratory_observation_belief(
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
    allow_pointwise_covariance_fallback: bool = False,
) -> ObservationBeliefExportV1:
    """Export an explicitly exploratory observation belief.

    The output retains v1 artifact and causal-stream schemas. The distinct function
    name prevents an uncalibrated run from being mistaken for the claim-bearing API.
    """

    return _v1.export_observation_belief(
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
        uncertainty_model=uncertainty_model,
        gauge_covariance_calibration=gauge_covariance_calibration,
        point_uncertainty_calibration=point_uncertainty_calibration,
        allow_uncalibrated_exploratory_covariance=True,
        allow_pointwise_covariance_fallback=allow_pointwise_covariance_fallback,
    )


def export_calibrated_observation_belief(
    manifest_path: str | Path,
    *,
    case_id: str,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchor,
    gauge_covariance_calibration: GaugeCovarianceCalibrationV1,
    point_uncertainty_calibration: PointUncertaintyCalibrationV1,
    source_revision: str,
    pixel_stride: int = 4,
    sampling_mode: SamplingMode = "fixed_grid",
    effective_samples_per_group: float = 64.0,
    minimum_prior_reliability: float = 0.05,
    max_gauge_rank: int | None = 64,
    minimum_retained_gauge_trace: float = 0.999,
    view_name: str = "camera0",
    covariance_cluster_size: int = DEFAULT_COVARIANCE_CLUSTER_SIZE,
    gauge_covariance_method: str = DENSE_ALIGNMENT_COVARIANCE_METHOD,
    point_covariance_method: str = POINT_UNCERTAINTY_COVARIANCE_METHOD,
) -> ObservationBeliefExportV1:
    """Export a claim-bearing observation after strict compatibility validation."""

    target = load_prediction_calibration_target(
        manifest_path,
        covariance_cluster_size=covariance_cluster_size,
        gauge_covariance_method=gauge_covariance_method,
        point_covariance_method=point_covariance_method,
    )
    assert_calibration_pair_compatible(
        gauge_covariance_calibration,
        point_uncertainty_calibration,
        target,
    )
    return _v1.export_calibrated_observation_belief(
        manifest_path,
        case_id=case_id,
        causal_frame_stop=causal_frame_stop,
        metric_anchor=metric_anchor,
        gauge_covariance_calibration=gauge_covariance_calibration,
        point_uncertainty_calibration=point_uncertainty_calibration,
        source_revision=source_revision,
        pixel_stride=pixel_stride,
        sampling_mode=sampling_mode,
        effective_samples_per_group=effective_samples_per_group,
        minimum_prior_reliability=minimum_prior_reliability,
        gauge_mode="sequential",
        max_gauge_rank=max_gauge_rank,
        minimum_retained_gauge_trace=minimum_retained_gauge_trace,
        view_name=view_name,
        allow_pointwise_covariance_fallback=False,
    )


__all__ = [
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
    "PROB4D_PROVIDER_API_VERSION",
    "PROVIDER_API_VERSION",
    "CalibrationCompatibilityError",
    "CausalOverlapSelection",
    "GaugeCovarianceCalibrationV1",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "PointUncertaintyCalibrationV1",
    "PredictionCalibrationTargetV1",
    "SamplingMode",
    "SelectedOverlapWindow",
    "assert_calibration_pair_compatible",
    "bind_causal_stream_contract_v2",
    "export_calibrated_observation_belief",
    "export_exploratory_observation_belief",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_point_uncertainty_calibration",
    "load_prediction_calibration_target",
    "motioncrafter_model_identifier",
    "prob4d_provider_manifest",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_observation_belief_export",
    "save_point_uncertainty_calibration",
    "select_causal_source",
    "write_observation_factor_bundle",
]
