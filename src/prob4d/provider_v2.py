"""Safe-by-default Prob4D provider API for claim-bearing development.

Version 1 remains frozen for exact reproduction. Version 2 separates exploratory
and calibrated export entry points, validates calibration compatibility before
opening prediction payloads, and binds provider/runtime provenance into exports.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from . import provider_v1 as _v1
from .calibration_compatibility import (
    CalibrationCompatibilityError,
    PredictionCalibrationTargetV1,
    assert_calibration_pair_compatible,
    load_prediction_calibration_target,
    motioncrafter_model_identifier,
)
from .composition_jacobian import (
    CompositionJacobianMode,
    composition_jacobian_mode,
)
from .covariance_root import CovarianceRootMode, covariance_root_mode
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
from .runtime_revision import (
    RuntimeRevisionAttestation,
    assert_runtime_revision,
    inspect_runtime_revision,
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
        "analytic_sim3_composition_jacobians",
        "canonical_repeated_eigenspace_covariance_root",
        "explicit_exploratory_and_claim_bearing_exports",
        "provider_attested_observation_artifacts",
        "runtime_revision_attestation",
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
            "composition_jacobian_semantics": (
                "provider v2 propagates sequential Sim(3) gauge covariance with "
                "closed-form derivatives in log-scale, axis-angle, and translation "
                "coordinates; the SO(3) log branch cut fails closed and provider v1 "
                "retains its frozen finite-difference behavior"
            ),
            "covariance_root_semantics": (
                "version 2 uses a context-local canonical basis for numerically "
                "repeated covariance eigenspaces and fails closed if a rank boundary "
                "would split one; provider v1 retains its frozen legacy basis"
            ),
            "export_mode_semantics": (
                "exploratory and calibrated entry points are distinct; calibrated "
                "export fixes sequential gauge propagation, uses canonical repeated-"
                "eigenspace covariance roots, and forbids pointwise covariance fallback"
            ),
            "provider_attestation_semantics": (
                "every provider-v2 export embeds the version-2 manifest identity, "
                "export mode, covariance-root and composition-Jacobian modes, and "
                "runtime-revision evidence; claim-bearing export fails closed on "
                "unavailable, mismatched, dirty, or non-independent runtime provenance"
            ),
        }
    )
    limitations = dict(cast(dict[str, object], inherited["limitations"]))
    limitations["uncalibrated_export_is_default"] = False
    limitations["deployment_environment_revision_is_independent_vcs_evidence"] = False
    descriptor: dict[str, object] = {
        **inherited,
        "provider_api_version": PROVIDER_API_VERSION,
        "capabilities": capabilities,
        "limitations": limitations,
        "metadata": metadata,
    }
    manifest_id = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    return {"manifest_id": manifest_id, **descriptor}


def _provider_attested_artifact(
    artifact: ObservationBeliefExportV1,
    *,
    export_mode: str,
    covariance_root_mode_name: CovarianceRootMode,
    composition_jacobian_mode_name: CompositionJacobianMode,
    calibration_compatibility_validated: bool,
    runtime_attestation: RuntimeRevisionAttestation,
) -> ObservationBeliefExportV1:
    metadata = dict(artifact.metadata)
    manifest = prob4d_provider_manifest(
        provider_revision=artifact.source_revision,
    )
    calibration = metadata.get("covariance_calibration")
    calibration_ids: dict[str, object] = {
        "gauge_artifact_id": None,
        "point_artifact_id": None,
    }
    if isinstance(calibration, Mapping):
        calibration_ids = {
            "gauge_artifact_id": calibration.get("gauge_artifact_id"),
            "point_artifact_id": calibration.get("point_artifact_id"),
        }
    metadata["prob4d_provider_attestation"] = {
        "provider_api_version": PROVIDER_API_VERSION,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_revision": artifact.source_revision,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": export_mode,
        "claim_bearing": export_mode == "calibrated",
        "calibration_compatibility_validated": bool(
            calibration_compatibility_validated
        ),
        "calibration_artifact_ids": calibration_ids,
        "covariance_root_mode": covariance_root_mode_name,
        "composition_jacobian_mode": composition_jacobian_mode_name,
        "runtime_revision": runtime_attestation.as_metadata(),
    }
    return replace(artifact, metadata=metadata)


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
    gauge_root_mode: CovarianceRootMode = "canonical_eigenspaces",
    minimum_retained_gauge_trace: float = 0.999,
    view_name: str = "camera0",
    source_revision: str | None = None,
    uncertainty_model: DepthDisagreementModel | None = None,
    gauge_covariance_calibration: GaugeCovarianceCalibrationV1 | None = None,
    point_uncertainty_calibration: PointUncertaintyCalibrationV1 | None = None,
    allow_pointwise_covariance_fallback: bool = False,
) -> ObservationBeliefExportV1:
    """Export an explicitly exploratory, provider-attested observation belief.

    The output retains v1 artifact and causal-stream schemas. The distinct function
    name prevents an uncalibrated run from being mistaken for the claim-bearing API.
    Runtime provenance is recorded but is not required to be independently verified.
    """

    with (
        covariance_root_mode(gauge_root_mode),
        composition_jacobian_mode("analytic"),
    ):
        artifact = _v1.export_observation_belief(
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
    runtime_attestation = inspect_runtime_revision(artifact.source_revision)
    return _provider_attested_artifact(
        artifact,
        export_mode="exploratory",
        covariance_root_mode_name=gauge_root_mode,
        composition_jacobian_mode_name="analytic",
        calibration_compatibility_validated=False,
        runtime_attestation=runtime_attestation,
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
) -> ObservationBeliefExportV1:
    """Export a claim-bearing observation after strict provenance validation."""

    runtime_attestation = assert_runtime_revision(source_revision)
    target = load_prediction_calibration_target(manifest_path)
    assert_calibration_pair_compatible(
        gauge_covariance_calibration,
        point_uncertainty_calibration,
        target,
    )
    with (
        covariance_root_mode("canonical_eigenspaces"),
        composition_jacobian_mode("analytic"),
    ):
        artifact = _v1.export_calibrated_observation_belief(
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
    return _provider_attested_artifact(
        artifact,
        export_mode="calibrated",
        covariance_root_mode_name="canonical_eigenspaces",
        composition_jacobian_mode_name="analytic",
        calibration_compatibility_validated=True,
        runtime_attestation=runtime_attestation,
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
    "CompositionJacobianMode",
    "CovarianceRootMode",
    "GaugeCovarianceCalibrationV1",
    "MetricGaugeAnchor",
    "ObservationBeliefExportV1",
    "ObservationFactorBundle",
    "PointUncertaintyCalibrationV1",
    "PredictionCalibrationTargetV1",
    "RuntimeRevisionAttestation",
    "SamplingMode",
    "SelectedOverlapWindow",
    "assert_calibration_pair_compatible",
    "assert_runtime_revision",
    "bind_causal_stream_contract_v2",
    "export_calibrated_observation_belief",
    "export_exploratory_observation_belief",
    "inspect_runtime_revision",
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
