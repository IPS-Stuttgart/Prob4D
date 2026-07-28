from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import prob4d.provider_v2 as provider
from prob4d.composition_jacobian import current_composition_jacobian_mode
from prob4d.covariance_root import current_covariance_root_mode
from prob4d.observation_contract import ObservationBeliefExportV1
from prob4d.runtime_revision import RuntimeRevisionAttestation

GAUGE_CALIBRATION_ID = "1" * 64
POINT_CALIBRATION_ID = "2" * 64


def _observation() -> ObservationBeliefExportV1:
    return ObservationBeliefExportV1(
        case_id="case-a",
        stream_id="prob4d:test",
        causal_frame_stop=2,
        view_names=("camera0",),
        window_names=("window0",),
        factor_names=(),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="d" * 64,
        declared_frame_ids=np.asarray([0]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]]),
        frame_ids=np.asarray([0]),
        entity_ids=np.asarray([0]),
        view_indices=np.asarray([0]),
        window_indices=np.asarray([0]),
        correlation_group_ids=np.asarray([0]),
        factor_group_ids=np.asarray([0]),
        prior_reliability=np.asarray([1.0]),
        association_probability=np.asarray([1.0]),
        local_covariance_m2=np.asarray([np.eye(3)]),
        low_rank_factor_m=np.empty((1, 3, 0)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([1.0]),
        group_composite_weight=np.asarray([1.0]),
        metadata={
            "existing": True,
            "covariance_calibration": {
                "gauge_artifact_id": GAUGE_CALIBRATION_ID,
                "point_artifact_id": POINT_CALIBRATION_ID,
            },
        },
    )


def _runtime_attestation(
    *,
    independently_verified: bool = True,
) -> RuntimeRevisionAttestation:
    return RuntimeRevisionAttestation(
        expected_revision="a" * 40,
        observed_revision="a" * 40,
        source=(
            "source_checkout"
            if independently_verified
            else "deployment_environment"
        ),
        clean_checkout=True if independently_verified else None,
        matched=True,
        independently_verified=independently_verified,
    )


def test_provider_v2_exposes_safe_capabilities() -> None:
    assert provider.PROVIDER_API_VERSION == 2
    assert provider.PROB4D_PROVIDER_API_VERSION == 2
    assert provider.JOINT_OBSERVATION_FACTOR_SCHEMA_VERSION == 4
    manifest = provider.prob4d_provider_manifest(provider_revision="a" * 40)
    assert manifest["provider_api_version"] == 2
    assert "strict_prediction_calibration_compatibility" in manifest["capabilities"]
    assert (
        "canonical_repeated_eigenspace_covariance_root" in manifest["capabilities"]
    )
    assert "analytic_sim3_composition_jacobians" in manifest["capabilities"]
    assert "runtime_revision_attestation" in manifest["capabilities"]
    assert "provider_attested_observation_artifacts" in manifest["capabilities"]
    assert "joint_observation_factor_gauge_covariance" in manifest["capabilities"]
    assert manifest["artifact_schema_versions"]["JointObservationFactorBundle"] == 4
    assert manifest["metadata"]["python_import_boundary"] == "prob4d.provider_v2"
    assert "canonical basis" in manifest["metadata"]["covariance_root_semantics"]
    assert "closed-form derivatives" in manifest["metadata"][
        "composition_jacobian_semantics"
    ]
    assert manifest["limitations"]["uncalibrated_export_is_default"] is False


def test_exploratory_export_is_explicit_context_local_and_attested(
    monkeypatch,
) -> None:
    original = _observation()
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured.update(
            manifest_path=manifest_path,
            root_mode=current_covariance_root_mode(),
            composition_mode=current_composition_jacobian_mode(),
            **kwargs,
        )
        return original

    monkeypatch.setattr(provider._v1, "export_observation_belief", fake_export)
    monkeypatch.setattr(
        provider,
        "inspect_runtime_revision",
        lambda revision: _runtime_attestation(independently_verified=False),
    )
    result = provider.export_exploratory_observation_belief(
        Path("predictions.json"),
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        sampling_mode="information_stratified",
        allow_pointwise_covariance_fallback=True,
        source_revision="a" * 40,
    )

    assert result.artifact_id != original.artifact_id
    assert captured["root_mode"] == "canonical_eigenspaces"
    assert captured["composition_mode"] == "analytic"
    assert current_covariance_root_mode() == "legacy_eigenvectors"
    assert current_composition_jacobian_mode() == "legacy_finite_difference"
    assert captured["allow_uncalibrated_exploratory_covariance"] is True
    assert captured["allow_pointwise_covariance_fallback"] is True
    assert captured["sampling_mode"] == "information_stratified"
    attestation = result.metadata["prob4d_provider_attestation"]
    assert attestation["schema_name"] == provider.PROVIDER_ATTESTATION_SCHEMA
    assert attestation["schema_version"] == provider.PROVIDER_ATTESTATION_VERSION
    assert attestation["export_mode"] == "exploratory"
    assert attestation["claim_bearing"] is False
    assert attestation["calibration_compatibility_validated"] is False
    assert attestation["composition_jacobian_mode"] == "analytic"
    assert attestation["provider_manifest_id"] == attestation["provider_manifest"][
        "manifest_id"
    ]
    assert attestation["runtime_revision"]["independently_verified"] is False
    provider.validate_provider_attestation(
        attestation,
        source_revision=result.source_revision,
    )


def test_exploratory_export_can_reproduce_legacy_root_basis(monkeypatch) -> None:
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured["root_mode"] = current_covariance_root_mode()
        captured["composition_mode"] = current_composition_jacobian_mode()
        return _observation()

    monkeypatch.setattr(provider._v1, "export_observation_belief", fake_export)
    monkeypatch.setattr(
        provider,
        "inspect_runtime_revision",
        lambda revision: _runtime_attestation(),
    )
    result = provider.export_exploratory_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        gauge_root_mode="legacy_eigenvectors",
        source_revision="a" * 40,
    )
    assert captured["root_mode"] == "legacy_eigenvectors"
    assert captured["composition_mode"] == "analytic"
    assert result.metadata["prob4d_provider_attestation"][
        "covariance_root_mode"
    ] == "legacy_eigenvectors"
    assert result.metadata["prob4d_provider_attestation"][
        "composition_jacobian_mode"
    ] == "analytic"


def test_calibrated_export_validates_runtime_and_calibration_before_delegating(
    monkeypatch,
) -> None:
    original = _observation()
    target = object()
    gauge = object()
    point = object()
    calls = []

    def fake_runtime(revision):
        calls.append(("runtime", revision))
        return _runtime_attestation()

    def fake_target(manifest_path, **kwargs):
        calls.append(("target", manifest_path, kwargs))
        return target

    def fake_assert(gauge_calibration, point_calibration, supplied_target):
        calls.append(
            (
                "assert",
                gauge_calibration,
                point_calibration,
                supplied_target,
            )
        )

    def fake_export(manifest_path, **kwargs):
        calls.append(
            (
                "export",
                manifest_path,
                kwargs,
                current_covariance_root_mode(),
                current_composition_jacobian_mode(),
            )
        )
        return original

    monkeypatch.setattr(provider, "assert_runtime_revision", fake_runtime)
    monkeypatch.setattr(provider, "load_prediction_calibration_target", fake_target)
    monkeypatch.setattr(provider, "assert_calibration_pair_compatible", fake_assert)
    monkeypatch.setattr(
        provider._v1,
        "export_calibrated_observation_belief",
        fake_export,
    )

    result = provider.export_calibrated_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        gauge_covariance_calibration=gauge,
        point_uncertainty_calibration=point,
        source_revision="a" * 40,
    )

    assert [call[0] for call in calls] == ["runtime", "target", "assert", "export"]
    export_kwargs = calls[-1][2]
    assert calls[-1][3] == "canonical_eigenspaces"
    assert calls[-1][4] == "analytic"
    assert current_covariance_root_mode() == "legacy_eigenvectors"
    assert current_composition_jacobian_mode() == "legacy_finite_difference"
    assert export_kwargs["gauge_mode"] == "sequential"
    assert export_kwargs["allow_pointwise_covariance_fallback"] is False
    assert export_kwargs["source_revision"] == "a" * 40
    attestation = result.metadata["prob4d_provider_attestation"]
    assert attestation["provider_api_version"] == 2
    assert attestation["export_mode"] == "calibrated"
    assert attestation["claim_bearing"] is True
    assert attestation["calibration_compatibility_validated"] is True
    assert attestation["composition_jacobian_mode"] == "analytic"
    assert attestation["calibration_artifact_ids"] == {
        "gauge_artifact_id": GAUGE_CALIBRATION_ID,
        "point_artifact_id": POINT_CALIBRATION_ID,
    }
    assert attestation["provider_manifest_id"] == attestation["provider_manifest"][
        "manifest_id"
    ]
    assert attestation["runtime_revision"]["matched"] is True
    provider.validate_provider_attestation(
        attestation,
        source_revision=result.source_revision,
        require_claim_bearing=True,
    )


def test_calibrated_export_stops_before_manifest_on_runtime_failure(monkeypatch) -> None:
    def fail_runtime(revision):
        raise RuntimeError("revision mismatch")

    monkeypatch.setattr(provider, "assert_runtime_revision", fail_runtime)

    def fail_target(*args, **kwargs):
        raise AssertionError("prediction manifest must remain unopened")

    monkeypatch.setattr(provider, "load_prediction_calibration_target", fail_target)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        provider.export_calibrated_observation_belief(
            "predictions.json",
            case_id="case-a",
            causal_frame_stop=10,
            metric_anchor=object(),
            gauge_covariance_calibration=object(),
            point_uncertainty_calibration=object(),
            source_revision="a" * 40,
        )


def test_calibrated_export_requires_explicit_source_revision() -> None:
    with pytest.raises(TypeError, match="source_revision"):
        provider.export_calibrated_observation_belief(
            "predictions.json",
            case_id="case-a",
            causal_frame_stop=10,
            metric_anchor=object(),
            gauge_covariance_calibration=object(),
            point_uncertainty_calibration=object(),
        )
