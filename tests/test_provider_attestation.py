from __future__ import annotations

import copy

import pytest

import prob4d.provider_v2 as provider
from prob4d.provider_attestation import (
    PROVIDER_ATTESTATION_SCHEMA,
    PROVIDER_ATTESTATION_VERSION,
    build_provider_attestation,
    compute_provider_manifest_id,
    validate_provider_attestation,
)


def _runtime(*, source: str = "source_checkout") -> dict[str, object]:
    if source == "source_checkout":
        return {
            "expected_revision": "a" * 40,
            "observed_revision": "a" * 40,
            "source": source,
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        }
    if source == "deployment_environment":
        return {
            "expected_revision": "a" * 40,
            "observed_revision": "a" * 40,
            "source": source,
            "clean_checkout": None,
            "matched": True,
            "independently_verified": False,
        }
    return {
        "expected_revision": "a" * 40,
        "observed_revision": None,
        "source": "unavailable",
        "clean_checkout": None,
        "matched": False,
        "independently_verified": False,
    }


def _calibrated_attestation() -> dict[str, object]:
    return build_provider_attestation(
        provider_manifest=provider.prob4d_provider_manifest(
            provider_revision="a" * 40
        ),
        provider_revision="a" * 40,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": "1" * 64,
            "point_artifact_id": "2" * 64,
        },
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision=_runtime(),
    )


def test_provider_attestation_embeds_a_verifiable_complete_manifest() -> None:
    attestation = _calibrated_attestation()
    validated = validate_provider_attestation(
        attestation,
        source_revision="a" * 40,
        require_claim_bearing=True,
    )

    assert validated["schema_name"] == PROVIDER_ATTESTATION_SCHEMA
    assert validated["schema_version"] == PROVIDER_ATTESTATION_VERSION
    assert validated["provider_manifest_id"] == validated["provider_manifest"][
        "manifest_id"
    ]
    assert compute_provider_manifest_id(validated["provider_manifest"]) == validated[
        "provider_manifest_id"
    ]
    assert validated["runtime_revision"]["independently_verified"] is True


def test_manifest_payload_tampering_is_rejected() -> None:
    attestation = _calibrated_attestation()
    tampered = copy.deepcopy(attestation)
    tampered["provider_manifest"]["provider_version"] = "999.0"

    with pytest.raises(ValueError, match="manifest ID does not match"):
        validate_provider_attestation(tampered, source_revision="a" * 40)


def test_rehashed_manifest_without_required_capability_is_rejected() -> None:
    attestation = _calibrated_attestation()
    tampered = copy.deepcopy(attestation)
    manifest = tampered["provider_manifest"]
    manifest["capabilities"].remove("runtime_revision_attestation")
    manifest["manifest_id"] = compute_provider_manifest_id(manifest)
    tampered["provider_manifest_id"] = manifest["manifest_id"]

    with pytest.raises(ValueError, match="required claim-bearing capabilities"):
        validate_provider_attestation(tampered, source_revision="a" * 40)


def test_attestation_is_bound_to_observation_source_revision() -> None:
    with pytest.raises(ValueError, match="observation source revision"):
        validate_provider_attestation(
            _calibrated_attestation(),
            source_revision="b" * 40,
        )


def test_claim_bearing_attestation_rejects_environment_only_revision() -> None:
    with pytest.raises(ValueError, match="independently matched runtime code"):
        build_provider_attestation(
            provider_manifest=provider.prob4d_provider_manifest(
                provider_revision="a" * 40
            ),
            provider_revision="a" * 40,
            export_mode="calibrated",
            calibration_compatibility_validated=True,
            calibration_artifact_ids={
                "gauge_artifact_id": "1" * 64,
                "point_artifact_id": "2" * 64,
            },
            covariance_root_mode="canonical_eigenspaces",
            composition_jacobian_mode="analytic",
            runtime_revision=_runtime(source="deployment_environment"),
        )


def test_exploratory_attestation_can_record_unavailable_runtime_and_no_calibration() -> None:
    attestation = build_provider_attestation(
        provider_manifest=provider.prob4d_provider_manifest(
            provider_revision="a" * 40
        ),
        provider_revision="a" * 40,
        export_mode="exploratory",
        calibration_compatibility_validated=False,
        calibration_artifact_ids={
            "gauge_artifact_id": None,
            "point_artifact_id": None,
        },
        covariance_root_mode="legacy_eigenvectors",
        composition_jacobian_mode="legacy_finite_difference",
        runtime_revision=_runtime(source="unavailable"),
    )

    validated = validate_provider_attestation(
        attestation,
        source_revision="a" * 40,
    )
    assert validated["claim_bearing"] is False
    assert validated["runtime_revision"]["matched"] is False


def test_attestation_schema_rejects_undeclared_fields() -> None:
    attestation = _calibrated_attestation()
    attestation["unexpected"] = True

    with pytest.raises(ValueError, match="fields changed"):
        validate_provider_attestation(attestation, source_revision="a" * 40)
