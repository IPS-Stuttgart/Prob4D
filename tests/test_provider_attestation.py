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


def _runtime(
    *,
    revision: str = "a" * 40,
    source: str = "source_checkout",
) -> dict[str, object]:
    if source == "source_checkout":
        return {
            "expected_revision": revision,
            "observed_revision": revision,
            "source": source,
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        }
    if source == "deployment_environment":
        return {
            "expected_revision": revision,
            "observed_revision": revision,
            "source": source,
            "clean_checkout": None,
            "matched": True,
            "independently_verified": False,
        }
    return {
        "expected_revision": revision,
        "observed_revision": None,
        "source": "unavailable",
        "clean_checkout": None,
        "matched": False,
        "independently_verified": False,
    }


def _calibrated_attestation(
    *,
    revision: str = "a" * 40,
) -> dict[str, object]:
    return build_provider_attestation(
        provider_manifest=provider.prob4d_provider_manifest(
            provider_revision=revision
        ),
        provider_revision=revision,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": "1" * 64,
            "point_artifact_id": "2" * 64,
        },
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision=_runtime(revision=revision),
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


def test_calibration_digest_is_not_coerced_from_an_integer() -> None:
    attestation = _calibrated_attestation()
    attestation["calibration_artifact_ids"]["gauge_artifact_id"] = int("1" * 64)

    with pytest.raises(ValueError, match="calibration gauge_artifact_id must be a string"):
        validate_provider_attestation(attestation, source_revision="a" * 40)


def test_runtime_revision_is_not_coerced_from_an_integer() -> None:
    revision = "1" * 40
    attestation = _calibrated_attestation(revision=revision)
    attestation["runtime_revision"]["observed_revision"] = int(revision)

    with pytest.raises(ValueError, match="runtime observed revision must be a string"):
        validate_provider_attestation(attestation, source_revision=revision)


def test_builder_rejects_truthy_non_boolean_compatibility_flag() -> None:
    with pytest.raises(ValueError, match="calibration compatibility flag must be Boolean"):
        build_provider_attestation(
            provider_manifest=provider.prob4d_provider_manifest(
                provider_revision="a" * 40
            ),
            provider_revision="a" * 40,
            export_mode="calibrated",
            calibration_compatibility_validated="false",  # type: ignore[arg-type]
            calibration_artifact_ids={
                "gauge_artifact_id": "1" * 64,
                "point_artifact_id": "2" * 64,
            },
            covariance_root_mode="canonical_eigenspaces",
            composition_jacobian_mode="analytic",
            runtime_revision=_runtime(),
        )


def test_manifest_hashing_rejects_nested_non_string_mapping_keys() -> None:
    with pytest.raises(ValueError, match="object keys must be strings"):
        compute_provider_manifest_id(
            {
                "metadata": {
                    1: "must not be normalized into a string key",
                },
            }
        )


def test_attestation_schema_version_requires_an_exact_integer() -> None:
    attestation = _calibrated_attestation()
    attestation["schema_version"] = 1.0

    with pytest.raises(ValueError, match="schema version must be an integer"):
        validate_provider_attestation(attestation, source_revision="a" * 40)


def test_manifest_schema_versions_reject_boolean_integer_aliases() -> None:
    attestation = _calibrated_attestation()
    manifest = attestation["provider_manifest"]
    manifest["artifact_schema_versions"]["ObservationBeliefV1"] = True
    manifest["manifest_id"] = compute_provider_manifest_id(manifest)
    attestation["provider_manifest_id"] = manifest["manifest_id"]

    with pytest.raises(ValueError, match="ObservationBeliefV1 schema version must be an integer"):
        validate_provider_attestation(attestation, source_revision="a" * 40)
