"""Self-contained provider-v2 attestation schema and validation.

The attestation embeds the complete content-addressed provider manifest so a
consumer can validate its semantics without importing Prob4D. The observation
artifact's own content address then binds the attestation and every manifest byte.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._immutable_json import plain_json

PROVIDER_ATTESTATION_SCHEMA = "prob4d.provider-attestation"
PROVIDER_ATTESTATION_VERSION = 1
PROVIDER_NAME = "prob4d"
PROVIDER_API_VERSION = 2
PROVIDER_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROVIDER_IMPORT_BOUNDARY = "prob4d.provider_v2"

_REQUIRED_CAPABILITIES = frozenset(
    {
        "analytic_sim3_composition_jacobians",
        "canonical_repeated_eigenspace_covariance_root",
        "explicit_exploratory_and_claim_bearing_exports",
        "provider_attested_observation_artifacts",
        "runtime_revision_attestation",
        "strict_prediction_calibration_compatibility",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "provider_api_version",
        "provider_manifest_id",
        "provider_manifest",
        "provider_revision",
        "python_import_boundary",
        "export_mode",
        "claim_bearing",
        "calibration_compatibility_validated",
        "calibration_artifact_ids",
        "covariance_root_mode",
        "composition_jacobian_mode",
        "runtime_revision",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "expected_revision",
        "observed_revision",
        "source",
        "clean_checkout",
        "matched",
        "independently_verified",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {
        "gauge_artifact_id",
        "point_artifact_id",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_string_mapping_keys(
    value: Any,
    *,
    name: str,
    path: str = "$",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                isinstance(key, str),
                f"{name} object keys must be strings at {path}",
            )
            _require_string_mapping_keys(
                item,
                name=name,
                path=f"{path}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _require_string_mapping_keys(
                item,
                name=name,
                path=f"{path}[{index}]",
            )


def _finite_json_mapping(value: Any, *, name: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    _require_string_mapping_keys(value, name=name)
    try:
        normalized = json.loads(
            json.dumps(
                plain_json(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error
    _require(isinstance(normalized, dict), f"{name} must be a JSON object")
    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    _require(
        not missing and not extra,
        f"{name} fields changed; missing={sorted(missing)}, extra={sorted(extra)}",
    )


def _require_boolean(value: Any, *, name: str) -> bool:
    _require(isinstance(value, bool), f"{name} must be Boolean")
    return cast(bool, value)


def _require_integer(value: Any, *, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return cast(int, value)


def _require_sha256(value: Any, *, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be a string")
    _require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _require_revision(value: Any, *, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be a string")
    _require(
        len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value),
        f"{name} must be an exact lowercase Git commit",
    )
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_provider_manifest_id(manifest: Mapping[str, Any]) -> str:
    """Return the manifest ID after removing its self-declared identifier."""

    normalized = _finite_json_mapping(manifest, name="provider manifest")
    normalized.pop("manifest_id", None)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def validate_provider_manifest(
    manifest: Any,
    *,
    expected_revision: str,
) -> dict[str, Any]:
    """Validate the embedded provider-v2 capability descriptor."""

    normalized = _finite_json_mapping(manifest, name="provider manifest")
    declared_id = _require_sha256(
        normalized.get("manifest_id", ""),
        name="provider manifest_id",
    )
    _require(
        compute_provider_manifest_id(normalized) == declared_id,
        "provider manifest ID does not match its descriptor",
    )
    revision = _require_revision(
        normalized.get("provider_revision", ""),
        name="provider manifest revision",
    )
    _require(
        revision == _require_revision(expected_revision, name="expected provider revision"),
        "provider manifest revision differs from the observation source revision",
    )
    _require(
        normalized.get("provider_name") == PROVIDER_NAME,
        "provider manifest has changed provider name",
    )
    provider_api_version = _require_integer(
        normalized.get("provider_api_version"),
        name="provider manifest API version",
    )
    _require(
        provider_api_version == PROVIDER_API_VERSION,
        "provider manifest is not API version 2",
    )

    capabilities = normalized.get("capabilities")
    _require(
        isinstance(capabilities, Sequence)
        and not isinstance(capabilities, (str, bytes))
        and all(isinstance(item, str) and item for item in capabilities)
        and len(capabilities) == len(set(capabilities)),
        "provider manifest capabilities must be unique nonempty strings",
    )
    capability_list = cast(Sequence[Any], capabilities)
    _require(
        _REQUIRED_CAPABILITIES.issubset(capability_list),
        "provider manifest lacks required claim-bearing capabilities",
    )

    schemas = normalized.get("artifact_schema_versions")
    _require(isinstance(schemas, Mapping), "provider artifact schemas must be a mapping")
    schema_mapping = cast(Mapping[str, Any], schemas)
    observation_belief_version = _require_integer(
        schema_mapping.get("ObservationBeliefV1"),
        name="ObservationBeliefV1 schema version",
    )
    causal_stream_version = _require_integer(
        schema_mapping.get("Prob4DCausalObservationStream"),
        name="Prob4DCausalObservationStream schema version",
    )
    _require(
        observation_belief_version == 1 and causal_stream_version == 2,
        "provider manifest declares unsupported observation schemas",
    )

    limitations = normalized.get("limitations")
    _require(isinstance(limitations, Mapping), "provider limitations must be a mapping")
    limitation_mapping = cast(Mapping[str, Any], limitations)
    _require(
        limitation_mapping.get("uncalibrated_export_is_default") is False,
        "provider-v2 manifest must not default to uncalibrated export",
    )
    _require(
        limitation_mapping.get("deployment_environment_revision_is_independent_vcs_evidence")
        is False,
        "provider manifest misstates deployment revision evidence",
    )

    metadata = normalized.get("metadata")
    _require(isinstance(metadata, Mapping), "provider metadata must be a mapping")
    metadata_mapping = cast(Mapping[str, Any], metadata)
    _require(
        metadata_mapping.get("source_repository") == PROVIDER_SOURCE_REPOSITORY,
        "provider manifest source repository changed",
    )
    _require(
        metadata_mapping.get("python_import_boundary") == PROVIDER_IMPORT_BOUNDARY,
        "provider manifest import boundary changed",
    )
    return normalized


def _validate_runtime_revision(
    value: Any,
    *,
    provider_revision: str,
    claim_bearing: bool,
) -> dict[str, Any]:
    runtime = _finite_json_mapping(value, name="runtime revision attestation")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, name="runtime revision attestation")
    expected = _require_revision(
        runtime.get("expected_revision", ""),
        name="runtime expected revision",
    )
    _require(
        expected == provider_revision,
        "runtime expected revision differs from provider revision",
    )
    observed_value = runtime.get("observed_revision")
    observed = (
        None
        if observed_value is None
        else _require_revision(observed_value, name="runtime observed revision")
    )
    source = runtime.get("source")
    _require(
        isinstance(source, str)
        and source
        in {
            "installed_vcs_metadata",
            "source_checkout",
            "deployment_environment",
            "unavailable",
        },
        "runtime revision source is unsupported",
    )
    clean = runtime.get("clean_checkout")
    _require(
        clean is None or isinstance(clean, bool),
        "runtime clean_checkout must be Boolean or null",
    )
    clean_value = cast(bool | None, clean)
    matched_value = _require_boolean(
        runtime.get("matched"),
        name="runtime matched",
    )
    independent_value = _require_boolean(
        runtime.get("independently_verified"),
        name="runtime independently_verified",
    )
    _require(
        matched_value is (observed == expected),
        "runtime matched flag disagrees with its revisions",
    )
    expected_independent = bool(
        matched_value
        and source in {"installed_vcs_metadata", "source_checkout"}
        and clean_value is not False
    )
    _require(
        independent_value is expected_independent,
        "runtime independent-verification flag disagrees with its evidence",
    )
    if source == "source_checkout":
        _require(
            isinstance(clean_value, bool),
            "source-checkout runtime evidence must declare checkout cleanliness",
        )
    else:
        _require(
            clean_value is None,
            "non-checkout runtime evidence cannot declare checkout cleanliness",
        )
    if claim_bearing:
        _require(
            matched_value and independent_value,
            "claim-bearing provider attestation requires independently matched runtime code",
        )
    return runtime


def _validate_calibration_ids(
    value: Any,
    *,
    claim_bearing: bool,
) -> dict[str, Any]:
    calibration = _finite_json_mapping(value, name="calibration artifact IDs")
    _require_exact_fields(calibration, _CALIBRATION_FIELDS, name="calibration artifact IDs")
    for field in sorted(_CALIBRATION_FIELDS):
        artifact_id = calibration.get(field)
        if artifact_id is not None:
            calibration[field] = _require_sha256(
                artifact_id,
                name=f"calibration {field}",
            )
    if claim_bearing:
        _require(
            all(calibration[field] is not None for field in _CALIBRATION_FIELDS),
            "claim-bearing provider attestation requires both calibration artifact IDs",
        )
    return calibration


def validate_provider_attestation(
    attestation: Mapping[str, Any],
    *,
    source_revision: str,
    require_claim_bearing: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a self-contained provider-v2 attestation."""

    require_claim_bearing_value = _require_boolean(
        require_claim_bearing,
        name="require_claim_bearing",
    )
    normalized = _finite_json_mapping(attestation, name="provider attestation")
    _require_exact_fields(normalized, _ATTESTATION_FIELDS, name="provider attestation")
    _require(
        normalized.get("schema_name") == PROVIDER_ATTESTATION_SCHEMA,
        "unsupported provider-attestation schema",
    )
    schema_version = _require_integer(
        normalized.get("schema_version"),
        name="provider-attestation schema version",
    )
    _require(
        schema_version == PROVIDER_ATTESTATION_VERSION,
        "unsupported provider-attestation version",
    )
    provider_api_version = _require_integer(
        normalized.get("provider_api_version"),
        name="provider-attestation API version",
    )
    _require(
        provider_api_version == PROVIDER_API_VERSION,
        "provider attestation is not API version 2",
    )

    revision = _require_revision(
        normalized.get("provider_revision", ""),
        name="provider attestation revision",
    )
    _require(
        revision == _require_revision(source_revision, name="observation source revision"),
        "provider attestation revision differs from the observation source revision",
    )
    _require(
        normalized.get("python_import_boundary") == PROVIDER_IMPORT_BOUNDARY,
        "provider attestation import boundary changed",
    )

    manifest = validate_provider_manifest(
        normalized.get("provider_manifest"),
        expected_revision=revision,
    )
    declared_manifest_id = _require_sha256(
        normalized.get("provider_manifest_id", ""),
        name="provider attestation manifest ID",
    )
    _require(
        declared_manifest_id == manifest["manifest_id"],
        "provider attestation manifest ID differs from the embedded manifest",
    )

    export_mode = normalized.get("export_mode")
    _require(
        isinstance(export_mode, str) and export_mode in {"calibrated", "exploratory"},
        "provider attestation export mode is unsupported",
    )
    claim_bearing = _require_boolean(
        normalized.get("claim_bearing"),
        name="claim_bearing",
    )
    _require(
        claim_bearing is (export_mode == "calibrated"),
        "provider claim-bearing flag disagrees with export mode",
    )
    compatibility = _require_boolean(
        normalized.get("calibration_compatibility_validated"),
        name="calibration compatibility flag",
    )
    _require(
        compatibility is claim_bearing,
        "calibration compatibility flag disagrees with export mode",
    )
    if require_claim_bearing_value:
        _require(claim_bearing, "a claim-bearing provider-v2 artifact is required")
    calibration = _validate_calibration_ids(
        normalized.get("calibration_artifact_ids"),
        claim_bearing=claim_bearing,
    )
    covariance_mode = normalized.get("covariance_root_mode")
    _require(
        isinstance(covariance_mode, str)
        and covariance_mode in {"canonical_eigenspaces", "legacy_eigenvectors"},
        "provider covariance-root mode is unsupported",
    )
    composition_mode = normalized.get("composition_jacobian_mode")
    _require(
        isinstance(composition_mode, str)
        and composition_mode in {"analytic", "legacy_finite_difference"},
        "provider composition-Jacobian mode is unsupported",
    )
    if claim_bearing:
        _require(
            covariance_mode == "canonical_eigenspaces",
            "claim-bearing provider-v2 artifact requires canonical covariance roots",
        )
        _require(
            composition_mode == "analytic",
            "claim-bearing provider-v2 artifact requires analytic composition Jacobians",
        )

    runtime = _validate_runtime_revision(
        normalized.get("runtime_revision"),
        provider_revision=revision,
        claim_bearing=claim_bearing,
    )
    normalized["provider_manifest"] = manifest
    normalized["calibration_artifact_ids"] = calibration
    normalized["runtime_revision"] = runtime
    return normalized


def build_provider_attestation(
    *,
    provider_manifest: Mapping[str, Any],
    provider_revision: str,
    export_mode: str,
    calibration_compatibility_validated: bool,
    calibration_artifact_ids: Mapping[str, Any],
    covariance_root_mode: str,
    composition_jacobian_mode: str,
    runtime_revision: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct an attestation and validate the producer's own output."""

    compatibility = _require_boolean(
        calibration_compatibility_validated,
        name="calibration compatibility flag",
    )
    manifest = validate_provider_manifest(
        provider_manifest,
        expected_revision=provider_revision,
    )
    payload: dict[str, Any] = {
        "schema_name": PROVIDER_ATTESTATION_SCHEMA,
        "schema_version": PROVIDER_ATTESTATION_VERSION,
        "provider_api_version": PROVIDER_API_VERSION,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_manifest": manifest,
        "provider_revision": provider_revision,
        "python_import_boundary": PROVIDER_IMPORT_BOUNDARY,
        "export_mode": export_mode,
        "claim_bearing": export_mode == "calibrated",
        "calibration_compatibility_validated": compatibility,
        "calibration_artifact_ids": dict(calibration_artifact_ids),
        "covariance_root_mode": covariance_root_mode,
        "composition_jacobian_mode": composition_jacobian_mode,
        "runtime_revision": dict(runtime_revision),
    }
    return validate_provider_attestation(
        payload,
        source_revision=provider_revision,
        require_claim_bearing=export_mode == "calibrated",
    )


__all__ = [
    "PROVIDER_API_VERSION",
    "PROVIDER_ATTESTATION_SCHEMA",
    "PROVIDER_ATTESTATION_VERSION",
    "PROVIDER_IMPORT_BOUNDARY",
    "PROVIDER_NAME",
    "PROVIDER_SOURCE_REPOSITORY",
    "build_provider_attestation",
    "compute_provider_manifest_id",
    "validate_provider_attestation",
    "validate_provider_manifest",
]
