"""Provider-neutral identity validation for claim-bearing provider evaluation.

Current fused prediction artifacts carry a free-form metadata mapping. This
module gives provider evaluation one strict, versioned identity record inside
that mapping while preserving deterministic replay of historical MotionCrafter
artifacts. Per-case manifest and run identities are validated and reported, but
only provider-contract fields participate in the cross-case method signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from ._immutable_json import plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_mapping,
    require_revision,
    require_sha256,
)
from .data import DENSE_STORAGE_DTYPES
from .io import FusedPredictionMetadata
from .prediction_provider_manifest import (
    COORDINATE_SEMANTICS,
    FLOW_SEMANTICS,
    POINT_SEMANTICS,
    RAY_SEMANTICS,
    SOURCE_DEPENDENCY_SEMANTICS,
)

PROVIDER_EVALUATION_IDENTITY_METADATA_KEY: Final = "provider_identity"
PROVIDER_EVALUATION_IDENTITY_SCHEMA: Final = (
    "prob4d.provider-evaluation-provider-identity"
)
PROVIDER_EVALUATION_IDENTITY_VERSION: Final = 1

IdentityFormat = Literal[
    "prediction-provider-manifest-v1",
    "motioncrafter-legacy-v1",
]

_GENERIC_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "provider_manifest_id",
        "provider_manifest_sha256",
        "provider_family",
        "provider_repository",
        "provider_revision",
        "provider_run_id",
        "model_set_id",
        "loader_id",
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "source_dependency_semantics",
    }
)
_LEGACY_MOTIONCRAFTER_FIELDS: Final = frozenset(
    {
        "motioncrafter_revision",
        "motioncrafter_model_set_sha256",
        "motioncrafter_seed_policy",
        "prediction_manifest_sha256",
    }
)
_LEGACY_SEED_POLICIES: Final = frozenset({"legacy-common", "derived-per-call"})


def _repository(value: object, *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result.count("/") != 1 or result.startswith("/") or result.endswith("/"):
        raise ValueError(f"{name} must use canonical owner/name form")
    return result


def _choice(value: object, choices: tuple[str, ...], *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")
    return result


def _legacy_seed_policy(value: object, *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result not in _LEGACY_SEED_POLICIES:
        raise ValueError(
            f"{name} must declare one of {sorted(_LEGACY_SEED_POLICIES)}"
        )
    return result


@dataclass(frozen=True, slots=True)
class ProviderEvaluationIdentity:
    """Validated provider identity normalized across generic and legacy inputs."""

    identity_format: IdentityFormat
    provider_revision: str
    model_set_id: str
    provider_manifest_sha256: str
    provider_family: str | None = None
    provider_repository: str | None = None
    provider_run_id: str | None = None
    provider_manifest_id: str | None = None
    loader_id: str | None = None
    coordinate_semantics: str | None = None
    point_semantics: str | None = None
    flow_semantics: str | None = None
    ray_semantics: str | None = None
    source_dependency_semantics: str | None = None
    legacy_seed_policy: str | None = None

    def contract_signature(self) -> tuple[object, ...]:
        """Return the provider contract that must remain stable across cases."""

        if self.identity_format == "motioncrafter-legacy-v1":
            return (
                self.identity_format,
                self.provider_revision,
                self.model_set_id,
                self.legacy_seed_policy,
            )
        return (
            self.identity_format,
            self.provider_family,
            self.provider_repository,
            self.provider_revision,
            self.model_set_id,
            self.loader_id,
            self.coordinate_semantics,
            self.point_semantics,
            self.flow_semantics,
            self.ray_semantics,
            self.source_dependency_semantics,
        )

    def contract_record(self) -> dict[str, object]:
        """Return report-safe provider fields that are invariant across cases."""

        if self.identity_format == "motioncrafter-legacy-v1":
            return {
                "identity_format": self.identity_format,
                "provider_revision": self.provider_revision,
                "model_set_id": self.model_set_id,
                "legacy_seed_policy": self.legacy_seed_policy,
            }
        return {
            "identity_format": self.identity_format,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "coordinate_semantics": self.coordinate_semantics,
            "point_semantics": self.point_semantics,
            "flow_semantics": self.flow_semantics,
            "ray_semantics": self.ray_semantics,
            "source_dependency_semantics": self.source_dependency_semantics,
        }

    def evidence_record(self) -> dict[str, object]:
        """Return the complete normalized per-case identity for diagnostics."""

        return {
            **self.contract_record(),
            "provider_manifest_id": self.provider_manifest_id,
            "provider_manifest_sha256": self.provider_manifest_sha256,
            "provider_run_id": self.provider_run_id,
        }


def _generic_identity(value: object, *, path: Path) -> ProviderEvaluationIdentity:
    mapping = require_mapping(
        value,
        name=f"{path} metadata.{PROVIDER_EVALUATION_IDENTITY_METADATA_KEY}",
    )
    require_exact_fields(
        mapping,
        _GENERIC_IDENTITY_FIELDS,
        name=f"{path} provider identity",
    )
    if mapping["schema_name"] != PROVIDER_EVALUATION_IDENTITY_SCHEMA:
        raise ValueError(f"{path} provider identity schema changed")
    version = require_exact_integer(
        mapping["schema_version"],
        name=f"{path} provider identity schema_version",
        minimum=1,
    )
    if version != PROVIDER_EVALUATION_IDENTITY_VERSION:
        raise ValueError(f"{path} provider identity version changed")
    source_semantics = require_exact_string(
        mapping["source_dependency_semantics"],
        name=f"{path} provider identity source_dependency_semantics",
    )
    if source_semantics != SOURCE_DEPENDENCY_SEMANTICS:
        raise ValueError(f"{path} provider identity source dependency semantics changed")
    return ProviderEvaluationIdentity(
        identity_format="prediction-provider-manifest-v1",
        provider_family=require_exact_string(
            mapping["provider_family"],
            name=f"{path} provider identity provider_family",
        ),
        provider_repository=_repository(
            mapping["provider_repository"],
            name=f"{path} provider identity provider_repository",
        ),
        provider_revision=require_revision(
            mapping["provider_revision"],
            name=f"{path} provider identity provider_revision",
        ),
        provider_run_id=require_sha256(
            mapping["provider_run_id"],
            name=f"{path} provider identity provider_run_id",
        ),
        provider_manifest_id=require_sha256(
            mapping["provider_manifest_id"],
            name=f"{path} provider identity provider_manifest_id",
        ),
        provider_manifest_sha256=require_sha256(
            mapping["provider_manifest_sha256"],
            name=f"{path} provider identity provider_manifest_sha256",
        ),
        model_set_id=require_sha256(
            mapping["model_set_id"],
            name=f"{path} provider identity model_set_id",
        ),
        loader_id=require_sha256(
            mapping["loader_id"],
            name=f"{path} provider identity loader_id",
        ),
        coordinate_semantics=_choice(
            mapping["coordinate_semantics"],
            COORDINATE_SEMANTICS,
            name=f"{path} provider identity coordinate_semantics",
        ),
        point_semantics=_choice(
            mapping["point_semantics"],
            POINT_SEMANTICS,
            name=f"{path} provider identity point_semantics",
        ),
        flow_semantics=_choice(
            mapping["flow_semantics"],
            FLOW_SEMANTICS,
            name=f"{path} provider identity flow_semantics",
        ),
        ray_semantics=_choice(
            mapping["ray_semantics"],
            RAY_SEMANTICS,
            name=f"{path} provider identity ray_semantics",
        ),
        source_dependency_semantics=source_semantics,
    )


def _legacy_identity(
    details: Mapping[str, Any],
    *,
    path: Path,
) -> ProviderEvaluationIdentity:
    missing = sorted(_LEGACY_MOTIONCRAFTER_FIELDS - set(details))
    if missing:
        raise ValueError(
            f"{path} metadata requires provider_identity or the complete historical "
            f"MotionCrafter identity; missing={missing}"
        )
    return ProviderEvaluationIdentity(
        identity_format="motioncrafter-legacy-v1",
        provider_revision=require_revision(
            details["motioncrafter_revision"],
            name=f"{path} metadata.motioncrafter_revision",
        ),
        model_set_id=require_sha256(
            details["motioncrafter_model_set_sha256"],
            name=f"{path} metadata.motioncrafter_model_set_sha256",
        ),
        provider_manifest_sha256=require_sha256(
            details["prediction_manifest_sha256"],
            name=f"{path} metadata.prediction_manifest_sha256",
        ),
        legacy_seed_policy=_legacy_seed_policy(
            details["motioncrafter_seed_policy"],
            name=f"{path} metadata.motioncrafter_seed_policy",
        ),
    )


def _validate_optional_legacy_mirror(
    details: Mapping[str, Any],
    identity: ProviderEvaluationIdentity,
    *,
    path: Path,
) -> None:
    present = _LEGACY_MOTIONCRAFTER_FIELDS & set(details)
    if not present:
        return
    missing = sorted(_LEGACY_MOTIONCRAFTER_FIELDS - set(details))
    if missing:
        raise ValueError(
            f"{path} metadata mixes generic and partial MotionCrafter identity; "
            f"missing={missing}"
        )
    legacy_revision = require_revision(
        details["motioncrafter_revision"],
        name=f"{path} metadata.motioncrafter_revision",
    )
    legacy_model_set = require_sha256(
        details["motioncrafter_model_set_sha256"],
        name=f"{path} metadata.motioncrafter_model_set_sha256",
    )
    legacy_manifest_sha256 = require_sha256(
        details["prediction_manifest_sha256"],
        name=f"{path} metadata.prediction_manifest_sha256",
    )
    _legacy_seed_policy(
        details["motioncrafter_seed_policy"],
        name=f"{path} metadata.motioncrafter_seed_policy",
    )
    if identity.provider_family != "motioncrafter":
        raise ValueError(
            f"{path} metadata carries MotionCrafter compatibility fields for a "
            "different provider_family"
        )
    if legacy_revision != identity.provider_revision:
        raise ValueError(f"{path} MotionCrafter compatibility revision changed")
    if legacy_model_set != identity.model_set_id:
        raise ValueError(f"{path} MotionCrafter compatibility model set changed")
    if legacy_manifest_sha256 != identity.provider_manifest_sha256:
        raise ValueError(f"{path} MotionCrafter compatibility manifest digest changed")


def provider_identity_from_metadata(
    details: Mapping[str, Any],
    *,
    path: Path,
) -> ProviderEvaluationIdentity:
    """Load a generic provider identity or replay the historical adapter."""

    if PROVIDER_EVALUATION_IDENTITY_METADATA_KEY not in details:
        return _legacy_identity(details, path=path)
    identity = _generic_identity(
        details[PROVIDER_EVALUATION_IDENTITY_METADATA_KEY],
        path=path,
    )
    _validate_optional_legacy_mirror(details, identity, path=path)
    return identity


def validate_provider_evaluation_metadata(
    metadata: FusedPredictionMetadata,
    *,
    path: Path,
) -> ProviderEvaluationIdentity:
    """Validate common estimator metadata and return normalized provider identity."""

    details = metadata.metadata
    require_revision(
        details.get("prob4d_revision"),
        name=f"{path} metadata.prob4d_revision",
    )
    identity = provider_identity_from_metadata(details, path=path)
    if details.get("includes_covariance") is not True:
        raise ValueError(
            f"{path} metadata.includes_covariance must be true for provider evaluation"
        )
    dense_storage_dtype = details.get("dense_storage_dtype", "float64")
    if dense_storage_dtype not in DENSE_STORAGE_DTYPES:
        raise ValueError(
            f"{path} metadata.dense_storage_dtype must be one of "
            + ", ".join(DENSE_STORAGE_DTYPES)
        )
    for field in ("gauge_estimator", "uncertainty_calibration"):
        value = details.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} metadata.{field} must be nonempty text")
    return identity


def provider_evaluation_method_signature(
    metadata: FusedPredictionMetadata,
    identity: ProviderEvaluationIdentity,
) -> tuple[object, ...]:
    """Build the cross-case signature without case-local manifest identities."""

    details = metadata.metadata
    return (
        metadata.fusion_method,
        metadata.covariance_semantics,
        metadata.correlation_assumption,
        details.get("prob4d_revision"),
        *identity.contract_signature(),
        details.get("dense_storage_dtype", "float64"),
        details.get("gauge_estimator"),
        details.get("uncertainty_calibration"),
        details.get("gauge_covariance_calibration_artifact_id"),
        details.get("point_uncertainty_calibration_artifact_id"),
        details.get("source_reliability_artifact_id"),
    )


def provider_identity_report_record(
    identity: ProviderEvaluationIdentity,
) -> dict[str, object]:
    """Return a detached finite JSON record for provider-evaluation output."""

    return cast(dict[str, object], plain_json(identity.evidence_record()))


__all__ = [
    "PROVIDER_EVALUATION_IDENTITY_METADATA_KEY",
    "PROVIDER_EVALUATION_IDENTITY_SCHEMA",
    "PROVIDER_EVALUATION_IDENTITY_VERSION",
    "ProviderEvaluationIdentity",
    "provider_evaluation_method_signature",
    "provider_identity_from_metadata",
    "provider_identity_report_record",
    "validate_provider_evaluation_metadata",
]
