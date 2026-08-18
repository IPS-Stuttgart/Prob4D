"""Target-free lock contract for held-out provider promotion."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ._heldout_promotion_common import (
    _REQUIRED_ARM_ROLES,
    _SHA256,
    HELDOUT_PROMOTION_LOCK_SCHEMA,
    HELDOUT_PROMOTION_LOCK_VERSION,
    LOCK_CLAIM_BOUNDARY,
    PromotionArmV1,
    _atomic_write_json,
    _canonical_string_tuple,
    _digest_mapping,
    _exact_keys,
    _load_json,
    _nonnegative_real,
    _optional_real,
    _repository,
    _revision,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
    _string_tuple_from_json,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import _sha256_json, _strict_integer
from .prediction_provider_manifest import (
    COORDINATE_SEMANTICS,
    FLOW_SEMANTICS,
    POINT_SEMANTICS,
    RAY_SEMANTICS,
    SOURCE_DEPENDENCY_SEMANTICS,
    PredictionProviderManifestV1,
)

HELDOUT_PROMOTION_LOCK_V2_VERSION = 2
PROVIDER_PROMOTION_IDENTITY_SCHEMA = "prob4d.heldout-provider-promotion-identity"
PROVIDER_PROMOTION_IDENTITY_VERSION = 1

_PROVIDER_PROMOTION_IDENTITY_FIELDS = {
    "schema_name",
    "schema_version",
    "provider_family",
    "provider_repository",
    "provider_revision",
    "model_set_id",
    "loader_id",
    "coordinate_semantics",
    "point_semantics",
    "flow_semantics",
    "ray_semantics",
    "source_dependency_semantics",
}
_PROVIDER_PROMOTION_CONTRACT_FIELDS = (
    "provider_family",
    "provider_repository",
    "provider_revision",
    "model_set_id",
    "loader_id",
    "coordinate_semantics",
    "point_semantics",
    "flow_semantics",
    "ray_semantics",
    "source_dependency_semantics",
)


def _contract_choice(value: Any, choices: tuple[str, ...], *, name: str) -> str:
    result = _strict_string(value, name=name)
    if result not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")
    return result


@dataclass(frozen=True, slots=True)
class ProviderPromotionIdentityV1:
    """Provider contract frozen by a schema-v2 held-out promotion lock."""

    provider_family: str
    provider_repository: str
    provider_revision: str
    model_set_id: str
    loader_id: str
    coordinate_semantics: str
    point_semantics: str
    flow_semantics: str
    ray_semantics: str
    source_dependency_semantics: str = SOURCE_DEPENDENCY_SEMANTICS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_family",
            _strict_string(self.provider_family, name="provider_family"),
        )
        object.__setattr__(
            self,
            "provider_repository",
            _repository(self.provider_repository, name="provider_repository"),
        )
        object.__setattr__(
            self,
            "provider_revision",
            _revision(self.provider_revision, name="provider_revision"),
        )
        for field_name in ("model_set_id", "loader_id"):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        object.__setattr__(
            self,
            "coordinate_semantics",
            _contract_choice(
                self.coordinate_semantics,
                COORDINATE_SEMANTICS,
                name="coordinate_semantics",
            ),
        )
        object.__setattr__(
            self,
            "point_semantics",
            _contract_choice(
                self.point_semantics,
                POINT_SEMANTICS,
                name="point_semantics",
            ),
        )
        object.__setattr__(
            self,
            "flow_semantics",
            _contract_choice(
                self.flow_semantics,
                FLOW_SEMANTICS,
                name="flow_semantics",
            ),
        )
        object.__setattr__(
            self,
            "ray_semantics",
            _contract_choice(
                self.ray_semantics,
                RAY_SEMANTICS,
                name="ray_semantics",
            ),
        )
        source_semantics = _strict_string(
            self.source_dependency_semantics,
            name="source_dependency_semantics",
        )
        if source_semantics != SOURCE_DEPENDENCY_SEMANTICS:
            raise ValueError("unsupported source-dependency semantics")
        object.__setattr__(self, "source_dependency_semantics", source_semantics)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_PROMOTION_IDENTITY_SCHEMA,
            "schema_version": PROVIDER_PROMOTION_IDENTITY_VERSION,
            **{
                field_name: getattr(self, field_name)
                for field_name in _PROVIDER_PROMOTION_CONTRACT_FIELDS
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> ProviderPromotionIdentityV1:
        mapping = _strict_mapping(value, name="provider promotion identity")
        _exact_keys(
            mapping,
            _PROVIDER_PROMOTION_IDENTITY_FIELDS,
            name="provider promotion identity",
        )
        if mapping["schema_name"] != PROVIDER_PROMOTION_IDENTITY_SCHEMA:
            raise ValueError("unsupported provider promotion identity schema")
        if mapping["schema_version"] != PROVIDER_PROMOTION_IDENTITY_VERSION:
            raise ValueError("unsupported provider promotion identity version")
        return cls(
            provider_family=mapping["provider_family"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            coordinate_semantics=mapping["coordinate_semantics"],
            point_semantics=mapping["point_semantics"],
            flow_semantics=mapping["flow_semantics"],
            ray_semantics=mapping["ray_semantics"],
            source_dependency_semantics=mapping["source_dependency_semantics"],
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: PredictionProviderManifestV1,
    ) -> ProviderPromotionIdentityV1:
        if not isinstance(manifest, PredictionProviderManifestV1):
            raise ValueError("manifest must be PredictionProviderManifestV1")
        return cls(
            provider_family=manifest.provider_family,
            provider_repository=manifest.provider_repository,
            provider_revision=manifest.provider_revision,
            model_set_id=manifest.model_set_id,
            loader_id=manifest.loader_id,
            coordinate_semantics=manifest.coordinate_semantics,
            point_semantics=manifest.point_semantics,
            flow_semantics=manifest.flow_semantics,
            ray_semantics=manifest.ray_semantics,
            source_dependency_semantics=manifest.source_dependency_semantics,
        )

    def validate_contract(self, **observed: object) -> None:
        unexpected = set(observed) - set(_PROVIDER_PROMOTION_CONTRACT_FIELDS)
        missing = set(_PROVIDER_PROMOTION_CONTRACT_FIELDS) - set(observed)
        if unexpected or missing:
            raise ValueError(
                "provider contract fields differ from the registered promotion identity"
            )
        for field_name in _PROVIDER_PROMOTION_CONTRACT_FIELDS:
            if observed[field_name] != getattr(self, field_name):
                raise ValueError(f"target provider {field_name} differs from promotion lock")

    def validate_manifest(self, manifest: PredictionProviderManifestV1) -> None:
        observed = self.from_manifest(manifest)
        self.validate_contract(
            **{
                field_name: getattr(observed, field_name)
                for field_name in _PROVIDER_PROMOTION_CONTRACT_FIELDS
            }
        )


@dataclass(frozen=True, slots=True)
class HeldoutProviderPromotionLockV1:
    """Target-free lock for the decisive real provider and guarded-query gate."""

    experiment_id: str
    source_repository: str
    source_revision: str
    bayesian_phystwin_repository: str
    bayesian_phystwin_revision: str
    motioncrafter_revision: str
    model_set_id: str
    prediction_run_spec_id: str
    provider_evaluation_manifest_sha256: str
    frozen_artifact_ids: Mapping[str, Any]
    development_group_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    arms: tuple[PromotionArmV1, ...]
    provider_reference_arm_id: str
    primary_query_arm_id: str
    bootstrap_resamples: int
    bootstrap_seed: int
    minimum_target_group_count: int
    query_superiority_margin_mm: float
    harmful_update_margin_mm: float
    maximum_harmful_accepted_updates: int
    maximum_worst_group_regression_mm: float
    maximum_technical_failures: int
    minimum_mean_accepted_coverage: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            _strict_string(self.experiment_id, name="experiment_id"),
        )
        object.__setattr__(
            self,
            "source_repository",
            _repository(self.source_repository, name="source_repository"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _revision(self.source_revision, name="source_revision"),
        )
        object.__setattr__(
            self,
            "bayesian_phystwin_repository",
            _repository(
                self.bayesian_phystwin_repository,
                name="bayesian_phystwin_repository",
            ),
        )
        object.__setattr__(
            self,
            "bayesian_phystwin_revision",
            _revision(
                self.bayesian_phystwin_revision,
                name="bayesian_phystwin_revision",
            ),
        )
        object.__setattr__(
            self,
            "motioncrafter_revision",
            _revision(self.motioncrafter_revision, name="motioncrafter_revision"),
        )
        for field_name in (
            "model_set_id",
            "prediction_run_spec_id",
            "provider_evaluation_manifest_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        object.__setattr__(
            self,
            "frozen_artifact_ids",
            _digest_mapping(self.frozen_artifact_ids, name="frozen_artifact_ids"),
        )
        development = _canonical_string_tuple(
            self.development_group_ids,
            name="development_group_ids",
        )
        calibration = _canonical_string_tuple(
            self.calibration_group_ids,
            name="calibration_group_ids",
        )
        target = _canonical_string_tuple(
            self.target_group_ids,
            name="target_group_ids",
        )
        if set(development) & set(calibration):
            raise ValueError("development and calibration groups must be disjoint")
        if set(development) & set(target):
            raise ValueError("development and target groups must be disjoint")
        if set(calibration) & set(target):
            raise ValueError("calibration and target groups must be disjoint")
        object.__setattr__(self, "development_group_ids", development)
        object.__setattr__(self, "calibration_group_ids", calibration)
        object.__setattr__(self, "target_group_ids", target)

        if (
            type(self.arms) is not tuple
            or not self.arms
            or not all(isinstance(arm, PromotionArmV1) for arm in self.arms)
        ):
            raise ValueError("arms must be a nonempty tuple of PromotionArmV1")
        arms = tuple(self.arms)
        arm_ids = tuple(arm.arm_id for arm in arms)
        if arm_ids != tuple(sorted(arm_ids)) or len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arms must be sorted by unique arm_id")
        roles = [arm.role for arm in arms]
        missing_roles = sorted(_REQUIRED_ARM_ROLES - set(roles))
        if missing_roles:
            raise ValueError(f"promotion arms are missing required roles: {missing_roles}")
        for role in _REQUIRED_ARM_ROLES:
            if roles.count(role) != 1:
                raise ValueError(f"required promotion role {role!r} must occur exactly once")
        provider_methods = [
            arm.provider_method_id for arm in arms if arm.provider_method_id is not None
        ]
        query_methods = [arm.query_method_id for arm in arms]
        if len(set(provider_methods)) != len(provider_methods):
            raise ValueError("provider_method_id values must be unique")
        if len(set(query_methods)) != len(query_methods):
            raise ValueError("query_method_id values must be unique")
        object.__setattr__(self, "arms", arms)

        provider_reference = _strict_string(
            self.provider_reference_arm_id,
            name="provider_reference_arm_id",
        )
        primary_query = _strict_string(
            self.primary_query_arm_id,
            name="primary_query_arm_id",
        )
        arms_by_id = {arm.arm_id: arm for arm in arms}
        if provider_reference not in arms_by_id:
            raise ValueError("provider_reference_arm_id is not a registered arm")
        if arms_by_id[provider_reference].provider_method_id is None:
            raise ValueError("provider reference arm requires a provider method")
        if primary_query not in arms_by_id:
            raise ValueError("primary_query_arm_id is not a registered arm")
        primary_arm = arms_by_id[primary_query]
        if primary_arm.role in {"physical_fallback", "sensor_assisted", "diagnostic"}:
            raise ValueError("primary query arm must be a non-sensor, non-diagnostic candidate")
        object.__setattr__(self, "provider_reference_arm_id", provider_reference)
        object.__setattr__(self, "primary_query_arm_id", primary_query)

        bootstrap_resamples = _strict_integer(
            self.bootstrap_resamples,
            name="bootstrap_resamples",
            minimum=100,
        )
        bootstrap_seed = _strict_integer(
            self.bootstrap_seed,
            name="bootstrap_seed",
            minimum=0,
        )
        minimum_groups = _strict_integer(
            self.minimum_target_group_count,
            name="minimum_target_group_count",
            minimum=1,
        )
        if len(target) < minimum_groups:
            raise ValueError("target group count is below the frozen minimum")
        object.__setattr__(self, "bootstrap_resamples", bootstrap_resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        object.__setattr__(self, "minimum_target_group_count", minimum_groups)
        object.__setattr__(
            self,
            "query_superiority_margin_mm",
            _nonnegative_real(
                self.query_superiority_margin_mm,
                name="query_superiority_margin_mm",
            ),
        )
        object.__setattr__(
            self,
            "harmful_update_margin_mm",
            _nonnegative_real(
                self.harmful_update_margin_mm,
                name="harmful_update_margin_mm",
            ),
        )
        object.__setattr__(
            self,
            "maximum_harmful_accepted_updates",
            _strict_integer(
                self.maximum_harmful_accepted_updates,
                name="maximum_harmful_accepted_updates",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_worst_group_regression_mm",
            _nonnegative_real(
                self.maximum_worst_group_regression_mm,
                name="maximum_worst_group_regression_mm",
            ),
        )
        object.__setattr__(
            self,
            "maximum_technical_failures",
            _strict_integer(
                self.maximum_technical_failures,
                name="maximum_technical_failures",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_mean_accepted_coverage",
            _optional_real(
                self.minimum_mean_accepted_coverage,
                name="minimum_mean_accepted_coverage",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="promotion lock metadata"),
        )

    @property
    def arms_by_id(self) -> Mapping[str, PromotionArmV1]:
        return {arm.arm_id: arm for arm in self.arms}

    @property
    def physical_fallback_arm_id(self) -> str:
        return next(arm.arm_id for arm in self.arms if arm.role == "physical_fallback")

    @property
    def provider_method_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                cast(str, arm.provider_method_id)
                for arm in self.arms
                if arm.provider_method_id is not None
            )
        )

    @property
    def provider_reference_method_id(self) -> str:
        method = self.arms_by_id[self.provider_reference_arm_id].provider_method_id
        assert method is not None
        return method

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": HELDOUT_PROMOTION_LOCK_SCHEMA,
            "schema_version": HELDOUT_PROMOTION_LOCK_VERSION,
            "experiment_id": self.experiment_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "bayesian_phystwin_repository": self.bayesian_phystwin_repository,
            "bayesian_phystwin_revision": self.bayesian_phystwin_revision,
            "motioncrafter_revision": self.motioncrafter_revision,
            "model_set_id": self.model_set_id,
            "prediction_run_spec_id": self.prediction_run_spec_id,
            "provider_evaluation_manifest_sha256": (self.provider_evaluation_manifest_sha256),
            "frozen_artifact_ids": plain_json(self.frozen_artifact_ids),
            "development_group_ids": list(self.development_group_ids),
            "calibration_group_ids": list(self.calibration_group_ids),
            "target_group_ids": list(self.target_group_ids),
            "arms": [arm.to_dict() for arm in self.arms],
            "provider_reference_arm_id": self.provider_reference_arm_id,
            "primary_query_arm_id": self.primary_query_arm_id,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "minimum_target_group_count": self.minimum_target_group_count,
            "query_superiority_margin_mm": self.query_superiority_margin_mm,
            "harmful_update_margin_mm": self.harmful_update_margin_mm,
            "maximum_harmful_accepted_updates": (self.maximum_harmful_accepted_updates),
            "maximum_worst_group_regression_mm": (self.maximum_worst_group_regression_mm),
            "maximum_technical_failures": self.maximum_technical_failures,
            "minimum_mean_accepted_coverage": self.minimum_mean_accepted_coverage,
            "metadata": plain_json(self.metadata),
            "claim_boundary": LOCK_CLAIM_BOUNDARY,
        }

    @property
    def promotion_lock_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "promotion_lock_id": self.promotion_lock_id}


@dataclass(frozen=True, slots=True)
class HeldoutProviderPromotionLockV2(HeldoutProviderPromotionLockV1):
    """Provider-neutral held-out lock with a complete immutable provider contract."""

    provider_identity: ProviderPromotionIdentityV1 | None = None

    def __post_init__(self) -> None:
        HeldoutProviderPromotionLockV1.__post_init__(self)
        identity = self.provider_identity
        if not isinstance(identity, ProviderPromotionIdentityV1):
            raise ValueError("provider_identity must be ProviderPromotionIdentityV1")
        if self.motioncrafter_revision != identity.provider_revision:
            raise ValueError("internal provider revision disagrees with provider_identity")
        if self.model_set_id != identity.model_set_id:
            raise ValueError("internal model set disagrees with provider_identity")

    @property
    def provider_contract(self) -> ProviderPromotionIdentityV1:
        identity = self.provider_identity
        if not isinstance(identity, ProviderPromotionIdentityV1):
            raise AssertionError("validated v2 lock has no provider identity")
        return identity

    def descriptor(self) -> dict[str, object]:
        descriptor = HeldoutProviderPromotionLockV1.descriptor(self)
        descriptor["schema_version"] = HELDOUT_PROMOTION_LOCK_V2_VERSION
        descriptor.pop("motioncrafter_revision")
        descriptor.pop("model_set_id")
        descriptor["provider_identity"] = self.provider_contract.to_dict()
        return descriptor


_LOCK_FIELDS_V1 = {
    "schema_name",
    "schema_version",
    "experiment_id",
    "source_repository",
    "source_revision",
    "bayesian_phystwin_repository",
    "bayesian_phystwin_revision",
    "motioncrafter_revision",
    "model_set_id",
    "prediction_run_spec_id",
    "provider_evaluation_manifest_sha256",
    "frozen_artifact_ids",
    "development_group_ids",
    "calibration_group_ids",
    "target_group_ids",
    "arms",
    "provider_reference_arm_id",
    "primary_query_arm_id",
    "bootstrap_resamples",
    "bootstrap_seed",
    "minimum_target_group_count",
    "query_superiority_margin_mm",
    "harmful_update_margin_mm",
    "maximum_harmful_accepted_updates",
    "maximum_worst_group_regression_mm",
    "maximum_technical_failures",
    "minimum_mean_accepted_coverage",
    "metadata",
    "claim_boundary",
    "promotion_lock_id",
}
_LOCK_FIELDS_V2 = (_LOCK_FIELDS_V1 - {"motioncrafter_revision", "model_set_id"}) | {
    "provider_identity"
}
_LOCK_CONFIG_FIELDS_V1 = _LOCK_FIELDS_V1 - {
    "schema_name",
    "schema_version",
    "claim_boundary",
    "promotion_lock_id",
}
_LOCK_CONFIG_FIELDS_V2 = _LOCK_FIELDS_V2 - {
    "schema_name",
    "schema_version",
    "claim_boundary",
    "promotion_lock_id",
}


def _arms_from_mapping(mapping: Mapping[str, Any]) -> tuple[PromotionArmV1, ...]:
    raw_arms = _strict_list(mapping["arms"], name="arms")
    return tuple(
        sorted(
            (PromotionArmV1.from_dict(item) for item in raw_arms),
            key=lambda arm: arm.arm_id,
        )
    )


def _common_lock_values(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": mapping["experiment_id"],
        "source_repository": mapping["source_repository"],
        "source_revision": mapping["source_revision"],
        "bayesian_phystwin_repository": mapping["bayesian_phystwin_repository"],
        "bayesian_phystwin_revision": mapping["bayesian_phystwin_revision"],
        "prediction_run_spec_id": mapping["prediction_run_spec_id"],
        "provider_evaluation_manifest_sha256": mapping["provider_evaluation_manifest_sha256"],
        "frozen_artifact_ids": mapping["frozen_artifact_ids"],
        "development_group_ids": _string_tuple_from_json(
            mapping["development_group_ids"],
            name="development_group_ids",
        ),
        "calibration_group_ids": _string_tuple_from_json(
            mapping["calibration_group_ids"],
            name="calibration_group_ids",
        ),
        "target_group_ids": _string_tuple_from_json(
            mapping["target_group_ids"],
            name="target_group_ids",
        ),
        "arms": _arms_from_mapping(mapping),
        "provider_reference_arm_id": mapping["provider_reference_arm_id"],
        "primary_query_arm_id": mapping["primary_query_arm_id"],
        "bootstrap_resamples": mapping["bootstrap_resamples"],
        "bootstrap_seed": mapping["bootstrap_seed"],
        "minimum_target_group_count": mapping["minimum_target_group_count"],
        "query_superiority_margin_mm": mapping["query_superiority_margin_mm"],
        "harmful_update_margin_mm": mapping["harmful_update_margin_mm"],
        "maximum_harmful_accepted_updates": mapping["maximum_harmful_accepted_updates"],
        "maximum_worst_group_regression_mm": mapping["maximum_worst_group_regression_mm"],
        "maximum_technical_failures": mapping["maximum_technical_failures"],
        "minimum_mean_accepted_coverage": mapping["minimum_mean_accepted_coverage"],
        "metadata": _strict_mapping(mapping["metadata"], name="promotion lock metadata"),
    }


def _lock_v1_from_mapping(mapping: Mapping[str, Any]) -> HeldoutProviderPromotionLockV1:
    return HeldoutProviderPromotionLockV1(
        motioncrafter_revision=mapping["motioncrafter_revision"],
        model_set_id=mapping["model_set_id"],
        **_common_lock_values(mapping),
    )


def _lock_v2_from_mapping(mapping: Mapping[str, Any]) -> HeldoutProviderPromotionLockV2:
    identity = ProviderPromotionIdentityV1.from_dict(mapping["provider_identity"])
    return HeldoutProviderPromotionLockV2(
        motioncrafter_revision=identity.provider_revision,
        model_set_id=identity.model_set_id,
        provider_identity=identity,
        **_common_lock_values(mapping),
    )


def promotion_lock_from_config(
    value: Any,
) -> HeldoutProviderPromotionLockV1:
    """Build a canonical v2 lock or replay a historical MotionCrafter v1 config."""

    mapping = _strict_mapping(value, name="promotion lock configuration")
    has_provider_identity = "provider_identity" in mapping
    legacy_fields = {"motioncrafter_revision", "model_set_id"} & set(mapping)
    if has_provider_identity:
        if legacy_fields:
            raise ValueError(
                "provider_identity cannot be mixed with legacy MotionCrafter lock fields"
            )
        _exact_keys(
            mapping,
            _LOCK_CONFIG_FIELDS_V2,
            name="promotion lock configuration",
        )
        return _lock_v2_from_mapping(mapping)
    _exact_keys(
        mapping,
        _LOCK_CONFIG_FIELDS_V1,
        name="promotion lock configuration",
    )
    return _lock_v1_from_mapping(mapping)


def promotion_lock_from_dict(value: Any) -> HeldoutProviderPromotionLockV1:
    """Load and independently validate a v1 or provider-neutral v2 lock."""

    mapping = _strict_mapping(value, name="promotion lock")
    if mapping.get("schema_name") != HELDOUT_PROMOTION_LOCK_SCHEMA:
        raise ValueError("unsupported held-out promotion lock schema")
    version = _strict_integer(
        mapping.get("schema_version"),
        name="schema_version",
        minimum=1,
    )
    if version == HELDOUT_PROMOTION_LOCK_VERSION:
        fields = _LOCK_FIELDS_V1
        factory = _lock_v1_from_mapping
    elif version == HELDOUT_PROMOTION_LOCK_V2_VERSION:
        fields = _LOCK_FIELDS_V2
        factory = _lock_v2_from_mapping
    else:
        raise ValueError("unsupported held-out promotion lock version")
    _exact_keys(mapping, fields, name="promotion lock")
    if mapping["claim_boundary"] != LOCK_CLAIM_BOUNDARY:
        raise ValueError("promotion lock claim boundary changed")
    lock = factory(mapping)
    supplied = _strict_digest(
        mapping["promotion_lock_id"],
        name="promotion_lock_id",
        pattern=_SHA256,
    )
    if supplied != lock.promotion_lock_id:
        raise ValueError("promotion_lock_id mismatch")
    return lock


def validate_provider_manifest_against_lock(
    lock: HeldoutProviderPromotionLockV1,
    manifest: PredictionProviderManifestV1,
) -> None:
    """Require one target manifest to match the provider contract frozen by the lock."""

    if isinstance(lock, HeldoutProviderPromotionLockV2):
        lock.provider_contract.validate_manifest(manifest)
        return
    if manifest.provider_revision != lock.motioncrafter_revision:
        raise ValueError("target provider revision differs from promotion lock")
    if manifest.model_set_id != lock.model_set_id:
        raise ValueError("target provider model set differs from promotion lock")


def validate_provider_contract_against_lock(
    lock: HeldoutProviderPromotionLockV1,
    *,
    provider_family: str,
    provider_repository: str,
    provider_revision: str,
    model_set_id: str,
    loader_id: str,
    coordinate_semantics: str,
    point_semantics: str,
    flow_semantics: str,
    ray_semantics: str,
    source_dependency_semantics: str,
) -> None:
    """Require an admitted aggregate contract to match the frozen provider identity."""

    if isinstance(lock, HeldoutProviderPromotionLockV2):
        lock.provider_contract.validate_contract(
            provider_family=provider_family,
            provider_repository=provider_repository,
            provider_revision=provider_revision,
            model_set_id=model_set_id,
            loader_id=loader_id,
            coordinate_semantics=coordinate_semantics,
            point_semantics=point_semantics,
            flow_semantics=flow_semantics,
            ray_semantics=ray_semantics,
            source_dependency_semantics=source_dependency_semantics,
        )
        return
    if provider_revision != lock.motioncrafter_revision:
        raise ValueError("target provider admission provider revision changed")
    if model_set_id != lock.model_set_id:
        raise ValueError("target provider admission model set changed")


def write_promotion_lock(
    lock: HeldoutProviderPromotionLockV1,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(lock, HeldoutProviderPromotionLockV1):
        raise ValueError("lock must be HeldoutProviderPromotionLockV1")
    _atomic_write_json(Path(path), lock.to_dict(), overwrite=overwrite)


def load_promotion_lock(
    path: str | os.PathLike[str],
) -> HeldoutProviderPromotionLockV1:
    mapping, _ = _load_json(Path(path), name="promotion lock")
    return promotion_lock_from_dict(mapping)
