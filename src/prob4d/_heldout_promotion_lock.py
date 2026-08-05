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


_LOCK_FIELDS = {
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
_LOCK_CONFIG_FIELDS = _LOCK_FIELDS - {
    "schema_name",
    "schema_version",
    "claim_boundary",
    "promotion_lock_id",
}


def _lock_from_mapping(mapping: Mapping[str, Any]) -> HeldoutProviderPromotionLockV1:
    raw_arms = _strict_list(mapping["arms"], name="arms")
    arms = tuple(
        sorted(
            (PromotionArmV1.from_dict(item) for item in raw_arms),
            key=lambda arm: arm.arm_id,
        )
    )
    return HeldoutProviderPromotionLockV1(
        experiment_id=mapping["experiment_id"],
        source_repository=mapping["source_repository"],
        source_revision=mapping["source_revision"],
        bayesian_phystwin_repository=mapping["bayesian_phystwin_repository"],
        bayesian_phystwin_revision=mapping["bayesian_phystwin_revision"],
        motioncrafter_revision=mapping["motioncrafter_revision"],
        model_set_id=mapping["model_set_id"],
        prediction_run_spec_id=mapping["prediction_run_spec_id"],
        provider_evaluation_manifest_sha256=mapping["provider_evaluation_manifest_sha256"],
        frozen_artifact_ids=mapping["frozen_artifact_ids"],
        development_group_ids=_string_tuple_from_json(
            mapping["development_group_ids"],
            name="development_group_ids",
        ),
        calibration_group_ids=_string_tuple_from_json(
            mapping["calibration_group_ids"],
            name="calibration_group_ids",
        ),
        target_group_ids=_string_tuple_from_json(
            mapping["target_group_ids"],
            name="target_group_ids",
        ),
        arms=arms,
        provider_reference_arm_id=mapping["provider_reference_arm_id"],
        primary_query_arm_id=mapping["primary_query_arm_id"],
        bootstrap_resamples=mapping["bootstrap_resamples"],
        bootstrap_seed=mapping["bootstrap_seed"],
        minimum_target_group_count=mapping["minimum_target_group_count"],
        query_superiority_margin_mm=mapping["query_superiority_margin_mm"],
        harmful_update_margin_mm=mapping["harmful_update_margin_mm"],
        maximum_harmful_accepted_updates=mapping["maximum_harmful_accepted_updates"],
        maximum_worst_group_regression_mm=mapping["maximum_worst_group_regression_mm"],
        maximum_technical_failures=mapping["maximum_technical_failures"],
        minimum_mean_accepted_coverage=mapping["minimum_mean_accepted_coverage"],
        metadata=_strict_mapping(mapping["metadata"], name="promotion lock metadata"),
    )


def promotion_lock_from_config(value: Any) -> HeldoutProviderPromotionLockV1:
    """Build a canonical lock from target-free configuration fields."""

    mapping = _strict_mapping(value, name="promotion lock configuration")
    _exact_keys(mapping, _LOCK_CONFIG_FIELDS, name="promotion lock configuration")
    return _lock_from_mapping(mapping)


def promotion_lock_from_dict(value: Any) -> HeldoutProviderPromotionLockV1:
    """Load and independently validate one canonical promotion lock."""

    mapping = _strict_mapping(value, name="promotion lock")
    _exact_keys(mapping, _LOCK_FIELDS, name="promotion lock")
    if mapping["schema_name"] != HELDOUT_PROMOTION_LOCK_SCHEMA:
        raise ValueError("unsupported held-out promotion lock schema")
    if mapping["schema_version"] != HELDOUT_PROMOTION_LOCK_VERSION:
        raise ValueError("unsupported held-out promotion lock version")
    if mapping["claim_boundary"] != LOCK_CLAIM_BOUNDARY:
        raise ValueError("promotion lock claim boundary changed")
    lock = _lock_from_mapping(mapping)
    supplied = _strict_digest(
        mapping["promotion_lock_id"],
        name="promotion_lock_id",
        pattern=_SHA256,
    )
    if supplied != lock.promotion_lock_id:
        raise ValueError("promotion_lock_id mismatch")
    return lock


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
