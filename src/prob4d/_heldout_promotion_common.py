"""Shared constants and strict validation for held-out promotion artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._provider_evaluation_manifest import validate_finite_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_real,
    _strict_string,
)
from ._selection_evidence_common import (
    _strict_mapping as _base_strict_mapping,
)

HELDOUT_PROMOTION_LOCK_SCHEMA = "prob4d.heldout-provider-promotion-lock"
HELDOUT_PROMOTION_LOCK_VERSION = 1
HELDOUT_QUERY_RESULTS_SCHEMA = "prob4d.heldout-provider-query-results"
HELDOUT_QUERY_RESULTS_VERSION = 1
HELDOUT_PROMOTION_REPORT_SCHEMA = "prob4d.heldout-provider-promotion-report"
HELDOUT_PROMOTION_REPORT_VERSION = 1

PromotionArmRole = Literal[
    "physical_fallback",
    "visual_baseline",
    "rowwise_gauge_marginalized",
    "framewise_explicit_joint_gauge",
    "persistent_explicit_joint_gauge",
    "cross_window_identity_marginalized",
    "sensor_assisted",
    "diagnostic",
]

_REQUIRED_ARM_ROLES: frozenset[PromotionArmRole] = frozenset(
    {
        "physical_fallback",
        "visual_baseline",
        "rowwise_gauge_marginalized",
        "framewise_explicit_joint_gauge",
        "persistent_explicit_joint_gauge",
        "cross_window_identity_marginalized",
        "sensor_assisted",
    }
)
_ALLOWED_ARM_ROLES: frozenset[PromotionArmRole] = _REQUIRED_ARM_ROLES | {"diagnostic"}
_REQUIRED_FROZEN_ARTIFACTS = frozenset(
    {
        "provider_configuration",
        "gauge_calibration",
        "point_calibration",
        "source_reliability_calibration",
        "material_identity_calibration",
        "selection_lock",
        "bayesian_guard_configuration",
    }
)
_GIT_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")

LOCK_CLAIM_BOUNDARY = (
    "This lock authenticates a target-free held-out Prob4D-to-BayesianPhysTwin "
    "promotion protocol. It freezes complete object/session splits, comparison "
    "arms, source/model/calibration identities, bootstrap settings, and decision "
    "margins before target outcomes are opened. It is not empirical evidence."
)
QUERY_RESULTS_CLAIM_BOUNDARY = (
    "These rows retain complete target group-by-arm guarded-query outcomes under "
    "one immutable promotion lock. They do not by themselves establish provider "
    "competence or physical benefit."
)
REPORT_CLAIM_BOUNDARY = (
    "A passing report establishes only the declared held-out provider and guarded "
    "Bayesian physical-query gates for the exact frozen objects/sessions and source "
    "identities. It does not establish Causal4D intervention benefit or general "
    "state of the art."
)


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Validate mappings and fail closed on coercive schema header aliases."""

    mapping = _base_strict_mapping(value, name=name)
    if "schema_name" in mapping and "schema_version" in mapping:
        _strict_string(mapping["schema_name"], name=f"{name}.schema_name")
        _strict_integer(
            mapping["schema_version"],
            name=f"{name}.schema_version",
            minimum=1,
        )
    return mapping


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, *, name: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable or invalid JSON: {path}") from error
    mapping = _strict_mapping(value, name=name)
    validate_finite_json(mapping, name=name)
    return mapping, payload


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    payload = (
        json.dumps(
            plain_json(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    atomic_write_text(path, payload, overwrite=overwrite)


def _repository(value: Any, *, name: str) -> str:
    result = _strict_string(value, name=name)
    if result.count("/") != 1 or result.startswith("/") or result.endswith("/"):
        raise ValueError(f"{name} must have canonical owner/name form")
    return result


def _revision(value: Any, *, name: str) -> str:
    return _strict_digest(value, name=name, pattern=_GIT_REVISION)


def _optional_string(value: Any, *, name: str) -> str | None:
    return None if value is None else _strict_string(value, name=name)


def _optional_real(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    result = _strict_real(value, name=name)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _canonical_string_tuple(
    value: Any,
    *,
    name: str,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    result = tuple(
        _strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _string_tuple_from_json(value: Any, *, name: str) -> tuple[str, ...]:
    items = _strict_list(value, name=name)
    return _canonical_string_tuple(tuple(items), name=name)


def _digest_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    mapping = _strict_mapping(value, name=name)
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    normalized: dict[str, str] = {}
    for key, digest in mapping.items():
        canonical_key = _strict_string(key, name=f"{name} key")
        normalized[canonical_key] = _strict_digest(
            digest,
            name=f"{name}[{canonical_key!r}]",
            pattern=_SHA256,
        )
    if not _REQUIRED_FROZEN_ARTIFACTS.issubset(normalized):
        missing = sorted(_REQUIRED_FROZEN_ARTIFACTS - set(normalized))
        raise ValueError(f"{name} is missing required identities: {missing}")
    return frozen_finite_json_mapping(normalized, name=name)


@dataclass(frozen=True, slots=True)
class PromotionArmV1:
    """One frozen provider/query comparison arm and its scientific role."""

    arm_id: str
    role: PromotionArmRole
    query_method_id: str
    provider_method_id: str | None
    sensor_assisted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arm_id = _strict_string(self.arm_id, name="arm_id")
        role = _strict_string(self.role, name="role")
        if role not in _ALLOWED_ARM_ROLES:
            raise ValueError(f"role must be one of {sorted(_ALLOWED_ARM_ROLES)}")
        query_method = _strict_string(self.query_method_id, name="query_method_id")
        provider_method = _optional_string(
            self.provider_method_id,
            name="provider_method_id",
        )
        sensor_assisted = _strict_bool(
            self.sensor_assisted,
            name="sensor_assisted",
        )
        if role == "physical_fallback":
            if provider_method is not None:
                raise ValueError("physical fallback must not have a provider method")
            if sensor_assisted:
                raise ValueError("physical fallback cannot be sensor-assisted")
        elif provider_method is None:
            raise ValueError("every non-fallback arm requires a provider_method_id")
        if (role == "sensor_assisted") != sensor_assisted:
            raise ValueError("sensor_assisted must be true exactly for the sensor-assisted role")
        object.__setattr__(self, "arm_id", arm_id)
        object.__setattr__(self, "role", cast(PromotionArmRole, role))
        object.__setattr__(self, "query_method_id", query_method)
        object.__setattr__(self, "provider_method_id", provider_method)
        object.__setattr__(self, "sensor_assisted", sensor_assisted)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="arm metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "role": self.role,
            "query_method_id": self.query_method_id,
            "provider_method_id": self.provider_method_id,
            "sensor_assisted": self.sensor_assisted,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PromotionArmV1:
        mapping = _strict_mapping(value, name="promotion arm")
        _exact_keys(
            mapping,
            {
                "arm_id",
                "role",
                "query_method_id",
                "provider_method_id",
                "sensor_assisted",
                "metadata",
            },
            name="promotion arm",
        )
        return cls(
            arm_id=mapping["arm_id"],
            role=mapping["role"],
            query_method_id=mapping["query_method_id"],
            provider_method_id=mapping["provider_method_id"],
            sensor_assisted=mapping["sensor_assisted"],
            metadata=_strict_mapping(mapping["metadata"], name="arm metadata"),
        )
