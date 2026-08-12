"""Localize source-side covariance failure before point-model development.

This module connects already-existing source mean/identity evidence and joint
covariance diagnostics to the covariance stages of ``fresh_provider_readiness``.
It deliberately does not fit a new covariance model.  Point-uncertainty
improvement is authorized only when source mean and identity gates pass, shared
(gauge/dependence) residual energy is adequately calibrated, and the remaining
conditional subspace is miscalibrated under a policy frozen on source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
)
from .fresh_provider_readiness import ReadinessGateV1, unevaluated_gate
from .joint_covariance_metrics import (
    JOINT_COVARIANCE_CLAIM_BOUNDARY,
    JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA,
    JOINT_COVARIANCE_DIAGNOSTIC_VERSION,
)
from .source_provider_competence import (
    SourceProviderCompetenceReportV1,
    load_source_provider_competence,
)

SOURCE_COVARIANCE_LOCALIZATION_SCHEMA = "prob4d.source-covariance-localization"
SOURCE_COVARIANCE_LOCALIZATION_VERSION = 1
SOURCE_COVARIANCE_LOCALIZATION_CLAIM_BOUNDARY = (
    "This artifact localizes already-open source/calibration covariance evidence. "
    "It does not fit or select a richer covariance model, use protected target "
    "outcomes, authorize a BayesianPhysTwin update, establish provider transfer, "
    "Causal4D intervention benefit, deployment safety, or state of the art. Point "
    "uncertainty development is authorized only for the explicit "
    "point-covariance-localized classification."
)

LocalizationClassification = Literal[
    "source-mean-negative",
    "identity-or-association-negative",
    "gauge-or-dependence-negative",
    "point-covariance-localized",
    "covariance-adequate",
    "technical-failure",
]

_POLICY_FIELDS = frozenset(
    {
        "minimum_group_count",
        "normalized_nees_lower",
        "normalized_nees_upper",
        "minimum_joint_pass_fraction",
        "shared_energy_lower",
        "shared_energy_upper",
        "minimum_shared_pass_fraction",
        "conditional_energy_lower",
        "conditional_energy_upper",
        "minimum_conditional_pass_fraction",
        "require_shared_subspace",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "group_id",
        "normalized_nees",
        "shared_subspace_normalized_energy",
        "conditional_subspace_normalized_energy",
        "joint_in_band",
        "shared_in_band",
        "conditional_in_band",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "provider_manifest_id",
        "cohort_binding_id",
        "source_provider_competence_id",
        "joint_diagnostic_sha256",
        "joint_residual_source_sha256",
        "policy",
        "groups",
        "group_count",
        "joint_pass_fraction",
        "shared_pass_fraction",
        "conditional_pass_fraction",
        "classification",
        "reason_codes",
        "authorize_point_uncertainty_development",
        "metadata",
        "claim_boundary",
        "source_covariance_localization_id",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _probability(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_nonnegative(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative(value, name=name)


def _sorted_unique_strings(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _strings_from_json(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return _sorted_unique_strings(tuple(value), name=name)


def _classification(value: object) -> LocalizationClassification:
    result = require_exact_string(value, name="classification")
    allowed = {
        "source-mean-negative",
        "identity-or-association-negative",
        "gauge-or-dependence-negative",
        "point-covariance-localized",
        "covariance-adequate",
        "technical-failure",
    }
    if result not in allowed:
        raise ValueError("unsupported source covariance localization classification")
    return cast(LocalizationClassification, result)


def _json_bytes_object(payload: bytes, *, name: str) -> dict[str, Any]:
    class DuplicateKey(ValueError):
        pass

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise DuplicateKey(f"{name} contains duplicate JSON object key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise DuplicateKey(f"{name} contains non-finite JSON number {token!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


@dataclass(frozen=True, slots=True)
class SourceCovarianceLocalizationPolicyV1:
    """Frozen source-only thresholds for assigning the covariance failure stage."""

    minimum_group_count: int
    normalized_nees_lower: float
    normalized_nees_upper: float
    minimum_joint_pass_fraction: float
    shared_energy_lower: float
    shared_energy_upper: float
    minimum_shared_pass_fraction: float
    conditional_energy_lower: float
    conditional_energy_upper: float
    minimum_conditional_pass_fraction: float
    require_shared_subspace: bool = True

    def __post_init__(self) -> None:
        count = require_exact_integer(
            self.minimum_group_count,
            name="minimum_group_count",
            minimum=1,
        )
        bands: dict[str, tuple[float, float]] = {}
        for prefix in ("normalized_nees", "shared_energy", "conditional_energy"):
            lower = _nonnegative(getattr(self, f"{prefix}_lower"), name=f"{prefix}_lower")
            upper = _nonnegative(getattr(self, f"{prefix}_upper"), name=f"{prefix}_upper")
            if upper < lower:
                raise ValueError(f"{prefix}_upper must not be below {prefix}_lower")
            bands[prefix] = (lower, upper)
        for name in (
            "minimum_joint_pass_fraction",
            "minimum_shared_pass_fraction",
            "minimum_conditional_pass_fraction",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
        object.__setattr__(self, "minimum_group_count", count)
        for prefix, (lower, upper) in bands.items():
            object.__setattr__(self, f"{prefix}_lower", lower)
            object.__setattr__(self, f"{prefix}_upper", upper)
        object.__setattr__(
            self,
            "require_shared_subspace",
            _strict_bool(self.require_shared_subspace, name="require_shared_subspace"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_group_count": self.minimum_group_count,
            "normalized_nees_lower": self.normalized_nees_lower,
            "normalized_nees_upper": self.normalized_nees_upper,
            "minimum_joint_pass_fraction": self.minimum_joint_pass_fraction,
            "shared_energy_lower": self.shared_energy_lower,
            "shared_energy_upper": self.shared_energy_upper,
            "minimum_shared_pass_fraction": self.minimum_shared_pass_fraction,
            "conditional_energy_lower": self.conditional_energy_lower,
            "conditional_energy_upper": self.conditional_energy_upper,
            "minimum_conditional_pass_fraction": (
                self.minimum_conditional_pass_fraction
            ),
            "require_shared_subspace": self.require_shared_subspace,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceCovarianceLocalizationPolicyV1:
        mapping = require_mapping(value, name="source covariance localization policy")
        require_exact_fields(
            mapping,
            _POLICY_FIELDS,
            name="source covariance localization policy",
        )
        return cls(**mapping)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SourceCovarianceLocalizationGroupV1:
    """Per-independent-group subspace calibration classification."""

    group_id: str
    normalized_nees: float
    shared_subspace_normalized_energy: float | None
    conditional_subspace_normalized_energy: float | None
    joint_in_band: bool
    shared_in_band: bool
    conditional_in_band: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", require_exact_string(self.group_id, name="group_id"))
        object.__setattr__(
            self,
            "normalized_nees",
            _nonnegative(self.normalized_nees, name="normalized_nees"),
        )
        object.__setattr__(
            self,
            "shared_subspace_normalized_energy",
            _optional_nonnegative(
                self.shared_subspace_normalized_energy,
                name="shared_subspace_normalized_energy",
            ),
        )
        object.__setattr__(
            self,
            "conditional_subspace_normalized_energy",
            _optional_nonnegative(
                self.conditional_subspace_normalized_energy,
                name="conditional_subspace_normalized_energy",
            ),
        )
        for name in ("joint_in_band", "shared_in_band", "conditional_in_band"):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name=name))

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "normalized_nees": self.normalized_nees,
            "shared_subspace_normalized_energy": self.shared_subspace_normalized_energy,
            "conditional_subspace_normalized_energy": (
                self.conditional_subspace_normalized_energy
            ),
            "joint_in_band": self.joint_in_band,
            "shared_in_band": self.shared_in_band,
            "conditional_in_band": self.conditional_in_band,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceCovarianceLocalizationGroupV1:
        mapping = require_mapping(value, name="source covariance localization group")
        require_exact_fields(
            mapping,
            _GROUP_FIELDS,
            name="source covariance localization group",
        )
        return cls(**mapping)  # type: ignore[arg-type]


def _in_band(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value <= upper


def _normalize_joint_groups(
    report: Mapping[str, Any],
    *,
    policy: SourceCovarianceLocalizationPolicyV1,
) -> tuple[SourceCovarianceLocalizationGroupV1, ...]:
    if report.get("schema_name") != JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA:
        raise ValueError("joint covariance diagnostic schema changed")
    if report.get("schema_version") != JOINT_COVARIANCE_DIAGNOSTIC_VERSION:
        raise ValueError("joint covariance diagnostic version changed")
    if report.get("claim_boundary") != JOINT_COVARIANCE_CLAIM_BOUNDARY:
        raise ValueError("joint covariance diagnostic claim boundary changed")
    evaluation = require_mapping(report.get("evaluation"), name="joint covariance evaluation")
    raw_groups = evaluation.get("groups")
    if type(raw_groups) is not list or not raw_groups:
        raise ValueError("joint covariance diagnostic groups must be a nonempty JSON array")
    result: list[SourceCovarianceLocalizationGroupV1] = []
    for index, raw in enumerate(raw_groups):
        group = require_mapping(raw, name=f"joint covariance groups[{index}]")
        group_id = require_exact_string(
            group.get("factor_group_id"),
            name=f"joint covariance groups[{index}].factor_group_id",
        )
        nees = _nonnegative(
            group.get("normalized_nees"),
            name=f"joint covariance groups[{index}].normalized_nees",
        )
        shared = _optional_nonnegative(
            group.get("shared_subspace_normalized_energy"),
            name=(
                f"joint covariance groups[{index}].shared_subspace_normalized_energy"
            ),
        )
        conditional = _optional_nonnegative(
            group.get("conditional_subspace_normalized_energy"),
            name=(
                f"joint covariance groups[{index}].conditional_subspace_normalized_energy"
            ),
        )
        result.append(
            SourceCovarianceLocalizationGroupV1(
                group_id=group_id,
                normalized_nees=nees,
                shared_subspace_normalized_energy=shared,
                conditional_subspace_normalized_energy=conditional,
                joint_in_band=_in_band(
                    nees,
                    policy.normalized_nees_lower,
                    policy.normalized_nees_upper,
                ),
                shared_in_band=(
                    not policy.require_shared_subspace
                    if shared is None
                    else _in_band(
                        shared,
                        policy.shared_energy_lower,
                        policy.shared_energy_upper,
                    )
                ),
                conditional_in_band=_in_band(
                    conditional,
                    policy.conditional_energy_lower,
                    policy.conditional_energy_upper,
                ),
            )
        )
    ordered = tuple(sorted(result, key=lambda item: item.group_id))
    if len({item.group_id for item in ordered}) != len(ordered):
        raise ValueError("joint covariance factor_group_id values must be unique")
    return ordered


def _fraction(groups: tuple[SourceCovarianceLocalizationGroupV1, ...], field: str) -> float:
    return sum(bool(getattr(group, field)) for group in groups) / len(groups)


@dataclass(frozen=True, slots=True)
class SourceCovarianceLocalizationV1:
    """Content-addressed source-only covariance failure localization."""

    provider_manifest_id: str
    cohort_binding_id: str
    source_provider_competence_id: str
    joint_diagnostic_sha256: str
    joint_residual_source_sha256: str
    policy: SourceCovarianceLocalizationPolicyV1
    groups: tuple[SourceCovarianceLocalizationGroupV1, ...]
    source_mean_status: str = field(repr=False)
    identity_reliability_status: str = field(repr=False)
    source_mean_reasons: tuple[str, ...] = field(default=(), repr=False)
    identity_reliability_reasons: tuple[str, ...] = field(default=(), repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    group_count: int = field(init=False)
    joint_pass_fraction: float = field(init=False)
    shared_pass_fraction: float = field(init=False)
    conditional_pass_fraction: float = field(init=False)
    classification: LocalizationClassification = field(init=False)
    reason_codes: tuple[str, ...] = field(init=False)
    authorize_point_uncertainty_development: bool = field(init=False)
    source_covariance_localization_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "provider_manifest_id",
            "cohort_binding_id",
            "source_provider_competence_id",
            "joint_diagnostic_sha256",
            "joint_residual_source_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        if not isinstance(self.policy, SourceCovarianceLocalizationPolicyV1):
            raise TypeError("policy must be SourceCovarianceLocalizationPolicyV1")
        if type(self.groups) is not tuple or not all(
            isinstance(item, SourceCovarianceLocalizationGroupV1) for item in self.groups
        ):
            raise TypeError("groups must contain SourceCovarianceLocalizationGroupV1 values")
        groups = tuple(sorted(self.groups, key=lambda item: item.group_id))
        if len({item.group_id for item in groups}) != len(groups):
            raise ValueError("groups must have unique group_id values")
        for group in groups:
            expected_joint = _in_band(
                group.normalized_nees,
                self.policy.normalized_nees_lower,
                self.policy.normalized_nees_upper,
            )
            expected_shared = (
                not self.policy.require_shared_subspace
                if group.shared_subspace_normalized_energy is None
                else _in_band(
                    group.shared_subspace_normalized_energy,
                    self.policy.shared_energy_lower,
                    self.policy.shared_energy_upper,
                )
            )
            expected_conditional = _in_band(
                group.conditional_subspace_normalized_energy,
                self.policy.conditional_energy_lower,
                self.policy.conditional_energy_upper,
            )
            if (
                group.joint_in_band != expected_joint
                or group.shared_in_band != expected_shared
                or group.conditional_in_band != expected_conditional
            ):
                raise ValueError("group band classifications do not match the frozen policy")
        mean_status = require_exact_string(self.source_mean_status, name="source_mean_status")
        identity_status = require_exact_string(
            self.identity_reliability_status,
            name="identity_reliability_status",
        )
        if mean_status not in {"pass", "fail", "technical-failure"}:
            raise ValueError("unsupported source_mean_status")
        if identity_status not in {"pass", "fail", "technical-failure"}:
            raise ValueError("unsupported identity_reliability_status")
        mean_reasons = _sorted_unique_strings(self.source_mean_reasons, name="source_mean_reasons")
        identity_reasons = _sorted_unique_strings(
            self.identity_reliability_reasons,
            name="identity_reliability_reasons",
        )
        reserved_metadata = {
            "source_mean_status",
            "identity_reliability_status",
            "source_mean_reasons",
            "identity_reliability_reasons",
        }
        if reserved_metadata & self.metadata.keys():
            raise ValueError("metadata uses reserved source-status keys")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )
        count = len(groups)
        joint_fraction = 0.0 if not groups else _fraction(groups, "joint_in_band")
        shared_fraction = 0.0 if not groups else _fraction(groups, "shared_in_band")
        conditional_fraction = 0.0 if not groups else _fraction(groups, "conditional_in_band")

        reasons: list[str] = []
        if mean_status == "technical-failure" or identity_status == "technical-failure":
            classification: LocalizationClassification = "technical-failure"
            reasons.extend(
                mean_reasons
                or identity_reasons
                or ("source-competence-technical-failure",)
            )
        elif mean_status != "pass":
            classification = "source-mean-negative"
            reasons.extend(mean_reasons or ("source-mean-negative",))
        elif identity_status != "pass":
            classification = "identity-or-association-negative"
            reasons.extend(identity_reasons or ("identity-or-association-negative",))
        elif count < self.policy.minimum_group_count:
            classification = "technical-failure"
            reasons.append("insufficient-joint-covariance-groups")
        elif self.policy.require_shared_subspace and any(
            group.shared_subspace_normalized_energy is None for group in groups
        ):
            classification = "technical-failure"
            reasons.append("missing-required-shared-subspace-energy")
        elif any(group.conditional_subspace_normalized_energy is None for group in groups):
            classification = "technical-failure"
            reasons.append("missing-conditional-subspace-energy")
        elif shared_fraction < self.policy.minimum_shared_pass_fraction:
            classification = "gauge-or-dependence-negative"
            reasons.append("shared-subspace-energy-outside-frozen-band")
        elif conditional_fraction < self.policy.minimum_conditional_pass_fraction:
            classification = "point-covariance-localized"
            reasons.append("conditional-subspace-energy-outside-frozen-band")
        elif joint_fraction < self.policy.minimum_joint_pass_fraction:
            classification = "gauge-or-dependence-negative"
            reasons.append("joint-nees-outside-band-with-subspaces-adequate")
        else:
            classification = "covariance-adequate"
            reasons.append("covariance-diagnostics-within-frozen-bands")

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "source_mean_status", mean_status)
        object.__setattr__(self, "identity_reliability_status", identity_status)
        object.__setattr__(self, "source_mean_reasons", mean_reasons)
        object.__setattr__(self, "identity_reliability_reasons", identity_reasons)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "group_count", count)
        object.__setattr__(self, "joint_pass_fraction", joint_fraction)
        object.__setattr__(self, "shared_pass_fraction", shared_fraction)
        object.__setattr__(self, "conditional_pass_fraction", conditional_fraction)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "reason_codes", tuple(sorted(set(reasons))))
        object.__setattr__(
            self,
            "authorize_point_uncertainty_development",
            classification == "point-covariance-localized",
        )
        object.__setattr__(
            self,
            "source_covariance_localization_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": SOURCE_COVARIANCE_LOCALIZATION_SCHEMA,
            "schema_version": SOURCE_COVARIANCE_LOCALIZATION_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "cohort_binding_id": self.cohort_binding_id,
            "source_provider_competence_id": self.source_provider_competence_id,
            "joint_diagnostic_sha256": self.joint_diagnostic_sha256,
            "joint_residual_source_sha256": self.joint_residual_source_sha256,
            "policy": self.policy.to_dict(),
            "groups": [item.to_dict() for item in self.groups],
            "group_count": self.group_count,
            "joint_pass_fraction": self.joint_pass_fraction,
            "shared_pass_fraction": self.shared_pass_fraction,
            "conditional_pass_fraction": self.conditional_pass_fraction,
            "classification": self.classification,
            "reason_codes": list(self.reason_codes),
            "authorize_point_uncertainty_development": (
                self.authorize_point_uncertainty_development
            ),
            "metadata": {
                **plain_json(self.metadata),
                "source_mean_status": self.source_mean_status,
                "identity_reliability_status": self.identity_reliability_status,
                "source_mean_reasons": list(self.source_mean_reasons),
                "identity_reliability_reasons": list(self.identity_reliability_reasons),
            },
            "claim_boundary": SOURCE_COVARIANCE_LOCALIZATION_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "source_covariance_localization_id": self.source_covariance_localization_id,
        }

    def readiness_gates(self) -> tuple[ReadinessGateV1, ReadinessGateV1]:
        """Return the gauge/dependence and point-covariance gates in readiness order."""

        evidence_id = self.source_covariance_localization_id
        if self.source_mean_status != "pass" or self.identity_reliability_status != "pass":
            return (
                unevaluated_gate("gauge-dependence"),
                unevaluated_gate("point-covariance"),
            )
        if self.classification == "technical-failure":
            return (
                ReadinessGateV1(
                    gate_name="gauge-dependence",
                    status="technical-failure",
                    evidence_id=evidence_id,
                    reason_codes=self.reason_codes,
                ),
                unevaluated_gate("point-covariance"),
            )
        if self.classification == "gauge-or-dependence-negative":
            return (
                ReadinessGateV1(
                    gate_name="gauge-dependence",
                    status="fail",
                    evidence_id=evidence_id,
                    reason_codes=self.reason_codes,
                ),
                unevaluated_gate("point-covariance"),
            )
        gauge_gate = ReadinessGateV1(
            gate_name="gauge-dependence",
            status="pass",
            evidence_id=evidence_id,
        )
        if self.classification == "point-covariance-localized":
            point_gate = ReadinessGateV1(
                gate_name="point-covariance",
                status="fail",
                evidence_id=evidence_id,
                reason_codes=self.reason_codes,
            )
        else:
            point_gate = ReadinessGateV1(
                gate_name="point-covariance",
                status="pass",
                evidence_id=evidence_id,
            )
        return gauge_gate, point_gate

    @classmethod
    def from_dict(cls, value: object) -> SourceCovarianceLocalizationV1:
        mapping = require_mapping(value, name="source covariance localization")
        require_exact_fields(mapping, _ARTIFACT_FIELDS, name="source covariance localization")
        if mapping["schema"] != SOURCE_COVARIANCE_LOCALIZATION_SCHEMA:
            raise ValueError("source covariance localization schema changed")
        if mapping["schema_version"] != SOURCE_COVARIANCE_LOCALIZATION_VERSION:
            raise ValueError("source covariance localization version changed")
        if mapping["claim_boundary"] != SOURCE_COVARIANCE_LOCALIZATION_CLAIM_BOUNDARY:
            raise ValueError("source covariance localization claim boundary changed")
        metadata = require_mapping(mapping["metadata"], name="metadata")
        required_metadata = {
            "source_mean_status",
            "identity_reliability_status",
            "source_mean_reasons",
            "identity_reliability_reasons",
        }
        if not required_metadata <= metadata.keys():
            raise ValueError("source covariance localization status metadata is incomplete")
        user_metadata = {
            key: item for key, item in metadata.items() if key not in required_metadata
        }
        raw_groups = mapping["groups"]
        if type(raw_groups) is not list:
            raise ValueError("groups must be a JSON array")
        result = cls(
            provider_manifest_id=mapping["provider_manifest_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            source_provider_competence_id=mapping["source_provider_competence_id"],
            joint_diagnostic_sha256=mapping["joint_diagnostic_sha256"],
            joint_residual_source_sha256=mapping["joint_residual_source_sha256"],
            policy=SourceCovarianceLocalizationPolicyV1.from_dict(mapping["policy"]),
            groups=tuple(
                SourceCovarianceLocalizationGroupV1.from_dict(item)
                for item in raw_groups
            ),
            source_mean_status=metadata["source_mean_status"],
            identity_reliability_status=metadata["identity_reliability_status"],
            source_mean_reasons=_strings_from_json(
                metadata["source_mean_reasons"],
                name="source_mean_reasons",
            ),
            identity_reliability_reasons=_strings_from_json(
                metadata["identity_reliability_reasons"],
                name="identity_reliability_reasons",
            ),
            metadata=require_finite_json_mapping(user_metadata, name="metadata"),
        )
        supplied_id = require_sha256(
            mapping["source_covariance_localization_id"],
            name="source_covariance_localization_id",
        )
        if supplied_id != result.source_covariance_localization_id:
            raise ValueError("source covariance localization identity mismatch")
        if plain_json(mapping) != result.to_dict():
            raise ValueError("source covariance localization derived fields changed")
        return result


def localize_source_covariance(
    source_report: SourceProviderCompetenceReportV1,
    joint_diagnostic: Mapping[str, Any],
    *,
    joint_diagnostic_sha256: str,
    policy: SourceCovarianceLocalizationPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> SourceCovarianceLocalizationV1:
    """Build one localization from exact source competence and covariance evidence."""

    if not isinstance(source_report, SourceProviderCompetenceReportV1):
        raise TypeError("source_report must be SourceProviderCompetenceReportV1")
    if not isinstance(policy, SourceCovarianceLocalizationPolicyV1):
        raise TypeError("policy must be SourceCovarianceLocalizationPolicyV1")
    digest = require_sha256(joint_diagnostic_sha256, name="joint_diagnostic_sha256")
    groups = _normalize_joint_groups(joint_diagnostic, policy=policy)
    evaluable_source_groups = tuple(
        group.group_id for group in source_report.groups if group.evaluable
    )
    diagnostic_group_ids = tuple(group.group_id for group in groups)
    if diagnostic_group_ids != evaluable_source_groups:
        raise ValueError(
            "joint covariance factor groups do not match evaluable source competence groups"
        )
    residual_source_sha256 = require_sha256(
        joint_diagnostic.get("source_sha256"),
        name="joint diagnostic source_sha256",
    )
    return SourceCovarianceLocalizationV1(
        provider_manifest_id=source_report.provider_manifest_id,
        cohort_binding_id=source_report.cohort_binding_id,
        source_provider_competence_id=source_report.source_provider_competence_id,
        joint_diagnostic_sha256=digest,
        joint_residual_source_sha256=residual_source_sha256,
        policy=policy,
        groups=groups,
        source_mean_status=source_report.mean_quality_status,
        identity_reliability_status=source_report.identity_reliability_status,
        source_mean_reasons=source_report.mean_quality_reasons,
        identity_reliability_reasons=source_report.identity_reliability_reasons,
        metadata={} if metadata is None else metadata,
    )


def write_source_covariance_localization(
    path: str | Path,
    result: SourceCovarianceLocalizationV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(result, SourceCovarianceLocalizationV1):
        raise TypeError("result must be SourceCovarianceLocalizationV1")
    payload = json.dumps(result.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def load_source_covariance_localization(
    path: str | Path,
) -> SourceCovarianceLocalizationV1:
    return SourceCovarianceLocalizationV1.from_dict(
        load_json_object(path, name="source covariance localization")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source-competence", type=Path, required=True)
    evaluate.add_argument("--joint-diagnostic", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "evaluate":
        source = load_source_provider_competence(arguments.source_competence)
        joint_bytes = arguments.joint_diagnostic.read_bytes()
        joint = _json_bytes_object(joint_bytes, name="joint covariance diagnostic")
        policy = SourceCovarianceLocalizationPolicyV1.from_dict(
            load_json_object(arguments.policy, name="source covariance localization policy")
        )
        result = localize_source_covariance(
            source,
            joint,
            joint_diagnostic_sha256=hashlib.sha256(joint_bytes).hexdigest(),
            policy=policy,
        )
        write_source_covariance_localization(
            arguments.output,
            result,
            overwrite=arguments.overwrite,
        )
    else:
        result = load_source_covariance_localization(arguments.artifact)
    print(
        json.dumps(
            {
                "source_covariance_localization_id": result.source_covariance_localization_id,
                "classification": result.classification,
                "authorize_point_uncertainty_development": (
                    result.authorize_point_uncertainty_development
                ),
                "joint_pass_fraction": result.joint_pass_fraction,
                "shared_pass_fraction": result.shared_pass_fraction,
                "conditional_pass_fraction": result.conditional_pass_fraction,
            },
            sort_keys=True,
        )
    )
    return 0 if result.classification == "covariance-adequate" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SOURCE_COVARIANCE_LOCALIZATION_CLAIM_BOUNDARY",
    "SOURCE_COVARIANCE_LOCALIZATION_SCHEMA",
    "SOURCE_COVARIANCE_LOCALIZATION_VERSION",
    "SourceCovarianceLocalizationGroupV1",
    "SourceCovarianceLocalizationPolicyV1",
    "SourceCovarianceLocalizationV1",
    "load_source_covariance_localization",
    "localize_source_covariance",
    "write_source_covariance_localization",
]
