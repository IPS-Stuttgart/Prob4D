"""Immutable candidate, calibration-row, and deployment-decision records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _frozen_real_mapping,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_mapping,
    _strict_real,
    _strict_string,
)


@dataclass(frozen=True, slots=True)
class CandidateSpecV1:
    """One fully specified method/threshold candidate."""

    candidate_id: str
    method_id: str
    complexity_rank: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _strict_string(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "method_id",
            _strict_string(self.method_id, name="method_id"),
        )
        object.__setattr__(
            self,
            "complexity_rank",
            _strict_integer(self.complexity_rank, name="complexity_rank"),
        )
        object.__setattr__(
            self,
            "parameters",
            frozen_finite_json_mapping(self.parameters, name="parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "method_id": self.method_id,
            "complexity_rank": self.complexity_rank,
            "parameters": plain_json(self.parameters),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateSpecV1:
        mapping = _strict_mapping(value, name="candidate")
        _exact_keys(
            mapping,
            {"candidate_id", "method_id", "complexity_rank", "parameters", "metadata"},
            name="candidate",
        )
        return cls(
            candidate_id=mapping["candidate_id"],
            method_id=mapping["method_id"],
            complexity_rank=mapping["complexity_rank"],
            parameters=_strict_mapping(mapping["parameters"], name="parameters"),
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class CalibrationMetricRowV1:
    """One immutable object/session by candidate calibration row."""

    group_id: str
    candidate_id: str
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            _strict_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            _strict_string(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "metrics",
            _frozen_real_mapping(self.metrics, name="metrics"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "candidate_id": self.candidate_id,
            "metrics": plain_json(self.metrics),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CalibrationMetricRowV1:
        mapping = _strict_mapping(value, name="calibration row")
        _exact_keys(
            mapping,
            {"group_id", "candidate_id", "metrics", "metadata"},
            name="calibration row",
        )
        return cls(
            group_id=mapping["group_id"],
            candidate_id=mapping["candidate_id"],
            metrics=_strict_mapping(mapping["metrics"], name="metrics"),
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class DeploymentDecisionV1:
    """One guarded target decision with verifiable exact fallback semantics."""

    group_id: str
    candidate_id: str
    accepted: bool
    guard_name: str
    guard_value: float
    candidate_artifact_id: str
    fallback_artifact_id: str
    deployed_artifact_id: str
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            _strict_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "candidate_id",
            _strict_string(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "accepted",
            _strict_bool(self.accepted, name="accepted"),
        )
        object.__setattr__(
            self,
            "guard_name",
            _strict_string(self.guard_name, name="guard_name"),
        )
        object.__setattr__(
            self,
            "guard_value",
            _strict_real(self.guard_value, name="guard_value"),
        )
        for name in (
            "candidate_artifact_id",
            "fallback_artifact_id",
            "deployed_artifact_id",
        ):
            object.__setattr__(
                self,
                name,
                _strict_digest(getattr(self, name), name=name, pattern=_SHA256),
            )
        if self.candidate_artifact_id == self.fallback_artifact_id:
            raise ValueError("candidate and fallback artifact IDs must differ")
        expected = (
            self.candidate_artifact_id if self.accepted else self.fallback_artifact_id
        )
        if self.deployed_artifact_id != expected:
            mode = "candidate" if self.accepted else "fallback"
            raise ValueError(f"deployed_artifact_id must reproduce the declared {mode}")
        object.__setattr__(
            self,
            "reason",
            _strict_string(self.reason, name="reason"),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def exact_fallback_reproduced(self) -> bool:
        return self.accepted or self.deployed_artifact_id == self.fallback_artifact_id

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
            "guard_name": self.guard_name,
            "guard_value": self.guard_value,
            "candidate_artifact_id": self.candidate_artifact_id,
            "fallback_artifact_id": self.fallback_artifact_id,
            "deployed_artifact_id": self.deployed_artifact_id,
            "reason": self.reason,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> DeploymentDecisionV1:
        mapping = _strict_mapping(value, name="deployment decision")
        _exact_keys(
            mapping,
            {
                "group_id",
                "candidate_id",
                "accepted",
                "guard_name",
                "guard_value",
                "candidate_artifact_id",
                "fallback_artifact_id",
                "deployed_artifact_id",
                "reason",
                "metadata",
            },
            name="deployment decision",
        )
        return cls(
            group_id=mapping["group_id"],
            candidate_id=mapping["candidate_id"],
            accepted=mapping["accepted"],
            guard_name=mapping["guard_name"],
            guard_value=mapping["guard_value"],
            candidate_artifact_id=mapping["candidate_artifact_id"],
            fallback_artifact_id=mapping["fallback_artifact_id"],
            deployed_artifact_id=mapping["deployed_artifact_id"],
            reason=mapping["reason"],
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )
