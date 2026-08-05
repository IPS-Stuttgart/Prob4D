"""Shared validation and selection-rule types for replayable evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._immutable_json import frozen_finite_json_mapping, plain_json

SELECTION_EVIDENCE_SCHEMA = "prob4d.selection-evidence"
SELECTION_EVIDENCE_VERSION = 2
SELECTION_REPLAY_SCHEMA = "prob4d.selection-replay"
FINAL_TIE_BREAK = "complexity-rank-then-candidate-id"

Direction = Literal["minimize", "maximize"]
Aggregation = Literal["mean", "sum", "max", "min"]
ConstraintRelation = Literal["at_most", "at_least"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _strict_digest(value: Any, *, name: str, pattern: re.Pattern[str]) -> str:
    result = _strict_string(value, name=name)
    if pattern.fullmatch(result) is None:
        raise ValueError(f"{name} has a noncanonical digest format")
    return result


def _strict_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strict_real(value: Any, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _strict_list(value: Any, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise ValueError(f"{name} fields are invalid: {'; '.join(details)}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _frozen_real_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not value:
        raise ValueError(f"{name} must not be empty")
    normalized: dict[str, float] = {}
    for metric_name, metric_value in value.items():
        key = _strict_string(metric_name, name=f"{name} key")
        normalized[key] = _strict_real(metric_value, name=f"{name}[{key!r}]")
    return frozen_finite_json_mapping(normalized, name=name)


def _direction(value: Any, *, name: str) -> Direction:
    result = _strict_string(value, name=name)
    if result not in {"minimize", "maximize"}:
        raise ValueError(f"{name} must be 'minimize' or 'maximize'")
    return cast(Direction, result)


def _aggregation(value: Any, *, name: str) -> Aggregation:
    result = _strict_string(value, name=name)
    if result not in {"mean", "sum", "max", "min"}:
        raise ValueError(f"{name} must be one of mean, sum, max, or min")
    return cast(Aggregation, result)


def _constraint_relation(value: Any, *, name: str) -> ConstraintRelation:
    result = _strict_string(value, name=name)
    if result not in {"at_most", "at_least"}:
        raise ValueError(f"{name} must be 'at_most' or 'at_least'")
    return cast(ConstraintRelation, result)


@dataclass(frozen=True, slots=True)
class MetricOrderV1:
    """One aggregate metric and its deterministic ordering direction."""

    metric_name: str
    direction: Direction
    aggregation: Aggregation = "mean"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_name",
            _strict_string(self.metric_name, name="metric_name"),
        )
        object.__setattr__(
            self,
            "direction",
            _direction(self.direction, name="direction"),
        )
        object.__setattr__(
            self,
            "aggregation",
            _aggregation(self.aggregation, name="aggregation"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "direction": self.direction,
            "aggregation": self.aggregation,
        }

    @classmethod
    def from_dict(cls, value: Any) -> MetricOrderV1:
        mapping = _strict_mapping(value, name="metric order")
        _exact_keys(
            mapping,
            {"metric_name", "direction", "aggregation"},
            name="metric order",
        )
        return cls(
            metric_name=mapping["metric_name"],
            direction=mapping["direction"],
            aggregation=mapping["aggregation"],
        )


@dataclass(frozen=True, slots=True)
class MetricConstraintV1:
    """One source-calibration feasibility constraint."""

    metric_name: str
    relation: ConstraintRelation
    threshold: float
    aggregation: Aggregation = "mean"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_name",
            _strict_string(self.metric_name, name="metric_name"),
        )
        object.__setattr__(
            self,
            "relation",
            _constraint_relation(self.relation, name="relation"),
        )
        object.__setattr__(
            self,
            "threshold",
            _strict_real(self.threshold, name="threshold"),
        )
        object.__setattr__(
            self,
            "aggregation",
            _aggregation(self.aggregation, name="aggregation"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "relation": self.relation,
            "threshold": self.threshold,
            "aggregation": self.aggregation,
        }

    @classmethod
    def from_dict(cls, value: Any) -> MetricConstraintV1:
        mapping = _strict_mapping(value, name="metric constraint")
        _exact_keys(
            mapping,
            {"metric_name", "relation", "threshold", "aggregation"},
            name="metric constraint",
        )
        return cls(
            metric_name=mapping["metric_name"],
            relation=mapping["relation"],
            threshold=mapping["threshold"],
            aggregation=mapping["aggregation"],
        )


@dataclass(frozen=True, slots=True)
class SelectionRuleV1:
    """Frozen target-blind rule used to rank calibration candidates."""

    primary: MetricOrderV1
    tie_break_metrics: tuple[MetricOrderV1, ...] = ()
    constraints: tuple[MetricConstraintV1, ...] = ()
    final_tie_break: str = FINAL_TIE_BREAK

    def __post_init__(self) -> None:
        if not isinstance(self.primary, MetricOrderV1):
            raise ValueError("primary must be a MetricOrderV1")
        if type(self.tie_break_metrics) is not tuple or not all(
            isinstance(item, MetricOrderV1) for item in self.tie_break_metrics
        ):
            raise ValueError("tie_break_metrics must be a tuple of MetricOrderV1")
        if type(self.constraints) is not tuple or not all(
            isinstance(item, MetricConstraintV1) for item in self.constraints
        ):
            raise ValueError("constraints must be a tuple of MetricConstraintV1")
        metric_keys = [
            (self.primary.metric_name, self.primary.aggregation),
            *[
                (item.metric_name, item.aggregation)
                for item in self.tie_break_metrics
            ],
        ]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("primary and tie-break metric orders must be unique")
        constraint_keys = [
            (item.metric_name, item.aggregation, item.relation)
            for item in self.constraints
        ]
        if len(constraint_keys) != len(set(constraint_keys)):
            raise ValueError("constraints must be unique")
        if self.final_tie_break != FINAL_TIE_BREAK:
            raise ValueError(f"final_tie_break must equal {FINAL_TIE_BREAK!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary.to_dict(),
            "tie_break_metrics": [item.to_dict() for item in self.tie_break_metrics],
            "constraints": [item.to_dict() for item in self.constraints],
            "final_tie_break": self.final_tie_break,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SelectionRuleV1:
        mapping = _strict_mapping(value, name="selection rule")
        _exact_keys(
            mapping,
            {"primary", "tie_break_metrics", "constraints", "final_tie_break"},
            name="selection rule",
        )
        tie_break_values = _strict_list(
            mapping["tie_break_metrics"],
            name="tie_break_metrics",
        )
        constraint_values = _strict_list(mapping["constraints"], name="constraints")
        return cls(
            primary=MetricOrderV1.from_dict(mapping["primary"]),
            tie_break_metrics=tuple(
                MetricOrderV1.from_dict(item) for item in tie_break_values
            ),
            constraints=tuple(
                MetricConstraintV1.from_dict(item) for item in constraint_values
            ),
            final_tie_break=mapping["final_tie_break"],
        )
