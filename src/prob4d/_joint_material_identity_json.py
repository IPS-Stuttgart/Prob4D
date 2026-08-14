"""Strict JSON helpers for joint material-identity artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_POSTERIOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "posterior_id",
        "window_order",
        "maximum_joint_assignments",
        "unconstrained_assignment_count",
        "feasible_assignment_count",
        "rejected_assignment_count",
        "log_normalizer",
        "joint_entropy_nats",
        "effective_assignment_count",
        "constraint_semantics",
        "mixtures",
        "assignments",
        "marginals",
        "metadata",
        "claim_boundary",
    }
)
_MIXTURE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "mixture_id",
        "target_endpoint",
        "window_order",
        "causal_frame_stop",
        "association_rule_id",
        "calibration_id",
        "tracklet_producer_revision",
        "association_revision",
        "weight_semantics",
        "null_hypothesis_semantics",
        "candidates",
        "metadata",
        "claim_boundary",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "kind",
        "source_endpoint",
        "association_result_id",
        "source_score",
        "calibrated_log_weight",
        "metadata",
    }
)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read joint material-identity posterior {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("joint posterior root must be a JSON object")
    return value


def _fields(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value

