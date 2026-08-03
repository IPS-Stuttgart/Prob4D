"""Strict manifest contract for paired Prob4D provider evaluation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

PROVIDER_EVALUATION_SCHEMA = "prob4d.provider-evaluation"
PROVIDER_EVALUATION_VERSION = 1
PROVIDER_EVALUATION_DECISION_VERSION = 2
EvaluationModeName = Literal["metric", "prefix_aligned", "oracle_aligned"]
DecisionDirection = Literal["lower", "higher"]
DecisionCriterion = Literal["superiority", "noninferiority"]
_EVALUATION_MODES = {"metric", "prefix_aligned", "oracle_aligned"}
_DECISION_DIRECTIONS = {"lower", "higher"}
_DECISION_CRITERIA = {"superiority", "noninferiority"}
_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ProviderEvaluationCase:
    """One paired truth case with one prediction artifact per method."""

    case_id: str
    group_id: str
    truth_path: Path
    predictions: Mapping[str, Path]
    boundary_frames: tuple[int, ...] = ()
    prefix_frame_stop_exclusive: int | None = None

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        group_id = str(self.group_id).strip()
        if not case_id or not group_id:
            raise ValueError("provider-evaluation case and group IDs must be nonempty")
        predictions = {
            str(method).strip(): Path(path)
            for method, path in self.predictions.items()
        }
        if not predictions or any(not method for method in predictions):
            raise ValueError("provider-evaluation predictions must be a nonempty mapping")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in self.boundary_frames
        ):
            raise ValueError("provider-evaluation boundary frames must be integers")
        boundary_frames = tuple(int(value) for value in self.boundary_frames)
        if boundary_frames != tuple(sorted(set(boundary_frames))) or any(
            value < 0 for value in boundary_frames
        ):
            raise ValueError(
                "provider-evaluation boundary frames must be sorted unique "
                "nonnegative integers"
            )
        prefix_stop = self.prefix_frame_stop_exclusive
        if prefix_stop is not None:
            prefix_stop = int(prefix_stop)
            if prefix_stop < 1:
                raise ValueError("prefix_frame_stop_exclusive must be positive")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "truth_path", Path(self.truth_path))
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "boundary_frames", boundary_frames)
        object.__setattr__(self, "prefix_frame_stop_exclusive", prefix_stop)


@dataclass(frozen=True)
class ProviderEvaluationDecisionRule:
    """One preregistered paired comparison against the manifest reference method."""

    rule_id: str
    candidate_method: str
    metric: str
    direction: DecisionDirection
    criterion: DecisionCriterion
    margin: float = 0.0

    def __post_init__(self) -> None:
        rule_id = str(self.rule_id).strip()
        candidate_method = str(self.candidate_method).strip()
        metric = str(self.metric).strip()
        direction = str(self.direction).strip()
        criterion = str(self.criterion).strip()
        margin = float(self.margin)
        if not rule_id or not candidate_method:
            raise ValueError("decision rule and candidate IDs must be nonempty")
        if _METRIC_NAME.fullmatch(metric) is None:
            raise ValueError(
                "decision metric must be one unqualified evaluation metric name"
            )
        if direction not in _DECISION_DIRECTIONS:
            raise ValueError(
                f"decision direction must be one of {sorted(_DECISION_DIRECTIONS)}"
            )
        if criterion not in _DECISION_CRITERIA:
            raise ValueError(
                f"decision criterion must be one of {sorted(_DECISION_CRITERIA)}"
            )
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("decision margin must be finite and nonnegative")
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "candidate_method", candidate_method)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "criterion", criterion)
        object.__setattr__(self, "margin", margin)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "rule_id": self.rule_id,
            "candidate_method": self.candidate_method,
            "metric": self.metric,
            "direction": self.direction,
            "criterion": self.criterion,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class ProviderEvaluationDecisionPolicy:
    """Target-frozen provider competence rule set."""

    policy_id: str
    minimum_group_count: int
    rules: tuple[ProviderEvaluationDecisionRule, ...]

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id).strip()
        if not policy_id:
            raise ValueError("decision policy_id must be nonempty")
        minimum_group_count = self.minimum_group_count
        if (
            isinstance(minimum_group_count, bool)
            or not isinstance(minimum_group_count, int)
            or minimum_group_count < 1
        ):
            raise ValueError("decision minimum_group_count must be a positive integer")
        rules = tuple(self.rules)
        if not rules:
            raise ValueError("decision policy must contain at least one rule")
        rule_ids = tuple(rule.rule_id for rule in rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("decision rule IDs must be unique")
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "minimum_group_count", minimum_group_count)
        object.__setattr__(self, "rules", rules)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "minimum_group_count": self.minimum_group_count,
            "rules": [rule.to_dict() for rule in self.rules],
        }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def validate_finite_json(value: object, *, name: str) -> None:
    """Reject non-finite or non-JSON report and manifest values."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_json(item, name=f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            validate_finite_json(item, name=f"{name}.{key}")
        return
    raise ValueError(f"{name} contains an unsupported JSON value")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _resolve_path(root: Path, value: object, *, name: str) -> Path:
    raw = Path(_text(value, name=name))
    resolved = raw if raw.is_absolute() else root / raw
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _decision_policy(
    value: object,
    *,
    methods: tuple[str, ...],
    reference_method: str,
) -> ProviderEvaluationDecisionPolicy:
    policy = _mapping(value, name="decision_policy")
    if set(policy) != {"policy_id", "minimum_group_count", "rules"}:
        raise ValueError("decision_policy fields changed")
    raw_rules = policy.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("decision_policy.rules must be a nonempty array")
    rules: list[ProviderEvaluationDecisionRule] = []
    for index, raw_rule in enumerate(raw_rules):
        item = _mapping(raw_rule, name=f"decision_policy.rules[{index}]")
        fields = {
            "rule_id",
            "candidate_method",
            "metric",
            "direction",
            "criterion",
            "margin",
        }
        if set(item) != fields:
            raise ValueError(f"decision_policy.rules[{index}] fields changed")
        direction = _text(
            item.get("direction"),
            name=f"decision_policy.rules[{index}].direction",
        )
        criterion = _text(
            item.get("criterion"),
            name=f"decision_policy.rules[{index}].criterion",
        )
        margin_value = item.get("margin")
        if isinstance(margin_value, bool) or not isinstance(
            margin_value, (int, float)
        ):
            raise ValueError(
                f"decision_policy.rules[{index}].margin must be numeric"
            )
        rule = ProviderEvaluationDecisionRule(
            rule_id=_text(
                item.get("rule_id"),
                name=f"decision_policy.rules[{index}].rule_id",
            ),
            candidate_method=_text(
                item.get("candidate_method"),
                name=f"decision_policy.rules[{index}].candidate_method",
            ),
            metric=_text(
                item.get("metric"),
                name=f"decision_policy.rules[{index}].metric",
            ),
            direction=cast(DecisionDirection, direction),
            criterion=cast(DecisionCriterion, criterion),
            margin=float(margin_value),
        )
        if rule.candidate_method == reference_method:
            raise ValueError("decision candidate must differ from reference_method")
        if rule.candidate_method not in methods:
            raise ValueError(
                f"decision candidate {rule.candidate_method!r} is not a registered method"
            )
        rules.append(rule)
    minimum_group_count = policy.get("minimum_group_count")
    if isinstance(minimum_group_count, bool) or not isinstance(
        minimum_group_count, int
    ):
        raise ValueError("decision_policy.minimum_group_count must be an integer")
    return ProviderEvaluationDecisionPolicy(
        policy_id=_text(policy.get("policy_id"), name="decision_policy.policy_id"),
        minimum_group_count=minimum_group_count,
        rules=tuple(rules),
    )


def load_provider_evaluation_plan(
    manifest_path: Path,
) -> tuple[
    list[ProviderEvaluationCase],
    EvaluationModeName,
    str,
    dict[str, Any],
    ProviderEvaluationDecisionPolicy | None,
]:
    """Load paired cases and an optional target-frozen decision policy."""

    try:
        manifest = _mapping(
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_constant,
            ),
            name=str(manifest_path),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read provider-evaluation manifest: {manifest_path}"
        ) from error
    if manifest.get("schema_name") != PROVIDER_EVALUATION_SCHEMA:
        raise ValueError("unsupported provider-evaluation manifest schema")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in {
        PROVIDER_EVALUATION_VERSION,
        PROVIDER_EVALUATION_DECISION_VERSION,
    }:
        raise ValueError("unsupported provider-evaluation manifest version")
    expected_fields = {
        "schema_name",
        "schema_version",
        "primary_mode",
        "reference_method",
        "cases",
        "metadata",
    }
    if schema_version == PROVIDER_EVALUATION_DECISION_VERSION:
        expected_fields.add("decision_policy")
    actual_fields = set(manifest)
    if actual_fields != expected_fields:
        raise ValueError(
            "provider-evaluation manifest fields changed: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"extra={sorted(actual_fields - expected_fields)}"
        )
    primary_mode_value = _text(manifest.get("primary_mode"), name="primary_mode")
    if primary_mode_value not in _EVALUATION_MODES:
        raise ValueError(f"primary_mode must be one of {sorted(_EVALUATION_MODES)}")
    primary_mode = cast(EvaluationModeName, primary_mode_value)
    if (
        schema_version == PROVIDER_EVALUATION_DECISION_VERSION
        and primary_mode == "oracle_aligned"
    ):
        raise ValueError("decision-bearing evaluation cannot use oracle_aligned mode")
    reference_method = _text(
        manifest.get("reference_method"),
        name="reference_method",
    )
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("provider-evaluation cases must be a nonempty array")
    metadata = dict(_mapping(manifest.get("metadata"), name="metadata"))
    validate_finite_json(metadata, name="metadata")

    root = manifest_path.resolve().parent
    cases: list[ProviderEvaluationCase] = []
    case_ids: set[str] = set()
    expected_methods: tuple[str, ...] | None = None
    for index, raw_case in enumerate(raw_cases):
        item = _mapping(raw_case, name=f"cases[{index}]")
        case_fields = {
            "case_id",
            "group_id",
            "truth",
            "predictions",
            "boundary_frames",
            "prefix_frame_stop_exclusive",
        }
        if set(item) != case_fields:
            raise ValueError(f"cases[{index}] fields changed")
        predictions = _mapping(
            item.get("predictions"),
            name=f"cases[{index}].predictions",
        )
        resolved_predictions = {
            _text(method, name=f"cases[{index}].predictions method"): _resolve_path(
                root,
                path,
                name=f"cases[{index}].predictions[{method!r}]",
            )
            for method, path in predictions.items()
        }
        boundary = item.get("boundary_frames")
        if not isinstance(boundary, list):
            raise ValueError(f"cases[{index}].boundary_frames must be an array")
        prefix_stop = item.get("prefix_frame_stop_exclusive")
        if prefix_stop is not None and (
            isinstance(prefix_stop, bool) or not isinstance(prefix_stop, int)
        ):
            raise ValueError(
                f"cases[{index}].prefix_frame_stop_exclusive must be an integer or null"
            )
        case = ProviderEvaluationCase(
            case_id=_text(item.get("case_id"), name=f"cases[{index}].case_id"),
            group_id=_text(item.get("group_id"), name=f"cases[{index}].group_id"),
            truth_path=_resolve_path(
                root,
                item.get("truth"),
                name=f"cases[{index}].truth",
            ),
            predictions=resolved_predictions,
            boundary_frames=tuple(boundary),
            prefix_frame_stop_exclusive=prefix_stop,
        )
        if case.case_id in case_ids:
            raise ValueError(f"duplicate provider-evaluation case_id: {case.case_id}")
        case_ids.add(case.case_id)
        methods = tuple(sorted(case.predictions))
        if expected_methods is None:
            expected_methods = methods
        elif methods != expected_methods:
            raise ValueError(
                "all provider-evaluation cases must contain the same paired method set"
            )
        if primary_mode == "prefix_aligned" and case.prefix_frame_stop_exclusive is None:
            raise ValueError(
                "prefix_aligned primary evaluation requires a prefix stop for every case"
            )
        cases.append(case)
    assert expected_methods is not None
    if reference_method not in expected_methods:
        raise ValueError(
            "reference_method must identify one prediction method in every case"
        )
    policy = None
    if schema_version == PROVIDER_EVALUATION_DECISION_VERSION:
        policy = _decision_policy(
            manifest.get("decision_policy"),
            methods=expected_methods,
            reference_method=reference_method,
        )
    return cases, primary_mode, reference_method, metadata, policy


def load_provider_evaluation_cases(
    manifest_path: Path,
) -> tuple[list[ProviderEvaluationCase], EvaluationModeName, str, dict[str, Any]]:
    """Load only the frozen manifest-v1 paired-case surface."""

    cases, primary_mode, reference_method, metadata, policy = (
        load_provider_evaluation_plan(manifest_path)
    )
    if policy is not None:
        raise ValueError(
            "decision-bearing manifest requires load_provider_evaluation_plan"
        )
    return cases, primary_mode, reference_method, metadata


__all__ = [
    "PROVIDER_EVALUATION_DECISION_VERSION",
    "PROVIDER_EVALUATION_SCHEMA",
    "PROVIDER_EVALUATION_VERSION",
    "DecisionCriterion",
    "DecisionDirection",
    "EvaluationModeName",
    "ProviderEvaluationCase",
    "ProviderEvaluationDecisionPolicy",
    "ProviderEvaluationDecisionRule",
    "load_provider_evaluation_cases",
    "load_provider_evaluation_plan",
    "validate_finite_json",
]
