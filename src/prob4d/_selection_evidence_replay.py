"""Deterministic replay of candidate selection from retained calibration rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    SELECTION_REPLAY_SCHEMA,
    Aggregation,
    Direction,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionRuleV1,
    _sha256_json,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._selection_evidence_records import CalibrationMetricRowV1, CandidateSpecV1


def _aggregate(values: Sequence[float], aggregation: Aggregation) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty sequence")
    if aggregation == "mean":
        return math.fsum(values) / len(values)
    if aggregation == "sum":
        return math.fsum(values)
    if aggregation == "max":
        return max(values)
    return min(values)


def _metric_key(value: float, direction: Direction) -> float:
    return value if direction == "minimize" else -value


def _constraint_passes(value: float, constraint: MetricConstraintV1) -> bool:
    if constraint.relation == "at_most":
        return value <= constraint.threshold
    return value >= constraint.threshold


def _rectangular_rows(
    candidates: tuple[CandidateSpecV1, ...],
    rows: tuple[CalibrationMetricRowV1, ...],
) -> tuple[tuple[str, ...], dict[tuple[str, str], CalibrationMetricRowV1]]:
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    candidate_set = set(candidate_ids)
    by_key: dict[tuple[str, str], CalibrationMetricRowV1] = {}
    group_ids: set[str] = set()
    for row in rows:
        if row.candidate_id not in candidate_set:
            raise ValueError(f"calibration row references unknown candidate {row.candidate_id!r}")
        key = (row.group_id, row.candidate_id)
        if key in by_key:
            raise ValueError(f"duplicate calibration row {key!r}")
        by_key[key] = row
        group_ids.add(row.group_id)
    if not group_ids:
        raise ValueError("calibration_rows must contain at least one group")
    ordered_groups = tuple(sorted(group_ids))
    expected = {
        (group_id, candidate_id)
        for group_id in ordered_groups
        for candidate_id in candidate_ids
    }
    actual = set(by_key)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "calibration rows must form a complete group-by-candidate matrix; "
            f"missing={missing}, extra={extra}"
        )
    return ordered_groups, by_key


def _required_metric_orders(rule: SelectionRuleV1) -> tuple[MetricOrderV1, ...]:
    return (rule.primary, *rule.tie_break_metrics)


def _candidate_replay_summaries(
    candidates: tuple[CandidateSpecV1, ...],
    rows: tuple[CalibrationMetricRowV1, ...],
    rule: SelectionRuleV1,
) -> tuple[dict[str, object], ...]:
    groups, by_key = _rectangular_rows(candidates, rows)
    required_metrics = {
        order.metric_name for order in _required_metric_orders(rule)
    } | {constraint.metric_name for constraint in rule.constraints}
    summaries: list[dict[str, object]] = []
    for candidate in candidates:
        aggregate_cache: dict[tuple[str, Aggregation], float] = {}

        def aggregate(
            metric_name: str,
            aggregation: Aggregation,
            *,
            candidate_id: str = candidate.candidate_id,
            cache: dict[tuple[str, Aggregation], float] = aggregate_cache,
        ) -> float:
            key = (metric_name, aggregation)
            if key not in cache:
                values: list[float] = []
                for group_id in groups:
                    row = by_key[(group_id, candidate_id)]
                    if metric_name not in row.metrics:
                        raise ValueError(
                            f"row {(group_id, candidate_id)!r} lacks "
                            f"required metric {metric_name!r}"
                        )
                    values.append(float(row.metrics[metric_name]))
                cache[key] = _aggregate(values, aggregation)
            return cache[key]

        for metric_name in required_metrics:
            matching_aggregations = {
                order.aggregation
                for order in _required_metric_orders(rule)
                if order.metric_name == metric_name
            } | {
                constraint.aggregation
                for constraint in rule.constraints
                if constraint.metric_name == metric_name
            }
            for aggregation in matching_aggregations:
                aggregate(metric_name, aggregation)
        constraint_results = [
            {
                "metric_name": constraint.metric_name,
                "aggregation": constraint.aggregation,
                "relation": constraint.relation,
                "threshold": constraint.threshold,
                "value": aggregate(
                    constraint.metric_name,
                    constraint.aggregation,
                ),
                "passed": _constraint_passes(
                    aggregate(constraint.metric_name, constraint.aggregation),
                    constraint,
                ),
            }
            for constraint in rule.constraints
        ]
        summaries.append(
            {
                "candidate_id": candidate.candidate_id,
                "method_id": candidate.method_id,
                "complexity_rank": candidate.complexity_rank,
                "aggregates": {
                    f"{metric_name}:{aggregation}": value
                    for (metric_name, aggregation), value in sorted(
                        aggregate_cache.items()
                    )
                },
                "constraints": constraint_results,
                "feasible": all(bool(item["passed"]) for item in constraint_results),
            }
        )
    return tuple(summaries)


def replay_candidate_order(
    candidates: Sequence[CandidateSpecV1],
    calibration_rows: Sequence[CalibrationMetricRowV1],
    selection_rule: SelectionRuleV1,
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    """Reconstruct the complete deterministic candidate order from retained rows."""

    candidate_tuple = tuple(candidates)
    row_tuple = tuple(calibration_rows)
    if not candidate_tuple or not all(
        isinstance(candidate, CandidateSpecV1) for candidate in candidate_tuple
    ):
        raise ValueError("candidates must contain CandidateSpecV1 values")
    if not row_tuple or not all(
        isinstance(row, CalibrationMetricRowV1) for row in row_tuple
    ):
        raise ValueError("calibration_rows must contain CalibrationMetricRowV1 values")
    if not isinstance(selection_rule, SelectionRuleV1):
        raise ValueError("selection_rule must be a SelectionRuleV1")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_tuple}
    if len(candidate_by_id) != len(candidate_tuple):
        raise ValueError("candidate IDs must be unique")
    summaries = _candidate_replay_summaries(candidate_tuple, row_tuple, selection_rule)
    summary_by_id = {str(summary["candidate_id"]): summary for summary in summaries}

    def aggregate(summary: Mapping[str, object], order: MetricOrderV1) -> float:
        aggregates = _strict_mapping(summary["aggregates"], name="aggregates")
        return float(aggregates[f"{order.metric_name}:{order.aggregation}"])

    def feasible_key(candidate_id: str) -> tuple[Any, ...]:
        summary = summary_by_id[candidate_id]
        candidate = candidate_by_id[candidate_id]
        return (
            0,
            _metric_key(
                aggregate(summary, selection_rule.primary),
                selection_rule.primary.direction,
            ),
            *[
                _metric_key(aggregate(summary, order), order.direction)
                for order in selection_rule.tie_break_metrics
            ],
            candidate.complexity_rank,
            candidate.candidate_id,
        )

    def infeasible_key(candidate_id: str) -> tuple[Any, ...]:
        summary = summary_by_id[candidate_id]
        constraints = _strict_list(summary["constraints"], name="constraints")
        failed_count = sum(
            not bool(_strict_mapping(item, name="constraint result")["passed"])
            for item in constraints
        )
        candidate = candidate_by_id[candidate_id]
        return (1, failed_count, candidate.complexity_rank, candidate.candidate_id)

    feasible_ids = [
        candidate_id
        for candidate_id, summary in summary_by_id.items()
        if bool(summary["feasible"])
    ]
    if not feasible_ids:
        raise ValueError("selection rule leaves no feasible candidate")
    order = tuple(
        sorted(
            summary_by_id,
            key=lambda candidate_id: (
                feasible_key(candidate_id)
                if bool(summary_by_id[candidate_id]["feasible"])
                else infeasible_key(candidate_id)
            ),
        )
    )
    ordered_summaries = tuple(summary_by_id[candidate_id] for candidate_id in order)
    return order, ordered_summaries


@dataclass(frozen=True, slots=True)
class SelectionReplayReportV1:
    """Deterministic verifier output independent of experiment implementation."""

    evidence_artifact_id: str
    candidate_order: tuple[str, ...]
    selected_candidate_id: str
    candidate_summaries: tuple[Mapping[str, Any], ...]
    deployment_group_count: int
    accepted_update_count: int
    fallback_update_count: int
    exact_fallback_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_artifact_id",
            _strict_digest(
                self.evidence_artifact_id,
                name="evidence_artifact_id",
                pattern=_SHA256,
            ),
        )
        if type(self.candidate_order) is not tuple or not self.candidate_order:
            raise ValueError("candidate_order must be a nonempty tuple")
        if len(set(self.candidate_order)) != len(self.candidate_order):
            raise ValueError("candidate_order must be unique")
        for index, candidate_id in enumerate(self.candidate_order):
            _strict_string(candidate_id, name=f"candidate_order[{index}]")
        if self.selected_candidate_id != self.candidate_order[0]:
            raise ValueError("selected_candidate_id must be first in candidate_order")
        if type(self.candidate_summaries) is not tuple:
            raise ValueError("candidate_summaries must be a tuple")
        frozen_summaries = tuple(
            frozen_finite_json_mapping(summary, name="candidate summary")
            for summary in self.candidate_summaries
        )
        object.__setattr__(self, "candidate_summaries", frozen_summaries)
        for name in (
            "deployment_group_count",
            "accepted_update_count",
            "fallback_update_count",
            "exact_fallback_count",
        ):
            object.__setattr__(
                self,
                name,
                _strict_integer(getattr(self, name), name=name),
            )
        if self.accepted_update_count + self.fallback_update_count != (
            self.deployment_group_count
        ):
            raise ValueError("deployment counts are inconsistent")
        if self.exact_fallback_count != self.fallback_update_count:
            raise ValueError("every rejected update must reproduce exact fallback")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": SELECTION_REPLAY_SCHEMA,
            "schema_version": 1,
            "evidence_artifact_id": self.evidence_artifact_id,
            "candidate_order": list(self.candidate_order),
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_summaries": [
                plain_json(summary) for summary in self.candidate_summaries
            ],
            "deployment_group_count": self.deployment_group_count,
            "accepted_update_count": self.accepted_update_count,
            "fallback_update_count": self.fallback_update_count,
            "exact_fallback_count": self.exact_fallback_count,
        }

    @property
    def replay_digest(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["replay_digest"] = self.replay_digest
        return result
