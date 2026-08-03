"""Preregistered decision rules for held-out provider competence reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._provider_evaluation_manifest import (
    EvaluationModeName,
    ProviderEvaluationDecisionPolicy,
    ProviderEvaluationDecisionRule,
)


def _group_count(
    aggregate: Mapping[str, Any],
    *,
    reference_method: str,
) -> int:
    reference = aggregate.get(reference_method)
    if not isinstance(reference, Mapping):
        raise ValueError("decision policy reference method is missing from aggregate results")
    count = reference.get("group_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("aggregate reference group_count must be a positive integer")
    for method_id, summary in aggregate.items():
        if not isinstance(summary, Mapping) or summary.get("group_count") != count:
            raise ValueError(
                f"method {method_id!r} does not share the reference group count"
            )
    return count


def _rule_result(
    rule: ProviderEvaluationDecisionRule,
    *,
    comparison: Mapping[str, Any],
    primary_mode: EvaluationModeName,
) -> dict[str, Any]:
    metrics = comparison.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"candidate {rule.candidate_method!r} has no paired comparison metrics"
        )
    metric_path = f"{primary_mode}.metrics.{rule.metric}"
    summary = metrics.get(metric_path)
    if not isinstance(summary, Mapping):
        raise ValueError(
            f"decision rule {rule.rule_id!r} references unavailable metric "
            f"{metric_path!r}"
        )
    required = ("mean", "ci95_lower", "ci95_upper", "group_count")
    if any(name not in summary for name in required):
        raise ValueError(
            f"decision metric {metric_path!r} lacks a complete bootstrap summary"
        )
    estimate = float(summary["mean"])
    lower = float(summary["ci95_lower"])
    upper = float(summary["ci95_upper"])

    if rule.direction == "lower":
        bound_name = "ci95_upper"
        bound_value = upper
        threshold = -rule.margin if rule.criterion == "superiority" else rule.margin
        passed = bound_value <= threshold
    else:
        bound_name = "ci95_lower"
        bound_value = lower
        threshold = rule.margin if rule.criterion == "superiority" else -rule.margin
        passed = bound_value >= threshold

    return {
        "rule_id": rule.rule_id,
        "candidate_method": rule.candidate_method,
        "reference_method": comparison.get("reference_method"),
        "metric": rule.metric,
        "metric_path": metric_path,
        "direction": rule.direction,
        "criterion": rule.criterion,
        "margin": rule.margin,
        "difference_semantics": "candidate_minus_reference",
        "estimate": estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "decision_bound": bound_name,
        "decision_bound_value": bound_value,
        "pass_threshold": threshold,
        "passed": passed,
    }


def evaluate_provider_decision_policy(
    policy: ProviderEvaluationDecisionPolicy,
    *,
    aggregate: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    primary_mode: EvaluationModeName,
    reference_method: str,
) -> dict[str, Any]:
    """Evaluate one target-frozen policy without selecting metrics after scoring."""

    observed_group_count = _group_count(
        aggregate,
        reference_method=reference_method,
    )
    group_count_passed = observed_group_count >= policy.minimum_group_count
    results: list[dict[str, Any]] = []
    for rule in policy.rules:
        comparison = comparisons.get(rule.candidate_method)
        if not isinstance(comparison, Mapping):
            raise ValueError(
                f"decision candidate {rule.candidate_method!r} has no paired comparison"
            )
        if comparison.get("reference_method") != reference_method:
            raise ValueError(
                f"decision candidate {rule.candidate_method!r} changed reference method"
            )
        results.append(
            _rule_result(
                rule,
                comparison=comparison,
                primary_mode=primary_mode,
            )
        )

    passed_rule_count = sum(bool(result["passed"]) for result in results)
    return {
        "policy_id": policy.policy_id,
        "primary_mode": primary_mode,
        "reference_method": reference_method,
        "minimum_group_count": policy.minimum_group_count,
        "observed_group_count": observed_group_count,
        "group_count_passed": group_count_passed,
        "rule_count": len(results),
        "passed_rule_count": passed_rule_count,
        "rules": results,
        "overall_passed": group_count_passed and passed_rule_count == len(results),
        "decision_semantics": (
            "All preregistered rules and the minimum independent-group gate must pass. "
            "Lower-is-better rules use the upper paired-bootstrap bound; "
            "higher-is-better rules use the lower bound."
        ),
        "claim_boundary": (
            "A passing policy establishes only held-out provider competence under the "
            "registered observation protocol. It does not establish Bayesian-PhysTwin "
            "acceptance or Causal4D intervention benefit."
        ),
    }


__all__ = ["evaluate_provider_decision_policy"]
