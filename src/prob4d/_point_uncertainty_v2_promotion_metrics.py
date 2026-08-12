"""Equal-group summary and promotion criteria for point uncertainty v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ._point_uncertainty_v2_promotion_types import (
    _NOMINAL_COVERAGE,
    PointUncertaintyGroupMetricsV1,
    PointUncertaintyPromotionPolicyV1,
)

_SUMMARY_FIELDS = frozenset(
    {
        "group_count",
        "minimum_group_rows",
        "baseline_mean_nll",
        "candidate_mean_nll",
        "mean_nll_improvement",
        "group_win_fraction",
        "baseline_coverage90",
        "candidate_coverage90",
        "baseline_coverage_error",
        "candidate_coverage_error",
        "coverage_error_increase",
        "worst_group_coverage_error_increase",
        "baseline_mean_rms_width",
        "candidate_mean_rms_width",
        "mean_width_ratio",
        "worst_group_width_ratio",
        "worst_group_nll_regression",
        "baseline_normalized_energy",
        "candidate_normalized_energy",
    }
)
_CRITERIA_FIELDS = frozenset(
    {
        "fit_converged",
        "minimum_group_count",
        "minimum_rows_per_group",
        "mean_nll_improvement",
        "group_win_fraction",
        "coverage_nonworse",
        "worst_group_coverage_nonworse",
        "width_budget",
        "worst_group_width_budget",
        "worst_group_nll_regression",
    }
)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summary(
    groups: tuple[PointUncertaintyGroupMetricsV1, ...],
) -> dict[str, float | int]:
    baseline_nll = _mean([group.baseline_mean_nll for group in groups])
    candidate_nll = _mean([group.candidate_mean_nll for group in groups])
    baseline_coverage = _mean([group.baseline_coverage90 for group in groups])
    candidate_coverage = _mean([group.candidate_coverage90 for group in groups])
    baseline_width = _mean([group.baseline_mean_rms_width for group in groups])
    candidate_width = _mean([group.candidate_mean_rms_width for group in groups])
    baseline_coverage_error = abs(baseline_coverage - _NOMINAL_COVERAGE)
    candidate_coverage_error = abs(candidate_coverage - _NOMINAL_COVERAGE)
    return {
        "group_count": len(groups),
        "minimum_group_rows": min(group.count for group in groups),
        "baseline_mean_nll": baseline_nll,
        "candidate_mean_nll": candidate_nll,
        "mean_nll_improvement": baseline_nll - candidate_nll,
        "group_win_fraction": _mean(
            [float(group.candidate_mean_nll < group.baseline_mean_nll) for group in groups]
        ),
        "baseline_coverage90": baseline_coverage,
        "candidate_coverage90": candidate_coverage,
        "baseline_coverage_error": baseline_coverage_error,
        "candidate_coverage_error": candidate_coverage_error,
        "coverage_error_increase": candidate_coverage_error - baseline_coverage_error,
        "worst_group_coverage_error_increase": max(
            abs(group.candidate_coverage90 - _NOMINAL_COVERAGE)
            - abs(group.baseline_coverage90 - _NOMINAL_COVERAGE)
            for group in groups
        ),
        "baseline_mean_rms_width": baseline_width,
        "candidate_mean_rms_width": candidate_width,
        "mean_width_ratio": candidate_width / baseline_width,
        "worst_group_width_ratio": max(
            group.candidate_mean_rms_width / group.baseline_mean_rms_width
            for group in groups
        ),
        "worst_group_nll_regression": max(
            group.candidate_mean_nll - group.baseline_mean_nll for group in groups
        ),
        "baseline_normalized_energy": _mean(
            [group.baseline_normalized_energy for group in groups]
        ),
        "candidate_normalized_energy": _mean(
            [group.candidate_normalized_energy for group in groups]
        ),
    }


def _criteria(
    fit_converged: bool,
    summary: Mapping[str, float | int],
    policy: PointUncertaintyPromotionPolicyV1,
) -> dict[str, bool]:
    return {
        "fit_converged": fit_converged,
        "minimum_group_count": int(summary["group_count"]) >= policy.minimum_group_count,
        "minimum_rows_per_group": (
            int(summary["minimum_group_rows"]) >= policy.minimum_rows_per_group
        ),
        "mean_nll_improvement": (
            float(summary["mean_nll_improvement"])
            >= policy.minimum_mean_nll_improvement
        ),
        "group_win_fraction": (
            float(summary["group_win_fraction"])
            >= policy.minimum_group_win_fraction
        ),
        "coverage_nonworse": (
            float(summary["coverage_error_increase"])
            <= policy.maximum_coverage_error_increase
        ),
        "worst_group_coverage_nonworse": (
            float(summary["worst_group_coverage_error_increase"])
            <= policy.maximum_worst_group_coverage_error_increase
        ),
        "width_budget": (
            float(summary["mean_width_ratio"]) <= policy.maximum_mean_width_ratio
        ),
        "worst_group_width_budget": (
            float(summary["worst_group_width_ratio"])
            <= policy.maximum_worst_group_width_ratio
        ),
        "worst_group_nll_regression": (
            float(summary["worst_group_nll_regression"])
            <= policy.maximum_worst_group_nll_regression
        ),
    }


__all__ = []
