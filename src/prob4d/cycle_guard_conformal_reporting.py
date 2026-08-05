"""Reporting and frozen decision helpers for conformal cycle-guard studies."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

import numpy as np

from .causal_gauge_graph_monte_carlo import _bootstrap_summary, _stable_seed

_METHOD_ORDER = (
    "tree",
    "full_joint_graph",
    "raw_guarded_graph",
    "empirical_normalized_guard",
    "conformal_normalized_guard",
)
_CLEAN_SCENARIOS = (
    "independent_clean",
    "correlated_clean",
    "highly_correlated_clean",
)
_STRONG_OUTLIER_SCENARIOS = (
    "correlated_strong_outliers",
    "highly_correlated_strong_outliers",
)
_MILD_OUTLIER_SCENARIO = "correlated_mild_outliers"


def _real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _aggregate_records(
    records: Sequence[Mapping[str, object]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    by_scenario_method: dict[
        tuple[str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    tree_by_trial: dict[tuple[str, int], Mapping[str, object]] = {}
    for record in records:
        scenario_id = str(record["scenario_id"])
        method_id = str(record["method_id"])
        trial_index = _integer(record["trial_index"], name="trial_index")
        by_scenario_method[(scenario_id, method_id)].append(record)
        if method_id == "tree":
            tree_by_trial[(scenario_id, trial_index)] = record

    metric_names = (
        "endpoint_displacement_rmse",
        "mean_window_displacement_rmse",
        "p90_window_displacement_rmse",
        "drift_slope",
        "coverage_95",
        "coverage_shortfall_95",
        "mean_normalized_nees",
        "mean_normalized_covariance_trace",
    )
    aggregate: list[dict[str, object]] = []
    scenario_ids = sorted({key[0] for key in by_scenario_method})
    for scenario_id in scenario_ids:
        for method_id in _METHOD_ORDER:
            group = by_scenario_method[(scenario_id, method_id)]
            if not group:
                raise ValueError(
                    f"scenario {scenario_id!r} lacks method {method_id!r} records"
                )
            metrics = {
                metric: _bootstrap_summary(
                    [_real(record[metric], name=metric) for record in group],
                    resamples=bootstrap_resamples,
                    seed=_stable_seed(bootstrap_seed, scenario_id, method_id, metric),
                )
                for metric in metric_names
            }
            paired: list[float] = []
            for record in group:
                trial_index = _integer(record["trial_index"], name="trial_index")
                tree_record = tree_by_trial[(scenario_id, trial_index)]
                paired.append(
                    _real(
                        record["endpoint_displacement_rmse"],
                        name="endpoint_displacement_rmse",
                    )
                    - _real(
                        tree_record["endpoint_displacement_rmse"],
                        name="tree endpoint_displacement_rmse",
                    )
                )
            fallback_values = [
                bool(record["fallback_applied"])
                for record in group
                if record["fallback_applied"] is not None
            ]
            injected_fallback = [
                bool(record["fallback_applied"])
                for record in group
                if bool(record["outlier_injected"])
                and record["fallback_applied"] is not None
            ]
            clean_fallback = [
                bool(record["fallback_applied"])
                for record in group
                if not bool(record["outlier_injected"])
                and record["fallback_applied"] is not None
            ]
            aggregate.append(
                {
                    "scenario_id": scenario_id,
                    "method_id": method_id,
                    "trial_count": len(group),
                    "outlier_injected_count": sum(
                        bool(record["outlier_injected"]) for record in group
                    ),
                    "metrics": metrics,
                    "paired_endpoint_delta_vs_tree": _bootstrap_summary(
                        paired,
                        resamples=bootstrap_resamples,
                        seed=_stable_seed(
                            bootstrap_seed,
                            scenario_id,
                            method_id,
                            "paired-endpoint-delta",
                        ),
                    ),
                    "fallback_rate": (
                        None
                        if not fallback_values
                        else float(np.mean(fallback_values))
                    ),
                    "outlier_detection_rate": (
                        None
                        if not injected_fallback
                        else float(np.mean(injected_fallback))
                    ),
                    "clean_false_fallback_rate": (
                        None
                        if not clean_fallback
                        else float(np.mean(clean_fallback))
                    ),
                }
            )
    return aggregate


def _guard_summary(
    records: Sequence[Mapping[str, object]],
    *,
    method_id: str,
) -> dict[str, object]:
    selected = [record for record in records if record["method_id"] == method_id]
    if not selected:
        raise ValueError(f"guard method {method_id!r} has no records")
    clean_rates: dict[str, float] = {}
    for scenario_id in _CLEAN_SCENARIOS:
        values = [
            bool(record["fallback_applied"])
            for record in selected
            if record["scenario_id"] == scenario_id
        ]
        if not values:
            raise ValueError(f"guard method {method_id!r} lacks {scenario_id!r}")
        clean_rates[scenario_id] = float(np.mean(values))
    strong = [
        bool(record["fallback_applied"])
        for record in selected
        if record["scenario_id"] in _STRONG_OUTLIER_SCENARIOS
        and bool(record["outlier_injected"])
    ]
    mild = [
        bool(record["fallback_applied"])
        for record in selected
        if record["scenario_id"] == _MILD_OUTLIER_SCENARIO
        and bool(record["outlier_injected"])
    ]
    all_injected = [
        bool(record["fallback_applied"])
        for record in selected
        if bool(record["outlier_injected"])
    ]
    if not strong or not mild or not all_injected:
        raise ValueError("registered outlier scenarios produced no injected cases")
    return {
        "clean_false_fallback_rates": clean_rates,
        "worst_clean_false_fallback_rate": max(clean_rates.values()),
        "strong_outlier_detection_rate": float(np.mean(strong)),
        "mild_outlier_detection_rate": float(np.mean(mild)),
        "all_outlier_detection_rate": float(np.mean(all_injected)),
        "strong_injected_count": len(strong),
        "mild_injected_count": len(mild),
        "all_injected_count": len(all_injected),
    }


def _decision(
    records: Sequence[Mapping[str, object]],
    aggregate: Sequence[Mapping[str, object]],
    *,
    conformal_miscoverage: float,
    guaranteed_miscoverage_upper_bound: float,
    endpoint_noninferiority_margin: float,
    minimum_clean_coverage: float,
    maximum_clean_covariance_trace_ratio: float,
) -> dict[str, object]:
    raw = _guard_summary(records, method_id="raw_guarded_graph")
    empirical = _guard_summary(records, method_id="empirical_normalized_guard")
    conformal = _guard_summary(records, method_id="conformal_normalized_guard")
    lookup = {
        (str(item["scenario_id"]), str(item["method_id"])): item
        for item in aggregate
    }

    endpoint_upper_bounds: dict[str, float] = {}
    clean_coverages: dict[str, float] = {}
    covariance_trace_ratios: dict[str, float] = {}
    for scenario_id in _CLEAN_SCENARIOS:
        candidate = lookup[(scenario_id, "conformal_normalized_guard")]
        tree = lookup[(scenario_id, "tree")]
        paired = candidate["paired_endpoint_delta_vs_tree"]
        candidate_metrics = candidate["metrics"]
        tree_metrics = tree["metrics"]
        assert isinstance(paired, Mapping)
        assert isinstance(candidate_metrics, Mapping)
        assert isinstance(tree_metrics, Mapping)
        endpoint_upper_bounds[scenario_id] = _real(
            paired["ci95_upper"], name="paired ci95_upper"
        )
        candidate_coverage = candidate_metrics["coverage_95"]
        candidate_trace = candidate_metrics["mean_normalized_covariance_trace"]
        tree_trace = tree_metrics["mean_normalized_covariance_trace"]
        assert isinstance(candidate_coverage, Mapping)
        assert isinstance(candidate_trace, Mapping)
        assert isinstance(tree_trace, Mapping)
        clean_coverages[scenario_id] = _real(
            candidate_coverage["mean"], name="candidate coverage mean"
        )
        tree_trace_mean = _real(tree_trace["mean"], name="tree covariance trace mean")
        if tree_trace_mean <= 0.0:
            raise ValueError("tree covariance trace must be positive")
        covariance_trace_ratios[scenario_id] = (
            _real(
                candidate_trace["mean"],
                name="candidate covariance trace mean",
            )
            / tree_trace_mean
        )

    empirical_strong = _real(
        empirical["strong_outlier_detection_rate"],
        name="empirical strong detection",
    )
    empirical_mild = _real(
        empirical["mild_outlier_detection_rate"],
        name="empirical mild detection",
    )
    conformal_strong = _real(
        conformal["strong_outlier_detection_rate"],
        name="conformal strong detection",
    )
    conformal_mild = _real(
        conformal["mild_outlier_detection_rate"],
        name="conformal mild detection",
    )
    conformal_worst = _real(
        conformal["worst_clean_false_fallback_rate"],
        name="conformal worst clean fallback",
    )
    empirical_worst = _real(
        empirical["worst_clean_false_fallback_rate"],
        name="empirical worst clean fallback",
    )
    criteria = {
        "finite_sample_bound_at_most_requested_miscoverage": (
            guaranteed_miscoverage_upper_bound <= conformal_miscoverage
        ),
        "strong_detection_at_least_0_95": conformal_strong >= 0.95,
        "strong_detection_noninferior_to_empirical_by_0_05": (
            conformal_strong >= empirical_strong - 0.05
        ),
        "mild_detection_at_least_0_90": conformal_mild >= 0.90,
        "mild_detection_noninferior_to_empirical_by_0_05": (
            conformal_mild >= empirical_mild - 0.05
        ),
        "worst_clean_false_fallback_at_most_0_10": conformal_worst <= 0.10,
        "worst_clean_false_fallback_not_worse_than_empirical": (
            conformal_worst <= empirical_worst
        ),
        "clean_endpoint_noninferiority": (
            max(endpoint_upper_bounds.values()) <= endpoint_noninferiority_margin
        ),
        "clean_coverage_at_least_registered_minimum": (
            min(clean_coverages.values()) >= minimum_clean_coverage
        ),
        "clean_covariance_width_within_registered_ratio": (
            max(covariance_trace_ratios.values())
            <= maximum_clean_covariance_trace_ratio
        ),
    }
    return {
        "registered_primary_endpoints": [
            "strong_outlier_detection_rate",
            "mild_outlier_detection_rate",
            "worst_clean_false_fallback_rate",
        ],
        "raw_guard": raw,
        "empirical_normalized_guard": empirical,
        "conformal_normalized_guard": conformal,
        "secondary_clean_diagnostics": {
            "endpoint_delta_ci95_upper_by_scenario": endpoint_upper_bounds,
            "coverage_by_scenario": clean_coverages,
            "covariance_trace_ratio_vs_tree_by_scenario": covariance_trace_ratios,
        },
        "registered_margins": {
            "conformal_miscoverage": conformal_miscoverage,
            "guaranteed_miscoverage_upper_bound": (
                guaranteed_miscoverage_upper_bound
            ),
            "endpoint_noninferiority_margin": endpoint_noninferiority_margin,
            "minimum_clean_coverage": minimum_clean_coverage,
            "maximum_clean_covariance_trace_ratio": (
                maximum_clean_covariance_trace_ratio
            ),
        },
        "criteria": criteria,
        "overall_passed": all(criteria.values()),
        "decision_semantics": (
            "Every preregistered finite-sample, detection, clean-fallback, endpoint, "
            "coverage, and covariance-width criterion must pass. Calibration and "
            "target seeds are disjoint from the prior normalized-guard study."
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["scenario_id", "method_id", "trial_index", "seed"]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flat_aggregate(aggregate: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in aggregate:
        metrics = item["metrics"]
        paired = item["paired_endpoint_delta_vs_tree"]
        assert isinstance(metrics, Mapping)
        assert isinstance(paired, Mapping)
        row: dict[str, object] = {
            "scenario_id": item["scenario_id"],
            "method_id": item["method_id"],
            "trial_count": item["trial_count"],
            "outlier_injected_count": item["outlier_injected_count"],
            "fallback_rate": item["fallback_rate"],
            "outlier_detection_rate": item["outlier_detection_rate"],
            "clean_false_fallback_rate": item["clean_false_fallback_rate"],
            "paired_endpoint_delta_mean": paired["mean"],
            "paired_endpoint_delta_ci95_lower": paired["ci95_lower"],
            "paired_endpoint_delta_ci95_upper": paired["ci95_upper"],
        }
        for metric_name, summary in metrics.items():
            assert isinstance(summary, Mapping)
            row[f"{metric_name}_mean"] = summary["mean"]
            row[f"{metric_name}_ci95_lower"] = summary["ci95_lower"]
            row[f"{metric_name}_ci95_upper"] = summary["ci95_upper"]
        rows.append(row)
    return rows


def _format_interval(summary: Mapping[str, object]) -> str:
    return (
        f"{_real(summary['mean'], name='interval mean'):.5g} "
        f"[{_real(summary['ci95_lower'], name='interval ci95_lower'):.5g}, "
        f"{_real(summary['ci95_upper'], name='interval ci95_upper'):.5g}]"
    )


def _write_markdown(
    path: Path,
    *,
    raw_threshold: float,
    empirical_threshold: float,
    conformal_threshold: Mapping[str, object],
    aggregate: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
) -> None:
    lines = [
        "# Finite-sample cycle-guard calibration study",
        "",
        "This is a fresh-seed controlled synthetic source-guard study, not a "
        "physical-twin claim.",
        "",
        f"Raw threshold: `{raw_threshold:.8g}`. Empirical normalized threshold: "
        f"`{empirical_threshold:.8g}`. Split-conformal threshold: "
        f"`{_real(conformal_threshold['threshold'], name='conformal threshold'):.8g}`.",
        "",
        "| Scenario | Method | Endpoint | Delta vs tree | Coverage | NEES | "
        "Fallback | Detection | Clean false fallback |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregate:
        metrics = item["metrics"]
        paired = item["paired_endpoint_delta_vs_tree"]
        assert isinstance(metrics, Mapping)
        assert isinstance(paired, Mapping)

        def optional(value: object) -> str:
            if value is None:
                return "—"
            return f"{_real(value, name='optional rate'):.3f}"

        coverage = metrics["coverage_95"]
        nees = metrics["mean_normalized_nees"]
        assert isinstance(coverage, Mapping)
        assert isinstance(nees, Mapping)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["scenario_id"]),
                    str(item["method_id"]),
                    _format_interval(metrics["endpoint_displacement_rmse"]),
                    _format_interval(paired),
                    f"{_real(coverage['mean'], name='coverage mean'):.3f}",
                    f"{_real(nees['mean'], name='NEES mean'):.3f}",
                    optional(item["fallback_rate"]),
                    optional(item["outlier_detection_rate"]),
                    optional(item["clean_false_fallback_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"Overall result: **{'PASS' if decision['overall_passed'] else 'FAIL'}**.",
            "",
            "```json",
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "## Claim boundary",
            "",
            "The result concerns fresh-seed synthetic source-cycle calibration only. "
            "It does not establish held-out physical-object provider competence, "
            "BayesianPhysTwin benefit, harmful-update control, or Causal4D "
            "intervention benefit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "_METHOD_ORDER",
    "_aggregate_records",
    "_decision",
    "_flat_aggregate",
    "_write_csv",
    "_write_markdown",
]
