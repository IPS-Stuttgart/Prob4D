"""Evaluate finite-sample worst-group calibration for source-only cycle guards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from .causal_gauge_graph_monte_carlo import (
    DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS,
    GaugeGraphStudyScenario,
    _bootstrap_summary,
    _canonical_json,
    _EstimatorOutput,
    _evaluate_output,
    _inject_inconsistent_skip_edge,
    _make_problem,
    _run_estimators,
    _stable_seed,
    _validated_positive_integer,
    _validated_source_revision,
)
from .cycle_guard_group_robust import (
    GroupRobustCycleCalibration,
    cycle_source_uncertainty_feature,
    fit_group_robust_cycle_calibration,
)
from .cycle_guard_monte_carlo import (
    _assert_exact_fallback,
    _normalized_guard_output,
    _run_raw_audit,
)
from .observation_export import _build_alignments
from .uncertainty_normalized_cycles import (
    audit_uncertainty_normalized_alignment_cycles,
)

GROUP_ROBUST_CYCLE_GUARD_SCHEMA: Final = (
    "prob4d.group-robust-cycle-guard-monte-carlo"
)
GROUP_ROBUST_CYCLE_GUARD_VERSION: Final = 1
_METHOD_ORDER: Final = (
    "tree",
    "full_joint_graph",
    "raw_guarded_graph",
    "pooled_empirical_normalized_guard",
    "pooled_conformal_normalized_guard",
    "stratum_conformal_normalized_guard",
    "worst_group_conformal_normalized_guard",
)
_GUARD_METHODS: Final = _METHOD_ORDER[2:]
_PRIMARY_METHOD: Final = "worst_group_conformal_normalized_guard"
_CLEAN_SCENARIOS: Final = (
    "independent_clean",
    "correlated_clean",
    "highly_correlated_clean",
)
_STRONG_OUTLIER_SCENARIOS: Final = (
    "correlated_strong_outliers",
    "highly_correlated_strong_outliers",
)
_MILD_OUTLIER_SCENARIO: Final = "correlated_mild_outliers"
_PREVIOUS_NORMALIZED_STRONG_DETECTION: Final = 1.0
_PREVIOUS_NORMALIZED_MILD_DETECTION: Final = 1.0
_CLAIM_BOUNDARY: Final = (
    "Controlled synthetic source-cycle admission only; no held-out "
    "physical-object provider, BayesianPhysTwin, harmful-update, or Causal4D "
    "benefit claim."
)


def _clean_correlations(
    scenarios: Sequence[GaugeGraphStudyScenario],
) -> tuple[float, ...]:
    values = tuple(
        sorted(
            {
                scenario.correlation
                for scenario in scenarios
                if scenario.outlier_probability == 0.0
            }
        )
    )
    if not values:
        raise ValueError("group-robust study requires registered clean scenarios")
    return values


def _source_case(
    *,
    seed: int,
    correlation: float,
    representative_radius: float,
    minimum_uncertainty_scale: float,
    num_frames: int,
    height: int,
    width: int,
    window_size: int,
    overlap: int,
) -> tuple[Any, tuple[Any, ...], float, float, float, int]:
    problem = _make_problem(
        seed=seed,
        correlation=correlation,
        num_frames=num_frames,
        height=height,
        width=width,
        window_size=window_size,
        overlap=overlap,
    )
    alignments = tuple(_build_alignments(problem.overlap_windows))
    raw_audit = _run_raw_audit(
        alignments,
        representative_radius=representative_radius,
    )
    normalized_audit = audit_uncertainty_normalized_alignment_cycles(
        alignments,
        representative_radius=representative_radius,
        minimum_uncertainty_scale=minimum_uncertainty_scale,
    )
    if normalized_audit.cycle_count < 1:
        raise ValueError("registered synthetic geometry produced no gauge cycles")
    return (
        problem,
        alignments,
        cycle_source_uncertainty_feature(normalized_audit),
        normalized_audit.maximum_observed_normalized_score,
        float(raw_audit["maximum"]),
        normalized_audit.cycle_count,
    )


def _prepare_source_splits(
    scenarios: Sequence[GaugeGraphStudyScenario],
    *,
    development_trials: int,
    calibration_trials: int,
    development_seed: int,
    calibration_seed: int,
    conformal_alpha: float,
    minimum_calibration_per_stratum: int,
    empirical_quantile: float,
    representative_radius: float,
    minimum_uncertainty_scale: float,
    num_frames: int,
    height: int,
    width: int,
    window_size: int,
    overlap: int,
) -> tuple[
    GroupRobustCycleCalibration,
    float,
    float,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    correlations = _clean_correlations(scenarios)
    if development_trials < len(correlations):
        raise ValueError("development split must cover every clean source condition")
    if calibration_trials < len(correlations):
        raise ValueError("calibration split must cover every clean source condition")

    development_records: list[dict[str, object]] = []
    development_features: list[float] = []
    for index in range(development_trials):
        correlation = correlations[index % len(correlations)]
        seed = development_seed + index
        _, _, feature, score, raw_score, cycle_count = _source_case(
            seed=seed,
            correlation=correlation,
            representative_radius=representative_radius,
            minimum_uncertainty_scale=minimum_uncertainty_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            window_size=window_size,
            overlap=overlap,
        )
        development_features.append(feature)
        development_records.append(
            {
                "trial_index": index,
                "seed": seed,
                "simulation_correlation": correlation,
                "cycle_count": cycle_count,
                "source_uncertainty_feature": feature,
                "maximum_normalized_cycle_score": score,
                "maximum_raw_cycle_displacement": raw_score,
                "used_for": "observable_stratum_boundaries_only",
            }
        )

    calibration_records: list[dict[str, object]] = []
    calibration_features: list[float] = []
    calibration_scores: list[float] = []
    raw_scores: list[float] = []
    for index in range(calibration_trials):
        correlation = correlations[index % len(correlations)]
        seed = calibration_seed + index
        _, _, feature, score, raw_score, cycle_count = _source_case(
            seed=seed,
            correlation=correlation,
            representative_radius=representative_radius,
            minimum_uncertainty_scale=minimum_uncertainty_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            window_size=window_size,
            overlap=overlap,
        )
        calibration_features.append(feature)
        calibration_scores.append(score)
        raw_scores.append(raw_score)
        calibration_records.append(
            {
                "trial_index": index,
                "seed": seed,
                "simulation_correlation": correlation,
                "cycle_count": cycle_count,
                "source_uncertainty_feature": feature,
                "maximum_normalized_cycle_score": score,
                "maximum_raw_cycle_displacement": raw_score,
            }
        )

    calibration = fit_group_robust_cycle_calibration(
        development_features,
        calibration_features,
        calibration_scores,
        alpha=conformal_alpha,
        minimum_calibration_per_stratum=minimum_calibration_per_stratum,
    )
    for record in calibration_records:
        record["observable_source_stratum"] = calibration.strata.assign(
            float(record["source_uncertainty_feature"])
        )

    empirical_raw_threshold = max(
        float(
            np.quantile(
                np.asarray(raw_scores, dtype=np.float64),
                empirical_quantile,
                method="higher",
            )
        ),
        float(np.nextafter(0.0, 1.0)),
    )
    empirical_normalized_threshold = max(
        float(
            np.quantile(
                np.asarray(calibration_scores, dtype=np.float64),
                empirical_quantile,
                method="higher",
            )
        ),
        float(np.nextafter(0.0, 1.0)),
    )

    sensitivity: list[dict[str, object]] = []
    for sample_count in (48, 96, calibration_trials):
        if sample_count > calibration_trials:
            continue
        try:
            fitted = fit_group_robust_cycle_calibration(
                development_features,
                calibration_features[:sample_count],
                calibration_scores[:sample_count],
                alpha=conformal_alpha,
                minimum_calibration_per_stratum=minimum_calibration_per_stratum,
            )
        except ValueError as error:
            sensitivity.append(
                {
                    "calibration_sample_count": sample_count,
                    "status": "insufficient",
                    "reason": str(error),
                }
            )
        else:
            sensitivity.append(
                {
                    "calibration_sample_count": sample_count,
                    "status": "finite",
                    "pooled_threshold": fitted.pooled_threshold.threshold,
                    "stratum_thresholds": dict(fitted.threshold_by_stratum),
                    "worst_group_threshold": fitted.worst_group_threshold,
                    "stratum_sample_counts": {
                        name: item.sample_count
                        for name, item in fitted.stratum_thresholds
                    },
                }
            )

    return (
        calibration,
        empirical_raw_threshold,
        empirical_normalized_threshold,
        development_records,
        calibration_records,
        sensitivity,
    )


def _renamed_output(
    output: _EstimatorOutput,
    method_id: str,
) -> _EstimatorOutput:
    return replace(output, method_id=method_id)


def _guard_output(
    problem: Any,
    alignments: Sequence[Any],
    *,
    method_id: str,
    threshold: float,
    representative_radius: float,
    minimum_uncertainty_scale: float,
    minimum_edge_weight: float,
) -> _EstimatorOutput:
    output, _ = _normalized_guard_output(
        problem,
        alignments,
        threshold=threshold,
        representative_radius=representative_radius,
        minimum_uncertainty_scale=minimum_uncertainty_scale,
        minimum_edge_weight=minimum_edge_weight,
    )
    return _renamed_output(output, method_id)


def _aggregate_records(
    records: Sequence[Mapping[str, object]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    tree_by_trial: dict[tuple[str, int], Mapping[str, object]] = {}
    for record in records:
        scenario_id = str(record["scenario_id"])
        method_id = str(record["method_id"])
        trial_index = int(record["trial_index"])
        grouped[(scenario_id, method_id)].append(record)
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
    scenario_ids = sorted({scenario_id for scenario_id, _ in grouped})
    for scenario_id in scenario_ids:
        for method_id in _METHOD_ORDER:
            group = grouped[(scenario_id, method_id)]
            if not group:
                raise ValueError(
                    f"scenario {scenario_id!r} lacks method {method_id!r}"
                )
            metrics = {
                name: _bootstrap_summary(
                    [float(record[name]) for record in group],
                    resamples=bootstrap_resamples,
                    seed=_stable_seed(
                        bootstrap_seed,
                        scenario_id,
                        method_id,
                        name,
                    ),
                )
                for name in metric_names
            }
            paired = [
                float(record["endpoint_displacement_rmse"])
                - float(
                    tree_by_trial[(scenario_id, int(record["trial_index"]))][
                        "endpoint_displacement_rmse"
                    ]
                )
                for record in group
            ]
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
                            "paired_endpoint_delta",
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
    method_id: str,
) -> dict[str, object]:
    method_records = [
        record for record in records if str(record["method_id"]) == method_id
    ]
    clean_rates: dict[str, float] = {}
    for scenario_id in _CLEAN_SCENARIOS:
        values = [
            bool(record["fallback_applied"])
            for record in method_records
            if str(record["scenario_id"]) == scenario_id
        ]
        if not values:
            raise ValueError(f"guard {method_id!r} lacks clean scenario records")
        clean_rates[scenario_id] = float(np.mean(values))
    stratum_rates: dict[str, float] = {}
    stratum_ids = sorted(
        {str(record["observable_source_stratum"]) for record in method_records}
    )
    for stratum_id in stratum_ids:
        values = [
            bool(record["fallback_applied"])
            for record in method_records
            if str(record["observable_source_stratum"]) == stratum_id
        ]
        if not values:
            raise ValueError(f"guard {method_id!r} lacks source-stratum records")
        stratum_rates[stratum_id] = float(np.mean(values))
    strong = [
        bool(record["fallback_applied"])
        for record in method_records
        if str(record["scenario_id"]) in _STRONG_OUTLIER_SCENARIOS
        and bool(record["outlier_injected"])
    ]
    mild = [
        bool(record["fallback_applied"])
        for record in method_records
        if str(record["scenario_id"]) == _MILD_OUTLIER_SCENARIO
        and bool(record["outlier_injected"])
    ]
    all_injected = [
        bool(record["fallback_applied"])
        for record in method_records
        if bool(record["outlier_injected"])
    ]
    if not strong or not mild or not all_injected:
        raise ValueError("registered outlier scenarios produced no injected trials")
    fallback_records = [
        record
        for record in method_records
        if record["fallback_applied"] is True
    ]
    exact = all(
        record["exact_tree_fallback_verified"] is True
        for record in fallback_records
    )
    return {
        "clean_false_fallback_rates": clean_rates,
        "fallback_rates_by_observable_source_stratum": stratum_rates,
        "worst_clean_false_fallback_rate": max(clean_rates.values()),
        "strong_outlier_detection_rate": float(np.mean(strong)),
        "mild_outlier_detection_rate": float(np.mean(mild)),
        "all_outlier_detection_rate": float(np.mean(all_injected)),
        "strong_injected_count": len(strong),
        "mild_injected_count": len(mild),
        "all_injected_count": len(all_injected),
        "fallback_count": len(fallback_records),
        "exact_tree_fallback_verified": exact,
    }


def _decision(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summaries = {
        method_id: _guard_summary(records, method_id)
        for method_id in _GUARD_METHODS
    }
    primary = summaries[_PRIMARY_METHOD]
    strong = float(primary["strong_outlier_detection_rate"])
    mild = float(primary["mild_outlier_detection_rate"])
    worst_clean = float(primary["worst_clean_false_fallback_rate"])
    criteria = {
        "strong_detection_at_least_0_95": strong >= 0.95,
        "strong_detection_noninferior_to_previous_by_0_05": (
            strong >= _PREVIOUS_NORMALIZED_STRONG_DETECTION - 0.05
        ),
        "mild_detection_at_least_0_90": mild >= 0.90,
        "mild_detection_noninferior_to_previous_by_0_05": (
            mild >= _PREVIOUS_NORMALIZED_MILD_DETECTION - 0.05
        ),
        "worst_clean_false_fallback_at_most_0_10": worst_clean <= 0.10,
        "exact_tree_fallback_verified": bool(
            primary["exact_tree_fallback_verified"]
        ),
    }
    return {
        "primary_method": _PRIMARY_METHOD,
        "previous_normalized_guard_reference": {
            "strong_outlier_detection_rate": (
                _PREVIOUS_NORMALIZED_STRONG_DETECTION
            ),
            "mild_outlier_detection_rate": _PREVIOUS_NORMALIZED_MILD_DETECTION,
            "source_result": "prob4d-cycle-guard-normalization-v1",
        },
        "guard_summaries": summaries,
        "criteria": criteria,
        "overall_passed": all(criteria.values()),
        "decision_semantics": (
            "The maximum observable-source-stratum conformal threshold is "
            "primary and was fixed before target evaluation. Every registered "
            "detection, worst-clean, and exact-fallback criterion must pass."
        ),
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["scenario_id", "method_id", "trial_index", "seed"]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flat_aggregate(
    aggregate: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in aggregate:
        metrics = item["metrics"]
        paired = item["paired_endpoint_delta_vs_tree"]
        if not isinstance(metrics, Mapping) or not isinstance(paired, Mapping):
            raise TypeError("aggregate metric payload changed")
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
        for name, summary in metrics.items():
            if not isinstance(summary, Mapping):
                raise TypeError("aggregate metric summary changed")
            row[f"{name}_mean"] = summary["mean"]
            row[f"{name}_ci95_lower"] = summary["ci95_lower"]
            row[f"{name}_ci95_upper"] = summary["ci95_upper"]
        rows.append(row)
    return rows


def _format_interval(summary: Mapping[str, object]) -> str:
    return (
        f"{float(summary['mean']):.6g} "
        f"[{float(summary['ci95_lower']):.6g}, "
        f"{float(summary['ci95_upper']):.6g}]"
    )


def _write_markdown(
    path: Path,
    *,
    calibration: GroupRobustCycleCalibration,
    empirical_raw_threshold: float,
    empirical_normalized_threshold: float,
    aggregate: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
) -> None:
    lines = [
        "# Finite-sample group-robust cycle-guard study",
        "",
        "This is controlled synthetic source-admission evidence, not a physical-twin claim.",
        "",
        "## Frozen thresholds",
        "",
        f"- raw empirical control: `{empirical_raw_threshold:.9g}`",
        f"- pooled empirical normalized control: `{empirical_normalized_threshold:.9g}`",
        f"- pooled conformal normalized control: `{calibration.pooled_threshold.threshold:.9g}`",
        (
            "- observable source strata: "
            + ", ".join(
                f"`{name}`={threshold.threshold:.9g} (n={threshold.sample_count})"
                for name, threshold in calibration.stratum_thresholds
            )
        ),
        f"- primary worst-group threshold: `{calibration.worst_group_threshold:.9g}`",
        "",
        "## Aggregate results",
        "",
        "| Scenario | Method | Endpoint | Delta vs tree | Coverage | NEES | Fallback | Detection |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregate:
        metrics = item["metrics"]
        paired = item["paired_endpoint_delta_vs_tree"]
        if not isinstance(metrics, Mapping) or not isinstance(paired, Mapping):
            raise TypeError("aggregate payload changed")

        def optional(value: object) -> str:
            return "—" if value is None else f"{float(value):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["scenario_id"]),
                    str(item["method_id"]),
                    _format_interval(metrics["endpoint_displacement_rmse"]),
                    _format_interval(paired),
                    f"{float(metrics['coverage_95']['mean']):.3f}",
                    f"{float(metrics['mean_normalized_nees']['mean']):.3f}",
                    optional(item["fallback_rate"]),
                    optional(item["outlier_detection_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Preregistered primary decision",
            "",
            f"Overall result: **{'PASS' if decision['overall_passed'] else 'FAIL'}**.",
            "",
            "```json",
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "## Claim boundary",
            "",
            _CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_group_robust_cycle_guard_monte_carlo(
    output_directory: str | Path,
    *,
    scenarios: Sequence[
        GaugeGraphStudyScenario
    ] = DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS,
    development_trials: int = 96,
    calibration_trials: int = 192,
    target_trials_per_scenario: int = 128,
    development_seed: int = 380_260_804,
    calibration_seed: int = 480_260_804,
    target_seed: int = 571_260_804,
    conformal_alpha: float = 0.05,
    minimum_calibration_per_stratum: int = 20,
    empirical_quantile: float = 0.95,
    representative_radius: float = 1.0,
    minimum_uncertainty_scale: float = 1e-12,
    num_frames: int = 28,
    height: int = 4,
    width: int = 6,
    window_size: int = 11,
    overlap: int = 7,
    minimum_edge_weight: float = 0.0,
    bootstrap_resamples: int = 2_000,
    source_revision: str | None = None,
) -> dict[str, object]:
    """Run the development/calibration/target-separated registered study."""

    development_trials = _validated_positive_integer(
        development_trials,
        name="development_trials",
    )
    calibration_trials = _validated_positive_integer(
        calibration_trials,
        name="calibration_trials",
    )
    target_trials_per_scenario = _validated_positive_integer(
        target_trials_per_scenario,
        name="target_trials_per_scenario",
    )
    bootstrap_resamples = _validated_positive_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
    )
    normalized_scenarios = tuple(scenarios)
    if not normalized_scenarios:
        raise ValueError("at least one registered scenario is required")
    if len({scenario.scenario_id for scenario in normalized_scenarios}) != len(
        normalized_scenarios
    ):
        raise ValueError("registered scenario IDs must be unique")
    numeric = np.asarray(
        [
            conformal_alpha,
            empirical_quantile,
            representative_radius,
            minimum_uncertainty_scale,
            minimum_edge_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("group-robust study parameters must be finite")
    if not 0.0 < conformal_alpha < 1.0:
        raise ValueError("conformal_alpha must lie in (0, 1)")
    if not 0.0 < empirical_quantile <= 1.0:
        raise ValueError("empirical_quantile must lie in (0, 1]")
    if representative_radius <= 0.0 or minimum_uncertainty_scale <= 0.0:
        raise ValueError("radius and uncertainty floor must be positive")
    if minimum_edge_weight < 0.0:
        raise ValueError("minimum_edge_weight must be nonnegative")

    (
        calibration,
        empirical_raw_threshold,
        empirical_normalized_threshold,
        development_records,
        calibration_records,
        sensitivity,
    ) = _prepare_source_splits(
        normalized_scenarios,
        development_trials=development_trials,
        calibration_trials=calibration_trials,
        development_seed=int(development_seed),
        calibration_seed=int(calibration_seed),
        conformal_alpha=float(conformal_alpha),
        minimum_calibration_per_stratum=minimum_calibration_per_stratum,
        empirical_quantile=float(empirical_quantile),
        representative_radius=float(representative_radius),
        minimum_uncertainty_scale=float(minimum_uncertainty_scale),
        num_frames=num_frames,
        height=height,
        width=width,
        window_size=window_size,
        overlap=overlap,
    )

    records: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(normalized_scenarios):
        for trial_index in range(target_trials_per_scenario):
            seed = int(target_seed) + scenario_index * 100_000 + trial_index
            (
                problem,
                clean_alignments,
                clean_feature,
                _,
                _,
                _,
            ) = _source_case(
                seed=seed,
                correlation=scenario.correlation,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                num_frames=num_frames,
                height=height,
                width=width,
                window_size=window_size,
                overlap=overlap,
            )
            alignments, outlier_injected, outlier_edge_id = (
                _inject_inconsistent_skip_edge(
                    clean_alignments,
                    [window.window_id for window in problem.overlap_windows],
                    generator=np.random.default_rng(
                        _stable_seed(
                            target_seed,
                            scenario.scenario_id,
                            trial_index,
                            "group-robust-outlier",
                        )
                    ),
                    probability=scenario.outlier_probability,
                    translation_magnitude=scenario.outlier_translation,
                )
            )
            normalized_audit = audit_uncertainty_normalized_alignment_cycles(
                alignments,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
            )
            feature = cycle_source_uncertainty_feature(normalized_audit)
            feature_shift_from_injection = feature - clean_feature
            raw_audit = _run_raw_audit(
                alignments,
                representative_radius=representative_radius,
            )
            stratum_id = calibration.strata.assign(feature)
            stratum_threshold = calibration.threshold_for_feature(feature)

            base_outputs, _ = _run_estimators(
                problem,
                alignments,
                cycle_threshold=empirical_raw_threshold,
                representative_radius=representative_radius,
                minimum_edge_weight=minimum_edge_weight,
            )
            tree = base_outputs["tree"]
            full_graph = base_outputs["full_joint_graph"]
            raw_guard = _renamed_output(
                base_outputs["guarded_graph"],
                "raw_guarded_graph",
            )
            pooled_empirical = _guard_output(
                problem,
                alignments,
                method_id="pooled_empirical_normalized_guard",
                threshold=empirical_normalized_threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            pooled_conformal = _guard_output(
                problem,
                alignments,
                method_id="pooled_conformal_normalized_guard",
                threshold=calibration.pooled_threshold.threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            stratum_conformal = _guard_output(
                problem,
                alignments,
                method_id="stratum_conformal_normalized_guard",
                threshold=stratum_threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            worst_group_conformal = _guard_output(
                problem,
                alignments,
                method_id="worst_group_conformal_normalized_guard",
                threshold=calibration.worst_group_threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            methods = {
                "tree": tree,
                "full_joint_graph": full_graph,
                "raw_guarded_graph": raw_guard,
                "pooled_empirical_normalized_guard": pooled_empirical,
                "pooled_conformal_normalized_guard": pooled_conformal,
                "stratum_conformal_normalized_guard": stratum_conformal,
                "worst_group_conformal_normalized_guard": worst_group_conformal,
            }
            applied_thresholds = {
                "tree": None,
                "full_joint_graph": None,
                "raw_guarded_graph": empirical_raw_threshold,
                "pooled_empirical_normalized_guard": (
                    empirical_normalized_threshold
                ),
                "pooled_conformal_normalized_guard": (
                    calibration.pooled_threshold.threshold
                ),
                "stratum_conformal_normalized_guard": stratum_threshold,
                "worst_group_conformal_normalized_guard": (
                    calibration.worst_group_threshold
                ),
            }
            for method_id in _GUARD_METHODS:
                _assert_exact_fallback(methods[method_id], tree)
            for method_id in _METHOD_ORDER:
                output = methods[method_id]
                metrics = _evaluate_output(
                    output,
                    problem,
                    representative_radius=representative_radius,
                )
                records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "simulation_correlation": scenario.correlation,
                        "outlier_probability": scenario.outlier_probability,
                        "outlier_translation": scenario.outlier_translation,
                        "trial_index": trial_index,
                        "seed": seed,
                        "outlier_injected": outlier_injected,
                        "outlier_edge_id": outlier_edge_id,
                        "source_uncertainty_feature": feature,
                        "source_uncertainty_feature_without_injection": (
                            clean_feature
                        ),
                        "source_uncertainty_feature_shift_from_injection": (
                            feature_shift_from_injection
                        ),
                        "observable_source_stratum": stratum_id,
                        "maximum_raw_cycle_displacement": float(
                            raw_audit["maximum"]
                        ),
                        "maximum_normalized_cycle_score": (
                            normalized_audit.maximum_observed_normalized_score
                        ),
                        "applied_guard_threshold": applied_thresholds[method_id],
                        "exact_tree_fallback_verified": (
                            None
                            if output.fallback_applied is not True
                            else True
                        ),
                        **metrics,
                    }
                )

    aggregate = _aggregate_records(
        records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=int(target_seed) + 84_271,
    )
    decision = _decision(records)
    source_revision_value = _validated_source_revision(source_revision)
    report_body: dict[str, object] = {
        "schema_name": GROUP_ROBUST_CYCLE_GUARD_SCHEMA,
        "schema_version": GROUP_ROBUST_CYCLE_GUARD_VERSION,
        "source_revision": source_revision_value,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "development_trials": development_trials,
            "calibration_trials": calibration_trials,
            "target_trials_per_scenario": target_trials_per_scenario,
            "development_seed": int(development_seed),
            "calibration_seed": int(calibration_seed),
            "target_seed": int(target_seed),
            "conformal_alpha": float(conformal_alpha),
            "minimum_calibration_per_stratum": int(
                minimum_calibration_per_stratum
            ),
            "empirical_quantile": float(empirical_quantile),
            "representative_radius": float(representative_radius),
            "minimum_uncertainty_scale": float(minimum_uncertainty_scale),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "window_size": window_size,
            "overlap": overlap,
            "minimum_edge_weight": float(minimum_edge_weight),
            "bootstrap_resamples": bootstrap_resamples,
            "scenarios": [
                scenario.to_dict() for scenario in normalized_scenarios
            ],
            "method_order": list(_METHOD_ORDER),
            "primary_method": _PRIMARY_METHOD,
        },
        "development": {
            "records": development_records,
            "uses_hidden_correlation_in_method": False,
            "simulation_condition_recorded_for_audit_only": True,
        },
        "calibration": {
            "group_robust": calibration.to_dict(),
            "empirical_raw_threshold": empirical_raw_threshold,
            "empirical_normalized_threshold": empirical_normalized_threshold,
            "records": calibration_records,
            "sample_size_sensitivity": sensitivity,
        },
        "trials": records,
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    report_id = hashlib.sha256(_canonical_json(report_body)).hexdigest()
    report = {"report_id": report_id, **report_body}
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cycle_guard_group_robust.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "cycle_guard_group_robust_trials.csv", records)
    _write_csv(
        output / "cycle_guard_group_robust.csv",
        _flat_aggregate(aggregate),
    )
    _write_markdown(
        output / "cycle_guard_group_robust.md",
        calibration=calibration,
        empirical_raw_threshold=empirical_raw_threshold,
        empirical_normalized_threshold=empirical_normalized_threshold,
        aggregate=aggregate,
        decision=decision,
    )
    evidence_files = sorted(output.glob("cycle_guard_group_robust*"))
    (output / "SHA256SUMS").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in evidence_files
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-trials", type=int, default=96)
    parser.add_argument("--calibration-trials", type=int, default=192)
    parser.add_argument("--target-trials-per-scenario", type=int, default=128)
    parser.add_argument("--development-seed", type=int, default=380_260_804)
    parser.add_argument("--calibration-seed", type=int, default=480_260_804)
    parser.add_argument("--target-seed", type=int, default=571_260_804)
    parser.add_argument("--conformal-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-calibration-per-stratum", type=int, default=20)
    parser.add_argument("--empirical-quantile", type=float, default=0.95)
    parser.add_argument("--representative-radius", type=float, default=1.0)
    parser.add_argument("--minimum-uncertainty-scale", type=float, default=1e-12)
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=11)
    parser.add_argument("--overlap", type=int, default=7)
    parser.add_argument("--minimum-edge-weight", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--source-revision")
    arguments = parser.parse_args(argv)
    report = run_group_robust_cycle_guard_monte_carlo(
        arguments.output_dir,
        development_trials=arguments.development_trials,
        calibration_trials=arguments.calibration_trials,
        target_trials_per_scenario=arguments.target_trials_per_scenario,
        development_seed=arguments.development_seed,
        calibration_seed=arguments.calibration_seed,
        target_seed=arguments.target_seed,
        conformal_alpha=arguments.conformal_alpha,
        minimum_calibration_per_stratum=(
            arguments.minimum_calibration_per_stratum
        ),
        empirical_quantile=arguments.empirical_quantile,
        representative_radius=arguments.representative_radius,
        minimum_uncertainty_scale=arguments.minimum_uncertainty_scale,
        num_frames=arguments.num_frames,
        height=arguments.height,
        width=arguments.width,
        window_size=arguments.window_size,
        overlap=arguments.overlap,
        minimum_edge_weight=arguments.minimum_edge_weight,
        bootstrap_resamples=arguments.bootstrap_resamples,
        source_revision=arguments.source_revision,
    )
    print(arguments.output_dir / "cycle_guard_group_robust.json")
    return 0 if report["decision"]["overall_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GROUP_ROBUST_CYCLE_GUARD_SCHEMA",
    "GROUP_ROBUST_CYCLE_GUARD_VERSION",
    "run_group_robust_cycle_guard_monte_carlo",
]
