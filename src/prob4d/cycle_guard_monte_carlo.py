"""Compare raw and uncertainty-normalized source cycle guards."""

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
    _joint_output,
    _make_problem,
    _run_estimators,
    _stable_seed,
    _validated_positive_integer,
    _validated_source_revision,
)
from .observation_export import _build_alignments
from .uncertainty_guarded_causal_gauge_graph import (
    UncertaintyGuardedCausalGaugeGraphReport,
    estimate_uncertainty_guarded_causal_multi_edge_gauge_graph,
)
from .uncertainty_normalized_cycles import (
    audit_uncertainty_normalized_alignment_cycles,
)

CYCLE_GUARD_MONTE_CARLO_SCHEMA: Final = "prob4d.cycle-guard-monte-carlo"
CYCLE_GUARD_MONTE_CARLO_VERSION: Final = 1
_METHOD_ORDER: Final = (
    "tree",
    "full_joint_graph",
    "raw_guarded_graph",
    "uncertainty_guarded_graph",
)
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


def _calibrate_thresholds(
    scenarios: Sequence[GaugeGraphStudyScenario],
    *,
    calibration_trials: int,
    calibration_seed: int,
    threshold_quantile: float,
    representative_radius: float,
    minimum_uncertainty_scale: float,
    num_frames: int,
    height: int,
    width: int,
    window_size: int,
    overlap: int,
) -> tuple[float, float, list[dict[str, object]]]:
    clean_correlations = tuple(
        sorted(
            {
                scenario.correlation
                for scenario in scenarios
                if scenario.outlier_probability == 0.0
            }
        )
    )
    if not clean_correlations:
        raise ValueError("cycle-guard calibration requires clean registered scenarios")
    if calibration_trials < len(clean_correlations):
        raise ValueError(
            "calibration_trials must cover every clean correlation at least once"
        )
    raw_maxima: list[float] = []
    normalized_maxima: list[float] = []
    records: list[dict[str, object]] = []
    for index in range(calibration_trials):
        correlation = clean_correlations[index % len(clean_correlations)]
        seed = calibration_seed + index
        problem = _make_problem(
            seed=seed,
            correlation=correlation,
            num_frames=num_frames,
            height=height,
            width=width,
            window_size=window_size,
            overlap=overlap,
        )
        alignments = _build_alignments(problem.overlap_windows)
        raw_audit = _run_raw_audit(
            alignments,
            representative_radius=representative_radius,
        )
        normalized_audit = audit_uncertainty_normalized_alignment_cycles(
            alignments,
            representative_radius=representative_radius,
            minimum_uncertainty_scale=minimum_uncertainty_scale,
        )
        if raw_audit["cycle_count"] < 1 or normalized_audit.cycle_count < 1:
            raise ValueError("registered calibration geometry produced no cycles")
        raw_maximum = float(raw_audit["maximum"])
        normalized_maximum = normalized_audit.maximum_observed_normalized_score
        raw_maxima.append(raw_maximum)
        normalized_maxima.append(normalized_maximum)
        records.append(
            {
                "trial_index": index,
                "seed": seed,
                "correlation": correlation,
                "cycle_count": normalized_audit.cycle_count,
                "maximum_raw_displacement": raw_maximum,
                "maximum_uncertainty_normalized_score": normalized_maximum,
            }
        )
    raw_threshold = float(
        np.quantile(
            np.asarray(raw_maxima, dtype=np.float64),
            threshold_quantile,
            method="higher",
        )
    )
    normalized_threshold = float(
        np.quantile(
            np.asarray(normalized_maxima, dtype=np.float64),
            threshold_quantile,
            method="higher",
        )
    )
    return (
        max(raw_threshold, float(np.nextafter(0.0, 1.0))),
        max(normalized_threshold, float(np.nextafter(0.0, 1.0))),
        records,
    )


def _run_raw_audit(
    alignments: Sequence[Any],
    *,
    representative_radius: float,
) -> dict[str, float | int]:
    from .alignment_cycles import audit_alignment_cycles

    audit = audit_alignment_cycles(
        tuple(alignments),
        representative_radius=representative_radius,
    )
    return {
        "cycle_count": audit.cycle_count,
        "maximum": audit.maximum_observed_representative_displacement,
    }


def _normalized_guard_output(
    problem: Any,
    alignments: Sequence[Any],
    *,
    threshold: float,
    representative_radius: float,
    minimum_uncertainty_scale: float,
    minimum_edge_weight: float,
) -> tuple[_EstimatorOutput, UncertaintyGuardedCausalGaugeGraphReport]:
    window_ids = [window.window_id for window in problem.overlap_windows]
    first_id = window_ids[0]
    initial_transform = problem.true_overlap_gauges[first_id]
    anchor_standard_deviation = np.asarray(
        [1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3],
        dtype=np.float64,
    )
    posterior, report = (
        estimate_uncertainty_guarded_causal_multi_edge_gauge_graph(
            problem.overlap_windows,
            alignments,
            initial_transform=initial_transform,
            initial_covariance=np.diag(anchor_standard_deviation**2),
            maximum_normalized_cycle_score=threshold,
            representative_radius=representative_radius,
            minimum_uncertainty_scale=minimum_uncertainty_scale,
            minimum_cycles_per_multi_edge_child=1,
            minimum_edge_weight=minimum_edge_weight,
        )
    )
    return (
        _joint_output(
            "uncertainty_guarded_graph",
            posterior,
            fallback_applied=report.fallback_applied,
            graph_report=report.graph_report,
        ),
        report,
    )


def _assert_exact_fallback(
    candidate: _EstimatorOutput,
    tree: _EstimatorOutput,
) -> None:
    if candidate.fallback_applied is not True:
        return
    for window_id in tree.estimates:
        np.testing.assert_array_equal(
            candidate.estimates[window_id].as_vector(),
            tree.estimates[window_id].as_vector(),
        )
        np.testing.assert_array_equal(
            candidate.covariances[window_id],
            tree.covariances[window_id],
        )


def _aggregate_records(
    records: Sequence[Mapping[str, object]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    by_scenario_method: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    tree_by_trial: dict[tuple[str, int], Mapping[str, object]] = {}
    for record in records:
        scenario_id = str(record["scenario_id"])
        method_id = str(record["method_id"])
        trial_index = int(record["trial_index"])
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
                    [float(record[metric]) for record in group],
                    resamples=bootstrap_resamples,
                    seed=_stable_seed(bootstrap_seed, scenario_id, method_id, metric),
                )
                for metric in metric_names
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
            paired_summary = _bootstrap_summary(
                paired,
                resamples=bootstrap_resamples,
                seed=_stable_seed(bootstrap_seed, scenario_id, method_id, "paired"),
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
                    "paired_endpoint_delta_vs_tree": paired_summary,
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


def _guard_decision(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    guard_methods = ("raw_guarded_graph", "uncertainty_guarded_graph")
    summaries: dict[str, dict[str, object]] = {}
    for method_id in guard_methods:
        method_records = [record for record in records if record["method_id"] == method_id]
        clean_rates: dict[str, float] = {}
        for scenario_id in _CLEAN_SCENARIOS:
            values = [
                bool(record["fallback_applied"])
                for record in method_records
                if record["scenario_id"] == scenario_id
            ]
            clean_rates[scenario_id] = float(np.mean(values))
        strong_values = [
            bool(record["fallback_applied"])
            for record in method_records
            if record["scenario_id"] in _STRONG_OUTLIER_SCENARIOS
            and bool(record["outlier_injected"])
        ]
        mild_values = [
            bool(record["fallback_applied"])
            for record in method_records
            if record["scenario_id"] == _MILD_OUTLIER_SCENARIO
            and bool(record["outlier_injected"])
        ]
        total_injected = [
            bool(record["fallback_applied"])
            for record in method_records
            if bool(record["outlier_injected"])
        ]
        summaries[method_id] = {
            "clean_false_fallback_rates": clean_rates,
            "worst_clean_false_fallback_rate": max(clean_rates.values()),
            "strong_outlier_detection_rate": float(np.mean(strong_values)),
            "mild_outlier_detection_rate": float(np.mean(mild_values)),
            "all_outlier_detection_rate": float(np.mean(total_injected)),
            "strong_injected_count": len(strong_values),
            "mild_injected_count": len(mild_values),
            "all_injected_count": len(total_injected),
        }

    raw = summaries["raw_guarded_graph"]
    normalized = summaries["uncertainty_guarded_graph"]
    raw_worst = float(raw["worst_clean_false_fallback_rate"])
    normalized_worst = float(normalized["worst_clean_false_fallback_rate"])
    raw_strong = float(raw["strong_outlier_detection_rate"])
    normalized_strong = float(normalized["strong_outlier_detection_rate"])
    raw_mild = float(raw["mild_outlier_detection_rate"])
    normalized_mild = float(normalized["mild_outlier_detection_rate"])
    criteria = {
        "strong_detection_at_least_0_95": normalized_strong >= 0.95,
        "strong_detection_noninferior_to_raw_by_0_05": (
            normalized_strong >= raw_strong - 0.05
        ),
        "mild_detection_at_least_0_90": normalized_mild >= 0.90,
        "mild_detection_noninferior_to_raw_by_0_05": (
            normalized_mild >= raw_mild - 0.05
        ),
        "worst_clean_false_fallback_at_most_0_10": normalized_worst <= 0.10,
        "worst_clean_false_fallback_halved": (
            normalized_worst <= 0.5 * raw_worst
        ),
    }
    return {
        "registered_primary_endpoints": [
            "strong_outlier_detection_rate",
            "mild_outlier_detection_rate",
            "worst_clean_false_fallback_rate",
        ],
        "raw_guard": raw,
        "uncertainty_normalized_guard": normalized,
        "criteria": criteria,
        "overall_passed": all(criteria.values()),
        "decision_semantics": (
            "Every preregistered detection and worst-clean false-fallback criterion "
            "must pass. No criterion uses target truth to choose a threshold after "
            "target evaluation."
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
        f"{float(summary['mean']):.5g} "
        f"[{float(summary['ci95_lower']):.5g}, "
        f"{float(summary['ci95_upper']):.5g}]"
    )


def _write_markdown(
    path: Path,
    *,
    raw_threshold: float,
    normalized_threshold: float,
    aggregate: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
) -> None:
    lines = [
        "# Uncertainty-normalized cycle-guard study",
        "",
        "This is a controlled synthetic source-guard study, not a physical-twin claim.",
        "",
        f"Raw cycle threshold: `{raw_threshold:.8g}`. Uncertainty-normalized "
        f"threshold: `{normalized_threshold:.8g}`.",
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
                    optional(item["clean_false_fallback_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Preregistered guard decision",
            "",
            f"Overall result: **{'PASS' if decision['overall_passed'] else 'FAIL'}**.",
            "",
            "```json",
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "## Claim boundary",
            "",
            "The result concerns synthetic source-cycle admission only. It does not "
            "establish held-out physical-object provider competence, BayesianPhysTwin "
            "benefit, harmful-update control, or Causal4D intervention benefit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_cycle_guard_monte_carlo(
    output_directory: str | Path,
    *,
    scenarios: Sequence[GaugeGraphStudyScenario] = DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS,
    calibration_trials: int = 48,
    target_trials_per_scenario: int = 128,
    calibration_seed: int = 80_260_804,
    target_seed: int = 271_260_804,
    threshold_quantile: float = 0.95,
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
    """Calibrate raw and normalized guards, then evaluate a frozen target split."""

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
        raise ValueError("at least one cycle-guard scenario is required")
    if len({scenario.scenario_id for scenario in normalized_scenarios}) != len(
        normalized_scenarios
    ):
        raise ValueError("cycle-guard scenario IDs must be unique")
    threshold_quantile = float(threshold_quantile)
    representative_radius = float(representative_radius)
    minimum_uncertainty_scale = float(minimum_uncertainty_scale)
    minimum_edge_weight = float(minimum_edge_weight)
    if not np.isfinite(threshold_quantile) or not 0.0 < threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must lie in (0, 1]")
    if not np.isfinite(representative_radius) or representative_radius <= 0.0:
        raise ValueError("representative_radius must be finite and positive")
    if (
        not np.isfinite(minimum_uncertainty_scale)
        or minimum_uncertainty_scale <= 0.0
    ):
        raise ValueError("minimum_uncertainty_scale must be finite and positive")
    if not np.isfinite(minimum_edge_weight) or minimum_edge_weight < 0.0:
        raise ValueError("minimum_edge_weight must be finite and nonnegative")

    raw_threshold, normalized_threshold, calibration_records = (
        _calibrate_thresholds(
            normalized_scenarios,
            calibration_trials=calibration_trials,
            calibration_seed=int(calibration_seed),
            threshold_quantile=threshold_quantile,
            representative_radius=representative_radius,
            minimum_uncertainty_scale=minimum_uncertainty_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            window_size=window_size,
            overlap=overlap,
        )
    )

    records: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(normalized_scenarios):
        for trial_index in range(target_trials_per_scenario):
            seed = int(target_seed) + scenario_index * 100_000 + trial_index
            problem = _make_problem(
                seed=seed,
                correlation=scenario.correlation,
                num_frames=num_frames,
                height=height,
                width=width,
                window_size=window_size,
                overlap=overlap,
            )
            alignments = _build_alignments(problem.overlap_windows)
            alignments, outlier_injected, outlier_edge_id = (
                _inject_inconsistent_skip_edge(
                    alignments,
                    [window.window_id for window in problem.overlap_windows],
                    generator=np.random.default_rng(
                        _stable_seed(
                            target_seed,
                            scenario.scenario_id,
                            trial_index,
                            "cycle-guard-outlier",
                        )
                    ),
                    probability=scenario.outlier_probability,
                    translation_magnitude=scenario.outlier_translation,
                )
            )
            raw_outputs, raw_report = _run_estimators(
                problem,
                alignments,
                cycle_threshold=raw_threshold,
                representative_radius=representative_radius,
                minimum_edge_weight=minimum_edge_weight,
            )
            normalized_output, normalized_report = _normalized_guard_output(
                problem,
                alignments,
                threshold=normalized_threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            tree = raw_outputs["tree"]
            full_graph = raw_outputs["full_joint_graph"]
            raw_guard = replace(
                raw_outputs["guarded_graph"],
                method_id="raw_guarded_graph",
            )
            _assert_exact_fallback(raw_guard, tree)
            _assert_exact_fallback(normalized_output, tree)
            methods = {
                "tree": tree,
                "full_joint_graph": full_graph,
                "raw_guarded_graph": raw_guard,
                "uncertainty_guarded_graph": normalized_output,
            }
            raw_maximum = (
                raw_report.cycle_audit.maximum_observed_representative_displacement
            )
            normalized_maximum = (
                normalized_report.cycle_audit.maximum_observed_normalized_score
            )
            for method_id in _METHOD_ORDER:
                metrics = _evaluate_output(
                    methods[method_id],
                    problem,
                    representative_radius=representative_radius,
                )
                records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "correlation": scenario.correlation,
                        "outlier_probability": scenario.outlier_probability,
                        "outlier_translation": scenario.outlier_translation,
                        "trial_index": trial_index,
                        "seed": seed,
                        "outlier_injected": outlier_injected,
                        "outlier_edge_id": outlier_edge_id,
                        "maximum_raw_cycle_displacement": raw_maximum,
                        "maximum_uncertainty_normalized_cycle_score": (
                            normalized_maximum
                        ),
                        **metrics,
                    }
                )

    aggregate = _aggregate_records(
        records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=int(target_seed) + 41_237,
    )
    decision = _guard_decision(records)
    source_revision_value = _validated_source_revision(source_revision)
    report_body: dict[str, object] = {
        "schema_name": CYCLE_GUARD_MONTE_CARLO_SCHEMA,
        "schema_version": CYCLE_GUARD_MONTE_CARLO_VERSION,
        "source_revision": source_revision_value,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "calibration_trials": calibration_trials,
            "target_trials_per_scenario": target_trials_per_scenario,
            "calibration_seed": int(calibration_seed),
            "target_seed": int(target_seed),
            "threshold_quantile": threshold_quantile,
            "representative_radius": representative_radius,
            "minimum_uncertainty_scale": minimum_uncertainty_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "window_size": window_size,
            "overlap": overlap,
            "minimum_edge_weight": minimum_edge_weight,
            "bootstrap_resamples": bootstrap_resamples,
            "scenarios": [scenario.to_dict() for scenario in normalized_scenarios],
        },
        "calibration": {
            "raw_cycle_threshold": raw_threshold,
            "uncertainty_normalized_cycle_threshold": normalized_threshold,
            "records": calibration_records,
        },
        "trials": records,
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": (
            "Controlled synthetic source-cycle admission only; no held-out "
            "physical-object provider, BayesianPhysTwin, harmful-update, or "
            "Causal4D "
            "benefit claim."
        ),
    }
    report_id = hashlib.sha256(_canonical_json(report_body)).hexdigest()
    report = {"report_id": report_id, **report_body}
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cycle_guard_monte_carlo.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "cycle_guard_monte_carlo_trials.csv", records)
    _write_csv(output / "cycle_guard_monte_carlo.csv", _flat_aggregate(aggregate))
    _write_markdown(
        output / "cycle_guard_monte_carlo.md",
        raw_threshold=raw_threshold,
        normalized_threshold=normalized_threshold,
        aggregate=aggregate,
        decision=decision,
    )
    evidence_files = sorted(output.glob("cycle_guard_monte_carlo.*"))
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
    parser.add_argument("--calibration-trials", type=int, default=48)
    parser.add_argument("--target-trials-per-scenario", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=80_260_804)
    parser.add_argument("--target-seed", type=int, default=271_260_804)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
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
    report = run_cycle_guard_monte_carlo(
        arguments.output_dir,
        calibration_trials=arguments.calibration_trials,
        target_trials_per_scenario=arguments.target_trials_per_scenario,
        calibration_seed=arguments.calibration_seed,
        target_seed=arguments.target_seed,
        threshold_quantile=arguments.threshold_quantile,
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
    print(arguments.output_dir / "cycle_guard_monte_carlo.json")
    return 0 if report["decision"]["overall_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CYCLE_GUARD_MONTE_CARLO_SCHEMA",
    "CYCLE_GUARD_MONTE_CARLO_VERSION",
    "run_cycle_guard_monte_carlo",
]
