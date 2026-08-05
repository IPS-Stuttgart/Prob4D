"""Evaluate finite-sample calibration of the source-only normalized cycle guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np

from .causal_gauge_graph_monte_carlo import (
    DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS,
    GaugeGraphStudyScenario,
    _canonical_json,
    _evaluate_output,
    _inject_inconsistent_skip_edge,
    _make_problem,
    _run_estimators,
    _stable_seed,
    _validated_positive_integer,
    _validated_source_revision,
)
from .cycle_guard_conformal_reporting import (
    _METHOD_ORDER,
    _aggregate_records,
    _decision,
    _flat_aggregate,
    _write_csv,
    _write_markdown,
)
from .cycle_guard_monte_carlo import (
    _assert_exact_fallback,
    _calibrate_thresholds,
    _normalized_guard_output,
)
from .finite_sample_threshold import fit_finite_sample_upper_threshold
from .observation_export import _build_alignments

CYCLE_GUARD_CONFORMAL_SCHEMA: Final = "prob4d.cycle-guard-conformal-monte-carlo"
CYCLE_GUARD_CONFORMAL_VERSION: Final = 1


def run_cycle_guard_conformal_monte_carlo(
    output_directory: str | Path,
    *,
    scenarios: Sequence[GaugeGraphStudyScenario] = DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS,
    calibration_trials: int = 96,
    target_trials_per_scenario: int = 128,
    calibration_seed: int = 408_260_804,
    target_seed: int = 731_260_804,
    empirical_threshold_quantile: float = 0.95,
    conformal_miscoverage: float = 0.05,
    representative_radius: float = 1.0,
    minimum_uncertainty_scale: float = 1e-12,
    num_frames: int = 28,
    height: int = 4,
    width: int = 6,
    window_size: int = 11,
    overlap: int = 7,
    minimum_edge_weight: float = 0.0,
    bootstrap_resamples: int = 2_000,
    endpoint_noninferiority_margin: float = 0.001,
    minimum_clean_coverage: float = 0.90,
    maximum_clean_covariance_trace_ratio: float = 1.25,
    source_revision: str | None = None,
) -> dict[str, object]:
    """Run a fresh-seed empirical-versus-conformal normalized-guard study."""

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
        raise ValueError("at least one conformal cycle-guard scenario is required")
    if len({scenario.scenario_id for scenario in normalized_scenarios}) != len(
        normalized_scenarios
    ):
        raise ValueError("conformal cycle-guard scenario IDs must be unique")
    empirical_threshold_quantile = float(empirical_threshold_quantile)
    if isinstance(conformal_miscoverage, (bool, np.bool_)) or not isinstance(
        conformal_miscoverage,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("conformal_miscoverage must be a real number")
    conformal_miscoverage = float(conformal_miscoverage)
    representative_radius = float(representative_radius)
    minimum_uncertainty_scale = float(minimum_uncertainty_scale)
    minimum_edge_weight = float(minimum_edge_weight)
    endpoint_noninferiority_margin = float(endpoint_noninferiority_margin)
    minimum_clean_coverage = float(minimum_clean_coverage)
    maximum_clean_covariance_trace_ratio = float(
        maximum_clean_covariance_trace_ratio
    )
    for name, value in (
        ("representative_radius", representative_radius),
        ("minimum_uncertainty_scale", minimum_uncertainty_scale),
        ("endpoint_noninferiority_margin", endpoint_noninferiority_margin),
        ("maximum_clean_covariance_trace_ratio", maximum_clean_covariance_trace_ratio),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(minimum_edge_weight) or minimum_edge_weight < 0.0:
        raise ValueError("minimum_edge_weight must be finite and nonnegative")
    if (
        not np.isfinite(empirical_threshold_quantile)
        or not 0.0 < empirical_threshold_quantile <= 1.0
    ):
        raise ValueError("empirical_threshold_quantile must lie in (0, 1]")
    if not np.isfinite(conformal_miscoverage) or not 0.0 < conformal_miscoverage < 1.0:
        raise ValueError("conformal_miscoverage must lie strictly between zero and one")
    conformal_rank = int(
        np.ceil((calibration_trials + 1) * (1.0 - conformal_miscoverage))
    )
    if conformal_rank > calibration_trials:
        raise ValueError(
            "conformal_miscoverage is below the finite calibration resolution; "
            "increase calibration_trials or conformal_miscoverage"
        )
    if not np.isfinite(minimum_clean_coverage) or not 0.0 <= minimum_clean_coverage <= 1.0:
        raise ValueError("minimum_clean_coverage must lie in [0, 1]")

    raw_threshold, empirical_threshold, calibration_records = _calibrate_thresholds(
        normalized_scenarios,
        calibration_trials=calibration_trials,
        calibration_seed=int(calibration_seed),
        threshold_quantile=empirical_threshold_quantile,
        representative_radius=representative_radius,
        minimum_uncertainty_scale=minimum_uncertainty_scale,
        num_frames=num_frames,
        height=height,
        width=width,
        window_size=window_size,
        overlap=overlap,
    )
    normalized_scores = np.asarray(
        [
            float(record["maximum_uncertainty_normalized_score"])
            for record in calibration_records
        ],
        dtype=np.float64,
    )
    conformal = fit_finite_sample_upper_threshold(
        normalized_scores,
        miscoverage=conformal_miscoverage,
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
            alignments, outlier_injected, outlier_edge_id = _inject_inconsistent_skip_edge(
                alignments,
                [window.window_id for window in problem.overlap_windows],
                generator=np.random.default_rng(
                    _stable_seed(
                        target_seed,
                        scenario.scenario_id,
                        trial_index,
                        "conformal-cycle-guard-outlier",
                    )
                ),
                probability=scenario.outlier_probability,
                translation_magnitude=scenario.outlier_translation,
            )
            raw_outputs, raw_report = _run_estimators(
                problem,
                alignments,
                cycle_threshold=raw_threshold,
                representative_radius=representative_radius,
                minimum_edge_weight=minimum_edge_weight,
            )
            empirical_output, empirical_report = _normalized_guard_output(
                problem,
                alignments,
                threshold=empirical_threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            conformal_output, conformal_report = _normalized_guard_output(
                problem,
                alignments,
                threshold=conformal.threshold,
                representative_radius=representative_radius,
                minimum_uncertainty_scale=minimum_uncertainty_scale,
                minimum_edge_weight=minimum_edge_weight,
            )
            tree = raw_outputs["tree"]
            raw_guard = replace(
                raw_outputs["guarded_graph"],
                method_id="raw_guarded_graph",
            )
            empirical_output = replace(
                empirical_output,
                method_id="empirical_normalized_guard",
            )
            conformal_output = replace(
                conformal_output,
                method_id="conformal_normalized_guard",
            )
            _assert_exact_fallback(raw_guard, tree)
            _assert_exact_fallback(empirical_output, tree)
            _assert_exact_fallback(conformal_output, tree)
            methods = {
                "tree": tree,
                "full_joint_graph": raw_outputs["full_joint_graph"],
                "raw_guarded_graph": raw_guard,
                "empirical_normalized_guard": empirical_output,
                "conformal_normalized_guard": conformal_output,
            }
            raw_maximum = (
                raw_report.cycle_audit.maximum_observed_representative_displacement
            )
            empirical_maximum = (
                empirical_report.cycle_audit.maximum_observed_normalized_score
            )
            conformal_maximum = (
                conformal_report.cycle_audit.maximum_observed_normalized_score
            )
            if not np.isclose(empirical_maximum, conformal_maximum, atol=0.0, rtol=0.0):
                raise ValueError("normalized score changed with the admission threshold")
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
                            empirical_maximum
                        ),
                        **metrics,
                    }
                )

    aggregate = _aggregate_records(
        records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=int(target_seed) + 91_733,
    )
    decision = _decision(
        records,
        aggregate,
        conformal_miscoverage=conformal_miscoverage,
        guaranteed_miscoverage_upper_bound=(
            conformal.guaranteed_miscoverage_upper_bound
        ),
        endpoint_noninferiority_margin=endpoint_noninferiority_margin,
        minimum_clean_coverage=minimum_clean_coverage,
        maximum_clean_covariance_trace_ratio=maximum_clean_covariance_trace_ratio,
    )
    report_body: dict[str, object] = {
        "schema_name": CYCLE_GUARD_CONFORMAL_SCHEMA,
        "schema_version": CYCLE_GUARD_CONFORMAL_VERSION,
        "source_revision": _validated_source_revision(source_revision),
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
            "empirical_threshold_quantile": empirical_threshold_quantile,
            "conformal_miscoverage": conformal_miscoverage,
            "representative_radius": representative_radius,
            "minimum_uncertainty_scale": minimum_uncertainty_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "window_size": window_size,
            "overlap": overlap,
            "minimum_edge_weight": minimum_edge_weight,
            "bootstrap_resamples": bootstrap_resamples,
            "endpoint_noninferiority_margin": endpoint_noninferiority_margin,
            "minimum_clean_coverage": minimum_clean_coverage,
            "maximum_clean_covariance_trace_ratio": (
                maximum_clean_covariance_trace_ratio
            ),
            "scenarios": [scenario.to_dict() for scenario in normalized_scenarios],
        },
        "calibration": {
            "raw_cycle_threshold": raw_threshold,
            "empirical_normalized_cycle_threshold": empirical_threshold,
            "finite_sample_normalized_cycle_threshold": conformal.to_dict(),
            "records": calibration_records,
        },
        "trials": records,
        "aggregate": aggregate,
        "decision": decision,
        "claim_boundary": (
            "Controlled synthetic finite-sample source-cycle calibration only; no "
            "held-out physical-object provider, BayesianPhysTwin, harmful-update, "
            "or Causal4D benefit claim."
        ),
    }
    report = {
        "report_id": hashlib.sha256(_canonical_json(report_body)).hexdigest(),
        **report_body,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cycle_guard_conformal_monte_carlo.json"
    aggregate_path = output / "cycle_guard_conformal_monte_carlo.csv"
    trials_path = output / "cycle_guard_conformal_monte_carlo_trials.csv"
    markdown_path = output / "cycle_guard_conformal_monte_carlo.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(trials_path, records)
    _write_csv(aggregate_path, _flat_aggregate(aggregate))
    _write_markdown(
        markdown_path,
        raw_threshold=raw_threshold,
        empirical_threshold=empirical_threshold,
        conformal_threshold=conformal.to_dict(),
        aggregate=aggregate,
        decision=decision,
    )
    evidence_files = (json_path, aggregate_path, trials_path, markdown_path)
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
    parser.add_argument("--calibration-trials", type=int, default=96)
    parser.add_argument("--target-trials-per-scenario", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=408_260_804)
    parser.add_argument("--target-seed", type=int, default=731_260_804)
    parser.add_argument("--empirical-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--conformal-miscoverage", type=float, default=0.05)
    parser.add_argument("--representative-radius", type=float, default=1.0)
    parser.add_argument("--minimum-uncertainty-scale", type=float, default=1e-12)
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=11)
    parser.add_argument("--overlap", type=int, default=7)
    parser.add_argument("--minimum-edge-weight", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--endpoint-noninferiority-margin", type=float, default=0.001)
    parser.add_argument("--minimum-clean-coverage", type=float, default=0.90)
    parser.add_argument(
        "--maximum-clean-covariance-trace-ratio",
        type=float,
        default=1.25,
    )
    parser.add_argument("--source-revision")
    arguments = parser.parse_args(argv)
    report = run_cycle_guard_conformal_monte_carlo(
        arguments.output_dir,
        calibration_trials=arguments.calibration_trials,
        target_trials_per_scenario=arguments.target_trials_per_scenario,
        calibration_seed=arguments.calibration_seed,
        target_seed=arguments.target_seed,
        empirical_threshold_quantile=arguments.empirical_threshold_quantile,
        conformal_miscoverage=arguments.conformal_miscoverage,
        representative_radius=arguments.representative_radius,
        minimum_uncertainty_scale=arguments.minimum_uncertainty_scale,
        num_frames=arguments.num_frames,
        height=arguments.height,
        width=arguments.width,
        window_size=arguments.window_size,
        overlap=arguments.overlap,
        minimum_edge_weight=arguments.minimum_edge_weight,
        bootstrap_resamples=arguments.bootstrap_resamples,
        endpoint_noninferiority_margin=arguments.endpoint_noninferiority_margin,
        minimum_clean_coverage=arguments.minimum_clean_coverage,
        maximum_clean_covariance_trace_ratio=(
            arguments.maximum_clean_covariance_trace_ratio
        ),
        source_revision=arguments.source_revision,
    )
    print(arguments.output_dir / "cycle_guard_conformal_monte_carlo.json")
    decision = report["decision"]
    if not isinstance(decision, Mapping):
        raise RuntimeError("cycle-guard conformal report decision must be a mapping")
    return 0 if decision.get("overall_passed") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CYCLE_GUARD_CONFORMAL_SCHEMA",
    "CYCLE_GUARD_CONFORMAL_VERSION",
    "run_cycle_guard_conformal_monte_carlo",
]
