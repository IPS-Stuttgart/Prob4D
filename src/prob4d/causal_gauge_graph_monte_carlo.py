"""Run a deterministic Monte Carlo study of causal gauge-graph estimators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .alignment import WindowAlignment
from .alignment_cycles import alignment_edge_id, audit_alignment_cycles
from .causal_gauge_graph import (
    CausalGaugeGraphReport,
    estimate_causal_multi_edge_gauge_graph,
)
from .composition_jacobian import composition_jacobian_mode
from .gauge import RelativeGaugeConstraint, SequentialGaugeEstimator
from .guarded_causal_gauge_graph import (
    GuardedCausalGaugeGraphReport,
    estimate_guarded_causal_multi_edge_gauge_graph,
)
from .observation_export import JointGaugePosterior, _build_alignments, estimate_joint_gauge_tree
from .sim3 import Sim3, so3_log
from .synthetic import SyntheticProblem, make_synthetic_problem

FloatArray = NDArray[np.floating]
GAUGE_GRAPH_MONTE_CARLO_SCHEMA: Final = "prob4d.causal-gauge-graph-monte-carlo"
GAUGE_GRAPH_MONTE_CARLO_VERSION: Final = 1
_CHI_SQUARE_7_95: Final = 14.067140449340169
_METHOD_ORDER: Final = (
    "tree",
    "marginal_ci",
    "full_joint_graph",
    "guarded_graph",
)


@dataclass(frozen=True)
class GaugeGraphStudyScenario:
    """One target-frozen synthetic dependence and outlier condition."""

    scenario_id: str
    correlation: float
    outlier_probability: float = 0.0
    outlier_translation: float = 0.0

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        correlation = float(self.correlation)
        probability = float(self.outlier_probability)
        translation = float(self.outlier_translation)
        if not scenario_id:
            raise ValueError("scenario_id must be nonempty")
        if not math.isfinite(correlation) or not 0.0 <= correlation < 1.0:
            raise ValueError("scenario correlation must lie in [0, 1)")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("outlier_probability must lie in [0, 1]")
        if not math.isfinite(translation) or translation < 0.0:
            raise ValueError("outlier_translation must be finite and nonnegative")
        if probability > 0.0 and translation <= 0.0:
            raise ValueError("positive outlier probability requires positive translation")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "outlier_probability", probability)
        object.__setattr__(self, "outlier_translation", translation)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "correlation": self.correlation,
            "outlier_probability": self.outlier_probability,
            "outlier_translation": self.outlier_translation,
        }


DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS: Final = (
    GaugeGraphStudyScenario("independent_clean", correlation=0.0),
    GaugeGraphStudyScenario("correlated_clean", correlation=0.75),
    GaugeGraphStudyScenario("highly_correlated_clean", correlation=0.95),
    GaugeGraphStudyScenario(
        "correlated_mild_outliers",
        correlation=0.75,
        outlier_probability=0.25,
        outlier_translation=0.10,
    ),
    GaugeGraphStudyScenario(
        "correlated_strong_outliers",
        correlation=0.75,
        outlier_probability=0.25,
        outlier_translation=0.30,
    ),
    GaugeGraphStudyScenario(
        "highly_correlated_strong_outliers",
        correlation=0.95,
        outlier_probability=0.25,
        outlier_translation=0.30,
    ),
)


@dataclass(frozen=True)
class _EstimatorOutput:
    method_id: str
    posterior_mode: str
    estimates: Mapping[str, Sim3]
    covariances: Mapping[str, FloatArray]
    fallback_applied: bool | None = None
    mean_effective_edge_count: float | None = None


def _validated_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized != value or normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _validated_source_revision(value: str | None) -> str:
    revision = "" if value is None else str(value).strip()
    if not revision:
        revision = os.environ.get("GITHUB_HEAD_SHA", "").strip()
    if not revision:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
        else:
            revision = result.stdout.strip() or "unknown"
    if revision == "unknown":
        return revision
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("source_revision must be a lowercase 40- or 64-character commit")
    return revision


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(str(int(seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def _make_problem(
    *,
    seed: int,
    correlation: float,
    num_frames: int,
    height: int,
    width: int,
    window_size: int,
    overlap: int,
) -> SyntheticProblem:
    return make_synthetic_problem(
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        window_size=window_size,
        overlap=overlap,
        correlation=correlation,
    )


def _cycle_threshold_calibration(
    scenarios: Sequence[GaugeGraphStudyScenario],
    *,
    calibration_trials: int,
    calibration_seed: int,
    threshold_quantile: float,
    representative_radius: float,
    num_frames: int,
    height: int,
    width: int,
    window_size: int,
    overlap: int,
) -> tuple[float, list[dict[str, object]]]:
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
        clean_correlations = tuple(sorted({scenario.correlation for scenario in scenarios}))
    if calibration_trials < len(clean_correlations):
        raise ValueError(
            "calibration_trials must cover every registered clean correlation at least once"
        )
    maxima: list[float] = []
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
        audit = audit_alignment_cycles(
            alignments,
            representative_radius=representative_radius,
        )
        if audit.cycle_count < 1:
            raise ValueError("registered synthetic geometry produced no gauge cycles")
        maximum = audit.maximum_observed_representative_displacement
        maxima.append(maximum)
        records.append(
            {
                "trial_index": index,
                "seed": seed,
                "correlation": correlation,
                "cycle_count": audit.cycle_count,
                "maximum_cycle_displacement": maximum,
            }
        )
    threshold = float(
        np.quantile(
            np.asarray(maxima, dtype=np.float64),
            threshold_quantile,
            method="higher",
        )
    )
    threshold = max(threshold, float(np.nextafter(0.0, 1.0)))
    return threshold, records


def _inject_inconsistent_skip_edge(
    alignments: Sequence[WindowAlignment],
    window_ids: Sequence[str],
    *,
    generator: np.random.Generator,
    probability: float,
    translation_magnitude: float,
) -> tuple[tuple[WindowAlignment, ...], bool, str | None]:
    normalized = tuple(alignments)
    if probability <= 0.0 or generator.random() >= probability:
        return normalized, False, None
    positions = {window_id: index for index, window_id in enumerate(window_ids)}
    edges = {(item.reference_id, item.moving_id) for item in normalized}
    candidates: list[tuple[int, WindowAlignment]] = []
    for index, alignment in enumerate(normalized):
        reference_position = positions[alignment.reference_id]
        moving_position = positions[alignment.moving_id]
        if moving_position - reference_position < 2:
            continue
        has_path = any(
            (alignment.reference_id, middle_id) in edges
            and (middle_id, alignment.moving_id) in edges
            for middle_id in window_ids[reference_position + 1 : moving_position]
        )
        if has_path:
            candidates.append((index, alignment))
    if not candidates:
        raise ValueError("outlier injection requires one direct edge with a two-edge path")
    selected_position = int(generator.integers(0, len(candidates)))
    selected_index, selected = candidates[selected_position]
    direction = generator.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(np.float64).eps:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= norm
    transform = selected.result.transform
    perturbed = Sim3(
        scale=transform.scale,
        rotation=transform.rotation,
        translation=transform.translation + translation_magnitude * direction,
    )
    changed = list(normalized)
    changed[selected_index] = replace(
        selected,
        result=replace(selected.result, transform=perturbed),
    )
    return tuple(changed), True, alignment_edge_id(selected)


def _joint_output(
    method_id: str,
    posterior: JointGaugePosterior,
    *,
    fallback_applied: bool | None = None,
    graph_report: CausalGaugeGraphReport | None = None,
) -> _EstimatorOutput:
    covariances = {
        window_id: posterior.joint_covariance[
            7 * index : 7 * (index + 1),
            7 * index : 7 * (index + 1),
        ]
        for index, window_id in enumerate(posterior.window_ids)
    }
    effective_edge_count = None
    if graph_report is not None:
        multi_edge = [
            1.0 / float(np.sum(step.covariance_intersection_weights**2))
            for step in graph_report.steps
            if len(step.candidate_parent_ids) > 1
        ]
        effective_edge_count = None if not multi_edge else float(np.mean(multi_edge))
    return _EstimatorOutput(
        method_id=method_id,
        posterior_mode=posterior.mode,
        estimates=posterior.estimates,
        covariances=covariances,
        fallback_applied=fallback_applied,
        mean_effective_edge_count=effective_edge_count,
    )


def _run_estimators(
    problem: SyntheticProblem,
    alignments: Sequence[WindowAlignment],
    *,
    cycle_threshold: float,
    representative_radius: float,
    minimum_edge_weight: float,
) -> tuple[dict[str, _EstimatorOutput], GuardedCausalGaugeGraphReport]:
    windows = problem.overlap_windows
    window_ids = [window.window_id for window in windows]
    first_id = window_ids[0]
    initial_transform = problem.true_overlap_gauges[first_id]
    anchor_standard_deviation = np.asarray(
        [1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3],
        dtype=np.float64,
    )
    initial_covariance = np.diag(anchor_standard_deviation**2)
    with composition_jacobian_mode("analytic"):
        tree = estimate_joint_gauge_tree(
            windows,
            alignments,
            initial_transform=initial_transform,
            initial_covariance=initial_covariance,
        )
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in alignments
    ]
    marginal = SequentialGaugeEstimator().estimate(
        window_ids,
        constraints,
        initial_transform=initial_transform,
        initial_covariance=initial_covariance,
    )
    graph, graph_report = estimate_causal_multi_edge_gauge_graph(
        windows,
        alignments,
        initial_transform=initial_transform,
        initial_covariance=initial_covariance,
        minimum_edge_weight=minimum_edge_weight,
    )
    guarded, guarded_report = estimate_guarded_causal_multi_edge_gauge_graph(
        windows,
        alignments,
        initial_transform=initial_transform,
        initial_covariance=initial_covariance,
        maximum_cycle_displacement=cycle_threshold,
        representative_radius=representative_radius,
        minimum_cycles_per_multi_edge_child=1,
        minimum_edge_weight=minimum_edge_weight,
    )
    guarded_graph_report = guarded_report.graph_report
    outputs = {
        "tree": _joint_output("tree", tree),
        "marginal_ci": _EstimatorOutput(
            method_id="marginal_ci",
            posterior_mode="sequential_marginal_multi_parent_ci_v1",
            estimates={
                window_id: marginal[window_id].global_from_local
                for window_id in window_ids
            },
            covariances={
                window_id: marginal[window_id].covariance
                for window_id in window_ids
            },
        ),
        "full_joint_graph": _joint_output(
            "full_joint_graph",
            graph,
            graph_report=graph_report,
        ),
        "guarded_graph": _joint_output(
            "guarded_graph",
            guarded,
            fallback_applied=guarded_report.fallback_applied,
            graph_report=guarded_graph_report,
        ),
    }
    return outputs, guarded_report


def _representative_points(radius: float) -> FloatArray:
    axes = radius * np.eye(3, dtype=np.float64)
    return np.concatenate((np.zeros((1, 3)), axes, -axes), axis=0)


def _representative_displacement(
    estimated: Sim3,
    truth: Sim3,
    *,
    radius: float,
) -> float:
    points = _representative_points(radius)
    difference = estimated.transform_points(points) - truth.transform_points(points)
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def _parameter_error(estimated: Sim3, truth: Sim3) -> FloatArray:
    return np.concatenate(
        (
            np.asarray([math.log(estimated.scale / truth.scale)]),
            so3_log(estimated.rotation @ truth.rotation.T),
            estimated.translation - truth.translation,
        )
    )


def _normalized_nees(error: FloatArray, covariance: FloatArray) -> float:
    matrix = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    spectral_scale = max(float(np.max(eigenvalues)), np.finfo(np.float64).tiny)
    floor = max(1e-12, 1e-10 * spectral_scale)
    inverse = (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    return float(error @ inverse @ error / error.size)


def _normalized_covariance_trace(covariance: FloatArray, *, radius: float) -> float:
    normalizer = np.diag([radius, radius, radius, radius, 1.0, 1.0, 1.0])
    normalized = normalizer @ np.asarray(covariance, dtype=np.float64) @ normalizer
    return float(np.trace(normalized) / 7.0)


def _evaluate_output(
    output: _EstimatorOutput,
    problem: SyntheticProblem,
    *,
    representative_radius: float,
) -> dict[str, object]:
    window_ids = [window.window_id for window in problem.overlap_windows]
    displacements: list[float] = []
    normalized_nees: list[float] = []
    coverage: list[float] = []
    covariance_widths: list[float] = []
    for window_id in window_ids[1:]:
        estimated = output.estimates[window_id]
        truth = problem.true_overlap_gauges[window_id]
        covariance = output.covariances[window_id]
        displacement = _representative_displacement(
            estimated,
            truth,
            radius=representative_radius,
        )
        error = _parameter_error(estimated, truth)
        nees = _normalized_nees(error, covariance)
        displacements.append(displacement)
        normalized_nees.append(nees)
        coverage.append(float(nees * 7.0 <= _CHI_SQUARE_7_95))
        covariance_widths.append(
            _normalized_covariance_trace(
                covariance,
                radius=representative_radius,
            )
        )
    values = np.asarray(displacements, dtype=np.float64)
    positions = np.arange(1, len(values) + 1, dtype=np.float64)
    centered = positions - float(np.mean(positions))
    denominator = float(centered @ centered)
    slope = 0.0 if denominator == 0.0 else float(centered @ values / denominator)
    return {
        "method_id": output.method_id,
        "posterior_mode": output.posterior_mode,
        "fallback_applied": output.fallback_applied,
        "mean_effective_edge_count": output.mean_effective_edge_count,
        "endpoint_displacement_rmse": float(values[-1]),
        "mean_window_displacement_rmse": float(np.mean(values)),
        "p90_window_displacement_rmse": float(np.quantile(values, 0.90)),
        "drift_slope": slope,
        "coverage_95": float(np.mean(coverage)),
        "coverage_shortfall_95": max(0.0, 0.95 - float(np.mean(coverage))),
        "mean_normalized_nees": float(np.mean(normalized_nees)),
        "mean_normalized_covariance_trace": float(np.mean(covariance_widths)),
    }


def _bootstrap_summary(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    estimate = float(np.mean(array))
    if array.size == 1:
        lower = upper = estimate
    else:
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, array.size, size=(resamples, array.size))
        means = np.mean(array[indices], axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "mean": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "count": int(array.size),
    }


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
            injected = [bool(record["outlier_injected"]) for record in group]
            outlier_fallback = [
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
                    "outlier_injected_count": sum(injected),
                    "metrics": metrics,
                    "paired_endpoint_delta_vs_tree": paired_summary,
                    "endpoint_win_rate_vs_tree": float(
                        np.mean(np.asarray(paired) < -1e-12)
                    ),
                    "endpoint_harm_rate_vs_tree": float(
                        np.mean(np.asarray(paired) > 1e-12)
                    ),
                    "fallback_rate": (
                        None
                        if not fallback_values
                        else float(np.mean(fallback_values))
                    ),
                    "outlier_detection_rate": (
                        None
                        if not outlier_fallback
                        else float(np.mean(outlier_fallback))
                    ),
                    "clean_false_fallback_rate": (
                        None
                        if not clean_fallback
                        else float(np.mean(clean_fallback))
                    ),
                }
            )
    return aggregate


def _flat_aggregate_rows(aggregate: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in aggregate:
        metrics = summary["metrics"]
        assert isinstance(metrics, Mapping)
        paired = summary["paired_endpoint_delta_vs_tree"]
        assert isinstance(paired, Mapping)
        row: dict[str, object] = {
            "scenario_id": summary["scenario_id"],
            "method_id": summary["method_id"],
            "trial_count": summary["trial_count"],
            "outlier_injected_count": summary["outlier_injected_count"],
            "endpoint_win_rate_vs_tree": summary["endpoint_win_rate_vs_tree"],
            "endpoint_harm_rate_vs_tree": summary["endpoint_harm_rate_vs_tree"],
            "fallback_rate": summary["fallback_rate"],
            "outlier_detection_rate": summary["outlier_detection_rate"],
            "clean_false_fallback_rate": summary["clean_false_fallback_rate"],
            "paired_endpoint_delta_mean": paired["mean"],
            "paired_endpoint_delta_ci95_lower": paired["ci95_lower"],
            "paired_endpoint_delta_ci95_upper": paired["ci95_upper"],
        }
        for metric_name, metric_summary in metrics.items():
            assert isinstance(metric_summary, Mapping)
            row[f"{metric_name}_mean"] = metric_summary["mean"]
            row[f"{metric_name}_ci95_lower"] = metric_summary["ci95_lower"]
            row[f"{metric_name}_ci95_upper"] = metric_summary["ci95_upper"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "scenario_id",
        "method_id",
        "trial_count",
        "outlier_injected_count",
    ]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_interval(summary: Mapping[str, object]) -> str:
    return (
        f"{float(summary['mean']):.5g} "
        f"[{float(summary['ci95_lower']):.5g}, "
        f"{float(summary['ci95_upper']):.5g}]"
    )


def _write_markdown(
    path: Path,
    *,
    threshold: float,
    threshold_quantile: float,
    calibration_trial_count: int,
    aggregate: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# Causal gauge-graph Monte Carlo pilot",
        "",
        "This is a controlled mechanistic study, not a real-data or downstream physical-twin "
        "claim.",
        "",
        f"The source-only cycle threshold is `{threshold:.8g}` at calibration "
        f"quantile `{threshold_quantile:.3f}` from `{calibration_trial_count}` "
        "independent clean calibration trials.",
        "",
        "Every interval is a deterministic trial bootstrap. Endpoint differences are "
        "`candidate - production tree`, so negative values favor the candidate.",
        "",
        "| Scenario | Method | N | Endpoint displacement | Delta vs tree | Win rate | "
        "Coverage 95% | NEES / dimension | Covariance width | Fallback |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in aggregate:
        metrics = summary["metrics"]
        paired = summary["paired_endpoint_delta_vs_tree"]
        assert isinstance(metrics, Mapping)
        assert isinstance(paired, Mapping)
        fallback = summary["fallback_rate"]
        fallback_text = "—" if fallback is None else f"{float(fallback):.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["scenario_id"]),
                    str(summary["method_id"]),
                    str(summary["trial_count"]),
                    _format_interval(metrics["endpoint_displacement_rmse"]),
                    _format_interval(paired),
                    f"{float(summary['endpoint_win_rate_vs_tree']):.3f}",
                    f"{float(metrics['coverage_95']['mean']):.3f}",
                    f"{float(metrics['mean_normalized_nees']['mean']):.3f}",
                    f"{float(metrics['mean_normalized_covariance_trace']['mean']):.5g}",
                    fallback_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Guard detection diagnostics",
            "",
            "| Scenario | Injected trials | Detection rate | Clean false-fallback rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for summary in aggregate:
        if summary["method_id"] != "guarded_graph":
            continue
        detection = summary["outlier_detection_rate"]
        false_fallback = summary["clean_false_fallback_rate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["scenario_id"]),
                    str(summary["outlier_injected_count"]),
                    "—" if detection is None else f"{float(detection):.3f}",
                    (
                        "—"
                        if false_fallback is None
                        else f"{float(false_fallback):.3f}"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The study tests estimator mechanics under the repository's synthetic correlated "
            "window generator. It does not establish provider competence on held-out physical "
            "objects, BayesianPhysTwin acceptance or physical-prediction benefit, harmful "
            "accepted-update control, or Causal4D intervention benefit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gauge_graph_monte_carlo(
    output_directory: str | Path,
    *,
    scenarios: Sequence[GaugeGraphStudyScenario] = DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS,
    calibration_trials: int = 24,
    target_trials_per_scenario: int = 32,
    calibration_seed: int = 20_260_804,
    target_seed: int = 91_260_804,
    threshold_quantile: float = 0.95,
    representative_radius: float = 1.0,
    num_frames: int = 32,
    height: int = 5,
    width: int = 7,
    window_size: int = 12,
    overlap: int = 8,
    minimum_edge_weight: float = 0.0,
    bootstrap_resamples: int = 1_000,
    source_revision: str | None = None,
) -> dict[str, object]:
    """Calibrate a source-only guard and evaluate frozen target scenarios."""

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
        raise ValueError("at least one study scenario is required")
    scenario_ids = tuple(scenario.scenario_id for scenario in normalized_scenarios)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("study scenario IDs must be unique")
    threshold_quantile = float(threshold_quantile)
    representative_radius = float(representative_radius)
    minimum_edge_weight = float(minimum_edge_weight)
    if not math.isfinite(threshold_quantile) or not 0.0 < threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must lie in (0, 1]")
    if not math.isfinite(representative_radius) or representative_radius <= 0.0:
        raise ValueError("representative_radius must be finite and positive")
    if not math.isfinite(minimum_edge_weight) or minimum_edge_weight < 0.0:
        raise ValueError("minimum_edge_weight must be finite and nonnegative")
    for name, value in (
        ("num_frames", num_frames),
        ("height", height),
        ("width", width),
        ("window_size", window_size),
        ("overlap", overlap),
    ):
        _validated_positive_integer(value, name=name)
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    if num_frames <= window_size:
        raise ValueError("num_frames must exceed window_size")

    threshold, calibration_records = _cycle_threshold_calibration(
        normalized_scenarios,
        calibration_trials=calibration_trials,
        calibration_seed=int(calibration_seed),
        threshold_quantile=threshold_quantile,
        representative_radius=representative_radius,
        num_frames=num_frames,
        height=height,
        width=width,
        window_size=window_size,
        overlap=overlap,
    )

    trial_records: list[dict[str, object]] = []
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
            outlier_generator = np.random.default_rng(
                _stable_seed(target_seed, scenario.scenario_id, trial_index, "outlier")
            )
            alignments, outlier_injected, outlier_edge_id = (
                _inject_inconsistent_skip_edge(
                    alignments,
                    [window.window_id for window in problem.overlap_windows],
                    generator=outlier_generator,
                    probability=scenario.outlier_probability,
                    translation_magnitude=scenario.outlier_translation,
                )
            )
            source_audit = audit_alignment_cycles(
                alignments,
                representative_radius=representative_radius,
            )
            outputs, guarded_report = _run_estimators(
                problem,
                alignments,
                cycle_threshold=threshold,
                representative_radius=representative_radius,
                minimum_edge_weight=minimum_edge_weight,
            )
            for method_id in _METHOD_ORDER:
                metrics = _evaluate_output(
                    outputs[method_id],
                    problem,
                    representative_radius=representative_radius,
                )
                trial_records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "correlation": scenario.correlation,
                        "outlier_probability": scenario.outlier_probability,
                        "outlier_translation": scenario.outlier_translation,
                        "trial_index": trial_index,
                        "seed": seed,
                        "outlier_injected": outlier_injected,
                        "outlier_edge_id": outlier_edge_id,
                        "source_cycle_count": source_audit.cycle_count,
                        "maximum_source_cycle_displacement": (
                            source_audit.maximum_observed_representative_displacement
                        ),
                        "guard_cycle_passed": guarded_report.cycle_audit.passed,
                        **metrics,
                    }
                )

    aggregate = _aggregate_records(
        trial_records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=int(target_seed) + 7_919,
    )
    source_revision_value = _validated_source_revision(source_revision)
    report_body: dict[str, object] = {
        "schema_name": GAUGE_GRAPH_MONTE_CARLO_SCHEMA,
        "schema_version": GAUGE_GRAPH_MONTE_CARLO_VERSION,
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
            "cycle_threshold": threshold,
            "selection_semantics": (
                "higher empirical quantile of the maximum source-only cycle displacement "
                "per clean calibration trial"
            ),
            "records": calibration_records,
        },
        "trials": trial_records,
        "aggregate": aggregate,
        "claim_boundary": (
            "Controlled synthetic estimator mechanics only; no held-out physical-object, "
            "BayesianPhysTwin, harmful-update, or Causal4D benefit claim."
        ),
    }
    report_id = hashlib.sha256(_canonical_json(report_body)).hexdigest()
    report = {"report_id": report_id, **report_body}
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gauge_graph_monte_carlo.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "gauge_graph_monte_carlo_trials.csv", trial_records)
    _write_csv(
        output / "gauge_graph_monte_carlo.csv",
        _flat_aggregate_rows(aggregate),
    )
    _write_markdown(
        output / "gauge_graph_monte_carlo.md",
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        calibration_trial_count=calibration_trials,
        aggregate=aggregate,
    )
    (output / "SHA256SUMS").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in sorted(output.glob("gauge_graph_monte_carlo.*"))
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-trials", type=int, default=24)
    parser.add_argument("--target-trials-per-scenario", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=20_260_804)
    parser.add_argument("--target-seed", type=int, default=91_260_804)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--representative-radius", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--minimum-edge-weight", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--source-revision")
    arguments = parser.parse_args(argv)
    run_gauge_graph_monte_carlo(
        arguments.output_dir,
        calibration_trials=arguments.calibration_trials,
        target_trials_per_scenario=arguments.target_trials_per_scenario,
        calibration_seed=arguments.calibration_seed,
        target_seed=arguments.target_seed,
        threshold_quantile=arguments.threshold_quantile,
        representative_radius=arguments.representative_radius,
        num_frames=arguments.num_frames,
        height=arguments.height,
        width=arguments.width,
        window_size=arguments.window_size,
        overlap=arguments.overlap,
        minimum_edge_weight=arguments.minimum_edge_weight,
        bootstrap_resamples=arguments.bootstrap_resamples,
        source_revision=arguments.source_revision,
    )
    print(arguments.output_dir / "gauge_graph_monte_carlo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GAUGE_GRAPH_STUDY_SCENARIOS",
    "GAUGE_GRAPH_MONTE_CARLO_SCHEMA",
    "GAUGE_GRAPH_MONTE_CARLO_VERSION",
    "GaugeGraphStudyScenario",
    "run_gauge_graph_monte_carlo",
]
