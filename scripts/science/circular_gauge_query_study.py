#!/usr/bin/env python3
"""Reproduce analytic/synthetic circular-gauge controls without accessing data.

Run with PYTHONPATH=src. Only the optional --plots flag requires Matplotlib.
No provider inference, existing benchmark, or protected source/target is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.circular_gauge_query import (
    AffineCircularQuery,
    CircularPrior,
    bounded_risk_admissible,
    path_violation_probability,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols/circular-gauge-query-control-v1.json"


def write_new(path: Path, data: bytes) -> None:
    """No-clobber output; exact existing bytes are idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to change existing evidence: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def wilson_interval(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def gaussian_probability_below(mean: float, variance: float, threshold: float) -> float:
    if variance == 0:
        return float(mean < threshold)
    return 0.5 * math.erfc((mean - threshold) / math.sqrt(2 * variance))


def run() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text())
    radius = protocol["radius_m"]
    sigma = protocol["phase_stddev_rad"]
    threshold = protocol["violation_below_m"]
    risk_budget = protocol["illustrative_risk_budget"]
    prior = CircularPrior.wrapped_normal(0, sigma, prior_id="explicit-control-conditional-prior-v1")
    coordinate = AffineCircularQuery(np.array([0.0]), np.array([radius]), np.array([0.0]))
    violation = AffineCircularQuery(np.array([threshold]), np.array([-radius]), np.array([0.0]))
    moments = coordinate.moments(prior)
    probability = path_violation_probability(violation, prior, tail_tolerance=protocol["tail_tolerance"])
    true_mean, true_variance = float(moments.mean[0]), float(moments.covariance[0, 0])
    rows: list[dict[str, Any]] = []

    def add(name: str, mean: float, variance: float, risk: float, order: int | None = None) -> None:
        rows.append({
            "method": name,
            "quadrature_order": order,
            "mean_mm": mean * 1000,
            "stddev_mm": math.sqrt(max(0.0, variance)) * 1000,
            "variance_m2": variance,
            "violation_probability": risk,
            "probability_absolute_error": abs(risk - probability.lower),
            "mean_absolute_error_mm": abs(mean - true_mean) * 1000,
            "admitted_at_illustrative_budget": risk <= risk_budget,
        })

    add("first-order-Gaussian", radius, 0.0, gaussian_probability_below(radius, 0.0, threshold))
    sigma_points = np.array([-sigma, sigma])
    sigma_outputs = radius * np.cos(sigma_points)
    add("two-point-spherical-radial", float(sigma_outputs.mean()), float(sigma_outputs.var()), float(np.mean(sigma_outputs < threshold)), 2)
    for order in protocol["gauss_hermite_orders"]:
        locations, weights = np.polynomial.hermite.hermgauss(order)
        phases = math.sqrt(2) * sigma * locations
        weights = weights / math.sqrt(math.pi)
        outputs = radius * np.cos(phases)
        mean = float(weights @ outputs)
        variance = float(weights @ ((outputs - mean) ** 2))
        add(f"Gauss-Hermite-{order}", mean, variance, float(weights @ (outputs < threshold)), order)
    add("moment-exact-Gaussian", true_mean, true_variance, gaussian_probability_below(true_mean, true_variance, threshold))
    add("exact-circular-shared-phase", true_mean, true_variance, probability.upper)

    count = protocol["monte_carlo_samples"]
    random = np.random.default_rng(protocol["monte_carlo_seed"])
    actual = radius * np.cos(random.normal(0.0, sigma, count))
    failures = int(np.count_nonzero(actual < threshold))
    control = {
        "radius_mm": radius * 1000,
        "phase_stddev_rad": sigma,
        "violation_below_mm": threshold * 1000,
        "illustrative_risk_budget": risk_budget,
        "exact_mean_mm": true_mean * 1000,
        "exact_stddev_mm": math.sqrt(true_variance) * 1000,
        "exact_violation_probability": probability.lower,
        "omitted_normal_tail_bound": probability.omitted_tail_bound,
        "analytic_admission": bounded_risk_admissible(probability, maximum_risk=risk_budget),
        "monte_carlo": {
            "independent_controlled_phase_draws": count,
            "seed": protocol["monte_carlo_seed"],
            "violation_count": failures,
            "violation_fraction": failures / count,
            "wilson_95_interval": wilson_interval(failures, count),
            "mean_mm": float(actual.mean()) * 1000,
            "stddev_mm": float(actual.std(ddof=0)) * 1000,
            "standard_error_mean_mm": float(actual.std(ddof=1)) / math.sqrt(count) * 1000,
        },
    }

    uniform = CircularPrior.uniform(prior_id="explicit-control-uniform-shared-phase-v1")
    constraint_count = protocol["shared_phase_control"]["constraints"]
    half_width = math.pi * protocol["shared_phase_control"]["single_constraint_probability"]
    cosine_threshold = math.cos(half_width)
    repeated = AffineCircularQuery(np.full(constraint_count, -cosine_threshold), np.ones(constraint_count), np.zeros(constraint_count))
    centers = np.arange(constraint_count) * 2 * math.pi / constraint_count
    disjoint = AffineCircularQuery(np.full(constraint_count, -cosine_threshold), np.cos(centers), np.sin(centers))
    true_repeated = path_violation_probability(repeated, uniform).lower
    true_disjoint = path_violation_probability(disjoint, uniform).lower
    independent = 1 - (1 - protocol["shared_phase_control"]["single_constraint_probability"]) ** constraint_count
    uniform_draws = np.random.default_rng(protocol["uniform_dependence_seed"]).uniform(0, 2 * math.pi, count)
    dependence = []
    for name, query, truth, budget in [
        ("five-identical-constraints", repeated, true_repeated, protocol["shared_phase_control"]["repeated_constraint_budget"]),
        ("five-disjoint-phase-constraints", disjoint, true_disjoint, protocol["shared_phase_control"]["disjoint_constraint_budget"]),
    ]:
        observed = int(np.count_nonzero(np.any(query.evaluate(uniform_draws) > 0, axis=1)))
        dependence.append({
            "case": name,
            "constraint_count": constraint_count,
            "single_constraint_probability": protocol["shared_phase_control"]["single_constraint_probability"],
            "exact_joint_probability": truth,
            "incorrect_independent_probability": independent,
            "illustrative_risk_budget": budget,
            "exact_admission": truth <= budget,
            "incorrect_independent_admission": independent <= budget,
            "monte_carlo_violation_count": observed,
            "monte_carlo_violation_fraction": observed / count,
            "monte_carlo_wilson_95_interval": wilson_interval(observed, count),
        })
    return {
        "schema": "prob4d.circular-gauge-query-study-result",
        "version": 1,
        "evidence_class": protocol["evidence_class"],
        "date": protocol["date"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "baseline_implementation_boundary": "Independent implementations of mathematical approximation rules; not an execution of unmodified Prob4D, BayesianPhysTwin, or Causal4D pipelines.",
        "information_boundary": protocol["information_boundary"],
        "analytic_control": control,
        "method_results": rows,
        "shared_phase_controls": dependence,
        "interpretation": {
            "positive": "Exact conditional circular moments recover nonlinear radial uncertainty missed by the specified first-order and two-point rules. Shared-phase arc integration recovers joint event risk without assuming independent frames.",
            "important_control": "High-order Gauss-Hermite recovers smooth moments. Its finite-node indicator probabilities need not converge monotonically. Exact first and second moments alone do not determine an event probability.",
            "scope": "Declared one-axis rotation at a fixed observable quotient, with an explicit circular prior. Real-provider and complete-dynamics uncertainty are not evaluated.",
            "unsupported_claims": protocol["unsupported_claims"],
        },
    }


def plots(result: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Each figure is distinct; default Matplotlib colors and styling are used.
    sigma = np.linspace(0.02, 2.5, 200)
    std = 100 / math.sqrt(2) * (-np.expm1(-sigma**2))
    fig, axis = plt.subplots(figsize=(7.5, 4.7))
    axis.plot(sigma, std, label="Exact circular radial standard deviation")
    axis.plot(sigma, np.zeros_like(sigma), linestyle="--", label="First-order and two-point rules")
    axis.set(xlabel="Unobserved twist standard deviation (rad)", ylabel="Radial standard deviation (mm)", title="A one-dimensional gauge can create two covariance modes")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "radial_uncertainty.png", dpi=180)
    plt.close(fig)

    rows = [row for row in result["method_results"] if row["method"].startswith("Gauss-Hermite")]
    fig, axis = plt.subplots(figsize=(7.5, 4.7))
    axis.plot([row["quadrature_order"] for row in rows], [100 * row["violation_probability"] for row in rows], marker="o", label="Gauss-Hermite indicator integration")
    axis.axhline(100 * result["analytic_control"]["exact_violation_probability"], linestyle="--", label="Exact circular arc probability")
    axis.set_xscale("log", base=2)
    axis.set_xticks([row["quadrature_order"] for row in rows], [str(row["quadrature_order"]) for row in rows])
    axis.set(xlabel="Quadrature nodes", ylabel="Predicted violation probability (%)", title="Accurate moments do not guarantee accurate tail events")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "event_probability_quadrature.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.plots and any((output / name).exists() for name in ["radial_uncertainty.png", "event_probability_quadrature.png"]):
        raise FileExistsError("plots require an output directory without existing figures")
    result = run()
    write_new(output / "result.json", json_bytes(result))
    write_new(output / "protocol.json", PROTOCOL.read_bytes())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(result["method_results"][0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(result["method_results"])
    write_new(output / "method_results.csv", buffer.getvalue().encode())
    if arguments.plots:
        plots(result, output)
    files = [
        ROOT / "src/prob4d/circular_gauge_query.py",
        ROOT / "tests/test_circular_gauge_query.py",
        Path(__file__).resolve(),
        PROTOCOL,
    ]
    manifest = {
        "schema": "prob4d.circular-gauge-query-evidence-manifest",
        "version": 1,
        "reviewed_prob4d_revision": "224d13dc9a93731ac5297b479eb1e121b3dbe659",
        "publication_state": "local-unpublished-additive-patch",
        "source_files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "output_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"},
        "command": "PYTHONPATH=src python scripts/science/circular_gauge_query_study.py --output-dir <new-directory>" + (" --plots" if arguments.plots else ""),
        "real_data_or_models_accessed": False,
    }
    write_new(output / "manifest.json", json_bytes(manifest))
    print(json.dumps(result["analytic_control"], indent=2))
    print(json.dumps(result["shared_phase_controls"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
