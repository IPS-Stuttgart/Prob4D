"""Controlled nonlinear-gauge query study with continuous independent truth.

Run with ``python -m prob4d.axial_gauge_study --output result.json
--source-revision <40-hex-commit>``. No dataset, checkpoint, or downstream outcome
is read. The fixed simulation protocol is included in the output with its hash.
This is a conditional mechanism study, not real-provider calibration evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .axial_gauge import AxialGaugeOrbit, CircularQuadrature, GaussianQueryMixture

PROTOCOL: dict[str, Any] = {
    "schema": "prob4d.axial-gauge-query-control-v1",
    "evidence_kind": "controlled-conditional-mechanism",
    "seeds": [20260830, 20260831],
    "independent_draws_per_regime_per_seed": 16384,
    "angular_nodes": 512,
    "refinement_nodes": 1024,
    "refinement_observations": 512,
    "radius_mm": 50.0,
    "readout_std_mm": 3.0,
    "event": "noisy reference-frame query coordinate y > 0 mm",
    "regimes": [
        {"name": "narrow", "kind": "wrapped-normal", "std_rad": 0.05},
        {"name": "moderate", "kind": "wrapped-normal", "std_rad": 0.6},
        {"name": "broad", "kind": "wrapped-normal", "std_rad": 1.2},
        {"name": "uniform", "kind": "uniform"},
        {"name": "threefold", "kind": "threefold", "std_rad": 0.12},
    ],
    "arms": ["tangent-gaussian", "exact-moment-gaussian", "axial-orbit-mixture"],
    "truth": "continuous angular draw plus independent isotropic Gaussian readout noise",
    "statistical_unit": "one independent gauge-and-readout draw; not coordinates or atoms",
    "paired_interval": "mean +/- 1.96 sample-standard-error; Monte Carlo interval only",
    "closed_boundaries": ["real provider", "BayesianPhysTwin", "Causal4D", "protected cohorts"],
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def angular_rule(regime: dict[str, Any], nodes: int) -> CircularQuadrature:
    """Periodic trapezoidal quadrature, independent of all simulated outcomes."""
    theta = -np.pi + (np.arange(nodes, dtype=np.float64) + 0.5) * (2.0 * np.pi / nodes)
    if regime["kind"] == "uniform":
        weights = np.ones(nodes)
    else:
        std = float(regime["std_rad"])
        centers = (
            np.array([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0])
            if regime["kind"] == "threefold"
            else np.array([0.0])
        )
        weights = np.zeros(nodes)
        # Five images cover more than 8 standard deviations in the broadest regime.
        for center in centers:
            for winding in range(-2, 3):
                weights += np.exp(-0.5 * ((theta - center + winding * 2.0 * np.pi) / std) ** 2)
    return CircularQuadrature(theta, weights)


def continuous_truth(
    regime: dict[str, Any], rng: np.random.Generator, count: int
) -> np.ndarray:
    """Independent continuous simulator; no quadrature atoms or core rotations used."""
    if regime["kind"] == "uniform":
        theta = rng.uniform(-np.pi, np.pi, count)
    elif regime["kind"] == "threefold":
        centers = np.array([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0])
        theta = centers[rng.integers(0, 3, count)] + rng.normal(0, regime["std_rad"], count)
    else:
        theta = rng.normal(0, regime["std_rad"], count)
    radius = PROTOCOL["radius_mm"]
    positions = np.column_stack((np.zeros(count), radius * np.cos(theta), radius * np.sin(theta)))
    return positions + rng.normal(0, PROTOCOL["readout_std_mm"], (count, 3))


def normal_interval(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    error = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return {"mean": mean, "lower": mean - error, "upper": mean + error}


def run_study(source_revision: str) -> dict[str, Any]:
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise ValueError("source_revision must be a full lowercase hexadecimal Git commit")
    orbit = AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    query = np.array([[0.0, PROTOCOL["radius_mm"], 0.0]])
    noise = np.eye(3) * PROTOCOL["readout_std_mm"] ** 2
    event_normal = np.array([0.0, 1.0, 0.0])
    records: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(PROTOCOL["regimes"]):
        rule = angular_rule(regime, PROTOCOL["angular_nodes"])
        nonlinear = orbit.pushforward(query, rule, noise_covariance=noise)
        mean, covariance = orbit.moments(query, rule)
        moment_gaussian = GaussianQueryMixture(mean.reshape(1, 3), np.ones(1), covariance + noise)
        if regime["kind"] == "uniform":
            angle_variance = np.pi**2 / 3.0
        elif regime["kind"] == "threefold":
            angle_variance = regime["std_rad"] ** 2 + 8.0 * np.pi**2 / 27.0
        else:
            angle_variance = regime["std_rad"] ** 2
        tangent_covariance = noise.copy()
        tangent_covariance[2, 2] += PROTOCOL["radius_mm"] ** 2 * angle_variance
        tangent = GaussianQueryMixture(query, np.ones(1), tangent_covariance)
        methods = {
            "tangent-gaussian": tangent,
            "exact-moment-gaussian": moment_gaussian,
            "axial-orbit-mixture": nonlinear,
        }
        refinement = orbit.pushforward(
            query, angular_rule(regime, PROTOCOL["refinement_nodes"]), noise_covariance=noise
        )
        log_scores: dict[str, list[np.ndarray]] = {name: [] for name in methods}
        brier_scores: dict[str, list[np.ndarray]] = {name: [] for name in methods}
        errors: dict[str, list[np.ndarray]] = {name: [] for name in methods}
        event_values: list[np.ndarray] = []
        refinements: list[float] = []
        seed_records: list[dict[str, Any]] = []
        for seed in PROTOCOL["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, regime_index]))
            truth = continuous_truth(regime, rng, PROTOCOL["independent_draws_per_regime_per_seed"])
            event = (truth[:, 1] > 0.0).astype(np.float64)
            event_values.append(event)
            seed_metrics: dict[str, Any] = {"seed": seed, "arms": {}}
            for name, method in methods.items():
                nll = -method.logpdf(truth)
                probability = method.halfspace_probability(event_normal, 0.0)
                brier = (probability - event) ** 2
                squared_error = np.sum((truth - method.mean) ** 2, axis=1)
                log_scores[name].append(nll)
                brier_scores[name].append(brier)
                errors[name].append(squared_error)
                seed_metrics["arms"][name] = {
                    "nll_nats": float(np.mean(nll)),
                    "brier": float(np.mean(brier)),
                    "position_rmse_mm": float(np.sqrt(np.mean(squared_error))),
                }
            prefix = truth[: PROTOCOL["refinement_observations"]]
            refinements.append(
                float(np.max(np.abs(nonlinear.logpdf(prefix) - refinement.logpdf(prefix))))
            )
            seed_records.append(seed_metrics)
        combined_nll = {name: np.concatenate(values) for name, values in log_scores.items()}
        combined_brier = {name: np.concatenate(values) for name, values in brier_scores.items()}
        record: dict[str, Any] = {
            "regime": regime["name"],
            "independent_draws": int(sum(values.size for values in event_values)),
            "observed_event_frequency": float(np.mean(np.concatenate(event_values))),
            "quadrature_refinement_max_abs_nll_nats": max(refinements),
            "same_mean_max_abs_mm": float(np.max(np.abs(nonlinear.mean - moment_gaussian.mean))),
            "same_covariance_max_abs_mm2": float(
                np.max(np.abs(nonlinear.covariance - moment_gaussian.covariance))
            ),
            "arms": {},
            "orbit_minus_moment_gaussian": {
                "paired_nll_nats": normal_interval(
                    combined_nll["axial-orbit-mixture"] - combined_nll["exact-moment-gaussian"]
                ),
                "paired_brier": normal_interval(
                    combined_brier["axial-orbit-mixture"] - combined_brier["exact-moment-gaussian"]
                ),
            },
            "per_seed": seed_records,
        }
        for name, method in methods.items():
            record["arms"][name] = {
                "nll_nats": float(np.mean(combined_nll[name])),
                "brier": float(np.mean(combined_brier[name])),
                "event_probability": method.halfspace_probability(event_normal, 0.0),
                "position_rmse_mm": float(np.sqrt(np.mean(np.concatenate(errors[name])))),
            }
        records.append(record)
    source_files = [Path(__file__), Path(__file__).with_name("axial_gauge.py")]
    return {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(canonical_json(PROTOCOL).encode()).hexdigest(),
        "source_revision": source_revision,
        "source_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "records": records,
        "claim_boundary": (
            "Conditional controlled mechanism evidence with known angular laws and readout noise. "
            "Not an end-to-end gauge estimator, real-provider result, "
            "empirical calibration guarantee, "
            "physical-state correction, intervention benefit, or deployment-safety result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing evidence")
    result = run_study(args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "protocol_sha256": result["protocol_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
