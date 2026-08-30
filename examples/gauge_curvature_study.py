"""Reproduce controlled shared-curvature evidence; never opens a dataset.

Run after installing the patched Prob4D checkout:
    python examples/gauge_curvature_study.py --output-dir outputs/gauge-curvature-v1

All cases and comparisons are fixed below. Results are local synthetic analysis,
not provider calibration, BPT guard execution, or Causal4D intervention evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np

import prob4d.gauge_curvature as curvature_module
import prob4d.sim3 as sim3_module
from prob4d.gauge_curvature import finite_difference_gauge_moments, sim3_chain_gauge_moments

SIGMAS = (0.025, 0.05, 0.1, 0.2, 0.35)
LEVERS_M = (0.25, 0.5, 1.0)
POINT_COUNTS = (1, 16, 64, 256)
PRIOR_SDS_M = (0.002, 0.005, 0.01)
POINT_SD_M = 0.001
STEPS = (0.01, 0.001, 0.0003)
MONTE_CARLO_SEED = 20260830
MONTE_CARLO_EPISODES = 131072
UPSTREAM_PROB4D_COMMIT = "224d13dc9a93731ac5297b479eb1e121b3dbe659"
UPSTREAM_SIM3_BLOB = "dc84a889c1aecb6b7d7c16ad83604861744b995d"


def exact_sine_product_variance(sigma: float, lever: float) -> float:
    """Independent analytic Gaussian characteristic-function result."""
    return lever**2 * (-math.expm1(-2.0 * sigma**2) / 2.0)**2


def axis_cubature(function, rank: int) -> tuple[float, float]:
    samples = np.array([
        function(sign * math.sqrt(rank) * np.eye(rank)[i])
        for i in range(rank)
        for sign in (-1.0, 1.0)
    ])
    return float(samples.mean()), float(np.mean((samples - samples.mean())**2))


def fifth_degree_cubature(function, rank: int) -> tuple[float, float]:
    """Classical fully symmetric degree-five Gaussian rule, 2*r*r+1 nodes.

    The axial weights become negative for r > 4. This is an established
    comparator, not the proposed method. Do not clip a negative variance.
    """
    values, weights = [function(np.zeros(rank))], [2.0 / (rank + 2.0)]
    for i in range(rank):
        for sign in (-1.0, 1.0):
            values.append(function(sign * math.sqrt(rank + 2.0) * np.eye(rank)[i]))
            weights.append((4.0 - rank) / (2.0 * (rank + 2.0)**2))
    for i in range(rank):
        for j in range(i + 1, rank):
            for si, sj in itertools.product((-1.0, 1.0), repeat=2):
                node = math.sqrt((rank + 2.0) / 2.0) * (si * np.eye(rank)[i] + sj * np.eye(rank)[j])
                values.append(function(node))
                weights.append(1.0 / (rank + 2.0)**2)
    values, weights = np.asarray(values), np.asarray(weights)
    mean = float(weights @ values)
    return mean, float(weights @ ((values - mean)**2))


def gauss_hermite_product(function, rank: int, order: int) -> tuple[float, float]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, weights = math.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)
    values, masses = [], []
    for indices in itertools.product(range(order), repeat=rank):
        values.append(function(nodes[list(indices)]))
        masses.append(float(np.prod(weights[list(indices)])))
    values, masses = np.asarray(values), np.asarray(masses)
    mean = float(masses @ values)
    return mean, float(masses @ ((values - mean)**2))


def curvature_case(sigma: float, lever: float, step: float):
    root = np.zeros((14, 2))
    root[3, 0] = sigma
    root[9, 1] = sigma
    return sim3_chain_gauge_moments(
        np.zeros((2, 7)), root, [[0, 0, lever]],
        query_matrix=[[0, 1, 0]], step=step,
    )


def gaussian_update_metrics(
    prior_variance: float, actual_noise: float, assumed_noise: float | None,
):
    """Exact expected LMMSE risk using moments, despite non-Gaussian noise."""
    if assumed_noise is None:
        gain, posterior_variance = 0.0, prior_variance
    else:
        gain = prior_variance / (prior_variance + assumed_noise)
        posterior_variance = prior_variance * assumed_noise / (prior_variance + assumed_noise)
    mse = (1.0 - gain)**2 * prior_variance + gain**2 * actual_noise
    return {
        "gain": gain,
        "posterior_variance_m2": posterior_variance,
        "expected_mse_m2": mse,
        "expected_rmse_mm": 1000.0 * math.sqrt(mse),
        "expected_nees": mse / posterior_variance,
        "expected_gaussian_nll_nats_in_meters": 0.5 * (
            math.log(2.0 * math.pi * posterior_variance) + mse / posterior_variance
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    if Path(curvature_module.__file__).resolve() != root / "src/prob4d/gauge_curvature.py":
        raise ValueError("imported curvature module is not from this checkout")
    if Path(sim3_module.__file__).resolve() != root / "src/prob4d/sim3.py":
        raise ValueError("imported Sim3 module is not from this checkout")
    sim3_bytes = (root / "src/prob4d/sim3.py").read_bytes()
    sim3_blob = hashlib.sha1(
        f"blob {len(sim3_bytes)}\0".encode() + sim3_bytes, usedforsecurity=False,
    ).hexdigest()
    if sim3_blob != UPSTREAM_SIM3_BLOB:
        raise ValueError("Sim3 source differs from the pinned upstream validation blob")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; retained results are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_rows, update_rows = [], []
    numerical_step_relative_variance_spreads = []
    for sigma, lever in itertools.product(SIGMAS, LEVERS_M):
        truth = exact_sine_product_variance(sigma, lever)
        def query(z, sigma=sigma, lever=lever):
            return lever * math.sin(sigma * z[0]) * math.sin(sigma * z[1])
        curvatures = [curvature_case(sigma, lever, step) for step in STEPS]
        corrected = float(curvatures[1].marginal_variance[0])
        step_vars = [float(c.marginal_variance[0]) for c in curvatures]
        numerical_step_relative_variance_spreads.append(
            (max(step_vars) - min(step_vars)) / corrected
        )
        axis_mean, axis_variance = axis_cubature(query, 2)
        fifth_mean, fifth_variance = fifth_degree_cubature(query, 2)
        gh3_mean, gh3_variance = gauss_hermite_product(query, 2, 3)
        gh15_mean, gh15_variance = gauss_hermite_product(query, 2, 15)
        methods = {
            "first_order": (0.0, 0.0, 1),
            "spherical_radial": (axis_mean, axis_variance, 4),
            "shared_curvature": (float(curvatures[1].mean[0]), corrected, 9),
            "fifth_degree": (fifth_mean, fifth_variance, 9),
            "gauss_hermite3": (gh3_mean, gh3_variance, 9),
            "gauss_hermite15_reference": (gh15_mean, gh15_variance, 225),
        }
        for method, (mean, variance, calls) in methods.items():
            transform_rows.append({
                "sigma_rad": sigma, "lever_m": lever, "method": method,
                "mean_m": mean, "variance_m2": variance,
                "sd_mm": 1000.0 * math.sqrt(variance),
                "exact_variance_m2": truth,
                "exact_sd_mm": 1000.0 * math.sqrt(truth),
                "relative_variance_error": variance / truth - 1.0,
                "forward_evaluation_count": calls,
            })
        for count, prior_sd in itertools.product(POINT_COUNTS, PRIOR_SDS_M):
            conditional_average_var = POINT_SD_M**2 / count
            noises = {
                "physical_fallback": None,
                "first_order": conditional_average_var,
                "spherical_radial": conditional_average_var + axis_variance,
                "pointwise_curvature": (POINT_SD_M**2 + corrected) / count,
                "shared_curvature": conditional_average_var + corrected,
                "fifth_degree_shared": conditional_average_var + fifth_variance,
                "gauss_hermite3_shared": conditional_average_var + gh3_variance,
                "oracle_second_moments_shared": conditional_average_var + truth,
            }
            for method, assumed_noise in noises.items():
                update_rows.append({
                    "sigma_rad": sigma, "lever_m": lever,
                    "point_count": count, "prior_sd_mm": 1000.0 * prior_sd,
                    "method": method,
                    **gaussian_update_metrics(
                        prior_sd**2, truth + conditional_average_var, assumed_noise,
                    ),
                })
    # Independent synthetic episodes check coverage. It is not deduced from NEES.
    headline = [r for r in update_rows if r["sigma_rad"] == 0.1 and r["lever_m"] == 0.5
                and r["point_count"] == 64 and r["prior_sd_mm"] == 2.0]
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    alpha, beta, state, averaged_noise = rng.normal(size=(4, MONTE_CARLO_EPISODES))
    state *= 0.002
    observation = state + 0.5 * np.sin(0.1 * alpha) * np.sin(0.1 * beta)
    observation += POINT_SD_M / math.sqrt(64) * averaged_noise
    coverage_rows = []
    normal_90 = 1.6448536269514722
    for row in headline:
        error = row["gain"] * observation - state
        posterior_sd = math.sqrt(row["posterior_variance_m2"])
        covered = np.abs(error) <= normal_90 * posterior_sd
        coverage = float(covered.mean())
        coverage_rows.append({
            "method": row["method"], "synthetic_episode_count": MONTE_CARLO_EPISODES,
            "empirical_rmse_mm": 1000.0 * float(np.sqrt(np.mean(error**2))),
            "nominal_90_coverage": coverage,
            "coverage_binomial_standard_error": math.sqrt(
                coverage * (1.0 - coverage) / MONTE_CARLO_EPISODES
            ),
            "interval_full_width_mm": 2000.0 * normal_90 * posterior_sd,
        })
    # Adversarial quadratic q=z0*z1 under 32 arbitrary whitening bases.
    basis_rows = []
    basis_rng = np.random.default_rng(64393)
    rank = 7
    rotations = [np.eye(rank)]
    rotations += [np.linalg.qr(basis_rng.normal(size=(rank, rank)))[0] for _ in range(31)]
    for index, rotation in enumerate(rotations):
        def function(z, rotation=rotation):
            rotated = rotation @ z
            return float(rotated[0] * rotated[1])
        _, axis_variance = axis_cubature(function, rank)
        _, fifth_variance = fifth_degree_cubature(function, rank)
        curvature = finite_difference_gauge_moments(
            lambda x: np.array([x[0] * x[1]]), np.zeros(rank), rotation, step=0.01,
        )
        basis_rows.append({
            "basis_index": index, "rank": rank, "true_variance": 1.0,
            "axis_variance": axis_variance,
            "fifth_degree_variance": fifth_variance,
            "shared_curvature_variance": float(curvature.marginal_variance[0]),
        })
    by_method = {row["method"]: row for row in headline}
    result = {
        "classification": "local-controlled-method-study-not-real-provider-evidence",
        "date": "2026-08-30",
        "target_outcomes_used": False,
        "public_datasets_opened": False,
        "bpt_guard_executed": False,
        "causal4d_pipeline_executed": False,
        "upstream_prob4d_commit": UPSTREAM_PROB4D_COMMIT,
        "upstream_sim3_git_blob": UPSTREAM_SIM3_BLOB,
        "upstream_sim3_blob_verified": True,
        "candidate_commit": None,
        "candidate_identity": "uncommitted-additive-patch; SHA256 source identities in manifest",
        "design": {
            "sigmas_rad": SIGMAS, "levers_m": LEVERS_M, "point_counts": POINT_COUNTS,
            "prior_sds_m": PRIOR_SDS_M, "point_sd_m": POINT_SD_M,
            "finite_difference_steps": STEPS,
            "primary_method_step": 0.001,
            "transform_case_count": len(SIGMAS) * len(LEVERS_M),
            "update_case_count": len(SIGMAS) * len(LEVERS_M) * len(POINT_COUNTS) * len(PRIOR_SDS_M),
            "update_arms": 8,
            "monte_carlo_seed": MONTE_CARLO_SEED,
            "monte_carlo_episodes": MONTE_CARLO_EPISODES,
            "selection": (
                "fixed illustrative geometry and grid; exploratory, "
                "not an independently preregistered claim"
            ),
        },
        "headline_transform": [
            r for r in transform_rows if r["sigma_rad"] == 0.1 and r["lever_m"] == 0.5
        ],
        "headline_update": headline,
        "headline_coverage": coverage_rows,
        "headline_rmse_reduction_vs_first_order": (
            1.0 - by_method["shared_curvature"]["expected_rmse_mm"]
            / by_method["first_order"]["expected_rmse_mm"]
        ),
        "headline_rmse_reduction_vs_physical_fallback": (
            1.0 - by_method["shared_curvature"]["expected_rmse_mm"]
            / by_method["physical_fallback"]["expected_rmse_mm"]
        ),
        "max_gh15_relative_variance_error": max(
            abs(r["relative_variance_error"]) for r in transform_rows
            if r["method"] == "gauss_hermite15_reference"
        ),
        "max_step_relative_variance_spread": max(numerical_step_relative_variance_spreads),
        "axis_basis_variance_range": [
            min(r["axis_variance"] for r in basis_rows),
            max(r["axis_variance"] for r in basis_rows),
        ],
        "max_curvature_basis_variance_error": max(
            abs(r["shared_curvature_variance"] - 1.0) for r in basis_rows
        ),
        "max_fifth_degree_basis_variance_error": max(
            abs(r["fifth_degree_variance"] - 1.0) for r in basis_rows
        ),
        "evaluation_count_scaling_not_walltime": [
            {
                "rank": r, "axis": 2*r, "curvature": 1+2*r*r,
                "fifth_degree": 1+2*r*r, "tensor_gh3": 3**r,
            }
            for r in (2, 4, 7, 14, 28)
        ],
        "limits": [
            (
                "Quadratic Gaussian moments and fifth-degree cubature are established methods, "
                "not claimed as new."
            ),
            (
                "The contribution candidate is joint curvature support for tangent-null "
                "physical queries and its downstream consequence."
            ),
            (
                "The Taylor closure is not exact for general nonlinear maps and is not "
                "uniformly covariance-order-two accurate."
            ),
            (
                "At rank two, three-node tensor Gauss-Hermite has the same evaluation count "
                "and is more accurate on this sine example."
            ),
            (
                "At general rank, classical fifth-degree cubature also has quadratic node "
                "count; no novel complexity order is claimed."
            ),
            (
                "Shared factors describe moments only; Gaussian interval coverage need not "
                "equal nominal coverage."
            ),
            (
                "This is an analytic scalar physical-update analogue, not execution of "
                "BayesianPhysTwin or Causal4D."
            ),
            (
                "No existing terminal result, claim registry, provider status, or sealed "
                "data-access boundary has changed."
            ),
        ],
    }
    write_csv(output_dir / "transform_results.csv", transform_rows)
    write_csv(output_dir / "update_results.csv", update_rows)
    write_csv(output_dir / "coverage_results.csv", coverage_rows)
    write_csv(output_dir / "whitening_basis_results.csv", basis_rows)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    root = Path(__file__).resolve().parents[1]
    code_paths = [
        root / "src/prob4d/gauge_curvature.py",
        root / "src/prob4d/sim3.py",
        root / "tests/test_gauge_curvature.py",
        root / "tests/test_gauge_curvature_study.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "upstream_prob4d_commit": UPSTREAM_PROB4D_COMMIT,
        "candidate_commit": None,
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code_paths
        },
        "outputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output_dir.iterdir()) if p.is_file()
        },
        "reproduce": (
            "python examples/gauge_curvature_study.py "
            "--output-dir outputs/gauge-curvature-v1"
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({
        "classification": result["classification"],
        "update_case_count": result["design"]["update_case_count"],
        "headline_rmse_reduction_vs_first_order": result["headline_rmse_reduction_vs_first_order"],
        "headline_rmse_reduction_vs_physical_fallback": (
            result["headline_rmse_reduction_vs_physical_fallback"]
        ),
        "max_gh15_relative_variance_error": result["max_gh15_relative_variance_error"],
        "max_step_relative_variance_spread": result["max_step_relative_variance_spread"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
