"""Frozen, CPU-only local-Gaussian mechanism study; no provider/data access.

Execute only after committing the protocol and implementation. Raw inputs,
proposals, decisions and seed-level scores are retained. Query tolerances express
model-conditional precision, not a frequentist safety or no-harm guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

from prob4d.query_information_audit import QueryInformationPolicy, assess_query_update

ARMS = (
    "physical_fallback", "reject_rank_deficient", "point_transform",
    "pseudoinverse_covariance", "ridge_completion", "exact_subspace",
    "query_without_audit", "audited_query",
)
PROPOSALS = ("exact", "nullspace_precision", "nullspace_mean", "repeated_information")
WORLDS = ("matched_prior", "shifted_prior")
METRICS = ("squared_error", "nll", "coverage90", "width90", "accepted", "harmful_accepted")
CHI2_90_3 = 6.251388631170325
NORMAL_95 = 1.6448536269514722


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def point_jacobian(point: np.ndarray, scale: float) -> np.ndarray:
    x, y, z = point
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.column_stack((point, -cross, scale * np.eye(3)))


def geometry(thickness: float, count: int, extent: float) -> tuple[np.ndarray, float]:
    parameter = np.linspace(-1.0, 1.0, count)
    cloud = extent * np.column_stack((
        parameter, thickness * np.sin(np.pi * parameter),
        0.5 * thickness * np.cos(2.0 * np.pi * parameter),
    ))
    cloud -= cloud.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(cloud**2, axis=1))))
    return np.vstack([point_jacobian(point, scale) for point in cloud]), scale


def score(
    means: np.ndarray, covariance: np.ndarray | None, truths: np.ndarray,
    jacobian: np.ndarray, prior_errors: np.ndarray, accepted: np.ndarray,
) -> np.ndarray:
    error = (means - truths) @ jacobian.T
    squared = np.sum(error**2, axis=1)
    result = np.zeros((len(truths), len(METRICS)))
    result[:, 0] = squared
    result[:, 4] = accepted
    result[:, 5] = accepted & (squared > prior_errors + 1e-12)
    if covariance is None:
        result[:, 1:4] = np.nan  # Missing uncertainty is not repaired by an epsilon.
        return result
    projected = jacobian @ covariance @ jacobian.T
    projected = 0.5 * (projected + projected.T)
    sign, logdet = np.linalg.slogdet(projected)
    if sign <= 0 or np.linalg.eigvalsh(projected)[0] <= 0:
        raise ValueError("registered point query must have nonsingular predictive covariance")
    mahalanobis = np.sum(error * np.linalg.solve(projected, error.T).T, axis=1)
    result[:, 1] = 0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + mahalanobis)
    result[:, 2] = mahalanobis <= CHI2_90_3
    result[:, 3] = 2.0 * NORMAL_95 * np.sqrt(np.diag(projected)).mean()
    return result


def estimate(values: np.ndarray, bootstrap: np.ndarray, *, root: bool = False) -> dict:
    if np.any(~np.isfinite(values)):
        return {"value": None, "interval95": None}
    distribution = values[bootstrap].mean(axis=1)
    central = float(values.mean())
    if root:
        central = float(np.sqrt(max(central, 0.0)))
        distribution = np.sqrt(np.maximum(distribution, 0.0))
    return {"value": central, "interval95": np.quantile(distribution, [0.025, 0.975]).tolist()}


def run(protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != "prob4d.query-information-audit-study.v1":
        raise ValueError("unsupported protocol")
    if output.exists():
        raise FileExistsError("refusing to overwrite a retained study directory")
    output.mkdir(parents=True)
    json_write(output / "protocol.json", protocol)
    start = time.perf_counter()
    policy = QueryInformationPolicy(**protocol["policy"])
    settings = list(itertools.product(
        protocol["thickness_ratios"], protocol["point_counts"],
        protocol["noise_std_m"], protocol["prior_std_multipliers"],
        protocol["shared_window_correlations"],
    ))
    seeds = np.arange(protocol["seed_start"], protocol["seed_start"] + protocol["seed_count"])
    n, s = len(settings), len(seeds)
    # Metric accumulation preserves the complete seed as statistical unit.
    sums = np.zeros((2, 4, 8, s, 6))
    rank_sums = np.zeros((2, 4, 2, 8, s, 6))
    rank_counts = np.zeros(2, dtype=int)
    curves = np.zeros((2, 4, len(protocol["tolerance_grid_m"]), s, 3))
    raw_truth = np.empty((2, n, s, 7))
    raw_natural = np.empty_like(raw_truth)
    raw_means = np.empty((2, 4, n, s, 7))
    raw_covariances = np.empty((4, n, 7, 7))
    raw_prior = np.empty((n, 7, 7))
    raw_information = np.empty_like(raw_prior)
    raw_jacobians = np.empty((n, 3, 3, 7))
    raw_admitted = np.empty((2, 4, n, 3, s), dtype=bool)
    raw_valid = np.empty((2, 4, n, s), dtype=bool)
    route_counts = {}
    configurations = []
    prior_mean = np.asarray(protocol["prior_mean_local"])
    query_points = np.asarray(protocol["query_points_m"])
    for config_index, (thickness, count, noise, strength, correlation) in enumerate(settings):
        design, scale = geometry(thickness, count, protocol["cloud_half_extent_m"])
        spectrum, basis = np.linalg.eigh(design.T @ design)
        retained = spectrum > spectrum[-1] * protocol["geometry_rank_rtol"]
        observed = basis[:, retained]
        nullspace = basis[:, ~retained]
        rank = int(retained.sum())
        rank_index = int(rank == 7)
        rank_counts[rank_index] += len(query_points)
        variance = noise**2 * (
            correlation + (1.0 - correlation) / protocol["windows"]
        )
        precision = spectrum[retained] / variance
        information = (observed * precision) @ observed.T
        pseudo = (observed / precision) @ observed.T
        prior = np.diag((strength * np.asarray(protocol["prior_std_local"]))**2)
        prior_inverse = np.diag(1.0 / np.diag(prior))
        exact_cov = np.linalg.solve(prior_inverse + information, np.eye(7))
        fabricated = float(np.median(precision)) * (nullspace @ nullspace.T)
        ridge_cov = np.linalg.solve(prior_inverse + information + fabricated, np.eye(7))
        repeated_scale = 1.0 + (protocol["windows"] - 1.0) * correlation
        repeated_cov = np.linalg.solve(prior_inverse + repeated_scale * information, np.eye(7))
        shift = nullspace @ (nullspace.T @ np.eye(7)[:, 1])
        if np.linalg.norm(shift) > 1e-12:
            shift *= protocol["unsupported_mean_shift_rad"] / np.linalg.norm(shift)
        else:
            shift[:] = 0.0
        jacobians = np.asarray([point_jacobian(point, scale) for point in query_points])
        raw_prior[config_index], raw_information[config_index] = prior, information
        raw_jacobians[config_index] = jacobians
        covariances = (exact_cov, ridge_cov, exact_cov, repeated_cov)
        raw_covariances[:, config_index] = covariances
        configurations.append({
            "thickness_ratio": thickness, "point_count": count, "noise_std_m": noise,
            "prior_std_multiplier": strength, "shared_window_correlation": correlation,
            "rank": rank, "cloud_scale_m": scale,
        })
        # Every seed is independent; configurations/worlds are paired within it.
        normals = np.asarray([
            np.random.default_rng(np.random.SeedSequence([int(seed), config_index])).normal(
                size=7 + rank
            ) for seed in seeds
        ])
        base_truth = prior_mean + normals[:, :7] @ np.linalg.cholesky(prior).T
        for world_index in range(2):
            truths = base_truth.copy()
            if world_index:
                truths[:, 1] += protocol["stress_prior_shift_rad"]
            natural = truths @ information + (
                normals[:, 7:] * np.sqrt(precision)
            ) @ observed.T
            exact_mean = (natural + prior_inverse @ prior_mean) @ exact_cov.T
            ridge_mean = (natural + prior_inverse @ prior_mean) @ ridge_cov.T
            repeated_mean = (
                repeated_scale * natural + prior_inverse @ prior_mean
            ) @ repeated_cov.T
            proposals = (exact_mean, ridge_mean, exact_mean + shift, repeated_mean)
            raw_truth[world_index, config_index] = truths
            raw_natural[world_index, config_index] = natural
            raw_means[world_index, :, config_index] = proposals
            fallback_mean = np.broadcast_to(prior_mean, (s, 7))
            point_mean = natural @ pseudo.T
            for proposal_index, (proposal_mean, proposal_cov) in enumerate(zip(
                proposals, covariances, strict=True
            )):
                for query_index, jacobian in enumerate(jacobians):
                    # Call the actual proposed gate for every seed, not a surrogate.
                    decisions = [assess_query_update(
                        prior_mean=prior_mean, prior_covariance=prior,
                        candidate_mean=proposal_mean[k], candidate_covariance=proposal_cov,
                        likelihood_information=information,
                        likelihood_natural_parameter=natural[k], query_jacobian=jacobian,
                        query_tolerances=np.full(3, protocol["primary_tolerance_m"]),
                        policy=policy,
                    ) for k in range(s)]
                    valid = np.asarray([decision.audit.valid for decision in decisions])
                    admission = np.asarray([decision.admitted for decision in decisions])
                    covariance_only_admission = np.asarray([
                        decision.maximum_standardized_variance <= 1.0
                        and decision.variance_reduction_fraction
                            >= policy.minimum_variance_reduction
                        for decision in decisions
                    ])
                    raw_valid[world_index, proposal_index, config_index] = valid
                    raw_admitted[world_index, proposal_index, config_index, query_index] = admission
                    for decision in decisions:
                        key = f"{WORLDS[world_index]}/{PROPOSALS[proposal_index]}/{decision.route}"
                        route_counts[key] = route_counts.get(key, 0) + 1
                    prior_errors = np.sum(((fallback_mean - truths) @ jacobian.T)**2, axis=1)
                    fixed = (
                        (fallback_mean, prior, np.zeros(s, dtype=bool)),
                        (exact_mean if rank == 7 else fallback_mean,
                         exact_cov if rank == 7 else prior, np.full(s, rank == 7)),
                        (point_mean, None, np.ones(s, dtype=bool)),
                        (point_mean, pseudo, np.ones(s, dtype=bool)),
                        (ridge_mean, ridge_cov, np.ones(s, dtype=bool)),
                        (exact_mean, exact_cov, np.ones(s, dtype=bool)),
                    )
                    for arm_index, (means, covariance, accepted) in enumerate(fixed):
                        metrics = score(means, covariance, truths, jacobian, prior_errors, accepted)
                        sums[world_index, proposal_index, arm_index] += metrics
                        rank_sums[world_index, proposal_index, rank_index, arm_index] += metrics
                    baseline = score(
                        fallback_mean, prior, truths, jacobian, prior_errors,
                        np.zeros(s, dtype=bool),
                    )
                    candidate = score(
                        proposal_mean, proposal_cov, truths, jacobian, prior_errors,
                        np.ones(s, dtype=bool),
                    )
                    for arm_index, accepted in ((6, covariance_only_admission), (7, admission)):
                        metrics = np.where(accepted[:, None], candidate, baseline)
                        sums[world_index, proposal_index, arm_index] += metrics
                        rank_sums[world_index, proposal_index, rank_index, arm_index] += metrics
                    for grid_index, tolerance in enumerate(protocol["tolerance_grid_m"]):
                        accepted = np.asarray([
                            decision.audit.valid
                            and decision.maximum_standardized_variance * (
                                protocol["primary_tolerance_m"] / tolerance
                            )**2 <= 1.0
                            and decision.variance_reduction_fraction
                            >= policy.minimum_variance_reduction
                            for decision in decisions
                        ])
                        chosen = np.where(accepted[:, None], candidate, baseline)
                        curves[world_index, proposal_index, grid_index] += chosen[:, [0, 4, 5]]
    denominator = n * len(query_points)
    means = sums / denominator
    by_rank = rank_sums / rank_counts[None, None, :, None, None, None]
    curves /= denominator
    bootstrap = np.random.default_rng(protocol["bootstrap_seed"]).integers(
        0, s, size=(protocol["bootstrap_repetitions"], s)
    )
    tables = {}
    rank_tables = {}
    paired = {}
    for wi, world in enumerate(WORLDS):
        for pi, proposal in enumerate(PROPOSALS):
            label = f"{world}/{proposal}"
            tables[label] = {}
            rank_tables[label] = {}
            for ai, arm in enumerate(ARMS):
                tables[label][arm] = {
                    ("rmse_m" if mi == 0 else metric): estimate(
                        means[wi, pi, ai, :, mi], bootstrap, root=(mi == 0)
                    ) for mi, metric in enumerate(METRICS)
                }
            for ri, rank_label in enumerate(("rank_deficient", "full_rank")):
                rank_tables[label][rank_label] = {
                    arm: {
                        ("rmse_m" if mi == 0 else metric): estimate(
                            by_rank[wi, pi, ri, ai, :, mi], bootstrap, root=(mi == 0)
                        ) for mi, metric in enumerate(METRICS)
                    } for ai, arm in enumerate(ARMS)
                }
            paired[label] = {
                "audited_minus_query_without_audit": {
                    metric: estimate(means[wi, pi, 7, :, mi] - means[wi, pi, 6, :, mi], bootstrap)
                    for mi, metric in enumerate(METRICS)
                }
            }
    np.savez_compressed(
        output / "raw_inputs_and_proposals.npz", seeds=seeds, truth=raw_truth,
        natural=raw_natural, proposal_means=raw_means, proposal_covariances=raw_covariances,
        prior_covariances=raw_prior, prior_mean=prior_mean, information=raw_information,
        query_jacobians=raw_jacobians, admitted=raw_admitted, audit_valid=raw_valid,
    )
    np.savez_compressed(
        output / "seed_metrics.npz", means=means, by_rank=by_rank, curves=curves,
        seeds=seeds, metric_names=np.asarray(METRICS), arm_names=np.asarray(ARMS),
        proposal_names=np.asarray(PROPOSALS), world_names=np.asarray(WORLDS),
    )
    summary = {
        "schema": "prob4d.query-information-audit-study-result.v1",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256(protocol_path), "configurations": configurations,
        "independent_seed_count": s, "queries_per_seed_per_panel": denominator,
        "bootstrap_unit": "complete paired seed; all configurations and queries nested",
        "tables": tables, "by_rank": rank_tables, "paired_differences": paired,
        "routes": route_counts, "elapsed_seconds": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {
            "scripts/science/query_information_audit_study.py": sha256(Path(__file__)),
            "src/prob4d/query_information_audit.py": sha256(
                Path(__file__).resolve().parents[2] / "src/prob4d/query_information_audit.py"
            ),
        },
        "artifact_sha256": {
            name: sha256(output / name)
            for name in ("raw_inputs_and_proposals.npz", "seed_metrics.npz", "protocol.json")
        },
    }
    json_write(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.protocol, args.output)
    print(json.dumps({
        "independent_seeds": result["independent_seed_count"],
        "configurations": len(result["configurations"]),
        "elapsed_seconds": result["elapsed_seconds"],
    }))


if __name__ == "__main__":
    main()
