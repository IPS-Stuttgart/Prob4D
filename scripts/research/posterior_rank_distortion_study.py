"""Reproduce the posterior rank--distortion frontier and baseline regret study.

The study uses frozen, jointly Gaussian query models with the same construction
as the independent dense theorem tests.  It compares the globally optimal
factor projection against two natural same-rank baselines:

* the Euclidean posterior-response SVD ordering used by the exact compressor;
* latent covariance-energy PCA based on ``U.T @ U``.

This is a prevalence and effect-size experiment, not real-data evidence.  The
registered distortion is the normalized posterior-covariance trace contraction
implemented by :mod:`prob4d.posterior_rank_distortion`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from prob4d.posterior_rank_distortion import posterior_rank_distortion_frontier

Array: TypeAlias = NDArray[np.float64]
Row: TypeAlias = dict[str, int | float | str | bool | None]


@dataclass(frozen=True, slots=True)
class Configuration:
    """One factor-rank and query-dimension study stratum."""

    factor_rank: int
    query_dimension: int

    @property
    def label(self) -> str:
        return f"r{self.factor_rank}-q{self.query_dimension}"


class DenseInnovation:
    """Study-only dense solver implementing Prob4D's innovation protocol."""

    def __init__(self, covariance: Array) -> None:
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.dimension = int(self.covariance.shape[0])
        self.observation_count = self.dimension // 3

    def solve(self, value: object) -> Array:
        raw = np.asarray(value, dtype=np.float64)
        return np.linalg.solve(
            self.covariance,
            raw.reshape(self.dimension, -1),
        ).reshape(raw.shape)


def _parse_configuration(value: str) -> Configuration:
    try:
        rank_text, query_text = value.split(":", maxsplit=1)
        rank = int(rank_text)
        query_dimension = int(query_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "configuration must have the form FACTOR_RANK:QUERY_DIMENSION"
        ) from exc
    if rank < 1 or query_dimension < 1:
        raise argparse.ArgumentTypeError("configuration dimensions must be positive")
    if query_dimension > 8:
        raise argparse.ArgumentTypeError(
            "query dimension must not exceed the frozen state dimension 8"
        )
    return Configuration(rank, query_dimension)


def _model(
    seed: int,
    configuration: Configuration,
    observation_count: int,
) -> tuple[Array, Array, Array, Array, Array]:
    rng = np.random.default_rng(seed)
    dimension = 3 * observation_count
    state_dimension = 8
    diagonal = np.diag(rng.uniform(0.3, 1.5, dimension))
    factor = rng.normal(size=(dimension, configuration.factor_rank)) / np.sqrt(
        configuration.factor_rank
    )
    state_map = rng.normal(size=(dimension, state_dimension)) / 2.0
    query_map = rng.normal(size=(configuration.query_dimension, state_dimension))
    prior = query_map @ query_map.T
    cross = query_map @ state_map.T
    remainder = diagonal + state_map @ state_map.T
    innovation = remainder + factor @ factor.T
    return factor, prior, cross, remainder, innovation


def _latent_statistics(
    factor: Array,
    prior: Array,
    cross: Array,
    innovation: Array,
) -> tuple[Array, Array, Array, Array]:
    solved = np.linalg.solve(
        innovation,
        np.concatenate((factor, cross.T), axis=1),
    )
    rank = factor.shape[1]
    solved_factor = solved[:, :rank]
    solved_cross = solved[:, rank:]
    posterior = 0.5 * (
        prior - cross @ solved_cross + (prior - cross @ solved_cross).T
    )
    posterior_root = np.linalg.cholesky(posterior)
    gram = 0.5 * (
        factor.T @ solved_factor + (factor.T @ solved_factor).T
    )
    remainder_metric = 0.5 * (
        np.eye(rank) - gram + (np.eye(rank) - gram).T
    )
    np.linalg.cholesky(remainder_metric)
    response = factor.T @ solved_cross
    normalized_response = np.linalg.solve(posterior_root, response.T).T
    relevance = 0.5 * (
        normalized_response @ normalized_response.T
        + (normalized_response @ normalized_response.T).T
    )
    svd_basis, _, _ = np.linalg.svd(normalized_response, full_matrices=True)
    _, energy_basis = np.linalg.eigh(
        0.5 * (factor.T @ factor + (factor.T @ factor).T)
    )
    energy_basis = energy_basis[:, ::-1]
    return remainder_metric, relevance, svd_basis, energy_basis


def _candidate_distortion(
    remainder_metric: Array,
    relevance: Array,
    ordered_basis: Array,
    retained_rank: int,
) -> float:
    discarded = ordered_basis[:, retained_rank:]
    if discarded.shape[1] == 0:
        return 0.0
    metric = 0.5 * (
        discarded.T @ remainder_metric @ discarded
        + (discarded.T @ remainder_metric @ discarded).T
    )
    objective = 0.5 * (
        discarded.T @ relevance @ discarded
        + (discarded.T @ relevance @ discarded).T
    )
    distortion = float(np.trace(np.linalg.solve(metric, objective)))
    return max(distortion, 0.0)


def _relative_regret(candidate: float, optimum: float) -> float | None:
    if optimum <= 1e-10:
        return None
    return max(candidate - optimum, 0.0) / optimum


def _evaluate_configuration(
    configuration: Configuration,
    *,
    seeds: int,
    observation_count: int,
) -> list[Row]:
    rows: list[Row] = []
    expected_exact_rank = min(
        configuration.factor_rank,
        configuration.query_dimension,
    )
    for seed in range(seeds):
        factor, prior, cross, _, innovation = _model(
            seed,
            configuration,
            observation_count,
        )
        frontier = posterior_rank_distortion_frontier(
            factor.reshape(observation_count, 3, configuration.factor_rank),
            prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=DenseInnovation(innovation),
        )
        if frontier.numerical_exact_rank != expected_exact_rank:
            raise RuntimeError(
                "random study model did not realize its expected generic exact rank"
            )
        remainder_metric, relevance, svd_basis, energy_basis = _latent_statistics(
            factor,
            prior,
            cross,
            innovation,
        )
        for retained_rank in range(configuration.factor_rank + 1):
            point = frontier.point(retained_rank)
            optimum = point.audited_normalized_covariance_trace_loss
            svd_distortion = _candidate_distortion(
                remainder_metric,
                relevance,
                svd_basis,
                retained_rank,
            )
            energy_distortion = _candidate_distortion(
                remainder_metric,
                relevance,
                energy_basis,
                retained_rank,
            )
            scale = max(optimum, svd_distortion, energy_distortion, 1.0)
            if svd_distortion < optimum - 2e-9 * scale:
                raise RuntimeError("SVD baseline beat the claimed global optimum")
            if energy_distortion < optimum - 2e-9 * scale:
                raise RuntimeError("energy baseline beat the claimed global optimum")
            rows.append(
                {
                    "configuration": configuration.label,
                    "seed": seed,
                    "factor_rank": configuration.factor_rank,
                    "query_dimension": configuration.query_dimension,
                    "retained_rank": retained_rank,
                    "compression_fraction": retained_rank
                    / configuration.factor_rank,
                    "optimal_distortion": optimum,
                    "closed_form_distortion": (
                        point.optimal_normalized_covariance_trace_loss
                    ),
                    "audit_absolute_error": abs(
                        point.optimal_normalized_covariance_trace_loss - optimum
                    ),
                    "svd_distortion": svd_distortion,
                    "svd_additive_regret": max(svd_distortion - optimum, 0.0),
                    "svd_relative_regret": _relative_regret(
                        svd_distortion,
                        optimum,
                    ),
                    "energy_distortion": energy_distortion,
                    "energy_additive_regret": max(
                        energy_distortion - optimum,
                        0.0,
                    ),
                    "energy_relative_regret": _relative_regret(
                        energy_distortion,
                        optimum,
                    ),
                    "exact_posterior": point.exact_posterior,
                }
            )
    return rows


def _numeric_values(rows: list[Row], key: str) -> Array:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return np.asarray(values, dtype=np.float64)


def _regret_summary(rows: list[Row], prefix: str) -> dict[str, float]:
    additive = _numeric_values(rows, f"{prefix}_additive_regret")
    relative = _numeric_values(rows, f"{prefix}_relative_regret")
    threshold = np.asarray(
        [1e-8 * max(float(row["optimal_distortion"]), 1.0) for row in rows],
        dtype=np.float64,
    )
    return {
        "strictly_suboptimal_fraction": float(np.mean(additive > threshold)),
        "mean_additive_regret": float(np.mean(additive)),
        "median_additive_regret": float(np.median(additive)),
        "maximum_additive_regret": float(np.max(additive)),
        "mean_relative_regret": float(np.mean(relative)),
        "median_relative_regret": float(np.median(relative)),
        "p95_relative_regret": float(np.quantile(relative, 0.95)),
        "maximum_relative_regret": float(np.max(relative)),
    }


def _aggregate(rows: list[Row], configurations: list[Configuration]) -> dict[str, object]:
    approximate_rows = [
        row for row in rows if float(row["optimal_distortion"]) > 1e-10
    ]
    if not approximate_rows:
        raise RuntimeError("study produced no positive-distortion rank points")
    control = next(
        (
            row
            for row in rows
            if row["configuration"] == "r7-q3"
            and row["seed"] == 93
            and row["retained_rank"] == 1
        ),
        None,
    )
    if control is None:
        raise RuntimeError("the frozen r7-q3 seed-93 control is missing")
    worst_svd = max(
        approximate_rows,
        key=lambda row: float(row["svd_relative_regret"]),
    )
    return {
        "schema": "prob4d.posterior-rank-distortion-study.v1",
        "claim_boundary": (
            "frozen jointly Gaussian synthetic query models; normalized posterior-"
            "covariance trace contraction within orthogonal U-to-UV projections"
        ),
        "configuration_count": len(configurations),
        "model_count": len(configurations)
        * len({int(row["seed"]) for row in rows}),
        "frontier_point_count": len(rows),
        "positive_distortion_point_count": len(approximate_rows),
        "maximum_closed_form_audit_error": float(
            np.max(_numeric_values(rows, "audit_absolute_error"))
        ),
        "svd": _regret_summary(approximate_rows, "svd"),
        "energy": _regret_summary(approximate_rows, "energy"),
        "frozen_control": {
            "configuration": control["configuration"],
            "seed": control["seed"],
            "retained_rank": control["retained_rank"],
            "optimal_distortion": control["optimal_distortion"],
            "svd_distortion": control["svd_distortion"],
            "svd_ratio_to_optimum": float(control["svd_distortion"])
            / float(control["optimal_distortion"]),
        },
        "worst_svd_relative_regret_case": worst_svd,
    }


def run_study(
    configurations: list[Configuration],
    *,
    seeds: int,
    observation_count: int,
) -> tuple[list[Row], dict[str, object]]:
    """Run all frozen configurations and return rows plus aggregate summary."""
    if seeds <= 93:
        raise ValueError("seeds must exceed 93 so the frozen control is included")
    if observation_count < 1:
        raise ValueError("observation_count must be positive")
    rows: list[Row] = []
    for configuration in configurations:
        rows.extend(
            _evaluate_configuration(
                configuration,
                seeds=seeds,
                observation_count=observation_count,
            )
        )
    return rows, _aggregate(rows, configurations)


def _write_outputs(output: Path, rows: list[Row], summary: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output / "frontier-points.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        action="append",
        type=_parse_configuration,
        dest="configurations",
        help="repeatable FACTOR_RANK:QUERY_DIMENSION stratum",
    )
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--observation-count", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/posterior-rank-distortion-study"),
    )
    args = parser.parse_args()
    configurations = args.configurations or [
        Configuration(7, 1),
        Configuration(7, 3),
        Configuration(14, 3),
        Configuration(28, 5),
    ]
    rows, summary = run_study(
        configurations,
        seeds=args.seeds,
        observation_count=args.observation_count,
    )
    _write_outputs(args.output, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
