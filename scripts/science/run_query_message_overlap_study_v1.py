#!/usr/bin/env python3
"""Controlled unknown-dependence study for prior-anchored query messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.query_message import (
    GaussianQueryMessage,
    apply_gaussian_query_message,
    compress_gaussian_query_posterior,
    fuse_gaussian_query_messages_covariance_intersection,
    select_pairwise_covariance_intersection,
)
from prob4d.query_posterior import GaussianQueryPosterior

SCHEMA = "prob4d.query-message-overlap-study-protocol"
RESULT_SCHEMA = "prob4d.query-message-overlap-study-result"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol must be a JSON object")
    if value.get("schema") != SCHEMA or value.get("schema_version") != 1:
        raise ValueError("protocol schema is unsupported")
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id", None)
    if not isinstance(protocol_id, str) or len(protocol_id) != 64:
        raise ValueError("protocol_id must be a SHA-256 identifier")
    if _content_id(unsigned) != protocol_id:
        raise ValueError("protocol identity mismatch")
    return value


def _positive_diagonal(value: object, *, name: str, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {dimension}")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _posterior(
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    observation_dimension: int,
) -> GaussianQueryPosterior:
    reduction = 0.5 * (
        prior_covariance
        - posterior_covariance
        + (prior_covariance - posterior_covariance).T
    )
    return GaussianQueryPosterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        mean_shift=posterior_mean - prior_mean,
        covariance_reduction=reduction,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        innovation_precision_quadratic=0.0,
        innovation_log_determinant=0.0,
        innovation_negative_log_likelihood=0.0,
        observation_dimension=observation_dimension,
    )


def _metrics(
    truth: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    chi_square_threshold: float,
) -> dict[str, float]:
    errors = truth - mean
    precision = np.linalg.inv(covariance)
    mahalanobis = np.einsum(
        "ni,ij,nj->n",
        errors,
        precision,
        errors,
        optimize=True,
    )
    dimension = truth.shape[1]
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("evaluated covariance must be positive definite")
    return {
        "coordinate_rmse": float(
            math.sqrt(float(np.mean(np.sum(errors * errors, axis=1))) / dimension)
        ),
        "normalized_nees": float(np.mean(mahalanobis) / dimension),
        "coverage_90": float(np.mean(mahalanobis <= chi_square_threshold)),
        "gaussian_nll_per_dimension": float(
            np.mean(
                0.5
                * (
                    dimension * math.log(2.0 * math.pi)
                    + log_determinant
                    + mahalanobis
                )
                / dimension
            )
        ),
        "covariance_log_determinant": float(log_determinant),
    }


def _message_for_sample(
    *,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    evidence_id: str,
) -> GaussianQueryMessage:
    posterior = _posterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        observation_dimension=prior_mean.size,
    )
    return compress_gaussian_query_posterior(
        posterior,
        query_id="controlled-three-axis-query",
        prior_id="controlled-anchor-prior-v1",
        evidence_ids=(evidence_id,),
    )


def _weight_for_message(
    fused: GaussianQueryMessage,
    message: GaussianQueryMessage,
) -> float:
    mapping = dict(zip(fused.component_message_ids, fused.component_weights, strict=True))
    return float(mapping[message.message_id])


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    dimension = int(protocol["dimension"])
    if dimension != 3:
        raise ValueError("v1 study requires dimension 3")
    sample_count = int(protocol["sample_count"])
    if sample_count < 1000:
        raise ValueError("sample_count must be at least 1000")
    seed = int(protocol["seed"])
    prior_mean = np.asarray(protocol["prior_mean"], dtype=np.float64)
    if prior_mean.shape != (dimension,) or not np.all(np.isfinite(prior_mean)):
        raise ValueError("prior_mean must be a finite dimension-vector")
    prior_diagonal = _positive_diagonal(
        protocol["prior_covariance_diagonal"],
        name="prior_covariance_diagonal",
        dimension=dimension,
    )
    first_noise_diagonal = _positive_diagonal(
        protocol["window_noise_diagonals"][0],
        name="first window noise diagonal",
        dimension=dimension,
    )
    second_noise_diagonal = _positive_diagonal(
        protocol["window_noise_diagonals"][1],
        name="second window noise diagonal",
        dimension=dimension,
    )
    correlations = [float(value) for value in protocol["correlation_values"]]
    if not correlations or any(
        not math.isfinite(value) or value < 0.0 or value >= 1.0
        for value in correlations
    ):
        raise ValueError("correlation_values must be nonempty and lie in [0, 1)")
    grid_size = int(protocol["ci_grid_size"])
    objective = str(protocol["ci_objective"])
    chi_square_threshold = float(protocol["chi_square_threshold_90"])

    prior_covariance = np.diag(prior_diagonal)
    prior_precision = np.diag(1.0 / prior_diagonal)
    prior_natural = prior_precision @ prior_mean
    first_noise = np.diag(first_noise_diagonal)
    second_noise = np.diag(second_noise_diagonal)
    first_precision = np.diag(1.0 / first_noise_diagonal)
    second_precision = np.diag(1.0 / second_noise_diagonal)
    first_root = np.diag(np.sqrt(first_noise_diagonal))
    second_root = np.diag(np.sqrt(second_noise_diagonal))
    identity = np.eye(dimension, dtype=np.float64)
    joint_design = np.vstack((identity, identity))

    first_covariance = np.linalg.inv(prior_precision + first_precision)
    second_covariance = np.linalg.inv(prior_precision + second_precision)
    naive_covariance = np.linalg.inv(
        prior_precision + first_precision + second_precision
    )

    rows: list[dict[str, Any]] = []
    maximum_api_mean_error = 0.0
    maximum_api_covariance_error = 0.0
    maximum_message_information_error = 0.0
    maximum_message_natural_error = 0.0
    maximum_duplicate_mean_error = 0.0
    maximum_duplicate_covariance_error = 0.0

    for index, correlation in enumerate(correlations):
        cross_noise = correlation * first_root @ second_root.T
        joint_noise = np.block(
            [
                [first_noise, cross_noise],
                [cross_noise.T, second_noise],
            ]
        )
        joint_root = np.linalg.cholesky(joint_noise)
        joint_noise_precision = np.linalg.inv(joint_noise)
        exact_covariance = np.linalg.inv(
            prior_precision
            + joint_design.T @ joint_noise_precision @ joint_design
        )

        rng = np.random.default_rng(seed + 1009 * index)
        truth = rng.multivariate_normal(
            prior_mean,
            prior_covariance,
            size=sample_count,
        )
        noise = rng.standard_normal((sample_count, 2 * dimension)) @ joint_root.T
        first_observation = truth + noise[:, :dimension]
        second_observation = truth + noise[:, dimension:]
        first_natural = first_observation @ first_precision.T
        second_natural = second_observation @ second_precision.T
        first_mean = (
            prior_natural[None, :] + first_natural
        ) @ first_covariance.T
        second_mean = (
            prior_natural[None, :] + second_natural
        ) @ second_covariance.T

        first_message = _message_for_sample(
            posterior_mean=first_mean[0],
            posterior_covariance=first_covariance,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            evidence_id=f"rho-{correlation:.6f}-window-a",
        )
        second_message = _message_for_sample(
            posterior_mean=second_mean[0],
            posterior_covariance=second_covariance,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            evidence_id=f"rho-{correlation:.6f}-window-b",
        )
        selected = select_pairwise_covariance_intersection(
            first_message,
            second_message,
            grid_size=grid_size,
            objective=objective,
        )
        first_weight = _weight_for_message(selected, first_message)
        second_weight = _weight_for_message(selected, second_message)
        ci_covariance = np.linalg.inv(
            prior_precision
            + first_weight * first_precision
            + second_weight * second_precision
        )

        exact_natural = (
            np.concatenate((first_observation, second_observation), axis=1)
            @ (joint_noise_precision @ joint_design)
        )
        exact_mean = (
            prior_natural[None, :] + exact_natural
        ) @ exact_covariance.T
        naive_mean = (
            prior_natural[None, :] + first_natural + second_natural
        ) @ naive_covariance.T
        ci_mean = (
            prior_natural[None, :]
            + first_weight * first_natural
            + second_weight * second_natural
        ) @ ci_covariance.T

        for sample_index in range(min(8, sample_count)):
            first_sample_message = _message_for_sample(
                posterior_mean=first_mean[sample_index],
                posterior_covariance=first_covariance,
                prior_mean=prior_mean,
                prior_covariance=prior_covariance,
                evidence_id=f"rho-{correlation:.6f}-window-a-sample-{sample_index}",
            )
            second_sample_message = _message_for_sample(
                posterior_mean=second_mean[sample_index],
                posterior_covariance=second_covariance,
                prior_mean=prior_mean,
                prior_covariance=prior_covariance,
                evidence_id=f"rho-{correlation:.6f}-window-b-sample-{sample_index}",
            )
            fused_sample = fuse_gaussian_query_messages_covariance_intersection(
                (first_sample_message, second_sample_message),
                weights=(first_weight, second_weight),
            )
            belief = apply_gaussian_query_message(fused_sample)
            maximum_api_mean_error = max(
                maximum_api_mean_error,
                float(np.max(np.abs(belief.mean - ci_mean[sample_index]))),
            )
            maximum_api_covariance_error = max(
                maximum_api_covariance_error,
                float(np.max(np.abs(belief.covariance - ci_covariance))),
            )

        maximum_message_information_error = max(
            maximum_message_information_error,
            float(
                np.max(
                    np.abs(first_message.information_increment - first_precision)
                )
            ),
            float(
                np.max(
                    np.abs(second_message.information_increment - second_precision)
                )
            ),
        )
        maximum_message_natural_error = max(
            maximum_message_natural_error,
            float(
                np.max(
                    np.abs(
                        first_message.natural_parameter_increment
                        - first_natural[0]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        second_message.natural_parameter_increment
                        - second_natural[0]
                    )
                )
            ),
        )
        duplicate = fuse_gaussian_query_messages_covariance_intersection(
            (first_message, first_message),
            weights=(0.5, 0.5),
        )
        duplicate_belief = apply_gaussian_query_message(duplicate)
        first_belief = apply_gaussian_query_message(first_message)
        maximum_duplicate_mean_error = max(
            maximum_duplicate_mean_error,
            float(np.max(np.abs(duplicate_belief.mean - first_belief.mean))),
        )
        maximum_duplicate_covariance_error = max(
            maximum_duplicate_covariance_error,
            float(
                np.max(
                    np.abs(
                        duplicate_belief.covariance - first_belief.covariance
                    )
                )
            ),
        )

        methods = {
            "exact_joint": _metrics(
                truth,
                exact_mean,
                exact_covariance,
                chi_square_threshold=chi_square_threshold,
            ),
            "naive_independent": _metrics(
                truth,
                naive_mean,
                naive_covariance,
                chi_square_threshold=chi_square_threshold,
            ),
            "query_covariance_intersection": _metrics(
                truth,
                ci_mean,
                ci_covariance,
                chi_square_threshold=chi_square_threshold,
            ),
            "window_a_only": _metrics(
                truth,
                first_mean,
                first_covariance,
                chi_square_threshold=chi_square_threshold,
            ),
            "window_b_only": _metrics(
                truth,
                second_mean,
                second_covariance,
                chi_square_threshold=chi_square_threshold,
            ),
        }
        rows.append(
            {
                "correlation": correlation,
                "ci_weights": {
                    "window_a": first_weight,
                    "window_b": second_weight,
                },
                "methods": methods,
            }
        )

    criteria = protocol["criteria"]
    exact_nees_error = max(
        abs(row["methods"]["exact_joint"]["normalized_nees"] - 1.0)
        for row in rows
    )
    exact_coverage_error = max(
        abs(row["methods"]["exact_joint"]["coverage_90"] - 0.9)
        for row in rows
    )
    high_rows = [
        row
        for row in rows
        if row["correlation"] >= float(criteria["high_correlation_minimum"])
    ]
    naive_high_min_nees = min(
        row["methods"]["naive_independent"]["normalized_nees"]
        for row in high_rows
    )
    naive_high_max_coverage = max(
        row["methods"]["naive_independent"]["coverage_90"]
        for row in high_rows
    )
    ci_max_nees = max(
        row["methods"]["query_covariance_intersection"]["normalized_nees"]
        for row in rows
    )
    ci_min_coverage = min(
        row["methods"]["query_covariance_intersection"]["coverage_90"]
        for row in rows
    )
    ci_better_than_single = all(
        row["methods"]["query_covariance_intersection"]["coordinate_rmse"]
        < min(
            row["methods"]["window_a_only"]["coordinate_rmse"],
            row["methods"]["window_b_only"]["coordinate_rmse"],
        )
        for row in rows
    )
    independent_row = next(row for row in rows if row["correlation"] == 0.0)
    independent_rmse_error = abs(
        independent_row["methods"]["exact_joint"]["coordinate_rmse"]
        - independent_row["methods"]["naive_independent"]["coordinate_rmse"]
    )
    independent_covariance_error = abs(
        independent_row["methods"]["exact_joint"]["covariance_log_determinant"]
        - independent_row["methods"]["naive_independent"][
            "covariance_log_determinant"
        ]
    )

    checks = {
        "exact_joint_normalized_nees": exact_nees_error
        <= float(criteria["maximum_exact_normalized_nees_error"]),
        "exact_joint_coverage": exact_coverage_error
        <= float(criteria["maximum_exact_coverage_error"]),
        "naive_high_correlation_overconfidence": naive_high_min_nees
        >= float(criteria["minimum_naive_high_correlation_normalized_nees"]),
        "naive_high_correlation_undercoverage": naive_high_max_coverage
        <= float(criteria["maximum_naive_high_correlation_coverage"]),
        "ci_not_overconfident": ci_max_nees
        <= float(criteria["maximum_ci_normalized_nees"]),
        "ci_coverage_floor": ci_min_coverage
        >= float(criteria["minimum_ci_coverage"]),
        "ci_improves_over_either_single_window": ci_better_than_single,
        "independent_naive_matches_exact_rmse": independent_rmse_error
        <= float(criteria["maximum_independent_parity_error"]),
        "independent_naive_matches_exact_covariance": independent_covariance_error
        <= float(criteria["maximum_independent_parity_error"]),
        "query_message_information_parity": maximum_message_information_error
        <= float(criteria["maximum_api_parity_error"]),
        "query_message_natural_parity": maximum_message_natural_error
        <= float(criteria["maximum_api_parity_error"]),
        "query_ci_mean_parity": maximum_api_mean_error
        <= float(criteria["maximum_api_parity_error"]),
        "query_ci_covariance_parity": maximum_api_covariance_error
        <= float(criteria["maximum_api_parity_error"]),
        "duplicate_mean_idempotence": maximum_duplicate_mean_error
        <= float(criteria["maximum_api_parity_error"]),
        "duplicate_covariance_idempotence": maximum_duplicate_covariance_error
        <= float(criteria["maximum_api_parity_error"]),
    }
    summary = {
        "exact_joint_maximum_normalized_nees_error": exact_nees_error,
        "exact_joint_maximum_coverage_error": exact_coverage_error,
        "naive_high_correlation_minimum_normalized_nees": naive_high_min_nees,
        "naive_high_correlation_maximum_coverage": naive_high_max_coverage,
        "ci_maximum_normalized_nees": ci_max_nees,
        "ci_minimum_coverage": ci_min_coverage,
        "ci_improves_over_either_single_window": ci_better_than_single,
        "maximum_api_mean_error": maximum_api_mean_error,
        "maximum_api_covariance_error": maximum_api_covariance_error,
        "maximum_message_information_error": maximum_message_information_error,
        "maximum_message_natural_error": maximum_message_natural_error,
        "maximum_duplicate_mean_error": maximum_duplicate_mean_error,
        "maximum_duplicate_covariance_error": maximum_duplicate_covariance_error,
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "decision": (
            "controlled-overlap-passed"
            if all(checks.values())
            else "controlled-overlap-failed"
        ),
        "sample_count_per_correlation": sample_count,
        "independent_unit": "simulated_complete_query_case",
        "rows": rows,
        "checks": checks,
        "summary": summary,
        "claim_boundary": protocol["claim_boundary"],
    }
    unsigned_result = dict(result)
    result["result_id"] = _content_id(unsigned_result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    result = run(protocol)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {
        "decision": result["decision"],
        "protocol_id": result["protocol_id"],
        "result_id": result["result_id"],
        "summary": result["summary"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if result["decision"] == "controlled-overlap-passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
