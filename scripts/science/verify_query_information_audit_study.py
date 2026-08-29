"""Independently verify a retained study without importing its estimator/scorer.

Uses measurement-space conditioning, explicitly inverted 3-D scoring matrices,
and the registered corruption definitions. No random scientific data are drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    for name, expected in summary["artifact_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    with np.load(directory / "raw_inputs_and_proposals.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(directory / "seed_metrics.npz", allow_pickle=False) as archive:
        retained = archive["means"].copy()
    truth = data["truth"]
    count, seeds = truth.shape[1:3]
    recomputed = np.zeros_like(retained)
    maximum_mean_error, maximum_covariance_error = 0.0, 0.0
    admission_cases = 0
    invalid_count, naive_invalid_count = [0] * 4, [0] * 4
    audited_invalid_count = [0] * 4
    good_rank_deficient_admissions = [0] * 3
    good_rank_deficient_denominator = [0] * 3
    for ci, config in enumerate(summary["configurations"]):
        prior = data["prior_covariances"][ci]
        mean0 = data["prior_mean"]
        information = data["information"][ci]
        values, vectors = np.linalg.eigh(information)
        keep = values > 1e-10 * values[-1]
        h_matrix = vectors[:, keep].T
        noise = np.diag(1.0 / values[keep])
        gain = prior @ h_matrix.T @ np.linalg.inv(h_matrix @ prior @ h_matrix.T + noise)
        measurement_covariance = prior - gain @ h_matrix @ prior
        maximum_covariance_error = max(maximum_covariance_error, float(np.max(np.abs(
            measurement_covariance - data["proposal_covariances"][0, ci]
        ))))
        pseudo = h_matrix.T @ noise @ h_matrix
        prior_means = np.broadcast_to(mean0, (seeds, 7))
        for wi in range(2):
            y = data["natural"][wi, ci] @ h_matrix.T @ noise
            measurement_mean = mean0 + (y - h_matrix @ mean0) @ gain.T
            maximum_mean_error = max(maximum_mean_error, float(np.max(np.abs(
                measurement_mean - data["proposal_means"][wi, 0, ci]
            ))))
            point_mean = data["natural"][wi, ci] @ pseudo.T
            for pi in range(4):
                corrupt = (
                    (pi in (1, 2) and config["rank"] < 7)
                    or (pi == 3 and config["shared_window_correlation"] > 0.0)
                )
                if not np.all(data["audit_valid"][wi, pi, ci] == (not corrupt)):
                    raise ValueError("audit validity disagrees with registered corruption")
                candidate_mean = data["proposal_means"][wi, pi, ci]
                candidate_cov = data["proposal_covariances"][pi, ci]
                for qi, jacobian in enumerate(data["query_jacobians"][ci]):
                    initial_error = (prior_means - truth[wi, ci]) @ jacobian.T
                    initial_squared = np.sum(initial_error**2, axis=1)
                    cq = jacobian @ candidate_cov @ jacobian.T
                    pq = jacobian @ prior @ jacobian.T
                    maximum = np.linalg.eigvalsh(0.5 * (cq + cq.T))[-1]
                    reduction = 1.0 - np.trace(cq) / np.trace(pq)
                    naive_accept = (
                        maximum <= protocol["primary_tolerance_m"]**2
                        and reduction >= protocol["policy"]["minimum_variance_reduction"]
                    )
                    accepted = naive_accept and not corrupt
                    if not np.all(data["admitted"][wi, pi, ci, qi] == accepted):
                        raise ValueError("retained admission differs from independent rule")
                    admission_cases += seeds
                    if wi == 0:
                        if corrupt:
                            invalid_count[pi] += seeds
                            naive_invalid_count[pi] += int(naive_accept) * seeds
                            audited_invalid_count[pi] += int(accepted) * seeds
                        if pi == 0 and config["rank"] < 7:
                            good_rank_deficient_denominator[qi] += seeds
                            good_rank_deficient_admissions[qi] += int(accepted) * seeds
                    full_rank = config["rank"] == 7
                    definitions = (
                        (prior_means, prior, False),
                        (measurement_mean if full_rank else prior_means,
                         measurement_covariance if full_rank else prior, full_rank),
                        (point_mean, None, True),
                        (point_mean, pseudo, True),
                        (data["proposal_means"][wi, 1, ci],
                         data["proposal_covariances"][1, ci], True),
                        (measurement_mean, measurement_covariance, True),
                        (candidate_mean if naive_accept else prior_means,
                         candidate_cov if naive_accept else prior, naive_accept),
                        (candidate_mean if accepted else prior_means,
                         candidate_cov if accepted else prior, accepted),
                    )
                    for ai, (mean, covariance, accept) in enumerate(definitions):
                        error = (mean - truth[wi, ci]) @ jacobian.T
                        squared = np.sum(error**2, axis=1)
                        metrics = np.zeros((seeds, 6))
                        metrics[:, 0] = squared
                        metrics[:, 4] = accept
                        metrics[:, 5] = accept & (squared > initial_squared + 1e-12)
                        if covariance is None:
                            metrics[:, 1:4] = np.nan
                        else:
                            projected = jacobian @ covariance @ jacobian.T
                            projected = (projected + projected.T) / 2.0
                            quadratic = np.einsum(
                                "ni,ij,nj->n", error, np.linalg.inv(projected), error
                            )
                            metrics[:, 1] = 0.5 * (
                                3.0 * np.log(2.0 * np.pi)
                                + np.log(np.linalg.det(projected)) + quadratic
                            )
                            metrics[:, 2] = quadratic <= 6.251388631170325
                            metrics[:, 3] = (
                                2.0 * 1.6448536269514722
                                * np.sqrt(np.diag(projected)).mean()
                            )
                        recomputed[wi, pi, ai] += metrics
    recomputed /= count * 3
    np.testing.assert_allclose(recomputed, retained, rtol=1e-8, atol=2e-8, equal_nan=True)
    if maximum_mean_error > 1e-9 or maximum_covariance_error > 1e-10:
        raise ValueError("measurement-space posterior comparison failed")
    result = {
        "schema": "prob4d.query-information-audit-study-verification.v1",
        "verified": True,
        "independently_verified_candidate_query_decisions": admission_cases,
        "independently_verified_seed_metric_cells": int(recomputed.size),
        "maximum_measurement_space_mean_absolute_difference": maximum_mean_error,
        "maximum_measurement_space_covariance_absolute_difference": maximum_covariance_error,
        "maximum_seed_metric_absolute_difference": float(np.nanmax(np.abs(recomputed - retained))),
        "matched_prior_invalid_proposals_by_condition": invalid_count,
        "matched_prior_invalid_admissions_without_audit": naive_invalid_count,
        "matched_prior_invalid_admissions_with_audit": audited_invalid_count,
        "matched_prior_clean_rank_deficient_admissions_by_query": good_rank_deficient_admissions,
        "matched_prior_clean_rank_deficient_denominators_by_query": good_rank_deficient_denominator,
        "condition_order": protocol["proposals"], "query_order": protocol["query_names"],
        "summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": (
            "Artifact/algebra/score verification, not scientific promotion or real-data validation."
        ),
    }
    (directory / "verification.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))


if __name__ == "__main__":
    main()
