#!/usr/bin/env python3
"""Evaluate optimal posterior rank--distortion on real cloth trajectories.

The study reuses the frozen recording-disjoint Tracking Cloth query-portfolio
model construction.  For every fold and registered query portfolio it compares,
at every supplied-factor rank and therefore at the same factor/projection byte
budget:

* the generalized-eigen posterior-trace optimum;
* the posterior-normalized response-SVD ordering used by the exact compressor;
* shared-factor covariance-energy PCA.

Rank selection uses only each training-fold Gaussian model.  Held-out recording
windows are used afterwards for RMSE, NEES, coverage, and NLL diagnostics.  Raw
trajectories are never written to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from prob4d.posterior_rank_distortion import posterior_rank_distortion_frontier

SCHEMA = "prob4d.tracking-cloth-rank-distortion-frontier.v1"
RESULT_SCHEMA = "prob4d.tracking-cloth-rank-distortion-frontier-result.v1"
BASE_SCRIPT_PATH = Path("scripts/science/run_tracking_cloth_query_portfolio_v1.py")
BASE_SCRIPT_GIT_BLOB_SHA1 = "dfb557c95bf04ac79af7b93447598ed4822d72cf"
BASE_PROTOCOL_PATH = Path("protocols/tracking-cloth-query-portfolio-v1.json")
BASE_PROTOCOL_GIT_BLOB_SHA1 = "0607e4c2e116cd6f613ae563fcc46e6dc785d988"
METHOD_OPTIMAL = "generalized-eigen-optimal"
METHOD_RESPONSE_SVD = "posterior-response-svd"
METHOD_FACTOR_PCA = "shared-factor-pca"
METHODS = (METHOD_OPTIMAL, METHOD_RESPONSE_SVD, METHOD_FACTOR_PCA)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate_hex(value: object, *, label: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must contain {length} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("schema") != SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unsupported rank-distortion protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _validate_hex(protocol_id, label="protocol_id", length=64)
    if _content_id(unsigned) != protocol_id:
        raise ValueError("rank-distortion protocol identity mismatch")
    expected = {
        "base_script_path": str(BASE_SCRIPT_PATH),
        "base_script_git_blob_sha1": BASE_SCRIPT_GIT_BLOB_SHA1,
        "base_protocol_path": str(BASE_PROTOCOL_PATH),
        "base_protocol_git_blob_sha1": BASE_PROTOCOL_GIT_BLOB_SHA1,
        "sizes": ["A2", "A3"],
        "query_counts": [1, 2, 4, 8, 12, 20],
        "methods": list(METHODS),
        "frontier_ranks": "all",
        "selection_uses_heldout_outcomes": False,
        "expected_csv_files": 120,
        "expected_compatible_recordings": 80,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"registered {key} changed")
    budgets = protocol.get("trace_budgets_per_query_dimension")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("trace budgets must be a nonempty list")
    numeric_budgets = [float(value) for value in budgets]
    if numeric_budgets != sorted(set(numeric_budgets)):
        raise ValueError("trace budgets must be unique and increasing")
    if not all(0.0 < value < 1.0 for value in numeric_budgets):
        raise ValueError("trace budgets must lie in (0, 1)")
    if float(protocol["primary_trace_budget_per_query_dimension"]) not in numeric_budgets:
        raise ValueError("primary trace budget must be registered in the budget grid")
    return protocol


def _load_base_module(protocol: Mapping[str, Any]) -> tuple[ModuleType, dict[str, Any]]:
    script_bytes = BASE_SCRIPT_PATH.read_bytes()
    if _git_blob_sha1(script_bytes) != protocol["base_script_git_blob_sha1"]:
        raise ValueError("registered Tracking Cloth model-construction script changed")
    protocol_bytes = BASE_PROTOCOL_PATH.read_bytes()
    if _git_blob_sha1(protocol_bytes) != protocol["base_protocol_git_blob_sha1"]:
        raise ValueError("registered Tracking Cloth model protocol changed")
    base_protocol = json.loads(protocol_bytes)
    if base_protocol.get("schema") != "prob4d.tracking-cloth-query-portfolio.v1":
        raise ValueError("base Tracking Cloth protocol schema changed")
    inherited = protocol["inherited_model_settings"]
    for key, expected in inherited.items():
        if base_protocol.get(key) != expected:
            raise ValueError(f"inherited model setting {key} changed")

    name = "prob4d_tracking_cloth_query_portfolio_base_v1"
    spec = importlib.util.spec_from_file_location(name, BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the registered Tracking Cloth model script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, base_protocol


def _whiten_symmetric(root: np.ndarray, value: np.ndarray) -> np.ndarray:
    left = np.linalg.solve(root, value)
    result = np.linalg.solve(root, left.T).T
    return 0.5 * (result + result.T)


def _projection_evaluation(
    base: ModuleType,
    *,
    prior: np.ndarray,
    cross: np.ndarray,
    conditional: np.ndarray,
    shared: np.ndarray,
    full_observation_covariance: np.ndarray,
    full_gain: np.ndarray,
    full_posterior: np.ndarray,
    full_posterior_root: np.ndarray,
    projection: np.ndarray,
    validity_margin: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    rank = int(projection.shape[1])
    reduced_factor = shared @ projection
    solver = base.BlockLowRankSolver(conditional, reduced_factor)
    solved = solver.solve(cross.T)
    gain = solved.T
    posterior = prior - cross @ solved
    posterior = 0.5 * (posterior + posterior.T)

    loss = 0.5 * ((full_posterior - posterior) + (full_posterior - posterior).T)
    normalized_loss = _whiten_symmetric(full_posterior_root, loss)
    loss_eigenvalues = np.linalg.eigvalsh(normalized_loss)
    scale = max(float(np.max(np.abs(loss_eigenvalues))), 1.0)
    if float(loss_eigenvalues[0]) < -1e-8 * scale:
        raise ValueError("a projected factor produced non-PSD posterior contraction")
    loss_eigenvalues = np.maximum(loss_eigenvalues, 0.0)
    trace_loss = max(float(np.sum(loss_eigenvalues)), 0.0)
    maximum_contraction = max(float(loss_eigenvalues[-1]), 0.0)

    gain_change = gain - full_gain
    mean_error_covariance = gain_change @ full_observation_covariance @ gain_change.T
    normalized_mean_error = _whiten_symmetric(
        full_posterior_root,
        mean_error_covariance,
    )
    mean_shift_risk = max(
        float(np.trace(normalized_mean_error)) / prior.shape[0],
        0.0,
    )
    posterior_eigenvalues = np.linalg.eigvalsh(posterior)
    maximum_posterior_eigenvalue = max(float(posterior_eigenvalues[-1]), 0.0)
    minimum_posterior_eigenvalue = float(posterior_eigenvalues[0])
    valid = bool(minimum_posterior_eigenvalue > 0.0 and maximum_contraction < 1.0 - validity_margin)
    condition_number = (
        maximum_posterior_eigenvalue / minimum_posterior_eigenvalue
        if minimum_posterior_eigenvalue > 0.0
        else None
    )
    if condition_number is not None and not math.isfinite(condition_number):
        condition_number = None
    return (
        {
            "retained_rank": rank,
            "discarded_dimension": int(shared.shape[1] - rank),
            "normalized_trace_loss": trace_loss,
            "normalized_trace_loss_per_query_dimension": trace_loss / prior.shape[0],
            "maximum_normalized_covariance_contraction": maximum_contraction,
            "mean_shift_risk": mean_shift_risk,
            "valid_posterior": valid,
            "minimum_posterior_eigenvalue": minimum_posterior_eigenvalue,
            "posterior_condition_number": condition_number,
            "raw_factor_bytes": int(reduced_factor.nbytes),
            "raw_projection_bytes": int(projection.nbytes),
            "raw_factor_plus_projection_bytes": int(reduced_factor.nbytes + projection.nbytes),
        },
        gain,
        posterior,
    )


def compute_projection_frontiers(
    base: ModuleType,
    *,
    prior: np.ndarray,
    cross: np.ndarray,
    conditional: np.ndarray,
    shared: np.ndarray,
    numerical_relative_tolerance: float,
    validity_margin: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[np.ndarray]], dict[str, Any]]:
    full_solver = base.BlockLowRankSolver(conditional, shared)
    solved_cross = full_solver.solve(cross.T)
    full_gain = solved_cross.T
    full_posterior = prior - cross @ solved_cross
    full_posterior = 0.5 * (full_posterior + full_posterior.T)
    full_posterior_root = np.linalg.cholesky(full_posterior)
    full_observation_covariance = conditional + shared @ shared.T
    original_rank = int(shared.shape[1])

    optimum = posterior_rank_distortion_frontier(
        shared.reshape(shared.shape[0] // 3, 3, original_rank),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=full_solver,
        numerical_relative_tolerance=numerical_relative_tolerance,
    )

    response = shared.T @ solved_cross
    normalized_response = np.linalg.solve(full_posterior_root, response.T).T
    response_basis, _, _ = np.linalg.svd(normalized_response, full_matrices=True)
    _, _, factor_right = np.linalg.svd(shared, full_matrices=True)
    pca_basis = factor_right.T

    projections: dict[str, list[np.ndarray]] = {
        METHOD_OPTIMAL: [point.latent_projection for point in optimum.points],
        METHOD_RESPONSE_SVD: [response_basis[:, :rank].copy() for rank in range(original_rank + 1)],
        METHOD_FACTOR_PCA: [pca_basis[:, :rank].copy() for rank in range(original_rank + 1)],
    }
    rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    gain_posterior_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    maximum_identity_error = 0.0
    maximum_optimality_violation = 0.0

    for method in METHODS:
        for rank, projection in enumerate(projections[method]):
            row, gain, posterior = _projection_evaluation(
                base,
                prior=prior,
                cross=cross,
                conditional=conditional,
                shared=shared,
                full_observation_covariance=full_observation_covariance,
                full_gain=full_gain,
                full_posterior=full_posterior,
                full_posterior_root=full_posterior_root,
                projection=projection,
                validity_margin=validity_margin,
            )
            row["method"] = method
            if method == METHOD_OPTIMAL:
                point = optimum.point(rank)
                scale = max(
                    row["normalized_trace_loss"],
                    point.optimal_normalized_covariance_trace_loss,
                    1.0,
                )
                identity_error = (
                    abs(
                        row["normalized_trace_loss"]
                        - point.optimal_normalized_covariance_trace_loss
                    )
                    / scale
                )
                maximum_identity_error = max(maximum_identity_error, identity_error)
                mean_scale = max(row["mean_shift_risk"], point.mean_shift_risk, 1.0)
                maximum_identity_error = max(
                    maximum_identity_error,
                    abs(row["mean_shift_risk"] - point.mean_shift_risk) / mean_scale,
                )
                row.update(
                    {
                        "closed_form_normalized_trace_loss": (
                            point.optimal_normalized_covariance_trace_loss
                        ),
                        "boundary_generalized_eigengap": (point.boundary_generalized_eigengap),
                        "optimal_subspace_unique": point.optimal_subspace_unique,
                        "exact_posterior": point.exact_posterior,
                    }
                )
            rows[method].append(row)
            gain_posterior_cache[(method, rank)] = (gain, posterior)

    for rank in range(original_rank + 1):
        optimal_loss = rows[METHOD_OPTIMAL][rank]["normalized_trace_loss"]
        for method in (METHOD_RESPONSE_SVD, METHOD_FACTOR_PCA):
            violation = optimal_loss - rows[method][rank]["normalized_trace_loss"]
            maximum_optimality_violation = max(maximum_optimality_violation, violation)
    if maximum_identity_error > 5e-8:
        raise ValueError("optimal frontier failed its independent dense identity audit")
    if maximum_optimality_violation > 5e-8:
        raise ValueError("a baseline beat the claimed same-rank optimum")

    context = {
        "full_gain": full_gain,
        "full_posterior": full_posterior,
        "gain_posterior_cache": gain_posterior_cache,
        "original_rank": original_rank,
        "query_dimension": int(prior.shape[0]),
        "numerical_exact_rank": int(optimum.numerical_exact_rank),
        "generalized_eigenvalues": optimum.generalized_eigenvalues.tolist(),
        "maximum_identity_relative_error": maximum_identity_error,
        "maximum_optimality_violation": maximum_optimality_violation,
    }
    return rows, projections, context


def minimum_rank_for_budget(
    rows: list[Mapping[str, Any]],
    budget_per_query_dimension: float,
) -> int:
    for row in rows:
        if (
            bool(row["valid_posterior"])
            and float(row["normalized_trace_loss_per_query_dimension"])
            <= budget_per_query_dimension
        ):
            return int(row["retained_rank"])
    raise ValueError("the complete factor did not satisfy the registered budget")


def _selected_rank_reasons(
    frontiers: Mapping[str, list[Mapping[str, Any]]],
    context: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, dict[int, list[str]]]:
    original_rank = int(context["original_rank"])
    exact_rank = int(context["numerical_exact_rank"])
    reasons: dict[str, dict[int, list[str]]] = {method: defaultdict(list) for method in METHODS}
    for method in METHODS:
        for rank in protocol["heldout_fixed_ranks"]:
            numeric = int(rank)
            if 0 <= numeric <= original_rank:
                reasons[method][numeric].append("fixed-rank")
        reasons[method][original_rank].append("full-factor")
        reasons[method][exact_rank].append("optimal-exact-rank-reference")
        if exact_rank:
            reasons[method][exact_rank - 1].append("pre-exact-rank-reference")
        first_valid = next(
            int(row["retained_rank"]) for row in frontiers[method] if bool(row["valid_posterior"])
        )
        reasons[method][first_valid].append("minimum-valid-rank")
        for budget in protocol["trace_budgets_per_query_dimension"]:
            rank = minimum_rank_for_budget(frontiers[method], float(budget))
            reasons[method][rank].append(f"trace-budget-{float(budget):g}")
    return {
        method: {rank: sorted(set(values)) for rank, values in selected.items()}
        for method, selected in reasons.items()
    }


def _evaluate_fold(
    base: ModuleType,
    *,
    train: list[Any],
    test: list[Any],
    model_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    size: str,
    fold_index: int,
) -> list[dict[str, Any]]:
    train_y = np.concatenate([record.observations_m for record in train])
    train_q_all = np.concatenate([record.future_displacements_m for record in train])
    test_y = np.concatenate([record.observations_m for record in test])
    test_q_all = np.concatenate([record.future_displacements_m for record in test])
    marker_count = int(train_y.shape[1])
    mean, joint = base._joint_covariance(
        train_q_all,
        train_y,
        float(model_protocol["joint_covariance_shrinkage"]),
        float(model_protocol["joint_covariance_ridge_fraction"]),
    )
    full_query_dimension = 3 * marker_count
    observation_mean = mean[full_query_dimension:]
    observation_covariance = joint[full_query_dimension:, full_query_dimension:]
    conditional, shared, conditional_fraction = base.decompose_shared_covariance(
        observation_covariance,
        float(model_protocol["maximum_conditional_block_fraction"]),
        float(model_protocol["factor_eigenvalue_relative_tolerance"]),
    )

    cases: list[dict[str, Any]] = []
    for query_count in [int(value) for value in protocol["query_counts"]]:
        if query_count > marker_count:
            continue
        marker_indices = base.select_query_indices(marker_count, query_count)
        coordinate_indices = np.concatenate(
            [np.arange(3 * index, 3 * index + 3) for index in marker_indices]
        )
        prior = joint[np.ix_(coordinate_indices, coordinate_indices)]
        cross = joint[
            np.ix_(
                coordinate_indices,
                full_query_dimension + np.arange(3 * marker_count),
            )
        ]
        query_mean = mean[coordinate_indices]
        test_queries = test_q_all[:, marker_indices, :]

        frontiers, _, context = compute_projection_frontiers(
            base,
            prior=prior,
            cross=cross,
            conditional=conditional,
            shared=shared,
            numerical_relative_tolerance=float(protocol["numerical_relative_tolerance"]),
            validity_margin=float(protocol["posterior_validity_margin"]),
        )
        selection_reasons = _selected_rank_reasons(
            frontiers,
            context,
            protocol,
        )
        full_metrics = base._metrics(
            test_queries,
            test_y,
            query_mean,
            observation_mean,
            context["full_gain"],
            context["full_posterior"],
        )
        heldout: list[dict[str, Any]] = []
        cache = context["gain_posterior_cache"]
        for method in METHODS:
            for rank, reasons in sorted(selection_reasons[method].items()):
                gain, posterior = cache[(method, rank)]
                metrics = base._metrics(
                    test_queries,
                    test_y,
                    query_mean,
                    observation_mean,
                    gain,
                    posterior,
                )
                heldout.append(
                    {
                        "method": method,
                        "retained_rank": rank,
                        "selection_reasons": reasons,
                        "metrics": metrics,
                    }
                )

        budget_selections: list[dict[str, Any]] = []
        for budget in protocol["trace_budgets_per_query_dimension"]:
            ranks = {
                method: minimum_rank_for_budget(frontiers[method], float(budget))
                for method in METHODS
            }
            budget_selections.append(
                {
                    "trace_budget_per_query_dimension": float(budget),
                    "selected_ranks": ranks,
                    "rank_savings_vs_response_svd": (
                        ranks[METHOD_RESPONSE_SVD] - ranks[METHOD_OPTIMAL]
                    ),
                    "rank_savings_vs_factor_pca": (
                        ranks[METHOD_FACTOR_PCA] - ranks[METHOD_OPTIMAL]
                    ),
                }
            )

        cases.append(
            {
                "size": size,
                "fold": fold_index,
                "query_marker_count": query_count,
                "query_dimension": int(prior.shape[0]),
                "query_marker_indices": marker_indices.tolist(),
                "train_recording_count": len(train),
                "test_recording_count": len(test),
                "train_window_count": int(len(train_y)),
                "test_window_count": int(len(test_y)),
                "original_shared_rank": int(context["original_rank"]),
                "numerical_exact_rank": int(context["numerical_exact_rank"]),
                "conditional_block_fraction": float(conditional_fraction),
                "generalized_eigenvalues": context["generalized_eigenvalues"],
                "frontiers": frontiers,
                "budget_selections": budget_selections,
                "heldout": heldout,
                "full_heldout_metrics": full_metrics,
                "audits": {
                    "maximum_identity_relative_error": context["maximum_identity_relative_error"],
                    "maximum_optimality_violation": context["maximum_optimality_violation"],
                },
            }
        )
    return cases


def _heldout_lookup(case: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(row["method"]), int(row["retained_rank"])): row["metrics"] for row in case["heldout"]
    }


def _weighted_method_metrics(
    cases: list[Mapping[str, Any]],
    budget: float,
    method: str,
) -> dict[str, Any]:
    weights: list[float] = []
    rmse: list[float] = []
    nees: list[float] = []
    coverage: list[float] = []
    nll_per_dimension: list[float] = []
    valid = 0
    for case in cases:
        selection = next(
            row
            for row in case["budget_selections"]
            if float(row["trace_budget_per_query_dimension"]) == budget
        )
        rank = int(selection["selected_ranks"][method])
        metrics = _heldout_lookup(case)[(method, rank)]
        weight = float(case["test_window_count"])
        weights.append(weight)
        rmse.append(float(metrics["rmse_per_coordinate_m"]))
        if metrics["mean_normalized_nees"] is not None:
            valid += 1
            nees.append(float(metrics["mean_normalized_nees"]))
            coverage.append(float(metrics["joint_coverage_90"]))
            nll_per_dimension.append(float(metrics["mean_nll_nats"]) / int(case["query_dimension"]))
        else:
            nees.append(float("nan"))
            coverage.append(float("nan"))
            nll_per_dimension.append(float("nan"))
    weight_array = np.asarray(weights)
    finite = np.isfinite(nll_per_dimension)
    return {
        "case_count": len(cases),
        "valid_posterior_case_count": valid,
        "pooled_rmse_per_coordinate_m": float(
            math.sqrt(np.average(np.square(rmse), weights=weight_array))
        ),
        "weighted_mean_normalized_nees": (
            float(np.average(np.asarray(nees)[finite], weights=weight_array[finite]))
            if np.any(finite)
            else None
        ),
        "weighted_joint_coverage_90": (
            float(
                np.average(
                    np.asarray(coverage)[finite],
                    weights=weight_array[finite],
                )
            )
            if np.any(finite)
            else None
        ),
        "weighted_mean_nll_per_query_dimension": (
            float(
                np.average(
                    np.asarray(nll_per_dimension)[finite],
                    weights=weight_array[finite],
                )
            )
            if np.any(finite)
            else None
        ),
    }


def _aggregate(cases: list[dict[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    maximum_identity_error = max(
        float(case["audits"]["maximum_identity_relative_error"]) for case in cases
    )
    maximum_optimality_violation = max(
        float(case["audits"]["maximum_optimality_violation"]) for case in cases
    )
    regret: dict[str, list[float]] = {
        METHOD_RESPONSE_SVD: [],
        METHOD_FACTOR_PCA: [],
    }
    strict: Counter[str] = Counter()
    point_count: Counter[str] = Counter()
    for case in cases:
        qdim = int(case["query_dimension"])
        optimal = case["frontiers"][METHOD_OPTIMAL]
        for method in (METHOD_RESPONSE_SVD, METHOD_FACTOR_PCA):
            for rank, baseline in enumerate(case["frontiers"][method]):
                value = (
                    float(baseline["normalized_trace_loss"])
                    - float(optimal[rank]["normalized_trace_loss"])
                ) / qdim
                if value < -5e-8:
                    raise ValueError("same-rank empirical regret became negative")
                regret[method].append(max(value, 0.0))
                point_count[method] += 1
                if value > 1e-8:
                    strict[method] += 1

    budget_rows: list[dict[str, Any]] = []
    for budget_value in protocol["trace_budgets_per_query_dimension"]:
        budget = float(budget_value)
        selected = [
            next(
                row
                for row in case["budget_selections"]
                if float(row["trace_budget_per_query_dimension"]) == budget
            )
            for case in cases
        ]
        ranks = {
            method: np.asarray(
                [row["selected_ranks"][method] for row in selected],
                dtype=float,
            )
            for method in METHODS
        }
        savings_svd = ranks[METHOD_RESPONSE_SVD] - ranks[METHOD_OPTIMAL]
        savings_pca = ranks[METHOD_FACTOR_PCA] - ranks[METHOD_OPTIMAL]
        if np.min(savings_svd) < 0.0 or np.min(savings_pca) < 0.0:
            raise ValueError("the optimal frontier used more rank than a baseline")
        budget_rows.append(
            {
                "trace_budget_per_query_dimension": budget,
                "case_count": len(cases),
                "mean_selected_rank": {
                    method: float(np.mean(values)) for method, values in ranks.items()
                },
                "median_selected_rank": {
                    method: float(np.median(values)) for method, values in ranks.items()
                },
                "strict_rank_savings_vs_response_svd_count": int(
                    np.count_nonzero(savings_svd > 0.0)
                ),
                "strict_rank_savings_vs_factor_pca_count": int(np.count_nonzero(savings_pca > 0.0)),
                "mean_rank_savings_vs_response_svd": float(np.mean(savings_svd)),
                "mean_rank_savings_vs_factor_pca": float(np.mean(savings_pca)),
                "maximum_rank_savings_vs_response_svd": int(np.max(savings_svd)),
                "maximum_rank_savings_vs_factor_pca": int(np.max(savings_pca)),
                "heldout_by_method": {
                    method: _weighted_method_metrics(cases, budget, method) for method in METHODS
                },
            }
        )

    portfolio_rows: list[dict[str, Any]] = []
    primary_budget = float(protocol["primary_trace_budget_per_query_dimension"])
    for size, query_count in sorted({(case["size"], case["query_marker_count"]) for case in cases}):
        selected_cases = [
            case
            for case in cases
            if case["size"] == size and case["query_marker_count"] == query_count
        ]
        selections = [
            next(
                row
                for row in case["budget_selections"]
                if float(row["trace_budget_per_query_dimension"]) == primary_budget
            )
            for case in selected_cases
        ]
        portfolio_rows.append(
            {
                "size": size,
                "query_marker_count": query_count,
                "query_dimension": int(selected_cases[0]["query_dimension"]),
                "fold_count": len(selected_cases),
                "original_shared_rank": int(selected_cases[0]["original_shared_rank"]),
                "mean_numerical_exact_rank": float(
                    np.mean([case["numerical_exact_rank"] for case in selected_cases])
                ),
                "primary_budget": primary_budget,
                "mean_selected_rank": {
                    method: float(np.mean([row["selected_ranks"][method] for row in selections]))
                    for method in METHODS
                },
                "strict_savings_vs_response_svd_folds": int(
                    sum(row["rank_savings_vs_response_svd"] > 0 for row in selections)
                ),
                "strict_savings_vs_factor_pca_folds": int(
                    sum(row["rank_savings_vs_factor_pca"] > 0 for row in selections)
                ),
            }
        )

    exact_ratios = [
        int(case["numerical_exact_rank"]) / int(case["original_shared_rank"]) for case in cases
    ]
    return {
        "case_count": len(cases),
        "curve_point_count": sum(
            len(case["frontiers"][METHOD_OPTIMAL]) * len(METHODS) for case in cases
        ),
        "maximum_identity_relative_error": maximum_identity_error,
        "maximum_optimality_violation": maximum_optimality_violation,
        "mean_exact_rank_fraction": float(np.mean(exact_ratios)),
        "maximum_exact_rank_fraction": float(np.max(exact_ratios)),
        "same_rank_regret": {
            method: {
                "point_count": int(point_count[method]),
                "strict_improvement_count": int(strict[method]),
                "mean_regret_per_query_dimension": float(np.mean(regret[method])),
                "median_regret_per_query_dimension": float(np.median(regret[method])),
                "maximum_regret_per_query_dimension": float(np.max(regret[method])),
            }
            for method in (METHOD_RESPONSE_SVD, METHOD_FACTOR_PCA)
        },
        "budget_rows": budget_rows,
        "primary_budget_portfolios": portfolio_rows,
    }


def _summary(result: Mapping[str, Any]) -> str:
    if result["status"] != "evaluated-real-rank-distortion-frontiers":
        return (
            "# Tracking Cloth posterior rank--distortion frontier\n\n"
            f"Status: **{result['status']}**\n\n{result.get('reason', '')}\n"
        )
    aggregate = result["aggregate"]
    primary = next(
        row
        for row in aggregate["budget_rows"]
        if row["trace_budget_per_query_dimension"]
        == result["primary_trace_budget_per_query_dimension"]
    )
    lines = [
        "# Tracking Cloth posterior rank--distortion frontier",
        "",
        "Status: **evaluated on recording-disjoint real trajectories**",
        "",
        f"- Accepted recordings: {result['inventory']['accepted_recording_count']} / {result['inventory']['csv_file_count']}",
        f"- Fold/query cases: {aggregate['case_count']}",
        f"- Rank/method curve points: {aggregate['curve_point_count']}",
        f"- Maximum independent frontier-identity error: {aggregate['maximum_identity_relative_error']:.3e}",
        f"- Maximum optimality violation: {aggregate['maximum_optimality_violation']:.3e}",
        f"- Mean exact-rank/original-rank fraction: {aggregate['mean_exact_rank_fraction']:.3f}",
        "",
        f"## Primary average trace budget: {primary['trace_budget_per_query_dimension']:.3f} per query dimension",
        "",
        f"- Mean optimal rank: {primary['mean_selected_rank'][METHOD_OPTIMAL]:.3f}",
        f"- Mean response-SVD rank: {primary['mean_selected_rank'][METHOD_RESPONSE_SVD]:.3f}",
        f"- Mean factor-PCA rank: {primary['mean_selected_rank'][METHOD_FACTOR_PCA]:.3f}",
        f"- Strict rank savings versus response SVD: {primary['strict_rank_savings_vs_response_svd_count']} / {primary['case_count']}",
        f"- Strict rank savings versus factor PCA: {primary['strict_rank_savings_vs_factor_pca_count']} / {primary['case_count']}",
        "",
        "| Size | Query markers | Q dim | Original rank | Exact rank | Optimal rank | Response-SVD rank | Factor-PCA rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["primary_budget_portfolios"]:
        lines.append(
            "| {size} | {count} | {qdim} | {original} | {exact:.2f} | {optimal:.2f} | {svd:.2f} | {pca:.2f} |".format(
                size=row["size"],
                count=row["query_marker_count"],
                qdim=row["query_dimension"],
                original=row["original_shared_rank"],
                exact=row["mean_numerical_exact_rank"],
                optimal=row["mean_selected_rank"][METHOD_OPTIMAL],
                svd=row["mean_selected_rank"][METHOD_RESPONSE_SVD],
                pca=row["mean_selected_rank"][METHOD_FACTOR_PCA],
            )
        )
    lines.extend(
        [
            "",
            "Ranks are selected from each training fold without held-out outcomes. Held-out RMSE, NEES, coverage, and NLL are diagnostics rather than enforced superiority claims. All same-rank methods have identical factor/projection scalar counts.",
            "",
            "The result concerns one frozen local Gaussian query model per fold. It does not preserve observation likelihood, establish recursive exactness, validate a learned 4D provider, or demonstrate BayesianPhysTwin/Causal4D control benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    dataset_root: Path,
    protocol_path: Path,
    output_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = load_protocol(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    base, model_protocol = _load_base_module(protocol)

    csv_paths = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    file_manifest: list[dict[str, Any]] = []
    samples: list[Any] = []
    excluded_reasons: Counter[str] = Counter()
    accepted_by_size: Counter[str] = Counter()
    for path in csv_paths:
        relative_path = path.relative_to(dataset_root).as_posix()
        file_manifest.append(
            {
                "relative_path": relative_path,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        try:
            recording = base.parse_recording(path, dataset_root)
            sample = base.make_samples(recording, model_protocol)
        except (OSError, UnicodeError, ValueError) as exc:
            excluded_reasons[type(exc).__name__ + ":" + str(exc)] += 1
            continue
        samples.append(sample)
        accepted_by_size[recording.size] += 1

    inventory = {
        "csv_file_count": len(csv_paths),
        "accepted_recording_count": len(samples),
        "accepted_by_size": dict(sorted(accepted_by_size.items())),
        "excluded_recording_count": len(csv_paths) - len(samples),
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
        "dataset_manifest_sha256": _content_id(file_manifest),
        "raw_data_copied_to_output": False,
    }
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "not-evaluated",
        "source_revision": _validate_hex(
            source_revision,
            label="source_revision",
            length=40,
        ),
        "protocol_id": protocol["protocol_id"],
        "inventory": inventory,
        "primary_trace_budget_per_query_dimension": protocol[
            "primary_trace_budget_per_query_dimension"
        ],
        "claim_boundary": protocol["claim_boundary"],
    }
    cases: list[dict[str, Any]] = []
    try:
        if len(csv_paths) != int(protocol["expected_csv_files"]):
            raise ValueError("dataset CSV count differs from the registered release")
        if len(samples) != int(protocol["expected_compatible_recordings"]):
            raise ValueError("compatible recording count differs from the registered release")
        folds = int(model_protocol["fold_count"])
        minimum = int(model_protocol["minimum_recordings_per_size"])
        for size in protocol["sizes"]:
            group = [record for record in samples if record.size == size]
            if len(group) < minimum:
                raise ValueError(f"{size} has insufficient compatible recordings")
            assignments = base._fold_assignments(group, folds)
            for fold in range(folds):
                train = [record for record in group if assignments[record.relative_path] != fold]
                test = [record for record in group if assignments[record.relative_path] == fold]
                cases.extend(
                    _evaluate_fold(
                        base,
                        train=train,
                        test=test,
                        model_protocol=model_protocol,
                        protocol=protocol,
                        size=size,
                        fold_index=fold,
                    )
                )
        aggregate = _aggregate(cases, protocol)
        if aggregate["case_count"] != int(protocol["expected_case_count"]):
            raise ValueError("evaluated fold/query case count changed")
        if aggregate["maximum_identity_relative_error"] > float(
            protocol["required_maximum_identity_relative_error"]
        ):
            raise ValueError("frontier identity error exceeded the registered limit")
        if aggregate["maximum_optimality_violation"] > float(
            protocol["required_maximum_optimality_violation"]
        ):
            raise ValueError("same-rank optimality violation exceeded the registered limit")
        result.update(
            {
                "status": "evaluated-real-rank-distortion-frontiers",
                "cases": cases,
                "aggregate": aggregate,
            }
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        result.update(
            {
                "status": "technical-or-support-negative",
                "reason": str(exc),
                "cases": cases,
            }
        )

    result["result_id"] = _content_id(result)
    result_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    manifest = {
        "schema": "prob4d.tracking-cloth-rank-distortion-frontier-manifest.v1",
        "source_revision": result["source_revision"],
        "protocol_id": result["protocol_id"],
        "result_id": result["result_id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "inventory_sha256": _sha256_file(output_dir / "inventory.json"),
        "dataset_manifest_sha256": inventory["dataset_manifest_sha256"],
        "base_script_git_blob_sha1": BASE_SCRIPT_GIT_BLOB_SHA1,
        "base_protocol_git_blob_sha1": BASE_PROTOCOL_GIT_BLOB_SHA1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "raw_data_copied_to_output": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root is unavailable: {args.dataset_root}")
    result = run(
        dataset_root=args.dataset_root.resolve(),
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        source_revision=args.source_revision,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_id": result["result_id"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "evaluated-real-rank-distortion-frontiers" else 2


if __name__ == "__main__":
    raise SystemExit(main())
