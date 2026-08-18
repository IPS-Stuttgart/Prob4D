"""Deterministic nonlinear closure numerics for joint ``Sim(3)`` chains."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ._gauge_linearization_contract import (
    _HARD_BRANCH_CUT_TOLERANCE,
    FloatArray,
    GaugeLinearizationCaseV1,
    GaugeLinearizationPolicyV1,
    JsonReport,
    _require_symmetric_psd,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._observation_factor_bundle import sim3_point_jacobian
from .composition_jacobian import analytic_sim3_compose_jacobians
from .sim3 import Sim3


def _compose_transform_chain(transform_vectors: FloatArray) -> tuple[Sim3, float]:
    transforms = [Sim3.from_vector(vector) for vector in transform_vectors]
    current = transforms[0]
    clearance = math.pi - float(np.linalg.norm(current.as_vector()[1:4]))
    for transform in transforms[1:]:
        clearance = min(
            clearance,
            math.pi - float(np.linalg.norm(transform.as_vector()[1:4])),
        )
        current = current.compose(transform)
        clearance = min(
            clearance,
            math.pi - float(np.linalg.norm(current.as_vector()[1:4])),
        )
    return current, max(0.0, clearance)


def linearize_sim3_chain(transform_vectors: FloatArray) -> tuple[Sim3, FloatArray]:
    """Return the composed transform and derivative against all input coordinates."""

    vectors = np.asarray(transform_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 7 or vectors.shape[0] < 1:
        raise ValueError("transform_vectors must have shape (K, 7) with K >= 1")
    transforms = [Sim3.from_vector(vector) for vector in vectors]
    derivative = np.zeros((7, 7 * len(transforms)), dtype=np.float64)
    derivative[:, :7] = np.eye(7)
    current = transforms[0]
    for index, transform in enumerate(transforms[1:], start=1):
        parent_jacobian, relative_jacobian = analytic_sim3_compose_jacobians(
            current,
            transform,
        )
        updated = np.zeros_like(derivative)
        updated[:, : 7 * index] = parent_jacobian @ derivative[:, : 7 * index]
        updated[:, 7 * index : 7 * (index + 1)] = relative_jacobian
        derivative = updated
        current = current.compose(transform)
    return current, derivative


def _pivoted_psd_root(covariance: FloatArray, *, relative_tolerance: float) -> FloatArray:
    matrix = _require_symmetric_psd(covariance, name="covariance")
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0)
    tolerance = relative_tolerance * scale
    residual = matrix.copy()
    columns: list[FloatArray] = []
    for _ in range(len(matrix)):
        diagonal = np.diag(residual)
        pivot_index = int(np.argmax(diagonal))
        pivot = float(diagonal[pivot_index])
        if pivot <= tolerance:
            break
        column = residual[:, pivot_index] / math.sqrt(pivot)
        columns.append(column)
        residual -= np.outer(column, column)
        residual = 0.5 * (residual + residual.T)
        if float(np.min(np.diag(residual), initial=0.0)) < -100.0 * tolerance:
            raise ValueError("covariance pivoted root became numerically indefinite")
    root = (
        np.zeros((len(matrix), 0), dtype=np.float64)
        if not columns
        else np.stack(columns, axis=1)
    )
    if not np.allclose(
        root @ root.T,
        matrix,
        atol=max(100.0 * tolerance, 1e-14),
        rtol=1e-9,
    ):
        raise ValueError("covariance rank tolerance discards material variance")
    return root


def _sigma_points(root: FloatArray) -> tuple[tuple[FloatArray, float], ...]:
    rank = int(root.shape[1])
    if rank == 0:
        return ((np.zeros(root.shape[0], dtype=np.float64), 1.0),)
    radius = math.sqrt(float(rank))
    weight = 1.0 / (2.0 * rank)
    return tuple(
        (sign * radius * root[:, index], weight)
        for index in range(rank)
        for sign in (-1.0, 1.0)
    )


def _covariance_metrics(
    linear_covariance: FloatArray,
    nonlinear_covariance: FloatArray,
    *,
    relative_tolerance: float,
) -> tuple[float, float, float]:
    linear = 0.5 * (linear_covariance + linear_covariance.T)
    nonlinear = 0.5 * (nonlinear_covariance + nonlinear_covariance.T)
    denominator = max(
        float(np.linalg.norm(linear, ord="fro")),
        float(np.linalg.norm(nonlinear, ord="fro")),
        np.finfo(np.float64).eps,
    )
    frobenius = float(np.linalg.norm(nonlinear - linear, ord="fro")) / denominator
    eigenvalues, eigenvectors = np.linalg.eigh(linear)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    supported = eigenvalues > relative_tolerance * scale
    directional = 0.0
    if np.any(supported):
        basis = eigenvectors[:, supported] / np.sqrt(eigenvalues[supported])[None, :]
        ratio_matrix = basis.T @ nonlinear @ basis
        ratio_matrix = 0.5 * (ratio_matrix + ratio_matrix.T)
        ratios = np.linalg.eigvalsh(ratio_matrix)
        directional = float(np.max(np.abs(ratios - 1.0), initial=0.0))
    null_basis = eigenvectors[:, ~supported]
    nonlinear_trace = max(float(np.trace(nonlinear)), 0.0)
    outside = 0.0
    if nonlinear_trace > np.finfo(np.float64).eps and null_basis.shape[1] > 0:
        outside = max(float(np.trace(null_basis.T @ nonlinear @ null_basis)), 0.0)
        outside = min(1.0, outside / nonlinear_trace)
    return frobenius, directional, outside


def _normalized_mean_shift(
    linear_mean: FloatArray,
    nonlinear_mean: FloatArray,
    linear_covariance: FloatArray,
    nonlinear_covariance: FloatArray,
) -> float:
    scale = max(
        float(np.trace(linear_covariance)),
        float(np.trace(nonlinear_covariance)),
        np.finfo(np.float64).eps,
    )
    return float(np.linalg.norm(nonlinear_mean - linear_mean)) / math.sqrt(scale)


def _branch_failure_report(
    case: GaugeLinearizationCaseV1,
    *,
    gauge_rank: int,
    clearance: float,
) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "group_id": case.group_id,
        "transform_count": case.transform_count,
        "point_count": case.point_count,
        "query_dimension": case.query_dimension,
        "gauge_rank": gauge_rank,
        "minimum_branch_cut_clearance_radians": clearance,
        "branch_cut_safe": False,
        "maximum_point_normalized_mean_shift": None,
        "maximum_point_relative_covariance_frobenius_error": None,
        "maximum_point_directional_variance_ratio_deviation": None,
        "maximum_point_variance_outside_linear_support_fraction": None,
        "query_normalized_mean_shift": None,
        "query_relative_covariance_frobenius_error": None,
        "query_directional_variance_ratio_deviation": None,
        "query_variance_outside_linear_support_fraction": None,
        "closure_passed": False,
        "failure_reasons": ["mean-chain-branch-cut"],
    }


def evaluate_gauge_linearization_case(
    case: GaugeLinearizationCaseV1,
    policy: GaugeLinearizationPolicyV1,
) -> dict[str, object]:
    """Compare analytic propagation with deterministic nonlinear cubature."""

    if not isinstance(case, GaugeLinearizationCaseV1):
        raise ValueError("case must be GaugeLinearizationCaseV1")
    if not isinstance(policy, GaugeLinearizationPolicyV1):
        raise ValueError("policy must be GaugeLinearizationPolicyV1")
    root = _pivoted_psd_root(
        case.joint_covariance,
        relative_tolerance=policy.covariance_rank_relative_tolerance,
    )
    gauge_rank = int(root.shape[1])
    mean_transform, mean_clearance = _compose_transform_chain(case.transform_vectors)
    if mean_clearance <= _HARD_BRANCH_CUT_TOLERANCE:
        return _branch_failure_report(
            case,
            gauge_rank=gauge_rank,
            clearance=mean_clearance,
        )

    composed, chain_jacobian = linearize_sim3_chain(case.transform_vectors)
    point_jacobian = np.einsum(
        "nac,cd->nad",
        sim3_point_jacobian(composed, case.points_local_m),
        chain_jacobian,
        optimize=True,
    )
    linear_mean = mean_transform.transform_points(case.points_local_m)
    linear_covariance = np.einsum(
        "nai,ij,nbj->nab",
        point_jacobian,
        case.joint_covariance,
        point_jacobian,
        optimize=True,
    )
    nonlinear_mean = np.zeros_like(linear_mean)
    nonlinear_second = np.zeros((case.point_count, 3, 3), dtype=np.float64)

    query_linear_mean: FloatArray | None = None
    query_linear_covariance: FloatArray | None = None
    query_nonlinear_mean: FloatArray | None = None
    query_nonlinear_second: FloatArray | None = None
    if case.query_jacobian is not None:
        query_linear_mean = np.einsum(
            "qni,ni->q",
            case.query_jacobian,
            linear_mean,
            optimize=True,
        )
        query_jacobian = np.einsum(
            "qni,nid->qd",
            case.query_jacobian,
            point_jacobian,
            optimize=True,
        )
        query_linear_covariance = (
            query_jacobian @ case.joint_covariance @ query_jacobian.T
        )
        query_nonlinear_mean = np.zeros(case.query_dimension, dtype=np.float64)
        query_nonlinear_second = np.zeros(
            (case.query_dimension, case.query_dimension),
            dtype=np.float64,
        )

    minimum_clearance = mean_clearance
    flat_mean = case.transform_vectors.reshape(-1)
    for perturbation, weight in _sigma_points(root):
        vectors = (flat_mean + perturbation).reshape(case.transform_vectors.shape)
        transform, clearance = _compose_transform_chain(vectors)
        minimum_clearance = min(minimum_clearance, clearance)
        points = transform.transform_points(case.points_local_m)
        nonlinear_mean += weight * points
        nonlinear_second += weight * np.einsum(
            "ni,nj->nij",
            points,
            points,
            optimize=True,
        )
        if case.query_jacobian is not None:
            assert query_nonlinear_mean is not None
            assert query_nonlinear_second is not None
            query = np.einsum(
                "qni,ni->q",
                case.query_jacobian,
                points,
                optimize=True,
            )
            query_nonlinear_mean += weight * query
            query_nonlinear_second += weight * np.outer(query, query)

    nonlinear_covariance = nonlinear_second - np.einsum(
        "ni,nj->nij",
        nonlinear_mean,
        nonlinear_mean,
        optimize=True,
    )
    nonlinear_covariance = 0.5 * (
        nonlinear_covariance + np.swapaxes(nonlinear_covariance, 1, 2)
    )
    point_metrics = []
    for index in range(case.point_count):
        point_metrics.append(
            (
                _normalized_mean_shift(
                    linear_mean[index],
                    nonlinear_mean[index],
                    linear_covariance[index],
                    nonlinear_covariance[index],
                ),
                *_covariance_metrics(
                    linear_covariance[index],
                    nonlinear_covariance[index],
                    relative_tolerance=policy.covariance_rank_relative_tolerance,
                ),
            )
        )
    point_columns = tuple(zip(*point_metrics, strict=True))
    point_maxima = tuple(max(column) for column in point_columns)

    query_metrics: tuple[float, float, float, float] | None = None
    if case.query_jacobian is not None:
        assert query_linear_mean is not None
        assert query_linear_covariance is not None
        assert query_nonlinear_mean is not None
        assert query_nonlinear_second is not None
        query_nonlinear_covariance = query_nonlinear_second - np.outer(
            query_nonlinear_mean,
            query_nonlinear_mean,
        )
        query_nonlinear_covariance = 0.5 * (
            query_nonlinear_covariance + query_nonlinear_covariance.T
        )
        query_metrics = (
            _normalized_mean_shift(
                query_linear_mean,
                query_nonlinear_mean,
                query_linear_covariance,
                query_nonlinear_covariance,
            ),
            *_covariance_metrics(
                query_linear_covariance,
                query_nonlinear_covariance,
                relative_tolerance=policy.covariance_rank_relative_tolerance,
            ),
        )

    reasons: list[str] = []
    if minimum_clearance < policy.minimum_branch_cut_clearance_radians:
        reasons.append("branch-cut-clearance")
    metric_rules = (
        (point_maxima[0], policy.maximum_normalized_mean_shift, "point-mean-shift"),
        (
            point_maxima[1],
            policy.maximum_relative_covariance_frobenius_error,
            "point-covariance-frobenius",
        ),
        (
            point_maxima[2],
            policy.maximum_directional_variance_ratio_deviation,
            "point-directional-variance",
        ),
        (
            point_maxima[3],
            policy.maximum_variance_outside_linear_support_fraction,
            "point-off-support-variance",
        ),
    )
    reasons.extend(reason for value, threshold, reason in metric_rules if value > threshold)
    if query_metrics is not None:
        query_rules = (
            (query_metrics[0], policy.maximum_normalized_mean_shift, "query-mean-shift"),
            (
                query_metrics[1],
                policy.maximum_relative_covariance_frobenius_error,
                "query-covariance-frobenius",
            ),
            (
                query_metrics[2],
                policy.maximum_directional_variance_ratio_deviation,
                "query-directional-variance",
            ),
            (
                query_metrics[3],
                policy.maximum_variance_outside_linear_support_fraction,
                "query-off-support-variance",
            ),
        )
        reasons.extend(
            reason for value, threshold, reason in query_rules if value > threshold
        )

    return {
        "case_id": case.case_id,
        "group_id": case.group_id,
        "transform_count": case.transform_count,
        "point_count": case.point_count,
        "query_dimension": case.query_dimension,
        "gauge_rank": gauge_rank,
        "minimum_branch_cut_clearance_radians": minimum_clearance,
        "branch_cut_safe": minimum_clearance
        >= policy.minimum_branch_cut_clearance_radians,
        "maximum_point_normalized_mean_shift": point_maxima[0],
        "maximum_point_relative_covariance_frobenius_error": point_maxima[1],
        "maximum_point_directional_variance_ratio_deviation": point_maxima[2],
        "maximum_point_variance_outside_linear_support_fraction": point_maxima[3],
        "query_normalized_mean_shift": None if query_metrics is None else query_metrics[0],
        "query_relative_covariance_frobenius_error": (
            None if query_metrics is None else query_metrics[1]
        ),
        "query_directional_variance_ratio_deviation": (
            None if query_metrics is None else query_metrics[2]
        ),
        "query_variance_outside_linear_support_fraction": (
            None if query_metrics is None else query_metrics[3]
        ),
        "closure_passed": not reasons,
        "failure_reasons": reasons,
    }


def _case_reports(
    cases: Sequence[GaugeLinearizationCaseV1],
    policy: GaugeLinearizationPolicyV1,
) -> tuple[JsonReport, ...]:
    return tuple(
        frozen_finite_json_mapping(
            evaluate_gauge_linearization_case(case, policy),
            name="gauge linearization case report",
        )
        for case in cases
    )


def _group_reports(reports: Sequence[JsonReport]) -> tuple[JsonReport, ...]:
    grouped: dict[str, list[JsonReport]] = {}
    for report in reports:
        grouped.setdefault(str(report["group_id"]), []).append(report)
    result = []
    for group_id in sorted(grouped):
        group = sorted(grouped[group_id], key=lambda report: str(report["case_id"]))
        passing = sum(bool(report["closure_passed"]) for report in group)
        result.append(
            frozen_finite_json_mapping(
                {
                    "group_id": group_id,
                    "case_ids": [str(report["case_id"]) for report in group],
                    "passing_case_count": passing,
                    "closure_passed": passing == len(group),
                },
                name="gauge linearization group report",
            )
        )
    return tuple(result)


def _reports_as_plain(reports: Sequence[JsonReport]) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], plain_json(report)) for report in reports]
