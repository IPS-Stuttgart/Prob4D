"""Deterministic covariance intersection for uncertain ``Sim(3)`` estimates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .sim3 import Sim3

FloatArray = NDArray[np.floating]
_GAUGE_DIMENSION = 7


@dataclass(frozen=True)
class _Candidate:
    original_index: int
    transform: Sim3
    covariance: np.ndarray
    log_determinant: float
    key: bytes


def _validated_covariance(
    values: FloatArray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (_GAUGE_DIMENSION, _GAUGE_DIMENSION):
        raise ValueError(f"{name} must have shape (7, 7)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(1.0, float(np.max(np.abs(symmetric))))
    symmetry_tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if not np.allclose(
        matrix,
        symmetric,
        atol=symmetry_tolerance,
        rtol=1e-12,
    ):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues), initial=0.0)),
    )
    negative_tolerance = 128.0 * np.finfo(np.float64).eps * spectral_scale
    if float(np.min(eigenvalues)) < -negative_tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    eigenvalue_floor = max(
        float(np.max(eigenvalues, initial=0.0)) * 1e-12,
        1e-12,
    )
    regularized = np.maximum(eigenvalues, eigenvalue_floor)
    precision = (eigenvectors * (1.0 / regularized)) @ eigenvectors.T
    log_determinant = float(np.sum(np.log(regularized)))
    return symmetric, 0.5 * (precision + precision.T), log_determinant


def _canonical_candidate_key(transform: Sim3, covariance: np.ndarray) -> bytes:
    values = np.concatenate((transform.as_vector(), covariance.reshape(-1)))
    return np.asarray(values, dtype="<f8").tobytes()


def _central_numerical_jacobian(function, vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    baseline = np.asarray(function(values), dtype=np.float64)
    jacobian = np.empty((baseline.size, values.size), dtype=np.float64)
    for index in range(values.size):
        step = 1e-6 * max(1.0, abs(float(values[index])))
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.asarray(function(plus), dtype=np.float64)
            - np.asarray(function(minus), dtype=np.float64)
        ) / (2.0 * step)
    return jacobian


def _information_inverse(information: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(
            "covariance-intersection information must be positive definite"
        )
    covariance = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def _pair_optimum(
    base_information: np.ndarray,
    first_precision: np.ndarray,
    second_precision: np.ndarray,
    total_weight: float,
    lower: float,
    upper: float,
    *,
    line_search_iterations: int,
) -> float:
    direction = first_precision - second_precision
    origin = base_information + total_weight * second_precision

    def derivative(weight: float) -> float:
        covariance = _information_inverse(origin + weight * direction)
        return -float(np.einsum("ij,ji->", covariance, direction))

    if derivative(lower) >= 0.0:
        return lower
    if derivative(upper) <= 0.0:
        return upper
    left, right = lower, upper
    for _ in range(line_search_iterations):
        midpoint = 0.5 * (left + right)
        if derivative(midpoint) <= 0.0:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def _optimize_weights(
    precisions: np.ndarray,
    *,
    minimum_weight: float,
    max_sweeps: int,
    line_search_iterations: int,
    tolerance: float,
) -> np.ndarray:
    count = len(precisions)
    if count == 1:
        return np.ones(1, dtype=np.float64)
    if not np.isfinite(minimum_weight) or minimum_weight < 0.0:
        raise ValueError("minimum_weight must be finite and non-negative")
    if minimum_weight * count >= 1.0:
        raise ValueError("minimum_weight must leave positive simplex mass")
    if max_sweeps < 1 or line_search_iterations < 1:
        raise ValueError("CI iteration counts must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("CI tolerance must be finite and positive")

    weights = np.full(count, 1.0 / count, dtype=np.float64)
    for _ in range(max_sweeps):
        maximum_change = 0.0
        for first in range(count - 1):
            for second in range(first + 1, count):
                total = float(weights[first] + weights[second])
                lower = minimum_weight
                upper = total - minimum_weight
                if upper <= lower:
                    continue
                base = np.einsum("i,ijk->jk", weights, precisions)
                base -= weights[first] * precisions[first]
                base -= weights[second] * precisions[second]
                optimized = _pair_optimum(
                    base,
                    precisions[first],
                    precisions[second],
                    total,
                    lower,
                    upper,
                    line_search_iterations=line_search_iterations,
                )
                change = abs(optimized - float(weights[first]))
                weights[first] = optimized
                weights[second] = total - optimized
                maximum_change = max(maximum_change, change)
        if maximum_change <= tolerance:
            break
    weights /= float(np.sum(weights))
    return weights


def fuse_sim3_covariance_intersection(
    candidates: Sequence[tuple[Sim3, FloatArray]],
    *,
    minimum_weight: float = 0.0,
    max_sweeps: int = 64,
    line_search_iterations: int = 48,
    tolerance: float = 1e-10,
) -> tuple[Sim3, np.ndarray, np.ndarray]:
    """Fuse correlated ``Sim(3)`` estimates in one deterministic local chart.

    Candidate ordering is canonicalized before optimization. Means and
    covariances are transported into the tangent chart of the most precise
    candidate, fused by n-way covariance intersection, and transported back to
    the global seven-vector coordinates. Returned weights correspond to the
    caller's original candidate order.
    """

    if not candidates:
        raise ValueError("at least one Sim3 candidate is required")

    entries: list[_Candidate] = []
    for original_index, (transform, covariance) in enumerate(candidates):
        if not isinstance(transform, Sim3):
            raise TypeError("candidate transforms must be Sim3 instances")
        matrix, _, log_determinant = _validated_covariance(
            covariance,
            name=f"candidate covariance {original_index}",
        )
        entries.append(
            _Candidate(
                original_index=original_index,
                transform=transform,
                covariance=matrix,
                log_determinant=log_determinant,
                key=_canonical_candidate_key(transform, matrix),
            )
        )
    entries.sort(key=lambda item: item.key)
    reference = min(
        entries,
        key=lambda item: (item.log_determinant, item.key),
    ).transform
    reference_inverse = reference.inverse()

    local_means: list[np.ndarray] = []
    local_precisions: list[np.ndarray] = []
    for position, entry in enumerate(entries):
        vector = entry.transform.as_vector()

        def to_local(value: np.ndarray) -> np.ndarray:
            return reference_inverse.compose(Sim3.from_vector(value)).as_vector()

        local_mean = to_local(vector)
        jacobian = _central_numerical_jacobian(to_local, vector)
        local_covariance = jacobian @ entry.covariance @ jacobian.T
        _, precision, _ = _validated_covariance(
            local_covariance,
            name=f"transported candidate covariance {position}",
        )
        local_means.append(local_mean)
        local_precisions.append(precision)

    mean_array = np.asarray(local_means, dtype=np.float64)
    precision_array = np.asarray(local_precisions, dtype=np.float64)
    weights = _optimize_weights(
        precision_array,
        minimum_weight=minimum_weight,
        max_sweeps=max_sweeps,
        line_search_iterations=line_search_iterations,
        tolerance=tolerance,
    )
    information = np.einsum("i,ijk->jk", weights, precision_array)
    local_covariance = _information_inverse(information)
    information_vector = np.einsum(
        "i,ijk,ik->j",
        weights,
        precision_array,
        mean_array,
    )
    local_mean = local_covariance @ information_vector
    fused_transform = reference.compose(Sim3.from_vector(local_mean))

    def from_local(value: np.ndarray) -> np.ndarray:
        return reference.compose(Sim3.from_vector(value)).as_vector()

    output_jacobian = _central_numerical_jacobian(from_local, local_mean)
    output_covariance = output_jacobian @ local_covariance @ output_jacobian.T
    output_covariance = 0.5 * (output_covariance + output_covariance.T)
    original_weights = np.empty(len(entries), dtype=np.float64)
    for sorted_index, entry in enumerate(entries):
        original_weights[entry.original_index] = weights[sorted_index]
    return fused_transform, output_covariance, original_weights


__all__ = ["fuse_sim3_covariance_intersection"]
