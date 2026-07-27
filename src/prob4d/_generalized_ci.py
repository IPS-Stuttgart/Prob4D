"""Deterministic generalized covariance intersection for dense estimates."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from .covariance import regularized_inverse_psd

FloatArray = NDArray[np.floating]


def _inverse_psd(values: FloatArray, *, name: str) -> FloatArray:
    return regularized_inverse_psd(values, name=name, eigenvalue_floor=1e-12)


def _canonical_order(
    means: FloatArray,
    covariances: FloatArray,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    keys: list[bytes] = []
    for index in range(means.shape[0]):
        digest = hashlib.sha256()
        for value in (means[index], covariances[index]):
            contiguous = np.ascontiguousarray(value)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(memoryview(contiguous).cast("B"))
        keys.append(digest.digest())
    order = np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    return order, inverse


def _project_simplex(values: FloatArray) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("simplex projection requires a finite nonempty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    positive = ordered - cumulative / np.arange(1, vector.size + 1) > 0.0
    rho = int(np.flatnonzero(positive)[-1]) if np.any(positive) else 0
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    projected /= float(np.sum(projected))
    return projected


def _objective_and_gradient(
    weights: FloatArray,
    information: FloatArray,
) -> tuple[float, FloatArray]:
    combined = np.einsum("k,nkij->nij", weights, information, optimize=True)
    covariance = _inverse_psd(
        combined,
        name="generalized covariance-intersection information",
    )
    sign, log_determinant = np.linalg.slogdet(covariance)
    if np.any(sign <= 0.0) or not np.all(np.isfinite(log_determinant)):
        raise ValueError("generalized covariance intersection lost positive definiteness")
    gradient = -np.mean(
        np.einsum("nij,nkji->nk", covariance, information, optimize=True),
        axis=0,
    )
    return float(np.mean(log_determinant)), gradient


def _optimize_weights(
    information: FloatArray,
    *,
    minimum_weight: float,
    maximum_iterations: int,
    tolerance: float,
) -> FloatArray:
    contributor_count = information.shape[1]
    free_mass = 1.0 - contributor_count * minimum_weight
    simplex = np.full(contributor_count, 1.0 / contributor_count, dtype=np.float64)
    weights = minimum_weight + free_mass * simplex
    score, gradient = _objective_and_gradient(weights, information)
    step_hint = 1.0

    for _ in range(maximum_iterations):
        simplex_gradient = free_mass * (gradient - float(np.mean(gradient)))
        projected = _project_simplex(simplex - simplex_gradient)
        if np.linalg.norm(projected - simplex, ord=np.inf) <= tolerance:
            break
        step = step_hint
        accepted = False
        for _ in range(40):
            candidate_simplex = _project_simplex(simplex - step * simplex_gradient)
            delta = candidate_simplex - simplex
            candidate_weights = minimum_weight + free_mass * candidate_simplex
            candidate_score, candidate_gradient = _objective_and_gradient(
                candidate_weights,
                information,
            )
            directional_derivative = float(simplex_gradient @ delta)
            if candidate_score <= score + 1e-4 * directional_derivative + 1e-14:
                simplex = candidate_simplex
                weights = candidate_weights
                score = candidate_score
                gradient = candidate_gradient
                step_hint = min(1.5 * step, 64.0)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        if np.linalg.norm(delta) <= tolerance * max(1.0, np.linalg.norm(simplex)):
            break
    return weights


def fuse_nway_covariance_intersection(
    means: FloatArray,
    covariances: FloatArray,
    *,
    minimum_weight: float = 0.0,
    weight_sample_size: int = 4_096,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
    chunk_size: int = 16_384,
    canonicalize: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fuse at least three estimates through one generalized-CI simplex."""

    values = np.asarray(means, dtype=np.float64)
    matrices = np.asarray(covariances, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 3:
        raise ValueError("n-way CI means must have shape (K, ..., D) with K >= 3")
    if not np.all(np.isfinite(values)):
        raise ValueError("n-way CI means must be finite")
    contributor_count = values.shape[0]
    dimension = values.shape[-1]
    if matrices.shape != values.shape + (dimension,):
        raise ValueError("n-way CI covariance shape does not match means")
    if not 0.0 <= minimum_weight < 1.0 / contributor_count:
        raise ValueError("minimum_weight must lie in [0, 1 / contributor_count)")
    if weight_sample_size < 1 or maximum_iterations < 1 or chunk_size < 1:
        raise ValueError("n-way CI sample, iteration, and chunk sizes must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("n-way CI tolerance must be finite and positive")

    inverse_order: NDArray[np.int64] | None = None
    if canonicalize:
        order, inverse_order = _canonical_order(values, matrices)
        values = values[order]
        matrices = matrices[order]

    leading_shape = values.shape[1:-1]
    flat_means = values.reshape(contributor_count, -1, dimension).transpose(1, 0, 2)
    flat_covariances = matrices.reshape(
        contributor_count,
        -1,
        dimension,
        dimension,
    ).transpose(1, 0, 2, 3)
    if flat_means.shape[0] == 0:
        raise ValueError("n-way CI requires at least one sample")
    information = _inverse_psd(flat_covariances, name="n-way CI covariance")
    sample = (
        np.arange(flat_means.shape[0])
        if flat_means.shape[0] <= weight_sample_size
        else np.linspace(
            0,
            flat_means.shape[0] - 1,
            weight_sample_size,
            dtype=np.int64,
        )
    )
    weights = _optimize_weights(
        information[sample],
        minimum_weight=minimum_weight,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )

    output_mean = np.empty((flat_means.shape[0], dimension), dtype=np.float64)
    output_covariance = np.empty(
        (flat_means.shape[0], dimension, dimension),
        dtype=np.float64,
    )
    for start in range(0, flat_means.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_means.shape[0])
        combined_information = np.einsum(
            "k,nkij->nij",
            weights,
            information[start:stop],
            optimize=True,
        )
        covariance = _inverse_psd(combined_information, name="n-way CI information")
        information_vector = np.einsum(
            "k,nkij,nkj->ni",
            weights,
            information[start:stop],
            flat_means[start:stop],
            optimize=True,
        )
        output_mean[start:stop] = np.einsum(
            "nij,nj->ni",
            covariance,
            information_vector,
            optimize=True,
        )
        output_covariance[start:stop] = covariance

    if inverse_order is not None:
        weights = weights[inverse_order]
    return (
        output_mean.reshape(leading_shape + (dimension,)),
        output_covariance.reshape(leading_shape + (dimension, dimension)),
        weights,
    )


__all__ = ["fuse_nway_covariance_intersection"]
