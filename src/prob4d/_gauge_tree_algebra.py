"""Matrix-free algebra for sparse gauge-tree priors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    FloatArray,
    coerce_gauge_blocks,
    restore_gauge_layout,
)


def covariance_action(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    value: Any,
) -> FloatArray:
    gauge_count = len(parents)
    blocks, layout = coerce_gauge_blocks(value, gauge_count=gauge_count, name="value")
    adjoint = blocks.copy()
    for child in range(gauge_count - 1, 0, -1):
        parent = int(parents[child])
        adjoint[parent] += transitions[child].T @ adjoint[child]
    weighted = np.empty_like(adjoint)
    for index in range(gauge_count):
        scale = scales[index]
        weighted[index] = scale @ (scale.T @ adjoint[index])
    result = np.empty_like(weighted)
    result[0] = weighted[0]
    for child in range(1, gauge_count):
        parent = int(parents[child])
        result[child] = transitions[child] @ result[parent] + weighted[child]
    return restore_gauge_layout(result, layout)


def innovation_coordinates(
    parents: np.ndarray,
    transitions: np.ndarray,
    value: Any,
) -> FloatArray:
    blocks, layout = coerce_gauge_blocks(value, gauge_count=len(parents), name="value")
    innovations = blocks.copy()
    for child in range(1, len(parents)):
        parent = int(parents[child])
        innovations[child] -= transitions[child] @ blocks[parent]
    return restore_gauge_layout(innovations, layout)


def standardized_innovations(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    value: Any,
) -> FloatArray:
    innovations, layout = coerce_gauge_blocks(
        innovation_coordinates(parents, transitions, value),
        gauge_count=len(parents),
        name="innovation coordinates",
    )
    standardized = np.empty_like(innovations)
    for index in range(len(parents)):
        standardized[index] = np.linalg.solve(scales[index], innovations[index])
    return restore_gauge_layout(standardized, layout)


def information_action(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    value: Any,
) -> FloatArray:
    blocks, layout = coerce_gauge_blocks(value, gauge_count=len(parents), name="value")
    innovations = blocks.copy()
    for child in range(1, len(parents)):
        parent = int(parents[child])
        innovations[child] -= transitions[child] @ blocks[parent]
    weighted = np.empty_like(innovations)
    for index in range(len(parents)):
        whitened = np.linalg.solve(scales[index], innovations[index])
        weighted[index] = np.linalg.solve(scales[index].T, whitened)
    result = weighted.copy()
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[parent] -= transitions[child].T @ weighted[child]
    return restore_gauge_layout(result, layout)


def information_quadratic(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    value: Any,
) -> float:
    blocks, layout = coerce_gauge_blocks(value, gauge_count=len(parents), name="value")
    if layout not in {"flat-vector", "block-vector"}:
        raise ValueError("information_quadratic requires exactly one gauge-state vector")
    standardized = np.asarray(
        standardized_innovations(parents, transitions, scales, blocks[:, :, 0])
    )
    return float(np.sum(np.square(standardized)))


def log_determinant_covariance(scales: np.ndarray) -> float:
    diagonal = np.diagonal(scales, axis1=1, axis2=2)
    return float(2.0 * np.sum(np.log(diagonal)))


def innovation_covariance_blocks(scales: np.ndarray) -> FloatArray:
    return np.einsum("kij,klj->kil", scales, scales, optimize=True)


def diagonal_covariance_blocks(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
) -> FloatArray:
    result = np.empty((len(parents), GAUGE_DIMENSION, GAUGE_DIMENSION), dtype=np.float64)
    result[0] = scales[0] @ scales[0].T
    for child in range(1, len(parents)):
        parent = int(parents[child])
        covariance = (
            transitions[child] @ result[parent] @ transitions[child].T
            + scales[child] @ scales[child].T
        )
        result[child] = 0.5 * (covariance + covariance.T)
    return result


def sample(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    *,
    seed: int,
    sample_count: int,
) -> FloatArray:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a genuine integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be a genuine integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    generator = np.random.default_rng(int(seed))
    standard = generator.standard_normal((int(sample_count), len(parents), GAUGE_DIMENSION))
    innovations = np.einsum("kij,nkj->nki", scales, standard, optimize=True)
    result = np.empty_like(innovations)
    result[:, 0] = innovations[:, 0]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[:, child] = (
            np.einsum(
                "ij,nj->ni",
                transitions[child],
                result[:, parent],
                optimize=True,
            )
            + innovations[:, child]
        )
    return result
