"""Strict construction and dense-parity checks for gauge-tree priors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._gauge_tree_algebra import covariance_action
from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    joint_covariance_sha256,
    strict_scale_tril,
    validate_gauge_ids,
    validate_parent_indices,
)


def transition_factors(
    *,
    gauge_ids: Sequence[str],
    parent_indices: Any,
    transition_matrices: Any,
    innovation_covariances: Any,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    ids = validate_gauge_ids(gauge_ids)
    parents = validate_parent_indices(parent_indices, gauge_count=len(ids))
    transitions = np.asarray(transition_matrices, dtype=np.float64)
    covariances = np.asarray(innovation_covariances, dtype=np.float64)
    expected = (len(ids), GAUGE_DIMENSION, GAUGE_DIMENSION)
    if transitions.shape != expected:
        raise ValueError(f"transition_matrices must have shape {expected}")
    if covariances.shape != expected:
        raise ValueError(f"innovation_covariances must have shape {expected}")
    scales = np.empty_like(covariances)
    for index, gauge_id in enumerate(ids):
        label = (
            "root gauge covariance"
            if index == 0
            else (f"innovation covariance for gauge {gauge_id!r}")
        )
        scales[index] = strict_scale_tril(covariances[index], name=label)
    return ids, parents, transitions, scales


def dense_factors(
    *,
    gauge_ids: Sequence[str],
    parent_indices: Any,
    joint_covariance: Any,
    parity_atol: float,
    parity_rtol: float,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, str]:
    ids = validate_gauge_ids(gauge_ids)
    parents = validate_parent_indices(parent_indices, gauge_count=len(ids))
    matrix = np.asarray(joint_covariance, dtype=np.float64)
    dimension = GAUGE_DIMENSION * len(ids)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"joint_covariance must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("joint_covariance must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(matrix, symmetric, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError("joint_covariance must be symmetric")
    if parity_atol < 0.0 or parity_rtol < 0.0:
        raise ValueError("parity tolerances must be nonnegative")
    transitions: np.ndarray = np.zeros(
        (len(ids), GAUGE_DIMENSION, GAUGE_DIMENSION),
        dtype=np.float64,
    )
    scales = np.empty_like(transitions)
    scales[0] = strict_scale_tril(
        symmetric[:GAUGE_DIMENSION, :GAUGE_DIMENSION],
        name="root gauge covariance",
    )
    for child in range(1, len(ids)):
        parent = int(parents[child])
        parent_slice = slice(GAUGE_DIMENSION * parent, GAUGE_DIMENSION * (parent + 1))
        child_slice = slice(GAUGE_DIMENSION * child, GAUGE_DIMENSION * (child + 1))
        parent_covariance = symmetric[parent_slice, parent_slice]
        child_parent = symmetric[child_slice, parent_slice]
        try:
            transition = np.linalg.solve(parent_covariance, child_parent.T).T
        except np.linalg.LinAlgError as error:
            raise ValueError(f"parent covariance for gauge {ids[child]!r} is singular") from error
        innovation = symmetric[child_slice, child_slice] - (
            transition @ parent_covariance @ transition.T
        )
        transitions[child] = transition
        scales[child] = strict_scale_tril(
            0.5 * (innovation + innovation.T),
            name=f"innovation covariance for gauge {ids[child]!r}",
        )
    verify_dense(
        parents,
        transitions,
        scales,
        symmetric,
        atol=parity_atol,
        rtol=parity_rtol,
    )
    return ids, parents, transitions, scales, joint_covariance_sha256(symmetric)


def materialize_dense(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    dimension = GAUGE_DIMENSION * len(parents)
    return np.asarray(
        covariance_action(parents, transitions, scales, np.eye(dimension, dtype=np.float64))
    )


def verify_dense(
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    joint_covariance: Any,
    *,
    atol: float,
    rtol: float,
) -> str:
    dimension = GAUGE_DIMENSION * len(parents)
    matrix = np.asarray(joint_covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError("joint_covariance has changed shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("joint_covariance must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    if not np.allclose(matrix, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError("joint_covariance must be symmetric")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("verification tolerances must be nonnegative")
    reconstructed = materialize_dense(parents, transitions, scales)
    if not np.allclose(reconstructed, symmetric, atol=atol, rtol=rtol):
        maximum_error = float(np.max(np.abs(reconstructed - symmetric)))
        raise ValueError(
            "joint_covariance is not representable by the declared causal tree; "
            f"maximum absolute mismatch is {maximum_error:.6e}"
        )
    return joint_covariance_sha256(symmetric)
