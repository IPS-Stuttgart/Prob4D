"""Observation-space products for sparse gauge-tree priors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._gauge_tree_common import FloatArray, validate_observation_design

CovarianceAction = Callable[[Any], FloatArray]


def row_marginal_covariance(
    diagonal_blocks: np.ndarray,
    local_gauge_jacobian: Any,
    gauge_indices: Any,
) -> FloatArray:
    jacobian, indices = validate_observation_design(
        local_gauge_jacobian,
        gauge_indices,
        gauge_count=len(diagonal_blocks),
    )
    result = np.einsum(
        "mia,mab,mjb->mij",
        jacobian,
        diagonal_blocks[indices],
        jacobian,
        optimize=True,
    )
    return 0.5 * (result + result.swapaxes(1, 2))


def observation_covariance_action(
    covariance_action: CovarianceAction,
    gauge_count: int,
    local_gauge_jacobian: Any,
    gauge_indices: Any,
    value: Any,
) -> FloatArray:
    jacobian, indices = validate_observation_design(
        local_gauge_jacobian,
        gauge_indices,
        gauge_count=gauge_count,
    )
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == (len(jacobian), 3):
        values = raw[:, :, None]
        squeeze = True
    elif raw.ndim == 3 and raw.shape[:2] == (len(jacobian), 3):
        values = raw
        squeeze = False
    else:
        raise ValueError("value must have shape (M, 3) or (M, 3, R)")
    if values.shape[2] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("value must contain finite values and at least one right-hand side")
    row_contributions = np.einsum("mij,mir->mjr", jacobian, values, optimize=True)
    gauge_rhs = np.zeros((gauge_count, 7, values.shape[2]), dtype=np.float64)
    np.add.at(gauge_rhs, indices, row_contributions)
    gauge_response = np.asarray(covariance_action(gauge_rhs))
    result = np.einsum(
        "mij,mjr->mir",
        jacobian,
        gauge_response[indices],
        optimize=True,
    )
    return result[:, :, 0] if squeeze else result


def marginal_observation_covariance_action(
    covariance_action: CovarianceAction,
    gauge_count: int,
    local_gauge_jacobian: Any,
    gauge_indices: Any,
    conditional_covariance: Any,
    value: Any,
) -> FloatArray:
    jacobian, indices = validate_observation_design(
        local_gauge_jacobian,
        gauge_indices,
        gauge_count=gauge_count,
    )
    local = np.asarray(conditional_covariance, dtype=np.float64)
    if local.shape != (len(jacobian), 3, 3):
        raise ValueError("conditional_covariance must have shape (M, 3, 3)")
    if not np.all(np.isfinite(local)):
        raise ValueError("conditional_covariance must be finite")
    symmetric = 0.5 * (local + local.swapaxes(1, 2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(1, 2), initial=0.0), 1.0)
    if not np.allclose(
        local,
        symmetric,
        atol=1e-12 * scale[:, None, None],
        rtol=1e-10,
    ):
        raise ValueError("conditional_covariance must be symmetric")
    if np.any(np.min(np.linalg.eigvalsh(symmetric), axis=1) < -1e-12 * scale):
        raise ValueError("conditional_covariance must be positive semidefinite")
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == (len(jacobian), 3):
        local_action = np.einsum("mij,mj->mi", local, raw, optimize=True)
    elif raw.ndim == 3 and raw.shape[:2] == (len(jacobian), 3):
        local_action = np.einsum("mij,mjr->mir", local, raw, optimize=True)
    else:
        raise ValueError("value must have shape (M, 3) or (M, 3, R)")
    return local_action + observation_covariance_action(
        covariance_action,
        gauge_count,
        jacobian,
        indices,
        raw,
    )
