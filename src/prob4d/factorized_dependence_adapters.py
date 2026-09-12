"""Fail-closed adapters for factorized marginal-preserving dependence."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .factorized_dependence import BlockSharedGaussianCovariance, FloatArray


def _matrix(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[0] != result.shape[1]
    ):
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def factorized_from_covariance_endpoints(
    local_covariance: ArrayLike,
    shared_covariance: ArrayLike,
    block_dimension: int,
    strength: float,
    *,
    covariance_rtol: float = 1.0e-9,
    covariance_atol: float = 1.0e-12,
    rank_rtol: float = 1.0e-10,
    rank_atol: float = 1.0e-13,
) -> BlockSharedGaussianCovariance:
    """Convert compatible dense endpoints into a block/shared representation.

    The local endpoint must be block diagonal.  Every complete diagonal block
    of the shared endpoint must equal the corresponding local block.  These
    requirements are deliberately stronger than equality of scalar variances.
    A mismatch raises instead of changing either endpoint.

    The shared endpoint is factored by a symmetric eigendecomposition.  Only
    eigenvalues below the declared numerical rank threshold are discarded.
    The reconstructed covariance must still match the supplied shared endpoint
    within the covariance tolerance.
    """

    local = _matrix(local_covariance, name="local_covariance")
    shared = _matrix(shared_covariance, name="shared_covariance")
    width = _positive_integer(block_dimension, name="block_dimension")
    if local.shape != shared.shape:
        raise ValueError("covariance endpoints must have matching shape")
    if local.shape[0] % width:
        raise ValueError("joint covariance dimension must be divisible by block_dimension")
    for value, name in (
        (covariance_rtol, "covariance_rtol"),
        (covariance_atol, "covariance_atol"),
        (rank_rtol, "rank_rtol"),
        (rank_atol, "rank_atol"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    scale = max(1.0, float(np.max(np.abs(np.concatenate((local, shared))))))
    tolerance = max(covariance_atol, covariance_rtol * scale)
    for matrix, name in ((local, "local_covariance"), (shared, "shared_covariance")):
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
            raise ValueError(f"{name} must be symmetric")
    local = 0.5 * (local + local.T)
    shared = 0.5 * (shared + shared.T)

    local_eigenvalues = np.linalg.eigvalsh(local)
    shared_eigenvalues, shared_eigenvectors = np.linalg.eigh(shared)
    if float(np.min(local_eigenvalues)) < -tolerance:
        raise ValueError("local_covariance must be positive semidefinite")
    if float(np.min(shared_eigenvalues)) < -tolerance:
        raise ValueError("shared_covariance must be positive semidefinite")

    group_count = local.shape[0] // width
    blocks = np.empty((group_count, width, width), dtype=np.float64)
    block_local = np.zeros_like(local)
    for group in range(group_count):
        start = group * width
        stop = start + width
        blocks[group] = local[start:stop, start:stop]
        block_local[start:stop, start:stop] = blocks[group]
        if not np.allclose(
            shared[start:stop, start:stop],
            blocks[group],
            rtol=covariance_rtol,
            atol=covariance_atol,
        ):
            raise ValueError(
                "shared and local endpoints do not preserve the same complete "
                f"marginal block for group {group}"
            )
    if not np.allclose(local, block_local, rtol=covariance_rtol, atol=covariance_atol):
        maximum = float(np.max(np.abs(local - block_local)))
        raise ValueError(
            "local_covariance must be block diagonal; maximum off-block magnitude "
            f"is {maximum:.6g}"
        )

    maximum_eigenvalue = max(0.0, float(np.max(shared_eigenvalues)))
    threshold = max(rank_atol, rank_rtol * maximum_eigenvalue)
    retained = shared_eigenvalues > threshold
    if not np.any(retained):
        raise ValueError("shared_covariance has zero numerical rank")
    factor = shared_eigenvectors[:, retained] * np.sqrt(shared_eigenvalues[retained])
    reconstructed = factor @ factor.T
    if not np.allclose(
        reconstructed,
        shared,
        rtol=covariance_rtol,
        atol=covariance_atol,
    ):
        maximum = float(np.max(np.abs(reconstructed - shared)))
        raise ValueError(
            "rank truncation does not reproduce shared_covariance within tolerance; "
            f"maximum absolute mismatch is {maximum:.6g}"
        )
    return BlockSharedGaussianCovariance(
        marginal_blocks=blocks,
        shared_factors=factor.reshape(group_count, width, -1),
        strength=strength,
        matching_rtol=covariance_rtol,
        matching_atol=covariance_atol,
    )
