"""Shared validation and linear-algebra helpers for structured observation Gaussians."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .observation_covariance_queries import ObservationFactorStack
from .observation_factors import StackedObservationFactors
from .sparse_observation_factors import SparseStackedObservationFactors
from .tree_sparse_observation_factors import TreeSparseStackedObservationFactors

FloatArray: TypeAlias = NDArray[np.floating[Any]]


class _GaugePrecisionBackend(Protocol):
    @property
    def factorization_backend(self) -> str:
        ...

    @property
    def log_determinant_increment(self) -> float:
        ...

    @property
    def factor_storage_nbytes(self) -> int:
        ...

    def apply(self, conditional_precision_response: np.ndarray) -> np.ndarray:
        """Apply the complete observation precision to a conditional response."""

        ...


def _require_stack(value: object) -> ObservationFactorStack:
    if not isinstance(
        value,
        (
            StackedObservationFactors,
            SparseStackedObservationFactors,
            TreeSparseStackedObservationFactors,
        ),
    ):
        raise TypeError(
            "stacked must be a StackedObservationFactors, "
            "SparseStackedObservationFactors, or TreeSparseStackedObservationFactors"
        )
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    result.setflags(write=False)
    return cast(np.ndarray, result)


def _symmetric(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"{name} must contain square matrices")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
    scale = np.maximum(
        np.max(np.abs(symmetric), axis=(-2, -1), initial=0.0),
        1.0,
    )
    if not np.allclose(
        matrix,
        symmetric,
        atol=1e-12 * scale[..., None, None],
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    return cast(np.ndarray, symmetric)


def _strict_cholesky(value: np.ndarray, *, name: str) -> np.ndarray:
    symmetric = _symmetric(value, name=name)
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be strictly positive definite") from error
    return cast(np.ndarray, factor)


def _cholesky_solve(factor: np.ndarray, value: np.ndarray) -> np.ndarray:
    forward = np.linalg.solve(factor, value)
    return cast(
        np.ndarray,
        np.linalg.solve(np.swapaxes(factor, -1, -2), forward),
    )


def _conditional_factorization(
    stacked: ObservationFactorStack,
) -> tuple[np.ndarray, float]:
    count = len(stacked.world_mean_m)
    conditional = np.asarray(
        stacked.conditional_world_covariance_m2,
        dtype=np.float64,
    )
    if conditional.shape != (count, 3, 3):
        raise ValueError(
            "conditional_world_covariance_m2 must have shape (M, 3, 3)"
        )
    factor = _strict_cholesky(
        conditional,
        name="conditional_world_covariance_m2",
    )
    log_determinant = float(
        2.0
        * np.sum(
            np.log(np.diagonal(factor, axis1=-2, axis2=-1)),
            dtype=np.float64,
        )
    )
    return _readonly(factor), log_determinant


def _conditional_solve(factor: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = _cholesky_solve(factor, value)
    if result.shape != value.shape or not np.all(np.isfinite(result)):
        raise RuntimeError("conditional covariance solve returned malformed values")
    return result


def _validated_rhs(
    value: object,
    *,
    observation_count: int,
) -> tuple[np.ndarray, bool]:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == (observation_count, 3):
        result = raw[:, :, None]
        squeeze = True
    elif raw.ndim == 3 and raw.shape[:2] == (observation_count, 3):
        result = raw
        squeeze = False
    else:
        raise ValueError("value must have shape (M, 3) or (M, 3, R)")
    if result.shape[2] < 1 or not np.all(np.isfinite(result)):
        raise ValueError("value must contain finite values and at least one right-hand side")
    return result, squeeze


def _covariance_root_psd(value: np.ndarray, *, name: str) -> np.ndarray:
    covariance = _symmetric(value, name=name)
    if covariance.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    tolerance = 1e-12 * scale
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    active = clipped > 0.0
    if not np.any(active):
        return np.empty((covariance.shape[0], 0), dtype=np.float64)
    root = eigenvectors[:, active] * np.sqrt(clipped[active])
    return _readonly(root)


def _factor_log_determinant(factor: np.ndarray) -> float:
    if factor.size == 0:
        return 0.0
    return float(2.0 * np.sum(np.log(np.diag(factor)), dtype=np.float64))
