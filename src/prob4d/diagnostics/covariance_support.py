"""Observability-aware diagnostics for errors under positive-semidefinite covariance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._scientific_scalars import require_finite_real
from ..covariance import validated_covariance_psd

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class CovarianceSupportDiagnostic:
    """Decompose one error into observable and covariance-nullspace components.

    ``observable_normalized_squared_error`` is the Moore--Penrose quadratic on
    the covariance range. It is a valid NEES-style statistic only when
    ``support_consistent`` is true. The explicit support fields prevent a
    nullspace error from being silently assigned zero cost.
    """

    dimension: int
    rank: int
    observable_normalized_squared_error: float
    rank_normalized_observable_squared_error: float
    error_norm: float
    nullspace_error_norm: float
    support_tolerance: float
    support_consistent: bool
    eigenvalue_threshold: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float


def covariance_support_diagnostic(
    error: FloatArray,
    covariance: FloatArray,
    *,
    rank_relative_tolerance: float = 1e-10,
    rank_absolute_tolerance: float = 0.0,
    support_relative_tolerance: float = 1e-9,
    support_absolute_tolerance: float = 1e-12,
) -> CovarianceSupportDiagnostic:
    """Evaluate an error without hiding covariance-nullspace inconsistency.

    The covariance is validated as symmetric positive semidefinite without
    adding jitter. Eigenvalues above the declared scale-aware threshold define
    the observable range. The residual component outside that range is reported
    explicitly and compared with a scale-aware support tolerance.
    """

    vector = np.asarray(error, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("error must be a nonempty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("error must be finite")

    rank_relative_tolerance = require_finite_real(
        rank_relative_tolerance,
        name="rank_relative_tolerance",
        minimum=0.0,
    )
    rank_absolute_tolerance = require_finite_real(
        rank_absolute_tolerance,
        name="rank_absolute_tolerance",
        minimum=0.0,
    )
    support_relative_tolerance = require_finite_real(
        support_relative_tolerance,
        name="support_relative_tolerance",
        minimum=0.0,
    )
    support_absolute_tolerance = require_finite_real(
        support_absolute_tolerance,
        name="support_absolute_tolerance",
        minimum=0.0,
    )

    validated = validated_covariance_psd(
        covariance,
        name="diagnostic covariance",
        shape=(vector.size, vector.size),
        readonly=False,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(validated)
    minimum_eigenvalue = float(np.min(eigenvalues))
    maximum_eigenvalue = float(np.max(eigenvalues))
    eigenvalue_threshold = (
        rank_absolute_tolerance + rank_relative_tolerance * maximum_eigenvalue
    )
    observable = eigenvalues > eigenvalue_threshold
    rank = int(np.count_nonzero(observable))

    coordinates = eigenvectors.T @ vector
    if rank:
        observable_normalized_squared_error = float(
            np.sum(coordinates[observable] ** 2 / eigenvalues[observable])
        )
        rank_normalized = observable_normalized_squared_error / rank
    else:
        observable_normalized_squared_error = 0.0
        rank_normalized = 0.0

    nullspace_error_norm = float(np.linalg.norm(coordinates[~observable]))
    error_norm = float(np.linalg.norm(vector))
    support_tolerance = (
        support_absolute_tolerance + support_relative_tolerance * error_norm
    )

    return CovarianceSupportDiagnostic(
        dimension=int(vector.size),
        rank=rank,
        observable_normalized_squared_error=observable_normalized_squared_error,
        rank_normalized_observable_squared_error=float(rank_normalized),
        error_norm=error_norm,
        nullspace_error_norm=nullspace_error_norm,
        support_tolerance=float(support_tolerance),
        support_consistent=bool(nullspace_error_norm <= support_tolerance),
        eigenvalue_threshold=float(eigenvalue_threshold),
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
    )


__all__ = ["CovarianceSupportDiagnostic", "covariance_support_diagnostic"]
