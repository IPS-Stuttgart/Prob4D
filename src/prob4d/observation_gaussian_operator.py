"""Structured Gaussian solves for explicit-gauge observation-factor stacks.

Prob4D represents the joint observation covariance as

``blockdiag(R_1, ..., R_M) + J Sigma_g J.T``.

The public operator delegates to dense/sparse covariance-root Woodbury backends
or exact causal-tree block-information elimination without materializing the
full ``3M x 3M`` observation covariance.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from ._observation_gaussian_common import (
    FloatArray,
    _conditional_factorization,
    _conditional_solve,
    _GaugePrecisionBackend,
    _require_stack,
    _validated_rhs,
)
from ._observation_gaussian_tree import _TreeGaugeInformation
from ._observation_gaussian_woodbury import (
    _DenseGaugeWoodbury,
    _SparseGaugeWoodbury,
)
from .observation_covariance_queries import ObservationFactorStack
from .sparse_observation_factors import SparseStackedObservationFactors
from .tree_sparse_observation_factors import TreeSparseStackedObservationFactors


class ObservationGaussianOperator:
    """Cached structured factorization of one joint observation Gaussian.

    The operator requires every conditional ``3 x 3`` row covariance to be
    strictly positive definite. Gauge priors may be positive semidefinite for
    dense and sparse stacks. The tree-sparse prior remains the already validated
    strictly positive-definite innovation model.
    """

    __slots__ = (
        "_backend",
        "_conditional_factor",
        "_dimension",
        "_log_determinant",
        "_observation_count",
    )

    def __init__(self, stacked: ObservationFactorStack) -> None:
        validated = _require_stack(stacked)
        conditional_factor, conditional_log_determinant = _conditional_factorization(
            validated
        )
        backend: _GaugePrecisionBackend
        if isinstance(validated, TreeSparseStackedObservationFactors):
            backend = _TreeGaugeInformation.build(validated, conditional_factor)
        elif isinstance(validated, SparseStackedObservationFactors):
            backend = _SparseGaugeWoodbury.build(validated, conditional_factor)
        else:
            backend = _DenseGaugeWoodbury.build(validated, conditional_factor)
        log_determinant = (
            conditional_log_determinant + backend.log_determinant_increment
        )
        if not math.isfinite(log_determinant):
            raise RuntimeError("structured observation log determinant is not finite")
        self._backend = backend
        self._conditional_factor = conditional_factor
        self._observation_count = int(len(validated.world_mean_m))
        self._dimension = 3 * self._observation_count
        self._log_determinant = float(log_determinant)

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def factorization_backend(self) -> str:
        return self._backend.factorization_backend

    @property
    def factor_storage_nbytes(self) -> int:
        """Bytes retained by the cached factorization, excluding the input stack."""

        return int(
            self._conditional_factor.nbytes + self._backend.factor_storage_nbytes
        )

    @property
    def dense_covariance_nbytes(self) -> int:
        """Bytes required by a dense ``3M x 3M`` float64 covariance."""

        return int(self._dimension**2 * np.dtype(np.float64).itemsize)

    @property
    def factor_storage_ratio_to_dense(self) -> float:
        """Cached-factor storage divided by full dense covariance storage."""

        return self.factor_storage_nbytes / self.dense_covariance_nbytes

    @property
    def log_determinant(self) -> float:
        """Return ``log(det(Sigma_y))`` for the joint observation covariance."""

        return self._log_determinant

    def solve(self, value: object) -> FloatArray:
        """Return ``Sigma_y^{-1} value`` without materializing ``Sigma_y``.

        ``value`` may have shape ``(M, 3)`` or ``(M, 3, R)``. The returned array
        has the same shape.
        """

        values, squeeze = _validated_rhs(
            value,
            observation_count=self._observation_count,
        )
        conditional_response = _conditional_solve(
            self._conditional_factor,
            values,
        )
        result = self._backend.apply(conditional_response)
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("structured observation solve returned malformed values")
        return cast(FloatArray, result[:, :, 0] if squeeze else result)

    def precision_quadratic(self, value: object) -> float:
        """Return ``value.T @ Sigma_y^{-1} @ value`` for one residual field."""

        residual = np.asarray(value, dtype=np.float64)
        if residual.shape != (self._observation_count, 3):
            raise ValueError("value must have shape (M, 3)")
        if not np.all(np.isfinite(residual)):
            raise ValueError("value must be finite")
        response = self.solve(residual)
        result = float(np.sum(residual * response, dtype=np.float64))
        scale = max(float(np.sum(np.abs(residual * response), dtype=np.float64)), 1.0)
        if result < -1e-10 * scale:
            raise RuntimeError("structured observation precision produced negative energy")
        return max(result, 0.0)

    def gaussian_nll(self, residual: object, *, per_dimension: bool = False) -> float:
        """Return the zero-mean Gaussian negative log likelihood of ``residual``."""

        if type(per_dimension) is not bool:
            raise TypeError("per_dimension must be a bool")
        quadratic = self.precision_quadratic(residual)
        result = 0.5 * (
            self._dimension * math.log(2.0 * math.pi)
            + self._log_determinant
            + quadratic
        )
        return result / self._dimension if per_dimension else result


def build_observation_gaussian_operator(
    stacked: ObservationFactorStack,
) -> ObservationGaussianOperator:
    """Build and cache the structured Gaussian factorization for ``stacked``."""

    return ObservationGaussianOperator(stacked)


def solve_observation_covariance(
    stacked: ObservationFactorStack,
    value: object,
) -> FloatArray:
    """Return one inverse covariance action through a temporary operator."""

    return ObservationGaussianOperator(stacked).solve(value)


def observation_precision_quadratic(
    stacked: ObservationFactorStack,
    value: object,
) -> float:
    """Return one joint precision quadratic through a temporary operator."""

    return ObservationGaussianOperator(stacked).precision_quadratic(value)


def observation_log_determinant(stacked: ObservationFactorStack) -> float:
    """Return the joint observation covariance log determinant."""

    return ObservationGaussianOperator(stacked).log_determinant


def observation_gaussian_nll(
    stacked: ObservationFactorStack,
    residual: object,
    *,
    per_dimension: bool = False,
) -> float:
    """Return one joint Gaussian negative log likelihood."""

    return ObservationGaussianOperator(stacked).gaussian_nll(
        residual,
        per_dimension=per_dimension,
    )


__all__ = [
    "ObservationGaussianOperator",
    "build_observation_gaussian_operator",
    "observation_gaussian_nll",
    "observation_log_determinant",
    "observation_precision_quadratic",
    "solve_observation_covariance",
]
