"""Local Gaussian query conditioning with structured Prob4D covariances.

The base :class:`~prob4d.observation_gaussian_operator.ObservationGaussianOperator`
represents the correlation-aware provider covariance ``R``.  A physical model
can add a low-rank positive-semidefinite term ``F F.T`` without materializing a
dense observation covariance.  For a fixed local linearization this yields the
innovation covariance

``S = R + F F.T``.

Given the prior moments of a registered query and its cross covariance with the
innovation, :func:`condition_gaussian_query` then evaluates the exact Gaussian
conditioning equations using only structured solves with ``S``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .observation_gaussian_operator import ObservationGaussianOperator

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def _readonly(value: object, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    copied = result.copy()
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _readonly_vector(value: object, *, name: str) -> FloatArray:
    result = _readonly(value, name=name)
    if result.ndim != 1 or result.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty vector")
    return result


def _readonly_symmetric_psd(value: object, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (result + result.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(result, symmetric, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    if minimum_eigenvalue < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    copied = symmetric.copy()
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _cho_solve(factor: FloatArray, value: FloatArray) -> FloatArray:
    forward = np.linalg.solve(factor, value)
    return cast(FloatArray, np.linalg.solve(factor.T, forward))


class LowRankUpdatedObservationGaussianOperator:
    """Structured operator for ``S = R + F F.T``.

    ``R`` is supplied by one cached :class:`ObservationGaussianOperator`.
    ``F`` has shape ``(M, 3, K)`` (or ``(M, 3)`` for rank one).  The class
    caches the Woodbury core and therefore supports repeated solves, log
    determinants, proper scores, and query conditioning without forming the
    dense ``3M x 3M`` covariance.

    The factor is copied at construction, so later caller-side mutation cannot
    invalidate the cached factorization.
    """

    __slots__ = (
        "_base",
        "_base_solved_factor",
        "_core_factor",
        "_factor",
        "_log_determinant",
    )

    def __init__(
        self,
        base: ObservationGaussianOperator,
        low_rank_factor: object,
    ) -> None:
        if not isinstance(base, ObservationGaussianOperator):
            raise TypeError("base must be an ObservationGaussianOperator")
        raw = np.asarray(low_rank_factor, dtype=np.float64)
        if raw.shape == (base.observation_count, 3):
            factor = raw[:, :, None]
        elif raw.ndim == 3 and raw.shape[:2] == (base.observation_count, 3):
            factor = raw
        else:
            raise ValueError(
                "low_rank_factor must have shape (M, 3) or (M, 3, K)"
            )
        if factor.shape[2] < 1 or not np.all(np.isfinite(factor)):
            raise ValueError(
                "low_rank_factor must contain finite values and at least one column"
            )

        retained_factor = np.asarray(factor, dtype=np.float64).copy()
        base_solved_factor = np.asarray(
            base.solve(retained_factor),
            dtype=np.float64,
        )
        gram = np.einsum(
            "mik,mil->kl",
            retained_factor,
            base_solved_factor,
            optimize=True,
        )
        core = np.eye(factor.shape[2], dtype=np.float64) + 0.5 * (gram + gram.T)
        try:
            core_factor = np.linalg.cholesky(core)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "low-rank Woodbury core is not strictly positive definite"
            ) from error
        log_determinant = base.log_determinant + 2.0 * float(
            np.sum(np.log(np.diag(core_factor)), dtype=np.float64)
        )
        if not math.isfinite(log_determinant):
            raise RuntimeError("updated observation log determinant is not finite")

        retained_factor.setflags(write=False)
        base_solved_factor = base_solved_factor.copy()
        base_solved_factor.setflags(write=False)
        core_factor = core_factor.copy()
        core_factor.setflags(write=False)

        self._base = base
        self._factor = cast(FloatArray, retained_factor)
        self._base_solved_factor = cast(FloatArray, base_solved_factor)
        self._core_factor = cast(FloatArray, core_factor)
        self._log_determinant = float(log_determinant)

    @property
    def base_operator(self) -> ObservationGaussianOperator:
        """Return the unchanged correlation-aware base operator."""

        return self._base

    @property
    def observation_count(self) -> int:
        return self._base.observation_count

    @property
    def dimension(self) -> int:
        return self._base.dimension

    @property
    def update_rank(self) -> int:
        return int(self._factor.shape[2])

    @property
    def low_rank_factor(self) -> FloatArray:
        """Return the retained read-only low-rank factor."""

        return self._factor

    @property
    def factorization_backend(self) -> str:
        return f"{self._base.factorization_backend}+low-rank-woodbury-v1"

    @property
    def factor_storage_nbytes(self) -> int:
        return int(
            self._base.factor_storage_nbytes
            + self._factor.nbytes
            + self._base_solved_factor.nbytes
            + self._core_factor.nbytes
        )

    @property
    def dense_covariance_nbytes(self) -> int:
        return self._base.dense_covariance_nbytes

    @property
    def factor_storage_ratio_to_dense(self) -> float:
        return self.factor_storage_nbytes / self.dense_covariance_nbytes

    @property
    def log_determinant(self) -> float:
        """Return ``log(det(R + F F.T))``."""

        return self._log_determinant

    def solve(self, value: object) -> FloatArray:
        """Return ``(R + F F.T)^{-1} value`` by one Woodbury correction."""

        base_response = np.asarray(self._base.solve(value), dtype=np.float64)
        if base_response.ndim == 2:
            core_rhs = np.einsum(
                "mik,mi->k",
                self._factor,
                base_response,
                optimize=True,
            )
            core_response = _cho_solve(self._core_factor, core_rhs)
            correction = np.einsum(
                "mik,k->mi",
                self._base_solved_factor,
                core_response,
                optimize=True,
            )
        elif base_response.ndim == 3:
            core_rhs = np.einsum(
                "mik,mir->kr",
                self._factor,
                base_response,
                optimize=True,
            )
            core_response = _cho_solve(self._core_factor, core_rhs)
            correction = np.einsum(
                "mik,kr->mir",
                self._base_solved_factor,
                core_response,
                optimize=True,
            )
        else:
            raise RuntimeError("base observation solve returned malformed values")
        result = base_response - correction
        if result.shape != base_response.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("updated observation solve returned malformed values")
        return cast(FloatArray, result)

    def precision_quadratic(self, value: object) -> float:
        """Return ``value.T @ (R + F F.T)^{-1} @ value``."""

        residual = np.asarray(value, dtype=np.float64)
        if residual.shape != (self.observation_count, 3):
            raise ValueError("value must have shape (M, 3)")
        if not np.all(np.isfinite(residual)):
            raise ValueError("value must be finite")
        response = self.solve(residual)
        result = float(np.sum(residual * response, dtype=np.float64))
        scale = max(
            float(np.sum(np.abs(residual * response), dtype=np.float64)),
            1.0,
        )
        if result < -1e-10 * scale:
            raise RuntimeError("updated observation precision produced negative energy")
        return max(result, 0.0)

    def gaussian_nll(self, residual: object, *, per_dimension: bool = False) -> float:
        """Return the zero-mean Gaussian negative log likelihood."""

        if type(per_dimension) is not bool:
            raise TypeError("per_dimension must be a bool")
        quadratic = self.precision_quadratic(residual)
        result = 0.5 * (
            self.dimension * math.log(2.0 * math.pi)
            + self.log_determinant
            + quadratic
        )
        return result / self.dimension if per_dimension else result


InnovationOperator: TypeAlias = (
    ObservationGaussianOperator | LowRankUpdatedObservationGaussianOperator
)


@dataclass(frozen=True, slots=True)
class GaussianQueryPosterior:
    """Exact posterior moments and diagnostics for one fixed Gaussian query."""

    prior_mean: FloatArray
    prior_covariance: FloatArray
    mean_shift: FloatArray
    covariance_reduction: FloatArray
    posterior_mean: FloatArray
    posterior_covariance: FloatArray
    innovation_precision_quadratic: float
    innovation_log_determinant: float
    innovation_negative_log_likelihood: float
    observation_dimension: int

    def __post_init__(self) -> None:
        prior_mean = _readonly_vector(self.prior_mean, name="prior_mean")
        prior_covariance = _readonly_symmetric_psd(
            self.prior_covariance,
            name="prior_covariance",
        )
        mean_shift = _readonly_vector(self.mean_shift, name="mean_shift")
        covariance_reduction = _readonly_symmetric_psd(
            self.covariance_reduction,
            name="covariance_reduction",
        )
        posterior_mean = _readonly_vector(self.posterior_mean, name="posterior_mean")
        posterior_covariance = _readonly_symmetric_psd(
            self.posterior_covariance,
            name="posterior_covariance",
        )
        query_dimension = prior_mean.shape[0]
        if (
            prior_covariance.shape != (query_dimension, query_dimension)
            or mean_shift.shape != (query_dimension,)
            or covariance_reduction.shape != (query_dimension, query_dimension)
            or posterior_mean.shape != (query_dimension,)
            or posterior_covariance.shape != (query_dimension, query_dimension)
        ):
            raise ValueError("query posterior fields have inconsistent dimensions")
        vector_scale = max(
            float(np.max(np.abs(posterior_mean), initial=0.0)),
            1.0,
        )
        if not np.allclose(
            posterior_mean,
            prior_mean + mean_shift,
            atol=1e-12 * vector_scale,
            rtol=1e-10,
        ):
            raise ValueError("posterior_mean must equal prior_mean plus mean_shift")
        covariance_scale = max(
            float(np.max(np.abs(posterior_covariance), initial=0.0)),
            1.0,
        )
        if not np.allclose(
            posterior_covariance,
            prior_covariance - covariance_reduction,
            atol=1e-12 * covariance_scale,
            rtol=1e-10,
        ):
            raise ValueError(
                "posterior_covariance must equal prior_covariance minus "
                "covariance_reduction"
            )
        if (
            not math.isfinite(self.innovation_precision_quadratic)
            or self.innovation_precision_quadratic < 0.0
        ):
            raise ValueError(
                "innovation_precision_quadratic must be finite and nonnegative"
            )
        if not math.isfinite(self.innovation_log_determinant):
            raise ValueError("innovation_log_determinant must be finite")
        if not math.isfinite(self.innovation_negative_log_likelihood):
            raise ValueError("innovation_negative_log_likelihood must be finite")
        if type(self.observation_dimension) is not int or self.observation_dimension < 1:
            raise ValueError("observation_dimension must be a positive integer")

        object.__setattr__(self, "prior_mean", prior_mean)
        object.__setattr__(self, "prior_covariance", prior_covariance)
        object.__setattr__(self, "mean_shift", mean_shift)
        object.__setattr__(self, "covariance_reduction", covariance_reduction)
        object.__setattr__(self, "posterior_mean", posterior_mean)
        object.__setattr__(self, "posterior_covariance", posterior_covariance)

    @property
    def query_dimension(self) -> int:
        return int(self.posterior_mean.shape[0])


def augment_observation_gaussian_operator(
    base: ObservationGaussianOperator,
    low_rank_factor: object,
) -> LowRankUpdatedObservationGaussianOperator:
    """Return a cached operator for ``R + F F.T``."""

    return LowRankUpdatedObservationGaussianOperator(base, low_rank_factor)


def _require_innovation_operator(value: object) -> InnovationOperator:
    if not isinstance(
        value,
        (
            ObservationGaussianOperator,
            LowRankUpdatedObservationGaussianOperator,
        ),
    ):
        raise TypeError(
            "innovation_operator must be an ObservationGaussianOperator or "
            "LowRankUpdatedObservationGaussianOperator"
        )
    return value


def _validated_cross_covariance(
    value: object,
    *,
    query_dimension: int,
    observation_count: int,
) -> FloatArray:
    raw = np.asarray(value, dtype=np.float64)
    if query_dimension == 1 and raw.shape == (observation_count, 3):
        result = raw[None]
    elif raw.shape == (query_dimension, 3 * observation_count):
        result = raw.reshape(query_dimension, observation_count, 3)
    elif raw.shape == (query_dimension, observation_count, 3):
        result = raw
    else:
        raise ValueError(
            "query_observation_cross_covariance must have shape "
            "(M, 3) for a scalar query, (Q, 3M), or (Q, M, 3)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("query_observation_cross_covariance must be finite")
    return cast(FloatArray, result)


def condition_gaussian_query(
    *,
    prior_mean: object,
    prior_covariance: object,
    innovation: object,
    query_observation_cross_covariance: object,
    innovation_operator: InnovationOperator,
) -> GaussianQueryPosterior:
    """Condition a registered query on one structured Gaussian innovation.

    For prior query moments ``(m_q, P_q)``, innovation ``nu``, cross covariance
    ``C_qy``, and a supplied operator for the *full* innovation covariance
    ``S``, this computes

    ``m_q|y = m_q + C_qy S^{-1} nu``

    and

    ``P_q|y = P_q - C_qy S^{-1} C_qy.T``.

    The calculation performs one batched structured solve and materializes only
    ``Q x Q`` query matrices.  It is exact for the supplied fixed local
    linear-Gaussian model.  The caller remains responsible for constructing the
    physical prior, linearization, cross covariance, and any frozen robust
    weights.
    """

    operator = _require_innovation_operator(innovation_operator)
    validated_prior_mean = _readonly_vector(prior_mean, name="prior_mean")
    query_dimension = int(validated_prior_mean.shape[0])
    validated_prior_covariance = _readonly_symmetric_psd(
        prior_covariance,
        name="prior_covariance",
    )
    if validated_prior_covariance.shape != (query_dimension, query_dimension):
        raise ValueError("prior_covariance shape must match prior_mean")

    residual = np.asarray(innovation, dtype=np.float64)
    if residual.shape != (operator.observation_count, 3):
        raise ValueError("innovation must have shape (M, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("innovation must be finite")

    cross_covariance = _validated_cross_covariance(
        query_observation_cross_covariance,
        query_dimension=query_dimension,
        observation_count=operator.observation_count,
    )
    right_hand_sides = np.concatenate(
        (
            residual[:, :, None],
            np.moveaxis(cross_covariance, 0, -1),
        ),
        axis=2,
    )
    solved = np.asarray(operator.solve(right_hand_sides), dtype=np.float64)
    innovation_response = solved[:, :, 0]
    cross_response = solved[:, :, 1:]

    mean_shift = np.einsum(
        "qmi,mi->q",
        cross_covariance,
        innovation_response,
        optimize=True,
    )
    raw_reduction = np.einsum(
        "qmi,mir->qr",
        cross_covariance,
        cross_response,
        optimize=True,
    )
    reduction_scale = max(
        float(np.max(np.abs(raw_reduction), initial=0.0)),
        1.0,
    )
    covariance_reduction = 0.5 * (raw_reduction + raw_reduction.T)
    if not np.allclose(
        raw_reduction,
        covariance_reduction,
        atol=1e-11 * reduction_scale,
        rtol=1e-9,
    ):
        raise RuntimeError("query covariance reduction is numerically asymmetric")
    validated_reduction = _readonly_symmetric_psd(
        covariance_reduction,
        name="covariance_reduction",
    )

    posterior_mean = validated_prior_mean + mean_shift
    posterior_covariance = validated_prior_covariance - validated_reduction
    try:
        validated_posterior_covariance = _readonly_symmetric_psd(
            posterior_covariance,
            name="posterior_covariance",
        )
    except ValueError as error:
        raise ValueError(
            "query moments are inconsistent with the supplied innovation covariance; "
            "posterior_covariance is not positive semidefinite"
        ) from error

    precision_quadratic = float(
        np.sum(residual * innovation_response, dtype=np.float64)
    )
    precision_scale = max(
        float(np.sum(np.abs(residual * innovation_response), dtype=np.float64)),
        1.0,
    )
    if precision_quadratic < -1e-10 * precision_scale:
        raise RuntimeError("innovation precision produced negative energy")
    precision_quadratic = max(precision_quadratic, 0.0)
    negative_log_likelihood = 0.5 * (
        operator.dimension * math.log(2.0 * math.pi)
        + operator.log_determinant
        + precision_quadratic
    )

    return GaussianQueryPosterior(
        prior_mean=validated_prior_mean,
        prior_covariance=validated_prior_covariance,
        mean_shift=mean_shift,
        covariance_reduction=validated_reduction,
        posterior_mean=posterior_mean,
        posterior_covariance=validated_posterior_covariance,
        innovation_precision_quadratic=precision_quadratic,
        innovation_log_determinant=operator.log_determinant,
        innovation_negative_log_likelihood=negative_log_likelihood,
        observation_dimension=operator.dimension,
    )


__all__ = [
    "GaussianQueryPosterior",
    "LowRankUpdatedObservationGaussianOperator",
    "augment_observation_gaussian_operator",
    "condition_gaussian_query",
]
