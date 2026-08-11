"""Project conditional-plus-low-rank observation covariance into query space.

Prob4D owns the structured observation covariance. A downstream consumer owns
its query Jacobian and any decision about whether joint covariance is worth the
added inference cost. This module performs only the neutral covariance
projection and reports how much query-space variance comes from shared modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]

QUERY_COVARIANCE_RELEVANCE_SCHEMA = "prob4d.query-covariance-relevance"
QUERY_COVARIANCE_RELEVANCE_VERSION = 1
QUERY_COVARIANCE_RELEVANCE_CLAIM_BOUNDARY = (
    "This diagnostic projects a supplied Prob4D conditional-plus-low-rank "
    "covariance through a caller-supplied query Jacobian. It does not define "
    "the physical query, select a covariance treatment, authorize an update, "
    "or establish BayesianPhysTwin or Causal4D benefit."
)


def _readonly(value: np.ndarray) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _validated_relative_rank_tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("relative_rank_tolerance must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("relative_rank_tolerance must lie in [0, 1)")
    return result


def _strict_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_covariance(
    value: object,
    *,
    name: str,
    expected_dimension: int | None = None,
) -> FloatArray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if covariance.shape[0] < 1:
        raise ValueError(f"{name} must have positive dimension")
    if expected_dimension is not None and covariance.shape != (
        expected_dimension,
        expected_dimension,
    ):
        raise ValueError(f"{name} has an invalid shape")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (covariance + covariance.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0)) < -(
        1e-12 + 1e-10 * scale
    ):
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _validated_covariance_rows(
    local_covariance_m2: object,
    *,
    sample_count: int,
) -> FloatArray:
    local = np.asarray(local_covariance_m2, dtype=np.float64)
    if local.shape != (sample_count, 3, 3):
        raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
    if not np.all(np.isfinite(local)):
        raise ValueError("local_covariance_m2 must be finite")
    symmetric = 0.5 * (local + local.swapaxes(1, 2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(1, 2)), 1.0)
    asymmetry = np.max(np.abs(local - local.swapaxes(1, 2)), axis=(1, 2))
    if np.any(asymmetry > 1e-12 + 1e-10 * scale):
        raise ValueError("local_covariance_m2 must be symmetric")
    minimum = np.min(np.linalg.eigvalsh(symmetric), axis=1)
    if np.any(minimum < -(1e-12 + 1e-10 * scale)):
        raise ValueError("local_covariance_m2 must be positive semidefinite")
    return symmetric


def _validated_inputs(
    query_jacobian: object,
    local_covariance_m2: object,
    low_rank_factor_m: object,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    jacobian = np.asarray(query_jacobian, dtype=np.float64)
    if jacobian.ndim == 2:
        if jacobian.shape[1:] != (3,):
            raise ValueError("scalar query_jacobian must have shape (N, 3)")
        jacobian = jacobian[None, ...]
    elif jacobian.ndim != 3 or jacobian.shape[2] != 3:
        raise ValueError("query_jacobian must have shape (Q, N, 3) or (N, 3)")
    if jacobian.shape[0] < 1 or jacobian.shape[1] < 1:
        raise ValueError("query_jacobian must contain at least one query and one row")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("query_jacobian must be finite")

    sample_count = int(jacobian.shape[1])
    local = _validated_covariance_rows(
        local_covariance_m2,
        sample_count=sample_count,
    )
    factor = np.asarray(low_rank_factor_m, dtype=np.float64)
    if factor.ndim != 3 or factor.shape[:2] != (sample_count, 3):
        raise ValueError("low_rank_factor_m must have shape (N, 3, R)")
    if not np.all(np.isfinite(factor)):
        raise ValueError("low_rank_factor_m must be finite")
    return jacobian, local, factor


def _effective_rank(value: np.ndarray, *, relative_tolerance: float) -> int:
    eigenvalues = np.linalg.eigvalsh(0.5 * (value + value.T))
    maximum = float(np.max(eigenvalues, initial=0.0))
    if maximum <= 0.0:
        return 0
    return int(np.count_nonzero(eigenvalues > relative_tolerance * maximum))


def _directional_shared_fractions(
    shared_covariance: np.ndarray,
    total_covariance: np.ndarray,
    *,
    relative_tolerance: float,
) -> tuple[int, float | None, float | None, float | None]:
    total_eigenvalues, total_eigenvectors = np.linalg.eigh(total_covariance)
    maximum = float(np.max(total_eigenvalues, initial=0.0))
    if maximum <= 0.0:
        return 0, None, None, None
    active = total_eigenvalues > relative_tolerance * maximum
    active_dimension = int(np.count_nonzero(active))
    if active_dimension == 0:
        return 0, None, None, None

    basis = total_eigenvectors[:, active]
    inverse_root = 1.0 / np.sqrt(total_eigenvalues[active])
    whitened_shared = basis.T @ shared_covariance @ basis
    whitened_shared = inverse_root[:, None] * whitened_shared * inverse_root[None, :]
    whitened_shared = 0.5 * (whitened_shared + whitened_shared.T)
    fractions = np.linalg.eigvalsh(whitened_shared)
    if np.any(fractions < -1e-10) or np.any(fractions > 1.0 + 1e-10):
        raise ValueError("shared covariance is not a valid component of total covariance")
    fractions = np.clip(fractions, 0.0, 1.0)
    return (
        active_dimension,
        float(np.min(fractions)),
        float(np.mean(fractions)),
        float(np.max(fractions)),
    )


def _coordinate_shared_fractions(
    shared_covariance: np.ndarray,
    total_covariance: np.ndarray,
) -> tuple[float | None, ...]:
    shared_diagonal = np.diag(shared_covariance)
    total_diagonal = np.diag(total_covariance)
    result: list[float | None] = []
    for shared, total in zip(shared_diagonal, total_diagonal, strict=True):
        result.append(None if total <= 0.0 else float(np.clip(shared / total, 0.0, 1.0)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class QueryCovarianceProjectionV1:
    """Immutable query-space decomposition and relevance diagnostics."""

    conditional_covariance: FloatArray
    shared_query_factor: FloatArray
    observation_count: int
    relative_rank_tolerance: float = 1e-10
    shared_covariance: FloatArray = field(init=False)
    total_covariance: FloatArray = field(init=False)
    query_dimension: int = field(init=False)
    shared_rank_column_count: int = field(init=False)
    total_effective_rank: int = field(init=False)
    shared_effective_rank: int = field(init=False)
    active_query_dimension: int = field(init=False)
    conditional_trace: float = field(init=False)
    shared_trace: float = field(init=False)
    total_trace: float = field(init=False)
    shared_trace_fraction: float | None = field(init=False)
    shared_frobenius_fraction: float | None = field(init=False)
    coordinate_shared_fractions: tuple[float | None, ...] = field(init=False)
    minimum_directional_shared_fraction: float | None = field(init=False)
    mean_directional_shared_fraction: float | None = field(init=False)
    maximum_directional_shared_fraction: float | None = field(init=False)

    def __post_init__(self) -> None:
        observation_count = _strict_positive_integer(
            self.observation_count,
            name="observation_count",
        )
        tolerance = _validated_relative_rank_tolerance(self.relative_rank_tolerance)
        conditional = _validated_covariance(
            self.conditional_covariance,
            name="conditional_covariance",
        )
        query_dimension = int(conditional.shape[0])
        shared_factor = np.asarray(self.shared_query_factor, dtype=np.float64)
        if shared_factor.ndim != 2 or shared_factor.shape[0] != query_dimension:
            raise ValueError("shared_query_factor must have shape (Q, R)")
        if not np.all(np.isfinite(shared_factor)):
            raise ValueError("shared_query_factor must be finite")

        shared = _validated_covariance(
            shared_factor @ shared_factor.T,
            name="shared_covariance",
            expected_dimension=query_dimension,
        )
        total = _validated_covariance(
            conditional + shared,
            name="total_covariance",
            expected_dimension=query_dimension,
        )
        conditional_trace = float(np.trace(conditional))
        shared_trace = float(np.trace(shared))
        total_trace = float(np.trace(total))
        total_norm = float(np.linalg.norm(total, ord="fro"))
        shared_trace_fraction = (
            None
            if total_trace <= 0.0
            else float(np.clip(shared_trace / total_trace, 0.0, 1.0))
        )
        shared_frobenius_fraction = (
            None
            if total_norm == 0.0
            else float(np.clip(np.linalg.norm(shared, ord="fro") / total_norm, 0.0, 1.0))
        )
        (
            active_dimension,
            minimum_directional,
            mean_directional,
            maximum_directional,
        ) = _directional_shared_fractions(
            shared,
            total,
            relative_tolerance=tolerance,
        )

        object.__setattr__(self, "conditional_covariance", _readonly(conditional))
        object.__setattr__(self, "shared_query_factor", _readonly(shared_factor))
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "relative_rank_tolerance", tolerance)
        object.__setattr__(self, "shared_covariance", _readonly(shared))
        object.__setattr__(self, "total_covariance", _readonly(total))
        object.__setattr__(self, "query_dimension", query_dimension)
        object.__setattr__(self, "shared_rank_column_count", int(shared_factor.shape[1]))
        object.__setattr__(
            self,
            "total_effective_rank",
            _effective_rank(total, relative_tolerance=tolerance),
        )
        object.__setattr__(
            self,
            "shared_effective_rank",
            _effective_rank(shared, relative_tolerance=tolerance),
        )
        object.__setattr__(self, "active_query_dimension", active_dimension)
        object.__setattr__(self, "conditional_trace", conditional_trace)
        object.__setattr__(self, "shared_trace", shared_trace)
        object.__setattr__(self, "total_trace", total_trace)
        object.__setattr__(self, "shared_trace_fraction", shared_trace_fraction)
        object.__setattr__(self, "shared_frobenius_fraction", shared_frobenius_fraction)
        object.__setattr__(
            self,
            "coordinate_shared_fractions",
            _coordinate_shared_fractions(shared, total),
        )
        object.__setattr__(
            self,
            "minimum_directional_shared_fraction",
            minimum_directional,
        )
        object.__setattr__(self, "mean_directional_shared_fraction", mean_directional)
        object.__setattr__(
            self,
            "maximum_directional_shared_fraction",
            maximum_directional,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact JSON-compatible diagnostic summary."""

        return {
            "schema": QUERY_COVARIANCE_RELEVANCE_SCHEMA,
            "version": QUERY_COVARIANCE_RELEVANCE_VERSION,
            "observation_count": self.observation_count,
            "query_dimension": self.query_dimension,
            "shared_rank_column_count": self.shared_rank_column_count,
            "total_effective_rank": self.total_effective_rank,
            "shared_effective_rank": self.shared_effective_rank,
            "active_query_dimension": self.active_query_dimension,
            "conditional_trace": self.conditional_trace,
            "shared_trace": self.shared_trace,
            "total_trace": self.total_trace,
            "shared_trace_fraction": self.shared_trace_fraction,
            "shared_frobenius_fraction": self.shared_frobenius_fraction,
            "coordinate_shared_fractions": list(self.coordinate_shared_fractions),
            "minimum_directional_shared_fraction": (
                self.minimum_directional_shared_fraction
            ),
            "mean_directional_shared_fraction": self.mean_directional_shared_fraction,
            "maximum_directional_shared_fraction": (
                self.maximum_directional_shared_fraction
            ),
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "claim_boundary": QUERY_COVARIANCE_RELEVANCE_CLAIM_BOUNDARY,
        }


def project_joint_covariance_to_query(
    query_jacobian: object,
    local_covariance_m2: object,
    low_rank_factor_m: object,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> QueryCovarianceProjectionV1:
    """Project ``blockdiag(D_i) + U U.T`` through a block query Jacobian.

    ``query_jacobian[q, i, :]`` is the derivative of query coordinate ``q``
    with respect to observation row ``i``. A two-dimensional ``(N, 3)`` input
    denotes a scalar query. The implementation never materializes the dense
    observation covariance or a dense ``Q x 3N`` Jacobian.
    """

    tolerance = _validated_relative_rank_tolerance(relative_rank_tolerance)
    jacobian, local, factor = _validated_inputs(
        query_jacobian,
        local_covariance_m2,
        low_rank_factor_m,
    )
    conditional = np.einsum(
        "qni,nij,rnj->qr",
        jacobian,
        local,
        jacobian,
        optimize=True,
    )
    shared_query_factor = np.einsum(
        "qni,nik->qk",
        jacobian,
        factor,
        optimize=True,
    )
    return QueryCovarianceProjectionV1(
        conditional_covariance=conditional,
        shared_query_factor=shared_query_factor,
        observation_count=int(jacobian.shape[1]),
        relative_rank_tolerance=tolerance,
    )


__all__ = [
    "QUERY_COVARIANCE_RELEVANCE_CLAIM_BOUNDARY",
    "QUERY_COVARIANCE_RELEVANCE_SCHEMA",
    "QUERY_COVARIANCE_RELEVANCE_VERSION",
    "QueryCovarianceProjectionV1",
    "project_joint_covariance_to_query",
]
