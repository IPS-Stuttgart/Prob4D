"""Decompose shared covariance into query-coupled and query-orthogonal factors.

For a shared observation factor ``U`` and registered linear query Jacobians
``J_h``, the stacked query factor is ``B = stack_h(J_h U)``.  The row space of
``B`` is the unique minimum-dimensional latent subspace that preserves every
registered query covariance and cross-covariance.  Its orthogonal complement is
invisible to the registered projections but remains observation uncertainty.

This module computes a deterministic numerical basis for both subspaces.  The
full observation covariance is preserved by retaining both returned factors.
The query-orthogonal factor must not be discarded from a Gaussian likelihood or
conditioning operation merely because it is invisible under the registered
forward projections.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]

EXACT_QUERY_COVARIANCE_DECOMPOSITION_SCHEMA = (
    "prob4d.exact-query-covariance-decomposition"
)
EXACT_QUERY_COVARIANCE_DECOMPOSITION_VERSION = 1
EXACT_QUERY_COVARIANCE_DECOMPOSITION_CLAIM_BOUNDARY = (
    "This experimental source-side diagnostic decomposes a supplied shared "
    "observation factor relative to caller-supplied linear query Jacobians. "
    "Both returned factors are required to preserve the full observation "
    "likelihood. It does not define a physical query, authorize removal of "
    "query-orthogonal uncertainty, alter a claim-bearing observation, establish "
    "provider competence, authorize a BayesianPhysTwin update, or establish "
    "Causal4D benefit."
)


def _readonly(value: object, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    copied = result.copy()
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _validated_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validated_rank(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validated_optional_rank(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _validated_rank(value, name=name)


def _validated_relative_tolerance(value: object, *, name: str) -> float:
    result = _validated_nonnegative(value, name=name)
    if result >= 1.0:
        raise ValueError(f"{name} must lie in [0, 1)")
    return result


def _validated_factor(value: object) -> FloatArray:
    factor = np.asarray(value, dtype=np.float64)
    if factor.ndim != 3 or factor.shape[0] < 1 or factor.shape[1] != 3:
        raise ValueError("low_rank_factor_m must have shape (N, 3, R) with N positive")
    if not np.all(np.isfinite(factor)):
        raise ValueError("low_rank_factor_m must be finite")
    return cast(FloatArray, factor)


def _validated_query(value: object, *, sample_count: int, name: str) -> FloatArray:
    jacobian = np.asarray(value, dtype=np.float64)
    if jacobian.ndim == 2:
        if jacobian.shape != (sample_count, 3):
            raise ValueError(
                f"query_jacobians[{name!r}] must have shape (N, 3) or (Q, N, 3)"
            )
        jacobian = jacobian[None, ...]
    elif jacobian.ndim != 3 or jacobian.shape[1:] != (sample_count, 3):
        raise ValueError(
            f"query_jacobians[{name!r}] must have shape (N, 3) or (Q, N, 3)"
        )
    if jacobian.shape[0] < 1:
        raise ValueError(f"query_jacobians[{name!r}] must contain a query coordinate")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError(f"query_jacobians[{name!r}] must be finite")
    return cast(FloatArray, jacobian)


def _validated_named_queries(
    query_jacobians: Mapping[str, object],
    *,
    sample_count: int,
) -> tuple[tuple[str, FloatArray], ...]:
    if not isinstance(query_jacobians, Mapping):
        raise TypeError("query_jacobians must be a mapping from names to arrays")
    if not query_jacobians:
        raise ValueError("query_jacobians must not be empty")
    queries: list[tuple[str, FloatArray]] = []
    for raw_name, value in query_jacobians.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("every query name must be a nonempty string")
        queries.append(
            (
                raw_name,
                _validated_query(value, sample_count=sample_count, name=raw_name),
            )
        )
    queries.sort(key=lambda item: item[0])
    return tuple(queries)


def _query_factor(jacobian: FloatArray, factor: FloatArray) -> FloatArray:
    return cast(
        FloatArray,
        np.einsum("qni,nir->qr", jacobian, factor, optimize=True),
    )


def _canonical_projector_basis(vectors: FloatArray) -> FloatArray:
    dimension, rank = vectors.shape
    if rank == 0:
        return np.empty((dimension, 0), dtype=np.float64)
    projector = 0.5 * (vectors @ vectors.T + (vectors @ vectors.T).T)
    basis: list[FloatArray] = []
    tolerance = 512.0 * np.finfo(np.float64).eps * max(1, dimension)
    for axis in range(dimension):
        candidate = projector[:, axis].copy()
        for _ in range(2):
            for previous in basis:
                candidate -= previous * float(previous @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        basis.append(cast(FloatArray, candidate))
        if len(basis) == rank:
            break
    if len(basis) != rank:
        raise RuntimeError("failed to construct a canonical latent-subspace basis")
    result = np.column_stack(basis)
    if not np.allclose(
        result.T @ result,
        np.eye(rank),
        atol=1e-11,
        rtol=1e-11,
    ):
        raise RuntimeError("canonical latent-subspace basis lost orthogonality")
    return cast(FloatArray, result)


def _relative_frobenius_error(value: FloatArray, reference: FloatArray) -> float:
    denominator = float(np.linalg.norm(reference, ord="fro"))
    if denominator <= 0.0:
        return 0.0
    return float(np.linalg.norm(value, ord="fro") / denominator)


@dataclass(frozen=True, slots=True)
class ExactQueryCovarianceDiagnosticV1:
    """Numerical preservation diagnostics for one registered query."""

    name: str
    query_dimension: int
    full_trace: float
    relative_covariance_error: float
    relative_query_orthogonal_factor_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        query_dimension = _validated_rank(
            self.query_dimension,
            name="query_dimension",
        )
        if query_dimension < 1:
            raise ValueError("query_dimension must be positive")
        full_trace = _validated_nonnegative(self.full_trace, name="full_trace")
        covariance_error = _validated_nonnegative(
            self.relative_covariance_error,
            name="relative_covariance_error",
        )
        orthogonal_norm = _validated_nonnegative(
            self.relative_query_orthogonal_factor_norm,
            name="relative_query_orthogonal_factor_norm",
        )
        object.__setattr__(self, "query_dimension", query_dimension)
        object.__setattr__(self, "full_trace", full_trace)
        object.__setattr__(self, "relative_covariance_error", covariance_error)
        object.__setattr__(
            self,
            "relative_query_orthogonal_factor_norm",
            orthogonal_norm,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "query_dimension": self.query_dimension,
            "full_trace": self.full_trace,
            "relative_covariance_error": self.relative_covariance_error,
            "relative_query_orthogonal_factor_norm": (
                self.relative_query_orthogonal_factor_norm
            ),
        }


@dataclass(frozen=True, slots=True)
class ExactQueryCovarianceDecompositionV1:
    """Minimum numerical query subspace plus its required orthogonal complement."""

    query_coupled_factor_m: FloatArray
    query_orthogonal_factor_m: FloatArray
    query_coupled_projection: FloatArray
    query_orthogonal_projection: FloatArray
    singular_values: FloatArray
    query_diagnostics: tuple[ExactQueryCovarianceDiagnosticV1, ...]
    original_rank: int
    minimum_exact_query_rank: int
    numerical_rank_threshold: float
    relative_rank_tolerance: float
    maximum_query_rank: int | None
    decomposition_applied: bool
    fallback_reason: str | None
    retained_query_rank: int = field(init=False)
    query_orthogonal_rank: int = field(init=False)

    def __post_init__(self) -> None:
        original_rank = _validated_rank(self.original_rank, name="original_rank")
        exact_rank = _validated_rank(
            self.minimum_exact_query_rank,
            name="minimum_exact_query_rank",
        )
        if exact_rank > original_rank:
            raise ValueError("minimum_exact_query_rank exceeds original_rank")
        threshold = _validated_nonnegative(
            self.numerical_rank_threshold,
            name="numerical_rank_threshold",
        )
        relative_tolerance = _validated_relative_tolerance(
            self.relative_rank_tolerance,
            name="relative_rank_tolerance",
        )
        maximum_rank = _validated_optional_rank(
            self.maximum_query_rank,
            name="maximum_query_rank",
        )

        coupled = np.asarray(self.query_coupled_factor_m, dtype=np.float64)
        orthogonal = np.asarray(self.query_orthogonal_factor_m, dtype=np.float64)
        if coupled.ndim != 3 or coupled.shape[0] < 1 or coupled.shape[1] != 3:
            raise ValueError("query_coupled_factor_m must have shape (N, 3, K)")
        if orthogonal.ndim != 3 or orthogonal.shape[:2] != coupled.shape[:2]:
            raise ValueError(
                "query_orthogonal_factor_m must have shape (N, 3, R-K)"
            )
        if not np.all(np.isfinite(coupled)) or not np.all(np.isfinite(orthogonal)):
            raise ValueError("decomposition factors must be finite")

        coupled_projection = np.asarray(
            self.query_coupled_projection,
            dtype=np.float64,
        )
        orthogonal_projection = np.asarray(
            self.query_orthogonal_projection,
            dtype=np.float64,
        )
        if coupled_projection.shape != (original_rank, coupled.shape[2]):
            raise ValueError("query_coupled_projection has an invalid shape")
        if orthogonal_projection.shape != (original_rank, orthogonal.shape[2]):
            raise ValueError("query_orthogonal_projection has an invalid shape")
        if not np.all(np.isfinite(coupled_projection)) or not np.all(
            np.isfinite(orthogonal_projection)
        ):
            raise ValueError("decomposition projections must be finite")

        combined_projection = np.concatenate(
            (coupled_projection, orthogonal_projection),
            axis=1,
        )
        if combined_projection.shape != (original_rank, original_rank):
            raise ValueError("decomposition projections must span the latent space")
        if not np.allclose(
            combined_projection.T @ combined_projection,
            np.eye(original_rank),
            atol=1e-11,
            rtol=1e-11,
        ):
            raise ValueError("decomposition projections must be jointly orthonormal")

        singular_values = np.asarray(self.singular_values, dtype=np.float64)
        if singular_values.ndim != 1 or singular_values.shape[0] > original_rank:
            raise ValueError("singular_values must be a vector no longer than original_rank")
        if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0.0):
            raise ValueError("singular_values must be finite and nonnegative")
        if np.any(np.diff(singular_values) > 1e-12):
            raise ValueError("singular_values must be sorted in descending order")

        diagnostic_names = tuple(
            diagnostic.name for diagnostic in self.query_diagnostics
        )
        if not diagnostic_names or diagnostic_names != tuple(sorted(diagnostic_names)):
            raise ValueError("query_diagnostics must be nonempty and sorted by name")
        if len(diagnostic_names) != len(set(diagnostic_names)):
            raise ValueError("query_diagnostics names must be unique")

        retained_rank = int(coupled.shape[2])
        orthogonal_rank = int(orthogonal.shape[2])
        if self.decomposition_applied:
            if self.fallback_reason is not None:
                raise ValueError("an applied decomposition cannot carry a fallback_reason")
            if retained_rank != exact_rank:
                raise ValueError("applied decomposition must use the minimum query rank")
            if orthogonal_rank != original_rank - exact_rank:
                raise ValueError("applied decomposition has an invalid orthogonal rank")
        else:
            if self.fallback_reason != "exact-query-rank-exceeds-cap":
                raise ValueError("an unapplied decomposition needs the rank-cap reason")
            if maximum_rank is None or exact_rank <= maximum_rank:
                raise ValueError("rank-cap fallback requires an exceeded maximum_query_rank")
            if retained_rank != original_rank or orthogonal_rank != 0:
                raise ValueError("rank-cap fallback must retain the exact full factor")
            if not np.array_equal(coupled_projection, np.eye(original_rank)):
                raise ValueError("rank-cap fallback must use the identity projection")

        object.__setattr__(self, "original_rank", original_rank)
        object.__setattr__(self, "minimum_exact_query_rank", exact_rank)
        object.__setattr__(self, "numerical_rank_threshold", threshold)
        object.__setattr__(self, "relative_rank_tolerance", relative_tolerance)
        object.__setattr__(self, "maximum_query_rank", maximum_rank)
        object.__setattr__(
            self,
            "query_coupled_factor_m",
            _readonly(coupled, name="query_coupled_factor_m"),
        )
        object.__setattr__(
            self,
            "query_orthogonal_factor_m",
            _readonly(orthogonal, name="query_orthogonal_factor_m"),
        )
        object.__setattr__(
            self,
            "query_coupled_projection",
            _readonly(coupled_projection, name="query_coupled_projection"),
        )
        object.__setattr__(
            self,
            "query_orthogonal_projection",
            _readonly(orthogonal_projection, name="query_orthogonal_projection"),
        )
        object.__setattr__(
            self,
            "singular_values",
            _readonly(singular_values, name="singular_values"),
        )
        object.__setattr__(self, "retained_query_rank", retained_rank)
        object.__setattr__(self, "query_orthogonal_rank", orthogonal_rank)

    @property
    def strict_query_rank_reduction(self) -> bool:
        return self.retained_query_rank < self.original_rank

    def summary(self) -> dict[str, object]:
        return {
            "schema": EXACT_QUERY_COVARIANCE_DECOMPOSITION_SCHEMA,
            "version": EXACT_QUERY_COVARIANCE_DECOMPOSITION_VERSION,
            "original_rank": self.original_rank,
            "minimum_exact_query_rank": self.minimum_exact_query_rank,
            "retained_query_rank": self.retained_query_rank,
            "query_orthogonal_rank": self.query_orthogonal_rank,
            "strict_query_rank_reduction": self.strict_query_rank_reduction,
            "decomposition_applied": self.decomposition_applied,
            "fallback_reason": self.fallback_reason,
            "numerical_rank_threshold": self.numerical_rank_threshold,
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "maximum_query_rank": self.maximum_query_rank,
            "query_diagnostics": [
                diagnostic.summary() for diagnostic in self.query_diagnostics
            ],
            "claim_boundary": EXACT_QUERY_COVARIANCE_DECOMPOSITION_CLAIM_BOUNDARY,
        }


def _diagnostics(
    query_factors: tuple[tuple[str, FloatArray], ...],
    *,
    coupled_projection: FloatArray,
    orthogonal_projection: FloatArray,
) -> tuple[ExactQueryCovarianceDiagnosticV1, ...]:
    diagnostics: list[ExactQueryCovarianceDiagnosticV1] = []
    for name, full_query_factor in query_factors:
        coupled_query_factor = full_query_factor @ coupled_projection
        orthogonal_query_factor = full_query_factor @ orthogonal_projection
        full_covariance = full_query_factor @ full_query_factor.T
        coupled_covariance = coupled_query_factor @ coupled_query_factor.T
        full_factor_norm = float(np.linalg.norm(full_query_factor, ord="fro"))
        orthogonal_factor_norm = float(
            np.linalg.norm(orthogonal_query_factor, ord="fro")
        )
        diagnostics.append(
            ExactQueryCovarianceDiagnosticV1(
                name=name,
                query_dimension=int(full_query_factor.shape[0]),
                full_trace=float(np.sum(full_query_factor * full_query_factor)),
                relative_covariance_error=_relative_frobenius_error(
                    full_covariance - coupled_covariance,
                    full_covariance,
                ),
                relative_query_orthogonal_factor_norm=(
                    0.0
                    if full_factor_norm <= 0.0
                    else orthogonal_factor_norm / full_factor_norm
                ),
            )
        )
    return tuple(diagnostics)


def decompose_shared_factor_for_exact_queries(
    low_rank_factor_m: object,
    query_jacobians: Mapping[str, object],
    *,
    maximum_query_rank: int | None = None,
    relative_rank_tolerance: float = 0.0,
) -> ExactQueryCovarianceDecompositionV1:
    """Return the minimum numerical query subspace and its exact complement.

    Let ``B`` stack every registered query factor ``J_h U``.  The query-coupled
    projection spans the numerical row space of ``B``.  The orthogonal projection
    spans its null space.  Consequently, retaining both factors reconstructs
    ``U U.T`` exactly up to floating-point arithmetic, while the coupled factor
    alone preserves every registered query covariance and cross-covariance up to
    the declared numerical-rank tolerance.

    ``maximum_query_rank`` is an admission cap, not a truncation request.  When
    the minimum required query rank exceeds that cap, the routine fails closed by
    returning the untouched full factor as the query-coupled factor and an empty
    orthogonal factor.
    """

    factor = _validated_factor(low_rank_factor_m)
    sample_count = int(factor.shape[0])
    original_rank = int(factor.shape[2])
    maximum_rank = _validated_optional_rank(
        maximum_query_rank,
        name="maximum_query_rank",
    )
    relative_tolerance = _validated_relative_tolerance(
        relative_rank_tolerance,
        name="relative_rank_tolerance",
    )
    queries = _validated_named_queries(
        query_jacobians,
        sample_count=sample_count,
    )
    query_factors = tuple(
        (name, _query_factor(jacobian, factor)) for name, jacobian in queries
    )
    stacked_query_factor = np.concatenate(
        [query_factor for _, query_factor in query_factors],
        axis=0,
    )

    if original_rank == 0:
        singular_values = np.empty(0, dtype=np.float64)
        numerical_threshold = 0.0
        exact_rank = 0
        coupled_projection = np.empty((0, 0), dtype=np.float64)
        orthogonal_projection = np.empty((0, 0), dtype=np.float64)
    else:
        _, singular_values, right_vectors = np.linalg.svd(
            stacked_query_factor,
            full_matrices=True,
        )
        spectral_scale = (
            float(singular_values[0]) if singular_values.shape[0] else 0.0
        )
        machine_threshold = (
            np.finfo(np.float64).eps
            * max(stacked_query_factor.shape)
            * spectral_scale
        )
        numerical_threshold = max(
            machine_threshold,
            relative_tolerance * spectral_scale,
        )
        exact_rank = int(np.count_nonzero(singular_values > numerical_threshold))
        coupled_projection = _canonical_projector_basis(
            cast(FloatArray, right_vectors[:exact_rank].T)
        )
        orthogonal_projection = _canonical_projector_basis(
            cast(FloatArray, right_vectors[exact_rank:].T)
        )

    if maximum_rank is not None and exact_rank > maximum_rank:
        coupled_projection = np.eye(original_rank, dtype=np.float64)
        orthogonal_projection = np.empty((original_rank, 0), dtype=np.float64)
        coupled_factor = factor.copy()
        orthogonal_factor = np.empty((sample_count, 3, 0), dtype=np.float64)
        return ExactQueryCovarianceDecompositionV1(
            query_coupled_factor_m=coupled_factor,
            query_orthogonal_factor_m=orthogonal_factor,
            query_coupled_projection=coupled_projection,
            query_orthogonal_projection=orthogonal_projection,
            singular_values=singular_values,
            query_diagnostics=_diagnostics(
                query_factors,
                coupled_projection=coupled_projection,
                orthogonal_projection=orthogonal_projection,
            ),
            original_rank=original_rank,
            minimum_exact_query_rank=exact_rank,
            numerical_rank_threshold=numerical_threshold,
            relative_rank_tolerance=relative_tolerance,
            maximum_query_rank=maximum_rank,
            decomposition_applied=False,
            fallback_reason="exact-query-rank-exceeds-cap",
        )

    flattened = factor.reshape(sample_count * 3, original_rank)
    coupled_factor = (flattened @ coupled_projection).reshape(
        sample_count,
        3,
        exact_rank,
    )
    orthogonal_factor = (flattened @ orthogonal_projection).reshape(
        sample_count,
        3,
        original_rank - exact_rank,
    )
    return ExactQueryCovarianceDecompositionV1(
        query_coupled_factor_m=coupled_factor,
        query_orthogonal_factor_m=orthogonal_factor,
        query_coupled_projection=coupled_projection,
        query_orthogonal_projection=orthogonal_projection,
        singular_values=singular_values,
        query_diagnostics=_diagnostics(
            query_factors,
            coupled_projection=coupled_projection,
            orthogonal_projection=orthogonal_projection,
        ),
        original_rank=original_rank,
        minimum_exact_query_rank=exact_rank,
        numerical_rank_threshold=numerical_threshold,
        relative_rank_tolerance=relative_tolerance,
        maximum_query_rank=maximum_rank,
        decomposition_applied=True,
        fallback_reason=None,
    )


__all__ = [
    "EXACT_QUERY_COVARIANCE_DECOMPOSITION_CLAIM_BOUNDARY",
    "EXACT_QUERY_COVARIANCE_DECOMPOSITION_SCHEMA",
    "EXACT_QUERY_COVARIANCE_DECOMPOSITION_VERSION",
    "ExactQueryCovarianceDecompositionV1",
    "ExactQueryCovarianceDiagnosticV1",
    "decompose_shared_factor_for_exact_queries",
]
