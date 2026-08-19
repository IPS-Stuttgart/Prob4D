"""Reduce shared observation covariance while preserving registered queries.

The shared Prob4D factor can contain many latent columns even when a downstream
physical query is sensitive to only a small subspace.  Observation-trace-only
truncation may nevertheless discard a low-energy direction that dominates that
query.  This experimental module constructs one deterministic latent subspace,
then admits the smallest whole-eigenspace rank that satisfies both observation-
and query-space loss limits.

The routine does not modify a claim-bearing observation, choose a physical
query, or authorize a Bayesian update.  When no admissible reduction exists
within the requested rank budget, it returns the exact caller-supplied full-rank
factor and records an explicit fallback reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]

QUERY_PRESERVING_COMPRESSION_SCHEMA = "prob4d.query-preserving-compression"
QUERY_PRESERVING_COMPRESSION_VERSION = 1
QUERY_PRESERVING_COMPRESSION_CLAIM_BOUNDARY = (
    "This experimental diagnostic compresses a supplied shared observation "
    "factor against caller-supplied local query Jacobians. It does not define "
    "the physical query, alter a claim-bearing observation artifact, establish "
    "provider competence, authorize a BayesianPhysTwin update, or establish "
    "Causal4D benefit."
)


def _readonly(value: object) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _validated_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


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


def _validated_positive_tolerance(value: object, *, name: str) -> float:
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
    return factor


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
    return jacobian


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


def _validated_query_weights(
    names: tuple[str, ...],
    query_weights: Mapping[str, object] | None,
) -> tuple[float, ...]:
    if query_weights is None:
        return tuple(1.0 for _ in names)
    if not isinstance(query_weights, Mapping):
        raise TypeError("query_weights must be a mapping when supplied")
    unknown = sorted(set(query_weights) - set(names))
    missing = sorted(set(names) - set(query_weights))
    if unknown or missing:
        raise ValueError(
            "query_weights must name exactly the registered queries; "
            f"missing={missing}, unknown={unknown}"
        )
    return tuple(
        _validated_nonnegative(query_weights[name], name=f"query_weights[{name!r}]")
        for name in names
    )


def _eigenvalues_equal(
    first: float,
    second: float,
    *,
    spectral_scale: float,
    relative_tolerance: float,
) -> bool:
    tolerance = 1e-14 + relative_tolerance * spectral_scale
    return abs(float(first) - float(second)) <= tolerance


def _canonical_projector_basis(vectors: FloatArray) -> FloatArray:
    dimension, rank = vectors.shape
    if rank == 0:
        return np.empty((dimension, 0), dtype=np.float64)
    projector = vectors @ vectors.T
    basis: list[FloatArray] = []
    tolerance = 256.0 * np.finfo(np.float64).eps * max(1, dimension)
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
        basis.append(candidate)
        if len(basis) == rank:
            break
    if len(basis) != rank:
        raise RuntimeError("failed to construct a canonical latent eigenspace basis")
    result = np.column_stack(basis)
    if not np.allclose(
        result.T @ result,
        np.eye(rank),
        atol=1e-11,
        rtol=1e-11,
    ):
        raise RuntimeError("canonical latent eigenspace basis lost orthogonality")
    return result


def _canonical_score_basis(
    score: FloatArray,
    *,
    relative_eigenspace_tolerance: float,
) -> tuple[FloatArray, FloatArray, tuple[int, ...]]:
    symmetric = 0.5 * (score + score.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        float(np.finfo(np.float64).tiny),
    )
    basis_parts: list[FloatArray] = []
    boundaries: list[int] = []
    position = 0
    while position < len(eigenvalues):
        stop = position + 1
        while stop < len(eigenvalues) and _eigenvalues_equal(
            float(eigenvalues[stop - 1]),
            float(eigenvalues[stop]),
            spectral_scale=spectral_scale,
            relative_tolerance=relative_eigenspace_tolerance,
        ):
            stop += 1
        basis_parts.append(_canonical_projector_basis(eigenvectors[:, position:stop]))
        boundaries.append(stop)
        position = stop
    basis = (
        np.concatenate(basis_parts, axis=1)
        if basis_parts
        else np.empty((0, 0), dtype=np.float64)
    )
    return basis, eigenvalues, tuple(boundaries)


def _query_factor(jacobian: FloatArray, factor: FloatArray) -> FloatArray:
    return np.einsum("qni,nir->qr", jacobian, factor, optimize=True)


def _trace_fraction(retained: float, total: float) -> float:
    if total <= 0.0:
        return 1.0
    return float(np.clip(retained / total, 0.0, 1.0))


def _loss_fractions(full_factor: FloatArray, reduced_factor: FloatArray) -> tuple[float, float]:
    full_covariance = full_factor @ full_factor.T
    reduced_covariance = reduced_factor @ reduced_factor.T
    loss = 0.5 * (
        (full_covariance - reduced_covariance)
        + (full_covariance - reduced_covariance).T
    )
    scale = max(float(np.max(np.abs(full_covariance), initial=0.0)), 1.0)
    minimum = float(np.min(np.linalg.eigvalsh(loss), initial=0.0))
    if minimum < -(1e-12 + 1e-10 * scale):
        raise RuntimeError("latent projection increased a registered query covariance")
    loss = 0.5 * (loss + loss.T)
    full_trace = float(np.trace(full_covariance))
    trace_loss = 0.0 if full_trace <= 0.0 else float(np.trace(loss) / full_trace)
    full_spectral = float(np.linalg.norm(full_covariance, ord=2))
    spectral_loss = (
        0.0
        if full_spectral <= 0.0
        else float(np.linalg.norm(loss, ord=2) / full_spectral)
    )
    return (
        float(np.clip(trace_loss, 0.0, 1.0)),
        float(np.clip(spectral_loss, 0.0, 1.0)),
    )


@dataclass(frozen=True, slots=True)
class QueryPreservingCompressionPolicyV1:
    """Frozen admission limits and latent-subspace weighting."""

    minimum_observation_trace_fraction: float
    maximum_query_trace_loss_fraction: float
    maximum_query_spectral_loss_fraction: float
    maximum_rank: int | None = None
    minimum_rank: int = 0
    observation_weight: float = 1.0
    query_weights: Mapping[str, object] | None = None
    relative_eigenspace_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        minimum_observation = _validated_fraction(
            self.minimum_observation_trace_fraction,
            name="minimum_observation_trace_fraction",
        )
        maximum_query_trace = _validated_fraction(
            self.maximum_query_trace_loss_fraction,
            name="maximum_query_trace_loss_fraction",
        )
        maximum_query_spectral = _validated_fraction(
            self.maximum_query_spectral_loss_fraction,
            name="maximum_query_spectral_loss_fraction",
        )
        minimum_rank = _validated_rank(self.minimum_rank, name="minimum_rank")
        maximum_rank = (
            None
            if self.maximum_rank is None
            else _validated_rank(self.maximum_rank, name="maximum_rank")
        )
        if maximum_rank is not None and maximum_rank < minimum_rank:
            raise ValueError("maximum_rank must be at least minimum_rank")
        observation_weight = _validated_nonnegative(
            self.observation_weight,
            name="observation_weight",
        )
        relative_tolerance = _validated_positive_tolerance(
            self.relative_eigenspace_tolerance,
            name="relative_eigenspace_tolerance",
        )
        if self.query_weights is None:
            copied_weights: Mapping[str, object] | None = None
        else:
            if not isinstance(self.query_weights, Mapping):
                raise TypeError("query_weights must be a mapping when supplied")
            validated_weights: dict[str, float] = {}
            for raw_name, raw_weight in self.query_weights.items():
                if not isinstance(raw_name, str) or not raw_name:
                    raise ValueError("every query weight name must be a nonempty string")
                validated_weights[raw_name] = _validated_nonnegative(
                    raw_weight,
                    name=f"query_weights[{raw_name!r}]",
                )
            copied_weights = MappingProxyType(
                {name: validated_weights[name] for name in sorted(validated_weights)}
            )
        object.__setattr__(
            self,
            "minimum_observation_trace_fraction",
            minimum_observation,
        )
        object.__setattr__(
            self,
            "maximum_query_trace_loss_fraction",
            maximum_query_trace,
        )
        object.__setattr__(
            self,
            "maximum_query_spectral_loss_fraction",
            maximum_query_spectral,
        )
        object.__setattr__(self, "minimum_rank", minimum_rank)
        object.__setattr__(self, "maximum_rank", maximum_rank)
        object.__setattr__(self, "observation_weight", observation_weight)
        object.__setattr__(self, "query_weights", copied_weights)
        object.__setattr__(
            self,
            "relative_eigenspace_tolerance",
            relative_tolerance,
        )


@dataclass(frozen=True, slots=True)
class QueryCompressionDiagnosticV1:
    """One registered query's covariance loss after compression."""

    name: str
    query_dimension: int
    full_trace: float
    trace_loss_fraction: float
    spectral_loss_fraction: float

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
        trace_loss = _validated_fraction(
            self.trace_loss_fraction,
            name="trace_loss_fraction",
        )
        spectral_loss = _validated_fraction(
            self.spectral_loss_fraction,
            name="spectral_loss_fraction",
        )
        object.__setattr__(self, "query_dimension", query_dimension)
        object.__setattr__(self, "full_trace", full_trace)
        object.__setattr__(self, "trace_loss_fraction", trace_loss)
        object.__setattr__(self, "spectral_loss_fraction", spectral_loss)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "query_dimension": self.query_dimension,
            "full_trace": self.full_trace,
            "trace_loss_fraction": self.trace_loss_fraction,
            "spectral_loss_fraction": self.spectral_loss_fraction,
        }


@dataclass(frozen=True, slots=True)
class QueryPreservingCompressionResultV1:
    """Immutable compressed factor or exact full-rank fallback."""

    compressed_factor_m: FloatArray
    latent_projection: FloatArray
    score_eigenvalues: FloatArray
    query_diagnostics: tuple[QueryCompressionDiagnosticV1, ...]
    original_rank: int
    observation_trace_fraction: float
    compression_applied: bool
    fallback_reason: str | None
    policy: QueryPreservingCompressionPolicyV1
    retained_rank: int = field(init=False)

    def __post_init__(self) -> None:
        original_rank = _validated_rank(self.original_rank, name="original_rank")
        factor = np.asarray(self.compressed_factor_m, dtype=np.float64)
        if factor.ndim != 3 or factor.shape[1] != 3:
            raise ValueError("compressed_factor_m must have shape (N, 3, K)")
        projection = np.asarray(self.latent_projection, dtype=np.float64)
        if projection.shape != (original_rank, factor.shape[2]):
            raise ValueError("latent_projection has an invalid shape")
        eigenvalues = np.asarray(self.score_eigenvalues, dtype=np.float64)
        if eigenvalues.shape != (original_rank,):
            raise ValueError("score_eigenvalues has an invalid shape")
        if not np.all(np.isfinite(factor)) or not np.all(np.isfinite(projection)):
            raise ValueError("compression result arrays must be finite")
        if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0.0):
            raise ValueError("score_eigenvalues must be finite and nonnegative")
        if projection.shape[1] and not np.allclose(
            projection.T @ projection,
            np.eye(projection.shape[1]),
            atol=1e-11,
            rtol=1e-11,
        ):
            raise ValueError("latent_projection columns must be orthonormal")
        if np.any(np.diff(eigenvalues) > 1e-12):
            raise ValueError("score_eigenvalues must be sorted in descending order")
        diagnostic_names = tuple(
            diagnostic.name for diagnostic in self.query_diagnostics
        )
        if not diagnostic_names or diagnostic_names != tuple(sorted(diagnostic_names)):
            raise ValueError("query_diagnostics must be nonempty and sorted by name")
        if len(diagnostic_names) != len(set(diagnostic_names)):
            raise ValueError("query_diagnostics names must be unique")
        observation_fraction = _validated_fraction(
            self.observation_trace_fraction,
            name="observation_trace_fraction",
        )
        if self.compression_applied and factor.shape[2] >= original_rank:
            raise ValueError("compression_applied requires a strict rank reduction")
        if self.compression_applied and self.fallback_reason is not None:
            raise ValueError("a compressed result cannot carry a fallback_reason")
        if not self.compression_applied and self.fallback_reason is None:
            raise ValueError("an uncompressed result requires a fallback_reason")
        if self.compression_applied:
            if factor.shape[2] < self.policy.minimum_rank:
                raise ValueError("compressed rank is below the policy minimum")
            if (
                self.policy.maximum_rank is not None
                and factor.shape[2] > self.policy.maximum_rank
            ):
                raise ValueError("compressed rank exceeds the policy maximum")
            if (
                observation_fraction
                < self.policy.minimum_observation_trace_fraction - 1e-12
            ):
                raise ValueError("compressed result violates observation trace policy")
            if any(
                diagnostic.trace_loss_fraction
                > self.policy.maximum_query_trace_loss_fraction + 1e-12
                or diagnostic.spectral_loss_fraction
                > self.policy.maximum_query_spectral_loss_fraction + 1e-12
                for diagnostic in self.query_diagnostics
            ):
                raise ValueError("compressed result violates a registered query policy")
        else:
            if factor.shape[2] != original_rank:
                raise ValueError("full-rank fallback must retain every original column")
            if not np.array_equal(projection, np.eye(original_rank)):
                raise ValueError("full-rank fallback must use the identity projection")
            if observation_fraction != 1.0:
                raise ValueError("full-rank fallback must retain all observation trace")
            if any(
                diagnostic.trace_loss_fraction != 0.0
                or diagnostic.spectral_loss_fraction != 0.0
                for diagnostic in self.query_diagnostics
            ):
                raise ValueError("full-rank fallback must report zero query loss")
        object.__setattr__(self, "original_rank", original_rank)
        object.__setattr__(self, "compressed_factor_m", _readonly(factor))
        object.__setattr__(self, "latent_projection", _readonly(projection))
        object.__setattr__(self, "score_eigenvalues", _readonly(eigenvalues))
        object.__setattr__(self, "observation_trace_fraction", observation_fraction)
        object.__setattr__(self, "retained_rank", int(factor.shape[2]))

    def summary(self) -> dict[str, object]:
        return {
            "schema": QUERY_PRESERVING_COMPRESSION_SCHEMA,
            "version": QUERY_PRESERVING_COMPRESSION_VERSION,
            "original_rank": self.original_rank,
            "retained_rank": self.retained_rank,
            "compression_applied": self.compression_applied,
            "fallback_reason": self.fallback_reason,
            "observation_trace_fraction": self.observation_trace_fraction,
            "query_diagnostics": [
                diagnostic.summary() for diagnostic in self.query_diagnostics
            ],
            "policy": {
                "minimum_observation_trace_fraction": (
                    self.policy.minimum_observation_trace_fraction
                ),
                "maximum_query_trace_loss_fraction": (
                    self.policy.maximum_query_trace_loss_fraction
                ),
                "maximum_query_spectral_loss_fraction": (
                    self.policy.maximum_query_spectral_loss_fraction
                ),
                "minimum_rank": self.policy.minimum_rank,
                "maximum_rank": self.policy.maximum_rank,
                "observation_weight": self.policy.observation_weight,
                "query_weights": (
                    None
                    if self.policy.query_weights is None
                    else dict(self.policy.query_weights)
                ),
                "relative_eigenspace_tolerance": (
                    self.policy.relative_eigenspace_tolerance
                ),
            },
            "claim_boundary": QUERY_PRESERVING_COMPRESSION_CLAIM_BOUNDARY,
        }


def compress_shared_factor_for_queries(
    low_rank_factor_m: object,
    query_jacobians: Mapping[str, object],
    *,
    policy: QueryPreservingCompressionPolicyV1,
) -> QueryPreservingCompressionResultV1:
    """Return the smallest admitted whole-eigenspace latent projection.

    The latent score combines normalized observation energy and normalized
    registered-query energy.  Candidate ranks are considered only at complete
    numerical eigenspace boundaries.  The first candidate satisfying every
    frozen loss limit is returned.  If no strict reduction satisfies the limits
    within ``maximum_rank``, the original factor and an identity projection are
    returned unchanged as an explicit full-rank fallback.
    """

    if not isinstance(policy, QueryPreservingCompressionPolicyV1):
        raise TypeError("policy must be QueryPreservingCompressionPolicyV1")
    factor = _validated_factor(low_rank_factor_m)
    sample_count = int(factor.shape[0])
    original_rank = int(factor.shape[2])
    if policy.minimum_rank > original_rank:
        raise ValueError("minimum_rank exceeds the factor rank")

    queries = _validated_named_queries(
        query_jacobians,
        sample_count=sample_count,
    )
    names = tuple(name for name, _ in queries)
    query_weights = _validated_query_weights(names, policy.query_weights)
    if policy.observation_weight == 0.0 and not any(
        weight > 0.0 for weight in query_weights
    ):
        raise ValueError("at least one observation or query weight must be positive")

    flattened = factor.reshape(sample_count * 3, original_rank)
    observation_gram = flattened.T @ flattened
    observation_trace = float(np.trace(observation_gram))
    query_factors = tuple(
        (name, _query_factor(jacobian, factor)) for name, jacobian in queries
    )

    score = np.zeros((original_rank, original_rank), dtype=np.float64)
    if observation_trace > 0.0 and policy.observation_weight > 0.0:
        score += policy.observation_weight * observation_gram / observation_trace
    for weight, (_, query_factor) in zip(query_weights, query_factors, strict=True):
        query_trace = float(np.sum(query_factor * query_factor))
        if weight > 0.0 and query_trace > 0.0:
            score += weight * (query_factor.T @ query_factor) / query_trace

    basis, score_eigenvalues, boundaries = _canonical_score_basis(
        score,
        relative_eigenspace_tolerance=policy.relative_eigenspace_tolerance,
    )
    candidate_ranks = [0, *boundaries]
    maximum_candidate_rank = (
        original_rank
        if policy.maximum_rank is None
        else min(policy.maximum_rank, original_rank)
    )

    selected_projection: FloatArray | None = None
    selected_factor: FloatArray | None = None
    selected_observation_fraction = 1.0
    selected_diagnostics: tuple[QueryCompressionDiagnosticV1, ...] | None = None

    for rank in candidate_ranks:
        if rank < policy.minimum_rank or rank > maximum_candidate_rank:
            continue
        projection = basis[:, :rank]
        reduced_flattened = flattened @ projection
        reduced = reduced_flattened.reshape(sample_count, 3, rank)
        retained_observation_trace = float(
            np.trace(projection.T @ observation_gram @ projection)
        )
        observation_fraction = _trace_fraction(
            retained_observation_trace,
            observation_trace,
        )
        diagnostics: list[QueryCompressionDiagnosticV1] = []
        queries_pass = True
        for name, full_query_factor in query_factors:
            reduced_query_factor = full_query_factor @ projection
            trace_loss, spectral_loss = _loss_fractions(
                full_query_factor,
                reduced_query_factor,
            )
            diagnostics.append(
                QueryCompressionDiagnosticV1(
                    name=name,
                    query_dimension=int(full_query_factor.shape[0]),
                    full_trace=float(np.sum(full_query_factor * full_query_factor)),
                    trace_loss_fraction=trace_loss,
                    spectral_loss_fraction=spectral_loss,
                )
            )
            queries_pass = queries_pass and (
                trace_loss <= policy.maximum_query_trace_loss_fraction + 1e-12
                and spectral_loss
                <= policy.maximum_query_spectral_loss_fraction + 1e-12
            )
        if (
            observation_fraction
            >= policy.minimum_observation_trace_fraction - 1e-12
            and queries_pass
        ):
            selected_projection = projection
            selected_factor = reduced
            selected_observation_fraction = observation_fraction
            selected_diagnostics = tuple(diagnostics)
            break

    if selected_factor is not None and selected_factor.shape[2] < original_rank:
        assert selected_projection is not None
        assert selected_diagnostics is not None
        return QueryPreservingCompressionResultV1(
            compressed_factor_m=selected_factor,
            latent_projection=selected_projection,
            score_eigenvalues=score_eigenvalues,
            query_diagnostics=selected_diagnostics,
            original_rank=original_rank,
            observation_trace_fraction=selected_observation_fraction,
            compression_applied=True,
            fallback_reason=None,
            policy=policy,
        )

    identity = np.eye(original_rank, dtype=np.float64)
    full_diagnostics = tuple(
        QueryCompressionDiagnosticV1(
            name=name,
            query_dimension=int(query_factor.shape[0]),
            full_trace=float(np.sum(query_factor * query_factor)),
            trace_loss_fraction=0.0,
            spectral_loss_fraction=0.0,
        )
        for name, query_factor in query_factors
    )
    reason = (
        "full-rank-required"
        if policy.maximum_rank is None or policy.maximum_rank >= original_rank
        else "no-admissible-reduction-within-rank-cap"
    )
    return QueryPreservingCompressionResultV1(
        compressed_factor_m=factor,
        latent_projection=identity,
        score_eigenvalues=score_eigenvalues,
        query_diagnostics=full_diagnostics,
        original_rank=original_rank,
        observation_trace_fraction=1.0,
        compression_applied=False,
        fallback_reason=reason,
        policy=policy,
    )


__all__ = [
    "QUERY_PRESERVING_COMPRESSION_CLAIM_BOUNDARY",
    "QUERY_PRESERVING_COMPRESSION_SCHEMA",
    "QUERY_PRESERVING_COMPRESSION_VERSION",
    "QueryCompressionDiagnosticV1",
    "QueryPreservingCompressionPolicyV1",
    "QueryPreservingCompressionResultV1",
    "compress_shared_factor_for_queries",
]
