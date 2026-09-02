"""Globally optimal rank--distortion frontier for one frozen Gaussian query.

The exact posterior-preserving compressor identifies the zero-distortion rank.
This module extends the same supplied-factor family to every retained rank.

For innovation covariance ``S = A + U U.T``, full query posterior covariance
``P``, response ``R = U.T S^-1 C.T``, and shared precision Gram matrix
``G = U.T S^-1 U``, let ``M = I - G``.  If ``N`` spans the discarded latent
subspace, the posterior covariance contraction caused by replacing ``U`` with
``U V`` (where ``V`` is the Euclidean orthogonal complement of ``N``) is

    L_N = R.T N (N.T M N)^-1 N.T R.

We use the dimensionless distortion

    D(N) = trace(P^-1 L_N).

Writing ``B = R P^-1 R.T``, the distortion is the generalized trace quotient

    trace((N.T M N)^-1 (N.T B N)).

Therefore, among all discarded subspaces of dimension ``d``, the global
minimum is the sum of the ``d`` smallest generalized eigenvalues of
``B x = lambda M x``.  This gives a nested globally optimal frontier for every
retained rank within the same orthogonal factor-projection family as the exact
theorem.  Zero distortion recovers the existing exact rank result.

The objective is posterior covariance contraction, not observation likelihood,
full posterior KL, or end-to-end task loss.  A rank cut inside a repeated
generalized-eigenvalue block has a unique optimum value but a non-unique factor
subspace; every point reports that distinction and independently audits the exact
contraction and expected posterior-normalized mean-shift risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from prob4d.posterior_preserving_compression import InnovationSolver

Array: TypeAlias = NDArray[np.float64]


def _array(value: object, name: str) -> Array:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _symmetric(value: Array, name: str) -> Array:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    scale = max(float(np.linalg.norm(value, ord="fro")), 1.0)
    if float(np.linalg.norm(value - value.T, ord="fro")) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (value + value.T)


def _cholesky(value: Array, name: str) -> Array:
    try:
        return np.linalg.cholesky(_symmetric(value, name))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite") from exc


def _whiten(root: Array, value: Array) -> Array:
    left = np.linalg.solve(root, value)
    return np.linalg.solve(root, left.T).T


def _readonly(value: Array) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _orthonormal_columns(value: Array, ambient_dimension: int) -> Array:
    """Return a deterministic Euclidean basis for the supplied column span."""
    if value.ndim != 2 or value.shape[0] != ambient_dimension:
        raise ValueError("subspace input has an invalid shape")
    if value.shape[1] == 0:
        return np.empty((ambient_dimension, 0), dtype=np.float64)
    basis, _ = np.linalg.qr(value, mode="reduced")
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis


def _orthogonal_complement(discarded: Array, ambient_dimension: int) -> Array:
    if discarded.shape == (ambient_dimension, 0):
        return np.eye(ambient_dimension, dtype=np.float64)
    _, _, right = np.linalg.svd(discarded.T, full_matrices=True)
    retained = right[discarded.shape[1] :].T.copy()
    for column in range(retained.shape[1]):
        pivot = int(np.argmax(np.abs(retained[:, column])))
        if retained[pivot, column] < 0.0:
            retained[:, column] *= -1.0
    return retained


@dataclass(frozen=True, slots=True)
class PosteriorRankDistortionPoint:
    """One globally optimal point, including uniqueness of its rank cut.

    ``optimal_subspace_unique`` is true only when the generalized-eigenvalue
    boundary is strict (or the retained/discarded subspace is trivial). When
    false, the optimum value is unique but multiple optimal factor covariances
    exist.
    """

    retained_rank: int
    discarded_dimension: int
    latent_projection: Array
    compressed_factor_m: Array
    optimal_normalized_covariance_trace_loss: float
    audited_normalized_covariance_trace_loss: float
    maximum_normalized_covariance_contraction: float
    mean_shift_risk: float
    mean_shift_risk_upper_bound: float
    boundary_generalized_eigengap: float | None
    optimal_subspace_unique: bool
    exact_posterior: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "latent_projection", _readonly(self.latent_projection))
        object.__setattr__(self, "compressed_factor_m", _readonly(self.compressed_factor_m))

    def summary(self) -> dict[str, object]:
        return {
            "retained_rank": self.retained_rank,
            "discarded_dimension": self.discarded_dimension,
            "optimal_normalized_covariance_trace_loss": (
                self.optimal_normalized_covariance_trace_loss
            ),
            "audited_normalized_covariance_trace_loss": (
                self.audited_normalized_covariance_trace_loss
            ),
            "maximum_normalized_covariance_contraction": (
                self.maximum_normalized_covariance_contraction
            ),
            "mean_shift_risk": self.mean_shift_risk,
            "mean_shift_risk_upper_bound": self.mean_shift_risk_upper_bound,
            "boundary_generalized_eigengap": self.boundary_generalized_eigengap,
            "optimal_subspace_unique": self.optimal_subspace_unique,
            "exact_posterior": self.exact_posterior,
        }


@dataclass(frozen=True, slots=True)
class PosteriorRankDistortionFrontier:
    """Nested globally optimal frontier within ``U -> U V`` projections."""

    generalized_eigenvalues: Array
    original_rank: int
    query_dimension: int
    numerical_exact_rank: int
    shared_precision_max_eigenvalue: float
    points: tuple[PosteriorRankDistortionPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generalized_eigenvalues", _readonly(self.generalized_eigenvalues))

    def point(self, retained_rank: int) -> PosteriorRankDistortionPoint:
        if not 0 <= retained_rank <= self.original_rank:
            raise ValueError("retained_rank lies outside the frontier")
        return self.points[retained_rank]

    def minimum_rank_for_trace_budget(self, budget: float) -> PosteriorRankDistortionPoint:
        if isinstance(budget, (bool, np.bool_)) or not isinstance(
            budget, (int, float, np.integer, np.floating)
        ):
            raise TypeError("budget must be a nonnegative real scalar")
        threshold = float(budget)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("budget must be finite and nonnegative")
        for point in self.points:
            if point.audited_normalized_covariance_trace_loss <= threshold:
                return point
        return self.points[-1]

    def summary(self) -> dict[str, object]:
        return {
            "original_rank": self.original_rank,
            "query_dimension": self.query_dimension,
            "numerical_exact_rank": self.numerical_exact_rank,
            "shared_precision_max_eigenvalue": self.shared_precision_max_eigenvalue,
            "generalized_eigenvalues": self.generalized_eigenvalues.tolist(),
            "points": [point.summary() for point in self.points],
            "claim_boundary": (
                "globally optimal normalized posterior-covariance trace contraction "
                "within one frozen U->UV factor-projection family; not observation evidence"
            ),
        }


def posterior_rank_distortion_frontier(
    low_rank_factor_m: object,
    *,
    prior_query_covariance: object,
    query_observation_cross_covariance: object,
    innovation_operator: InnovationSolver,
    numerical_relative_tolerance: float = 1e-12,
) -> PosteriorRankDistortionFrontier:
    """Return the globally optimal posterior trace-distortion frontier.

    The inputs have the same semantics as
    :func:`compress_shared_factor_for_posterior`: the solver represents the
    complete innovation covariance ``S`` and the factor is the supplied shared
    component ``U`` in ``S=A+UU.T``.  All non-shared covariance terms, rows,
    prior, cross covariance, linearization, and robust weights remain fixed.

    For every retained rank from zero through the original factor rank, the
    returned projection globally minimizes

    ``trace(P_full^-1 (P_full - P_reduced))``

    over all orthogonal latent factor projections of that rank.  The complete
    factor appears as the final zero-distortion point.  The routine does not use
    an observed innovation and does not preserve observation likelihood.
    """
    if isinstance(numerical_relative_tolerance, (bool, np.bool_)) or not isinstance(
        numerical_relative_tolerance, (int, float, np.integer, np.floating)
    ):
        raise TypeError("numerical_relative_tolerance must be a real scalar")
    tolerance = float(numerical_relative_tolerance)
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("numerical_relative_tolerance must lie in [0, 1)")

    factor = _array(low_rank_factor_m, "low_rank_factor_m")
    if factor.ndim != 3 or factor.shape[0] < 1 or factor.shape[1] != 3:
        raise ValueError("low_rank_factor_m must have shape (N, 3, R), N > 0")
    count, _, rank = factor.shape
    if innovation_operator.observation_count != count or innovation_operator.dimension != 3 * count:
        raise ValueError("innovation_operator dimensions do not match the factor")

    prior = _array(prior_query_covariance, "prior_query_covariance")
    if prior.ndim != 2 or prior.shape[0] < 1 or prior.shape[0] != prior.shape[1]:
        raise ValueError("prior_query_covariance must be a nonempty square matrix")
    _cholesky(prior, "prior_query_covariance")
    query_dimension = prior.shape[0]
    cross = _array(
        query_observation_cross_covariance,
        "query_observation_cross_covariance",
    )
    if cross.shape not in (
        (query_dimension, count, 3),
        (query_dimension, 3 * count),
    ):
        raise ValueError("query_observation_cross_covariance must have shape (Q,N,3) or (Q,3N)")
    cross = cross.reshape(query_dimension, 3 * count)
    u = factor.reshape(3 * count, rank)

    if rank == 0:
        point = PosteriorRankDistortionPoint(
            retained_rank=0,
            discarded_dimension=0,
            latent_projection=np.empty((0, 0)),
            compressed_factor_m=factor,
            optimal_normalized_covariance_trace_loss=0.0,
            audited_normalized_covariance_trace_loss=0.0,
            maximum_normalized_covariance_contraction=0.0,
            mean_shift_risk=0.0,
            mean_shift_risk_upper_bound=0.0,
            boundary_generalized_eigengap=None,
            optimal_subspace_unique=True,
            exact_posterior=True,
        )
        return PosteriorRankDistortionFrontier(
            generalized_eigenvalues=np.empty(0),
            original_rank=0,
            query_dimension=query_dimension,
            numerical_exact_rank=0,
            shared_precision_max_eigenvalue=0.0,
            points=(point,),
        )

    rhs = np.concatenate((u, cross.T), axis=1).reshape(count, 3, rank + query_dimension)
    solved = _array(innovation_operator.solve(rhs), "innovation solve")
    if solved.shape != rhs.shape:
        raise ValueError("innovation solve returned an incorrect shape")
    solved = solved.reshape(3 * count, rank + query_dimension)
    su, sc = solved[:, :rank], solved[:, rank:]

    posterior = _symmetric(prior - cross @ sc, "full query posterior")
    posterior_root = _cholesky(posterior, "full query posterior")
    gram = _symmetric(u.T @ su, "shared precision Gram matrix")
    remainder_metric = _symmetric(np.eye(rank) - gram, "innovation remainder")
    remainder_root = _cholesky(remainder_metric, "innovation remainder")
    response = u.T @ sc

    whitened_response = np.linalg.solve(posterior_root, response.T)
    relevance = _symmetric(
        whitened_response.T @ whitened_response,
        "posterior-normalized latent relevance",
    )
    left = np.linalg.solve(remainder_root, relevance)
    generalized_matrix = _symmetric(
        np.linalg.solve(remainder_root, left.T).T,
        "generalized relevance matrix",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(generalized_matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-10 * scale:
        raise ValueError("generalized posterior relevance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    generalized_vectors = np.linalg.solve(remainder_root.T, eigenvectors)

    relevance_singular = np.linalg.svd(whitened_response.T, compute_uv=False)
    numerical_exact_rank = (
        int(np.count_nonzero(relevance_singular > tolerance * relevance_singular[0]))
        if relevance_singular.size and relevance_singular[0] > 0.0
        else 0
    )
    gamma = max(float(np.linalg.eigvalsh(gram)[-1]), 0.0)
    if not gamma < 1.0:
        raise ValueError("shared precision Gram matrix violates the innovation remainder")

    points: list[PosteriorRankDistortionPoint] = []
    cumulative = np.concatenate(([0.0], np.cumsum(eigenvalues)))
    for retained_rank in range(rank + 1):
        discarded_dimension = rank - retained_rank
        if 0 < discarded_dimension < rank:
            boundary_gap = max(
                float(eigenvalues[discarded_dimension] - eigenvalues[discarded_dimension - 1]),
                0.0,
            )
            optimal_subspace_unique = boundary_gap > tolerance * scale
        else:
            boundary_gap = None
            optimal_subspace_unique = True

        if discarded_dimension:
            discarded = _orthonormal_columns(generalized_vectors[:, :discarded_dimension], rank)
        else:
            discarded = np.empty((rank, 0), dtype=np.float64)
        projection = _orthogonal_complement(discarded, rank)
        compressed = (u @ projection).reshape(count, 3, retained_rank)

        if discarded_dimension:
            metric = _symmetric(
                discarded.T @ remainder_metric @ discarded,
                "discarded remainder metric",
            )
            metric_root = _cholesky(metric, "discarded remainder metric")
            residual = response.T @ discarded
            solved_residual = np.linalg.solve(
                metric_root.T,
                np.linalg.solve(metric_root, residual.T),
            )
            covariance_loss = _symmetric(
                residual @ solved_residual,
                "posterior covariance contraction",
            )
            white_loss = _symmetric(
                _whiten(posterior_root, covariance_loss),
                "normalized posterior covariance contraction",
            )
            audited_trace = max(float(np.trace(white_loss)), 0.0)
            maximum_contraction = max(float(np.linalg.eigvalsh(white_loss)[-1]), 0.0)

            discarded_gram = _symmetric(
                discarded.T @ gram @ discarded,
                "discarded shared precision Gram matrix",
            )
            mean_error_covariance = _symmetric(
                solved_residual.T @ discarded_gram @ solved_residual,
                "posterior mean-shift covariance",
            )
            white_mean_error = _symmetric(
                _whiten(posterior_root, mean_error_covariance),
                "normalized posterior mean-shift covariance",
            )
            mean_shift_risk = max(
                float(np.trace(white_mean_error)) / query_dimension,
                0.0,
            )
        else:
            audited_trace = 0.0
            maximum_contraction = 0.0
            mean_shift_risk = 0.0

        optimal_trace = max(float(cumulative[discarded_dimension]), 0.0)
        audit_scale = max(optimal_trace, audited_trace, 1.0)
        if abs(audited_trace - optimal_trace) > 1e-9 * audit_scale:
            raise ValueError("generalized-eigenvalue distortion identity failed its audit")
        mean_bound = (
            gamma / (1.0 - gamma) * audited_trace / query_dimension if audited_trace else 0.0
        )
        if mean_shift_risk > mean_bound + 1e-10 * max(mean_bound, 1.0):
            raise ValueError("posterior mean-shift bound failed its audit")

        points.append(
            PosteriorRankDistortionPoint(
                retained_rank=retained_rank,
                discarded_dimension=discarded_dimension,
                latent_projection=projection,
                compressed_factor_m=compressed,
                optimal_normalized_covariance_trace_loss=optimal_trace,
                audited_normalized_covariance_trace_loss=audited_trace,
                maximum_normalized_covariance_contraction=maximum_contraction,
                mean_shift_risk=mean_shift_risk,
                mean_shift_risk_upper_bound=mean_bound,
                boundary_generalized_eigengap=boundary_gap,
                optimal_subspace_unique=optimal_subspace_unique,
                exact_posterior=audited_trace <= 1e-10 * max(1.0, optimal_trace),
            )
        )

    return PosteriorRankDistortionFrontier(
        generalized_eigenvalues=eigenvalues,
        original_rank=rank,
        query_dimension=query_dimension,
        numerical_exact_rank=numerical_exact_rank,
        shared_precision_max_eigenvalue=gamma,
        points=tuple(points),
    )
