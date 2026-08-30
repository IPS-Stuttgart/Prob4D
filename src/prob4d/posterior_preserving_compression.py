"""Shared-noise compression for a fixed, jointly Gaussian physical query.

For innovation covariance S = A + U U.T, A positive definite, and query/data
cross covariance C, retain range(U.T @ solve(S, C.T)).  This preserves the
query's conditional mean for every innovation and its conditional covariance.
It does NOT preserve the observation likelihood or an arbitrary next update.

This experimental numerical kernel does not change any provider/export API.
See docs/posterior-preserving-compression.md for the theorem and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class InnovationSolver(Protocol):
    """Covariance-solve surface shared by Prob4D's innovation operators."""

    @property
    def observation_count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def solve(self, value: object) -> Array: ...


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
    scale = float(np.linalg.norm(value, ord="fro"))
    if float(np.linalg.norm(value - value.T, ord="fro")) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (value + value.T)


def _cholesky(value: Array, name: str) -> Array:
    try:
        return np.linalg.cholesky(_symmetric(value, name))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite") from exc


def _cho_solve(root: Array, value: Array) -> Array:
    return np.linalg.solve(root.T, np.linalg.solve(root, value))


def _whiten(root: Array, value: Array) -> Array:
    left = np.linalg.solve(root, value)
    return np.linalg.solve(root, left.T).T


def _fraction(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must lie in [0, 1)")
    if positive and result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly(value: Array) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PosteriorPreservingCompression:
    """A query-bound factor, not a general-purpose replacement observation.

    ``mean_shift_risk`` is E[||P_post^-1/2 (m_reduced-m_full)||^2] / Q
    under the full innovation distribution.  Zero in exact arithmetic.
    The other errors compare the gain and posterior covariance, not trace
    retention in observation space. Arrays are independent read-only copies.
    """

    compressed_factor_m: Array
    latent_projection: Array
    numerical_required_rank: int
    query_dimension: int
    relative_gain_error: float
    relative_covariance_error: float
    mean_shift_risk: float
    exact_fallback: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "compressed_factor_m", _readonly(self.compressed_factor_m))
        object.__setattr__(self, "latent_projection", _readonly(self.latent_projection))

    @property
    def original_rank(self) -> int:
        return int(self.latent_projection.shape[0])

    @property
    def retained_rank(self) -> int:
        return int(self.latent_projection.shape[1])

    def summary(self) -> dict[str, object]:
        return {
            "original_rank": self.original_rank,
            "retained_rank": self.retained_rank,
            "numerical_required_rank": self.numerical_required_rank,
            "query_dimension": self.query_dimension,
            "relative_gain_error": self.relative_gain_error,
            "relative_covariance_error": self.relative_covariance_error,
            "mean_shift_risk": self.mean_shift_risk,
            "exact_fallback": self.exact_fallback,
            "reason": self.reason,
            "claim_boundary": "fixed-local-Gaussian-query-only; not observation evidence",
        }


def compress_shared_factor_for_posterior(
    low_rank_factor_m: object,
    *,
    prior_query_covariance: object,
    query_observation_cross_covariance: object,
    innovation_operator: InnovationSolver,
    maximum_rank: int | None = None,
    rank_relative_tolerance: float = 1e-12,
    parity_relative_tolerance: float = 1e-9,
) -> PosteriorPreservingCompression:
    """Return a minimum-range shared factor for one frozen Gaussian query.

    Shapes: U=(N,3,R), prior query covariance=(Q,Q), C=(Q,N,3) or
    (Q,3N). The solver must represent the FULL innovation covariance S,
    including U U.T and the physical prior's observation covariance.
    The same prior, cross covariance, row order, linearization and fixed
    robust weights must be used by the consumer. No observed outcome is used
    in constructing the projection.

    In exact arithmetic the required rank is rank(U.T S^-1 C.T) <= Q.
    SVD selects a numerical range; a downdate audit checks the actual gain,
    relative posterior covariance loss and expected normalized mean error.
    Numerical failures cause ValueError, never a ridge or silent clipping.
    If no reduced rank meets the parity limits and rank cap, the exact
    original factor is returned. The cap never forces information loss.

    The implementation requires nondegenerate query prior and posterior
    covariances for relative error auditing; deterministic/duplicated query
    coordinates must first be reduced to an independent query basis.
    """
    factor = _array(low_rank_factor_m, "low_rank_factor_m")
    if factor.ndim != 3 or factor.shape[0] < 1 or factor.shape[1] != 3:
        raise ValueError("low_rank_factor_m must have shape (N, 3, R), N > 0")
    count, _, rank = factor.shape
    if (
        innovation_operator.observation_count != count
        or innovation_operator.dimension != 3 * count
    ):
        raise ValueError("innovation_operator dimensions do not match the factor")
    if maximum_rank is not None:
        if isinstance(maximum_rank, (bool, np.bool_)) or not isinstance(
            maximum_rank, (int, np.integer)
        ):
            raise TypeError("maximum_rank must be a nonnegative integer or None")
        if maximum_rank < 0:
            raise ValueError("maximum_rank must be nonnegative")
    rank_tol = _fraction(rank_relative_tolerance, "rank_relative_tolerance", positive=False)
    parity_tol = _fraction(parity_relative_tolerance, "parity_relative_tolerance", positive=True)
    prior = _array(prior_query_covariance, "prior_query_covariance")
    if prior.ndim != 2 or prior.shape[0] < 1 or prior.shape[0] != prior.shape[1]:
        raise ValueError("prior_query_covariance must be a nonempty square matrix")
    _cholesky(prior, "prior_query_covariance")
    qdim = prior.shape[0]
    cross = _array(query_observation_cross_covariance, "query_observation_cross_covariance")
    if cross.shape not in ((qdim, count, 3), (qdim, 3 * count)):
        raise ValueError("query_observation_cross_covariance must have shape (Q,N,3) or (Q,3N)")
    cross = cross.reshape(qdim, 3 * count)
    u = factor.reshape(3 * count, rank)
    rhs = np.concatenate((u, cross.T), axis=1).reshape(count, 3, rank + qdim)
    solved = _array(innovation_operator.solve(rhs), "innovation solve")
    if solved.shape != rhs.shape:
        raise ValueError("innovation solve returned an incorrect shape")
    solved = solved.reshape(3 * count, rank + qdim)
    su, sc = solved[:, :rank], solved[:, rank:]
    gain = sc.T
    posterior = _symmetric(prior - cross @ sc, "full query posterior")
    post_root = _cholesky(posterior, "full query posterior")
    gram = _symmetric(u.T @ su, "shared precision Gram matrix")
    # Equivalent to A = S - U U.T being positive definite, for an SPD S.
    _cholesky(np.eye(rank) - gram, "innovation remainder")
    response = u.T @ sc
    normalized_response = np.linalg.solve(post_root, response.T).T
    left, singular, _ = np.linalg.svd(normalized_response, full_matrices=True)
    required = int(np.count_nonzero(singular > rank_tol * singular[0])) if singular.size else 0
    white_gain = np.linalg.solve(post_root, gain)
    gain_norm = float(np.linalg.norm(white_gain, ord="fro"))

    def fallback(reason: str) -> PosteriorPreservingCompression:
        return PosteriorPreservingCompression(
            factor, np.eye(rank), required, qdim, 0.0, 0.0, 0.0, True, reason
        )

    if rank == 0:
        return PosteriorPreservingCompression(
            factor, np.empty((0, 0)), 0, qdim, 0.0, 0.0, 0.0, False,
            "no-shared-factor",
        )
    limit = min(rank - 1, int(maximum_rank) if maximum_rank is not None else rank - 1)
    for retained in range(required, limit + 1):
        complement = left[:, retained:]
        discarded_gram = complement.T @ gram @ complement
        # The input Gram is already symmetric; cancellation near its nullspace
        # requires symmetrizing the product, not a relative-to-zero rejection.
        discarded_gram = 0.5 * (discarded_gram + discarded_gram.T)
        core = _cholesky(np.eye(rank - retained) - discarded_gram, "compression downdate")
        residual = response.T @ complement
        solved_residual = _cho_solve(core, residual.T)
        covariance_loss = residual @ solved_residual
        gain_change = solved_residual.T @ (su @ complement).T
        mean_error_covariance = solved_residual.T @ discarded_gram @ solved_residual
        white_loss = _whiten(post_root, covariance_loss)
        white_mean_error = _whiten(post_root, mean_error_covariance)
        covariance_error = max(
            float(np.linalg.eigvalsh(0.5 * (white_loss + white_loss.T))[-1]), 0.0
        )
        mean_risk = max(float(np.trace(white_mean_error)) / qdim, 0.0)
        gain_change_norm = float(np.linalg.norm(np.linalg.solve(post_root, gain_change), ord="fro"))
        gain_error = gain_change_norm / gain_norm if gain_norm else gain_change_norm
        if not np.all(np.isfinite([covariance_error, mean_risk, gain_error])):
            raise ValueError("compression audit produced nonfinite errors")
        if (
            gain_error <= parity_tol
            and covariance_error <= parity_tol
            and mean_risk <= parity_tol**2
        ):
            projection = left[:, :retained].copy()
            # Fix column signs; repeated-subspace bases may still rotate by LAPACK version.
            for column in range(retained):
                pivot = int(np.argmax(np.abs(projection[:, column])))
                if projection[pivot, column] < 0.0:
                    projection[:, column] *= -1.0
            return PosteriorPreservingCompression(
                (u @ projection).reshape(count, 3, retained), projection,
                required, qdim, gain_error, covariance_error, mean_risk, False,
                "posterior-parity-validated",
            )
    reason = (
        "full-rank-required" if required == rank
        else "no-parity-preserving-reduction-within-cap"
    )
    return fallback(reason)
