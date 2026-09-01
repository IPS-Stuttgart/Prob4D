"""Factorized marginal-preserving Gaussian dependence.

The covariance family implemented here separates block-local uncertainty from a
shared low-rank latent effect while preserving every marginal covariance block.
It is useful for correlated point/query predictions whose means and marginal
uncertainties are already fixed.  The strength is supplied by the caller; this
module does not fit it or attach a calibration claim to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _numeric_array(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _strength(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("strength must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("strength must be finite and lie in [0, 1]")
    return result


@dataclass(frozen=True)
class BlockSharedGaussianCovariance:
    """Block-local plus shared-low-rank covariance with fixed marginals.

    Let ``M_i`` denote the covariance block for output group ``i`` and let
    ``F_i`` be that group's rows of a shared factor.  Construction requires

    ``F_i @ F_i.T == M_i``

    up to the declared numerical tolerance.  For strength ``alpha`` the joint
    covariance is

    ``Sigma(alpha) = (1-alpha) blockdiag(M_i) + alpha F F.T``.

    Consequently, every diagonal block equals ``M_i`` for every alpha.  Only
    cross-group dependence changes.  For ``alpha < 1``, solves and log
    determinants use block solves and a rank-sized Woodbury core; no dense
    joint covariance is formed.  ``alpha == 1`` remains available for moments
    and sampling, but precision operations may be singular and are rejected.
    """

    marginal_blocks: FloatArray
    shared_factors: FloatArray
    strength: float
    matching_rtol: float = 1.0e-9
    matching_atol: float = 1.0e-12

    def __post_init__(self) -> None:
        blocks = _numeric_array(self.marginal_blocks, name="marginal_blocks")
        factors = _numeric_array(self.shared_factors, name="shared_factors")
        alpha = _strength(self.strength)
        if blocks.ndim != 3 or blocks.shape[0] == 0 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError("marginal_blocks must have nonempty shape (N, D, D)")
        if factors.ndim != 3 or factors.shape[:2] != blocks.shape[:2] or factors.shape[2] == 0:
            raise ValueError("shared_factors must have shape (N, D, R) with R > 0")
        if not math.isfinite(self.matching_rtol) or self.matching_rtol < 0.0:
            raise ValueError("matching_rtol must be finite and nonnegative")
        if not math.isfinite(self.matching_atol) or self.matching_atol < 0.0:
            raise ValueError("matching_atol must be finite and nonnegative")

        scale = max(1.0, float(np.max(np.abs(blocks))))
        tolerance = max(float(self.matching_atol), float(self.matching_rtol) * scale)
        if not np.allclose(blocks, np.swapaxes(blocks, 1, 2), rtol=0.0, atol=tolerance):
            raise ValueError("marginal covariance blocks must be symmetric")
        blocks = 0.5 * (blocks + np.swapaxes(blocks, 1, 2))
        eigenvalues = np.linalg.eigvalsh(blocks)
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("marginal covariance blocks must be positive semidefinite")
        if alpha < 1.0 and float(np.min(eigenvalues)) <= 0.0:
            raise ValueError("marginal covariance blocks must be positive definite when strength < 1")

        shared_marginals = np.einsum("idr,ier->ide", factors, factors)
        if not np.allclose(
            shared_marginals,
            blocks,
            rtol=float(self.matching_rtol),
            atol=float(self.matching_atol),
        ):
            maximum = float(np.max(np.abs(shared_marginals - blocks)))
            raise ValueError(
                "shared factors must reproduce every marginal covariance block; "
                f"maximum absolute mismatch is {maximum:.6g}"
            )

        blocks.setflags(write=False)
        factors.setflags(write=False)
        object.__setattr__(self, "marginal_blocks", blocks)
        object.__setattr__(self, "shared_factors", factors)
        object.__setattr__(self, "strength", alpha)

    @property
    def group_count(self) -> int:
        return int(self.marginal_blocks.shape[0])

    @property
    def block_dimension(self) -> int:
        return int(self.marginal_blocks.shape[1])

    @property
    def latent_rank(self) -> int:
        return int(self.shared_factors.shape[2])

    @property
    def dimension(self) -> int:
        return self.group_count * self.block_dimension

    @property
    def storage_bytes(self) -> int:
        """Bytes occupied by the two numerical arrays retained by the model."""
        return int(self.marginal_blocks.nbytes + self.shared_factors.nbytes)

    @property
    def dense_storage_bytes(self) -> int:
        """Bytes required by one float64 dense joint covariance."""
        return self.dimension * self.dimension * np.dtype(np.float64).itemsize

    @property
    def storage_reduction_factor(self) -> float:
        return self.dense_storage_bytes / self.storage_bytes

    @property
    def marginal_covariances(self) -> FloatArray:
        result = np.array(self.marginal_blocks, copy=True)
        result.setflags(write=False)
        return result

    def cross_covariance(self, first_group: int, second_group: int) -> FloatArray:
        """Return one cross-group covariance block.

        The diagonal case returns the preserved marginal exactly rather than a
        numerically reconstructed copy.
        """
        for value, name in ((first_group, "first_group"), (second_group, "second_group")):
            if isinstance(value, (bool, np.bool_)) or not 0 <= int(value) < self.group_count:
                raise IndexError(f"{name} is outside the group range")
        first = int(first_group)
        second = int(second_group)
        if first == second:
            result = np.array(self.marginal_blocks[first], copy=True)
        else:
            result = self.strength * self.shared_factors[first] @ self.shared_factors[second].T
        result.setflags(write=False)
        return result

    def dense_covariance(self) -> FloatArray:
        """Materialize the joint covariance for validation or small problems."""
        local = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        width = self.block_dimension
        for index, block in enumerate(self.marginal_blocks):
            start = index * width
            local[start : start + width, start : start + width] = block
        factor = self.shared_factors.reshape(self.dimension, self.latent_rank)
        result = (1.0 - self.strength) * local + self.strength * (factor @ factor.T)
        result = 0.5 * (result + result.T)
        result.setflags(write=False)
        return result

    def _require_precision_form(self) -> float:
        local_weight = 1.0 - self.strength
        if local_weight <= 0.0:
            raise np.linalg.LinAlgError(
                "strength=1 has no positive block-local component; use an explicit "
                "singular-Gaussian treatment or a strength below one"
            )
        return local_weight

    def _solve_blocks(self, right_hand_side: FloatArray) -> FloatArray:
        return np.linalg.solve(self.marginal_blocks, right_hand_side)

    def solve(self, right_hand_side: ArrayLike) -> FloatArray:
        """Apply the inverse covariance using the Woodbury identity.

        ``right_hand_side`` may have shape ``(dimension,)`` or
        ``(dimension, K)``.  The returned shape matches the input shape.
        """
        local_weight = self._require_precision_form()
        right = _numeric_array(right_hand_side, name="right_hand_side")
        vector = right.ndim == 1
        if vector:
            if right.shape != (self.dimension,):
                raise ValueError("a vector right_hand_side must match the covariance dimension")
            right = right[:, None]
        elif right.ndim != 2 or right.shape[0] != self.dimension:
            raise ValueError("right_hand_side must have shape (dimension,) or (dimension, K)")

        columns = right.shape[1]
        grouped = right.reshape(self.group_count, self.block_dimension, columns)
        local_inverse_right = self._solve_blocks(grouped) / local_weight

        factor = self.shared_factors
        local_inverse_factor = self._solve_blocks(factor) / local_weight
        core = np.eye(self.latent_rank) + self.strength * np.einsum(
            "idr,ids->rs", factor, local_inverse_factor
        )
        projected = math.sqrt(self.strength) * np.einsum(
            "idr,idk->rk", factor, local_inverse_right
        )
        correction_coefficients = np.linalg.solve(core, projected)
        correction = math.sqrt(self.strength) * np.einsum(
            "idr,rk->idk", local_inverse_factor, correction_coefficients
        )
        result = (local_inverse_right - correction).reshape(self.dimension, columns)
        if vector:
            result = result[:, 0]
        result.setflags(write=False)
        return result

    def log_determinant(self) -> float:
        """Return ``log(det(Sigma))`` without materializing ``Sigma``."""
        local_weight = self._require_precision_form()
        signs, block_log_determinants = np.linalg.slogdet(self.marginal_blocks)
        if np.any(signs <= 0.0):
            raise np.linalg.LinAlgError("marginal covariance block is not positive definite")
        local_inverse_factor = self._solve_blocks(self.shared_factors) / local_weight
        core = np.eye(self.latent_rank) + self.strength * np.einsum(
            "idr,ids->rs", self.shared_factors, local_inverse_factor
        )
        core_sign, core_log_determinant = np.linalg.slogdet(core)
        if core_sign <= 0.0:
            raise np.linalg.LinAlgError("Woodbury core is not positive definite")
        return float(
            np.sum(block_log_determinants)
            + self.dimension * math.log(local_weight)
            + core_log_determinant
        )

    def quadratic_form(self, residual: ArrayLike) -> float:
        vector = _numeric_array(residual, name="residual")
        if vector.shape != (self.dimension,):
            raise ValueError("residual must match the covariance dimension")
        solved = self.solve(vector)
        value = float(vector @ solved)
        tolerance = 1.0e-10 * max(1.0, float(vector @ vector))
        if value < -tolerance:
            raise RuntimeError("computed a negative covariance quadratic form")
        return max(value, 0.0)

    def gaussian_nll(self, residual: ArrayLike, *, normalized: bool = False) -> float:
        """Gaussian negative log likelihood for a zero-mean residual."""
        value = 0.5 * (
            self.dimension * math.log(2.0 * math.pi)
            + self.log_determinant()
            + self.quadratic_form(residual)
        )
        return value / self.dimension if normalized else value

    def sample(
        self,
        random_generator: np.random.Generator,
        sample_count: int,
    ) -> FloatArray:
        """Draw zero-mean samples with shape ``(sample_count, N, D)``."""
        if not isinstance(random_generator, np.random.Generator):
            raise TypeError("random_generator must be numpy.random.Generator")
        if isinstance(sample_count, (bool, np.bool_)) or int(sample_count) != sample_count:
            raise ValueError("sample_count must be a positive integer")
        count = int(sample_count)
        if count <= 0:
            raise ValueError("sample_count must be a positive integer")

        local_noise = random_generator.standard_normal(
            (count, self.group_count, self.block_dimension)
        )
        shared_noise = random_generator.standard_normal((count, self.latent_rank))
        local_roots = np.linalg.cholesky(self.marginal_blocks)
        local = np.einsum("idj,sij->sid", local_roots, local_noise)
        shared = np.einsum("idr,sr->sid", self.shared_factors, shared_noise)
        result = math.sqrt(1.0 - self.strength) * local + math.sqrt(self.strength) * shared
        result.setflags(write=False)
        return result
