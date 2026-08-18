"""Causal-tree block-information backend for structured observation Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ._observation_gaussian_common import (
    _cholesky_solve,
    _conditional_solve,
    _factor_log_determinant,
    _readonly,
    _strict_cholesky,
)
from .tree_sparse_observation_factors import TreeSparseStackedObservationFactors


@dataclass(frozen=True, slots=True)
class _TreeGaugeInformation:
    conditional_factor: np.ndarray
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    parent_indices: np.ndarray
    off_diagonal_blocks: np.ndarray
    eliminated_factors: tuple[np.ndarray, ...]
    prior_log_determinant: float
    factorization_backend: str = "tree-block-information-v1"

    @classmethod
    def build(
        cls,
        stacked: TreeSparseStackedObservationFactors,
        conditional_factor: np.ndarray,
    ) -> _TreeGaugeInformation:
        count = len(stacked.world_mean_m)
        prior = stacked.gauge_tree_prior
        gauge_count = prior.gauge_count
        jacobian = np.asarray(stacked.local_gauge_jacobian, dtype=np.float64)
        indices = np.asarray(stacked.gauge_indices, dtype=np.int64)
        if jacobian.shape != (count, 3, 7):
            raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
        if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= gauge_count):
            raise ValueError("gauge_indices reference an unknown gauge")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("local_gauge_jacobian must be finite")

        precision_jacobian = _conditional_solve(conditional_factor, jacobian)
        row_information = np.einsum(
            "mij,mik->mjk",
            jacobian,
            precision_jacobian,
            optimize=True,
        )
        diagonal = np.zeros((gauge_count, 7, 7), dtype=np.float64)
        np.add.at(diagonal, indices, row_information)

        parents = np.asarray(prior.parent_indices, dtype=np.int64)
        transitions = np.asarray(prior.transition_matrices, dtype=np.float64)
        innovation_factors = np.asarray(prior.innovation_scale_tril, dtype=np.float64)
        off_diagonal = np.zeros_like(diagonal)
        identity = np.eye(7, dtype=np.float64)
        for gauge_index in range(gauge_count):
            innovation_precision = _cholesky_solve(
                innovation_factors[gauge_index],
                identity,
            )
            diagonal[gauge_index] += innovation_precision
            if gauge_index == 0:
                continue
            parent = int(parents[gauge_index])
            transition = transitions[gauge_index]
            off_diagonal[gauge_index] = -(innovation_precision @ transition)
            diagonal[parent] += transition.T @ innovation_precision @ transition

        factors: list[np.ndarray | None] = [None] * gauge_count
        for gauge_index in range(gauge_count - 1, 0, -1):
            factor = _strict_cholesky(
                diagonal[gauge_index],
                name=f"eliminated gauge information block {gauge_index}",
            )
            factors[gauge_index] = _readonly(factor)
            parent = int(parents[gauge_index])
            solved_edge = _cholesky_solve(factor, off_diagonal[gauge_index])
            diagonal[parent] -= off_diagonal[gauge_index].T @ solved_edge
            diagonal[parent] = 0.5 * (diagonal[parent] + diagonal[parent].T)
        root_factor = _strict_cholesky(
            diagonal[0],
            name="eliminated root gauge information block",
        )
        factors[0] = _readonly(root_factor)
        complete_factors = tuple(cast(np.ndarray, factor) for factor in factors)
        return cls(
            conditional_factor=conditional_factor,
            local_gauge_jacobian=_readonly(jacobian),
            gauge_indices=indices,
            parent_indices=parents,
            off_diagonal_blocks=_readonly(off_diagonal),
            eliminated_factors=complete_factors,
            prior_log_determinant=float(prior.log_determinant_covariance()),
        )

    def __post_init__(self) -> None:
        self.gauge_indices.setflags(write=False)
        self.parent_indices.setflags(write=False)

    @property
    def gauge_count(self) -> int:
        return len(self.eliminated_factors)

    @property
    def log_determinant_increment(self) -> float:
        information_log_determinant = sum(
            _factor_log_determinant(factor) for factor in self.eliminated_factors
        )
        return self.prior_log_determinant + information_log_determinant

    @property
    def factor_storage_nbytes(self) -> int:
        return int(
            self.off_diagonal_blocks.nbytes
            + sum(factor.nbytes for factor in self.eliminated_factors)
        )

    def _solve_gauge_information(self, gauge_rhs: np.ndarray) -> np.ndarray:
        work = gauge_rhs.copy()
        for gauge_index in range(self.gauge_count - 1, 0, -1):
            parent = int(self.parent_indices[gauge_index])
            solved = _cholesky_solve(
                self.eliminated_factors[gauge_index],
                work[gauge_index],
            )
            work[parent] -= self.off_diagonal_blocks[gauge_index].T @ solved

        solution = np.empty_like(work)
        solution[0] = _cholesky_solve(self.eliminated_factors[0], work[0])
        for gauge_index in range(1, self.gauge_count):
            parent = int(self.parent_indices[gauge_index])
            rhs = work[gauge_index] - (
                self.off_diagonal_blocks[gauge_index] @ solution[parent]
            )
            solution[gauge_index] = _cholesky_solve(
                self.eliminated_factors[gauge_index],
                rhs,
            )
        return cast(np.ndarray, solution)

    def apply(self, conditional_precision_response: np.ndarray) -> np.ndarray:
        row_rhs = np.einsum(
            "mij,mir->mjr",
            self.local_gauge_jacobian,
            conditional_precision_response,
            optimize=True,
        )
        gauge_rhs = np.zeros(
            (self.gauge_count, 7, conditional_precision_response.shape[2]),
            dtype=np.float64,
        )
        np.add.at(gauge_rhs, self.gauge_indices, row_rhs)
        gauge_solution = self._solve_gauge_information(gauge_rhs)
        gauge_observation = np.einsum(
            "mij,mjr->mir",
            self.local_gauge_jacobian,
            gauge_solution[self.gauge_indices],
            optimize=True,
        )
        correction = _conditional_solve(self.conditional_factor, gauge_observation)
        return cast(np.ndarray, conditional_precision_response - correction)
