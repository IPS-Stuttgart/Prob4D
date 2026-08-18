"""Dense and sparse covariance-root Woodbury backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ._observation_gaussian_common import (
    _cholesky_solve,
    _conditional_solve,
    _covariance_root_psd,
    _factor_log_determinant,
    _readonly,
    _strict_cholesky,
)
from .observation_factors import StackedObservationFactors
from .sparse_observation_factors import SparseStackedObservationFactors


@dataclass(frozen=True, slots=True)
class _DenseGaugeWoodbury:
    conditional_factor: np.ndarray
    gauge_jacobian: np.ndarray
    gauge_root: np.ndarray
    latent_factor: np.ndarray
    factorization_backend: str = "dense-gauge-root-woodbury-v1"

    @classmethod
    def build(
        cls,
        stacked: StackedObservationFactors,
        conditional_factor: np.ndarray,
    ) -> _DenseGaugeWoodbury:
        count = len(stacked.world_mean_m)
        jacobian = np.asarray(stacked.gauge_jacobian, dtype=np.float64)
        gauge_dimension = int(jacobian.shape[2]) if jacobian.ndim == 3 else -1
        if jacobian.shape != (count, 3, gauge_dimension):
            raise ValueError("gauge_jacobian must have shape (M, 3, G)")
        prior = np.asarray(stacked.gauge_prior_covariance, dtype=np.float64)
        if prior.shape != (gauge_dimension, gauge_dimension):
            raise ValueError("gauge_prior_covariance has changed shape")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("gauge_jacobian must be finite")
        root = _covariance_root_psd(prior, name="gauge_prior_covariance")
        latent_factor: np.ndarray
        if root.shape[1] == 0:
            latent_factor = np.empty((0, 0), dtype=np.float64)
        else:
            precision_jacobian = _conditional_solve(conditional_factor, jacobian)
            gauge_information = np.einsum(
                "mig,mih->gh",
                jacobian,
                precision_jacobian,
                optimize=True,
            )
            latent_information = np.eye(root.shape[1], dtype=np.float64) + (
                root.T @ gauge_information @ root
            )
            latent_factor = _strict_cholesky(
                latent_information,
                name="latent gauge information",
            )
        return cls(
            conditional_factor=conditional_factor,
            gauge_jacobian=_readonly(jacobian),
            gauge_root=root,
            latent_factor=_readonly(latent_factor),
        )

    @property
    def log_determinant_increment(self) -> float:
        return _factor_log_determinant(self.latent_factor)

    @property
    def factor_storage_nbytes(self) -> int:
        return int(self.gauge_root.nbytes + self.latent_factor.nbytes)

    def apply(self, conditional_precision_response: np.ndarray) -> np.ndarray:
        if self.gauge_root.shape[1] == 0:
            return conditional_precision_response
        gauge_rhs = np.einsum(
            "mig,mir->gr",
            self.gauge_jacobian,
            conditional_precision_response,
            optimize=True,
        )
        latent_rhs = self.gauge_root.T @ gauge_rhs
        latent_solution = _cholesky_solve(self.latent_factor, latent_rhs)
        gauge_solution = self.gauge_root @ latent_solution
        gauge_observation = np.einsum(
            "mig,gr->mir",
            self.gauge_jacobian,
            gauge_solution,
            optimize=True,
        )
        correction = _conditional_solve(self.conditional_factor, gauge_observation)
        return cast(np.ndarray, conditional_precision_response - correction)


@dataclass(frozen=True, slots=True)
class _SparseGaugeWoodbury:
    conditional_factor: np.ndarray
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    gauge_root: np.ndarray
    latent_factor: np.ndarray
    gauge_count: int
    factorization_backend: str = "sparse-gauge-root-woodbury-v1"

    @classmethod
    def build(
        cls,
        stacked: SparseStackedObservationFactors,
        conditional_factor: np.ndarray,
    ) -> _SparseGaugeWoodbury:
        count = len(stacked.world_mean_m)
        gauge_count = len(stacked.gauge_ids)
        jacobian = np.asarray(stacked.local_gauge_jacobian, dtype=np.float64)
        indices = np.asarray(stacked.gauge_indices, dtype=np.int64)
        if jacobian.shape != (count, 3, 7):
            raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
        if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= gauge_count):
            raise ValueError("gauge_indices reference an unknown gauge")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("local_gauge_jacobian must be finite")
        prior = np.asarray(stacked.gauge_prior_covariance, dtype=np.float64)
        gauge_dimension = 7 * gauge_count
        if prior.shape != (gauge_dimension, gauge_dimension):
            raise ValueError("gauge_prior_covariance has changed shape")
        root = _covariance_root_psd(prior, name="gauge_prior_covariance")
        latent_factor: np.ndarray
        if root.shape[1] == 0:
            latent_factor = np.empty((0, 0), dtype=np.float64)
        else:
            precision_jacobian = _conditional_solve(conditional_factor, jacobian)
            row_information = np.einsum(
                "mij,mik->mjk",
                jacobian,
                precision_jacobian,
                optimize=True,
            )
            block_information: np.ndarray = np.zeros(
                (gauge_count, 7, 7),
                dtype=np.float64,
            )
            np.add.at(block_information, indices, row_information)
            root_blocks = root.reshape(gauge_count, 7, root.shape[1])
            information_root = np.einsum(
                "kij,kjr->kir",
                block_information,
                root_blocks,
                optimize=True,
            )
            latent_information = np.eye(root.shape[1], dtype=np.float64) + np.einsum(
                "kir,kis->rs",
                root_blocks,
                information_root,
                optimize=True,
            )
            latent_factor = _strict_cholesky(
                latent_information,
                name="latent gauge information",
            )
        return cls(
            conditional_factor=conditional_factor,
            local_gauge_jacobian=_readonly(jacobian),
            gauge_indices=indices,
            gauge_root=root,
            latent_factor=_readonly(latent_factor),
            gauge_count=gauge_count,
        )

    def __post_init__(self) -> None:
        self.gauge_indices.setflags(write=False)

    @property
    def log_determinant_increment(self) -> float:
        return _factor_log_determinant(self.latent_factor)

    @property
    def factor_storage_nbytes(self) -> int:
        return int(self.gauge_root.nbytes + self.latent_factor.nbytes)

    def apply(self, conditional_precision_response: np.ndarray) -> np.ndarray:
        if self.gauge_root.shape[1] == 0:
            return conditional_precision_response
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
        flat_rhs = gauge_rhs.reshape(7 * self.gauge_count, -1)
        latent_rhs = self.gauge_root.T @ flat_rhs
        latent_solution = _cholesky_solve(self.latent_factor, latent_rhs)
        gauge_solution = (self.gauge_root @ latent_solution).reshape(
            self.gauge_count,
            7,
            -1,
        )
        gauge_observation = np.einsum(
            "mij,mjr->mir",
            self.local_gauge_jacobian,
            gauge_solution[self.gauge_indices],
            optimize=True,
        )
        correction = _conditional_solve(self.conditional_factor, gauge_observation)
        return cast(np.ndarray, conditional_precision_response - correction)
