"""Public operations mixed into the immutable gauge-tree prior contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._gauge_tree_algebra import (
    covariance_action,
    diagonal_covariance_blocks,
    information_action,
    information_quadratic,
    innovation_coordinates,
    innovation_covariance_blocks,
    log_determinant_covariance,
    sample,
    standardized_innovations,
)
from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    FloatArray,
    IntArray,
    joint_covariance_sha256,
)
from ._gauge_tree_factorization import verify_dense
from ._gauge_tree_observation import (
    marginal_observation_covariance_action,
    observation_covariance_action,
    row_marginal_covariance,
)
from ._gauge_tree_selection import cross_covariance, selected_covariance


class GaugeTreePriorMethods:
    """Matrix-free operations shared by the versioned gauge-tree prior."""

    gauge_ids: tuple[str, ...]
    parent_indices: IntArray
    transition_matrices: FloatArray
    innovation_scale_tril: FloatArray
    source_joint_covariance_sha256: str | None

    @property
    def gauge_count(self) -> int:
        return len(self.gauge_ids)

    @property
    def parent_gauge_ids(self) -> tuple[str | None, ...]:
        return (None,) + tuple(
            self.gauge_ids[int(self.parent_indices[index])] for index in range(1, self.gauge_count)
        )

    @property
    def dimension(self) -> int:
        return GAUGE_DIMENSION * self.gauge_count

    @property
    def factor_storage_nbytes(self) -> int:
        return int(
            self.parent_indices.nbytes
            + self.transition_matrices.nbytes
            + self.innovation_scale_tril.nbytes
        )

    @property
    def dense_covariance_nbytes(self) -> int:
        return int(self.dimension**2 * np.dtype(np.float64).itemsize)

    @property
    def storage_ratio_to_dense(self) -> float:
        return self.factor_storage_nbytes / self.dense_covariance_nbytes

    def covariance_action(self, value: Any) -> FloatArray:
        return covariance_action(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            value,
        )

    def solve_information(self, value: Any) -> FloatArray:
        return self.covariance_action(value)

    def innovation_coordinates(self, value: Any) -> FloatArray:
        return innovation_coordinates(self.parent_indices, self.transition_matrices, value)

    def standardized_innovations(self, value: Any) -> FloatArray:
        return standardized_innovations(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            value,
        )

    def information_action(self, value: Any) -> FloatArray:
        return information_action(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            value,
        )

    def solve_covariance(self, value: Any) -> FloatArray:
        return self.information_action(value)

    def information_quadratic(self, value: Any) -> float:
        return information_quadratic(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            value,
        )

    def log_determinant_covariance(self) -> float:
        return log_determinant_covariance(self.innovation_scale_tril)

    def innovation_covariance_blocks(self) -> FloatArray:
        return innovation_covariance_blocks(self.innovation_scale_tril)

    def diagonal_covariance_blocks(self) -> FloatArray:
        return diagonal_covariance_blocks(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
        )

    def cross_covariance(
        self,
        left_gauge_ids: Sequence[str],
        right_gauge_ids: Sequence[str],
    ) -> FloatArray:
        return cross_covariance(
            self.gauge_ids,
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            left_gauge_ids,
            right_gauge_ids,
        )

    def selected_covariance(self, gauge_ids: Sequence[str]) -> FloatArray:
        return selected_covariance(
            self.gauge_ids,
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            gauge_ids,
        )

    def materialize_dense_covariance(self, *, maximum_gauges: int = 128) -> FloatArray:
        if isinstance(maximum_gauges, bool) or not isinstance(
            maximum_gauges,
            (int, np.integer),
        ):
            raise TypeError("maximum_gauges must be a genuine integer")
        if maximum_gauges < 1:
            raise ValueError("maximum_gauges must be positive")
        if self.gauge_count > int(maximum_gauges):
            raise ValueError(
                f"dense covariance materialization is limited to {maximum_gauges} gauges"
            )
        return self.selected_covariance(self.gauge_ids)

    def verify_dense_covariance(
        self,
        joint_covariance: Any,
        *,
        atol: float = 1e-10,
        rtol: float = 1e-8,
        require_source_digest: bool = False,
    ) -> None:
        if require_source_digest:
            if self.source_joint_covariance_sha256 is None:
                raise ValueError("the gauge-tree prior has no source covariance digest")
            if joint_covariance_sha256(joint_covariance) != self.source_joint_covariance_sha256:
                raise ValueError("joint_covariance does not match the bound source digest")
        verify_dense(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            joint_covariance,
            atol=atol,
            rtol=rtol,
        )

    def row_marginal_covariance(
        self,
        local_gauge_jacobian: Any,
        gauge_indices: Any,
    ) -> FloatArray:
        return row_marginal_covariance(
            self.diagonal_covariance_blocks(),
            local_gauge_jacobian,
            gauge_indices,
        )

    def observation_covariance_action(
        self,
        local_gauge_jacobian: Any,
        gauge_indices: Any,
        value: Any,
    ) -> FloatArray:
        return observation_covariance_action(
            self.covariance_action,
            self.gauge_count,
            local_gauge_jacobian,
            gauge_indices,
            value,
        )

    def marginal_observation_covariance_action(
        self,
        local_gauge_jacobian: Any,
        gauge_indices: Any,
        conditional_covariance: Any,
        value: Any,
    ) -> FloatArray:
        return marginal_observation_covariance_action(
            self.covariance_action,
            self.gauge_count,
            local_gauge_jacobian,
            gauge_indices,
            conditional_covariance,
            value,
        )

    def sample(self, *, seed: int, sample_count: int = 1) -> FloatArray:
        return sample(
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            seed=seed,
            sample_count=sample_count,
        )
