"""Tree-backed sparse stacking for explicit-gauge observation-factor updates.

The historical sparse stack removes the ``M x 3 x 7K`` row expansion but retains
one dense ``7K x 7K`` gauge covariance. This module binds the same immutable row
representation to :class:`GaugeTreeSquareRootPriorV1` after exact dense parity
verification, then drops the retained dense covariance from the returned object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .gauge_tree_prior import GaugeTreeSquareRootPriorV1
from .observation_factors import ObservationFactorBundle
from .sparse_observation_factors import (
    SparseStackedObservationFactors,
    stack_sparse_observation_factors,
)

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]

_ROW_ARRAY_FIELDS = (
    "world_mean_m",
    "conditional_world_covariance_m2",
    "marginal_world_covariance_m2",
    "local_gauge_jacobian",
    "gauge_indices",
    "association_probability",
    "prior_reliability",
    "prior_nominal_probability",
    "composite_weight",
    "point_ids",
    "frame_indices",
)
_TRANSFER_FIELDS = (
    *_ROW_ARRAY_FIELDS,
    "view_ids",
    "factor_ids",
    "correlation_group_ids",
    "causal_frame_stop",
)


def _require_marginal_parity(
    stacked: SparseStackedObservationFactors,
    gauge_tree_prior: GaugeTreeSquareRootPriorV1,
) -> None:
    with np.errstate(invalid="ignore", over="ignore"):
        gauge_marginal = gauge_tree_prior.row_marginal_covariance(
            stacked.local_gauge_jacobian,
            stacked.gauge_indices,
        )
        expected = stacked.conditional_world_covariance_m2 + gauge_marginal
        expected = 0.5 * (expected + expected.swapaxes(1, 2))
    if not np.allclose(
        stacked.marginal_world_covariance_m2,
        expected,
        atol=1e-12,
        rtol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(
            "marginal_world_covariance_m2 does not match the tree gauge prior"
        )


def _require_immutable_rows(stacked: SparseStackedObservationFactors) -> None:
    for name in _ROW_ARRAY_FIELDS:
        if np.asarray(getattr(stacked, name)).flags.writeable:
            raise ValueError(f"stacked {name} must be immutable")


@dataclass(frozen=True, slots=True, init=False)
class TreeSparseStackedObservationFactors:
    """Factory-built sparse rows backed by an ``O(K)`` causal gauge-tree prior.

    Construct instances through :func:`bind_gauge_tree_prior` or
    :func:`stack_tree_sparse_observation_factors`. The factory verifies the tree
    against the complete dense schema-v4 prior and the retained row marginals
    before transferring the already immutable row arrays.
    """

    world_mean_m: FloatArray
    conditional_world_covariance_m2: FloatArray
    marginal_world_covariance_m2: FloatArray
    local_gauge_jacobian: FloatArray
    gauge_indices: IntArray
    gauge_tree_prior: GaugeTreeSquareRootPriorV1
    association_probability: FloatArray
    prior_reliability: FloatArray
    prior_nominal_probability: FloatArray
    composite_weight: FloatArray
    point_ids: IntArray
    frame_indices: IntArray
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    causal_frame_stop: int

    def __init__(self) -> None:
        raise TypeError(
            "use bind_gauge_tree_prior or stack_tree_sparse_observation_factors"
        )

    @classmethod
    def _from_verified_sparse_stack(
        cls,
        stacked: SparseStackedObservationFactors,
        gauge_tree_prior: GaugeTreeSquareRootPriorV1,
    ) -> TreeSparseStackedObservationFactors:
        result = object.__new__(cls)
        for name in _TRANSFER_FIELDS:
            object.__setattr__(result, name, getattr(stacked, name))
        object.__setattr__(result, "gauge_tree_prior", gauge_tree_prior)
        return result

    @property
    def observation_count(self) -> int:
        return len(self.world_mean_m)

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        return self.gauge_tree_prior.gauge_ids

    @property
    def gauge_count(self) -> int:
        return self.gauge_tree_prior.gauge_count

    @property
    def dense_gauge_dimension(self) -> int:
        return self.gauge_tree_prior.dimension

    @property
    def sparse_gauge_design_nbytes(self) -> int:
        return int(self.local_gauge_jacobian.nbytes + self.gauge_indices.nbytes)

    @property
    def dense_gauge_design_nbytes(self) -> int:
        return int(
            self.observation_count
            * 3
            * self.dense_gauge_dimension
            * np.dtype(np.float64).itemsize
        )

    @property
    def gauge_prior_storage_nbytes(self) -> int:
        return self.gauge_tree_prior.factor_storage_nbytes

    @property
    def dense_gauge_prior_nbytes(self) -> int:
        return self.gauge_tree_prior.dense_covariance_nbytes

    @property
    def gauge_prior_storage_ratio_to_dense(self) -> float:
        return self.gauge_tree_prior.storage_ratio_to_dense

    def dense_gauge_jacobian(self) -> FloatArray:
        """Materialize the historical dense row design as a compatibility copy."""

        result: FloatArray = np.zeros(
            (self.observation_count, 3, self.dense_gauge_dimension),
            dtype=np.float64,
        )
        for gauge_index in range(self.gauge_count):
            selected = self.gauge_indices == gauge_index
            start = 7 * gauge_index
            result[selected, :, start : start + 7] = self.local_gauge_jacobian[selected]
        return result

    def apply_gauge_delta(self, gauge_delta: FloatArray) -> FloatArray:
        """Apply one flat or block-shaped gauge perturbation to every row."""

        delta = np.asarray(gauge_delta, dtype=np.float64)
        blocks: FloatArray
        if delta.shape == (self.dense_gauge_dimension,):
            blocks = delta.reshape(self.gauge_count, 7)
        elif delta.shape == (self.gauge_count, 7):
            blocks = delta
        else:
            raise ValueError(
                "gauge_delta must have shape "
                f"({self.dense_gauge_dimension},) or ({self.gauge_count}, 7)"
            )
        if not np.all(np.isfinite(blocks)):
            raise ValueError("gauge_delta must be finite")
        return np.einsum(
            "nij,nj->ni",
            self.local_gauge_jacobian,
            blocks[self.gauge_indices],
            optimize=True,
        )

    def gauge_marginal_covariance_m2(self) -> FloatArray:
        """Return each row's ``J Sigma_gg J^T`` contribution."""

        return self.gauge_tree_prior.row_marginal_covariance(
            self.local_gauge_jacobian,
            self.gauge_indices,
        )

    def gauge_covariance_action(self, value: Any) -> FloatArray:
        return self.gauge_tree_prior.covariance_action(value)

    def gauge_information_action(self, value: Any) -> FloatArray:
        return self.gauge_tree_prior.information_action(value)

    def observation_gauge_covariance_action(self, value: Any) -> FloatArray:
        return self.gauge_tree_prior.observation_covariance_action(
            self.local_gauge_jacobian,
            self.gauge_indices,
            value,
        )

    def marginal_observation_covariance_action(self, value: Any) -> FloatArray:
        return self.gauge_tree_prior.marginal_observation_covariance_action(
            self.local_gauge_jacobian,
            self.gauge_indices,
            self.conditional_world_covariance_m2,
            value,
        )

    def materialize_dense_gauge_prior(
        self,
        *,
        maximum_gauges: int = 128,
    ) -> FloatArray:
        """Explicitly materialize the dense prior behind the existing size guard."""

        return self.gauge_tree_prior.materialize_dense_covariance(
            maximum_gauges=maximum_gauges,
        )


def bind_gauge_tree_prior(
    stacked: SparseStackedObservationFactors,
    gauge_tree_prior: GaugeTreeSquareRootPriorV1,
) -> TreeSparseStackedObservationFactors:
    """Verify a sparse tree against a dense stack and release the dense prior.

    The returned object reuses the stack's already immutable row arrays. It does
    not retain ``stacked.gauge_prior_covariance``.
    """

    if not isinstance(stacked, SparseStackedObservationFactors):
        raise TypeError("stacked must be a SparseStackedObservationFactors")
    if not isinstance(gauge_tree_prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("gauge_tree_prior must be a GaugeTreeSquareRootPriorV1")
    if gauge_tree_prior.gauge_ids != stacked.gauge_ids:
        raise ValueError("gauge-tree prior order does not match the sparse stack")
    gauge_tree_prior.verify_dense_covariance(
        stacked.gauge_prior_covariance,
        require_source_digest=(
            gauge_tree_prior.source_joint_covariance_sha256 is not None
        ),
    )
    _require_immutable_rows(stacked)
    _require_marginal_parity(stacked, gauge_tree_prior)
    return TreeSparseStackedObservationFactors._from_verified_sparse_stack(
        stacked,
        gauge_tree_prior,
    )


def stack_tree_sparse_observation_factors(
    bundle: ObservationFactorBundle,
    gauge_tree_prior: GaugeTreeSquareRootPriorV1,
    *,
    include_invalid: bool = False,
) -> TreeSparseStackedObservationFactors:
    """Stack rows, verify the tree prior, and retain no dense gauge covariance."""

    if not isinstance(bundle, ObservationFactorBundle):
        raise TypeError("bundle must be an ObservationFactorBundle")
    stacked = stack_sparse_observation_factors(
        bundle,
        include_invalid=include_invalid,
    )
    return bind_gauge_tree_prior(stacked, gauge_tree_prior)


__all__ = [
    "TreeSparseStackedObservationFactors",
    "bind_gauge_tree_prior",
    "stack_tree_sparse_observation_factors",
]
