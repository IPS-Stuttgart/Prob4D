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


def _readonly(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _integer_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain genuine integers")
    return np.asarray(raw, dtype=np.int64)


def _validate_covariance_rows(value: np.ndarray, *, name: str) -> None:
    symmetric = 0.5 * (value + value.swapaxes(1, 2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(1, 2), initial=0.0), 1.0)
    if not np.allclose(
        value,
        symmetric,
        atol=1e-12 * scale[:, None, None],
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    if np.any(np.min(np.linalg.eigvalsh(symmetric), axis=1) < -1e-12 * scale):
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True, slots=True)
class TreeSparseStackedObservationFactors:
    """Sparse observation rows backed by an ``O(K)`` causal gauge-tree prior.

    ``conditional_world_covariance_m2`` is the covariance for explicit-gauge
    inference. ``marginal_world_covariance_m2`` is retained only for diagnostics
    and parity checks; adding it to an explicit gauge design would count the
    gauge uncertainty twice.
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

    def __post_init__(self) -> None:
        if not isinstance(self.gauge_tree_prior, GaugeTreeSquareRootPriorV1):
            raise TypeError("gauge_tree_prior must be a GaugeTreeSquareRootPriorV1")
        mean = np.asarray(self.world_mean_m, dtype=np.float64)
        conditional = np.asarray(
            self.conditional_world_covariance_m2,
            dtype=np.float64,
        )
        marginal = np.asarray(
            self.marginal_world_covariance_m2,
            dtype=np.float64,
        )
        local_jacobian = np.asarray(self.local_gauge_jacobian, dtype=np.float64)
        gauge_indices = _integer_vector(self.gauge_indices, name="gauge_indices")
        association = np.asarray(self.association_probability, dtype=np.float64)
        reliability = np.asarray(self.prior_reliability, dtype=np.float64)
        nominal = np.asarray(self.prior_nominal_probability, dtype=np.float64)
        composite = np.asarray(self.composite_weight, dtype=np.float64)
        point_ids = _integer_vector(self.point_ids, name="point_ids")
        frame_indices = _integer_vector(self.frame_indices, name="frame_indices")
        count = len(mean)
        gauge_count = self.gauge_tree_prior.gauge_count

        if count < 1:
            raise ValueError("tree-sparse observation-factor stack must contain rows")
        if mean.shape != (count, 3):
            raise ValueError("world_mean_m must have shape (M, 3)")
        if conditional.shape != (count, 3, 3):
            raise ValueError(
                "conditional_world_covariance_m2 must have shape (M, 3, 3)"
            )
        if marginal.shape != (count, 3, 3):
            raise ValueError("marginal_world_covariance_m2 must have shape (M, 3, 3)")
        if local_jacobian.shape != (count, 3, 7):
            raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
        if gauge_indices.shape != (count,):
            raise ValueError("gauge_indices must have shape (M,)")
        if np.any(gauge_indices < 0) or np.any(gauge_indices >= gauge_count):
            raise ValueError("gauge_indices reference an unknown gauge")

        for name, values in (
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional),
            ("marginal_world_covariance_m2", marginal),
            ("local_gauge_jacobian", local_jacobian),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        _validate_covariance_rows(
            conditional,
            name="conditional_world_covariance_m2",
        )
        _validate_covariance_rows(
            marginal,
            name="marginal_world_covariance_m2",
        )

        probabilities = (
            ("association_probability", association, True),
            ("prior_reliability", reliability, True),
            ("prior_nominal_probability", nominal, True),
            ("composite_weight", composite, False),
        )
        for name, values, allow_zero in probabilities:
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape (M,)")
            lower = values >= 0.0 if allow_zero else values > 0.0
            if (
                not np.all(np.isfinite(values))
                or not np.all(lower)
                or np.any(values > 1.0)
            ):
                interval = "[0, 1]" if allow_zero else "(0, 1]"
                raise ValueError(f"{name} must lie in {interval}")

        if point_ids.shape != (count,) or frame_indices.shape != (count,):
            raise ValueError("row identity vectors must have shape (M,)")
        if np.any(frame_indices < 0):
            raise ValueError("frame identities must be nonnegative")
        raw_causal_frame_stop = self.causal_frame_stop
        if (
            isinstance(raw_causal_frame_stop, (bool, np.bool_))
            or not isinstance(raw_causal_frame_stop, (int, np.integer))
        ):
            raise TypeError("causal_frame_stop must be a genuine integer")
        causal_frame_stop = int(raw_causal_frame_stop)
        if causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be positive")
        if np.any(frame_indices >= causal_frame_stop):
            raise ValueError("stacked rows cross the exclusive causal frame stop")

        string_fields = {
            "view_ids": self.view_ids,
            "factor_ids": self.factor_ids,
            "correlation_group_ids": self.correlation_group_ids,
        }
        normalized_strings: dict[str, tuple[str, ...]] = {}
        for name, raw_values in string_fields.items():
            if not isinstance(raw_values, tuple) or any(
                not isinstance(value, str) for value in raw_values
            ):
                raise TypeError(f"{name} must be a tuple of strings")
            values = tuple(raw_values)
            if len(values) != count or any(not value for value in values):
                raise ValueError(f"{name} must contain one nonempty string per row")
            normalized_strings[name] = values

        gauge_marginal = self.gauge_tree_prior.row_marginal_covariance(
            local_jacobian,
            gauge_indices,
        )
        expected_marginal = conditional + gauge_marginal
        expected_marginal = 0.5 * (
            expected_marginal + expected_marginal.swapaxes(1, 2)
        )
        if not np.allclose(
            marginal,
            expected_marginal,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "marginal_world_covariance_m2 does not match the tree gauge prior"
            )

        for name, value in (
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional),
            ("marginal_world_covariance_m2", marginal),
            ("local_gauge_jacobian", local_jacobian),
            ("gauge_indices", gauge_indices),
            ("association_probability", association),
            ("prior_reliability", reliability),
            ("prior_nominal_probability", nominal),
            ("composite_weight", composite),
            ("point_ids", point_ids),
            ("frame_indices", frame_indices),
        ):
            object.__setattr__(self, name, _readonly(value))
        for name, values in normalized_strings.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)

    @classmethod
    def _from_verified_sparse_stack(
        cls,
        stacked: SparseStackedObservationFactors,
        gauge_tree_prior: GaugeTreeSquareRootPriorV1,
    ) -> TreeSparseStackedObservationFactors:
        """Transfer immutable row arrays after exact dense-prior verification."""

        result = object.__new__(cls)
        for name in (
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
            "view_ids",
            "factor_ids",
            "correlation_group_ids",
            "causal_frame_stop",
        ):
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
