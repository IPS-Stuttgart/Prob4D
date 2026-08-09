"""Tree-backed sparse stacking for explicit-gauge observation-factor updates.

The historical sparse stack removes the ``M x 3 x 7K`` row expansion but retains
one dense ``7K x 7K`` gauge covariance. This module binds the same immutable row
representation to :class:`GaugeTreeSquareRootPriorV1` after exact dense parity
verification, or builds it directly from validated row arrays and a tree prior.
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
    _integer_vector,
    _readonly,
    _require_row_covariance,
    _string_tuple,
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


@dataclass(frozen=True, slots=True)
class _ValidatedDirectRows:
    world_mean_m: np.ndarray
    conditional_world_covariance_m2: np.ndarray
    marginal_world_covariance_m2: np.ndarray
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    association_probability: np.ndarray
    prior_reliability: np.ndarray
    prior_nominal_probability: np.ndarray
    composite_weight: np.ndarray
    point_ids: np.ndarray
    frame_indices: np.ndarray
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    causal_frame_stop: int


def _probability_vector(
    value: object,
    *,
    name: str,
    count: int,
    allow_zero: bool,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape (M,)")
    lower = result >= 0.0 if allow_zero else result > 0.0
    if not np.all(np.isfinite(result)) or not np.all(lower) or np.any(result > 1.0):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _causal_frame_stop(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError("causal_frame_stop must be a genuine integer")
    result = int(value)
    if result < 1:
        raise ValueError("causal_frame_stop must be positive")
    return result


def _require_row_metadata_consistency(
    *,
    gauge_indices: np.ndarray,
    point_ids: np.ndarray,
    frame_indices: np.ndarray,
    view_ids: tuple[str, ...],
    factor_ids: tuple[str, ...],
    correlation_group_ids: tuple[str, ...],
    prior_nominal_probability: np.ndarray,
    composite_weight: np.ndarray,
) -> None:
    factor_metadata: dict[str, tuple[object, ...]] = {}
    group_settings: dict[str, tuple[float, float]] = {}
    factor_points: set[tuple[str, int]] = set()
    for index, factor_id in enumerate(factor_ids):
        metadata = (
            view_ids[index],
            int(frame_indices[index]),
            int(gauge_indices[index]),
            correlation_group_ids[index],
            float(prior_nominal_probability[index]),
            float(composite_weight[index]),
        )
        previous_metadata = factor_metadata.setdefault(factor_id, metadata)
        if previous_metadata != metadata:
            raise ValueError("rows with one factor_id must share factor metadata")

        group_id = correlation_group_ids[index]
        settings = (
            float(prior_nominal_probability[index]),
            float(composite_weight[index]),
        )
        previous_settings = group_settings.setdefault(group_id, settings)
        if previous_settings != settings:
            raise ValueError(
                "rows in one correlation group must share nominal probability and composite weight"
            )

        point_key = (factor_id, int(point_ids[index]))
        if point_key in factor_points:
            raise ValueError("point_ids must be unique within each factor")
        factor_points.add(point_key)


def _validate_direct_rows(
    gauge_tree_prior: GaugeTreeSquareRootPriorV1,
    *,
    world_mean_m: object,
    conditional_world_covariance_m2: object,
    local_gauge_jacobian: object,
    gauge_indices: object,
    association_probability: object,
    prior_reliability: object,
    prior_nominal_probability: object,
    composite_weight: object,
    point_ids: object,
    frame_indices: object,
    view_ids: object,
    factor_ids: object,
    correlation_group_ids: object,
    causal_frame_stop: object,
) -> _ValidatedDirectRows:
    mean = np.asarray(world_mean_m, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[0] < 1 or mean.shape[1] != 3:
        raise ValueError("world_mean_m must have shape (M, 3) with M >= 1")
    count = mean.shape[0]
    conditional = np.asarray(conditional_world_covariance_m2, dtype=np.float64)
    local_jacobian = np.asarray(local_gauge_jacobian, dtype=np.float64)
    indices = _integer_vector(np.asarray(gauge_indices), name="gauge_indices")
    points = _integer_vector(np.asarray(point_ids), name="point_ids")
    frames = _integer_vector(np.asarray(frame_indices), name="frame_indices")

    if conditional.shape != (count, 3, 3):
        raise ValueError("conditional_world_covariance_m2 must have shape (M, 3, 3)")
    if local_jacobian.shape != (count, 3, 7):
        raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
    if indices.shape != (count,):
        raise ValueError("gauge_indices must have shape (M,)")
    if points.shape != (count,) or frames.shape != (count,):
        raise ValueError("row identity vectors must have shape (M,)")
    if not np.all(np.isfinite(mean)):
        raise ValueError("world_mean_m must be finite")
    if not np.all(np.isfinite(conditional)):
        raise ValueError("conditional_world_covariance_m2 must be finite")
    if not np.all(np.isfinite(local_jacobian)):
        raise ValueError("local_gauge_jacobian must be finite")
    if np.any(indices < 0) or np.any(indices >= gauge_tree_prior.gauge_count):
        raise ValueError("gauge_indices reference an unknown gauge")

    _require_row_covariance(
        conditional,
        name="conditional_world_covariance_m2",
    )
    association = _probability_vector(
        association_probability,
        name="association_probability",
        count=count,
        allow_zero=False,
    )
    reliability = _probability_vector(
        prior_reliability,
        name="prior_reliability",
        count=count,
        allow_zero=False,
    )
    nominal = _probability_vector(
        prior_nominal_probability,
        name="prior_nominal_probability",
        count=count,
        allow_zero=True,
    )
    composite = _probability_vector(
        composite_weight,
        name="composite_weight",
        count=count,
        allow_zero=False,
    )
    stop = _causal_frame_stop(causal_frame_stop)
    if np.any(frames < 0) or np.any(frames >= stop):
        raise ValueError("stacked rows cross the exclusive causal frame stop")

    views = _string_tuple(view_ids, name="view_ids", expected_length=count)
    factors = _string_tuple(factor_ids, name="factor_ids", expected_length=count)
    groups = _string_tuple(
        correlation_group_ids,
        name="correlation_group_ids",
        expected_length=count,
    )
    _require_row_metadata_consistency(
        gauge_indices=indices,
        point_ids=points,
        frame_indices=frames,
        view_ids=views,
        factor_ids=factors,
        correlation_group_ids=groups,
        prior_nominal_probability=nominal,
        composite_weight=composite,
    )

    gauge_marginal = gauge_tree_prior.row_marginal_covariance(
        local_jacobian,
        indices,
    )
    marginal = conditional + gauge_marginal
    marginal = 0.5 * (marginal + marginal.swapaxes(1, 2))
    if not np.all(np.isfinite(marginal)):
        raise ValueError("derived marginal_world_covariance_m2 must be finite")
    _require_row_covariance(
        marginal,
        name="derived marginal_world_covariance_m2",
    )

    return _ValidatedDirectRows(
        world_mean_m=mean,
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        local_gauge_jacobian=local_jacobian,
        gauge_indices=indices,
        association_probability=association,
        prior_reliability=reliability,
        prior_nominal_probability=nominal,
        composite_weight=composite,
        point_ids=points,
        frame_indices=frames,
        view_ids=views,
        factor_ids=factors,
        correlation_group_ids=groups,
        causal_frame_stop=stop,
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
        raise ValueError("marginal_world_covariance_m2 does not match the tree gauge prior")


def _require_immutable_rows(stacked: SparseStackedObservationFactors) -> None:
    for name in _ROW_ARRAY_FIELDS:
        if np.asarray(getattr(stacked, name)).flags.writeable:
            raise ValueError(f"stacked {name} must be immutable")


@dataclass(frozen=True, slots=True, init=False)
class TreeSparseStackedObservationFactors:
    """Factory-built sparse rows backed by an ``O(K)`` causal gauge-tree prior.

    Direct producers use :func:`build_tree_sparse_observation_factors`. Existing
    schema-v4 bundles use :func:`bind_gauge_tree_prior` or
    :func:`stack_tree_sparse_observation_factors`, which verify the tree against
    the complete dense prior before releasing it.
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
            "use build_tree_sparse_observation_factors, bind_gauge_tree_prior, "
            "or stack_tree_sparse_observation_factors"
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

    @classmethod
    def _from_direct_rows(
        cls,
        rows: _ValidatedDirectRows,
        gauge_tree_prior: GaugeTreeSquareRootPriorV1,
    ) -> TreeSparseStackedObservationFactors:
        result = object.__new__(cls)
        for name in _ROW_ARRAY_FIELDS:
            object.__setattr__(result, name, _readonly(getattr(rows, name)))
        object.__setattr__(result, "view_ids", rows.view_ids)
        object.__setattr__(result, "factor_ids", rows.factor_ids)
        object.__setattr__(result, "correlation_group_ids", rows.correlation_group_ids)
        object.__setattr__(result, "causal_frame_stop", rows.causal_frame_stop)
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
            self.observation_count * 3 * self.dense_gauge_dimension * np.dtype(np.float64).itemsize
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


def build_tree_sparse_observation_factors(
    gauge_tree_prior: GaugeTreeSquareRootPriorV1,
    *,
    world_mean_m: object,
    conditional_world_covariance_m2: object,
    local_gauge_jacobian: object,
    gauge_indices: object,
    association_probability: object,
    prior_reliability: object,
    prior_nominal_probability: object,
    composite_weight: object,
    point_ids: object,
    frame_indices: object,
    view_ids: object,
    factor_ids: object,
    correlation_group_ids: object,
    causal_frame_stop: object,
) -> TreeSparseStackedObservationFactors:
    """Build a tree-backed stack without constructing a dense gauge covariance.

    This is the native producer path. It accepts selected, finite execution rows,
    validates row and grouping semantics, derives marginal covariance from the
    sparse prior, and copies every numerical input into an immutable result.
    """

    if not isinstance(gauge_tree_prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("gauge_tree_prior must be a GaugeTreeSquareRootPriorV1")
    rows = _validate_direct_rows(
        gauge_tree_prior,
        world_mean_m=world_mean_m,
        conditional_world_covariance_m2=conditional_world_covariance_m2,
        local_gauge_jacobian=local_gauge_jacobian,
        gauge_indices=gauge_indices,
        association_probability=association_probability,
        prior_reliability=prior_reliability,
        prior_nominal_probability=prior_nominal_probability,
        composite_weight=composite_weight,
        point_ids=point_ids,
        frame_indices=frame_indices,
        view_ids=view_ids,
        factor_ids=factor_ids,
        correlation_group_ids=correlation_group_ids,
        causal_frame_stop=causal_frame_stop,
    )
    return TreeSparseStackedObservationFactors._from_direct_rows(
        rows,
        gauge_tree_prior,
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
        require_source_digest=(gauge_tree_prior.source_joint_covariance_sha256 is not None),
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
    "build_tree_sparse_observation_factors",
    "stack_tree_sparse_observation_factors",
]
