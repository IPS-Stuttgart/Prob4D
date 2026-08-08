"""Sparse in-memory stacking for explicit-gauge observation-factor updates.

The neutral schema-v4 :class:`~prob4d.observation_factors.ObservationFactorBundle`
stores one local seven-dimensional ``Sim(3)`` gauge block per observation row and
one joint prior over all gauges.  The historical dense stack expands every row
to ``3 x 7K`` even though exactly one seven-column block is nonzero.  This module
keeps the local block plus an integer gauge index, preserving the exact
statistics while reducing Jacobian storage from ``O(MK)`` to ``O(M)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .observation_factors import ObservationFactorBundle

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
    if raw.dtype.kind == "u" and raw.size:
        maximum = int(np.max(raw))
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(f"{name} values must fit in the int64 range")
    return np.asarray(raw, dtype=np.int64)


def _string_tuple(
    value: object,
    *,
    name: str,
    expected_length: int | None = None,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of literal strings")
    values = cast(tuple[object, ...], value)
    if any(type(item) is not str for item in values):
        raise TypeError(f"{name} must contain literal strings")
    if any(not item for item in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} entries")
    return cast(tuple[str, ...], values)


def _require_psd(value: np.ndarray, *, name: str, tolerance: float = 1e-12) -> None:
    symmetric = 0.5 * (value + value.T)
    if not np.allclose(value, symmetric, atol=tolerance, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(symmetric), initial=0.0) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")


def _require_row_covariance(
    value: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> None:
    finite_rows = np.all(np.isfinite(value), axis=(1, 2))
    if not np.any(finite_rows):
        return
    selected = value[finite_rows]
    symmetric = 0.5 * (selected + selected.swapaxes(1, 2))
    if not np.allclose(selected, symmetric, atol=tolerance, rtol=1e-10):
        raise ValueError(f"{name} finite rows must be symmetric")
    if np.any(np.linalg.eigvalsh(symmetric) < -tolerance):
        raise ValueError(f"{name} finite rows must be positive semidefinite")


def _row_gauge_marginal_covariance(
    local_gauge_jacobian: np.ndarray,
    gauge_indices: np.ndarray,
    gauge_prior_covariance: np.ndarray,
    *,
    gauge_count: int,
) -> np.ndarray:
    blocks: FloatArray = np.empty(
        (len(local_gauge_jacobian), 7, 7),
        dtype=np.float64,
    )
    for gauge_index in range(gauge_count):
        selected = gauge_indices == gauge_index
        start = 7 * gauge_index
        blocks[selected] = gauge_prior_covariance[
            start : start + 7,
            start : start + 7,
        ]
    with np.errstate(invalid="ignore", over="ignore"):
        result: FloatArray = np.einsum(
            "nia,nab,njb->nij",
            local_gauge_jacobian,
            blocks,
            local_gauge_jacobian,
            optimize=True,
        )
    return 0.5 * (result + result.swapaxes(1, 2))


@dataclass(frozen=True, slots=True)
class SparseStackedObservationFactors:
    """Flattened rows with one local gauge block and one gauge index per row.

    ``conditional_world_covariance_m2`` is the covariance to use when gauge
    errors remain explicit nuisance variables.  ``marginal_world_covariance_m2``
    is retained for diagnostics and consumers that integrate gauges out; it must
    not be added to the explicit gauge design.
    """

    world_mean_m: FloatArray
    conditional_world_covariance_m2: FloatArray
    marginal_world_covariance_m2: FloatArray
    local_gauge_jacobian: FloatArray
    gauge_indices: IntArray
    gauge_prior_covariance: FloatArray
    association_probability: FloatArray
    prior_reliability: FloatArray
    prior_nominal_probability: FloatArray
    composite_weight: FloatArray
    point_ids: IntArray
    frame_indices: IntArray
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    gauge_ids: tuple[str, ...]
    causal_frame_stop: int

    def __post_init__(self) -> None:
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
        gauge_prior = np.asarray(self.gauge_prior_covariance, dtype=np.float64)
        association = np.asarray(self.association_probability, dtype=np.float64)
        reliability = np.asarray(self.prior_reliability, dtype=np.float64)
        nominal = np.asarray(self.prior_nominal_probability, dtype=np.float64)
        composite = np.asarray(self.composite_weight, dtype=np.float64)
        point_ids = _integer_vector(self.point_ids, name="point_ids")
        frame_indices = _integer_vector(self.frame_indices, name="frame_indices")
        gauge_ids = _string_tuple(self.gauge_ids, name="gauge_ids")
        count = len(mean)
        gauge_count = len(gauge_ids)
        gauge_dimension = 7 * gauge_count

        if count < 1:
            raise ValueError("sparse observation-factor stack must contain rows")
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
        if not gauge_ids:
            raise ValueError("gauge_ids must contain at least one identifier")
        if len(set(gauge_ids)) != gauge_count:
            raise ValueError("gauge_ids must be unique")
        if np.any(gauge_indices < 0) or np.any(gauge_indices >= gauge_count):
            raise ValueError("gauge_indices reference an unknown gauge")
        if gauge_prior.shape != (gauge_dimension, gauge_dimension):
            raise ValueError(
                "gauge_prior_covariance must have shape "
                f"({gauge_dimension}, {gauge_dimension})"
            )
        if not np.all(np.isfinite(gauge_prior)):
            raise ValueError("gauge_prior_covariance must be finite")
        _require_psd(gauge_prior, name="gauge_prior_covariance")
        _require_row_covariance(
            conditional,
            name="conditional_world_covariance_m2",
        )
        _require_row_covariance(
            marginal,
            name="marginal_world_covariance_m2",
        )
        gauge_marginal = _row_gauge_marginal_covariance(
            local_jacobian,
            gauge_indices,
            gauge_prior,
            gauge_count=gauge_count,
        )
        with np.errstate(invalid="ignore", over="ignore"):
            expected_marginal = conditional + gauge_marginal
            expected_marginal = 0.5 * (
                expected_marginal + expected_marginal.swapaxes(1, 2)
            )
        if not np.allclose(
            marginal,
            expected_marginal,
            atol=1e-12,
            rtol=1e-10,
            equal_nan=True,
        ):
            raise ValueError(
                "marginal_world_covariance_m2 must equal conditional covariance "
                "plus the selected gauge-prior contribution"
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
        raw_causal_frame_stop = self.causal_frame_stop
        if (
            isinstance(raw_causal_frame_stop, (bool, np.bool_))
            or not isinstance(raw_causal_frame_stop, (int, np.integer))
        ):
            raise TypeError("causal_frame_stop must be a genuine integer")
        causal_frame_stop = int(raw_causal_frame_stop)
        if causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be positive")
        if np.any(frame_indices < 0) or np.any(frame_indices >= causal_frame_stop):
            raise ValueError("stacked rows cross the exclusive causal frame stop")

        string_fields = {
            "view_ids": _string_tuple(
                self.view_ids,
                name="view_ids",
                expected_length=count,
            ),
            "factor_ids": _string_tuple(
                self.factor_ids,
                name="factor_ids",
                expected_length=count,
            ),
            "correlation_group_ids": _string_tuple(
                self.correlation_group_ids,
                name="correlation_group_ids",
                expected_length=count,
            ),
        }

        for name, value in (
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional),
            ("marginal_world_covariance_m2", marginal),
            ("local_gauge_jacobian", local_jacobian),
            ("gauge_indices", gauge_indices),
            ("gauge_prior_covariance", gauge_prior),
            ("association_probability", association),
            ("prior_reliability", reliability),
            ("prior_nominal_probability", nominal),
            ("composite_weight", composite),
            ("point_ids", point_ids),
            ("frame_indices", frame_indices),
        ):
            object.__setattr__(self, name, _readonly(value))
        for name, values in string_fields.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)

    @property
    def observation_count(self) -> int:
        return len(self.world_mean_m)

    @property
    def gauge_count(self) -> int:
        return len(self.gauge_ids)

    @property
    def dense_gauge_dimension(self) -> int:
        return 7 * self.gauge_count

    @property
    def sparse_gauge_design_nbytes(self) -> int:
        """Bytes retained for the local Jacobians and row gauge indices."""

        return int(self.local_gauge_jacobian.nbytes + self.gauge_indices.nbytes)

    @property
    def dense_gauge_design_nbytes(self) -> int:
        """Bytes required by the equivalent float64 dense ``M x 3 x 7K`` design."""

        return int(
            self.observation_count
            * 3
            * self.dense_gauge_dimension
            * np.dtype(np.float64).itemsize
        )

    def dense_gauge_jacobian(self) -> FloatArray:
        """Materialize the historical dense design as a mutable compatibility copy."""

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
        """Apply one stacked or block-shaped gauge perturbation to every row."""

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

        return _row_gauge_marginal_covariance(
            self.local_gauge_jacobian,
            self.gauge_indices,
            self.gauge_prior_covariance,
            gauge_count=self.gauge_count,
        )


def stack_sparse_observation_factors(
    bundle: ObservationFactorBundle,
    *,
    include_invalid: bool = False,
) -> SparseStackedObservationFactors:
    """Stack a factor bundle without expanding zero gauge-Jacobian blocks."""

    if not isinstance(bundle, ObservationFactorBundle):
        raise TypeError("bundle must be an ObservationFactorBundle")
    gauge_ids = tuple(gauge.window_id for gauge in bundle.gauges)
    gauge_positions = {gauge_id: index for index, gauge_id in enumerate(gauge_ids)}

    means: list[np.ndarray] = []
    conditional_covariances: list[np.ndarray] = []
    marginal_covariances: list[np.ndarray] = []
    local_jacobians: list[np.ndarray] = []
    gauge_indices: list[np.ndarray] = []
    association_probabilities: list[np.ndarray] = []
    prior_reliabilities: list[np.ndarray] = []
    prior_nominal_probabilities: list[np.ndarray] = []
    composite_weights: list[np.ndarray] = []
    point_ids: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    view_ids: list[str] = []
    factor_ids: list[str] = []
    correlation_group_ids: list[str] = []

    for factor in bundle.factors:
        linearized = bundle.linearize(factor)
        selected = (
            np.ones(len(factor.point_ids), dtype=bool)
            if include_invalid
            else (
                factor.valid_mask
                & (factor.association_probability > 0.0)
                & (factor.prior_reliability > 0.0)
            )
        )
        selected_count = int(np.count_nonzero(selected))
        if selected_count == 0:
            continue
        gauge_index = gauge_positions[factor.gauge_id]
        means.append(linearized.world_mean_m[selected])
        conditional_covariances.append(
            linearized.conditional_world_covariance_m2[selected]
        )
        marginal_covariances.append(linearized.marginal_world_covariance_m2[selected])
        local_jacobians.append(linearized.gauge_jacobian[selected])
        gauge_indices.append(np.full(selected_count, gauge_index, dtype=np.int64))
        association_probabilities.append(factor.association_probability[selected])
        prior_reliabilities.append(factor.prior_reliability[selected])
        prior_nominal_probabilities.append(
            np.full(
                selected_count,
                factor.prior_nominal_probability,
                dtype=np.float64,
            )
        )
        composite_weights.append(
            np.full(selected_count, factor.composite_weight, dtype=np.float64)
        )
        point_ids.append(factor.point_ids[selected])
        frame_indices.append(
            np.full(selected_count, factor.frame_index, dtype=np.int64)
        )
        view_ids.extend([factor.view_id] * selected_count)
        factor_ids.extend([factor.factor_id] * selected_count)
        correlation_group_ids.extend(
            [factor.correlation_group_id] * selected_count
        )

    if not means:
        raise ValueError("observation-factor stack has no selected rows")

    return SparseStackedObservationFactors(
        world_mean_m=np.concatenate(means, axis=0),
        conditional_world_covariance_m2=np.concatenate(
            conditional_covariances,
            axis=0,
        ),
        marginal_world_covariance_m2=np.concatenate(
            marginal_covariances,
            axis=0,
        ),
        local_gauge_jacobian=np.concatenate(local_jacobians, axis=0),
        gauge_indices=np.concatenate(gauge_indices),
        gauge_prior_covariance=bundle.joint_gauge_covariance,
        association_probability=np.concatenate(association_probabilities),
        prior_reliability=np.concatenate(prior_reliabilities),
        prior_nominal_probability=np.concatenate(prior_nominal_probabilities),
        composite_weight=np.concatenate(composite_weights),
        point_ids=np.concatenate(point_ids),
        frame_indices=np.concatenate(frame_indices),
        view_ids=tuple(view_ids),
        factor_ids=tuple(factor_ids),
        correlation_group_ids=tuple(correlation_group_ids),
        gauge_ids=gauge_ids,
        causal_frame_stop=bundle.causal_frame_stop,
    )


__all__ = [
    "SparseStackedObservationFactors",
    "stack_sparse_observation_factors",
]
