"""Conservative geometric error budget for axial query-orbit coefficients.

This module converts separately supplied bounds on a unit axis, pivot, and
reference points into the operator-norm coefficient bound consumed by
``certify_axial_linear_query``.  It does not estimate those geometric bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _array(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _unit_axis(value: ArrayLike) -> FloatArray:
    axis = _array(value, name="estimated_axis")
    if axis.shape != (3,):
        raise ValueError("estimated_axis must have shape (3,)")
    scale = float(np.max(np.abs(axis)))
    if scale == 0.0:
        raise ValueError("estimated_axis must be nonzero")
    normalized = axis / scale
    normalized = normalized / float(np.linalg.norm(normalized))
    normalized.setflags(write=False)
    return normalized


def axial_point_orbit_coefficients(
    points: ArrayLike,
    *,
    axis: ArrayLike,
    pivot: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return cosine and sine coefficients for point rotations about a line."""

    reference = _array(points, name="points")
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != 3:
        raise ValueError("points must have nonempty shape (N, 3)")
    unit = _unit_axis(axis)
    center = _array(pivot, name="pivot")
    if center.shape != (3,):
        raise ValueError("pivot must have shape (3,)")
    offset = reference - center
    parallel = np.outer(offset @ unit, unit)
    cosine = offset - parallel
    sine = np.cross(unit, cosine)
    cosine.setflags(write=False)
    sine.setflags(write=False)
    return cosine, sine


def project_axial_query_coefficients(
    points: ArrayLike,
    *,
    axis: ArrayLike,
    pivot: ArrayLike,
    query_weights: ArrayLike,
) -> FloatArray:
    """Return ``B`` for a linear query of point coordinates.

    ``query_weights`` has shape ``(N,3)`` for one scalar query or ``(D,N,3)``
    for a vector query.
    """

    reference = _array(points, name="points")
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != 3:
        raise ValueError("points must have nonempty shape (N, 3)")
    weights = _array(query_weights, name="query_weights")
    if weights.ndim == 2:
        weights = weights[None, :, :]
    if weights.ndim != 3 or weights.shape[0] == 0 or weights.shape[1:] != reference.shape:
        raise ValueError("query_weights must have shape (N,3) or (D,N,3)")
    cosine, sine = axial_point_orbit_coefficients(
        reference,
        axis=axis,
        pivot=pivot,
    )
    result = np.column_stack(
        (
            np.einsum("dnc,nc->d", weights, cosine),
            np.einsum("dnc,nc->d", weights, sine),
        )
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AxialGeometryCoefficientBound:
    """Conservative ingredients of one query-coefficient error budget."""

    coefficient_operator_error_bound: float
    query_operator_norm: float
    stacked_coefficient_frobenius_bound: float
    projection_operator_error_bound: float
    point_offset_error_bounds: FloatArray
    cosine_coefficient_error_bounds: FloatArray
    sine_coefficient_error_bounds: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "coefficient_operator_error_bound",
            "query_operator_norm",
            "stacked_coefficient_frobenius_bound",
            "projection_operator_error_bound",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name=name),
            )
        for name in (
            "point_offset_error_bounds",
            "cosine_coefficient_error_bounds",
            "sine_coefficient_error_bounds",
        ):
            value = _array(getattr(self, name), name=name)
            if value.ndim != 1 or value.size == 0 or np.any(value < 0.0):
                raise ValueError(f"{name} must be a nonempty nonnegative vector")
            object.__setattr__(self, name, value)
        size = self.point_offset_error_bounds.size
        if (
            self.cosine_coefficient_error_bounds.size != size
            or self.sine_coefficient_error_bounds.size != size
        ):
            raise ValueError("all per-point error vectors must have equal length")


def bound_axial_query_coefficient_error(
    estimated_points: ArrayLike,
    *,
    estimated_axis: ArrayLike,
    estimated_pivot: ArrayLike,
    query_weights: ArrayLike,
    point_position_error_bounds: ArrayLike,
    axis_vector_error_bound: float,
    pivot_position_error_bound: float,
) -> AxialGeometryCoefficientBound:
    """Bound the operator error of projected axial orbit coefficients.

    The contract is

    ``||p_i - p_hat_i|| <= eps_point_i``,
    ``||c - c_hat|| <= eps_pivot``, and
    ``||u - u_hat|| <= eps_axis``

    for unit true and estimated axes.  The returned coefficient bound is valid
    simultaneously for all points under these deterministic Euclidean bounds.
    It is intentionally conservative and uses a Frobenius upper bound before
    applying the linear query map.
    """

    points = _array(estimated_points, name="estimated_points")
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError("estimated_points must have nonempty shape (N, 3)")
    axis = _unit_axis(estimated_axis)
    pivot = _array(estimated_pivot, name="estimated_pivot")
    if pivot.shape != (3,):
        raise ValueError("estimated_pivot must have shape (3,)")
    point_errors = _array(
        point_position_error_bounds,
        name="point_position_error_bounds",
    )
    if point_errors.shape != (points.shape[0],) or np.any(point_errors < 0.0):
        raise ValueError(
            "point_position_error_bounds must be a nonnegative vector of length N"
        )
    axis_error = _nonnegative(
        axis_vector_error_bound,
        name="axis_vector_error_bound",
    )
    if axis_error > 2.0:
        raise ValueError("axis_vector_error_bound cannot exceed two for unit axes")
    pivot_error = _nonnegative(
        pivot_position_error_bound,
        name="pivot_position_error_bound",
    )
    weights = _array(query_weights, name="query_weights")
    if weights.ndim == 2:
        weights = weights[None, :, :]
    if weights.ndim != 3 or weights.shape[0] == 0 or weights.shape[1:] != points.shape:
        raise ValueError("query_weights must have shape (N,3) or (D,N,3)")

    estimated_offset = points - pivot
    estimated_cosine, _ = axial_point_orbit_coefficients(
        points,
        axis=axis,
        pivot=pivot,
    )
    offset_errors = point_errors + pivot_error

    # ||u u^T - u_hat u_hat^T|| <= 2 ||u-u_hat|| for unit axes.
    projection_error = min(2.0, 2.0 * axis_error)
    cosine_errors = offset_errors + projection_error * np.linalg.norm(
        estimated_offset,
        axis=1,
    )

    # b = u x a.  Bound the true cosine coefficient by its estimate plus
    # its error, then use ||x cross y|| <= ||x|| ||y||.
    true_cosine_norm_bound = np.linalg.norm(estimated_cosine, axis=1) + cosine_errors
    sine_errors = cosine_errors + axis_error * true_cosine_norm_bound

    stacked_frobenius = float(
        np.sqrt(np.sum(cosine_errors**2) + np.sum(sine_errors**2))
    )
    flattened_weights = weights.reshape(weights.shape[0], -1)
    query_operator_norm = float(
        np.linalg.svd(flattened_weights, compute_uv=False)[0]
    )
    coefficient_bound = query_operator_norm * stacked_frobenius
    return AxialGeometryCoefficientBound(
        coefficient_operator_error_bound=coefficient_bound,
        query_operator_norm=query_operator_norm,
        stacked_coefficient_frobenius_bound=stacked_frobenius,
        projection_operator_error_bound=projection_error,
        point_offset_error_bounds=offset_errors,
        cosine_coefficient_error_bounds=cosine_errors,
        sine_coefficient_error_bounds=sine_errors,
    )
