"""Fail-closed query-identifiability certificates for approximate finite orbits.

The core certificate treats an axial linear-query orbit

    q(theta) = center + B [cos(theta), sin(theta)]^T

with a supplied spectral-norm error bound ``||B_true - B_hat||_2 <= eta``.
Weyl's singular-value perturbation bound then gives a certified interval for
its complete Euclidean orbit diameter.  The module does not estimate ``eta``;
that information must come from a separately justified geometric, bootstrap,
or ensemble bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
OrbitDecision = Literal[
    "certified-invariant",
    "certified-variant",
    "undetermined",
]


def _finite_array(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _nonnegative_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_scalar(value: float, *, name: str) -> float:
    result = _nonnegative_scalar(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _decision(lower: float, upper: float, tolerance: float) -> OrbitDecision:
    if upper <= tolerance:
        return "certified-invariant"
    if lower > tolerance:
        return "certified-variant"
    return "undetermined"


def batch_axial_orbit_diameters(coefficients: ArrayLike) -> FloatArray:
    """Return exact Euclidean diameters for matrices with two orbit columns.

    ``coefficients`` has shape ``(..., query_dimension, 2)``.  For
    ``q(theta) = c + B [cos(theta), sin(theta)]``, the complete pairwise
    Euclidean orbit diameter is exactly ``2 * sigma_max(B)``.  The closed form
    below diagonalizes only the two-by-two Gram matrix and is vectorized over
    arbitrary leading dimensions.
    """

    matrix = _finite_array(coefficients, name="coefficients")
    if matrix.ndim < 2 or matrix.shape[-2] == 0 or matrix.shape[-1] != 2:
        raise ValueError(
            "coefficients must have shape (..., query_dimension, 2)"
        )
    first = matrix[..., :, 0]
    second = matrix[..., :, 1]
    gram_00 = np.sum(first * first, axis=-1)
    gram_01 = np.sum(first * second, axis=-1)
    gram_11 = np.sum(second * second, axis=-1)
    discriminant = np.hypot(gram_00 - gram_11, 2.0 * gram_01)
    largest_eigenvalue = 0.5 * (gram_00 + gram_11 + discriminant)
    diameter = 2.0 * np.sqrt(np.maximum(largest_eigenvalue, 0.0))
    result = np.asarray(diameter, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RobustAxialOrbitCertificate:
    """Certified diameter interval for one approximate axial query orbit."""

    estimated_diameter: float
    lower_diameter: float
    upper_diameter: float
    coefficient_error_bound: float
    invariance_tolerance: float
    nominal_local_derivative_norm: float
    nominal_angle_radians: float
    decision: OrbitDecision

    def __post_init__(self) -> None:
        for name in (
            "estimated_diameter",
            "lower_diameter",
            "upper_diameter",
            "coefficient_error_bound",
            "invariance_tolerance",
            "nominal_local_derivative_norm",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_scalar(getattr(self, name), name=name),
            )
        angle = float(self.nominal_angle_radians)
        if not np.isfinite(angle):
            raise ValueError("nominal_angle_radians must be finite")
        object.__setattr__(self, "nominal_angle_radians", angle)
        if self.lower_diameter > self.upper_diameter:
            raise ValueError("lower_diameter cannot exceed upper_diameter")
        if self.decision != _decision(
            self.lower_diameter,
            self.upper_diameter,
            self.invariance_tolerance,
        ):
            raise ValueError("decision is inconsistent with the diameter interval")

    @property
    def update_admitted(self) -> bool:
        """Only a certified invariant query is admitted; uncertainty falls back."""

        return self.decision == "certified-invariant"


def certify_axial_linear_query(
    estimated_coefficients: ArrayLike,
    *,
    coefficient_error_bound: float,
    invariance_tolerance: float,
    nominal_angle_radians: float = 0.0,
) -> RobustAxialOrbitCertificate:
    """Certify invariance, variation, or indeterminacy of an axial query.

    The error bound is an operator-norm bound on the complete coefficient
    matrix.  Since the true diameter is ``2*sigma_max(B_true)``, singular-value
    perturbation gives

    ``d_true in [max(0, d_hat - 2*eta), d_hat + 2*eta]``.

    An ``undetermined`` result is deliberately not admitted.
    """

    matrix = _finite_array(
        estimated_coefficients,
        name="estimated_coefficients",
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 2:
        raise ValueError(
            "estimated_coefficients must have shape (query_dimension, 2)"
        )
    error = _nonnegative_scalar(
        coefficient_error_bound,
        name="coefficient_error_bound",
    )
    tolerance = _positive_scalar(
        invariance_tolerance,
        name="invariance_tolerance",
    )
    angle = float(nominal_angle_radians)
    if not np.isfinite(angle):
        raise ValueError("nominal_angle_radians must be finite")
    estimated = float(batch_axial_orbit_diameters(matrix))
    lower = max(0.0, estimated - 2.0 * error)
    upper = estimated + 2.0 * error
    tangent = np.array([-np.sin(angle), np.cos(angle)])
    derivative_norm = float(np.linalg.norm(matrix @ tangent))
    return RobustAxialOrbitCertificate(
        estimated_diameter=estimated,
        lower_diameter=lower,
        upper_diameter=upper,
        coefficient_error_bound=error,
        invariance_tolerance=tolerance,
        nominal_local_derivative_norm=derivative_norm,
        nominal_angle_radians=angle,
        decision=_decision(lower, upper, tolerance),
    )


def sampled_pairwise_diameter(values: ArrayLike) -> float:
    """Return the exact Euclidean diameter of a finite query sample."""

    points = _finite_array(values, name="values")
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] == 0:
        raise ValueError("values must have shape (sample_count>=2, dimension>=1)")
    maximum = 0.0
    for start in range(0, points.shape[0], 128):
        difference = points[start : start + 128, None, :] - points[None, :, :]
        maximum = max(
            maximum,
            float(np.max(np.linalg.norm(difference, axis=-1))),
        )
    return maximum


@dataclass(frozen=True)
class SampledPeriodicOrbitCertificate:
    """Lipschitz-certified diameter interval from a uniform periodic grid."""

    sampled_diameter: float
    lower_diameter: float
    upper_diameter: float
    lipschitz_bound: float
    angular_cover_radius: float
    sample_count: int
    invariance_tolerance: float
    decision: OrbitDecision

    def __post_init__(self) -> None:
        for name in (
            "sampled_diameter",
            "lower_diameter",
            "upper_diameter",
            "lipschitz_bound",
            "angular_cover_radius",
            "invariance_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_scalar(getattr(self, name), name=name),
            )
        if type(self.sample_count) is not int or self.sample_count < 3:
            raise ValueError("sample_count must be an integer of at least three")
        if self.invariance_tolerance == 0.0:
            raise ValueError("invariance_tolerance must be positive")
        if self.lower_diameter > self.upper_diameter:
            raise ValueError("lower_diameter cannot exceed upper_diameter")
        if self.decision != _decision(
            self.lower_diameter,
            self.upper_diameter,
            self.invariance_tolerance,
        ):
            raise ValueError("decision is inconsistent with the diameter interval")

    @property
    def update_admitted(self) -> bool:
        return self.decision == "certified-invariant"


def certify_uniformly_sampled_periodic_query(
    values: ArrayLike,
    *,
    angular_lipschitz_bound: float,
    invariance_tolerance: float,
) -> SampledPeriodicOrbitCertificate:
    """Certify a periodic query using a uniform grid and a Lipschitz bound.

    For ``K`` samples on ``[0, 2*pi)``, every angle lies within ``pi/K`` of a
    sample.  If the query is ``L``-Lipschitz in angle, its complete diameter is
    between the sampled diameter and ``sampled_diameter + 2*L*pi/K``.
    """

    points = _finite_array(values, name="values")
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] == 0:
        raise ValueError("values must have shape (sample_count>=3, dimension>=1)")
    lipschitz = _nonnegative_scalar(
        angular_lipschitz_bound,
        name="angular_lipschitz_bound",
    )
    tolerance = _positive_scalar(
        invariance_tolerance,
        name="invariance_tolerance",
    )
    sample_count = int(points.shape[0])
    cover_radius = float(np.pi / sample_count)
    sampled = sampled_pairwise_diameter(points)
    lower = sampled
    upper = sampled + 2.0 * lipschitz * cover_radius
    return SampledPeriodicOrbitCertificate(
        sampled_diameter=sampled,
        lower_diameter=lower,
        upper_diameter=upper,
        lipschitz_bound=lipschitz,
        angular_cover_radius=cover_radius,
        sample_count=sample_count,
        invariance_tolerance=tolerance,
        decision=_decision(lower, upper, tolerance),
    )
