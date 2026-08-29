"""Exact shared point/query moments for a declared axial gauge ambiguity.

Experimental, conditional geometric kernel; not a provider-v2 exporter or a
complete Sim(3) posterior. An exact stabilizer must be justified from the
measurement model, not inferred from a numerically small Hessian eigenvalue.
The angular law is supplied by the caller and is never fitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_ROUNDOFF = 128.0 * np.finfo(np.float64).eps


def _array(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _vector(value: ArrayLike, size: int, *, name: str) -> FloatArray:
    result = _array(value, name=name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


def _points(value: ArrayLike, *, name: str = "points") -> FloatArray:
    result = _array(value, name=name)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError(f"{name} must have nonempty shape (N, 3)")
    return result


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CircularMoments2:
    """Mean/covariance of (cos(theta), sin(theta)), not a full angular density.

    These are equivalent to the first two complex trigonometric moments. The
    centered representation avoids catastrophic subtraction for narrow wrapped
    normals. Matching these moments does not specify tails or credible regions.
    """

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = _vector(self.mean, 2, name="circular mean")
        covariance = _array(self.covariance, name="circular covariance")
        if covariance.shape != (2, 2):
            raise ValueError("circular covariance must have shape (2, 2)")
        if not np.allclose(covariance, covariance.T, atol=_ROUNDOFF, rtol=0.0):
            raise ValueError("circular covariance must be symmetric")
        symmetric = 0.5 * (covariance + covariance.T)
        if float(np.linalg.eigvalsh(symmetric)[0]) < -_ROUNDOFF:
            raise ValueError("circular covariance must be positive semidefinite")
        if not np.isclose(float(np.trace(symmetric) + mean @ mean), 1.0, atol=_ROUNDOFF, rtol=0.0):
            raise ValueError("circular moments must satisfy E[cos^2 + sin^2] = 1")
        symmetric.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", symmetric)

    @classmethod
    def uniform(cls) -> CircularMoments2:
        return cls(np.zeros(2), 0.5 * np.eye(2))

    @classmethod
    def wrapped_normal(cls, mean_radians: float, variance_radians2: float) -> CircularMoments2:
        """Exact moments of theta = Normal(mean, variance) modulo 2*pi."""
        mean = _finite_scalar(mean_radians, name="mean_radians")
        variance = _finite_scalar(variance_radians2, name="variance_radians2")
        if variance < 0.0:
            raise ValueError("variance_radians2 must be nonnegative")
        cosine, sine = float(np.cos(mean)), float(np.sin(mean))
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        attenuation = float(np.exp(-0.5 * variance))
        # expm1 retains the O(variance^2) radial term for narrow laws.
        radial = 0.5 * float(-np.expm1(-variance)) ** 2
        tangential = 0.5 * float(-np.expm1(-variance)) * (1.0 + float(np.exp(-variance)))
        covariance = rotation @ np.diag([radial, tangential]) @ rotation.T
        return cls(attenuation * np.array([cosine, sine]), covariance)

    @classmethod
    def from_atoms(cls, angles: ArrayLike, weights: ArrayLike) -> CircularMoments2:
        """Exact moments of a finite angular law; weights need not sum to one."""
        theta = _array(angles, name="angles")
        mass = _array(weights, name="weights")
        if theta.ndim != 1 or theta.size == 0 or mass.shape != theta.shape:
            raise ValueError("angles and weights must have the same nonempty 1D shape")
        if np.any(mass < 0.0) or float(np.max(mass)) <= 0.0:
            raise ValueError("weights must be nonnegative with positive total mass")
        scaled = mass / float(np.max(mass))
        probability = scaled / float(np.sum(scaled))
        values = np.column_stack((np.cos(theta), np.sin(theta)))
        mean = probability @ values
        centered = values - mean
        covariance = (centered.T * probability) @ centered
        return cls(mean, covariance)

    @property
    def first_moment(self) -> complex:
        return complex(float(self.mean[0]), float(self.mean[1]))

    @property
    def second_moment(self) -> complex:
        raw = self.covariance + np.outer(self.mean, self.mean)
        return complex(float(raw[0, 0] - raw[1, 1]), float(2.0 * raw[0, 1]))

    @property
    def covariance_root(self) -> FloatArray:
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
        root.setflags(write=False)
        return root


@dataclass(frozen=True)
class AxialQueryMoments:
    """Exact moments and componentwise full-orbit bounds for a linear query.

    Bounds describe all angles in [0, 2*pi), not a credible region and not
    necessarily the support of the caller's angular law. The covariance is only
    the shared axial contribution, without point noise or other gauge terms.
    """

    mean: FloatArray
    shared_factors: FloatArray
    orbit_center: FloatArray
    cosine_coefficients: FloatArray
    sine_coefficients: FloatArray

    def __post_init__(self) -> None:
        mean = _array(self.mean, name="query mean")
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("query mean must be a nonempty vector")
        factors = _array(self.shared_factors, name="query shared_factors")
        if factors.shape != (mean.size, 2):
            raise ValueError("query shared_factors must have shape (Q, 2)")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "shared_factors", factors)
        for name in ("orbit_center", "cosine_coefficients", "sine_coefficients"):
            object.__setattr__(self, name, _vector(getattr(self, name), mean.size, name=name))

    @property
    def covariance(self) -> FloatArray:
        return _array(self.shared_factors @ self.shared_factors.T, name="query covariance")

    @property
    def orbit_amplitude(self) -> FloatArray:
        return _array(
            np.hypot(self.cosine_coefficients, self.sine_coefficients), name="orbit amplitude"
        )

    @property
    def full_orbit_bounds(self) -> FloatArray:
        amplitude = self.orbit_amplitude
        return _array(
            np.column_stack((self.orbit_center - amplitude, self.orbit_center + amplitude)),
            name="full orbit bounds",
        )


@dataclass(frozen=True)
class AxialPointMoments:
    """N point means and one shared rank-at-most-two covariance factor.

    Flatten ``shared_factors`` to (3*N, 2); its product with its transpose is the
    complete joint axial covariance. Never treat its rows as independent noise.
    """

    mean: FloatArray
    shared_factors: FloatArray
    orbit_center: FloatArray
    cosine_coefficients: FloatArray
    sine_coefficients: FloatArray

    def __post_init__(self) -> None:
        mean = _points(self.mean, name="point mean")
        factors = _array(self.shared_factors, name="point shared_factors")
        if factors.shape != (*mean.shape, 2):
            raise ValueError("point shared_factors must have shape (N, 3, 2)")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "shared_factors", factors)
        for name in ("orbit_center", "cosine_coefficients", "sine_coefficients"):
            value = _points(getattr(self, name), name=name)
            if value.shape != mean.shape:
                raise ValueError(f"{name} must match point mean shape")
            object.__setattr__(self, name, value)

    @property
    def marginal_covariance(self) -> FloatArray:
        return _array(
            np.einsum("nik,njk->nij", self.shared_factors, self.shared_factors),
            name="marginal covariance",
        )

    def project(self, weights: ArrayLike) -> AxialQueryMoments:
        """Project q_d = sum_{n,c} weights[d,n,c] * point[n,c].

        Shape (N, 3) denotes one scalar query; (Q, N, 3) denotes Q queries.
        Point ordering/identity and any additional covariance are caller-owned.
        """
        matrix = _array(weights, name="query weights")
        if matrix.ndim == 2:
            matrix = matrix[None, :, :]
        if matrix.ndim != 3 or matrix.shape[0] == 0 or matrix.shape[1:] != self.mean.shape:
            raise ValueError("query weights must have shape (N, 3) or (Q, N, 3)")
        return AxialQueryMoments(
            mean=np.einsum("qnc,nc->q", matrix, self.mean),
            shared_factors=np.einsum("qnc,nck->qk", matrix, self.shared_factors),
            orbit_center=np.einsum("qnc,nc->q", matrix, self.orbit_center),
            cosine_coefficients=np.einsum("qnc,nc->q", matrix, self.cosine_coefficients),
            sine_coefficients=np.einsum("qnc,nc->q", matrix, self.sine_coefficients),
        )


@dataclass(frozen=True)
class AxialGaugeOrbit:
    """Declared rotations about a fixed world-frame line through ``pivot``.

    The observable quotient (axis, pivot, scale and reference points) is held
    fixed. For an uncertain quotient use conditional angular moments and the
    law of total covariance; adding independent covariance is not valid when
    that quotient and the angle are correlated.
    """

    axis: FloatArray
    pivot: FloatArray

    def __post_init__(self) -> None:
        axis = _vector(self.axis, 3, name="axis")
        pivot = _vector(self.pivot, 3, name="pivot")
        magnitude = float(np.max(np.abs(axis)))
        if magnitude == 0.0:
            raise ValueError("axis must be nonzero")
        scaled = axis / magnitude
        unit = scaled / float(np.linalg.norm(scaled))
        unit.setflags(write=False)
        object.__setattr__(self, "axis", unit)
        object.__setattr__(self, "pivot", pivot)

    def coefficients(self, points: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        reference = _points(points)
        offset = reference - self.pivot
        parallel = np.outer(offset @ self.axis, self.axis)
        cosine = offset - parallel
        sine = np.cross(self.axis, cosine)
        return (
            _array(self.pivot + parallel, name="orbit center"),
            _array(cosine, name="cosine coefficients"),
            _array(sine, name="sine coefficients"),
        )

    def support_orbit_diameter(self, support_points: ArrayLike) -> FloatArray:
        """Per-point maximum movement over the full orbit, equal to 2*radius.

        Exact collinear point support has zero diameter. A small nonzero value
        is not a proof of an exact likelihood symmetry.
        """
        _, cosine, _ = self.coefficients(support_points)
        return _array(2.0 * np.linalg.norm(cosine, axis=1), name="support orbit diameter")

    def point_moments(self, points: ArrayLike, angular: CircularMoments2) -> AxialPointMoments:
        if not isinstance(angular, CircularMoments2):
            raise TypeError("angular must be CircularMoments2")
        center, cosine, sine = self.coefficients(points)
        design = np.stack((cosine, sine), axis=-1)
        mean = center + design @ angular.mean
        factors = design @ angular.covariance_root
        return AxialPointMoments(mean, factors, center, cosine, sine)
