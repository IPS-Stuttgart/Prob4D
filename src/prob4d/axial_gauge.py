"""Nonlinear query laws for an exactly unobserved rotation about a line.

This experimental kernel is conditional on an observable gauge/geometry. It
preserves a caller-supplied conditional angular law and can update it under an
explicit positional likelihood. It does not complete a physical belief or
authorize a real-provider update. Circular
quadrature is shared across all query coordinates, retaining cross-point
uncertainty. A nearly collinear cloud is not silently treated as an exact
symmetry. See ``docs/axial-gauge-query.md`` for the statistical boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def _array(value: object, *, name: str, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _mass(value: object, *, name: str) -> FloatArray:
    result = _array(value, name=name, ndim=1).copy()
    if not result.size or np.any(result < 0.0) or not np.any(result > 0.0):
        raise ValueError(f"{name} must contain nonnegative mass with positive total")
    result /= np.max(result)
    result /= np.sum(result)
    result.setflags(write=False)
    return result


def _covariance(value: object, *, dimension: int) -> FloatArray:
    result = _array(value, name="noise_covariance", ndim=2).copy()
    if result.shape != (dimension, dimension):
        raise ValueError("noise_covariance has the wrong query dimension")
    if not np.allclose(result, result.T, rtol=1e-12, atol=1e-12):
        raise ValueError("noise_covariance must be symmetric")
    result = (result + result.T) / 2.0
    if np.min(np.linalg.eigvalsh(result)) < 0.0:
        raise ValueError("noise_covariance must be positive semidefinite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CircularQuadrature:
    """A finite angular law; weights are masses, not unweighted density values."""

    angles: FloatArray
    weights: FloatArray

    def __post_init__(self) -> None:
        angles = _array(self.angles, name="angles", ndim=1)
        weights = _mass(self.weights, name="weights")
        if angles.shape != weights.shape:
            raise ValueError("angles and weights must have the same nonempty shape")
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "weights", weights)

    def moment(self, order: int) -> complex:
        """Return the trigonometric moment E[exp(i * order * angle)]."""
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise TypeError("order must be an integer")
        return complex(np.sum(self.weights * np.exp(1j * int(order) * self.angles)))


@dataclass(frozen=True)
class GaussianQueryMixture:
    """Joint query atoms with a shared additive Gaussian noise covariance.

    Rows in ``atoms`` are complete query vectors, not independent point draws.
    A zero covariance represents an exact discrete pushforward, for which
    ``logpdf`` is deliberately unavailable. Noise is assumed independent of the
    gauge atom; heteroscedastic or correlated gauge/readout noise is not modeled.
    """

    atoms: FloatArray
    weights: FloatArray
    noise_covariance: FloatArray

    def __post_init__(self) -> None:
        atoms = _array(self.atoms, name="atoms", ndim=2)
        weights = _mass(self.weights, name="weights")
        if atoms.shape[0] != weights.size or atoms.shape[1] < 1:
            raise ValueError("atoms must have shape (number of masses, positive dimension)")
        noise = _covariance(self.noise_covariance, dimension=atoms.shape[1])
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "noise_covariance", noise)

    @property
    def mean(self) -> FloatArray:
        return np.asarray(self.weights @ self.atoms, dtype=np.float64)

    @property
    def covariance(self) -> FloatArray:
        centered = self.atoms - self.mean
        result = centered.T @ (self.weights[:, None] * centered) + self.noise_covariance
        return np.asarray((result + result.T) / 2.0, dtype=np.float64)

    def logpdf(self, observations: FloatArray, *, batch_size: int = 256) -> FloatArray:
        """Evaluate the continuous mixture density in bounded-memory batches."""
        values = _array(observations, name="observations", ndim=2)
        if values.shape[1] != self.atoms.shape[1]:
            raise ValueError("observations have the wrong query dimension")
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            cholesky = np.linalg.cholesky(self.noise_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("logpdf requires positive-definite readout noise") from exc
        inverse_root = np.linalg.solve(cholesky, np.eye(cholesky.shape[0]))
        positive = self.weights > 0.0
        atoms = self.atoms[positive] @ inverse_root.T
        logweights = np.log(self.weights[positive])
        constant = (
            self.atoms.shape[1] * np.log(2.0 * np.pi)
            + 2.0 * np.sum(np.log(np.diag(cholesky)))
        ) / 2.0
        result = np.empty(values.shape[0], dtype=np.float64)
        for start in range(0, values.shape[0], batch_size):
            whitened = values[start : start + batch_size] @ inverse_root.T
            delta = whitened[:, None, :] - atoms[None, :, :]
            logcomponents = logweights - np.sum(delta * delta, axis=2) / 2.0 - constant
            maximum = np.max(logcomponents, axis=1)
            result[start : start + batch_size] = maximum + np.log(
                np.sum(np.exp(logcomponents - maximum[:, None]), axis=1)
            )
        return result

    def halfspace_probability(self, normal: FloatArray, threshold: float) -> float:
        """Return P(normal @ query > threshold), without Gaussianizing the law."""
        direction = _array(normal, name="normal", ndim=1)
        if direction.shape != (self.atoms.shape[1],) or not np.isfinite(threshold):
            raise ValueError("normal/threshold must define a finite query-dimensional halfspace")
        means = self.atoms @ direction
        variance = float(direction @ self.noise_covariance @ direction)
        if variance == 0.0:
            return float(self.weights @ (means > threshold))
        if variance < 0.0:
            raise ValueError("negative projected noise variance")
        standardized = (float(threshold) - means) / np.sqrt(2.0 * variance)
        probabilities = np.array([0.5 * erfc(float(value)) for value in standardized])
        return float(self.weights @ probabilities)


@dataclass(frozen=True)
class AxialGaugeOrbit:
    """Rotations about a fixed line in the reference/world coordinate frame."""

    center: FloatArray
    axis: FloatArray

    def __post_init__(self) -> None:
        center = _array(self.center, name="center", ndim=1)
        axis = _array(self.axis, name="axis", ndim=1)
        if center.shape != (3,) or axis.shape != (3,):
            raise ValueError("center and axis must have shape (3,)")
        length = float(np.linalg.norm(axis))
        if not np.isfinite(length) or length == 0.0:
            raise ValueError("axis must have a finite positive length")
        axis = axis / length
        axis.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "axis", axis)

    @classmethod
    def from_line(
        cls, reference_points: FloatArray, *, tolerance: float = 1e-10
    ) -> AxialGaugeOrbit:
        """Verify a nonzero exact line to a declared relative numerical tolerance.

        ``tolerance`` bounds maximum transverse residual / RMS cloud radius.
        It is not an observability threshold for a weakly curved object. Freeze
        it before outcomes and do not relax it to turn weak geometry into symmetry.
        The largest-magnitude axis coordinate is positive (deterministic sign).
        """
        points = _array(reference_points, name="reference_points", ndim=2)
        if points.shape[0] < 2 or points.shape[1] != 3:
            raise ValueError("reference_points must have shape (N>=2, 3)")
        if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
            raise ValueError("tolerance must be finite and lie strictly between zero and one")
        center = np.mean(points, axis=0)
        centered = points - center
        radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        if not np.isfinite(radius) or radius == 0.0:
            raise ValueError("a zero-extent cloud does not identify an axial stabilizer")
        _, _, right = np.linalg.svd(centered / radius, full_matrices=False)
        axis = right[0].copy()
        if axis[int(np.argmax(np.abs(axis)))] < 0.0:
            axis = -axis
        transverse = centered - np.outer(centered @ axis, axis)
        if float(np.max(np.linalg.norm(transverse, axis=1))) > tolerance * radius:
            raise ValueError("reference geometry is not an exact line at the declared tolerance")
        return cls(center=center, axis=axis)

    def _components(
        self, reference_queries: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        points = _array(reference_queries, name="reference_queries", ndim=2)
        if points.shape[0] < 1 or points.shape[1] != 3:
            raise ValueError("reference_queries must have shape (N>=1, 3)")
        relative = points - self.center
        parallel = np.outer(relative @ self.axis, self.axis)
        cosine = relative - parallel
        sine = np.cross(self.axis, cosine)
        base = self.center + parallel
        return base, cosine, sine

    def positions(self, reference_queries: FloatArray, angles: FloatArray) -> FloatArray:
        """Return (angle, point, coordinate) atoms with one shared rotation."""
        theta = _array(angles, name="angles", ndim=1)
        base, cosine, sine = self._components(reference_queries)
        return (
            base[None, :, :]
            + np.cos(theta)[:, None, None] * cosine[None, :, :]
            + np.sin(theta)[:, None, None] * sine[None, :, :]
        )

    def condition_on_correspondences(
        self,
        reference_points: FloatArray,
        observed_points: FloatArray,
        angular_prior: CircularQuadrature,
        *,
        noise_covariance: FloatArray,
    ) -> CircularQuadrature:
        """Bayes-update the angle using a complete, fixed positional likelihood.

        Geometry is conditional/fixed. The covariance is for the point-major
        stacked observation residual and may include cross-point dependence.
        Exact-line observations leave the prior unchanged. Off-axis or weakly
        curved observations may inform the angle and must not be discarded by
        declaring a symmetry from a local rank threshold. Source uncertainty or
        angle-dependent covariance requires a different likelihood.
        """
        observed = _array(observed_points, name="observed_points", ndim=2)
        predicted = self.positions(reference_points, angular_prior.angles)
        if observed.shape != predicted.shape[1:]:
            raise ValueError("observed_points must match the reference point shape")
        covariance = _covariance(noise_covariance, dimension=observed.size)
        try:
            root = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("conditioning requires positive-definite residual noise") from exc
        residuals = observed.reshape(1, -1) - predicted.reshape(predicted.shape[0], -1)
        whitened = np.linalg.solve(root, residuals.T).T
        log_likelihood = -0.5 * np.sum(whitened * whitened, axis=1)
        positive = angular_prior.weights > 0.0
        log_mass = np.full(angular_prior.weights.shape, -np.inf)
        log_mass[positive] = np.log(angular_prior.weights[positive]) + log_likelihood[positive]
        maximum = float(np.max(log_mass))
        if not np.isfinite(maximum):
            raise ValueError("correspondence likelihood has no finite prior-supported mass")
        return CircularQuadrature(angular_prior.angles, np.exp(log_mass - maximum))

    def moments(
        self, reference_queries: FloatArray, angular_law: CircularQuadrature
    ) -> tuple[FloatArray, FloatArray]:
        """Exact finite-law joint moments from only the first two harmonics.

        The mean is (N, 3); covariance uses point-major flattened coordinates.
        This is a moment calculation, not a claim that the query is Gaussian.
        """
        base, cosine, sine = self._components(reference_queries)
        first, second = angular_law.moment(1), angular_law.moment(2)
        mean = base + first.real * cosine + first.imag * sine
        coefficient_covariance = np.array(
            [
                [
                    (1.0 + second.real) / 2.0 - first.real**2,
                    second.imag / 2.0 - first.real * first.imag,
                ],
                [
                    second.imag / 2.0 - first.real * first.imag,
                    (1.0 - second.real) / 2.0 - first.imag**2,
                ],
            ]
        )
        basis = np.column_stack((cosine.reshape(-1), sine.reshape(-1)))
        covariance = basis @ coefficient_covariance @ basis.T
        return mean, (covariance + covariance.T) / 2.0

    def pushforward(
        self,
        reference_queries: FloatArray,
        angular_law: CircularQuadrature,
        *,
        noise_covariance: FloatArray | None = None,
    ) -> GaussianQueryMixture:
        """Retain the full shared-angle query law, optionally with readout noise."""
        positions = self.positions(reference_queries, angular_law.angles)
        dimension = 3 * positions.shape[1]
        noise = np.zeros((dimension, dimension)) if noise_covariance is None else noise_covariance
        return GaussianQueryMixture(
            atoms=positions.reshape(positions.shape[0], dimension),
            weights=angular_law.weights,
            noise_covariance=noise,
        )
