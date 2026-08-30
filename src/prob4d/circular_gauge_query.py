"""Experimental nonlinear query propagation for a declared circular gauge.

This is NOT a replacement for the Sim(3) estimator or a provider admission gate.
The caller must supply the CONDITIONAL circular prior at a fixed observable
quotient and justify a common, global one-axis rotation symmetry. A local
rank-deficient Jacobian alone does not establish that symmetry.

For q(phi) = c + a*cos(phi) + b*sin(phi), moments are analytic. Joint events
q_j(phi) > 0 are unions of circular arcs sharing ONE phi. Wrapped-normal arc
probabilities retain an explicit omitted-normal-tail bound. That bound excludes
floating-point rounding, model error, and uncertainty in the fixed quotient.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, erfc, erf, floor, fsum, hypot, pi, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_TAU = 2.0 * pi
_SQRT2 = sqrt(2.0)


def _vector(value: Any, name: str, *, allow_empty: bool = False) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim != 1 or (result.size == 0 and not allow_empty):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else ''}vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar, not a boolean or array")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class CircularPrior:
    """An explicit mixture of wrapped normals and a uniform circular density.

    Positive standard deviations ensure absolute continuity, so isolated arc
    endpoints have zero probability. Weights must already sum to one; the only
    normalization performed is removal of accepted floating-point roundoff.
    ``prior_id`` is caller provenance, not a calibration or validation claim.
    """

    weights: FloatArray
    means: FloatArray
    stddevs: FloatArray
    uniform_weight: float
    prior_id: str

    def __post_init__(self) -> None:
        weights = _vector(self.weights, "weights", allow_empty=True)
        means = _vector(self.means, "means", allow_empty=True)
        stddevs = _vector(self.stddevs, "stddevs", allow_empty=True)
        if weights.shape != means.shape or weights.shape != stddevs.shape:
            raise ValueError("weights, means, and stddevs must have identical shapes")
        if np.any(weights < 0.0) or np.any(stddevs <= 0.0):
            raise ValueError("weights must be nonnegative and stddevs strictly positive")
        uniform = _finite_scalar(self.uniform_weight, "uniform_weight")
        if not 0.0 <= uniform <= 1.0:
            raise ValueError("uniform_weight must lie in [0, 1]")
        total = fsum([uniform, *weights.tolist()])
        if abs(total - 1.0) > 1e-12:
            raise ValueError("complete prior weights must sum to one")
        if not isinstance(self.prior_id, str) or not self.prior_id.strip():
            raise ValueError("prior_id must be a nonempty string")
        weights = _vector(weights / total, "weights", allow_empty=True)
        means = _vector(np.remainder(means, _TAU), "means", allow_empty=True)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "stddevs", stddevs)
        object.__setattr__(self, "uniform_weight", uniform / total)

    @classmethod
    def wrapped_normal(cls, mean: float, stddev: float, *, prior_id: str) -> CircularPrior:
        return cls(np.array([1.0]), np.array([mean]), np.array([stddev]), 0.0, prior_id)

    @classmethod
    def uniform(cls, *, prior_id: str) -> CircularPrior:
        return cls(np.empty(0), np.empty(0), np.empty(0), 1.0, prior_id)

    def trigonometric_moment(self, order: int) -> complex:
        """Return E[exp(i*order*phi)]; negative orders use conjugacy."""
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
            raise ValueError("order must be an integer")
        if order == 0:
            return 1.0 + 0.0j
        if abs(order) > 1_000_000:
            raise ValueError("order exceeds the supported numerical range")
        with np.errstate(over="ignore", under="ignore"):
            magnitude = np.exp(-0.5 * (float(order) * self.stddevs) ** 2)
        return complex(np.sum(self.weights * magnitude * np.exp(1j * order * self.means)))

    def trigonometric_mean_covariance(self) -> tuple[FloatArray, FloatArray]:
        """Stable moments of [cos(phi), sin(phi)], including narrow priors.

        Component-centered formulas and the law of total covariance avoid
        cancellation in Var(cos(phi)) as a wrapped-normal variance tends to zero.
        """
        component_means: list[FloatArray] = []
        component_covariances: list[FloatArray] = []
        component_weights: list[float] = []
        if self.uniform_weight:
            component_means.append(np.zeros(2))
            component_covariances.append(0.5 * np.eye(2))
            component_weights.append(self.uniform_weight)
        for weight, mean, stddev in zip(self.weights, self.means, self.stddevs):
            if weight == 0.0:
                continue
            with np.errstate(over="ignore", under="ignore"):
                variance = float(np.square(stddev))
                rho = float(np.exp(-0.5 * variance))
                var_cos = 0.5 * float(-np.expm1(-variance)) ** 2
                var_sin = 0.5 * float(-np.expm1(-2.0 * variance))
            cosine, sine = float(np.cos(mean)), float(np.sin(mean))
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            component_means.append(rho * rotation[:, 0])
            component_covariances.append(rotation @ np.diag([var_cos, var_sin]) @ rotation.T)
            component_weights.append(float(weight))
        result_mean = sum(w * m for w, m in zip(component_weights, component_means))
        result_covariance = np.zeros((2, 2))
        for weight, mean, covariance in zip(
            component_weights, component_means, component_covariances
        ):
            delta = mean - result_mean
            result_covariance += weight * (covariance + np.outer(delta, delta))
        result_mean = np.asarray(result_mean, dtype=np.float64)
        result_covariance = 0.5 * (result_covariance + result_covariance.T)
        result_mean.setflags(write=False)
        result_covariance.setflags(write=False)
        return result_mean, result_covariance


@dataclass(frozen=True)
class QueryMoments:
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = _vector(self.mean, "mean")
        covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        if covariance.shape != (mean.size, mean.size) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be a finite square matrix matching the mean")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
            raise ValueError("covariance must be symmetric")
        scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-10 * scale:
            raise ValueError("covariance must be positive semidefinite")
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class QueryMomentFactor:
    """Linear-memory exact moment representation; covariance = factor @ factor.T.

    The two columns are harmonic covariance modes, not two independent phases.
    """

    mean: FloatArray
    factor: FloatArray

    def __post_init__(self) -> None:
        mean = _vector(self.mean, "mean")
        factor = np.asarray(self.factor, dtype=np.float64).copy()
        if factor.shape != (mean.size, 2) or not np.all(np.isfinite(factor)):
            raise ValueError("factor must be finite with shape (query_dimension, 2)")
        factor.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "factor", factor)

    @property
    def marginal_variance(self) -> FloatArray:
        result = np.sum(self.factor**2, axis=1)
        result.setflags(write=False)
        return result

    def dense(self, *, maximum_dimension: int = 2048) -> QueryMoments:
        maximum_dimension = _positive_integer(maximum_dimension, "maximum_dimension")
        if self.mean.size > maximum_dimension:
            raise ValueError("dense covariance exceeds maximum_dimension; retain the factor")
        return QueryMoments(self.mean, self.factor @ self.factor.T)


@dataclass(frozen=True)
class AffineCircularQuery:
    """A complete, jointly rotated query q(phi)=offset+cosine*cos(phi)+sine*sin(phi).

    Every output coordinate shares the same phase. The query can stack points,
    frames, or linear clearance constraints. Those are not independent samples.
    """

    offset: FloatArray
    cosine: FloatArray
    sine: FloatArray

    def __post_init__(self) -> None:
        for name in ("offset", "cosine", "sine"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if self.offset.shape != self.cosine.shape or self.offset.shape != self.sine.shape:
            raise ValueError("offset, cosine, and sine must have identical shapes")

    def evaluate(self, phase: Any) -> FloatArray:
        phase_array = np.asarray(phase, dtype=np.float64)
        if not np.all(np.isfinite(phase_array)):
            raise ValueError("phase must be finite")
        return (
            self.offset
            + np.cos(phase_array)[..., None] * self.cosine
            + np.sin(phase_array)[..., None] * self.sine
        )

    def project(self, matrix: Any, *, offset: Any | None = None) -> AffineCircularQuery:
        projection = np.asarray(matrix, dtype=np.float64)
        if (
            projection.ndim != 2
            or projection.shape[0] < 1
            or projection.shape[1] != self.offset.size
            or not np.all(np.isfinite(projection))
        ):
            raise ValueError("matrix must be finite with shape (Q, original_dimension)")
        shift = np.zeros(projection.shape[0]) if offset is None else _vector(offset, "offset")
        if shift.shape != (projection.shape[0],):
            raise ValueError("offset must match the projected dimension")
        return AffineCircularQuery(
            projection @ self.offset + shift,
            projection @ self.cosine,
            projection @ self.sine,
        )

    def low_rank_moments(self, prior: CircularPrior) -> QueryMomentFactor:
        """Exact shared-phase moments in O(query_dimension) storage."""
        mean, covariance = prior.trigonometric_mean_covariance()
        mapping = np.column_stack((self.cosine, self.sine))
        values, vectors = np.linalg.eigh(covariance)
        root = vectors * np.sqrt(np.maximum(values, 0.0))
        return QueryMomentFactor(self.offset + mapping @ mean, mapping @ root)

    def moments(self, prior: CircularPrior) -> QueryMoments:
        """Dense conditional moments for small queries; not a Gaussian assertion."""
        return self.low_rank_moments(prior).dense()


def point_rotation_orbit(
    points: Any, *, axis_origin: Any, axis_direction: Any
) -> AffineCircularQuery:
    """Build a stacked point orbit by Rodrigues' formula in a declared frame.

    The supplied points are already conditional on all observable quotient
    coordinates. This function does not fit Sim(3) or prove a gauge symmetry.
    """
    points_array = np.asarray(points, dtype=np.float64)
    if (
        points_array.ndim != 2
        or points_array.shape[0] < 1
        or points_array.shape[1] != 3
        or not np.all(np.isfinite(points_array))
    ):
        raise ValueError("points must be a finite, nonempty (N, 3) matrix")
    origin = _vector(axis_origin, "axis_origin")
    direction = _vector(axis_direction, "axis_direction")
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("axis origin and direction must have three coordinates")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0 or not np.isfinite(direction_norm):
        raise ValueError("axis_direction must have a finite nonzero norm")
    direction = direction / direction_norm
    relative = points_array - origin
    parallel = np.outer(relative @ direction, direction)
    perpendicular = relative - parallel
    return AffineCircularQuery(
        (origin + parallel).reshape(-1),
        perpendicular.reshape(-1),
        np.cross(direction, perpendicular).reshape(-1),
    )


def validate_declared_line_support(
    points: Any, *, axis_origin: Any, axis_direction: Any, tolerance: float = 1e-12
) -> float:
    """Check line geometry only; return its maximum transverse residual.

    A local nullspace is not an admissible substitute for this geometric
    premise. Even this check does not prove that a provider likelihood, a
    physical model, or a conditional prior has the required global symmetry.
    """
    tolerance = _finite_scalar(tolerance, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    orbit = point_rotation_orbit(points, axis_origin=axis_origin, axis_direction=axis_direction)
    residual = float(np.max(np.linalg.norm(orbit.cosine.reshape(-1, 3), axis=1)))
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[0] < 2 or float(np.linalg.norm(np.ptp(point_array, axis=0))) <= tolerance:
        raise ValueError("line support must contain at least two separated points")
    if residual > tolerance:
        raise ValueError("declared support is not on the rotation axis")
    return residual


def violation_arcs(query: AffineCircularQuery) -> tuple[tuple[float, float], ...]:
    """Disjoint intervals for the joint event ANY q_j(phi)>0, on [0,2*pi].

    Endpoints have zero mass under every supported prior. Geometric roundoff
    near exact tangencies is not included in the subsequent normal-tail bound.
    """
    intervals: list[tuple[float, float]] = []
    for offset, cosine, sine in zip(query.offset, query.cosine, query.sine):
        amplitude = hypot(float(cosine), float(sine))
        if amplitude == 0.0:
            if offset > 0.0:
                return ((0.0, _TAU),)
            continue
        threshold = -float(offset) / amplitude
        if threshold <= -1.0:
            return ((0.0, _TAU),)
        if threshold >= 1.0:
            continue
        center = atan2(float(sine), float(cosine)) % _TAU
        width = acos(threshold)
        left, right = center - width, center + width
        if left < 0.0:
            intervals.extend(((0.0, right), (left + _TAU, _TAU)))
        elif right > _TAU:
            intervals.extend(((0.0, right - _TAU), (left, _TAU)))
        else:
            intervals.append((left, right))
    merged: list[tuple[float, float]] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return tuple(merged)


def _normal_interval(left: float, right: float) -> float:
    """Stable standard-normal mass without subtraction of two values near 1."""
    if right <= 0.0:
        return 0.5 * (erfc(-right / _SQRT2) - erfc(-left / _SQRT2))
    if left >= 0.0:
        return 0.5 * (erfc(left / _SQRT2) - erfc(right / _SQRT2))
    return 0.5 * (erf(right / _SQRT2) - erf(left / _SQRT2))


@dataclass(frozen=True)
class ProbabilityBounds:
    """Analytic probability bracket for omitted tails, not interval arithmetic."""

    lower: float
    upper: float
    omitted_tail_bound: float
    arc_count: int

    def __post_init__(self) -> None:
        lower = _finite_scalar(self.lower, "lower")
        upper = _finite_scalar(self.upper, "upper")
        bound = _finite_scalar(self.omitted_tail_bound, "omitted_tail_bound")
        if not 0.0 <= lower <= upper <= 1.0 or not 0.0 <= bound <= 1.0:
            raise ValueError("invalid probability bounds")
        if upper - lower > bound + 1e-14:
            raise ValueError("bracket width exceeds its tail bound")
        if isinstance(self.arc_count, (bool, np.bool_)) or not isinstance(
            self.arc_count, (int, np.integer)
        ) or self.arc_count < 0:
            raise ValueError("arc_count must be a nonnegative integer")


def path_violation_probability(
    query: AffineCircularQuery,
    prior: CircularPrior,
    *,
    tail_tolerance: float = 1e-12,
    max_periods: int = 100_000,
) -> ProbabilityBounds:
    """Probability that any stacked constraint fails under one shared phase.

    Exact arc geometry, exact uniform integration, and normal-CDF sums with an
    explicit omitted-tail bound. No per-frame independence approximation or
    Gaussian approximation of the transformed query is used. The fixed-quotient
    model excludes extra measurement, dynamics, and contact uncertainty.
    """
    tolerance = _finite_scalar(tail_tolerance, "tail_tolerance")
    if not 0.0 < tolerance < 0.25:
        raise ValueError("tail_tolerance must lie in (0, 0.25)")
    max_periods = _positive_integer(max_periods, "max_periods")
    arcs = violation_arcs(query)
    if not arcs:
        return ProbabilityBounds(0.0, 0.0, 0.0, 0)
    if arcs == ((0.0, _TAU),):
        return ProbabilityBounds(1.0, 1.0, 0.0, 1)
    probability_terms = [prior.uniform_weight * fsum(right - left for left, right in arcs) / _TAU]
    tail_terms: list[float] = []
    cutoff = 4.0
    while erfc(cutoff / _SQRT2) > tolerance:
        cutoff += 1.0
    for weight, mean, stddev in zip(prior.weights, prior.means, prior.stddevs):
        if weight == 0.0:
            continue
        scaled_cutoff = cutoff * float(stddev)
        if not np.isfinite(scaled_cutoff) or scaled_cutoff / _TAU > max_periods:
            raise ValueError("prior exceeds max_periods; no probability estimate was produced")
        first = floor((float(mean) - scaled_cutoff) / _TAU)
        last = floor((float(mean) + scaled_cutoff) / _TAU)
        if last - first + 1 > max_periods:
            raise ValueError("prior exceeds max_periods; no probability estimate was produced")
        component_terms = []
        for period in range(first, last + 1):
            for left, right in arcs:
                component_terms.append(
                    _normal_interval(
                        (left + _TAU * period - mean) / stddev,
                        (right + _TAU * period - mean) / stddev,
                    )
                )
        lower_cut = (first * _TAU - mean) / stddev
        upper_cut = ((last + 1) * _TAU - mean) / stddev
        tail = 0.5 * erfc(-lower_cut / _SQRT2) + 0.5 * erfc(upper_cut / _SQRT2)
        probability_terms.append(float(weight) * fsum(component_terms))
        tail_terms.append(float(weight) * tail)
    lower = float(np.clip(fsum(probability_terms), 0.0, 1.0))
    tail_bound = float(np.clip(fsum(tail_terms), 0.0, 1.0))
    upper = min(1.0, lower + tail_bound)
    return ProbabilityBounds(lower, upper, tail_bound, len(arcs))


def bounded_risk_admissible(bounds: ProbabilityBounds, *, maximum_risk: float) -> bool:
    """Model-conditional decision only; the consumer owns complete-belief fallback."""
    risk = _finite_scalar(maximum_risk, "maximum_risk")
    if not 0.0 <= risk <= 1.0:
        raise ValueError("maximum_risk must lie in [0, 1]")
    return bounds.upper <= risk


__all__ = [
    "AffineCircularQuery",
    "CircularPrior",
    "ProbabilityBounds",
    "QueryMoments",
    "QueryMomentFactor",
    "bounded_risk_admissible",
    "path_violation_probability",
    "point_rotation_orbit",
    "validate_declared_line_support",
    "violation_arcs",
]
