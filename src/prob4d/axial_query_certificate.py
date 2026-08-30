"""Finite-angle query bounds for a shared, unresolved axial gauge.

This is an analytic, conditional-model certificate, not a covariance estimate
or a provider-admission rule.  An affine query of points rotated about one
fixed line has the form ``c + a*cos(theta) + b*sin(theta)``.  Keeping that
shared angle permits exact action comparisons without completing a deficient
Sim(3) information matrix.  A zero derivative at one representative does not
imply that the query is constant on the orbit.

The caller must establish that the admitted uncertainty is contained in the
specified orbit, angle arc, and uniform advantage-error envelope.  A generic
rank-six Jacobian, a near-collinear cloud, or an arbitrary nonlinear physical
query does not establish those assumptions.  Rejected certificates must be
routed to the caller's original complete fallback belief by BayesianPhysTwin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
OrbitKey: TypeAlias = tuple[str, tuple[float, ...]]
_SCOPE = "conditional-shared-axial-orbit-affine-query-v1"


def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _gauge_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("shared_gauge_id must be an explicit nonempty string")
    return value


def _points(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), N positive")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    # A bytes-backed array cannot have its write flag re-enabled by a caller.
    return np.frombuffer(result.tobytes(), dtype=np.float64)


@dataclass(frozen=True)
class AngleArc:
    """Closed circular arc, parameterized as center +/- half_width radians.

    The full circle is the default.  An arc is a supplied support assumption,
    not a confidence level.  ``None`` represents an infeasible/empty arc in
    the functions below; it must never yield vacuous acceptance.
    """

    center: float = 0.0
    half_width: float = math.pi

    def __post_init__(self) -> None:
        center = _scalar(self.center, "center")
        width = _scalar(self.half_width, "half_width", nonnegative=True)
        if width > math.pi:
            raise ValueError("half_width must not exceed pi")
        object.__setattr__(self, "center", math.remainder(center, 2.0 * math.pi))
        object.__setattr__(self, "half_width", width)

    def contains(self, angle: float, *, atol: float = 0.0) -> bool:
        angle = _scalar(angle, "angle")
        tolerance = _scalar(atol, "atol", nonnegative=True)
        displacement = math.remainder(angle - self.center, 2.0 * math.pi)
        return abs(displacement) <= self.half_width + tolerance


_FULL_CIRCLE = AngleArc()


@dataclass(frozen=True)
class HarmonicQuery:
    """Scalar affine query on one declared, shared axial orbit.

    ``orbit_key`` binds both the shared latent identity and coordinate convention.
    Matching axis geometry alone does not establish shared uncertainty.
    """

    constant: float
    cosine: float
    sine: float
    orbit_key: OrbitKey

    def __post_init__(self) -> None:
        for name in ("constant", "cosine", "sine"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if not isinstance(self.orbit_key, tuple) or len(self.orbit_key) != 2:
            raise ValueError("orbit_key must bind a shared identity and geometry")
        identity, geometry = self.orbit_key
        identity = _gauge_id(identity)
        if len(geometry) != 6:
            raise ValueError("orbit geometry must contain an origin and an axis")
        key = tuple(_scalar(v, "orbit_key entry") for v in geometry)
        if not math.isclose(math.hypot(*key[3:]), 1.0, abs_tol=1e-12):
            raise ValueError("orbit_key must contain a unit axis")
        object.__setattr__(self, "orbit_key", (identity, key))

    @property
    def amplitude(self) -> float:
        value = math.hypot(self.cosine, self.sine)
        if not math.isfinite(value):
            raise ValueError("query amplitude overflowed")
        return value

    @property
    def derivative_at_zero(self) -> float:
        return self.sine

    def evaluate(self, angle: float) -> float:
        angle = _scalar(angle, "angle")
        value = math.fsum(
            (self.constant, self.cosine * math.cos(angle), self.sine * math.sin(angle))
        )
        if not math.isfinite(value):
            raise ValueError("query evaluation overflowed")
        return value

    def minus(self, other: HarmonicQuery) -> HarmonicQuery:
        """Subtract before bounding, preserving shared-angle cancellation."""
        if not isinstance(other, HarmonicQuery):
            raise TypeError("other must be a HarmonicQuery")
        if self.orbit_key != other.orbit_key:
            raise ValueError("queries must use the same shared orbit")
        return HarmonicQuery(
            self.constant - other.constant,
            self.cosine - other.cosine,
            self.sine - other.sine,
            self.orbit_key,
        )

    def bounds(self, arc: AngleArc = _FULL_CIRCLE) -> QueryBounds:
        """Analytic extrema, including interior stationary angles.

        Endpoints alone are insufficient when an arc crosses an extremum.
        Floating-point results are numerical evaluations of the exact formula,
        not an interval-arithmetic verification of every rounding operation.
        """
        if not isinstance(arc, AngleArc):
            raise TypeError("arc must be a nonempty AngleArc")
        radius = self.amplitude
        if radius == 0.0:
            return QueryBounds(self.constant, self.constant, arc.center, arc.center)
        if arc.half_width == math.pi:
            maximum_angle = math.atan2(self.sine, self.cosine)
            minimum_angle = math.remainder(maximum_angle + math.pi, 2.0 * math.pi)
            return QueryBounds(
                self.constant - radius,
                self.constant + radius,
                minimum_angle,
                maximum_angle,
            )
        candidates = [arc.center - arc.half_width, arc.center + arc.half_width]
        maximum_angle = math.atan2(self.sine, self.cosine)
        minimum_angle = math.remainder(maximum_angle + math.pi, 2.0 * math.pi)
        for angle in (minimum_angle, maximum_angle):
            if arc.contains(angle):
                candidates.append(angle)
        values = [self.evaluate(angle) for angle in candidates]
        lower_index = int(np.argmin(values))
        upper_index = int(np.argmax(values))
        return QueryBounds(
            values[lower_index],
            values[upper_index],
            candidates[lower_index],
            candidates[upper_index],
        )


@dataclass(frozen=True)
class QueryBounds:
    """Numerical analytic extrema and angles attaining them."""

    lower: float
    upper: float
    lower_angle: float
    upper_angle: float

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "lower_angle", "upper_angle"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")


@dataclass(frozen=True, eq=False)
class AxialRotationOrbit:
    """Rotations of representative metric points about a fixed reference line.

    Points must already be transformed by the fitted representative Sim(3).
    Constructing this object does not assert that a visual factor identifies
    the line or that the orbit exhausts a noisy seven-dimensional posterior.
    ``shared_gauge_id`` must identify the same latent angle and representative
    convention in both queries; it is never inferred from matching geometry.
    """

    origin: FloatArray
    axis: FloatArray
    shared_gauge_id: str

    def __post_init__(self) -> None:
        origin = _vector(self.origin, "origin")
        raw_axis = _vector(self.axis, "axis")
        norm = math.hypot(*(float(x) for x in raw_axis))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("axis must have finite positive norm")
        axis = _vector(raw_axis / norm, "normalized axis")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "shared_gauge_id", _gauge_id(self.shared_gauge_id))

    @property
    def key(self) -> OrbitKey:
        geometry = tuple(float(x) for x in np.concatenate((self.origin, self.axis)))
        return self.shared_gauge_id, geometry

    def transform(self, points: ArrayLike, angle: float) -> FloatArray:
        points_array = _points(points, "points")
        angle = _scalar(angle, "angle")
        centered = points_array - self.origin
        parallel = np.outer(centered @ self.axis, self.axis)
        perpendicular = centered - parallel
        transformed = (
            self.origin
            + parallel
            + math.cos(angle) * perpendicular
            + math.sin(angle) * np.cross(self.axis, perpendicular)
        )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("transformed points overflowed")
        return transformed

    def maximum_support_displacement(self, points: ArrayLike) -> float:
        """Exact full-orbit maximum point displacement from the representative.

        This is twice the largest distance to the axis.  It is zero for exact
        line support, but explicitly nonzero for near-collinear input.  There
        is no rank threshold that silently promotes a near-line to an exact
        stabilizer.
        """
        centered = _points(points, "points") - self.origin
        perpendicular = centered - np.outer(centered @ self.axis, self.axis)
        maximum = 2.0 * max(math.hypot(*(float(x) for x in row)) for row in perpendicular)
        if not math.isfinite(maximum):
            raise ValueError("support displacement overflowed")
        return maximum

    def affine_query(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        *,
        offset: float = 0.0,
    ) -> HarmonicQuery:
        """Represent ``offset + sum_j weights[j] @ rotated_points[j]``."""
        points_array = _points(points, "points")
        weights_array = _points(weights, "weights")
        if points_array.shape != weights_array.shape:
            raise ValueError("points and weights must have the same shape")
        offset = _scalar(offset, "offset")
        centered = points_array - self.origin
        parallel = np.outer(centered @ self.axis, self.axis)
        perpendicular = centered - parallel
        return HarmonicQuery(
            constant=offset + float(np.sum(weights_array * (self.origin + parallel))),
            cosine=float(np.sum(weights_array * perpendicular)),
            sine=float(np.sum(weights_array * np.cross(self.axis, perpendicular))),
            orbit_key=self.key,
        )

    def bounded_anchor_arc(
        self,
        representative_point: ArrayLike,
        observed_point: ArrayLike,
        *,
        error_radius: float,
    ) -> AngleArc | None:
        """Angles consistent with one Euclidean bounded-error metric anchor.

        Returns ``None`` for inconsistent support and a full circle for an
        uninformative feasible anchor.  The supplied radius is a deterministic
        bound; this method does not calibrate it or assign a confidence level.
        Reusing the visual observation as an independent anchor is not justified.
        """
        point = _vector(representative_point, "representative_point") - self.origin
        observed = _vector(observed_point, "observed_point") - self.origin
        radius = _scalar(error_radius, "error_radius", nonnegative=True)
        point_parallel = float(point @ self.axis)
        observed_parallel = float(observed @ self.axis)
        point_perpendicular = point - point_parallel * self.axis
        observed_perpendicular = observed - observed_parallel * self.axis
        cosine = float(point_perpendicular @ observed_perpendicular)
        sine = float(np.cross(self.axis, point_perpendicular) @ observed_perpendicular)
        amplitude = math.hypot(cosine, sine)
        constant = math.fsum(
            (
                (point_parallel - observed_parallel) ** 2,
                float(point_perpendicular @ point_perpendicular),
                float(observed_perpendicular @ observed_perpendicular),
            )
        )
        radius_squared = radius * radius
        if not all(math.isfinite(v) for v in (amplitude, constant, radius_squared)):
            raise ValueError("anchor geometry overflowed")
        if amplitude == 0.0:
            return AngleArc() if constant <= radius_squared else None
        threshold = (constant - radius_squared) / (2.0 * amplitude)
        if threshold > 1.0:
            return None
        if threshold <= -1.0:
            return AngleArc()
        return AngleArc(
            center=math.atan2(sine, cosine),
            half_width=math.acos(threshold),
        )


@dataclass(frozen=True)
class OrbitAdvantageCertificate:
    """A conditional advantage certificate; not a deployment authorization."""

    admitted: bool
    reason_codes: tuple[str, ...]
    lower_advantage: float | None
    upper_advantage: float | None
    required_margin: float
    advantage_error_bound: float
    numerical_slack: float
    scope: str = _SCOPE

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise TypeError("admitted must be a bool")
        if self.scope != _SCOPE:
            raise ValueError("certificate scope changed")
        reasons = tuple(self.reason_codes)
        allowed = {
            "orbit-model-scope-not-admitted",
            "infeasible-anchor-support",
            "nonpositive-robust-advantage",
        }
        if len(set(reasons)) != len(reasons) or any(r not in allowed for r in reasons):
            raise ValueError("invalid or duplicate rejection reasons")
        if self.admitted == bool(reasons):
            raise ValueError("admission and rejection reasons disagree")
        for name in ("required_margin", "advantage_error_bound", "numerical_slack"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, nonnegative=True))
        if (self.lower_advantage is None) != (self.upper_advantage is None):
            raise ValueError("both advantage bounds must be present or absent")
        if self.lower_advantage is None:
            if "infeasible-anchor-support" not in reasons:
                raise ValueError("missing bounds require infeasible support")
        else:
            lower = _scalar(self.lower_advantage, "lower_advantage")
            upper = _scalar(self.upper_advantage, "upper_advantage")
            if lower > upper:
                raise ValueError("advantage bounds are reversed")
            if self.admitted and not lower > self.required_margin + self.numerical_slack:
                raise ValueError("admitted certificate has no positive robust advantage")
        object.__setattr__(self, "reason_codes", reasons)


def certify_shared_orbit_advantage(
    *,
    fallback_loss: HarmonicQuery,
    candidate_loss: HarmonicQuery,
    scope_admitted: bool,
    arc: AngleArc | None = _FULL_CIRCLE,
    advantage_error_bound: float = 0.0,
    required_margin: float = 0.0,
    numerical_slack: float = 1e-12,
) -> OrbitAdvantageCertificate:
    """Admit only uniformly positive fallback-minus-candidate advantage.

    Both losses must use the same orbit.  Subtraction precedes optimization,
    preserving shared geometric uncertainty.  The uniform error bound covers
    all omitted effects on their difference, not just a fitted standard error.
    ``scope_admitted`` must come from the caller's independently justified
    source/model gate.  A false scope or empty anchor support always rejects.

    ``numerical_slack`` is an explicit absolute tolerance in loss units, not a
    statistical margin or a formal interval-arithmetic error bound.  Strict
    comparison rejects equality and numerically indistinguishable improvement.
    """
    if type(scope_admitted) is not bool:
        raise TypeError("scope_admitted must be a bool")
    error = _scalar(advantage_error_bound, "advantage_error_bound", nonnegative=True)
    margin = _scalar(required_margin, "required_margin", nonnegative=True)
    slack = _scalar(numerical_slack, "numerical_slack", nonnegative=True)
    difference = fallback_loss.minus(candidate_loss)
    reasons: list[str] = []
    if not scope_admitted:
        reasons.append("orbit-model-scope-not-admitted")
    lower: float | None = None
    upper: float | None = None
    if arc is None:
        reasons.append("infeasible-anchor-support")
    else:
        bounds = difference.bounds(arc)
        lower = _scalar(bounds.lower - error, "lower advantage")
        upper = _scalar(bounds.upper + error, "upper advantage")
        if not lower > margin + slack:
            reasons.append("nonpositive-robust-advantage")
    return OrbitAdvantageCertificate(
        admitted=not reasons,
        reason_codes=tuple(reasons),
        lower_advantage=lower,
        upper_advantage=upper,
        required_margin=margin,
        advantage_error_bound=error,
        numerical_slack=slack,
    )


__all__ = [
    "AngleArc",
    "AxialRotationOrbit",
    "HarmonicQuery",
    "OrbitAdvantageCertificate",
    "QueryBounds",
    "certify_shared_orbit_advantage",
]
