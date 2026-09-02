"""Continuous axial-symmetry query bounds with group-conformal support.

The geometric part is exact for a declared SO(2) orbit. The statistical part
calibrates a scalar nonconformity score across independent groups. It supplies a
marginal exchangeability statement for that score, not conditional coverage,
provider competence, or deployment safety.

For a vector first-harmonic query

    q(theta) = c + a cos(theta) + b sin(theta),

the weighted full-orbit diameter is exactly

    2 * sigma_max(W [a b]).

A calibrated axial tube combines a continuous angle arc with a Euclidean
remainder ball. Scalar affine-query bounds and squared-distance action
comparisons can then be enlarged by exact Lipschitz/error identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .axial_query_certificate import (
    AngleArc,
    AxialRotationOrbit,
    HarmonicQuery,
    QueryBounds,
)

FloatArray: TypeAlias = NDArray[np.float64]
OrbitKey: TypeAlias = tuple[str, tuple[float, ...]]
_SCOPE = "continuous-axial-group-conformal-support-v1"


def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _probability(value: object, name: str, *, include_one: bool = False) -> float:
    result = _scalar(value, name)
    upper_ok = result <= 1.0 if include_one else result < 1.0
    if not 0.0 < result or not upper_ok:
        bracket = "(0, 1]" if include_one else "(0, 1)"
        raise ValueError(f"{name} must lie in {bracket}")
    return result


def _vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    copy = np.array(result, copy=True)
    copy.setflags(write=False)
    return copy


def _readonly(value: ArrayLike, name: str, *, ndim: int | None = None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    copy = np.array(result, copy=True)
    copy.setflags(write=False)
    return copy


def _validate_orbit_key(value: object) -> OrbitKey:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("orbit_key must bind an identity and six-vector geometry")
    identity, geometry = value
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("orbit identity must be a nonempty string")
    if not isinstance(geometry, tuple) or len(geometry) != 6:
        raise ValueError("orbit geometry must contain origin and unit axis")
    numbers = tuple(_scalar(entry, "orbit_key entry") for entry in geometry)
    if not math.isclose(math.hypot(*numbers[3:]), 1.0, abs_tol=1e-12):
        raise ValueError("orbit_key must contain a unit axis")
    return identity, numbers


@dataclass(frozen=True, slots=True)
class GroupConformalUpperBound:
    """One-sided split-conformal upper bound for exchangeable group scores.

    If ``finite`` is false, the requested finite-sample coverage cannot be
    attained by a finite order statistic. Callers must reject rather than
    silently use the sample maximum as a stronger guarantee.
    """

    threshold: float | None
    miscoverage: float
    calibration_group_count: int
    order_statistic: int
    finite: bool
    scope: str = _SCOPE

    def __post_init__(self) -> None:
        alpha = _probability(self.miscoverage, "miscoverage")
        count = self.calibration_group_count
        order = self.order_statistic
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("calibration_group_count must be a positive integer")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order_statistic must be a positive integer")
        if type(self.finite) is not bool:
            raise TypeError("finite must be a bool")
        if self.scope != _SCOPE:
            raise ValueError("calibration scope changed")
        if self.finite != (self.threshold is not None):
            raise ValueError("finite flag and threshold disagree")
        if self.threshold is not None:
            _scalar(self.threshold, "threshold", nonnegative=True)
            if order > count:
                raise ValueError("finite conformal order exceeds sample count")
        elif order <= count:
            raise ValueError("missing threshold despite a finite conformal order")
        object.__setattr__(self, "miscoverage", alpha)

    @property
    def coverage_level(self) -> float:
        return 1.0 - self.miscoverage

    def summary(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "miscoverage": self.miscoverage,
            "coverage_level": self.coverage_level,
            "calibration_group_count": self.calibration_group_count,
            "order_statistic": self.order_statistic,
            "finite": self.finite,
            "guarantee": (
                "marginal-over-exchangeable-group-score; not conditional or selective coverage"
            ),
        }


def calibrate_group_conformal_upper_bound(
    scores: ArrayLike,
    *,
    miscoverage: float,
) -> GroupConformalUpperBound:
    """Calibrate the standard finite-sample upper order statistic.

    For exchangeable nonnegative group scores ``S_1,...,S_n,S_{n+1}``, the
    returned finite threshold is the ``ceil((n+1)(1-alpha))``-th order statistic
    of the first ``n`` scores. It gives marginal coverage at least ``1-alpha``
    for the next group score. If that order exceeds ``n``, no finite threshold
    is returned.
    """

    alpha = _probability(miscoverage, "miscoverage")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    order = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    if order > values.size:
        return GroupConformalUpperBound(
            threshold=None,
            miscoverage=alpha,
            calibration_group_count=int(values.size),
            order_statistic=order,
            finite=False,
        )
    threshold = float(np.partition(values, order - 1)[order - 1])
    return GroupConformalUpperBound(
        threshold=threshold,
        miscoverage=alpha,
        calibration_group_count=int(values.size),
        order_statistic=order,
        finite=True,
    )


def empirical_upper_quantile(values: ArrayLike, *, probability: float) -> float:
    """Return the conservative empirical upper quantile by order statistic."""

    level = _probability(probability, "probability", include_one=True)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("values must be finite and nonnegative")
    order = max(1, int(math.ceil(level * array.size)))
    return float(np.partition(array, order - 1)[order - 1])


@dataclass(frozen=True, slots=True)
class AxialTubeResidual:
    """Closest-orbit decomposition of one real point."""

    angle_radians: float
    euclidean_residual: float
    radial_scale: float
    angle_normalizer: float
    normalized_score: float

    def __post_init__(self) -> None:
        angle = _scalar(self.angle_radians, "angle_radians")
        residual = _scalar(
            self.euclidean_residual,
            "euclidean_residual",
            nonnegative=True,
        )
        scale = _scalar(self.radial_scale, "radial_scale", nonnegative=True)
        normalizer = _scalar(
            self.angle_normalizer,
            "angle_normalizer",
            nonnegative=True,
        )
        score = _scalar(self.normalized_score, "normalized_score", nonnegative=True)
        if scale == 0.0 or normalizer == 0.0:
            raise ValueError("radial_scale and angle_normalizer must be positive")
        expected = max(abs(angle) / normalizer, residual / scale)
        if not math.isclose(score, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("normalized_score does not match its components")


def axial_tube_residual(
    orbit: AxialRotationOrbit,
    representative_point: ArrayLike,
    observed_point: ArrayLike,
    *,
    angle_normalizer: float = math.pi,
    radial_scale: float | None = None,
) -> AxialTubeResidual:
    """Decompose a point into closest continuous-orbit angle and residual."""

    if not isinstance(orbit, AxialRotationOrbit):
        raise TypeError("orbit must be an AxialRotationOrbit")
    representative = _vector(representative_point, "representative_point")
    observed = _vector(observed_point, "observed_point")
    normalizer = _scalar(
        angle_normalizer,
        "angle_normalizer",
        nonnegative=True,
    )
    if normalizer == 0.0:
        raise ValueError("angle_normalizer must be positive")

    centered_rep = representative - orbit.origin
    rep_parallel = float(centered_rep @ orbit.axis)
    rep_radial = centered_rep - rep_parallel * orbit.axis
    rep_radius = float(np.linalg.norm(rep_radial))
    if rep_radius <= np.finfo(np.float64).tiny:
        raise ValueError("representative_point must be off the orbit axis")

    scale = rep_radius if radial_scale is None else _scalar(
        radial_scale,
        "radial_scale",
        nonnegative=True,
    )
    if scale == 0.0:
        raise ValueError("radial_scale must be positive")

    centered_observed = observed - orbit.origin
    observed_parallel = float(centered_observed @ orbit.axis)
    observed_radial = centered_observed - observed_parallel * orbit.axis
    observed_radius = float(np.linalg.norm(observed_radial))
    if observed_radius <= np.finfo(np.float64).tiny:
        angle = 0.0
    else:
        cosine_term = float(rep_radial @ observed_radial)
        sine_term = float(np.cross(orbit.axis, rep_radial) @ observed_radial)
        angle = math.atan2(sine_term, cosine_term)
    closest = orbit.transform(representative[None, :], angle)[0]
    residual = float(np.linalg.norm(observed - closest))
    score = max(abs(angle) / normalizer, residual / scale)
    return AxialTubeResidual(
        angle_radians=angle,
        euclidean_residual=residual,
        radial_scale=scale,
        angle_normalizer=normalizer,
        normalized_score=score,
    )


@dataclass(frozen=True, slots=True)
class AxialTubeSupport:
    """Continuous angle arc plus an additive Euclidean remainder ball."""

    orbit_key: OrbitKey
    arc: AngleArc
    euclidean_radius: float
    normalized_score_threshold: float
    radial_scale: float
    angle_normalizer: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "orbit_key", _validate_orbit_key(self.orbit_key))
        if not isinstance(self.arc, AngleArc):
            raise TypeError("arc must be an AngleArc")
        for name in (
            "euclidean_radius",
            "normalized_score_threshold",
            "radial_scale",
            "angle_normalizer",
        ):
            value = _scalar(getattr(self, name), name, nonnegative=True)
            if name in {"radial_scale", "angle_normalizer"} and value == 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def contains(
        self,
        orbit: AxialRotationOrbit,
        representative_point: ArrayLike,
        point: ArrayLike,
        *,
        atol: float = 1e-12,
    ) -> bool:
        if orbit.key != self.orbit_key:
            raise ValueError("support and orbit identities differ")
        tolerance = _scalar(atol, "atol", nonnegative=True)
        residual = axial_tube_residual(
            orbit,
            representative_point,
            point,
            angle_normalizer=self.angle_normalizer,
            radial_scale=self.radial_scale,
        )
        return self.arc.contains(residual.angle_radians, atol=tolerance) and (
            residual.euclidean_residual <= self.euclidean_radius + tolerance
        )

    def expand_scalar_bounds(
        self,
        query: HarmonicQuery,
        *,
        euclidean_lipschitz: float,
    ) -> QueryBounds:
        if query.orbit_key != self.orbit_key:
            raise ValueError("support and query identities differ")
        lipschitz = _scalar(
            euclidean_lipschitz,
            "euclidean_lipschitz",
            nonnegative=True,
        )
        raw = query.bounds(self.arc)
        padding = lipschitz * self.euclidean_radius
        return QueryBounds(
            lower=raw.lower - padding,
            upper=raw.upper + padding,
            lower_angle=raw.lower_angle,
            upper_angle=raw.upper_angle,
        )


def support_from_conformal_threshold(
    orbit: AxialRotationOrbit,
    *,
    normalized_score_threshold: float,
    radial_scale: float,
    angle_normalizer: float,
) -> AxialTubeSupport:
    """Instantiate a case-specific continuous support from one calibrated score."""

    if not isinstance(orbit, AxialRotationOrbit):
        raise TypeError("orbit must be an AxialRotationOrbit")
    threshold = _scalar(
        normalized_score_threshold,
        "normalized_score_threshold",
        nonnegative=True,
    )
    scale = _scalar(radial_scale, "radial_scale", nonnegative=True)
    normalizer = _scalar(
        angle_normalizer,
        "angle_normalizer",
        nonnegative=True,
    )
    if scale == 0.0 or normalizer == 0.0:
        raise ValueError("radial_scale and angle_normalizer must be positive")
    return AxialTubeSupport(
        orbit_key=orbit.key,
        arc=AngleArc(center=0.0, half_width=min(math.pi, threshold * normalizer)),
        euclidean_radius=threshold * scale,
        normalized_score_threshold=threshold,
        radial_scale=scale,
        angle_normalizer=normalizer,
    )


@dataclass(frozen=True, slots=True)
class HarmonicVectorQuery:
    """Vector-valued first harmonic on one continuous axial orbit."""

    constant: FloatArray
    cosine: FloatArray
    sine: FloatArray
    orbit_key: OrbitKey

    def __post_init__(self) -> None:
        constant = _readonly(self.constant, "constant", ndim=1)
        cosine = _readonly(self.cosine, "cosine", ndim=1)
        sine = _readonly(self.sine, "sine", ndim=1)
        if constant.size == 0 or cosine.shape != constant.shape or sine.shape != constant.shape:
            raise ValueError("harmonic vector coefficients must have one equal nonzero shape")
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "cosine", cosine)
        object.__setattr__(self, "sine", sine)
        object.__setattr__(self, "orbit_key", _validate_orbit_key(self.orbit_key))

    @property
    def dimension(self) -> int:
        return int(self.constant.size)

    def evaluate(self, angle: float) -> FloatArray:
        theta = _scalar(angle, "angle")
        result = (
            self.constant
            + math.cos(theta) * self.cosine
            + math.sin(theta) * self.sine
        )
        return _readonly(result, "query evaluation", ndim=1)

    def scalar_projection(self, weights: ArrayLike, *, offset: float = 0.0) -> HarmonicQuery:
        vector = np.asarray(weights, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("weights must be a finite query-dimension vector")
        shift = _scalar(offset, "offset")
        return HarmonicQuery(
            constant=shift + float(vector @ self.constant),
            cosine=float(vector @ self.cosine),
            sine=float(vector @ self.sine),
            orbit_key=self.orbit_key,
        )

    def full_circle_weighted_diameter(
        self,
        *,
        weight: ArrayLike | None = None,
        additive_query_radius: float = 0.0,
    ) -> float:
        """Exact weighted diameter of the continuous ellipse plus a query ball."""

        if weight is None:
            matrix = np.eye(self.dimension, dtype=np.float64)
        else:
            matrix = np.asarray(weight, dtype=np.float64)
            if (
                matrix.ndim != 2
                or matrix.shape[1] != self.dimension
                or matrix.shape[0] == 0
                or not np.all(np.isfinite(matrix))
            ):
                raise ValueError("weight must be a finite matrix with query_dimension columns")
        radius = _scalar(
            additive_query_radius,
            "additive_query_radius",
            nonnegative=True,
        )
        harmonic = matrix @ np.column_stack((self.cosine, self.sine))
        singular = np.linalg.svd(harmonic, compute_uv=False)
        orbit_diameter = 0.0 if singular.size == 0 else 2.0 * float(singular[0])
        weight_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
        result = orbit_diameter + 2.0 * weight_norm * radius
        if not math.isfinite(result):
            raise ValueError("weighted diameter overflowed")
        return result


def point_position_query(
    orbit: AxialRotationOrbit,
    representative_point: ArrayLike,
) -> HarmonicVectorQuery:
    """Return the exact 3-D point-position harmonic on the full SO(2) orbit."""

    if not isinstance(orbit, AxialRotationOrbit):
        raise TypeError("orbit must be an AxialRotationOrbit")
    point = _vector(representative_point, "representative_point")
    centered = point - orbit.origin
    parallel = float(centered @ orbit.axis) * orbit.axis
    radial = centered - parallel
    return HarmonicVectorQuery(
        constant=orbit.origin + parallel,
        cosine=radial,
        sine=np.cross(orbit.axis, radial),
        orbit_key=orbit.key,
    )


def squared_distance_query(
    orbit: AxialRotationOrbit,
    representative_point: ArrayLike,
    target_point: ArrayLike,
) -> HarmonicQuery:
    """Exact squared Euclidean distance from an orbiting point to a fixed point."""

    point_query = point_position_query(orbit, representative_point)
    target = _vector(target_point, "target_point")
    centered = point_query.constant - target
    radial = point_query.cosine
    tangent = point_query.sine
    return HarmonicQuery(
        constant=float(centered @ centered + radial @ radial),
        cosine=2.0 * float(centered @ radial),
        sine=2.0 * float(centered @ tangent),
        orbit_key=orbit.key,
    )


@dataclass(frozen=True, slots=True)
class ContinuousQueryDiameterCertificate:
    admitted: bool
    weighted_diameter: float
    tolerance: float
    scope_admitted: bool
    reason_codes: tuple[str, ...]
    scope: str = _SCOPE

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool or type(self.scope_admitted) is not bool:
            raise TypeError("admission flags must be bools")
        diameter = _scalar(
            self.weighted_diameter,
            "weighted_diameter",
            nonnegative=True,
        )
        tolerance = _scalar(self.tolerance, "tolerance", nonnegative=True)
        reasons = tuple(self.reason_codes)
        allowed = {"orbit-model-scope-not-admitted", "query-diameter-exceeds-tolerance"}
        if len(set(reasons)) != len(reasons) or any(reason not in allowed for reason in reasons):
            raise ValueError("invalid continuous-query rejection reasons")
        if self.admitted == bool(reasons):
            raise ValueError("admission and rejection reasons disagree")
        if self.admitted and diameter > tolerance:
            raise ValueError("admitted query exceeds its tolerance")
        if self.scope != _SCOPE:
            raise ValueError("certificate scope changed")
        object.__setattr__(self, "weighted_diameter", diameter)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "reason_codes", reasons)


def certify_full_circle_vector_query(
    query: HarmonicVectorQuery,
    *,
    tolerance: float,
    scope_admitted: bool,
    weight: ArrayLike | None = None,
    additive_query_radius: float = 0.0,
    numerical_slack: float = 1e-12,
) -> ContinuousQueryDiameterCertificate:
    """Certify a vector query over the complete continuous SO(2) orbit."""

    if not isinstance(query, HarmonicVectorQuery):
        raise TypeError("query must be a HarmonicVectorQuery")
    if type(scope_admitted) is not bool:
        raise TypeError("scope_admitted must be a bool")
    threshold = _scalar(tolerance, "tolerance", nonnegative=True)
    slack = _scalar(numerical_slack, "numerical_slack", nonnegative=True)
    diameter = query.full_circle_weighted_diameter(
        weight=weight,
        additive_query_radius=additive_query_radius,
    )
    reasons: list[str] = []
    if not scope_admitted:
        reasons.append("orbit-model-scope-not-admitted")
    if diameter > threshold + slack:
        reasons.append("query-diameter-exceeds-tolerance")
    return ContinuousQueryDiameterCertificate(
        admitted=not reasons,
        weighted_diameter=diameter,
        tolerance=threshold,
        scope_admitted=scope_admitted,
        reason_codes=tuple(reasons),
    )


__all__ = [
    "AxialTubeResidual",
    "AxialTubeSupport",
    "ContinuousQueryDiameterCertificate",
    "GroupConformalUpperBound",
    "HarmonicVectorQuery",
    "axial_tube_residual",
    "calibrate_group_conformal_upper_bound",
    "certify_full_circle_vector_query",
    "empirical_upper_quantile",
    "point_position_query",
    "squared_distance_query",
    "support_from_conformal_threshold",
]
