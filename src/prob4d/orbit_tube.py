"""Calibrated Euclidean tubes around estimated axial gauge orbits.

An estimated axial orbit is a circle in three-dimensional space.  The circle may
capture a residual rotation while remaining imperfect because its line, axial
coordinate, or radius were estimated from partial observations.  This module
provides:

* exact point-to-circle distance;
* exact affine-query bounds for a Euclidean tube around the circle;
* deterministic minimal-rotation transport for a canonical point baseline; and
* split-conformal calibration of one radius from maxima over independent groups.

The finite-sample statement is conditional on exchangeable independent groups
and the supplied nonconformity score.  It is not a provider-quality, physical
state-recovery, or deployment-safety guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return np.frombuffer(result.tobytes(), dtype=np.float64)


def _unit_vector(value: ArrayLike, name: str) -> FloatArray:
    result = _vector(value, name)
    norm = math.hypot(*(float(entry) for entry in result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have positive norm")
    return _vector(result / norm, f"normalized {name}")


def _validated_group_id(value: object, index: int) -> str:
    if type(value) is not str:
        raise TypeError(f"group_ids[{index}] must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"group_ids[{index}] must be a nonempty trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"group_ids[{index}] must not contain control characters")
    if len(value.encode("utf-8")) > 128:
        raise ValueError(f"group_ids[{index}] must contain at most 128 UTF-8 bytes")
    return value


@dataclass(frozen=True, slots=True)
class ScalarBounds:
    """Closed scalar interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = _scalar(self.lower, "lower")
        upper = _scalar(self.upper, "upper")
        if lower > upper:
            raise ValueError("lower must not exceed upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float, *, atol: float = 0.0) -> bool:
        point = _scalar(value, "value")
        tolerance = _scalar(atol, "atol", nonnegative=True)
        return self.lower - tolerance <= point <= self.upper + tolerance

    def threshold_sign(self, threshold: float = 0.0, *, margin: float = 0.0) -> int | None:
        """Return a certified threshold side, or ``None`` when the interval overlaps."""

        level = _scalar(threshold, "threshold")
        gap = _scalar(margin, "margin", nonnegative=True)
        if self.lower > level + gap:
            return 1
        if self.upper < level - gap:
            return -1
        return None


@dataclass(frozen=True, slots=True, eq=False)
class AxialCircleOrbit:
    """One estimated circular orbit around a fixed axis."""

    center: FloatArray
    axis: FloatArray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector(self.center, "center"))
        object.__setattr__(self, "axis", _unit_vector(self.axis, "axis"))
        object.__setattr__(
            self,
            "radius",
            _scalar(self.radius, "radius", nonnegative=True),
        )

    def point_distance(self, point: ArrayLike) -> float:
        """Exact Euclidean distance from a point to the circular orbit."""

        delta = _vector(point, "point") - self.center
        axial = float(delta @ self.axis)
        radial = delta - axial * self.axis
        radial_norm = math.hypot(*(float(entry) for entry in radial))
        result = math.hypot(axial, radial_norm - self.radius)
        if not math.isfinite(result):
            raise ValueError("point-to-orbit distance overflowed")
        return result

    def affine_query_bounds(
        self,
        direction: ArrayLike,
        *,
        offset: float = 0.0,
        tube_radius: float = 0.0,
    ) -> ScalarBounds:
        """Exact bounds of ``offset + direction @ point`` over an orbit tube.

        ``tube_radius`` expands the circle by a closed Euclidean ball.  The
        expansion is exact for an affine scalar query because its Lipschitz
        constant under Euclidean distance is ``||direction||``.
        """

        vector = _vector(direction, "direction")
        offset_value = _scalar(offset, "offset")
        tube = _scalar(tube_radius, "tube_radius", nonnegative=True)
        direction_norm = math.hypot(*(float(entry) for entry in vector))
        parallel = float(vector @ self.axis)
        perpendicular_norm = math.sqrt(max(direction_norm**2 - parallel**2, 0.0))
        center_value = offset_value + float(vector @ self.center)
        half_width = self.radius * perpendicular_norm + tube * direction_norm
        return ScalarBounds(center_value - half_width, center_value + half_width)


@dataclass(frozen=True, slots=True)
class GroupMaximumRadiusCalibration:
    """Split-conformal radius from complete-group maximum scores."""

    requested_miscoverage: float
    calibration_group_ids: tuple[str, ...]
    calibration_group_maximum_scores: tuple[float, ...]
    group_assignment_sha256: str
    order_statistic_rank: int
    finite_sample_coverage_level: float
    radius: float

    def __post_init__(self) -> None:
        alpha = _scalar(self.requested_miscoverage, "requested_miscoverage")
        if not 0.0 < alpha < 1.0:
            raise ValueError("requested_miscoverage must lie in (0, 1)")
        if not isinstance(self.calibration_group_ids, tuple):
            raise TypeError("calibration_group_ids must be a tuple")
        if not isinstance(self.calibration_group_maximum_scores, tuple):
            raise TypeError("calibration_group_maximum_scores must be a tuple")
        if len(self.calibration_group_ids) == 0:
            raise ValueError("at least one calibration group is required")
        if len(self.calibration_group_ids) != len(self.calibration_group_maximum_scores):
            raise ValueError("group IDs and maximum scores must have equal length")
        ids = tuple(
            _validated_group_id(value, index)
            for index, value in enumerate(self.calibration_group_ids)
        )
        if tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise ValueError("calibration_group_ids must be unique and sorted")
        maxima = tuple(
            _scalar(value, f"calibration_group_maximum_scores[{index}]", nonnegative=True)
            for index, value in enumerate(self.calibration_group_maximum_scores)
        )
        rank = self.order_statistic_rank
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("order_statistic_rank must be an integer")
        if not 1 <= rank <= len(ids):
            raise ValueError("order_statistic_rank is inconsistent with group count")
        level = _scalar(self.finite_sample_coverage_level, "finite_sample_coverage_level")
        expected_level = rank / (len(ids) + 1)
        if not math.isclose(level, expected_level, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("finite_sample_coverage_level does not equal rank/(n+1)")
        radius = _scalar(self.radius, "radius", nonnegative=True)
        expected_radius = sorted(maxima)[rank - 1]
        if not math.isclose(radius, expected_radius, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("radius does not equal the registered order statistic")
        payload = json.dumps(ids, ensure_ascii=True, separators=(",", ":")).encode()
        expected_digest = hashlib.sha256(payload).hexdigest()
        if self.group_assignment_sha256 != expected_digest:
            raise ValueError("group_assignment_sha256 does not match calibration_group_ids")
        object.__setattr__(self, "requested_miscoverage", alpha)
        object.__setattr__(self, "calibration_group_ids", ids)
        object.__setattr__(self, "calibration_group_maximum_scores", maxima)
        object.__setattr__(self, "finite_sample_coverage_level", level)
        object.__setattr__(self, "radius", radius)

    @property
    def group_count(self) -> int:
        return len(self.calibration_group_ids)

    def covers(self, score: float, *, atol: float = 0.0) -> bool:
        value = _scalar(score, "score", nonnegative=True)
        tolerance = _scalar(atol, "atol", nonnegative=True)
        return value <= self.radius + tolerance

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["group_count"] = self.group_count
        result["requested_coverage"] = 1.0 - self.requested_miscoverage
        return result


def calibrate_group_maximum_radius(
    scores: ArrayLike,
    group_ids: Sequence[str],
    *,
    miscoverage: float,
) -> GroupMaximumRadiusCalibration:
    """Calibrate a radius from maxima over exchangeable independent groups."""

    alpha = _scalar(miscoverage, "miscoverage")
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie in (0, 1)")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    if isinstance(group_ids, str | bytes):
        raise TypeError("group_ids must be a sequence")
    supplied = tuple(group_ids)
    if len(supplied) != values.size:
        raise ValueError("group_ids must contain one entry per score")
    validated = tuple(_validated_group_id(value, index) for index, value in enumerate(supplied))
    unique = tuple(sorted(set(validated)))
    maxima = tuple(
        float(np.max(values[np.asarray([group == selected for group in validated])]))
        for selected in unique
    )
    rank = math.ceil((len(unique) + 1) * (1.0 - alpha))
    if rank > len(unique):
        minimum = math.ceil(1.0 / alpha) - 1
        raise ValueError(
            "too few independent groups for a finite split-conformal radius; "
            f"need at least {minimum}"
        )
    radius = sorted(maxima)[rank - 1]
    payload = json.dumps(unique, ensure_ascii=True, separators=(",", ":")).encode()
    return GroupMaximumRadiusCalibration(
        requested_miscoverage=alpha,
        calibration_group_ids=unique,
        calibration_group_maximum_scores=maxima,
        group_assignment_sha256=hashlib.sha256(payload).hexdigest(),
        order_statistic_rank=rank,
        finite_sample_coverage_level=rank / (len(unique) + 1),
        radius=radius,
    )


def minimal_rotation_transport(
    vector: ArrayLike,
    source_axis: ArrayLike,
    target_axis: ArrayLike,
) -> FloatArray:
    """Transport a vector by a deterministic shortest rotation between axes."""

    value = _vector(vector, "vector")
    source = _unit_vector(source_axis, "source_axis")
    target = _unit_vector(target_axis, "target_axis")
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    cross = np.cross(source, target)
    sine = math.hypot(*(float(entry) for entry in cross))
    if sine <= 1e-12:
        if cosine >= 0.0:
            return _vector(value, "transported vector")
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
        rotation_axis = np.cross(source, basis)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotated = 2.0 * rotation_axis * float(rotation_axis @ value) - value
        return _vector(rotated, "transported vector")
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    return _vector(rotation @ value, "transported vector")
