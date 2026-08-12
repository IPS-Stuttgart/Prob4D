"""Shared validation and basis helpers for point uncertainty calibration v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ._scientific_scalars import require_finite_real, require_genuine_integer
from ._strict_json import require_exact_fields, require_exact_string, require_mapping

_POLICY_FIELDS = frozenset(
    {
        "ridge_strength",
        "lateral_coupling_strength",
        "minimum_group_count",
        "minimum_rows_per_group",
        "variance_floor",
        "log_variance_lower",
        "log_variance_upper",
        "newton_tolerance",
        "maximum_iterations",
    }
)


def positive(value: object, *, name: str) -> float:
    return require_finite_real(
        value,
        name=name,
        minimum=0.0,
        minimum_inclusive=False,
    )


def float_tuple(
    value: object,
    *,
    name: str,
    length: int,
    positive_only: bool = False,
) -> tuple[float, ...]:
    if type(value) not in {tuple, list}:
        raise TypeError(f"{name} must be a sequence")
    items = tuple(
        (
            positive(item, name=f"{name}[{index}]")
            if positive_only
            else require_finite_real(item, name=f"{name}[{index}]")
        )
        for index, item in enumerate(value)
    )
    if len(items) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return items


def string_tuple(
    value: object,
    *,
    name: str,
    sorted_unique: bool = False,
) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        raise TypeError(f"{name} must be a sequence")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    if sorted_unique and result != tuple(sorted(result)):
        raise ValueError(f"{name} must use canonical sorted order")
    return result


def float_matrix(value: object, *, name: str, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero vector")
    return value / norms[:, None]


def local_point_basis(
    ray_directions: object,
    tangent_reference: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ray, prediction-reference tangent, and orthogonal tangent axes."""

    rays = _normalize_rows(
        float_matrix(ray_directions, name="ray_directions", columns=3),
        name="ray_directions",
    )
    reference = float_matrix(
        tangent_reference,
        name="tangent_reference",
        columns=3,
    )
    if reference.shape[0] != rays.shape[0]:
        raise ValueError("tangent_reference and ray_directions must have matching rows")

    tangent = reference - np.sum(reference * rays, axis=1)[:, None] * rays
    fallback = np.linalg.norm(tangent, axis=1) <= 1e-10
    if np.any(fallback):
        indices = np.argmin(np.abs(rays[fallback]), axis=1)
        canonical = np.eye(3, dtype=np.float64)[indices]
        tangent[fallback] = (
            canonical
            - np.sum(canonical * rays[fallback], axis=1)[:, None] * rays[fallback]
        )
    tangent_one = _normalize_rows(tangent, name="projected tangent_reference")
    tangent_two = _normalize_rows(np.cross(rays, tangent_one), name="orthogonal tangent")
    return rays, tangent_one, tangent_two


@dataclass(frozen=True, slots=True)
class PointUncertaintyCalibrationPolicyV2:
    """Frozen source-only fit and numerical-stability policy."""

    ridge_strength: float = 1e-3
    lateral_coupling_strength: float = 1e-2
    minimum_group_count: int = 8
    minimum_rows_per_group: int = 64
    variance_floor: float = 1e-10
    log_variance_lower: float = -30.0
    log_variance_upper: float = 10.0
    newton_tolerance: float = 1e-8
    maximum_iterations: int = 80

    def __post_init__(self) -> None:
        for name in ("ridge_strength", "lateral_coupling_strength"):
            object.__setattr__(
                self,
                name,
                require_finite_real(getattr(self, name), name=name, minimum=0.0),
            )
        for name in ("minimum_group_count", "minimum_rows_per_group", "maximum_iterations"):
            object.__setattr__(
                self,
                name,
                require_genuine_integer(getattr(self, name), name=name, minimum=1),
            )
        object.__setattr__(
            self,
            "variance_floor",
            positive(self.variance_floor, name="variance_floor"),
        )
        lower = require_finite_real(self.log_variance_lower, name="log_variance_lower")
        upper = require_finite_real(self.log_variance_upper, name="log_variance_upper")
        if upper <= lower:
            raise ValueError("log_variance_upper must exceed log_variance_lower")
        object.__setattr__(self, "log_variance_lower", lower)
        object.__setattr__(self, "log_variance_upper", upper)
        object.__setattr__(
            self,
            "newton_tolerance",
            positive(self.newton_tolerance, name="newton_tolerance"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ridge_strength": self.ridge_strength,
            "lateral_coupling_strength": self.lateral_coupling_strength,
            "minimum_group_count": self.minimum_group_count,
            "minimum_rows_per_group": self.minimum_rows_per_group,
            "variance_floor": self.variance_floor,
            "log_variance_lower": self.log_variance_lower,
            "log_variance_upper": self.log_variance_upper,
            "newton_tolerance": self.newton_tolerance,
            "maximum_iterations": self.maximum_iterations,
        }

    @classmethod
    def from_dict(cls, value: object) -> PointUncertaintyCalibrationPolicyV2:
        mapping = require_mapping(value, name="point uncertainty v2 policy")
        require_exact_fields(mapping, _POLICY_FIELDS, name="point uncertainty v2 policy")
        return cls(**cast(dict[str, Any], dict(mapping)))


__all__ = [
    "PointUncertaintyCalibrationPolicyV2",
    "float_matrix",
    "float_tuple",
    "local_point_basis",
    "string_tuple",
]
