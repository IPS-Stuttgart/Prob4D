"""Validated row-level contracts for unfused Prob4D observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]


def _readonly(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _require_psd(value: np.ndarray, name: str, *, tolerance: float = 1e-12) -> None:
    symmetric = 0.5 * (value + value.swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True)
class ObservationFactor:
    """One unfused set of associated 3-D observations in a local window gauge."""

    factor_id: str
    frame_index: int
    view_id: str
    window_id: str
    gauge_id: str
    point_ids: IntArray
    points_local_m: FloatArray
    valid_mask: BoolArray
    local_covariance_m2: FloatArray
    association_probability: FloatArray
    correlation_group_id: str
    causal_frame_limit: int
    ray_directions_local: FloatArray | None = None

    def __post_init__(self) -> None:
        identifiers = {
            "factor_id": self.factor_id,
            "view_id": self.view_id,
            "window_id": self.window_id,
            "gauge_id": self.gauge_id,
            "correlation_group_id": self.correlation_group_id,
        }
        for name, value in identifiers.items():
            if not str(value):
                raise ValueError(f"{name} must not be empty")
        if int(self.frame_index) < 0:
            raise ValueError("frame_index must be non-negative")
        if int(self.causal_frame_limit) < int(self.frame_index):
            raise ValueError("observation factor crosses its causal frame limit")

        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        points = np.asarray(self.points_local_m, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        covariance = np.asarray(self.local_covariance_m2, dtype=np.float64)
        association = np.asarray(self.association_probability, dtype=np.float64)
        if point_ids.ndim != 1 or len(point_ids) == 0:
            raise ValueError("point_ids must be a non-empty vector")
        if len(np.unique(point_ids)) != len(point_ids):
            raise ValueError("point_ids must be unique within a factor")
        if points.shape != (len(point_ids), 3):
            raise ValueError("points_local_m must have shape (N, 3)")
        if valid.shape != (len(point_ids),):
            raise ValueError("valid_mask must have shape (N,)")
        if covariance.shape != (len(point_ids), 3, 3):
            raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
        if association.shape != (len(point_ids),):
            raise ValueError("association_probability must have shape (N,)")
        if not np.all(np.isfinite(association)) or np.any(
            (association < 0.0) | (association > 1.0)
        ):
            raise ValueError("association_probability must lie in [0, 1]")
        active = valid & (association > 0.0)
        if not np.all(np.isfinite(points[active])):
            raise ValueError("active local observations must be finite")
        if not np.all(np.isfinite(covariance[active])):
            raise ValueError("active local covariances must be finite")
        if np.any(active):
            if not np.allclose(
                covariance[active], covariance[active].swapaxes(1, 2), atol=1e-12
            ):
                raise ValueError("local covariances must be symmetric")
            _require_psd(covariance[active], "local covariances")

        rays: np.ndarray | None = None
        if self.ray_directions_local is not None:
            rays = np.asarray(self.ray_directions_local, dtype=np.float64).copy()
            if rays.shape != points.shape:
                raise ValueError("ray_directions_local must have shape (N, 3)")
            if not np.all(np.isfinite(rays[active])):
                raise ValueError("active ray directions must be finite")
            norms = np.linalg.norm(rays, axis=1)
            if np.any(active & (norms <= np.finfo(np.float64).eps)):
                raise ValueError("active ray directions must be nonzero")
            normalize = norms > np.finfo(np.float64).eps
            rays[normalize] /= norms[normalize, None]
            rays.setflags(write=False)

        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "causal_frame_limit", int(self.causal_frame_limit))
        object.__setattr__(self, "point_ids", _readonly(point_ids))
        object.__setattr__(self, "points_local_m", _readonly(points))
        object.__setattr__(self, "valid_mask", _readonly(valid))
        object.__setattr__(self, "local_covariance_m2", _readonly(covariance))
        object.__setattr__(self, "association_probability", _readonly(association))
        object.__setattr__(self, "ray_directions_local", rays)


@dataclass(frozen=True)
class LinearizedObservationFactor:
    """World-frame moments and gauge Jacobians for one observation factor."""

    factor_id: str
    frame_index: int
    view_id: str
    window_id: str
    gauge_id: str
    correlation_group_id: str
    point_ids: IntArray
    world_mean_m: FloatArray
    conditional_world_covariance_m2: FloatArray
    marginal_world_covariance_m2: FloatArray
    gauge_jacobian: FloatArray
    valid_mask: BoolArray
    association_probability: FloatArray
    ray_directions_world: FloatArray | None = None

    def __post_init__(self) -> None:
        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        mean = np.asarray(self.world_mean_m, dtype=np.float64)
        conditional_covariance = np.asarray(
            self.conditional_world_covariance_m2, dtype=np.float64
        )
        marginal_covariance = np.asarray(
            self.marginal_world_covariance_m2, dtype=np.float64
        )
        jacobian = np.asarray(self.gauge_jacobian, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        probability = np.asarray(self.association_probability, dtype=np.float64)
        count = len(point_ids)
        if mean.shape != (count, 3):
            raise ValueError("world_mean_m must have shape (N, 3)")
        if conditional_covariance.shape != (count, 3, 3):
            raise ValueError(
                "conditional_world_covariance_m2 must have shape (N, 3, 3)"
            )
        if marginal_covariance.shape != (count, 3, 3):
            raise ValueError(
                "marginal_world_covariance_m2 must have shape (N, 3, 3)"
            )
        if jacobian.shape != (count, 3, 7):
            raise ValueError("gauge_jacobian must have shape (N, 3, 7)")
        if valid.shape != (count,) or probability.shape != (count,):
            raise ValueError("linearized factor masks have changed shape")
        rays = None
        if self.ray_directions_world is not None:
            rays = np.asarray(self.ray_directions_world, dtype=np.float64)
            if rays.shape != (count, 3):
                raise ValueError("ray_directions_world must have shape (N, 3)")
            rays = _readonly(rays)
        for name, value in (
            ("point_ids", point_ids),
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional_covariance),
            ("marginal_world_covariance_m2", marginal_covariance),
            ("gauge_jacobian", jacobian),
            ("valid_mask", valid),
            ("association_probability", probability),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "ray_directions_world", rays)


@dataclass(frozen=True)
class StackedObservationFactors:
    """Flattened factor rows with block-structured gauge nuisance parameters."""

    world_mean_m: FloatArray
    conditional_world_covariance_m2: FloatArray
    marginal_world_covariance_m2: FloatArray
    gauge_jacobian: FloatArray
    gauge_prior_covariance: FloatArray
    association_probability: FloatArray
    point_ids: IntArray
    frame_indices: IntArray
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    gauge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = np.asarray(self.world_mean_m, dtype=np.float64)
        conditional_covariance = np.asarray(
            self.conditional_world_covariance_m2, dtype=np.float64
        )
        marginal_covariance = np.asarray(
            self.marginal_world_covariance_m2, dtype=np.float64
        )
        jacobian = np.asarray(self.gauge_jacobian, dtype=np.float64)
        gauge_prior = np.asarray(self.gauge_prior_covariance, dtype=np.float64)
        probability = np.asarray(self.association_probability, dtype=np.float64)
        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        count = len(mean)
        gauge_dimension = 7 * len(self.gauge_ids)
        if mean.shape != (count, 3):
            raise ValueError("stacked world means must have shape (M, 3)")
        if conditional_covariance.shape != (count, 3, 3):
            raise ValueError(
                "stacked conditional covariance must have shape (M, 3, 3)"
            )
        if marginal_covariance.shape != (count, 3, 3):
            raise ValueError(
                "stacked marginal covariance must have shape (M, 3, 3)"
            )
        if jacobian.shape != (count, 3, gauge_dimension):
            raise ValueError("stacked gauge Jacobian has changed shape")
        if gauge_prior.shape != (gauge_dimension, gauge_dimension):
            raise ValueError("gauge prior covariance has changed shape")
        if probability.shape != (count,):
            raise ValueError("stacked association_probability has changed shape")
        if point_ids.shape != (count,) or frame_indices.shape != (count,):
            raise ValueError("stacked integer metadata has changed shape")
        for values in (self.view_ids, self.factor_ids, self.correlation_group_ids):
            if len(values) != count:
                raise ValueError("stacked string metadata has changed length")
        for name, value in (
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional_covariance),
            ("marginal_world_covariance_m2", marginal_covariance),
            ("gauge_jacobian", jacobian),
            ("gauge_prior_covariance", gauge_prior),
            ("association_probability", probability),
            ("point_ids", point_ids),
            ("frame_indices", frame_indices),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "view_ids", tuple(map(str, self.view_ids)))
        object.__setattr__(self, "factor_ids", tuple(map(str, self.factor_ids)))
        object.__setattr__(
            self, "correlation_group_ids", tuple(map(str, self.correlation_group_ids))
        )
        object.__setattr__(self, "gauge_ids", tuple(map(str, self.gauge_ids)))
