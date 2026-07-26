"""Compact anisotropic uncertainty for dense geometry and motion."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .alignment import WindowAlignment
from .data import PredictionWindow
from .sim3 import Sim3

FloatArray = NDArray[np.floating]


def _readonly_copy(values: np.ndarray, *, dtype: type = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StructuredCovariance:
    """Along-ray and per-lateral-axis variance on a dense sample grid."""

    ray_directions: FloatArray
    parallel_variance: FloatArray
    lateral_variance: FloatArray

    def __post_init__(self) -> None:
        rays = np.asarray(self.ray_directions, dtype=np.float64).copy()
        parallel = np.asarray(self.parallel_variance, dtype=np.float64).copy()
        lateral = np.asarray(self.lateral_variance, dtype=np.float64).copy()
        if rays.shape[:-1] != parallel.shape or rays.shape[-1] != 3:
            raise ValueError("ray_directions must have shape parallel_variance.shape + (3,)")
        if lateral.shape != parallel.shape:
            raise ValueError("parallel and lateral variance must have matching shapes")
        if not np.all(np.isfinite(rays)):
            raise ValueError("ray_directions must be finite")
        if not np.all(np.isfinite(parallel)) or not np.all(np.isfinite(lateral)):
            raise ValueError("structured variances must be finite")
        if np.any(parallel <= 0) or np.any(lateral <= 0):
            raise ValueError("structured variances must be strictly positive")
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("ray_directions must be nonzero")
        rays = rays / norms
        object.__setattr__(self, "ray_directions", _readonly_copy(rays))
        object.__setattr__(self, "parallel_variance", _readonly_copy(parallel))
        object.__setattr__(self, "lateral_variance", _readonly_copy(lateral))

    def matrices(self) -> FloatArray:
        identity = np.eye(3)
        outer = np.einsum("...i,...j->...ij", self.ray_directions, self.ray_directions)
        return (
            self.lateral_variance[..., None, None] * identity
            + (self.parallel_variance - self.lateral_variance)[..., None, None] * outer
        )

    def transformed(self, transform: Sim3) -> StructuredCovariance:
        return StructuredCovariance(
            ray_directions=transform.rotate_directions(self.ray_directions),
            parallel_variance=transform.scale**2 * self.parallel_variance,
            lateral_variance=transform.scale**2 * self.lateral_variance,
        )


@dataclass
class DisagreementEvidence:
    """Accumulated overlap residual energy for one window."""

    parallel_sum: FloatArray
    lateral_sum: FloatArray
    count: FloatArray

    @classmethod
    def empty(cls, shape: tuple[int, int, int]) -> DisagreementEvidence:
        return cls(np.zeros(shape), np.zeros(shape), np.zeros(shape))

    @property
    def parallel_mean(self) -> FloatArray:
        return np.divide(
            self.parallel_sum,
            self.count,
            out=np.zeros_like(self.parallel_sum),
            where=self.count > 0,
        )

    @property
    def lateral_mean(self) -> FloatArray:
        return np.divide(
            self.lateral_sum,
            self.count,
            out=np.zeros_like(self.lateral_sum),
            where=self.count > 0,
        )


@dataclass(frozen=True)
class CalibrationReport:
    count: int
    parallel_scale_update: float
    lateral_scale_update: float
    parallel_normalized_mse: float
    lateral_normalized_mse: float

    def __post_init__(self) -> None:
        count = int(self.count)
        values = np.asarray(
            [
                self.parallel_scale_update,
                self.lateral_scale_update,
                self.parallel_normalized_mse,
                self.lateral_normalized_mse,
            ],
            dtype=np.float64,
        )
        if count < 1:
            raise ValueError("calibration report count must be positive")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("calibration report values must be finite and non-negative")
        if values[0] <= 0.0 or values[1] <= 0.0:
            raise ValueError("calibration scale updates must be strictly positive")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "parallel_scale_update", float(values[0]))
        object.__setattr__(self, "lateral_scale_update", float(values[1]))
        object.__setattr__(self, "parallel_normalized_mse", float(values[2]))
        object.__setattr__(self, "lateral_normalized_mse", float(values[3]))


@dataclass(frozen=True)
class DepthDisagreementModel:
    """Depth-aware variance model augmented with online overlap disagreement."""

    parallel_floor: float = 2.5e-4
    parallel_depth_coefficient: float = 2.5e-4
    lateral_floor: float = 2.5e-5
    lateral_depth_coefficient: float = 2.5e-5
    disagreement_gain: float = 0.5
    parallel_scale: float = 1.0
    lateral_scale: float = 1.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(
            [
                self.parallel_floor,
                self.parallel_depth_coefficient,
                self.lateral_floor,
                self.lateral_depth_coefficient,
                self.disagreement_gain,
            ],
            dtype=np.float64,
        )
        scales = np.asarray([self.parallel_scale, self.lateral_scale], dtype=np.float64)
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
            raise ValueError("uncertainty coefficients must be finite and non-negative")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("uncertainty scales must be finite and strictly positive")
        for name, value in (
            ("parallel_floor", coefficients[0]),
            ("parallel_depth_coefficient", coefficients[1]),
            ("lateral_floor", coefficients[2]),
            ("lateral_depth_coefficient", coefficients[3]),
            ("disagreement_gain", coefficients[4]),
            ("parallel_scale", scales[0]),
            ("lateral_scale", scales[1]),
        ):
            object.__setattr__(self, name, float(value))

    def predict(
        self,
        window: PredictionWindow,
        evidence: DisagreementEvidence | None = None,
    ) -> StructuredCovariance:
        depth_squared = np.sum(window.point_map**2, axis=-1)
        parallel = self.parallel_floor + self.parallel_depth_coefficient * depth_squared
        lateral = self.lateral_floor + self.lateral_depth_coefficient * depth_squared
        if evidence is not None:
            if evidence.count.shape != window.shape:
                raise ValueError("disagreement evidence shape does not match window")
            # Pairwise disagreement is split between the two contributing estimates.
            parallel += 0.5 * self.disagreement_gain * evidence.parallel_mean
            lateral += 0.5 * self.disagreement_gain * evidence.lateral_mean
        # Calibration is a multiplicative correction of the complete predictive
        # variance, including the overlap-disagreement contribution.
        parallel *= self.parallel_scale
        lateral *= self.lateral_scale
        return StructuredCovariance(
            ray_directions=window.rays(),
            parallel_variance=np.maximum(parallel, np.finfo(np.float64).eps),
            lateral_variance=np.maximum(lateral, np.finfo(np.float64).eps),
        )

    def calibrate(
        self,
        errors: FloatArray,
        covariance: StructuredCovariance,
        *,
        mask: NDArray[np.bool_] | None = None,
        trim_quantile: float = 0.99,
    ) -> tuple[DepthDisagreementModel, CalibrationReport]:
        """Scale variances using held-out prediction residuals."""

        errors = np.asarray(errors, dtype=np.float64)
        if errors.shape != covariance.ray_directions.shape:
            raise ValueError("errors and covariance rays must have matching shapes")
        if not np.isfinite(trim_quantile) or not 0.0 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must lie in (0, 1]")
        active = np.all(np.isfinite(errors), axis=-1)
        if mask is not None:
            supplied_mask = np.asarray(mask, dtype=bool)
            if supplied_mask.shape != active.shape:
                raise ValueError("calibration mask shape does not match errors")
            active &= supplied_mask
        if not np.any(active):
            raise ValueError("calibration set has no valid residuals")

        ray = covariance.ray_directions
        parallel_error = np.sum(errors * ray, axis=-1)
        total_squared = np.sum(errors**2, axis=-1)
        lateral_squared = np.maximum(total_squared - parallel_error**2, 0.0)
        parallel_ratio = parallel_error[active] ** 2 / covariance.parallel_variance[active]
        lateral_ratio = lateral_squared[active] / (2.0 * covariance.lateral_variance[active])

        def trimmed_mean(values: FloatArray) -> float:
            upper = float(np.quantile(values, trim_quantile))
            return max(float(np.mean(np.minimum(values, upper))), 1e-6)

        parallel_update = trimmed_mean(parallel_ratio)
        lateral_update = trimmed_mean(lateral_ratio)
        calibrated = replace(
            self,
            parallel_scale=self.parallel_scale * parallel_update,
            lateral_scale=self.lateral_scale * lateral_update,
        )
        return calibrated, CalibrationReport(
            count=int(np.count_nonzero(active)),
            parallel_scale_update=parallel_update,
            lateral_scale_update=lateral_update,
            parallel_normalized_mse=float(np.mean(parallel_ratio)),
            lateral_normalized_mse=float(np.mean(lateral_ratio)),
        )


def accumulate_disagreement(
    windows: dict[str, PredictionWindow],
    alignments: list[WindowAlignment],
) -> dict[str, DisagreementEvidence]:
    """Project pairwise overlap residuals into along-ray and lateral components."""

    evidence = {
        window_id: DisagreementEvidence.empty(window.shape) for window_id, window in windows.items()
    }
    for alignment in alignments:
        reference = windows[alignment.reference_id]
        moving = windows[alignment.moving_id]
        transform = alignment.result.transform
        for frame in alignment.common_frames:
            reference_index = reference.local_index(int(frame))
            moving_index = moving.local_index(int(frame))
            mask = reference.valid_mask[reference_index] & moving.valid_mask[moving_index]
            if not np.any(mask):
                continue
            target = reference.point_map[reference_index]
            source_aligned = transform.transform_points(moving.point_map[moving_index])
            residual = target - source_aligned
            reference_rays = reference.rays()[reference_index]
            moving_rays = transform.rotate_directions(moving.rays()[moving_index])
            rays = reference_rays + moving_rays
            ray_norm = np.linalg.norm(rays, axis=-1, keepdims=True)
            rays = np.divide(rays, ray_norm, out=reference_rays.copy(), where=ray_norm > 1e-12)
            parallel = np.sum(residual * rays, axis=-1) ** 2
            lateral = 0.5 * np.maximum(np.sum(residual**2, axis=-1) - parallel, 0.0)

            for window_id, local_index in (
                (reference.window_id, reference_index),
                (moving.window_id, moving_index),
            ):
                item = evidence[window_id]
                item.parallel_sum[local_index][mask] += parallel[mask]
                item.lateral_sum[local_index][mask] += lateral[mask]
                item.count[local_index][mask] += 1.0
    return evidence
