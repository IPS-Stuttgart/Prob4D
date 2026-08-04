"""Compact anisotropic uncertainty for dense geometry and motion."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ._scientific_scalars import require_finite_real, require_genuine_integer
from .alignment import WindowAlignment
from .calibration_aggregation import (
    GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
    UPPER_WINSORIZED_MEAN_V1,
    upper_winsorized_mean,
)
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
        count = require_genuine_integer(
            self.count,
            name="count",
            minimum=1,
        )
        parallel_scale_update = require_finite_real(
            self.parallel_scale_update,
            name="parallel_scale_update",
            minimum=0.0,
            minimum_inclusive=False,
        )
        lateral_scale_update = require_finite_real(
            self.lateral_scale_update,
            name="lateral_scale_update",
            minimum=0.0,
            minimum_inclusive=False,
        )
        parallel_normalized_mse = require_finite_real(
            self.parallel_normalized_mse,
            name="parallel_normalized_mse",
            minimum=0.0,
        )
        lateral_normalized_mse = require_finite_real(
            self.lateral_normalized_mse,
            name="lateral_normalized_mse",
            minimum=0.0,
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(
            self,
            "parallel_scale_update",
            parallel_scale_update,
        )
        object.__setattr__(self, "lateral_scale_update", lateral_scale_update)
        object.__setattr__(
            self,
            "parallel_normalized_mse",
            parallel_normalized_mse,
        )
        object.__setattr__(
            self,
            "lateral_normalized_mse",
            lateral_normalized_mse,
        )


@dataclass(frozen=True)
class GroupBalancedCalibrationReport:
    """Equal-group calibration audit with per-group scale diagnostics."""

    count: int
    trim_quantile: float
    parallel_scale_update: float
    lateral_scale_update: float
    parallel_normalized_mse: float
    lateral_normalized_mse: float
    group_ids: tuple[str, ...]
    group_counts: tuple[int, ...]
    group_parallel_scale_updates: tuple[float, ...]
    group_lateral_scale_updates: tuple[float, ...]
    group_parallel_normalized_mse: tuple[float, ...]
    group_lateral_normalized_mse: tuple[float, ...]

    def __post_init__(self) -> None:
        count = require_genuine_integer(
            self.count,
            name="count",
            minimum=1,
        )
        trim_quantile = require_finite_real(
            self.trim_quantile,
            name="trim_quantile",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        if not isinstance(self.group_ids, tuple) or any(
            type(value) is not str for value in self.group_ids
        ):
            raise TypeError("group_ids must be a canonical tuple of strings")
        group_ids = self.group_ids
        group_counts = tuple(
            require_genuine_integer(
                value,
                name=f"group_counts[{index}]",
                minimum=1,
            )
            for index, value in enumerate(self.group_counts)
        )
        raw_group_values = (
            (
                "group_parallel_scale_updates",
                self.group_parallel_scale_updates,
                True,
            ),
            (
                "group_lateral_scale_updates",
                self.group_lateral_scale_updates,
                True,
            ),
            (
                "group_parallel_normalized_mse",
                self.group_parallel_normalized_mse,
                False,
            ),
            (
                "group_lateral_normalized_mse",
                self.group_lateral_normalized_mse,
                False,
            ),
        )
        group_values = tuple(
            tuple(
                require_finite_real(
                    value,
                    name=f"{name}[{index}]",
                    minimum=0.0,
                    minimum_inclusive=not strictly_positive,
                )
                for index, value in enumerate(values)
            )
            for name, values, strictly_positive in raw_group_values
        )
        lengths = {
            len(group_ids),
            len(group_counts),
            *(len(values) for values in group_values),
        }
        if lengths != {len(group_ids)} or not group_ids:
            raise ValueError("group calibration fields must have one non-empty shared length")
        if len(set(group_ids)) != len(group_ids) or any(not value for value in group_ids):
            raise ValueError("group calibration IDs must be non-empty and unique")
        if group_ids != tuple(sorted(group_ids)):
            raise ValueError("group calibration IDs must use canonical sorted order")
        if sum(group_counts) != count:
            raise ValueError("group calibration counts must be positive and sum to count")
        aggregate = (
            require_finite_real(
                self.parallel_scale_update,
                name="parallel_scale_update",
                minimum=0.0,
                minimum_inclusive=False,
            ),
            require_finite_real(
                self.lateral_scale_update,
                name="lateral_scale_update",
                minimum=0.0,
                minimum_inclusive=False,
            ),
            require_finite_real(
                self.parallel_normalized_mse,
                name="parallel_normalized_mse",
                minimum=0.0,
            ),
            require_finite_real(
                self.lateral_normalized_mse,
                name="lateral_normalized_mse",
                minimum=0.0,
            ),
        )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "trim_quantile", trim_quantile)
        object.__setattr__(self, "parallel_scale_update", aggregate[0])
        object.__setattr__(self, "lateral_scale_update", aggregate[1])
        object.__setattr__(self, "parallel_normalized_mse", aggregate[2])
        object.__setattr__(self, "lateral_normalized_mse", aggregate[3])
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "group_counts", group_counts)
        object.__setattr__(self, "group_parallel_scale_updates", group_values[0])
        object.__setattr__(self, "group_lateral_scale_updates", group_values[1])
        object.__setattr__(self, "group_parallel_normalized_mse", group_values[2])
        object.__setattr__(self, "group_lateral_normalized_mse", group_values[3])

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def winsor_quantile(self) -> float:
        """Explicit alias for the legacy serialized ``trim_quantile`` field."""

        return self.trim_quantile

    @property
    def aggregation_semantics(self) -> str:
        return GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation_semantics,
            "count": self.count,
            "group_count": self.group_count,
            "trim_quantile": self.trim_quantile,
            "winsor_quantile": self.winsor_quantile,
            "parallel_scale_update": self.parallel_scale_update,
            "lateral_scale_update": self.lateral_scale_update,
            "parallel_normalized_mse": self.parallel_normalized_mse,
            "lateral_normalized_mse": self.lateral_normalized_mse,
            "groups": [
                {
                    "group_id": group_id,
                    "count": count,
                    "parallel_scale_update": parallel_update,
                    "lateral_scale_update": lateral_update,
                    "parallel_normalized_mse": parallel_mse,
                    "lateral_normalized_mse": lateral_mse,
                }
                for (
                    group_id,
                    count,
                    parallel_update,
                    lateral_update,
                    parallel_mse,
                    lateral_mse,
                ) in zip(
                    self.group_ids,
                    self.group_counts,
                    self.group_parallel_scale_updates,
                    self.group_lateral_scale_updates,
                    self.group_parallel_normalized_mse,
                    self.group_lateral_normalized_mse,
                    strict=True,
                )
            ],
        }


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

    @property
    def calibration_aggregation_semantics(self) -> str:
        """Aggregation used by pooled and per-group residual scale updates."""

        return UPPER_WINSORIZED_MEAN_V1

    def calibrate(
        self,
        errors: FloatArray,
        covariance: StructuredCovariance,
        *,
        mask: NDArray[np.bool_] | None = None,
        trim_quantile: float = 0.99,
    ) -> tuple[DepthDisagreementModel, CalibrationReport]:
        """Scale variances using upper-winsorized held-out residual ratios.

        ``trim_quantile`` is retained as a serialized compatibility name; rows
        above that quantile are clipped rather than removed.
        """

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

        parallel_update = upper_winsorized_mean(parallel_ratio, quantile=trim_quantile)
        lateral_update = upper_winsorized_mean(lateral_ratio, quantile=trim_quantile)
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

    def calibrate_group_balanced(
        self,
        errors: FloatArray,
        covariance: StructuredCovariance,
        group_ids: np.ndarray,
        *,
        mask: NDArray[np.bool_] | None = None,
        trim_quantile: float = 0.99,
    ) -> tuple[DepthDisagreementModel, GroupBalancedCalibrationReport]:
        """Scale variances while assigning equal mass to each declared group.

        Ratios are upper-winsorized independently inside every group. Values above
        the selected quantile are clipped, not removed. The final parallel and
        lateral updates are arithmetic means of those per-group values, so a
        long or densely sampled sequence cannot dominate merely by contributing
        more rows. Group IDs are canonicalized to sorted strings, and rows with
        invalid errors or a false mask value contribute to no group.
        """

        errors = np.asarray(errors, dtype=np.float64)
        if errors.shape != covariance.ray_directions.shape:
            raise ValueError("errors and covariance rays must have matching shapes")
        if not np.isfinite(trim_quantile) or not 0.0 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must lie in (0, 1]")
        groups = np.asarray(group_ids)
        if groups.shape != errors.shape[:-1]:
            raise ValueError("group_ids must match the calibration sample grid")
        active = np.all(np.isfinite(errors), axis=-1)
        if mask is not None:
            supplied_mask = np.asarray(mask, dtype=bool)
            if supplied_mask.shape != active.shape:
                raise ValueError("calibration mask shape does not match errors")
            active &= supplied_mask
        if not np.any(active):
            raise ValueError("calibration set has no valid residuals")

        active_groups = tuple(
            "" if value is None else str(value).strip() for value in groups[active].reshape(-1)
        )
        if any(not value for value in active_groups):
            raise ValueError("active group IDs must be non-empty")
        group_array = np.asarray(active_groups, dtype=str)
        canonical_group_ids = tuple(sorted(set(active_groups)))

        ray = covariance.ray_directions
        parallel_error = np.sum(errors * ray, axis=-1)
        total_squared = np.sum(errors**2, axis=-1)
        lateral_squared = np.maximum(total_squared - parallel_error**2, 0.0)
        parallel_ratio = (
            parallel_error[active] ** 2 / covariance.parallel_variance[active]
        ).reshape(-1)
        lateral_ratio = (
            lateral_squared[active] / (2.0 * covariance.lateral_variance[active])
        ).reshape(-1)

        def ordered_mean(values: np.ndarray) -> float:
            return float(np.mean(np.sort(np.asarray(values, dtype=np.float64))))

        counts: list[int] = []
        parallel_updates: list[float] = []
        lateral_updates: list[float] = []
        parallel_mse: list[float] = []
        lateral_mse: list[float] = []
        for group_id in canonical_group_ids:
            selected = group_array == group_id
            counts.append(int(np.count_nonzero(selected)))
            parallel_updates.append(
                upper_winsorized_mean(
                    parallel_ratio[selected],
                    quantile=trim_quantile,
                    canonicalize=True,
                )
            )
            lateral_updates.append(
                upper_winsorized_mean(
                    lateral_ratio[selected],
                    quantile=trim_quantile,
                    canonicalize=True,
                )
            )
            parallel_mse.append(ordered_mean(parallel_ratio[selected]))
            lateral_mse.append(ordered_mean(lateral_ratio[selected]))

        parallel_update = ordered_mean(np.asarray(parallel_updates))
        lateral_update = ordered_mean(np.asarray(lateral_updates))
        calibrated = replace(
            self,
            parallel_scale=self.parallel_scale * parallel_update,
            lateral_scale=self.lateral_scale * lateral_update,
        )
        report = GroupBalancedCalibrationReport(
            count=int(np.count_nonzero(active)),
            trim_quantile=trim_quantile,
            parallel_scale_update=parallel_update,
            lateral_scale_update=lateral_update,
            parallel_normalized_mse=ordered_mean(np.asarray(parallel_mse)),
            lateral_normalized_mse=ordered_mean(np.asarray(lateral_mse)),
            group_ids=canonical_group_ids,
            group_counts=tuple(counts),
            group_parallel_scale_updates=tuple(parallel_updates),
            group_lateral_scale_updates=tuple(lateral_updates),
            group_parallel_normalized_mse=tuple(parallel_mse),
            group_lateral_normalized_mse=tuple(lateral_mse),
        )
        return calibrated, report


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
