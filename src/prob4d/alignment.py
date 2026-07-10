"""Robust relative-gauge estimation from overlapping prediction windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import PredictionWindow
from .sim3 import Sim3, skew, so3_log, so3_right_jacobian

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class AlignmentResult:
    """A robust estimate mapping source points into target coordinates."""

    transform: Sim3
    covariance: FloatArray
    residual_rms: float
    inlier_fraction: float
    num_correspondences: int

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("alignment covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class WindowAlignment:
    """Relative transform from ``moving_id`` into ``reference_id`` coordinates."""

    reference_id: str
    moving_id: str
    common_frames: NDArray[np.integer]
    result: AlignmentResult


def _weighted_umeyama(source: FloatArray, target: FloatArray, weights: FloatArray) -> Sim3:
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= np.finfo(np.float64).eps:
        raise ValueError("alignment weights have zero total mass")
    normalized = weights / weight_sum
    source_mean = np.sum(normalized[:, None] * source, axis=0)
    target_mean = np.sum(normalized[:, None] * target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (normalized[:, None] * target_centered).T @ source_centered
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ correction @ right_transpose
    source_variance = float(np.sum(normalized * np.sum(source_centered**2, axis=1)))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("source correspondences have no spatial extent")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("estimated alignment scale is not positive")
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def _alignment_covariance(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    transform: Sim3,
) -> FloatArray:
    transformed = transform.transform_points(source)
    residuals = target - transformed
    information = np.zeros((7, 7), dtype=np.float64)
    rotation_jacobian = so3_right_jacobian(so3_log(transform.rotation))
    for point, weight in zip(source, weights, strict=True):
        scaled_rotated = transform.scale * (transform.rotation @ point)
        jacobian = np.empty((3, 7), dtype=np.float64)
        jacobian[:, 0] = scaled_rotated
        jacobian[:, 1:4] = -transform.scale * transform.rotation @ skew(point) @ rotation_jacobian
        jacobian[:, 4:7] = np.eye(3)
        information += float(weight) * jacobian.T @ jacobian

    active = int(np.count_nonzero(weights > 1e-3))
    degrees_of_freedom = max(1, 3 * active - 7)
    variance = float(np.sum(weights[:, None] * residuals**2) / degrees_of_freedom)
    covariance = variance * np.linalg.pinv(information, rcond=1e-10)
    floor = np.diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-12, 1e-12, 1e-12])
    return 0.5 * (covariance + covariance.T) + floor


def estimate_sim3_robust(
    source: FloatArray,
    target: FloatArray,
    *,
    weights: FloatArray | None = None,
    max_iterations: int = 20,
    huber_multiplier: float = 2.5,
    tolerance: float = 1e-8,
) -> AlignmentResult:
    """Estimate a similarity transform with Huber iteratively reweighted least squares."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if source.shape[0] < 4:
        raise ValueError("at least four correspondences are required")
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 4:
        raise ValueError("at least four finite correspondences are required")

    if weights is None:
        base_weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != finite.shape:
            raise ValueError("weights must have shape (N,)")
        base_weights = supplied[finite]
        if np.any(base_weights < 0) or not np.all(np.isfinite(base_weights)):
            raise ValueError("weights must be finite and non-negative")

    robust_weights = base_weights.copy()
    previous_vector: FloatArray | None = None
    cutoff = np.inf
    transform = Sim3.identity()
    for _ in range(max_iterations):
        transform = _weighted_umeyama(source, target, robust_weights)
        residual_norms = np.linalg.norm(target - transform.transform_points(source), axis=1)
        median = float(np.median(residual_norms))
        mad = float(np.median(np.abs(residual_norms - median)))
        robust_scale = max(1.4826 * mad, np.finfo(np.float64).eps)
        cutoff = max(median + huber_multiplier * robust_scale, np.finfo(np.float64).eps)
        huber_weights = np.minimum(1.0, cutoff / np.maximum(residual_norms, cutoff))
        robust_weights = base_weights * huber_weights
        current_vector = transform.as_vector()
        if (
            previous_vector is not None
            and np.linalg.norm(current_vector - previous_vector) < tolerance
        ):
            break
        previous_vector = current_vector

    residuals = target - transform.transform_points(source)
    residual_norms = np.linalg.norm(residuals, axis=1)
    covariance = _alignment_covariance(source, target, robust_weights, transform)
    weighted_squared_error = float(np.sum(robust_weights * residual_norms**2))
    weight_sum = max(float(robust_weights.sum()), np.finfo(np.float64).eps)
    return AlignmentResult(
        transform=transform,
        covariance=covariance,
        residual_rms=float(np.sqrt(weighted_squared_error / weight_sum)),
        inlier_fraction=float(np.mean(residual_norms <= cutoff)),
        num_correspondences=int(source.shape[0]),
    )


def overlapping_correspondences(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int = 100_000,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, NDArray[np.integer]]:
    """Collect same-frame, same-pixel points from two decoded windows."""

    if reference.shape[1:] != moving.shape[1:]:
        raise ValueError("overlapping windows must use the same spatial resolution")
    common_frames = reference.common_frames(moving)
    if common_frames.size == 0:
        raise ValueError("windows do not overlap")

    source_parts: list[FloatArray] = []
    target_parts: list[FloatArray] = []
    for frame in common_frames:
        reference_index = reference.local_index(int(frame))
        moving_index = moving.local_index(int(frame))
        mask = reference.valid_mask[reference_index] & moving.valid_mask[moving_index]
        source_parts.append(moving.point_map[moving_index][mask])
        target_parts.append(reference.point_map[reference_index][mask])

    source = np.concatenate(source_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    if source.shape[0] < 4:
        raise ValueError("overlap has fewer than four valid point correspondences")
    if source.shape[0] > max_correspondences:
        generator = np.random.default_rng(seed)
        selection = np.sort(
            generator.choice(source.shape[0], size=max_correspondences, replace=False)
        )
        source = source[selection]
        target = target[selection]
    return source, target, common_frames


def align_windows(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int = 100_000,
    seed: int = 0,
) -> WindowAlignment:
    """Estimate the transform from a moving window gauge to a reference gauge."""

    source, target, common_frames = overlapping_correspondences(
        reference,
        moving,
        max_correspondences=max_correspondences,
        seed=seed,
    )
    return WindowAlignment(
        reference_id=reference.window_id,
        moving_id=moving.window_id,
        common_frames=common_frames,
        result=estimate_sim3_robust(source, target),
    )
