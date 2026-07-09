"""Accuracy, drift, seam, and marginal calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import NDArray

from .fusion import FusedSequence


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TruthSequence:
    frame_indices: NDArray[np.integer]
    point_map: FloatArray
    valid_mask: NDArray[np.bool_]
    scene_flow: FloatArray | None = None
    deform_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        frames = np.asarray(self.frame_indices, dtype=np.int64)
        points = np.asarray(self.point_map, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=bool)
        if points.shape[:-1] != mask.shape or points.shape[-1] != 3:
            raise ValueError("truth point-map and mask shapes are inconsistent")
        if frames.shape != (points.shape[0],):
            raise ValueError("truth frame indices do not match point-map length")
        if (self.scene_flow is None) != (self.deform_mask is None):
            raise ValueError("truth scene_flow and deform_mask must be paired")


@dataclass(frozen=True)
class SequenceMetrics:
    metric_point_rmse: float
    metric_endpoint_point_rmse: float
    point_rmse: float
    endpoint_point_rmse: float
    drift_slope: float
    seam_rmse: float
    flow_epe: float | None
    coverage_95: float
    gaussian_nll: float
    mean_mahalanobis_squared: float
    evaluated_points: int
    fitted_alignment_scale: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _scale_translation_alignment(source: FloatArray, target: FloatArray) -> tuple[float, FloatArray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    denominator = float(np.sum(source_centered**2))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0, target_mean - source_mean
    scale = float(np.sum(source_centered * target_centered) / denominator)
    scale = max(scale, np.finfo(np.float64).eps)
    return scale, target_mean - scale * source_mean


def evaluate_sequence(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    boundary_frames: list[int] | None = None,
    align_scale_translation: bool = True,
) -> SequenceMetrics:
    """Evaluate one sequence after at most one global scale/translation alignment."""

    common_frames = np.intersect1d(prediction.frame_indices, truth.frame_indices)
    if common_frames.size == 0:
        raise ValueError("prediction and truth have no common frames")
    prediction_indices = [int(np.searchsorted(prediction.frame_indices, frame)) for frame in common_frames]
    truth_indices = [int(np.searchsorted(truth.frame_indices, frame)) for frame in common_frames]
    predicted_points = prediction.point_map[prediction_indices].copy()
    predicted_covariance = prediction.point_covariance[prediction_indices].copy()
    predicted_mask = prediction.valid_mask[prediction_indices]
    truth_points = truth.point_map[truth_indices]
    truth_mask = truth.valid_mask[truth_indices]
    active = predicted_mask & truth_mask
    if not np.any(active):
        raise ValueError("prediction and truth have no jointly valid points")

    metric_errors = predicted_points - truth_points
    metric_squared_norm = np.sum(metric_errors**2, axis=-1)
    metric_frame_rmse = np.array(
        [
            np.sqrt(np.mean(metric_squared_norm[index][active[index]]))
            if np.any(active[index])
            else np.nan
            for index in range(common_frames.size)
        ]
    )
    metric_finite_frames = np.flatnonzero(np.isfinite(metric_frame_rmse))
    metric_point_rmse = float(np.sqrt(np.mean(metric_squared_norm[active])))
    metric_endpoint_point_rmse = float(metric_frame_rmse[metric_finite_frames[-1]])

    scale = 1.0
    translation = np.zeros(3)
    if align_scale_translation:
        scale, translation = _scale_translation_alignment(
            predicted_points[active], truth_points[active]
        )
        predicted_points = scale * predicted_points + translation
        predicted_covariance *= scale**2

    errors = predicted_points - truth_points
    squared_norm = np.sum(errors**2, axis=-1)
    point_rmse = float(np.sqrt(np.mean(squared_norm[active])))
    frame_rmse = np.array(
        [
            np.sqrt(np.mean(squared_norm[index][active[index]]))
            if np.any(active[index])
            else np.nan
            for index in range(common_frames.size)
        ]
    )
    finite_frames = np.isfinite(frame_rmse)
    if np.count_nonzero(finite_frames) >= 2:
        drift_slope = float(
            np.polyfit(common_frames[finite_frames], frame_rmse[finite_frames], deg=1)[0]
        )
    else:
        drift_slope = 0.0
    endpoint_point_rmse = float(frame_rmse[np.flatnonzero(finite_frames)[-1]])

    seam_values: list[float] = []
    common_position = {int(frame): index for index, frame in enumerate(common_frames)}
    for boundary in boundary_frames or []:
        if boundary not in common_position or boundary - 1 not in common_position:
            continue
        current = common_position[boundary]
        previous = common_position[boundary - 1]
        seam_mask = active[current] & active[previous]
        if np.any(seam_mask):
            error_jump = errors[current][seam_mask] - errors[previous][seam_mask]
            seam_values.extend(np.sum(error_jump**2, axis=-1).tolist())
    seam_rmse = float(np.sqrt(np.mean(seam_values))) if seam_values else 0.0

    active_covariance = predicted_covariance[active]
    active_error = errors[active]
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (active_covariance + np.swapaxes(active_covariance, -1, -2))
    )
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    inverse_covariance = np.einsum(
        "...ij,...j,...kj->...ik", eigenvectors, 1.0 / eigenvalues, eigenvectors
    )
    mahalanobis = np.einsum(
        "...i,...ij,...j->...", active_error, inverse_covariance, active_error
    )
    log_determinant = np.sum(np.log(eigenvalues), axis=-1)
    nll = 0.5 * (3.0 * np.log(2.0 * np.pi) + log_determinant + mahalanobis)

    flow_epe: float | None = None
    if prediction.scene_flow is not None and truth.scene_flow is not None:
        predicted_flow = scale * prediction.scene_flow[prediction_indices]
        flow_mask = (
            prediction.deform_mask[prediction_indices]
            & truth.deform_mask[truth_indices]
        )
        if np.any(flow_mask):
            flow_epe = float(
                np.mean(
                    np.linalg.norm(
                        predicted_flow[flow_mask] - truth.scene_flow[truth_indices][flow_mask],
                        axis=-1,
                    )
                )
            )

    return SequenceMetrics(
        metric_point_rmse=metric_point_rmse,
        metric_endpoint_point_rmse=metric_endpoint_point_rmse,
        point_rmse=point_rmse,
        endpoint_point_rmse=endpoint_point_rmse,
        drift_slope=drift_slope,
        seam_rmse=seam_rmse,
        flow_epe=flow_epe,
        coverage_95=float(np.mean(mahalanobis <= 7.814727903251179)),
        gaussian_nll=float(np.mean(nll)),
        mean_mahalanobis_squared=float(np.mean(mahalanobis)),
        evaluated_points=int(np.count_nonzero(active)),
        fitted_alignment_scale=scale,
    )
