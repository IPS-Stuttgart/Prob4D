"""Accuracy, drift, seam, and marginal calibration metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .covariance import covariance_statistics
from .fusion import FusedSequence

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TruthSequence:
    """A validated immutable dense truth sequence.

    Public construction defensively copies every NumPy field, normalizes it to
    the canonical dtype, validates active geometry, and makes the retained arrays
    read-only. Inactive point and flow entries may contain arbitrary sentinels;
    only entries selected by their corresponding masks are required to be finite.
    """

    frame_indices: NDArray[np.integer]
    point_map: FloatArray
    valid_mask: NDArray[np.bool_]
    scene_flow: FloatArray | None = None
    deform_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        raw_frames = np.asarray(self.frame_indices)
        if not np.issubdtype(raw_frames.dtype, np.integer):
            raise ValueError("truth frame_indices must contain integers")
        if np.any(raw_frames < 0) or np.any(raw_frames > np.iinfo(np.int64).max):
            raise ValueError("truth frame_indices must fit the non-negative int64 range")
        frames = _readonly_copy(raw_frames, dtype=np.int64)
        points = _readonly_copy(self.point_map, dtype=np.float64)
        mask = _readonly_copy(self.valid_mask, dtype=bool)
        if frames.ndim != 1 or frames.size == 0:
            raise ValueError(
                "truth frame_indices must be a non-empty one-dimensional array"
            )
        if np.any(np.diff(frames) <= 0):
            raise ValueError("truth frame_indices must be strictly increasing")
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("truth point_map must have shape (T, H, W, 3)")
        if mask.shape != points.shape[:-1]:
            raise ValueError("truth valid_mask must have shape (T, H, W)")
        if frames.shape != (points.shape[0],):
            raise ValueError("truth frame_indices do not match point_map length")
        _validate_finite_active_vectors(
            points,
            active_mask=mask,
            name="active truth point_map",
        )
        if (self.scene_flow is None) != (self.deform_mask is None):
            raise ValueError("truth scene_flow and deform_mask must be paired")

        flow: np.ndarray | None = None
        flow_mask: np.ndarray | None = None
        if self.scene_flow is not None:
            assert self.deform_mask is not None
            flow = _readonly_copy(self.scene_flow, dtype=np.float64)
            flow_mask = _readonly_copy(self.deform_mask, dtype=bool)
            if flow.shape != points.shape:
                raise ValueError("truth scene_flow must match point_map shape")
            if flow_mask.shape != mask.shape:
                raise ValueError("truth deform_mask must match valid_mask shape")
            _validate_finite_active_vectors(
                flow,
                active_mask=flow_mask & mask,
                name="active truth scene_flow",
            )

        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "point_map", points)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "scene_flow", flow)
        object.__setattr__(self, "deform_mask", flow_mask)


@dataclass(frozen=True)
class SequenceMetrics:
    # Preserve the historical constructor order for downstream compatibility.
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

    # Extended provider scorecard.
    metric_frame_balanced_point_rmse: float
    frame_balanced_point_rmse: float
    coverage_50: float
    coverage_80: float
    coverage_90: float
    coverage_shortfall_95: float
    coverage_calibration_error: float
    median_mahalanobis_squared: float
    mean_covariance_trace: float
    mean_covariance_log_determinant: float
    uncertainty_error_spearman: float
    mean_relative_error: float
    relative_error_retained_50: float
    relative_error_retained_80: float
    selective_gain_80: float
    selective_oracle_fraction_80: float
    risk_coverage_auc: float
    evaluated_frames: int
    evaluated_flow_points: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class UncertaintyDiagnostics:
    # Preserve the historical constructor order for downstream compatibility.
    count: int
    coverage_50: float
    coverage_80: float
    coverage_90: float
    coverage_95: float
    coverage_shortfall_95: float
    coverage_calibration_error: float
    mean_mahalanobis_squared: float
    median_mahalanobis_squared: float
    gaussian_nll: float
    uncertainty_error_spearman: float
    mean_relative_error: float
    relative_error_retained_80: float
    relative_error_oracle_80: float
    selective_gain_80: float
    selective_oracle_fraction_80: float

    # Extended width and selective-risk diagnostics.
    mean_covariance_trace: float
    mean_covariance_log_determinant: float
    relative_error_retained_50: float
    risk_coverage_auc: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


_CHI_SQUARED_3 = {
    0.50: 2.3659738843753377,
    0.80: 4.64162767608745,
    0.90: 6.251388631170325,
    0.95: 7.814727903251179,
}


def _readonly_copy(value: ArrayLike, *, dtype: DTypeLike) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validate_finite_active_vectors(
    values: np.ndarray,
    *,
    active_mask: NDArray[np.bool_],
    name: str,
    chunk_size: int = 65_536,
) -> None:
    if chunk_size < 1:
        raise ValueError("truth vector validation chunk_size must be positive")
    flat_values = values.reshape(-1, values.shape[-1])
    flat_active = np.asarray(active_mask, dtype=bool).reshape(-1)
    for start in range(0, flat_values.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_values.shape[0])
        selected = flat_active[start:stop]
        if np.any(selected) and not np.all(np.isfinite(flat_values[start:stop][selected])):
            raise ValueError(f"{name} entries must be finite")


def _validated_support_mask(
    value: NDArray[np.bool_] | None,
    *,
    truth: TruthSequence,
    name: str,
) -> NDArray[np.bool_] | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    if mask.shape != truth.valid_mask.shape:
        raise ValueError(f"{name} must match truth valid_mask shape")
    return mask


def _rankdata(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _expected_risk_at_count(
    risks: FloatArray,
    uncertainty_scores: FloatArray,
    retain_count: int,
) -> float:
    """Return tie-invariant expected risk at one fixed retained sample count."""

    risks = np.asarray(risks, dtype=np.float64)
    scores = np.asarray(uncertainty_scores, dtype=np.float64)
    if risks.shape != scores.shape or risks.ndim != 1:
        raise ValueError("risks and uncertainty_scores must be equal-length vectors")
    if not 1 <= retain_count <= risks.size:
        raise ValueError("retain_count must select at least one and at most all samples")
    threshold = np.partition(scores, retain_count - 1)[retain_count - 1]
    lower = scores < threshold
    tied = scores == threshold
    lower_count = int(np.count_nonzero(lower))
    tied_count = int(np.count_nonzero(tied))
    tied_to_retain = retain_count - lower_count
    if tied_count < tied_to_retain or tied_to_retain < 1:
        raise RuntimeError("invalid uncertainty tie partition")
    retained_sum = float(np.sum(risks[lower]))
    retained_sum += float(tied_to_retain / tied_count) * float(np.sum(risks[tied]))
    return retained_sum / retain_count


def _tie_aware_risk_coverage_auc(
    risks: FloatArray,
    uncertainty_scores: FloatArray,
) -> float:
    """Integrate expected selective risk without depending on tie order."""

    risks = np.asarray(risks, dtype=np.float64)
    scores = np.asarray(uncertainty_scores, dtype=np.float64)
    if risks.shape != scores.shape or risks.ndim != 1 or risks.size == 0:
        raise ValueError("risks and uncertainty_scores must be nonempty equal vectors")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_risks = risks[order]
    retained_count = 0
    retained_sum = 0.0
    integrated_risk = 0.0
    start = 0
    while start < risks.size:
        stop = start + 1
        while stop < risks.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        group = sorted_risks[start:stop]
        group_mean = float(np.mean(group))
        within_group_counts = np.arange(1, group.size + 1, dtype=np.float64)
        integrated_risk += float(
            np.sum(
                (retained_sum + within_group_counts * group_mean)
                / (retained_count + within_group_counts)
            )
        )
        retained_count += group.size
        retained_sum += float(np.sum(group))
        start = stop
    return integrated_risk / risks.size


def uncertainty_diagnostics(
    errors: FloatArray,
    covariances: FloatArray,
    target_norms: FloatArray,
    *,
    uncertainty_normalizers: FloatArray | None = None,
) -> UncertaintyDiagnostics:
    """Evaluate calibration and relative-error ranking for sampled 3D residuals."""

    errors = np.asarray(errors, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    target_norms = np.asarray(target_norms, dtype=np.float64)
    normalizers = (
        target_norms
        if uncertainty_normalizers is None
        else np.asarray(uncertainty_normalizers, dtype=np.float64)
    )
    if errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError("errors must have shape (N, 3)")
    if covariances.shape != (errors.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N, 3, 3)")
    if target_norms.shape != (errors.shape[0],):
        raise ValueError("target_norms must have shape (N,)")
    if normalizers.shape != (errors.shape[0],):
        raise ValueError("uncertainty_normalizers must have shape (N,)")
    if not np.all(np.isfinite(covariances)):
        raise ValueError("diagnostic covariances must be finite")
    active = (
        np.all(np.isfinite(errors), axis=1)
        & np.isfinite(target_norms)
        & np.isfinite(normalizers)
    )
    if not np.any(active):
        raise ValueError("uncertainty diagnostics have no finite samples")
    errors = errors[active]
    covariances = covariances[active]
    target_norms = target_norms[active]
    normalizers = normalizers[active]

    symmetric, inverse, log_determinant = covariance_statistics(
        covariances,
        name="diagnostic covariance",
    )
    mahalanobis_squared = np.einsum("...i,...ij,...j->...", errors, inverse, errors)
    nll = 0.5 * (3.0 * np.log(2.0 * np.pi) + log_determinant + mahalanobis_squared)
    coverage = {
        level: float(np.mean(mahalanobis_squared <= threshold))
        for level, threshold in _CHI_SQUARED_3.items()
    }
    calibration_error = float(
        np.mean([abs(coverage[level] - level) for level in _CHI_SQUARED_3])
    )

    relative_error = np.linalg.norm(errors, axis=-1) / np.maximum(target_norms, 1e-2)
    # Match the relative point-error risk: covariance is in squared metric
    # units, so normalize it by the squared predicted point norm.
    uncertainty_score = np.trace(symmetric, axis1=-2, axis2=-1) / np.maximum(
        normalizers, 1e-2
    ) ** 2
    uncertainty_rank = _rankdata(uncertainty_score)
    error_rank = _rankdata(relative_error)
    if np.std(uncertainty_rank) <= 1e-12 or np.std(error_rank) <= 1e-12:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(uncertainty_rank, error_rank)[0, 1])

    retain_count_50 = max(1, int(np.floor(0.5 * relative_error.size)))
    retain_count_80 = max(1, int(np.floor(0.8 * relative_error.size)))
    risk_all = float(np.mean(relative_error))
    risk_retained_50 = _expected_risk_at_count(
        relative_error,
        uncertainty_score,
        retain_count_50,
    )
    risk_retained_80 = _expected_risk_at_count(
        relative_error,
        uncertainty_score,
        retain_count_80,
    )
    risk_oracle = _expected_risk_at_count(
        relative_error,
        relative_error,
        retain_count_80,
    )
    gain = risk_all - risk_retained_80
    oracle_gain = risk_all - risk_oracle

    return UncertaintyDiagnostics(
        count=int(relative_error.size),
        coverage_50=coverage[0.50],
        coverage_80=coverage[0.80],
        coverage_90=coverage[0.90],
        coverage_95=coverage[0.95],
        coverage_shortfall_95=max(0.0, 0.95 - coverage[0.95]),
        coverage_calibration_error=calibration_error,
        mean_mahalanobis_squared=float(np.mean(mahalanobis_squared)),
        median_mahalanobis_squared=float(np.median(mahalanobis_squared)),
        gaussian_nll=float(np.mean(nll)),
        mean_covariance_trace=float(
            np.mean(np.trace(symmetric, axis1=-2, axis2=-1))
        ),
        mean_covariance_log_determinant=float(np.mean(log_determinant)),
        uncertainty_error_spearman=spearman,
        mean_relative_error=risk_all,
        relative_error_retained_50=risk_retained_50,
        relative_error_retained_80=risk_retained_80,
        relative_error_oracle_80=risk_oracle,
        selective_gain_80=gain,
        selective_oracle_fraction_80=(gain / oracle_gain if oracle_gain > 1e-12 else 0.0),
        risk_coverage_auc=_tie_aware_risk_coverage_auc(
            relative_error,
            uncertainty_score,
        ),
    )


def _scale_translation_alignment(
    source: FloatArray, target: FloatArray
) -> tuple[float, FloatArray]:
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
    truth_support_mask: NDArray[np.bool_] | None = None,
    truth_flow_support_mask: NDArray[np.bool_] | None = None,
) -> SequenceMetrics:
    """Evaluate one sequence after at most one global scale/translation alignment."""

    support_mask = _validated_support_mask(
        truth_support_mask,
        truth=truth,
        name="truth_support_mask",
    )
    flow_support_mask = _validated_support_mask(
        truth_flow_support_mask,
        truth=truth,
        name="truth_flow_support_mask",
    )
    common_frames = np.intersect1d(prediction.frame_indices, truth.frame_indices)
    if common_frames.size == 0:
        raise ValueError("prediction and truth have no common frames")
    prediction_indices = [
        int(np.searchsorted(prediction.frame_indices, frame)) for frame in common_frames
    ]
    truth_indices = [
        int(np.searchsorted(truth.frame_indices, frame)) for frame in common_frames
    ]
    predicted_points = prediction.point_map[prediction_indices].copy()
    predicted_covariance = prediction.point_covariance[prediction_indices].copy()
    predicted_mask = prediction.valid_mask[prediction_indices]
    truth_points = truth.point_map[truth_indices]
    truth_mask = truth.valid_mask[truth_indices]
    if support_mask is not None:
        truth_mask = truth_mask & support_mask[truth_indices]
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
    metric_frame_balanced_point_rmse = float(
        np.mean(metric_frame_rmse[metric_finite_frames])
    )

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
    frame_balanced_point_rmse = float(np.mean(frame_rmse[finite_frames]))

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
    diagnostics = uncertainty_diagnostics(
        active_error,
        active_covariance,
        np.linalg.norm(truth_points[active], axis=-1),
        uncertainty_normalizers=np.linalg.norm(predicted_points[active], axis=-1),
    )

    flow_epe: float | None = None
    evaluated_flow_points = 0
    if prediction.scene_flow is not None and truth.scene_flow is not None:
        predicted_flow = scale * prediction.scene_flow[prediction_indices]
        truth_flow = truth.scene_flow[truth_indices]
        flow_mask = (
            prediction.deform_mask[prediction_indices]
            & truth.deform_mask[truth_indices]
            & predicted_mask
            & truth_mask
            & np.all(np.isfinite(predicted_flow), axis=-1)
            & np.all(np.isfinite(truth_flow), axis=-1)
        )
        if flow_support_mask is not None:
            flow_mask &= flow_support_mask[truth_indices]
        if np.any(flow_mask):
            evaluated_flow_points = int(np.count_nonzero(flow_mask))
            flow_epe = float(
                np.mean(
                    np.linalg.norm(
                        predicted_flow[flow_mask] - truth_flow[flow_mask],
                        axis=-1,
                    )
                )
            )

    return SequenceMetrics(
        metric_point_rmse=metric_point_rmse,
        metric_endpoint_point_rmse=metric_endpoint_point_rmse,
        metric_frame_balanced_point_rmse=metric_frame_balanced_point_rmse,
        point_rmse=point_rmse,
        endpoint_point_rmse=endpoint_point_rmse,
        frame_balanced_point_rmse=frame_balanced_point_rmse,
        drift_slope=drift_slope,
        seam_rmse=seam_rmse,
        flow_epe=flow_epe,
        coverage_50=diagnostics.coverage_50,
        coverage_80=diagnostics.coverage_80,
        coverage_90=diagnostics.coverage_90,
        coverage_95=diagnostics.coverage_95,
        coverage_shortfall_95=diagnostics.coverage_shortfall_95,
        coverage_calibration_error=diagnostics.coverage_calibration_error,
        gaussian_nll=diagnostics.gaussian_nll,
        mean_mahalanobis_squared=diagnostics.mean_mahalanobis_squared,
        median_mahalanobis_squared=diagnostics.median_mahalanobis_squared,
        mean_covariance_trace=diagnostics.mean_covariance_trace,
        mean_covariance_log_determinant=(
            diagnostics.mean_covariance_log_determinant
        ),
        uncertainty_error_spearman=diagnostics.uncertainty_error_spearman,
        mean_relative_error=diagnostics.mean_relative_error,
        relative_error_retained_50=diagnostics.relative_error_retained_50,
        relative_error_retained_80=diagnostics.relative_error_retained_80,
        selective_gain_80=diagnostics.selective_gain_80,
        selective_oracle_fraction_80=diagnostics.selective_oracle_fraction_80,
        risk_coverage_auc=diagnostics.risk_coverage_auc,
        evaluated_points=int(np.count_nonzero(active)),
        evaluated_frames=int(np.count_nonzero(finite_frames)),
        evaluated_flow_points=evaluated_flow_points,
        fitted_alignment_scale=scale,
    )
