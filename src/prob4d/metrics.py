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


DEFAULT_EVALUATION_CHUNK_SIZE = 65_536


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


class _UncertaintyAccumulator:
    """Accumulate uncertainty diagnostics with bounded covariance temporaries."""

    def __init__(self) -> None:
        self._count = 0
        self._coverage_counts = {level: 0 for level in _CHI_SQUARED_3}
        self._nll_sum = 0.0
        self._trace_sum = 0.0
        self._log_determinant_sum = 0.0
        self._mahalanobis_squared: list[np.ndarray] = []
        self._relative_error: list[np.ndarray] = []
        self._uncertainty_score: list[np.ndarray] = []

    def add(
        self,
        errors: FloatArray,
        covariances: FloatArray,
        target_norms: FloatArray,
        *,
        uncertainty_normalizers: FloatArray | None = None,
    ) -> None:
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
            return
        errors = errors[active]
        covariances = covariances[active]
        target_norms = target_norms[active]
        normalizers = normalizers[active]

        symmetric, inverse, log_determinant = covariance_statistics(
            covariances,
            name="diagnostic covariance",
        )
        mahalanobis_squared = np.einsum(
            "...i,...ij,...j->...",
            errors,
            inverse,
            errors,
        )
        nll = 0.5 * (
            3.0 * np.log(2.0 * np.pi) + log_determinant + mahalanobis_squared
        )
        count = int(errors.shape[0])
        self._count += count
        self._nll_sum += float(np.sum(nll, dtype=np.float64))
        covariance_trace = np.trace(symmetric, axis1=-2, axis2=-1)
        self._trace_sum += float(np.sum(covariance_trace, dtype=np.float64))
        self._log_determinant_sum += float(
            np.sum(log_determinant, dtype=np.float64)
        )
        for level, threshold in _CHI_SQUARED_3.items():
            self._coverage_counts[level] += int(
                np.count_nonzero(mahalanobis_squared <= threshold)
            )

        relative_error = np.linalg.norm(errors, axis=-1) / np.maximum(
            target_norms,
            1e-2,
        )
        uncertainty_score = covariance_trace / np.maximum(normalizers, 1e-2) ** 2
        self._mahalanobis_squared.append(mahalanobis_squared.copy())
        self._relative_error.append(relative_error.copy())
        self._uncertainty_score.append(uncertainty_score.copy())

    def finalize(self) -> UncertaintyDiagnostics:
        if self._count == 0:
            raise ValueError("uncertainty diagnostics have no finite samples")
        mahalanobis_squared = np.concatenate(self._mahalanobis_squared)
        self._mahalanobis_squared.clear()
        relative_error = np.concatenate(self._relative_error)
        self._relative_error.clear()
        uncertainty_score = np.concatenate(self._uncertainty_score)
        self._uncertainty_score.clear()
        coverage = {
            level: self._coverage_counts[level] / self._count
            for level in _CHI_SQUARED_3
        }
        calibration_error = float(
            np.mean([abs(coverage[level] - level) for level in _CHI_SQUARED_3])
        )

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
            count=self._count,
            coverage_50=coverage[0.50],
            coverage_80=coverage[0.80],
            coverage_90=coverage[0.90],
            coverage_95=coverage[0.95],
            coverage_shortfall_95=max(0.0, 0.95 - coverage[0.95]),
            coverage_calibration_error=calibration_error,
            mean_mahalanobis_squared=float(np.mean(mahalanobis_squared)),
            median_mahalanobis_squared=float(np.median(mahalanobis_squared)),
            gaussian_nll=self._nll_sum / self._count,
            mean_covariance_trace=self._trace_sum / self._count,
            mean_covariance_log_determinant=(
                self._log_determinant_sum / self._count
            ),
            uncertainty_error_spearman=spearman,
            mean_relative_error=risk_all,
            relative_error_retained_50=risk_retained_50,
            relative_error_retained_80=risk_retained_80,
            relative_error_oracle_80=risk_oracle,
            selective_gain_80=gain,
            selective_oracle_fraction_80=(
                gain / oracle_gain if oracle_gain > 1e-12 else 0.0
            ),
            risk_coverage_auc=_tie_aware_risk_coverage_auc(
                relative_error,
                uncertainty_score,
            ),
        )


def uncertainty_diagnostics(
    errors: FloatArray,
    covariances: FloatArray,
    target_norms: FloatArray,
    *,
    uncertainty_normalizers: FloatArray | None = None,
) -> UncertaintyDiagnostics:
    """Evaluate calibration and relative-error ranking for sampled 3D residuals."""

    accumulator = _UncertaintyAccumulator()
    accumulator.add(
        errors,
        covariances,
        target_norms,
        uncertainty_normalizers=uncertainty_normalizers,
    )
    return accumulator.finalize()


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


def _scale_translation_from_sufficient_statistics(
    *,
    count: int,
    source_sum: FloatArray,
    target_sum: FloatArray,
    source_squared_sum: float,
    source_target_sum: float,
) -> tuple[float, FloatArray]:
    if count < 1:
        raise ValueError("scale/translation alignment requires at least one sample")
    source_mean = np.asarray(source_sum, dtype=np.float64) / count
    target_mean = np.asarray(target_sum, dtype=np.float64) / count
    denominator = source_squared_sum - count * float(source_mean @ source_mean)
    if denominator <= np.finfo(np.float64).eps:
        return 1.0, target_mean - source_mean
    numerator = source_target_sum - count * float(source_mean @ target_mean)
    scale = max(numerator / denominator, np.finfo(np.float64).eps)
    return float(scale), target_mean - scale * source_mean


def _frame_support(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    prediction_index: int,
    truth_index: int,
    support_mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    active = prediction.valid_mask[prediction_index] & truth.valid_mask[truth_index]
    if support_mask is not None:
        active = active & support_mask[truth_index]
    return active


def evaluate_sequence(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    boundary_frames: list[int] | None = None,
    align_scale_translation: bool = True,
    truth_support_mask: NDArray[np.bool_] | None = None,
    truth_flow_support_mask: NDArray[np.bool_] | None = None,
    prediction_scale: float = 1.0,
    prediction_translation: ArrayLike | None = None,
    evaluation_chunk_size: int = DEFAULT_EVALUATION_CHUNK_SIZE,
) -> SequenceMetrics:
    """Evaluate one sequence with bounded point/covariance temporaries.

    The dense prediction and truth contracts remain unchanged, but metric
    accumulation processes one spatial chunk at a time. Only three scalar
    vectors needed for median, ranking, and selective-risk diagnostics grow with
    the evaluated point count.
    """

    if evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    if not np.isfinite(prediction_scale) or prediction_scale <= 0.0:
        raise ValueError("prediction_scale must be finite and positive")
    base_translation = (
        np.zeros(3, dtype=np.float64)
        if prediction_translation is None
        else np.asarray(prediction_translation, dtype=np.float64)
    )
    if base_translation.shape != (3,) or not np.all(np.isfinite(base_translation)):
        raise ValueError("prediction_translation must be a finite three-vector")
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
    prediction_indices = np.searchsorted(prediction.frame_indices, common_frames)
    truth_indices = np.searchsorted(truth.frame_indices, common_frames)

    metric_frame_rmse = np.full(common_frames.size, np.nan, dtype=np.float64)
    metric_squared_sum = 0.0
    evaluated_points = 0
    alignment_source_sum = np.zeros(3, dtype=np.float64)
    alignment_target_sum = np.zeros(3, dtype=np.float64)
    alignment_source_squared_sum = 0.0
    alignment_source_target_sum = 0.0

    for frame_position, (prediction_index, truth_index) in enumerate(
        zip(prediction_indices, truth_indices, strict=True)
    ):
        active = _frame_support(
            prediction,
            truth,
            prediction_index=int(prediction_index),
            truth_index=int(truth_index),
            support_mask=support_mask,
        ).reshape(-1)
        predicted = prediction.point_map[int(prediction_index)].reshape(-1, 3)
        target = truth.point_map[int(truth_index)].reshape(-1, 3)
        frame_squared_sum = 0.0
        frame_count = 0
        for start in range(0, active.size, evaluation_chunk_size):
            stop = min(start + evaluation_chunk_size, active.size)
            selected = active[start:stop]
            if not np.any(selected):
                continue
            source_values = (
                prediction_scale * predicted[start:stop][selected]
                + base_translation
            )
            target_values = target[start:stop][selected]
            errors = source_values - target_values
            squared_norm = np.einsum("ni,ni->n", errors, errors)
            count = int(source_values.shape[0])
            frame_count += count
            evaluated_points += count
            squared_sum = float(np.sum(squared_norm, dtype=np.float64))
            frame_squared_sum += squared_sum
            metric_squared_sum += squared_sum
            if align_scale_translation:
                alignment_source_sum += np.sum(
                    source_values,
                    axis=0,
                    dtype=np.float64,
                )
                alignment_target_sum += np.sum(
                    target_values,
                    axis=0,
                    dtype=np.float64,
                )
                alignment_source_squared_sum += float(
                    np.sum(source_values * source_values, dtype=np.float64)
                )
                alignment_source_target_sum += float(
                    np.sum(source_values * target_values, dtype=np.float64)
                )
        if frame_count:
            metric_frame_rmse[frame_position] = np.sqrt(
                frame_squared_sum / frame_count
            )

    if evaluated_points == 0:
        raise ValueError("prediction and truth have no jointly valid points")
    metric_finite_frames = np.flatnonzero(np.isfinite(metric_frame_rmse))
    metric_point_rmse = float(np.sqrt(metric_squared_sum / evaluated_points))
    metric_endpoint_point_rmse = float(metric_frame_rmse[metric_finite_frames[-1]])
    metric_frame_balanced_point_rmse = float(
        np.mean(metric_frame_rmse[metric_finite_frames])
    )

    alignment_scale = 1.0
    alignment_translation = np.zeros(3, dtype=np.float64)
    if align_scale_translation:
        alignment_scale, alignment_translation = (
            _scale_translation_from_sufficient_statistics(
                count=evaluated_points,
                source_sum=alignment_source_sum,
                target_sum=alignment_target_sum,
                source_squared_sum=alignment_source_squared_sum,
                source_target_sum=alignment_source_target_sum,
            )
        )
    effective_scale = alignment_scale * prediction_scale
    effective_translation = (
        alignment_scale * base_translation + alignment_translation
    )

    frame_rmse = np.full(common_frames.size, np.nan, dtype=np.float64)
    squared_sum = 0.0
    uncertainty = _UncertaintyAccumulator()
    flow_error_sum = 0.0
    evaluated_flow_points = 0
    covariance_scale = effective_scale**2

    for frame_position, (prediction_index, truth_index) in enumerate(
        zip(prediction_indices, truth_indices, strict=True)
    ):
        prediction_index = int(prediction_index)
        truth_index = int(truth_index)
        active_frame = _frame_support(
            prediction,
            truth,
            prediction_index=prediction_index,
            truth_index=truth_index,
            support_mask=support_mask,
        )
        active = active_frame.reshape(-1)
        predicted = prediction.point_map[prediction_index].reshape(-1, 3)
        target = truth.point_map[truth_index].reshape(-1, 3)
        covariance = prediction.point_covariance[prediction_index].reshape(-1, 3, 3)
        frame_squared_sum = 0.0
        frame_count = 0

        predicted_flow = (
            None
            if prediction.scene_flow is None
            else prediction.scene_flow[prediction_index].reshape(-1, 3)
        )
        target_flow = (
            None
            if truth.scene_flow is None
            else truth.scene_flow[truth_index].reshape(-1, 3)
        )
        flow_active = None
        if predicted_flow is not None and target_flow is not None:
            assert prediction.deform_mask is not None
            assert truth.deform_mask is not None
            flow_active = (
                active_frame
                & prediction.deform_mask[prediction_index]
                & truth.deform_mask[truth_index]
                & np.all(np.isfinite(prediction.scene_flow[prediction_index]), axis=-1)
                & np.all(np.isfinite(truth.scene_flow[truth_index]), axis=-1)
            )
            if flow_support_mask is not None:
                flow_active &= flow_support_mask[truth_index]
            flow_active = flow_active.reshape(-1)

        for start in range(0, active.size, evaluation_chunk_size):
            stop = min(start + evaluation_chunk_size, active.size)
            selected = active[start:stop]
            if np.any(selected):
                source_values = predicted[start:stop][selected]
                target_values = target[start:stop][selected]
                aligned_values = (
                    effective_scale * source_values + effective_translation
                )
                errors = aligned_values - target_values
                squared_norm = np.einsum("ni,ni->n", errors, errors)
                count = int(errors.shape[0])
                frame_count += count
                value = float(np.sum(squared_norm, dtype=np.float64))
                frame_squared_sum += value
                squared_sum += value
                selected_covariance = covariance[start:stop][selected]
                if covariance_scale != 1.0:
                    selected_covariance = selected_covariance * covariance_scale
                uncertainty.add(
                    errors,
                    selected_covariance,
                    np.linalg.norm(target_values, axis=-1),
                    uncertainty_normalizers=np.linalg.norm(
                        aligned_values,
                        axis=-1,
                    ),
                )

            if flow_active is not None:
                selected_flow = flow_active[start:stop]
                if np.any(selected_flow):
                    assert predicted_flow is not None
                    assert target_flow is not None
                    flow_errors = (
                        effective_scale * predicted_flow[start:stop][selected_flow]
                        - target_flow[start:stop][selected_flow]
                    )
                    flow_error_sum += float(
                        np.sum(np.linalg.norm(flow_errors, axis=-1), dtype=np.float64)
                    )
                    evaluated_flow_points += int(flow_errors.shape[0])
        if frame_count:
            frame_rmse[frame_position] = np.sqrt(frame_squared_sum / frame_count)

    finite_frames = np.isfinite(frame_rmse)
    if np.count_nonzero(finite_frames) >= 2:
        drift_slope = float(
            np.polyfit(common_frames[finite_frames], frame_rmse[finite_frames], deg=1)[0]
        )
    else:
        drift_slope = 0.0
    endpoint_point_rmse = float(frame_rmse[np.flatnonzero(finite_frames)[-1]])
    frame_balanced_point_rmse = float(np.mean(frame_rmse[finite_frames]))
    point_rmse = float(np.sqrt(squared_sum / evaluated_points))

    seam_squared_sum = 0.0
    seam_count = 0
    common_position = {int(frame): index for index, frame in enumerate(common_frames)}
    for boundary in boundary_frames or []:
        if boundary not in common_position or boundary - 1 not in common_position:
            continue
        current_position = common_position[boundary]
        previous_position = common_position[boundary - 1]
        current_prediction_index = int(prediction_indices[current_position])
        current_truth_index = int(truth_indices[current_position])
        previous_prediction_index = int(prediction_indices[previous_position])
        previous_truth_index = int(truth_indices[previous_position])
        current_active = _frame_support(
            prediction,
            truth,
            prediction_index=current_prediction_index,
            truth_index=current_truth_index,
            support_mask=support_mask,
        ).reshape(-1)
        previous_active = _frame_support(
            prediction,
            truth,
            prediction_index=previous_prediction_index,
            truth_index=previous_truth_index,
            support_mask=support_mask,
        ).reshape(-1)
        seam_active = current_active & previous_active
        current_prediction = prediction.point_map[current_prediction_index].reshape(-1, 3)
        current_target = truth.point_map[current_truth_index].reshape(-1, 3)
        previous_prediction = prediction.point_map[previous_prediction_index].reshape(
            -1,
            3,
        )
        previous_target = truth.point_map[previous_truth_index].reshape(-1, 3)
        for start in range(0, seam_active.size, evaluation_chunk_size):
            stop = min(start + evaluation_chunk_size, seam_active.size)
            selected = seam_active[start:stop]
            if not np.any(selected):
                continue
            current_error = (
                effective_scale * current_prediction[start:stop][selected]
                + effective_translation
                - current_target[start:stop][selected]
            )
            previous_error = (
                effective_scale * previous_prediction[start:stop][selected]
                + effective_translation
                - previous_target[start:stop][selected]
            )
            error_jump = current_error - previous_error
            seam_squared_sum += float(
                np.sum(error_jump * error_jump, dtype=np.float64)
            )
            seam_count += int(error_jump.shape[0])
    seam_rmse = (
        float(np.sqrt(seam_squared_sum / seam_count)) if seam_count else 0.0
    )

    diagnostics = uncertainty.finalize()
    flow_epe = (
        flow_error_sum / evaluated_flow_points if evaluated_flow_points else None
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
        evaluated_points=evaluated_points,
        evaluated_frames=int(np.count_nonzero(finite_frames)),
        evaluated_flow_points=evaluated_flow_points,
        fitted_alignment_scale=alignment_scale,
    )
