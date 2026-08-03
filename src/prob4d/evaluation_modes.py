"""Explicit metric, causal-prefix, and oracle alignment evaluation modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .fusion import FusedSequence
from .metrics import (
    DEFAULT_EVALUATION_CHUNK_SIZE,
    SequenceMetrics,
    TruthSequence,
    evaluate_sequence,
)

EvaluationMode = Literal["metric", "prefix_aligned", "oracle_aligned"]


@dataclass(frozen=True)
class EvaluationModeResult:
    """Metrics together with the truth-derived transform used by one mode."""

    mode: EvaluationMode
    metrics: SequenceMetrics
    fitted_scale: float
    fitted_translation: np.ndarray
    fit_frame_count: int
    fit_point_count: int
    fit_frame_stop_exclusive: int | None

    def __post_init__(self) -> None:
        translation = np.asarray(self.fitted_translation, dtype=np.float64).copy()
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("fitted_translation must be a finite three-vector")
        if not np.isfinite(self.fitted_scale) or self.fitted_scale <= 0.0:
            raise ValueError("fitted_scale must be finite and positive")
        if self.fit_frame_count < 0 or self.fit_point_count < 0:
            raise ValueError("alignment fit counts must be non-negative")
        translation.setflags(write=False)
        object.__setattr__(self, "fitted_translation", translation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fitted_scale": self.fitted_scale,
            "fitted_translation": self.fitted_translation.tolist(),
            "fit_frame_count": self.fit_frame_count,
            "fit_point_count": self.fit_point_count,
            "fit_frame_stop_exclusive": self.fit_frame_stop_exclusive,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationModes:
    """The three evaluation interpretations for one predicted sequence."""

    metric: EvaluationModeResult
    oracle_aligned: EvaluationModeResult
    prefix_aligned: EvaluationModeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.to_dict(),
            "prefix_aligned": (
                None
                if self.prefix_aligned is None
                else self.prefix_aligned.to_dict()
            ),
            "oracle_aligned": self.oracle_aligned.to_dict(),
        }


def _common_pairs(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    frame_stop_exclusive: int | None,
) -> list[tuple[int, int]]:
    common = np.intersect1d(prediction.frame_indices, truth.frame_indices)
    if frame_stop_exclusive is not None:
        common = common[common < frame_stop_exclusive]
    return [
        (
            int(np.searchsorted(prediction.frame_indices, frame)),
            int(np.searchsorted(truth.frame_indices, frame)),
        )
        for frame in common
    ]


def _fit_scale_translation_streaming(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    frame_stop_exclusive: int | None,
    truth_support_mask: np.ndarray | None,
    evaluation_chunk_size: int,
) -> tuple[float, np.ndarray, int, int]:
    """Fit one isotropic scale and translation with bounded temporaries."""

    if evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    pairs = _common_pairs(
        prediction,
        truth,
        frame_stop_exclusive=frame_stop_exclusive,
    )
    point_count = 0
    frame_count = 0
    source_sum = np.zeros(3, dtype=np.float64)
    target_sum = np.zeros(3, dtype=np.float64)
    source_squared_sum = 0.0
    cross_sum = 0.0
    for prediction_index, truth_index in pairs:
        source_frame = prediction.point_map[prediction_index].reshape(-1, 3)
        target_frame = truth.point_map[truth_index].reshape(-1, 3)
        active = (
            prediction.valid_mask[prediction_index]
            & truth.valid_mask[truth_index]
        )
        if truth_support_mask is not None:
            active &= truth_support_mask[truth_index]
        flat_active = active.reshape(-1)
        frame_points = 0
        for start in range(0, flat_active.size, evaluation_chunk_size):
            stop = min(start + evaluation_chunk_size, flat_active.size)
            selected = flat_active[start:stop]
            if not np.any(selected):
                continue
            source = source_frame[start:stop][selected]
            target = target_frame[start:stop][selected]
            count = int(source.shape[0])
            frame_points += count
            point_count += count
            source_sum += np.sum(source, axis=0, dtype=np.float64)
            target_sum += np.sum(target, axis=0, dtype=np.float64)
            source_squared_sum += float(
                np.sum(source * source, dtype=np.float64)
            )
            cross_sum += float(np.sum(source * target, dtype=np.float64))
        if frame_points:
            frame_count += 1

    if point_count == 0:
        boundary = (
            "all common frames"
            if frame_stop_exclusive is None
            else f"frames before {frame_stop_exclusive}"
        )
        raise ValueError(f"alignment fit has no jointly valid points in {boundary}")

    source_mean = source_sum / point_count
    target_mean = target_sum / point_count
    denominator = source_squared_sum - point_count * float(
        source_mean @ source_mean
    )
    if denominator <= np.finfo(np.float64).eps:
        scale = 1.0
    else:
        numerator = cross_sum - point_count * float(source_mean @ target_mean)
        scale = max(
            numerator / denominator,
            np.finfo(np.float64).eps,
        )
    translation = target_mean - scale * source_mean
    return float(scale), translation, frame_count, point_count


def _evaluate_transformed(
    mode: EvaluationMode,
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    scale: float,
    translation: np.ndarray,
    fit_frame_count: int,
    fit_point_count: int,
    fit_frame_stop_exclusive: int | None,
    boundary_frames: list[int] | None,
    truth_support_mask: np.ndarray | None,
    truth_flow_support_mask: np.ndarray | None,
    evaluation_chunk_size: int,
) -> EvaluationModeResult:
    metrics = evaluate_sequence(
        prediction,
        truth,
        boundary_frames=boundary_frames,
        align_scale_translation=False,
        truth_support_mask=truth_support_mask,
        truth_flow_support_mask=truth_flow_support_mask,
        prediction_scale=scale,
        prediction_translation=translation,
        evaluation_chunk_size=evaluation_chunk_size,
    )
    metrics = replace(metrics, fitted_alignment_scale=scale)
    return EvaluationModeResult(
        mode=mode,
        metrics=metrics,
        fitted_scale=scale,
        fitted_translation=translation,
        fit_frame_count=fit_frame_count,
        fit_point_count=fit_point_count,
        fit_frame_stop_exclusive=fit_frame_stop_exclusive,
    )


def evaluate_sequence_modes(
    prediction: FusedSequence,
    truth: TruthSequence,
    *,
    boundary_frames: list[int] | None = None,
    prefix_frame_stop_exclusive: int | None = None,
    truth_support_mask: np.ndarray | None = None,
    truth_flow_support_mask: np.ndarray | None = None,
    evaluation_chunk_size: int = DEFAULT_EVALUATION_CHUNK_SIZE,
) -> EvaluationModes:
    """Evaluate metric, causal-prefix-aligned, and oracle-aligned interpretations.

    The metric mode never uses truth to alter the prediction. The prefix mode fits
    one scale and translation only from common frames strictly before the supplied
    boundary and then evaluates all available frames. The oracle mode fits on all
    evaluated common frames and is therefore a reconstruction diagnostic rather
    than a causal prediction result.
    """

    if evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    support_mask = None
    if truth_support_mask is not None:
        support_mask = np.asarray(truth_support_mask, dtype=bool)
        if support_mask.shape != truth.valid_mask.shape:
            raise ValueError("truth_support_mask must match truth valid_mask shape")
    flow_support_mask = None
    if truth_flow_support_mask is not None:
        flow_support_mask = np.asarray(truth_flow_support_mask, dtype=bool)
        if flow_support_mask.shape != truth.valid_mask.shape:
            raise ValueError(
                "truth_flow_support_mask must match truth valid_mask shape"
            )
    metric = _evaluate_transformed(
        "metric",
        prediction,
        truth,
        scale=1.0,
        translation=np.zeros(3),
        fit_frame_count=0,
        fit_point_count=0,
        fit_frame_stop_exclusive=None,
        boundary_frames=boundary_frames,
        truth_support_mask=support_mask,
        truth_flow_support_mask=flow_support_mask,
        evaluation_chunk_size=evaluation_chunk_size,
    )

    oracle_scale, oracle_translation, oracle_frames, oracle_points = (
        _fit_scale_translation_streaming(
            prediction,
            truth,
            frame_stop_exclusive=None,
            truth_support_mask=support_mask,
            evaluation_chunk_size=evaluation_chunk_size,
        )
    )
    oracle = _evaluate_transformed(
        "oracle_aligned",
        prediction,
        truth,
        scale=oracle_scale,
        translation=oracle_translation,
        fit_frame_count=oracle_frames,
        fit_point_count=oracle_points,
        fit_frame_stop_exclusive=None,
        boundary_frames=boundary_frames,
        truth_support_mask=support_mask,
        truth_flow_support_mask=flow_support_mask,
        evaluation_chunk_size=evaluation_chunk_size,
    )

    prefix: EvaluationModeResult | None = None
    if prefix_frame_stop_exclusive is not None:
        prefix_scale, prefix_translation, prefix_frames, prefix_points = (
            _fit_scale_translation_streaming(
                prediction,
                truth,
                frame_stop_exclusive=prefix_frame_stop_exclusive,
                truth_support_mask=support_mask,
                evaluation_chunk_size=evaluation_chunk_size,
            )
        )
        prefix = _evaluate_transformed(
            "prefix_aligned",
            prediction,
            truth,
            scale=prefix_scale,
            translation=prefix_translation,
            fit_frame_count=prefix_frames,
            fit_point_count=prefix_points,
            fit_frame_stop_exclusive=prefix_frame_stop_exclusive,
            boundary_frames=boundary_frames,
            truth_support_mask=support_mask,
            truth_flow_support_mask=flow_support_mask,
            evaluation_chunk_size=evaluation_chunk_size,
        )

    return EvaluationModes(
        metric=metric,
        prefix_aligned=prefix,
        oracle_aligned=oracle,
    )


__all__ = [
    "EvaluationMode",
    "EvaluationModeResult",
    "EvaluationModes",
    "evaluate_sequence_modes",
]
