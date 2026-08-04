"""Source-side diagnostics for common-mode observation failures.

Overlap disagreement detects mutually inconsistent windows, but a shared visual
backbone can be confidently wrong in every window.  This module adds diagnostics
that remain independent of Bayesian-PhysTwin innovations and target labels:

- one-step consistency between decoded point motion and predicted scene flow;
- empirical dispersion across independently seeded predictions in an explicitly
  common gauge; and
- an evaluation-only quadrant audit that reports low-disagreement/high-error
  failures without feeding target truth back into reliability calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from .data import PredictionWindow
from .source_reliability import SourceReliabilityFeatures

FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SourceOnlyDiagnosticGrid:
    """Finite source-only diagnostic features on one prediction grid."""

    feature_names: tuple[str, ...]
    values: FloatArray
    available_mask: BoolArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.feature_names)
        if not names or any(not value for value in names):
            raise ValueError("diagnostic feature_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("diagnostic feature_names must be unique")
        values = np.asarray(self.values, dtype=np.float64)
        available = np.asarray(self.available_mask, dtype=bool)
        if values.ndim < 2 or values.shape[-1] != len(names):
            raise ValueError("diagnostic values must have shape (..., feature_count)")
        if available.shape != values.shape[:-1]:
            raise ValueError("diagnostic available_mask must match the feature grid")
        if not np.all(np.isfinite(values)):
            raise ValueError("source-only diagnostic values must be finite")

        normalized_metadata = frozen_finite_json_mapping(
            self.metadata,
            name="source-only diagnostic metadata",
        )
        required_false = (
            "uses_truth",
            "uses_downstream_physical_innovation",
            "uses_association_probability",
        )
        for name in required_false:
            if normalized_metadata.get(name) is not False:
                raise ValueError(f"source-only diagnostic metadata must declare {name}=false")

        masked_values = np.where(available[..., None], values, 0.0)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", _readonly(masked_values, dtype=np.float64))
        object.__setattr__(self, "available_mask", _readonly(available, dtype=bool))
        object.__setattr__(self, "metadata", normalized_metadata)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return self.available_mask.shape

    @property
    def available_count(self) -> int:
        return int(np.count_nonzero(self.available_mask))

    def summary(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "grid_shape": list(self.grid_shape),
            "available_count": self.available_count,
            "metadata": plain_json(self.metadata),
        }


def augment_source_reliability_features(
    base: SourceReliabilityFeatures,
    diagnostics: Sequence[SourceOnlyDiagnosticGrid],
) -> SourceReliabilityFeatures:
    """Append source-only diagnostics without changing the base valid-row set."""

    diagnostic_list = list(diagnostics)
    if not diagnostic_list:
        raise ValueError("at least one source-only diagnostic is required")

    names = list(base.feature_names)
    arrays = [np.asarray(base.values, dtype=np.float64)]
    summaries: list[dict[str, object]] = []
    for diagnostic in diagnostic_list:
        if diagnostic.grid_shape != base.valid_mask.shape:
            raise ValueError("source-only diagnostic grid differs from base features")
        duplicates = sorted(set(names) & set(diagnostic.feature_names))
        if duplicates:
            raise ValueError(f"source-only diagnostic feature names collide: {duplicates}")
        names.extend(diagnostic.feature_names)
        arrays.append(np.asarray(diagnostic.values, dtype=np.float64))
        summaries.append(diagnostic.summary())

    metadata = plain_json(base.metadata)
    metadata.update(
        {
            "semantics": "prob4d-source-only-reliability-features-augmented-v1",
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "uses_association_probability": False,
            "source_only_diagnostics": summaries,
        }
    )
    return SourceReliabilityFeatures(
        feature_names=tuple(names),
        values=np.concatenate(arrays, axis=-1),
        valid_mask=base.valid_mask,
        metadata=metadata,
    )


def build_flow_point_consistency_diagnostic(
    window: PredictionWindow,
) -> SourceOnlyDiagnosticGrid:
    """Compare one-step decoded point displacement with predicted scene flow.

    The diagnostic assumes that ``scene_flow[t]`` predicts motion from local frame
    ``t`` to local frame ``t + 1``.  It is opt-in and records this assumption in
    metadata so a producer with different flow semantics cannot silently reuse it.
    """

    shape = window.shape
    available = np.zeros(shape, dtype=bool)
    relative_residual = np.zeros(shape, dtype=np.float64)
    direction_disagreement = np.zeros(shape, dtype=np.float64)

    if (
        window.scene_flow is not None
        and window.deform_mask is not None
        and shape[0] > 1
    ):
        current = np.asarray(window.point_map[:-1], dtype=np.float64)
        following = np.asarray(window.point_map[1:], dtype=np.float64)
        flow = np.asarray(window.scene_flow[:-1], dtype=np.float64)
        active = (
            np.asarray(window.valid_mask[:-1], dtype=bool)
            & np.asarray(window.valid_mask[1:], dtype=bool)
            & np.asarray(window.deform_mask[:-1], dtype=bool)
        )
        displacement = following - current
        residual_norm = np.linalg.norm(displacement - flow, axis=-1)
        displacement_norm = np.linalg.norm(displacement, axis=-1)
        flow_norm = np.linalg.norm(flow, axis=-1)
        scale = displacement_norm + flow_norm
        epsilon = np.finfo(np.float64).eps
        local_relative = np.divide(
            residual_norm,
            scale,
            out=np.zeros_like(residual_norm),
            where=scale > epsilon,
        )
        both_nonzero = (displacement_norm > epsilon) & (flow_norm > epsilon)
        cosine = np.divide(
            np.einsum("...i,...i->...", displacement, flow),
            displacement_norm * flow_norm,
            out=np.ones_like(displacement_norm),
            where=both_nonzero,
        )
        local_direction = np.where(
            both_nonzero,
            0.5 * (1.0 - np.clip(cosine, -1.0, 1.0)),
            0.0,
        )
        available[:-1] = active
        relative_residual[:-1] = np.where(active, local_relative, 0.0)
        direction_disagreement[:-1] = np.where(active, local_direction, 0.0)

    values = np.stack(
        (
            available.astype(np.float64),
            np.log1p(relative_residual),
            direction_disagreement,
        ),
        axis=-1,
    )
    return SourceOnlyDiagnosticGrid(
        feature_names=(
            "has_flow_point_consistency",
            "log1p_relative_flow_point_residual",
            "flow_point_direction_disagreement",
        ),
        values=values,
        available_mask=available,
        metadata={
            "semantics": "prob4d-flow-point-consistency-v1",
            "flow_semantics": "scene_flow[t] predicts point_map[t+1] - point_map[t]",
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "uses_association_probability": False,
            "window_id": window.window_id,
            "frame_indices": [int(value) for value in window.frame_indices],
            "scene_flow_available": window.scene_flow is not None,
        },
    )


def build_common_gauge_seed_dispersion_diagnostic(
    windows: Sequence[PredictionWindow],
    *,
    common_gauge_id: str,
    model_set_id: str,
) -> SourceOnlyDiagnosticGrid:
    """Measure empirical point dispersion across independently seeded predictions.

    Every input must already be represented in the same declared gauge.  The
    function deliberately does not align samples itself because target-informed or
    differently fitted gauges would change the meaning of empirical dispersion.
    """

    samples = list(windows)
    if len(samples) < 2:
        raise ValueError("seed dispersion requires at least two prediction windows")
    gauge_id = str(common_gauge_id).strip()
    model_identity = str(model_set_id).strip()
    if not gauge_id or not model_identity:
        raise ValueError("common_gauge_id and model_set_id must be non-empty")
    window_ids = [window.window_id for window in samples]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("seed-dispersion window IDs must be unique")

    reference = samples[0]
    for window in samples[1:]:
        if window.shape != reference.shape:
            raise ValueError("seed-dispersion prediction grids differ")
        if not np.array_equal(window.frame_indices, reference.frame_indices):
            raise ValueError("seed-dispersion frame indices differ")

    count = np.zeros(reference.shape, dtype=np.int64)
    point_sum = np.zeros(reference.shape + (3,), dtype=np.float64)
    squared_norm_sum = np.zeros(reference.shape, dtype=np.float64)
    for window in samples:
        mask = np.asarray(window.valid_mask, dtype=bool)
        points = np.asarray(window.point_map, dtype=np.float64)
        count[mask] += 1
        point_sum[mask] += points[mask]
        squared_norm_sum[mask] += np.einsum(
            "...i,...i->...",
            points[mask],
            points[mask],
        )

    available = count >= 2
    safe_count = np.maximum(count, 1)
    mean = point_sum / safe_count[..., None]
    second_moment = squared_norm_sum / safe_count
    variance_trace = np.maximum(
        second_moment - np.einsum("...i,...i->...", mean, mean),
        0.0,
    )
    dispersion = np.sqrt(variance_trace)
    mean_radius = np.linalg.norm(mean, axis=-1)
    positive_radius = mean_radius[available & (mean_radius > 0.0)]
    radius_scale = float(np.median(positive_radius)) if positive_radius.size else 1.0
    denominator = np.maximum(mean_radius, max(radius_scale, 1.0) * 1e-9)
    relative_dispersion = np.where(available, dispersion / denominator, 0.0)
    contributor_fraction = np.where(
        available,
        count / float(len(samples)),
        0.0,
    )

    values = np.stack(
        (
            available.astype(np.float64),
            contributor_fraction,
            np.log1p(relative_dispersion),
        ),
        axis=-1,
    )
    return SourceOnlyDiagnosticGrid(
        feature_names=(
            "has_independent_seed_dispersion",
            "independent_seed_contributor_fraction",
            "log1p_relative_independent_seed_dispersion",
        ),
        values=values,
        available_mask=available,
        metadata={
            "semantics": "prob4d-common-gauge-seed-dispersion-v1",
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "uses_association_probability": False,
            "common_gauge_id": gauge_id,
            "model_set_id": model_identity,
            "seed_count": len(samples),
            "window_ids": window_ids,
            "frame_indices": [int(value) for value in reference.frame_indices],
            "alignment_performed_by_diagnostic": False,
            "radius_scale": radius_scale,
        },
    )


@dataclass(frozen=True)
class CommonModeFailureAudit:
    """Evaluation-only disagreement/error quadrant accounting."""

    disagreement_threshold: float
    error_threshold: float
    valid_count: int
    low_disagreement_low_error_count: int
    high_disagreement_low_error_count: int
    low_disagreement_high_error_count: int
    high_disagreement_high_error_count: int
    low_disagreement_high_error_mean: float
    low_disagreement_high_error_max: float

    def __post_init__(self) -> None:
        thresholds = np.asarray(
            [self.disagreement_threshold, self.error_threshold],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0.0):
            raise ValueError("common-mode audit thresholds must be finite and non-negative")
        count_names = (
            "valid_count",
            "low_disagreement_low_error_count",
            "high_disagreement_low_error_count",
            "low_disagreement_high_error_count",
            "high_disagreement_high_error_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.valid_count < 1:
            raise ValueError("common-mode audit requires at least one valid row")
        total = sum(getattr(self, name) for name in count_names[1:])
        if total != self.valid_count:
            raise ValueError("common-mode quadrant counts do not sum to valid_count")
        severity = np.asarray(
            [
                self.low_disagreement_high_error_mean,
                self.low_disagreement_high_error_max,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(severity)) or np.any(severity < 0.0):
            raise ValueError("common-mode failure severity must be finite and non-negative")

    @property
    def low_disagreement_high_error_rate(self) -> float:
        return self.low_disagreement_high_error_count / self.valid_count

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "low_disagreement_high_error_rate": (
                self.low_disagreement_high_error_rate
            ),
        }


def audit_common_mode_failures(
    disagreement_score: FloatArray,
    error: FloatArray,
    *,
    disagreement_threshold: float,
    error_threshold: float,
    valid_mask: BoolArray | None = None,
) -> CommonModeFailureAudit:
    """Report the low-disagreement/high-error quadrant on opened evaluation data.

    Thresholds are explicit inputs and should be frozen from development or
    calibration units.  This function is evaluation-only: its result must not be
    used to refit a source-reliability model on the same target outcomes.
    """

    disagreement = np.asarray(disagreement_score, dtype=np.float64)
    errors = np.asarray(error, dtype=np.float64)
    if disagreement.shape != errors.shape:
        raise ValueError("common-mode disagreement and error arrays must match")
    if not np.all(np.isfinite(disagreement)) or np.any(disagreement < 0.0):
        raise ValueError("common-mode disagreement scores must be finite and non-negative")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("common-mode errors must be finite and non-negative")
    if valid_mask is None:
        valid = np.ones(disagreement.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != disagreement.shape:
            raise ValueError("common-mode valid_mask must match the score arrays")
    if not np.any(valid):
        raise ValueError("common-mode audit has no valid rows")

    disagreement_limit = float(disagreement_threshold)
    error_limit = float(error_threshold)
    if (
        not np.isfinite(disagreement_limit)
        or disagreement_limit < 0.0
        or not np.isfinite(error_limit)
        or error_limit < 0.0
    ):
        raise ValueError("common-mode thresholds must be finite and non-negative")

    high_disagreement = disagreement > disagreement_limit
    high_error = errors > error_limit
    low_low = valid & ~high_disagreement & ~high_error
    high_low = valid & high_disagreement & ~high_error
    low_high = valid & ~high_disagreement & high_error
    high_high = valid & high_disagreement & high_error
    low_high_errors = errors[low_high]
    return CommonModeFailureAudit(
        disagreement_threshold=disagreement_limit,
        error_threshold=error_limit,
        valid_count=int(np.count_nonzero(valid)),
        low_disagreement_low_error_count=int(np.count_nonzero(low_low)),
        high_disagreement_low_error_count=int(np.count_nonzero(high_low)),
        low_disagreement_high_error_count=int(np.count_nonzero(low_high)),
        high_disagreement_high_error_count=int(np.count_nonzero(high_high)),
        low_disagreement_high_error_mean=(
            float(np.mean(low_high_errors)) if low_high_errors.size else 0.0
        ),
        low_disagreement_high_error_max=(
            float(np.max(low_high_errors)) if low_high_errors.size else 0.0
        ),
    )


__all__ = [
    "CommonModeFailureAudit",
    "SourceOnlyDiagnosticGrid",
    "audit_common_mode_failures",
    "augment_source_reliability_features",
    "build_common_gauge_seed_dispersion_diagnostic",
    "build_flow_point_consistency_diagnostic",
]
