"""Source-only, group-balanced calibration of observation prior reliability.

The calibrated probability is a prior statement about source-side nominality. It
must not use a Bayesian-PhysTwin physical innovation, a downstream posterior
responsibility, or association probability as a substitute label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from .data import PredictionWindow
from .uncertainty import DisagreementEvidence, StructuredCovariance

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]

SOURCE_RELIABILITY_SCHEMA = "prob4d.source-reliability-calibration"
SOURCE_RELIABILITY_VERSION = 1


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


@dataclass(frozen=True)
class SourceReliabilityFeatures:
    """Source-only feature grid and the rows eligible for calibration/use."""

    feature_names: tuple[str, ...]
    values: FloatArray
    valid_mask: BoolArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.feature_names)
        if not names or any(not value for value in names):
            raise ValueError("feature_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        values = np.asarray(self.values, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=bool)
        if values.ndim < 2 or values.shape[-1] != len(names):
            raise ValueError("feature values must have shape (..., feature_count)")
        if mask.shape != values.shape[:-1]:
            raise ValueError("valid_mask must match the feature leading dimensions")
        if not np.all(np.isfinite(values)):
            raise ValueError("source reliability features must be finite")
        if not np.any(mask):
            raise ValueError("source reliability feature grid has no valid rows")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", _readonly(values, dtype=np.float64))
        object.__setattr__(self, "valid_mask", _readonly(mask, dtype=bool))
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="source reliability feature metadata",
            ),
        )

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def row_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    def flattened(self) -> FloatArray:
        return self.values[self.valid_mask].copy()

    def summary(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_count": self.feature_count,
            "row_count": self.row_count,
            "grid_shape": list(self.valid_mask.shape),
        }


def _local_validity_deficit(valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=np.float64)
    padded = np.pad(valid, ((0, 0), (1, 1), (1, 1)), mode="constant")
    count = np.zeros_like(valid)
    height, width = valid.shape[1:]
    for row_offset in range(3):
        for column_offset in range(3):
            count += padded[
                :,
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return 1.0 - count / 9.0


def build_source_reliability_features(
    window: PredictionWindow,
    covariance: StructuredCovariance,
    evidence: DisagreementEvidence | None = None,
) -> SourceReliabilityFeatures:
    """Build finite features without truth or downstream physical residuals."""

    if covariance.parallel_variance.shape != window.shape:
        raise ValueError("structured covariance shape differs from prediction window")
    if evidence is not None and evidence.count.shape != window.shape:
        raise ValueError("disagreement evidence shape differs from prediction window")

    valid = np.asarray(window.valid_mask, dtype=bool)
    parallel = covariance.parallel_variance
    lateral = covariance.lateral_variance
    total_variance = parallel + 2.0 * lateral
    variance_scale = max(
        float(np.median(total_variance[valid])),
        np.finfo(np.float64).tiny,
    )
    log_relative_variance = np.log(
        np.maximum(total_variance / variance_scale, np.finfo(np.float64).tiny)
    )

    if evidence is None:
        has_overlap = np.zeros(window.shape, dtype=bool)
        normalized_disagreement = np.zeros(window.shape, dtype=np.float64)
    else:
        has_overlap = evidence.count > 0.0
        normalized_disagreement = (
            evidence.parallel_mean / np.maximum(parallel, 1e-15)
            + evidence.lateral_mean / np.maximum(lateral, 1e-15)
        )
        normalized_disagreement = np.where(
            has_overlap,
            normalized_disagreement,
            0.0,
        )
    log_disagreement = np.log1p(np.maximum(normalized_disagreement, 0.0))

    time_steps = window.shape[0]
    if time_steps == 1:
        temporal_edge = np.ones(1, dtype=np.float64)
    else:
        positions = np.arange(time_steps, dtype=np.float64)
        distance = np.minimum(positions, time_steps - 1.0 - positions)
        temporal_edge = 1.0 - distance / max((time_steps - 1.0) / 2.0, 1.0)
        temporal_edge = np.clip(temporal_edge, 0.0, 1.0)
    temporal_edge_grid = np.broadcast_to(
        temporal_edge[:, None, None],
        window.shape,
    )

    has_scene_flow = np.zeros(window.shape, dtype=bool)
    log_relative_flow = np.zeros(window.shape, dtype=np.float64)
    if window.scene_flow is not None and window.deform_mask is not None:
        has_scene_flow = window.deform_mask & valid
        flow_norm = np.linalg.norm(window.scene_flow, axis=-1)
        positive_flow = flow_norm[has_scene_flow & (flow_norm > 0.0)]
        flow_scale = (
            float(np.median(positive_flow))
            if len(positive_flow)
            else 1.0
        )
        flow_scale = max(flow_scale, np.finfo(np.float64).tiny)
        log_relative_flow = np.log1p(flow_norm / flow_scale)
        log_relative_flow = np.where(has_scene_flow, log_relative_flow, 0.0)

    validity_deficit = _local_validity_deficit(valid)
    values = np.stack(
        (
            has_overlap.astype(np.float64),
            log_disagreement,
            temporal_edge_grid,
            log_relative_variance,
            has_scene_flow.astype(np.float64),
            log_relative_flow,
            validity_deficit,
        ),
        axis=-1,
    )
    return SourceReliabilityFeatures(
        feature_names=(
            "has_overlap",
            "log1p_normalized_overlap_disagreement",
            "temporal_edge_proximity",
            "log_relative_total_variance",
            "has_scene_flow",
            "log1p_relative_scene_flow",
            "local_validity_deficit",
        ),
        values=values,
        valid_mask=valid,
        metadata={
            "semantics": "prob4d-source-only-reliability-features-v1",
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
            "uses_association_probability": False,
            "window_id": window.window_id,
            "frame_indices": [int(value) for value in window.frame_indices],
            "variance_scale": variance_scale,
            "overlap_evidence_available": evidence is not None,
            "scene_flow_available": window.scene_flow is not None,
        },
    )


@dataclass(frozen=True)
class SourceReliabilityCalibrationReport:
    """Optimization and held-in calibration diagnostics."""

    count: int
    group_count: int
    feature_count: int
    iterations: int
    converged: bool
    group_balanced_nominal_fraction: float
    weighted_log_loss: float
    weighted_brier_score: float
    ridge: float

    def __post_init__(self) -> None:
        integer_fields = ("count", "group_count", "feature_count", "iterations")
        for name in integer_fields:
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        diagnostics = np.asarray(
            [
                self.group_balanced_nominal_fraction,
                self.weighted_log_loss,
                self.weighted_brier_score,
                self.ridge,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("source reliability diagnostics must be finite and non-negative")
        if diagnostics[0] > 1.0:
            raise ValueError("nominal fraction must lie in [0, 1]")
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(
            self,
            "group_balanced_nominal_fraction",
            float(diagnostics[0]),
        )
        object.__setattr__(self, "weighted_log_loss", float(diagnostics[1]))
        object.__setattr__(self, "weighted_brier_score", float(diagnostics[2]))
        object.__setattr__(self, "ridge", float(diagnostics[3]))

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SourceReliabilityModelV1:
    """Content-addressed group-balanced logistic source-reliability model."""

    feature_names: tuple[str, ...]
    feature_center: FloatArray
    feature_scale: FloatArray
    coefficients: FloatArray
    minimum_probability: float
    maximum_probability: float
    label_definition: str
    group_definition: str
    calibration_group_ids: tuple[str, ...]
    report: SourceReliabilityCalibrationReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.feature_names)
        if not names or any(not value for value in names):
            raise ValueError("feature_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        feature_count = len(names)
        if center.shape != (feature_count,) or scale.shape != (feature_count,):
            raise ValueError("feature center and scale must match feature_names")
        if coefficients.shape != (feature_count + 1,):
            raise ValueError("coefficients must contain intercept plus one value per feature")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("source reliability model values must be finite")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("feature_scale must be finite and positive")
        minimum = float(self.minimum_probability)
        maximum = float(self.maximum_probability)
        if not 0.0 < minimum < maximum < 1.0:
            raise ValueError("probability limits must satisfy 0 < minimum < maximum < 1")
        label_definition = str(self.label_definition).strip()
        group_definition = str(self.group_definition).strip()
        if not label_definition or not group_definition:
            raise ValueError("label_definition and group_definition must be non-empty")
        groups = tuple(sorted(str(value).strip() for value in self.calibration_group_ids))
        if not groups or any(not value for value in groups) or len(set(groups)) != len(groups):
            raise ValueError("calibration_group_ids must be non-empty and unique")
        if self.report.feature_count != feature_count:
            raise ValueError("calibration report feature count changed")
        if self.report.group_count != len(groups):
            raise ValueError("calibration report group count changed")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_center", _readonly(center, dtype=np.float64))
        object.__setattr__(self, "feature_scale", _readonly(scale, dtype=np.float64))
        object.__setattr__(
            self,
            "coefficients",
            _readonly(coefficients, dtype=np.float64),
        )
        object.__setattr__(self, "minimum_probability", minimum)
        object.__setattr__(self, "maximum_probability", maximum)
        object.__setattr__(self, "label_definition", label_definition)
        object.__setattr__(self, "group_definition", group_definition)
        object.__setattr__(self, "calibration_group_ids", groups)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="source reliability calibration metadata",
            ),
        )

    def predict(self, features: SourceReliabilityFeatures | FloatArray) -> FloatArray:
        if isinstance(features, SourceReliabilityFeatures):
            if features.feature_names != self.feature_names:
                raise ValueError("source reliability feature names changed")
            values = features.values
        else:
            values = np.asarray(features, dtype=np.float64)
        if values.shape[-1:] != (len(self.feature_names),):
            raise ValueError("feature array has changed final dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        standardized = (values - self.feature_center) / self.feature_scale
        logits = self.coefficients[0] + np.einsum(
            "...f,f->...",
            standardized,
            self.coefficients[1:],
        )
        probability = _sigmoid(logits)
        probability = np.clip(
            probability,
            self.minimum_probability,
            self.maximum_probability,
        )
        if isinstance(features, SourceReliabilityFeatures):
            probability = np.where(features.valid_mask, probability, 0.0)
        return probability

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_RELIABILITY_SCHEMA,
            "version": SOURCE_RELIABILITY_VERSION,
            "feature_names": list(self.feature_names),
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "minimum_probability": self.minimum_probability,
            "maximum_probability": self.maximum_probability,
            "label_definition": self.label_definition,
            "group_definition": self.group_definition,
            "calibration_group_ids": list(self.calibration_group_ids),
            "report": self.report.to_dict(),
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceReliabilityModelV1:
        if payload.get("schema") != SOURCE_RELIABILITY_SCHEMA:
            raise ValueError("unexpected source reliability schema")
        if payload.get("version") != SOURCE_RELIABILITY_VERSION:
            raise ValueError("unsupported source reliability version")
        report_payload = payload.get("report")
        if not isinstance(report_payload, Mapping):
            raise ValueError("source reliability artifact is missing its report")
        try:
            report = SourceReliabilityCalibrationReport(
                count=int(report_payload["count"]),
                group_count=int(report_payload["group_count"]),
                feature_count=int(report_payload["feature_count"]),
                iterations=int(report_payload["iterations"]),
                converged=bool(report_payload["converged"]),
                group_balanced_nominal_fraction=float(
                    report_payload["group_balanced_nominal_fraction"]
                ),
                weighted_log_loss=float(report_payload["weighted_log_loss"]),
                weighted_brier_score=float(
                    report_payload["weighted_brier_score"]
                ),
                ridge=float(report_payload["ridge"]),
            )
            artifact = cls(
                feature_names=tuple(payload["feature_names"]),
                feature_center=np.asarray(payload["feature_center"], dtype=np.float64),
                feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
                coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
                minimum_probability=float(payload["minimum_probability"]),
                maximum_probability=float(payload["maximum_probability"]),
                label_definition=str(payload["label_definition"]),
                group_definition=str(payload["group_definition"]),
                calibration_group_ids=tuple(payload["calibration_group_ids"]),
                report=report,
                metadata=payload.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid source reliability artifact") from error
        if str(payload.get("artifact_id", "")) != artifact.artifact_id:
            raise ValueError("source reliability artifact_id does not match content")
        return artifact


def _canonical_training_rows(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys: list[np.ndarray] = [labels]
    keys.extend(features[:, index] for index in reversed(range(features.shape[1])))
    keys.append(groups)
    order = np.lexsort(tuple(keys))
    return features[order], labels[order], groups[order]


def _weighted_objective(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    parameters: np.ndarray,
    ridge: float,
) -> float:
    logits = design @ parameters
    data_loss = np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits))
    return float(data_loss + 0.5 * ridge * np.sum(parameters[1:] ** 2))


def fit_group_balanced_source_reliability(
    features: FloatArray,
    nominal_labels: BoolArray | FloatArray,
    group_ids: np.ndarray,
    *,
    feature_names: Sequence[str],
    mask: BoolArray | None = None,
    ridge: float = 1e-3,
    maximum_iterations: int = 100,
    convergence_tolerance: float = 1e-9,
    minimum_probability: float = 0.01,
    maximum_probability: float = 0.99,
    label_definition: str,
    group_definition: str,
    metadata: Mapping[str, Any] | None = None,
) -> SourceReliabilityModelV1:
    """Fit deterministic logistic nominality with equal mass per group."""

    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(nominal_labels, dtype=np.float64)
    groups = np.asarray(group_ids)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise ValueError("features must have shape (..., feature_count)")
    leading_shape = values.shape[:-1]
    if labels.shape != leading_shape or groups.shape != leading_shape:
        raise ValueError("labels and group_ids must match feature leading dimensions")
    names = tuple(str(value) for value in feature_names)
    if len(names) != values.shape[-1]:
        raise ValueError("feature_names length differs from feature count")
    active = np.ones(leading_shape, dtype=bool)
    if mask is not None:
        supplied_mask = np.asarray(mask, dtype=bool)
        if supplied_mask.shape != leading_shape:
            raise ValueError("mask must match feature leading dimensions")
        active &= supplied_mask
    if not np.any(active):
        raise ValueError("source reliability calibration has no active rows")
    active_values = values[active]
    active_labels = labels[active]
    raw_groups = groups[active].reshape(-1)
    if not np.all(np.isfinite(active_values)):
        raise ValueError("active calibration features must be finite")
    if not np.all(np.isfinite(active_labels)) or not np.all(
        (active_labels == 0.0) | (active_labels == 1.0)
    ):
        raise ValueError("nominal_labels must contain only zero or one")
    if len(np.unique(active_labels)) != 2:
        raise ValueError("source reliability calibration requires both label classes")
    normalized_groups = np.asarray(
        ["" if value is None else str(value).strip() for value in raw_groups],
        dtype=str,
    )
    if np.any(normalized_groups == ""):
        raise ValueError("active calibration group IDs must be non-empty")
    active_values, active_labels, normalized_groups = _canonical_training_rows(
        active_values,
        active_labels,
        normalized_groups,
    )
    canonical_groups = tuple(sorted(set(normalized_groups.tolist())))
    group_count = len(canonical_groups)
    weights = np.empty(len(active_labels), dtype=np.float64)
    for group_id in canonical_groups:
        selected = normalized_groups == group_id
        weights[selected] = 1.0 / (group_count * int(np.count_nonzero(selected)))
    if not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise RuntimeError("group-balanced calibration weights do not sum to one")

    ridge = float(ridge)
    tolerance = float(convergence_tolerance)
    maximum_iterations = int(maximum_iterations)
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be finite and positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not 0.0 < minimum_probability < maximum_probability < 1.0:
        raise ValueError("probability limits must satisfy 0 < minimum < maximum < 1")

    center = np.sum(weights[:, None] * active_values, axis=0)
    centered = active_values - center
    scale = np.sqrt(np.sum(weights[:, None] * centered**2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = centered / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    nominal_fraction = float(np.sum(weights * active_labels))
    initial_probability = float(np.clip(nominal_fraction, 1e-6, 1.0 - 1e-6))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[0] = np.log(initial_probability / (1.0 - initial_probability))
    regularizer = np.diag(np.r_[0.0, np.full(values.shape[-1], ridge)])
    converged = False
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        logits = design @ parameters
        probability = _sigmoid(logits)
        curvature = weights * probability * (1.0 - probability)
        gradient = design.T @ (weights * (probability - active_labels))
        gradient[1:] += ridge * parameters[1:]
        hessian = design.T @ (curvature[:, None] * design) + regularizer
        hessian += 1e-12 * np.eye(hessian.shape[0])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.linalg.norm(step) <= tolerance * (1.0 + np.linalg.norm(parameters)):
            converged = True
            break
        current_objective = _weighted_objective(
            design,
            active_labels,
            weights,
            parameters,
            ridge,
        )
        accepted = False
        fraction = 1.0
        for _ in range(20):
            candidate = parameters - fraction * step
            candidate_objective = _weighted_objective(
                design,
                active_labels,
                weights,
                candidate,
                ridge,
            )
            if candidate_objective < current_objective - 1e-14:
                parameters = candidate
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            break
        if np.linalg.norm(fraction * step) <= tolerance * (
            1.0 + np.linalg.norm(parameters)
        ):
            converged = True
            break

    fitted_probability = np.clip(
        _sigmoid(design @ parameters),
        minimum_probability,
        maximum_probability,
    )
    weighted_log_loss = float(
        -np.sum(
            weights
            * (
                active_labels * np.log(fitted_probability)
                + (1.0 - active_labels) * np.log(1.0 - fitted_probability)
            )
        )
    )
    weighted_brier = float(
        np.sum(weights * (fitted_probability - active_labels) ** 2)
    )
    report = SourceReliabilityCalibrationReport(
        count=len(active_labels),
        group_count=group_count,
        feature_count=values.shape[-1],
        iterations=iterations,
        converged=converged,
        group_balanced_nominal_fraction=nominal_fraction,
        weighted_log_loss=weighted_log_loss,
        weighted_brier_score=weighted_brier,
        ridge=ridge,
    )
    return SourceReliabilityModelV1(
        feature_names=names,
        feature_center=center,
        feature_scale=scale,
        coefficients=parameters,
        minimum_probability=minimum_probability,
        maximum_probability=maximum_probability,
        label_definition=label_definition,
        group_definition=group_definition,
        calibration_group_ids=canonical_groups,
        report=report,
        metadata={} if metadata is None else metadata,
    )


def save_source_reliability_model(
    model: SourceReliabilityModelV1,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(model.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_source_reliability_model(path: str | Path) -> SourceReliabilityModelV1:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source reliability artifact must be a JSON object")
    return SourceReliabilityModelV1.from_dict(payload)


__all__ = [
    "SOURCE_RELIABILITY_SCHEMA",
    "SOURCE_RELIABILITY_VERSION",
    "SourceReliabilityCalibrationReport",
    "SourceReliabilityFeatures",
    "SourceReliabilityModelV1",
    "build_source_reliability_features",
    "fit_group_balanced_source_reliability",
    "load_source_reliability_model",
    "save_source_reliability_model",
]
