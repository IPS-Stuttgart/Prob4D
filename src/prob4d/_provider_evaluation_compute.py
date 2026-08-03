"""Evaluation and paired group-bootstrap aggregation for Prob4D artifacts."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._provider_evaluation_manifest import ProviderEvaluationCase
from .data import DENSE_STORAGE_DTYPES
from .evaluation_modes import EvaluationModes, evaluate_sequence_modes
from .fusion import FusedSequence
from .io import (
    FusedPredictionArtifact,
    FusedPredictionMetadata,
    load_fused_prediction_artifact,
    load_truth,
)
from .metrics import DEFAULT_EVALUATION_CHUNK_SIZE, TruthSequence


def _numeric_leaves(value: object, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = str(key) if not prefix else f"{prefix}.{key}"
            result.update(_numeric_leaves(item, prefix=path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            result[prefix] = numeric
    return result


def _stable_seed(seed: int, method_id: str, metric: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(method_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(metric.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def _validated_revision(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an exact Git revision")
    revision = value.strip()
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            f"{name} must be a lowercase 40- or 64-character Git revision"
        )
    return revision


def _validated_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 digest")
    digest = value.strip()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validate_method_metadata(
    metadata: FusedPredictionMetadata,
    *,
    path: Path,
) -> None:
    details = metadata.metadata
    _validated_revision(
        details.get("prob4d_revision"),
        name=f"{path} metadata.prob4d_revision",
    )
    _validated_revision(
        details.get("motioncrafter_revision"),
        name=f"{path} metadata.motioncrafter_revision",
    )
    _validated_sha256(
        details.get("motioncrafter_model_set_sha256"),
        name=f"{path} metadata.motioncrafter_model_set_sha256",
    )
    _validated_sha256(
        details.get("prediction_manifest_sha256"),
        name=f"{path} metadata.prediction_manifest_sha256",
    )
    seed_policy = details.get("motioncrafter_seed_policy")
    if seed_policy not in {"legacy-common", "derived-per-call"}:
        raise ValueError(
            f"{path} metadata.motioncrafter_seed_policy must declare the exact "
            "MotionCrafter stochastic semantics"
        )
    if details.get("includes_covariance") is not True:
        raise ValueError(
            f"{path} metadata.includes_covariance must be true for provider evaluation"
        )
    dense_storage_dtype = details.get("dense_storage_dtype", "float64")
    if dense_storage_dtype not in DENSE_STORAGE_DTYPES:
        raise ValueError(
            f"{path} metadata.dense_storage_dtype must be one of "
            + ", ".join(DENSE_STORAGE_DTYPES)
        )
    for field in ("gauge_estimator", "uncertainty_calibration"):
        value = details.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} metadata.{field} must be nonempty text")


def _bootstrap_summary(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    estimate = float(np.mean(values))
    if values.size == 1:
        lower = upper = estimate
    else:
        generator = np.random.default_rng(seed)
        indices = generator.integers(
            0,
            values.size,
            size=(resamples, values.size),
        )
        means = np.mean(values[indices], axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "mean": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "group_count": int(values.size),
    }


def _method_signature(metadata: FusedPredictionMetadata) -> tuple[object, ...]:
    details = metadata.metadata
    return (
        metadata.fusion_method,
        metadata.covariance_semantics,
        metadata.correlation_assumption,
        details.get("prob4d_revision"),
        details.get("motioncrafter_revision"),
        details.get("motioncrafter_seed_policy"),
        details.get("motioncrafter_model_set_sha256"),
        details.get("dense_storage_dtype", "float64"),
        details.get("gauge_estimator"),
        details.get("uncertainty_calibration"),
        details.get("gauge_covariance_calibration_artifact_id"),
        details.get("point_uncertainty_calibration_artifact_id"),
        details.get("source_reliability_artifact_id"),
    )


def _paired_common_support(
    truth: TruthSequence,
    predictions: Mapping[str, FusedSequence],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Build truth-indexed support shared by every registered method in one case."""

    if not predictions:
        raise ValueError("provider evaluation requires at least one prediction method")
    spatial_shape = truth.valid_mask.shape[1:]
    common_frames = truth.frame_indices
    for method_id, prediction in sorted(predictions.items()):
        if prediction.valid_mask.shape[1:] != spatial_shape:
            raise ValueError(
                f"method {method_id!r} has spatial support "
                f"{prediction.valid_mask.shape[1:]}, expected {spatial_shape}"
            )
        common_frames = np.intersect1d(common_frames, prediction.frame_indices)
    if common_frames.size == 0:
        raise ValueError("registered methods and truth have no common frames")

    truth_indices = np.searchsorted(truth.frame_indices, common_frames)
    common_point_support = np.zeros_like(truth.valid_mask, dtype=bool)
    active_points = truth.valid_mask[truth_indices].copy()
    prediction_indices: dict[str, np.ndarray] = {}
    for method_id, prediction in sorted(predictions.items()):
        indices = np.searchsorted(prediction.frame_indices, common_frames)
        prediction_indices[method_id] = indices
        active_points &= prediction.valid_mask[indices]
    common_point_support[truth_indices] = active_points
    common_point_count = int(np.count_nonzero(active_points))
    if common_point_count == 0:
        raise ValueError(
            "registered methods and truth have no jointly valid common point support"
        )

    all_methods_have_flow = truth.scene_flow is not None and all(
        prediction.scene_flow is not None for prediction in predictions.values()
    )
    common_flow_support: np.ndarray | None = None
    common_flow_count = 0
    if truth.scene_flow is not None:
        common_flow_support = np.zeros_like(truth.valid_mask, dtype=bool)
        if all_methods_have_flow:
            assert truth.deform_mask is not None
            active_flow = truth.deform_mask[truth_indices] & active_points
            for method_id, prediction in sorted(predictions.items()):
                assert prediction.deform_mask is not None
                active_flow &= prediction.deform_mask[
                    prediction_indices[method_id]
                ]
            common_flow_support[truth_indices] = active_flow
            common_flow_count = int(np.count_nonzero(active_flow))

    truth_valid_on_common_frames = int(
        np.count_nonzero(truth.valid_mask[truth_indices])
    )
    return common_point_support, common_flow_support, {
        "truth_frame_count": int(truth.frame_indices.size),
        "common_frame_count": int(common_frames.size),
        "common_frame_fraction_of_truth": float(
            common_frames.size / truth.frame_indices.size
        ),
        "truth_valid_points_on_common_frames": truth_valid_on_common_frames,
        "common_valid_points": common_point_count,
        "common_point_fraction_of_truth_on_common_frames": float(
            common_point_count / truth_valid_on_common_frames
        ),
        "all_methods_have_flow": all_methods_have_flow,
        "common_valid_flow_points": common_flow_count,
    }


def _method_support_summary(
    *,
    common: EvaluationModes,
    native: EvaluationModes,
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    common_metrics = common.metric.metrics
    native_metrics = native.metric.metrics
    common_points = common_metrics.evaluated_points
    native_points = native_metrics.evaluated_points
    common_frames = common_metrics.evaluated_frames
    native_frames = native_metrics.evaluated_frames
    common_flow = common_metrics.evaluated_flow_points
    native_flow = native_metrics.evaluated_flow_points
    return {
        **paired,
        "native_valid_points": native_points,
        "common_point_fraction_of_native": float(common_points / native_points),
        "native_evaluated_frames": native_frames,
        "common_evaluated_frames": common_frames,
        "common_frame_fraction_of_native": float(common_frames / native_frames),
        "native_valid_flow_points": native_flow,
        "common_flow_fraction_of_native": (
            float(common_flow / native_flow) if native_flow > 0 else None
        ),
    }


def evaluate_provider_cases(
    cases: list[ProviderEvaluationCase],
    *,
    allow_legacy_artifacts: bool,
    evaluation_chunk_size: int = DEFAULT_EVALUATION_CHUNK_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate every paired case and enforce one semantic signature per method."""

    if evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")
    records: list[dict[str, Any]] = []
    signatures: dict[str, tuple[object, ...]] = {}
    method_metadata: dict[str, dict[str, Any]] = {}
    for case in cases:
        truth = load_truth(case.truth_path)
        artifacts: dict[str, FusedPredictionArtifact] = {}
        for method_id, prediction_path in sorted(case.predictions.items()):
            artifact = load_fused_prediction_artifact(prediction_path)
            artifacts[method_id] = artifact
            metadata = artifact.metadata
            if metadata.legacy_unspecified and not allow_legacy_artifacts:
                raise ValueError(
                    f"{prediction_path} has legacy unspecified covariance semantics; "
                    "regenerate it or pass --allow-legacy-artifacts for an explicitly "
                    "non-claim-bearing diagnostic"
                )
            if not metadata.legacy_unspecified and metadata.method_id != method_id:
                raise ValueError(
                    f"prediction method label {method_id!r} disagrees with artifact "
                    f"method_id {metadata.method_id!r}"
                )
            if not metadata.legacy_unspecified:
                _validate_method_metadata(metadata, path=prediction_path)
            signature = _method_signature(metadata)
            previous = signatures.setdefault(method_id, signature)
            if signature != previous:
                raise ValueError(
                    f"method {method_id!r} mixes covariance, model, estimator, or "
                    "revision semantics"
                )
            method_metadata.setdefault(method_id, metadata.to_dict())

        common_point_support, common_flow_support, paired_support = (
            _paired_common_support(
                truth,
                {
                    method_id: artifact.sequence
                    for method_id, artifact in artifacts.items()
                },
            )
        )
        for method_id, prediction_path in sorted(case.predictions.items()):
            artifact = artifacts[method_id]
            native_modes: EvaluationModes = evaluate_sequence_modes(
                artifact.sequence,
                truth,
                boundary_frames=list(case.boundary_frames),
                prefix_frame_stop_exclusive=case.prefix_frame_stop_exclusive,
                evaluation_chunk_size=evaluation_chunk_size,
            )
            common_modes: EvaluationModes = evaluate_sequence_modes(
                artifact.sequence,
                truth,
                boundary_frames=list(case.boundary_frames),
                prefix_frame_stop_exclusive=case.prefix_frame_stop_exclusive,
                truth_support_mask=common_point_support,
                truth_flow_support_mask=common_flow_support,
                evaluation_chunk_size=evaluation_chunk_size,
            )
            evaluation = common_modes.to_dict()
            native_evaluation = native_modes.to_dict()
            support = _method_support_summary(
                common=common_modes,
                native=native_modes,
                paired=paired_support,
            )
            numeric = _numeric_leaves(evaluation)
            numeric.update(
                _numeric_leaves(
                    native_evaluation,
                    prefix="native_support",
                )
            )
            numeric.update(_numeric_leaves(support, prefix="support"))
            records.append(
                {
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "method_id": method_id,
                    "prediction_path": str(prediction_path),
                    "truth_path": str(case.truth_path),
                    "artifact": artifact.metadata.to_dict(),
                    "evaluation": evaluation,
                    "native_support_evaluation": native_evaluation,
                    "support": support,
                    "_numeric": numeric,
                }
            )
    return records, method_metadata


def aggregate_provider_records(
    records: list[dict[str, Any]],
    *,
    reference_method: str,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate equal-weight groups and paired candidate-minus-reference changes."""

    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    method_case_counts: dict[str, int] = defaultdict(int)
    case_group: dict[str, str] = {}
    case_method_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        case_id = str(record["case_id"])
        method_id = str(record["method_id"])
        group_id = str(record["group_id"])
        previous_group = case_group.setdefault(case_id, group_id)
        if previous_group != group_id:
            raise ValueError(f"case {case_id!r} changed group_id across methods")
        method_case_counts[method_id] += 1
        numeric = record["_numeric"]
        assert isinstance(numeric, dict)
        normalized = {str(metric): float(value) for metric, value in numeric.items()}
        case_method_metrics[(case_id, method_id)] = normalized
        for metric, value in normalized.items():
            grouped[method_id][group_id][metric].append(value)

    aggregate: dict[str, Any] = {}
    for method_id, groups in sorted(grouped.items()):
        metric_names = sorted(
            set.intersection(*(set(metrics) for metrics in groups.values()))
        )
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            group_ids = sorted(groups)
            group_means = np.asarray(
                [np.mean(groups[group_id][metric]) for group_id in group_ids],
                dtype=np.float64,
            )
            summary = _bootstrap_summary(
                group_means,
                resamples=bootstrap_resamples,
                seed=_stable_seed(seed, method_id, metric),
            )
            if metric.endswith("coverage_shortfall_95"):
                worst_index = int(np.argmax(group_means))
                summary["worst_group_mean"] = float(group_means[worst_index])
                summary["worst_group_id"] = group_ids[worst_index]
            metrics[metric] = summary
        aggregate[method_id] = {
            "case_count": method_case_counts[method_id],
            "group_count": len(groups),
            "aggregation": "equal_group_mean_after_within_group_case_mean",
            "bootstrap_unit": "group_id",
            "metrics": metrics,
        }

    case_ids = sorted(case_group)
    comparisons: dict[str, Any] = {}
    for candidate_method in sorted(grouped):
        if candidate_method == reference_method:
            continue
        metric_sets = [
            set(case_method_metrics[(case_id, reference_method)])
            & set(case_method_metrics[(case_id, candidate_method)])
            for case_id in case_ids
        ]
        comparable_metrics = sorted(
            metric
            for metric in set.intersection(*metric_sets)
            if ".metrics." in metric
        )
        comparison_metrics: dict[str, Any] = {}
        for metric in comparable_metrics:
            group_differences: dict[str, list[float]] = defaultdict(list)
            for case_id in case_ids:
                difference = (
                    case_method_metrics[(case_id, candidate_method)][metric]
                    - case_method_metrics[(case_id, reference_method)][metric]
                )
                group_differences[case_group[case_id]].append(difference)
            group_means = np.asarray(
                [
                    np.mean(group_differences[group_id])
                    for group_id in sorted(group_differences)
                ],
                dtype=np.float64,
            )
            comparison_metrics[metric] = _bootstrap_summary(
                group_means,
                resamples=bootstrap_resamples,
                seed=_stable_seed(
                    seed,
                    f"{candidate_method}-minus-{reference_method}",
                    metric,
                ),
            )
        comparisons[candidate_method] = {
            "reference_method": reference_method,
            "difference_semantics": "candidate_minus_reference",
            "case_count": method_case_counts[candidate_method],
            "group_count": len(grouped[candidate_method]),
            "aggregation": (
                "equal_group_mean_of_within_group_paired_case_differences"
            ),
            "bootstrap_unit": "group_id",
            "metrics": comparison_metrics,
        }
    return aggregate, comparisons


__all__ = [
    "aggregate_provider_records",
    "evaluate_provider_cases",
]
