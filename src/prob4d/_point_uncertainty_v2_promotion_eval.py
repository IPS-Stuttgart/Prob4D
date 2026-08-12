"""Numerical replay for point uncertainty v2 promotion evidence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._point_uncertainty_v2_promotion_report import PointUncertaintyPromotionReportV1
from ._point_uncertainty_v2_promotion_types import (
    PointUncertaintyGroupMetricsV1,
    PointUncertaintyPromotionPolicyV1,
    _CHI2_DF3_90,
    _LOG_2PI,
)
from ._strict_calibration import PointUncertaintyCalibrationV1
from ._strict_json import require_exact_string, require_mapping, require_sha256
from .group_balanced_point_calibration import group_balanced_point_calibration_metadata
from .point_uncertainty_v2 import PointUncertaintyCalibrationV2, local_point_basis


def _baseline_training_group_ids(
    calibration: PointUncertaintyCalibrationV1,
) -> tuple[str, ...]:
    metadata = group_balanced_point_calibration_metadata(calibration)
    if metadata is None:
        raise ValueError(
            "baseline v1 calibration must carry equal-group calibration provenance"
        )
    report = require_mapping(metadata["report"], name="baseline group report")
    groups = report["groups"]
    if type(groups) not in {list, tuple}:
        raise ValueError("baseline group report groups changed")
    result: list[str] = []
    for index, raw in enumerate(groups):
        item = require_mapping(raw, name=f"baseline groups[{index}]")
        result.append(
            require_exact_string(item.get("group_id"), name=f"baseline groups[{index}].group_id")
        )
    canonical = tuple(sorted(set(result)))
    if len(canonical) != len(result):
        raise ValueError("baseline group report group IDs must be unique")
    return canonical


def _production_v1_variances(
    calibration: PointUncertaintyCalibrationV1,
    *,
    depth_squared: object,
    disagreement_parallel_mean: object,
    disagreement_lateral_mean: object,
    rows: int,
) -> np.ndarray:
    depth = np.asarray(depth_squared, dtype=np.float64)
    parallel_disagreement = np.asarray(disagreement_parallel_mean, dtype=np.float64)
    lateral_disagreement = np.asarray(disagreement_lateral_mean, dtype=np.float64)
    for name, values in (
        ("depth_squared", depth),
        ("disagreement_parallel_mean", parallel_disagreement),
        ("disagreement_lateral_mean", lateral_disagreement),
    ):
        if values.shape != (rows,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with shape ({rows},)")
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative")
    model = calibration.model
    parallel = (
        model.parallel_floor
        + model.parallel_depth_coefficient * depth
        + 0.5 * model.disagreement_gain * parallel_disagreement
    ) * model.parallel_scale
    lateral = (
        model.lateral_floor
        + model.lateral_depth_coefficient * depth
        + 0.5 * model.disagreement_gain * lateral_disagreement
    ) * model.lateral_scale
    epsilon = np.finfo(np.float64).eps
    return np.column_stack(
        (
            np.maximum(parallel, epsilon),
            np.maximum(lateral, epsilon),
            np.maximum(lateral, epsilon),
        )
    )


def _row_scores(
    residual_xyz: np.ndarray,
    rays: np.ndarray,
    tangent_one: np.ndarray,
    tangent_two: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = np.column_stack(
        (
            np.sum(residual_xyz * rays, axis=1),
            np.sum(residual_xyz * tangent_one, axis=1),
            np.sum(residual_xyz * tangent_two, axis=1),
        )
    )
    energy = np.sum(projected * projected / variances, axis=1)
    nll = 0.5 * (3.0 * _LOG_2PI + np.sum(np.log(variances), axis=1) + energy)
    rms_width = np.sqrt(np.mean(variances, axis=1))
    return nll, energy, rms_width


def evaluate_point_uncertainty_v2_promotion(
    calibration: PointUncertaintyCalibrationV2,
    *,
    residual_xyz: object,
    ray_directions: object,
    tangent_reference: object,
    features: object,
    feature_names: Sequence[str],
    group_ids: Sequence[str],
    baseline_calibration: PointUncertaintyCalibrationV1,
    depth_squared: object,
    disagreement_parallel_mean: object,
    disagreement_lateral_mean: object,
    provider_manifest_id: str,
    cohort_binding_id: str,
    validation_sha256: str,
    policy: PointUncertaintyPromotionPolicyV1,
) -> PointUncertaintyPromotionReportV1:
    """Compare v2 against v1 on disjoint independent validation groups."""

    if not isinstance(calibration, PointUncertaintyCalibrationV2):
        raise TypeError("calibration must be PointUncertaintyCalibrationV2")
    if not isinstance(baseline_calibration, PointUncertaintyCalibrationV1):
        raise TypeError("baseline_calibration must be PointUncertaintyCalibrationV1")
    baseline_training_groups = _baseline_training_group_ids(baseline_calibration)
    if baseline_training_groups != calibration.group_ids:
        raise ValueError(
            "baseline v1 and candidate v2 must use the same independent training groups"
        )
    provider_manifest_id = require_sha256(provider_manifest_id, name="provider_manifest_id")
    cohort_binding_id = require_sha256(cohort_binding_id, name="cohort_binding_id")
    validation_sha256 = require_sha256(validation_sha256, name="validation_sha256")
    if provider_manifest_id != calibration.provider_manifest_id:
        raise ValueError("validation provider manifest does not match the calibration")
    if cohort_binding_id != calibration.cohort_binding_id:
        raise ValueError("validation cohort binding does not match the calibration")
    names = tuple(feature_names)
    if names != calibration.feature_names:
        raise ValueError("validation feature_names must exactly match the calibration")

    residual = np.asarray(residual_xyz, dtype=np.float64)
    if residual.ndim != 2 or residual.shape[1] != 3 or not np.all(np.isfinite(residual)):
        raise ValueError("residual_xyz must be finite with shape (N, 3)")
    rows = residual.shape[0]
    if rows == 0:
        raise ValueError("validation data must not be empty")
    group_rows = tuple(
        require_exact_string(item, name=f"group_ids[{index}]")
        for index, item in enumerate(group_ids)
    )
    if len(group_rows) != rows:
        raise ValueError("group_ids must contain one ID per validation row")
    validation_groups = tuple(sorted(set(group_rows)))
    overlap = set(calibration.group_ids).intersection(validation_groups)
    if overlap:
        raise ValueError(
            "point uncertainty promotion requires disjoint training and validation "
            f"groups; overlap={sorted(overlap)}"
        )

    rays, tangent_one, tangent_two = local_point_basis(
        ray_directions,
        tangent_reference,
    )
    if rays.shape[0] != rows:
        raise ValueError("basis inputs and residual_xyz must have matching rows")
    feature_matrix = np.asarray(features, dtype=np.float64)
    if feature_matrix.shape != (rows, len(names)) or not np.all(np.isfinite(feature_matrix)):
        raise ValueError("features have invalid shape or non-finite values")
    candidate_variances = calibration.predict_variances(feature_matrix)
    baseline_variances = _production_v1_variances(
        baseline_calibration,
        depth_squared=depth_squared,
        disagreement_parallel_mean=disagreement_parallel_mean,
        disagreement_lateral_mean=disagreement_lateral_mean,
        rows=rows,
    )
    baseline_nll, baseline_energy, baseline_width = _row_scores(
        residual,
        rays,
        tangent_one,
        tangent_two,
        baseline_variances,
    )
    candidate_nll, candidate_energy, candidate_width = _row_scores(
        residual,
        rays,
        tangent_one,
        tangent_two,
        candidate_variances,
    )

    group_array = np.asarray(group_rows)
    metrics: list[PointUncertaintyGroupMetricsV1] = []
    for group_id in validation_groups:
        selected = group_array == group_id
        count = int(np.sum(selected))
        metrics.append(
            PointUncertaintyGroupMetricsV1(
                group_id=group_id,
                count=count,
                baseline_mean_nll=float(np.mean(baseline_nll[selected])),
                candidate_mean_nll=float(np.mean(candidate_nll[selected])),
                baseline_coverage90=float(
                    np.mean(baseline_energy[selected] <= _CHI2_DF3_90)
                ),
                candidate_coverage90=float(
                    np.mean(candidate_energy[selected] <= _CHI2_DF3_90)
                ),
                baseline_mean_rms_width=float(np.mean(baseline_width[selected])),
                candidate_mean_rms_width=float(np.mean(candidate_width[selected])),
                baseline_normalized_energy=float(np.mean(baseline_energy[selected]) / 3.0),
                candidate_normalized_energy=float(np.mean(candidate_energy[selected]) / 3.0),
            )
        )
    return PointUncertaintyPromotionReportV1(
        point_uncertainty_calibration_id=calibration.point_uncertainty_calibration_id,
        baseline_point_calibration_id=baseline_calibration.artifact_id,
        provider_manifest_id=provider_manifest_id,
        cohort_binding_id=cohort_binding_id,
        validation_sha256=validation_sha256,
        training_group_ids=calibration.group_ids,
        baseline_training_group_ids=baseline_training_groups,
        groups=tuple(metrics),
        policy=policy,
        fit_converged=calibration.fit_converged,
    )



__all__ = ["evaluate_point_uncertainty_v2_promotion"]
