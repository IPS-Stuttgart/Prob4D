"""Content-addressed equal-group point-uncertainty calibration helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._strict_calibration import PointUncertaintyCalibrationV1
from .calibration_aggregation import (
    GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
    LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1,
)
from .uncertainty import (
    CalibrationReport,
    DepthDisagreementModel,
    GroupBalancedCalibrationReport,
    StructuredCovariance,
)

_GROUP_BALANCED_METADATA_KEY = "group_balanced_point_uncertainty_calibration"


def fit_group_balanced_point_uncertainty_calibration(
    model: DepthDisagreementModel,
    errors: np.ndarray,
    covariance: StructuredCovariance,
    group_ids: np.ndarray,
    *,
    group_definition: str,
    calibration_case_ids: Sequence[str],
    source_repository: str,
    source_revision: str,
    motioncrafter_revision: str,
    model_identifier: str,
    covariance_method: str,
    mask: np.ndarray | None = None,
    trim_quantile: float = 0.99,
    image_resolution: tuple[int, int] | None = None,
    window_size: int | None = None,
    window_overlap: int | None = None,
    covariance_cluster_size: int | None = None,
    input_artifact_sha256: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[PointUncertaintyCalibrationV1, GroupBalancedCalibrationReport]:
    """Fit equal-group scales and bind the complete report into one artifact.

    `PointUncertaintyCalibrationV1` retains its existing schema and aggregate
    fields. The grouping definition, canonical group identities, per-group
    diagnostics, and aggregation semantics are stored in content-addressed
    provenance metadata, so a pooled artifact cannot be silently relabelled as
    group-balanced.
    """

    if not isinstance(group_definition, str) or not group_definition.strip():
        raise ValueError("group_definition must be a non-empty string")
    normalized_group_definition = group_definition.strip()
    if metadata is None:
        supplied_metadata: dict[str, Any] = {}
    elif not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    else:
        supplied_metadata = dict(metadata)
    if _GROUP_BALANCED_METADATA_KEY in supplied_metadata:
        raise ValueError(
            f"metadata key {_GROUP_BALANCED_METADATA_KEY!r} is reserved"
        )

    calibrated, group_report = model.calibrate_group_balanced(
        errors,
        covariance,
        group_ids,
        mask=mask,
        trim_quantile=trim_quantile,
    )
    aggregate_report = CalibrationReport(
        count=group_report.count,
        parallel_scale_update=group_report.parallel_scale_update,
        lateral_scale_update=group_report.lateral_scale_update,
        parallel_normalized_mse=group_report.parallel_normalized_mse,
        lateral_normalized_mse=group_report.lateral_normalized_mse,
    )
    supplied_metadata[_GROUP_BALANCED_METADATA_KEY] = {
        "group_definition": normalized_group_definition,
        "report": group_report.to_dict(),
    }
    artifact = PointUncertaintyCalibrationV1.from_model(
        calibrated,
        aggregate_report,
        calibration_case_ids=calibration_case_ids,
        source_repository=source_repository,
        source_revision=source_revision,
        motioncrafter_revision=motioncrafter_revision,
        model_identifier=model_identifier,
        covariance_method=covariance_method,
        trim_quantile=trim_quantile,
        image_resolution=image_resolution,
        window_size=window_size,
        window_overlap=window_overlap,
        covariance_cluster_size=covariance_cluster_size,
        input_artifact_sha256=input_artifact_sha256,
        metadata=supplied_metadata,
    )
    return artifact, group_report


def group_balanced_point_calibration_metadata(
    artifact: PointUncertaintyCalibrationV1,
) -> Mapping[str, Any] | None:
    """Return the validated equal-group provenance record when present."""

    value = artifact.metadata.get(_GROUP_BALANCED_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("group-balanced point calibration metadata is malformed")
    group_definition = value.get("group_definition")
    report = value.get("report")
    if not isinstance(group_definition, str) or not group_definition.strip():
        raise ValueError("group-balanced metadata has no group definition")
    if not isinstance(report, Mapping):
        raise ValueError("group-balanced metadata has no calibration report")
    aggregation = report.get("aggregation")
    if aggregation not in {
        GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
        LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1,
    }:
        raise ValueError("group-balanced metadata changed aggregation semantics")
    trim_quantile = report.get("trim_quantile")
    if isinstance(trim_quantile, bool) or not isinstance(trim_quantile, (int, float)):
        raise ValueError("group-balanced metadata trim_quantile must be numeric")
    normalized_quantile = float(trim_quantile)
    if not np.isfinite(normalized_quantile) or not 0.0 < normalized_quantile <= 1.0:
        raise ValueError("group-balanced metadata trim_quantile is invalid")
    winsor_quantile = report.get("winsor_quantile")
    if aggregation == GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2:
        if isinstance(winsor_quantile, bool) or not isinstance(
            winsor_quantile, (int, float)
        ):
            raise ValueError("winsorized group metadata has no winsor_quantile")
        if float(winsor_quantile) != normalized_quantile:
            raise ValueError("group-balanced winsor and legacy quantiles differ")
    elif winsor_quantile is not None:
        if isinstance(winsor_quantile, bool) or not isinstance(
            winsor_quantile, (int, float)
        ):
            raise ValueError("legacy winsor_quantile must be numeric when present")
        if float(winsor_quantile) != normalized_quantile:
            raise ValueError("legacy group-balanced quantile aliases differ")
    groups = report.get("groups")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("group-balanced metadata has no group diagnostics")
    supplied_group_count = report.get("group_count")
    if isinstance(supplied_group_count, bool) or not isinstance(
        supplied_group_count, int
    ):
        raise ValueError("group-balanced metadata group_count must be an integer")
    if supplied_group_count != len(groups):
        raise ValueError("group-balanced metadata group_count changed")
    return value


__all__ = [
    "fit_group_balanced_point_uncertainty_calibration",
    "group_balanced_point_calibration_metadata",
]
