"""Strict public calibration classes and content-addressed JSON loaders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ._gauge_calibration import (
    GaugeCovarianceCalibrationV1 as _GaugeCovarianceCalibrationV1,
    save_gauge_covariance_calibration,
)
from ._point_calibration import (
    PointUncertaintyCalibrationV1 as _PointUncertaintyCalibrationV1,
    save_point_uncertainty_calibration,
)
from ._strict_calibration_artifact import (
    validate_gauge_calibration_payload,
    validate_gauge_calibration_values,
    validate_point_calibration_payload,
    validate_point_calibration_values,
)
from ._strict_json import load_json_object


class GaugeCovarianceCalibrationV1(_GaugeCovarianceCalibrationV1):
    """Gauge calibration that rejects noncanonical constructor values."""

    def __post_init__(self) -> None:
        validate_gauge_calibration_values(self)
        super().__post_init__()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GaugeCovarianceCalibrationV1:
        validate_gauge_calibration_payload(payload)
        return cast(GaugeCovarianceCalibrationV1, super().from_dict(payload))


class PointUncertaintyCalibrationV1(_PointUncertaintyCalibrationV1):
    """Point calibration that rejects noncanonical constructor values."""

    def __post_init__(self) -> None:
        validate_point_calibration_values(self)
        super().__post_init__()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PointUncertaintyCalibrationV1:
        validate_point_calibration_payload(payload)
        return cast(PointUncertaintyCalibrationV1, super().from_dict(payload))


def load_gauge_covariance_calibration(
    path: str | Path,
) -> GaugeCovarianceCalibrationV1:
    payload = load_json_object(path, name="gauge covariance calibration")
    return GaugeCovarianceCalibrationV1.from_dict(payload)


def load_point_uncertainty_calibration(
    path: str | Path,
) -> PointUncertaintyCalibrationV1:
    payload = load_json_object(path, name="point uncertainty calibration")
    return PointUncertaintyCalibrationV1.from_dict(payload)


__all__ = [
    "GaugeCovarianceCalibrationV1",
    "PointUncertaintyCalibrationV1",
    "load_gauge_covariance_calibration",
    "load_point_uncertainty_calibration",
    "save_gauge_covariance_calibration",
    "save_point_uncertainty_calibration",
]
