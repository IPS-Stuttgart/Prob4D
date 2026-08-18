"""Preview version-1 calibration façade backed by :mod:`prob4d.api.v2`."""

from __future__ import annotations

from typing import Final

from .v2 import (
    CalibrationCompatibilityError,
    GaugeCovarianceCalibrationV1,
    MetricGaugeAnchor,
    PathwiseMaximumCalibrationV1,
    PathwiseUncertaintyDiagnostics,
    PointUncertaintyCalibrationV1,
    PredictionCalibrationTargetV1,
    assert_calibration_pair_compatible,
    fit_pathwise_maximum_calibration,
    load_gauge_covariance_calibration,
    load_metric_gauge_anchor,
    load_point_uncertainty_calibration,
    load_prediction_calibration_target,
    pathwise_uncertainty_diagnostics,
    save_gauge_covariance_calibration,
    save_metric_gauge_anchor,
    save_point_uncertainty_calibration,
)

FACADE_VERSION: Final = 1
LIFECYCLE: Final = "preview"

__all__ = [
    "CalibrationCompatibilityError",
    "FACADE_VERSION",
    "GaugeCovarianceCalibrationV1",
    "LIFECYCLE",
    "MetricGaugeAnchor",
    "PathwiseMaximumCalibrationV1",
    "PathwiseUncertaintyDiagnostics",
    "PointUncertaintyCalibrationV1",
    "PredictionCalibrationTargetV1",
    "assert_calibration_pair_compatible",
    "fit_pathwise_maximum_calibration",
    "load_gauge_covariance_calibration",
    "load_metric_gauge_anchor",
    "load_point_uncertainty_calibration",
    "load_prediction_calibration_target",
    "pathwise_uncertainty_diagnostics",
    "save_gauge_covariance_calibration",
    "save_metric_gauge_anchor",
    "save_point_uncertainty_calibration",
]
