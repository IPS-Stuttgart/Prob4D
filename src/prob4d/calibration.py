"""Content-addressed covariance calibration contracts for Prob4D providers."""

from ._calibration_common import (
    GAUGE_COVARIANCE_CALIBRATION_SCHEMA,
    GAUGE_COVARIANCE_CALIBRATION_VERSION,
    POINT_UNCERTAINTY_CALIBRATION_SCHEMA,
    POINT_UNCERTAINTY_CALIBRATION_VERSION,
)
from ._strict_calibration import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    load_gauge_covariance_calibration,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_point_uncertainty_calibration,
)
from .calibration_aggregation import (
    GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
    LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1,
    UPPER_WINSORIZED_MEAN_V1,
    upper_winsorized_mean,
)
from .group_balanced_point_calibration import (
    fit_group_balanced_point_uncertainty_calibration,
    group_balanced_point_calibration_metadata,
)

__all__ = [
    "GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2",
    "LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1",
    "UPPER_WINSORIZED_MEAN_V1",
    "GAUGE_COVARIANCE_CALIBRATION_SCHEMA",
    "GAUGE_COVARIANCE_CALIBRATION_VERSION",
    "POINT_UNCERTAINTY_CALIBRATION_SCHEMA",
    "POINT_UNCERTAINTY_CALIBRATION_VERSION",
    "GaugeCovarianceCalibrationV1",
    "PointUncertaintyCalibrationV1",
    "fit_group_balanced_point_uncertainty_calibration",
    "group_balanced_point_calibration_metadata",
    "load_gauge_covariance_calibration",
    "load_point_uncertainty_calibration",
    "save_gauge_covariance_calibration",
    "save_point_uncertainty_calibration",
    "upper_winsorized_mean",
]
