from __future__ import annotations

import numpy as np
import pytest

from prob4d.api.v2 import (
    PathwiseMaximumCalibrationV1,
    fit_pathwise_maximum_calibration,
    pathwise_uncertainty_diagnostics,
)


def _residuals_from_mahalanobis(values: np.ndarray) -> np.ndarray:
    result = np.zeros((*values.shape, 3), dtype=np.float64)
    result[..., 0] = np.sqrt(values)
    return result


def _identity_covariance(shape: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(np.eye(3, dtype=np.float64), (*shape, 3, 3)).copy()


def test_pathwise_maximum_calibration_uses_finite_sample_rank() -> None:
    maxima = np.arange(1.0, 21.0, dtype=np.float64)[:, None]
    calibration = fit_pathwise_maximum_calibration(
        _residuals_from_mahalanobis(maxima),
        _identity_covariance(maxima.shape),
        miscoverage=0.05,
    )

    assert isinstance(calibration, PathwiseMaximumCalibrationV1)
    assert calibration.calibration_trajectory_count == 20
    assert calibration.order_statistic_rank == 20
    assert calibration.finite_sample_coverage_level == pytest.approx(20 / 21)
    assert calibration.maximum_mahalanobis_squared_threshold == pytest.approx(20.0)

    too_few = np.arange(1.0, 19.0, dtype=np.float64)[:, None]
    with pytest.raises(ValueError, match="at least 19"):
        fit_pathwise_maximum_calibration(
            _residuals_from_mahalanobis(too_few),
            _identity_covariance(too_few.shape),
            miscoverage=0.05,
        )


def test_pathwise_diagnostics_expose_clustered_failures_and_missingness() -> None:
    mahalanobis = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 9.0, 10.0, 2.0, 1.0],
            [1.0, 2.0, 0.0, 0.0, 12.0],
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, False, False, True],
        ]
    )
    calibration = PathwiseMaximumCalibrationV1(
        requested_miscoverage=0.05,
        calibration_trajectory_count=20,
        order_statistic_rank=20,
        finite_sample_coverage_level=20 / 21,
        maximum_mahalanobis_squared_threshold=10.0,
    )
    diagnostics = pathwise_uncertainty_diagnostics(
        _residuals_from_mahalanobis(mahalanobis),
        _identity_covariance(mahalanobis.shape),
        valid_mask=valid,
        calibration=calibration,
    )

    assert diagnostics.trajectory_count == 3
    assert diagnostics.evaluated_step_count == 13
    assert diagnostics.marginal_coverage_95 == pytest.approx(10 / 13)
    assert diagnostics.all_steps_inside_marginal_95_fraction == pytest.approx(1 / 3)
    assert diagnostics.maximum_longest_marginal_95_failure_run == 2
    assert diagnostics.mean_longest_marginal_95_failure_run == pytest.approx(1.0)
    assert diagnostics.maximum_longest_unsupported_run == 2
    assert diagnostics.simultaneous_nominal_coverage == pytest.approx(0.95)
    assert diagnostics.simultaneous_coverage == pytest.approx(2 / 3)
    assert diagnostics.simultaneous_coverage_shortfall == pytest.approx(0.95 - 2 / 3)
    assert diagnostics.to_dict()["p95_max_mahalanobis_squared"] == pytest.approx(
        diagnostics.p95_max_mahalanobis_squared
    )


def test_all_step_marginal_fraction_is_not_reported_as_simultaneous_coverage() -> None:
    mahalanobis = np.ones((2, 3), dtype=np.float64)
    diagnostics = pathwise_uncertainty_diagnostics(
        _residuals_from_mahalanobis(mahalanobis),
        _identity_covariance(mahalanobis.shape),
    )

    assert diagnostics.all_steps_inside_marginal_95_fraction == 1.0
    assert diagnostics.simultaneous_nominal_coverage is None
    assert diagnostics.simultaneous_maximum_threshold is None
    assert diagnostics.simultaneous_coverage is None
    assert diagnostics.simultaneous_coverage_shortfall is None


def test_pathwise_inputs_reject_empty_trajectories_and_target_calibration_types() -> None:
    mahalanobis = np.ones((2, 2), dtype=np.float64)
    valid = np.asarray([[True, True], [False, False]])
    with pytest.raises(ValueError, match="at least one valid step"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            valid_mask=valid,
        )
    with pytest.raises(TypeError, match="PathwiseMaximumCalibrationV1"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            calibration=object(),  # type: ignore[arg-type]
        )
