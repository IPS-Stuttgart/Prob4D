from __future__ import annotations

from dataclasses import replace

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


def _calibration(
    group_count: int = 20,
    *,
    maximum: float = 20.0,
) -> PathwiseMaximumCalibrationV1:
    maxima = np.repeat(
        np.linspace(maximum / group_count, maximum, group_count, dtype=np.float64),
        2,
    )[:, None]
    group_ids = tuple(
        f"calibration-object-{index:02d}"
        for index in range(group_count)
        for _ in range(2)
    )
    return fit_pathwise_maximum_calibration(
        _residuals_from_mahalanobis(maxima),
        _identity_covariance(maxima.shape),
        group_ids=group_ids,
        independent_unit="physical-object",
        miscoverage=0.05,
    )


def test_pathwise_maximum_calibration_uses_independent_group_rank() -> None:
    calibration = _calibration()

    assert isinstance(calibration, PathwiseMaximumCalibrationV1)
    assert calibration.calibration_trajectory_count == 40
    assert calibration.calibration_group_count == 20
    assert calibration.calibration_group_trajectory_counts == (2,) * 20
    assert calibration.order_statistic_rank == 20
    assert calibration.finite_sample_coverage_level == pytest.approx(20 / 21)
    assert calibration.maximum_mahalanobis_squared_threshold == pytest.approx(20.0)
    assert len(calibration.calibration_group_assignment_sha256) == 64
    assert calibration.to_dict()["calibration_group_ids"] == (
        calibration.calibration_group_ids
    )

    too_few_group_ids = tuple(
        f"calibration-object-{index:02d}"
        for index in range(18)
        for _ in range(10)
    )
    too_few = np.ones((len(too_few_group_ids), 1), dtype=np.float64)
    with pytest.raises(ValueError, match="at least 19 independent groups"):
        fit_pathwise_maximum_calibration(
            _residuals_from_mahalanobis(too_few),
            _identity_covariance(too_few.shape),
            group_ids=too_few_group_ids,
            independent_unit="physical-object",
            miscoverage=0.05,
        )


def test_calibration_artifact_binds_complete_group_assignment() -> None:
    calibration = _calibration()
    with pytest.raises(ValueError, match="digest does not match"):
        replace(
            calibration,
            calibration_group_assignment_sha256="0" * 64,
        )


def test_pathwise_diagnostics_weight_complete_groups_equally() -> None:
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
    diagnostics = pathwise_uncertainty_diagnostics(
        _residuals_from_mahalanobis(mahalanobis),
        _identity_covariance(mahalanobis.shape),
        group_ids=("target-object-a", "target-object-a", "target-object-b"),
        independent_unit="physical-object",
        valid_mask=valid,
        calibration=_calibration(maximum=10.0),
    )

    assert diagnostics.group_count == 2
    assert diagnostics.trajectory_count == 3
    assert diagnostics.evaluated_step_count == 13
    assert diagnostics.equal_group_marginal_coverage_95 == pytest.approx(
        (0.8 + 2 / 3) / 2
    )
    assert diagnostics.all_groups_inside_marginal_95_fraction == 0.0
    assert diagnostics.maximum_trajectory_longest_marginal_95_failure_run == 2
    assert diagnostics.mean_trajectory_longest_marginal_95_failure_run == pytest.approx(
        1.0
    )
    assert diagnostics.maximum_trajectory_longest_unsupported_run == 2
    assert diagnostics.simultaneous_nominal_group_coverage == pytest.approx(0.95)
    assert diagnostics.simultaneous_group_coverage == pytest.approx(0.5)
    assert diagnostics.simultaneous_group_coverage_shortfall == pytest.approx(0.45)
    assert diagnostics.to_dict()["target_group_trajectory_counts"] == (2, 1)


def test_all_group_marginal_fraction_is_not_simultaneous_coverage() -> None:
    mahalanobis = np.ones((2, 3), dtype=np.float64)
    diagnostics = pathwise_uncertainty_diagnostics(
        _residuals_from_mahalanobis(mahalanobis),
        _identity_covariance(mahalanobis.shape),
        group_ids=("target-session-a", "target-session-b"),
        independent_unit="acquisition-session",
    )

    assert diagnostics.all_groups_inside_marginal_95_fraction == 1.0
    assert diagnostics.simultaneous_nominal_group_coverage is None
    assert diagnostics.simultaneous_group_maximum_threshold is None
    assert diagnostics.simultaneous_group_coverage is None
    assert diagnostics.simultaneous_group_coverage_shortfall is None


def test_target_groups_must_be_disjoint_from_calibration_groups() -> None:
    calibration = _calibration()
    mahalanobis = np.ones((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="must be disjoint"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            group_ids=(calibration.calibration_group_ids[0],),
            independent_unit="physical-object",
            calibration=calibration,
        )


def test_pathwise_inputs_reject_invalid_grouping_and_calibration_types() -> None:
    mahalanobis = np.ones((2, 2), dtype=np.float64)
    valid = np.asarray([[True, True], [False, False]])
    with pytest.raises(ValueError, match="at least one valid step"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            group_ids=("object-a", "object-b"),
            independent_unit="physical-object",
            valid_mask=valid,
        )
    with pytest.raises(ValueError, match="one entry per"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            group_ids=("object-a",),
            independent_unit="physical-object",
        )
    with pytest.raises(ValueError, match="independent_unit"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            group_ids=("track-a", "track-b"),
            independent_unit="trajectory",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="PathwiseMaximumCalibrationV1"):
        pathwise_uncertainty_diagnostics(
            _residuals_from_mahalanobis(mahalanobis),
            _identity_covariance(mahalanobis.shape),
            group_ids=("object-a", "object-b"),
            independent_unit="physical-object",
            calibration=object(),  # type: ignore[arg-type]
        )
