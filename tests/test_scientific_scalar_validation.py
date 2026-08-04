from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from prob4d.alignment import AlignmentResult, estimate_sim3_robust
from prob4d.cross_fitted_disagreement import CrossFittedDisagreementReport
from prob4d.sim3 import Sim3
from prob4d.uncertainty import (
    CalibrationReport,
    GroupBalancedCalibrationReport,
)

_INTEGER_ALIASES: tuple[object, ...] = (
    True,
    np.bool_(True),
    1.0,
    np.float64(1.0),
    "1",
)


def _alignment_result(**overrides: Any) -> AlignmentResult:
    values: dict[str, Any] = {
        "transform": Sim3.identity(),
        "covariance": np.eye(7),
        "residual_rms": 0.0,
        "inlier_fraction": 1.0,
        "num_correspondences": 8,
        "num_covariance_clusters": 8,
        "information_rank": 7,
        "information_condition": 1.0,
    }
    values.update(overrides)
    return AlignmentResult(**values)


def _cross_fitted_report(**overrides: Any) -> CrossFittedDisagreementReport:
    values: dict[str, Any] = {
        "alignment_count": 1,
        "requested_folds": 2,
        "candidate_folds": 2,
        "fitted_folds": 1,
        "skipped_folds": 1,
        "skipped_alignments": 0,
        "overlap_points": 10,
        "evaluated_points": 5,
        "cluster_size": 4,
        "maximum_training_correspondences": 100,
        "seed": 0,
    }
    values.update(overrides)
    return CrossFittedDisagreementReport(**values)


def _calibration_report(**overrides: Any) -> CalibrationReport:
    values: dict[str, Any] = {
        "count": 1,
        "parallel_scale_update": 1.0,
        "lateral_scale_update": 1.0,
        "parallel_normalized_mse": 1.0,
        "lateral_normalized_mse": 1.0,
    }
    values.update(overrides)
    return CalibrationReport(**values)


def _group_report(**overrides: Any) -> GroupBalancedCalibrationReport:
    values: dict[str, Any] = {
        "count": 2,
        "trim_quantile": 0.99,
        "parallel_scale_update": 1.0,
        "lateral_scale_update": 1.0,
        "parallel_normalized_mse": 1.0,
        "lateral_normalized_mse": 1.0,
        "group_ids": ("a", "b"),
        "group_counts": (1, 1),
        "group_parallel_scale_updates": (1.0, 1.0),
        "group_lateral_scale_updates": (1.0, 1.0),
        "group_parallel_normalized_mse": (1.0, 1.0),
        "group_lateral_normalized_mse": (1.0, 1.0),
    }
    values.update(overrides)
    return GroupBalancedCalibrationReport(**values)


def _nondegenerate_correspondences() -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    target = 1.2 * source + np.asarray([0.3, -0.2, 0.4])
    return source, target


@pytest.mark.parametrize("value", _INTEGER_ALIASES)
def test_alignment_result_rejects_coercible_integer_counts(value: object) -> None:
    with pytest.raises(TypeError, match="genuine integer"):
        _alignment_result(num_correspondences=value)


@pytest.mark.parametrize("value", _INTEGER_ALIASES)
def test_cross_fitted_report_rejects_coercible_integer_counts(value: object) -> None:
    with pytest.raises(TypeError, match="genuine integer"):
        _cross_fitted_report(alignment_count=value)


@pytest.mark.parametrize("value", _INTEGER_ALIASES)
def test_calibration_report_rejects_coercible_counts(value: object) -> None:
    with pytest.raises(TypeError, match="genuine integer"):
        _calibration_report(count=value)


def test_group_report_rejects_fractional_group_count_without_truncation() -> None:
    with pytest.raises(TypeError, match=r"group_counts\[0\].*genuine integer"):
        _group_report(group_counts=(1.5, 1))


@pytest.mark.parametrize(
    "group_ids",
    (
        (1, "b"),
        ["a", "b"],
        {"a", "b"},
        "ab",
    ),
)
def test_group_report_rejects_noncanonical_group_ids(
    group_ids: object,
) -> None:
    with pytest.raises(TypeError, match="canonical tuple of strings"):
        _group_report(group_ids=group_ids)


@pytest.mark.parametrize("value", (True, np.bool_(True), "1.0"))
def test_calibration_report_rejects_coercible_real_values(value: object) -> None:
    with pytest.raises(TypeError, match="genuine real scalar"):
        _calibration_report(parallel_scale_update=value)


def test_group_report_rejects_coercible_per_group_values() -> None:
    with pytest.raises(
        TypeError,
        match=r"group_parallel_scale_updates\[0\].*genuine real scalar",
    ):
        _group_report(group_parallel_scale_updates=(True, 1.0))


@pytest.mark.parametrize("max_iterations", (0, -1))
def test_alignment_rejects_nonpositive_iteration_budget(max_iterations: int) -> None:
    source, target = _nondegenerate_correspondences()
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        estimate_sim3_robust(source, target, max_iterations=max_iterations)


@pytest.mark.parametrize("value", _INTEGER_ALIASES)
def test_alignment_rejects_coercible_iteration_budget(value: object) -> None:
    source, target = _nondegenerate_correspondences()
    with pytest.raises(TypeError, match="max_iterations must be a genuine integer"):
        estimate_sim3_robust(source, target, max_iterations=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("huber_multiplier", 0.0, "huber_multiplier must be greater than 0.0"),
        ("huber_multiplier", np.inf, "huber_multiplier must be finite"),
        ("tolerance", 0.0, "tolerance must be greater than 0.0"),
        ("tolerance", np.nan, "tolerance must be finite"),
    ),
)
def test_alignment_rejects_invalid_robust_controls(
    keyword: str,
    value: float,
    message: str,
) -> None:
    source, target = _nondegenerate_correspondences()
    with pytest.raises(ValueError, match=message):
        estimate_sim3_robust(source, target, **{keyword: value})


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("huber_multiplier", True),
        ("huber_multiplier", "2.5"),
        ("tolerance", np.bool_(True)),
        ("tolerance", "1e-8"),
    ),
)
def test_alignment_rejects_coercible_robust_controls(
    keyword: str,
    value: object,
) -> None:
    source, target = _nondegenerate_correspondences()
    with pytest.raises(TypeError, match="genuine real scalar"):
        estimate_sim3_robust(source, target, **{keyword: value})  # type: ignore[arg-type]


def test_numpy_integer_and_float_scalars_remain_supported() -> None:
    source, target = _nondegenerate_correspondences()
    result = estimate_sim3_robust(
        source,
        target,
        max_iterations=np.int64(1),
        huber_multiplier=np.float32(2.5),
        tolerance=np.float64(1e-8),
    )
    report = _cross_fitted_report(
        alignment_count=np.int64(1),
        requested_folds=np.int32(2),
    )
    calibration = _calibration_report(
        count=np.int64(1),
        parallel_scale_update=np.float32(1.0),
    )
    grouped = _group_report(
        count=np.int64(2),
        group_counts=(np.int32(1), np.int64(1)),
    )

    assert result.num_correspondences == len(source)
    assert report.alignment_count == 1
    assert calibration.count == 1
    assert grouped.group_counts == (1, 1)
