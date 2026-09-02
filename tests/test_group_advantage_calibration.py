from __future__ import annotations

import numpy as np
import pytest

from prob4d.group_advantage_calibration import (
    SignedGroupConformalUpperBound,
    calibrate_signed_group_upper_bound,
    group_max_advantage_deficit,
)


def test_signed_calibration_can_remove_systematic_conservatism() -> None:
    result = calibrate_signed_group_upper_bound(
        [-5.0, -4.0, -3.0, -2.0],
        miscoverage=0.4,
    )
    assert result.finite
    assert result.threshold == -3.0
    assert result.lower_bound(1.0) == 4.0
    assert result.admits(1.0)


def test_finite_sample_failure_is_explicit() -> None:
    result = calibrate_signed_group_upper_bound(
        [1.0, 2.0],
        miscoverage=0.1,
    )
    assert not result.finite
    assert result.threshold is None
    with pytest.raises(ValueError, match="unavailable"):
        result.lower_bound(1.0)


def test_group_score_implies_simultaneous_case_lower_bounds() -> None:
    base = np.array([3.0, -2.0, 9.0, 0.5])
    realized = np.array([2.5, -1.0, 7.0, 0.4])
    score = group_max_advantage_deficit(base, realized)
    assert score == 2.0
    np.testing.assert_array_less(base - score - 1e-15, realized)

    calibration = SignedGroupConformalUpperBound(
        threshold=score,
        miscoverage=0.2,
        calibration_group_count=5,
        order_statistic=5,
        finite=True,
    )
    for predicted, actual in zip(base, realized, strict=True):
        assert actual >= calibration.lower_bound(predicted) - 1e-15
        if calibration.admits(predicted):
            assert actual > 0.0


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [[1.0]],
        [1.0, np.nan],
        [1.0, np.inf],
    ],
)
def test_invalid_calibration_scores_are_rejected(scores: object) -> None:
    with pytest.raises(ValueError):
        calibrate_signed_group_upper_bound(
            scores,
            miscoverage=0.2,
        )


def test_group_score_validates_shapes_and_does_not_mutate() -> None:
    base = np.array([-1.0, 2.0])
    realized = np.array([-2.0, 4.0])
    before = (base.copy(), realized.copy())
    assert group_max_advantage_deficit(base, realized) == 1.0
    np.testing.assert_array_equal(base, before[0])
    np.testing.assert_array_equal(realized, before[1])
    with pytest.raises(ValueError, match="match"):
        group_max_advantage_deficit(base, [1.0])
