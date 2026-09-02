from __future__ import annotations

import numpy as np
import pytest

from prob4d.group_risk_control import (
    harmful_accepted_loss,
    select_group_conformal_risk_control,
)


def test_selects_least_conservative_corrected_feasible_candidate() -> None:
    losses = np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.8, 0.1, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
        ]
    )
    result = select_group_conformal_risk_control(
        losses,
        [0.0, 0.1, 0.2, 0.3],
        target_risk=0.3,
    )
    assert result.feasible
    assert result.selected_index == 2
    assert result.selected_parameter == 0.2
    assert result.empirical_risk == 0.0
    assert result.corrected_risk == 0.2
    assert result.finite_sample_floor == 0.2


def test_fails_closed_when_finite_sample_floor_exceeds_target() -> None:
    result = select_group_conformal_risk_control(
        np.zeros((4, 2)),
        [0.0, 1.0],
        target_risk=0.1,
    )
    assert not result.feasible
    assert result.selected_index is None
    assert result.finite_sample_floor == 0.2


def test_non_nested_loss_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonincreasing"):
        select_group_conformal_risk_control(
            [[0.0, 1.0], [1.0, 0.0]],
            [0.0, 1.0],
            target_risk=0.5,
        )


def test_group_permutation_does_not_change_selection() -> None:
    rng = np.random.default_rng(42)
    losses = np.sort(rng.uniform(size=(20, 7)), axis=1)[:, ::-1]
    parameters = np.arange(7, dtype=np.float64)
    first = select_group_conformal_risk_control(
        losses,
        parameters,
        target_risk=0.4,
    )
    second = select_group_conformal_risk_control(
        losses[rng.permutation(20)],
        parameters,
        target_risk=0.4,
    )
    assert first.selected_index == second.selected_index
    np.testing.assert_array_equal(first.empirical_risk_curve, second.empirical_risk_curve)


def test_harmful_accepted_binary_loss() -> None:
    result = harmful_accepted_loss(
        np.array([True, True, False, False]),
        np.array([2.0, 1.0, 8.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(result, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="boolean"):
        harmful_accepted_loss([1, 0], [1.0, 1.0], [0.0, 0.0])


def test_invalid_bounds_and_nonfinite_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="\[0, loss_bound\]"):
        select_group_conformal_risk_control(
            [[0.0, 1.1]],
            [0.0, 1.0],
            target_risk=0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        select_group_conformal_risk_control(
            [[0.0, np.nan]],
            [0.0, 1.0],
            target_risk=0.5,
        )
