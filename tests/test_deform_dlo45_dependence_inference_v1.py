from __future__ import annotations

import runpy

import numpy as np
import pytest

MODULE = runpy.run_path(
    "benchmarks/information_contract_v1/adapters/"
    "deform_dlo45_dependence_inference_v1.py"
)
paired_group_inference = MODULE["paired_group_inference"]


def test_paired_group_inference_is_deterministic_and_paired() -> None:
    full_nll = np.array([0.0, 0.0, 0.0, 0.0])
    diagonal_nll = np.array([1.0, 2.0, 3.0, 4.0])
    full_nees = np.ones(4)
    diagonal_nees = np.full(4, 4.0)

    first = paired_group_inference(
        full_nll,
        diagonal_nll,
        full_nees,
        diagonal_nees,
        replicates=2_000,
        seed=7,
    )
    second = paired_group_inference(
        full_nll,
        diagonal_nll,
        full_nees,
        diagonal_nees,
        replicates=2_000,
        seed=7,
    )

    assert first == second
    assert first["paired_nll_gain_diagonal_minus_full"] == 2.5
    assert first["full_dependence_nll_win_count"] == 4
    assert first["diagonal_nll_win_count"] == 0
    assert first["nll_tie_count"] == 0
    assert first["paired_sign_test_two_sided_p"] == 0.125
    assert first["paired_nll_gain_ci95"][0] > 0.0
    assert first["paired_calibration_error_gain_ci95"][0] > 0.0


def test_paired_group_inference_retains_ties() -> None:
    result = paired_group_inference(
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
        replicates=20,
        seed=0,
    )

    assert result["full_dependence_nll_win_count"] == 0
    assert result["diagonal_nll_win_count"] == 0
    assert result["nll_tie_count"] == 2
    assert result["paired_sign_test_two_sided_p"] == 1.0


@pytest.mark.parametrize(
    "arguments,match",
    [
        (([1.0], [1.0], [1.0], [1.0]), "at least two groups"),
        (([1.0, 2.0], [1.0], [1.0, 1.0], [1.0, 1.0]), "identical length"),
        (([1.0, 2.0], [1.0, 2.0], [0.0, 1.0], [1.0, 1.0]), "must be positive"),
    ],
)
def test_paired_group_inference_rejects_invalid_inputs(
    arguments: tuple[object, object, object, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        paired_group_inference(*arguments, replicates=10, seed=1)
