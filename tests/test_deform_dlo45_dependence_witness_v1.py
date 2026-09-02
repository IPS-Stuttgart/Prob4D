from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest


MODULE = runpy.run_path(
    "benchmarks/information_contract_v1/adapters/"
    "deform_dlo45_dependence_witness_v1.py"
)
summary_vector = MODULE["summary_vector"]
ordered = MODULE["_ordered"]
equal_group_second_moment = MODULE["_equal_group_second_moment"]


def test_summary_vector_has_registered_linear_semantics() -> None:
    values = np.zeros((2, 8, 3), dtype=np.float64)
    values[1, 4:, :] = np.array([2.0, 4.0, 6.0])

    result = summary_vector(values.reshape(-1, 3), horizon_frames=2)

    np.testing.assert_allclose(result[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result[3:6], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(result[6:9], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(result[9:12], [1.0, 2.0, 3.0])


def test_constant_residual_has_no_bending_or_temporal_change() -> None:
    rows = np.repeat(np.array([[1.0, -2.0, 3.0]]), 24, axis=0)

    result = summary_vector(rows, horizon_frames=3)

    np.testing.assert_allclose(result[:3], [1.0, -2.0, 3.0])
    np.testing.assert_allclose(result[3:6], [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(result[6:], np.zeros(6))


@pytest.mark.parametrize(
    "values,horizon",
    [
        (np.zeros((15, 3)), 2),
        (np.zeros((16, 2)), 2),
        (np.full((16, 3), np.nan), 2),
    ],
)
def test_summary_vector_rejects_invalid_forecasts(
    values: np.ndarray, horizon: int
) -> None:
    with pytest.raises(ValueError, match="finite 8-node forecast"):
        summary_vector(values, horizon_frames=horizon)


def test_source_partition_order_is_input_order_independent() -> None:
    paths = tuple(Path(f"trajectory-{index:02d}.pkl") for index in range(12))

    forward = ordered(paths, "DLO4")
    reverse = ordered(tuple(reversed(paths)), "DLO4")

    assert forward == reverse
    assert set(forward) == set(paths)
    assert ordered(paths, "DLO5") != forward


def test_equal_group_second_moment_weights_complete_groups_equally() -> None:
    values = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ]
    )
    groups = np.array([0, 0, 1, 1], dtype=np.int64)

    result = equal_group_second_moment(values, groups, 2)

    np.testing.assert_allclose(result[0, 0], 0.5, atol=2e-8)
    np.testing.assert_allclose(result[1, 1], 2.0, atol=2e-8)
    np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(result, result.T)
    np.linalg.cholesky(result)


def test_equal_group_second_moment_rejects_missing_group() -> None:
    with pytest.raises(ValueError, match="empty source group"):
        equal_group_second_moment(
            np.ones((2, 2)), np.zeros(2, dtype=np.int64), 2
        )
