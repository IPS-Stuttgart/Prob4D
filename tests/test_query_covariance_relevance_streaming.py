from __future__ import annotations

import numpy as np
import pytest

from prob4d.query_covariance_relevance import (
    project_joint_covariance_blocks_to_query,
    project_joint_covariance_to_query,
)


def _random_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(991)
    jacobian = generator.normal(size=(4, 11, 3))
    roots = generator.normal(size=(11, 3, 3))
    local = np.einsum("nij,nkj->nik", roots, roots) + 0.05 * np.eye(3)[None, ...]
    factor = generator.normal(size=(11, 3, 5))
    return jacobian, local, factor


def test_streaming_projection_matches_monolithic_and_reverse_order() -> None:
    jacobian, local, factor = _random_problem()
    expected = project_joint_covariance_to_query(jacobian, local, factor)
    blocks = (
        (jacobian[:, :2], local[:2], factor[:2]),
        (jacobian[:, 2:7], local[2:7], factor[2:7]),
        (jacobian[:, 7:], local[7:], factor[7:]),
    )

    streamed = project_joint_covariance_blocks_to_query(iter(blocks))
    reversed_streamed = project_joint_covariance_blocks_to_query(iter(reversed(blocks)))

    for actual in (streamed, reversed_streamed):
        assert actual.observation_count == 11
        np.testing.assert_allclose(
            actual.conditional_covariance,
            expected.conditional_covariance,
        )
        np.testing.assert_allclose(
            actual.shared_query_factor,
            expected.shared_query_factor,
        )
        np.testing.assert_allclose(actual.shared_covariance, expected.shared_covariance)
        np.testing.assert_allclose(actual.total_covariance, expected.total_covariance)


def test_streaming_projection_accepts_scalar_query_blocks_and_rank_zero() -> None:
    blocks = (
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.repeat(np.eye(3)[None, ...], 2, axis=0),
            np.empty((2, 3, 0)),
        ),
        (
            np.array([[0.0, 0.0, 2.0]]),
            np.diag([1.0, 2.0, 3.0])[None, ...],
            np.empty((1, 3, 0)),
        ),
    )

    result = project_joint_covariance_blocks_to_query(blocks)

    assert result.query_dimension == 1
    assert result.observation_count == 3
    assert result.shared_rank_column_count == 0
    np.testing.assert_allclose(result.conditional_covariance, [[14.0]])
    assert result.shared_trace_fraction == 0.0


def test_streaming_projection_consumes_iterable_once() -> None:
    jacobian, local, factor = _random_problem()

    class OnePass:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("block stream was iterated more than once")
            yield jacobian[:, :5], local[:5], factor[:5]
            yield jacobian[:, 5:], local[5:], factor[5:]

    blocks = OnePass()
    result = project_joint_covariance_blocks_to_query(blocks)

    assert blocks.iterations == 1
    assert result.observation_count == 11


def test_streaming_projection_rejects_empty_or_malformed_blocks() -> None:
    with pytest.raises(ValueError, match="at least one observation block"):
        project_joint_covariance_blocks_to_query(())
    with pytest.raises(TypeError, match="three-tuple"):
        project_joint_covariance_blocks_to_query(
            [(np.ones((1, 3)), np.eye(3)[None])]
        )


def test_streaming_projection_rejects_dimension_or_rank_drift() -> None:
    local = np.eye(3)[None, ...]
    with pytest.raises(ValueError, match="same query dimension"):
        project_joint_covariance_blocks_to_query(
            (
                (np.ones((1, 1, 3)), local, np.ones((1, 3, 2))),
                (np.ones((2, 1, 3)), local, np.ones((1, 3, 2))),
            )
        )
    with pytest.raises(ValueError, match="same shared-factor rank"):
        project_joint_covariance_blocks_to_query(
            (
                (np.ones((1, 1, 3)), local, np.ones((1, 3, 2))),
                (np.ones((1, 1, 3)), local, np.ones((1, 3, 3))),
            )
        )
