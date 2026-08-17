from __future__ import annotations

import numpy as np
import pytest

from prob4d.api.v2 import (
    ProjectedObservationCovariance,
    observation_covariance_action,
    observation_covariance_quadratic,
    project_observation_covariance,
)
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.observation_factors import StackedObservationFactors
from prob4d.sparse_observation_factors import SparseStackedObservationFactors
from prob4d.tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    bind_gauge_tree_prior,
)


def _stacks() -> tuple[
    StackedObservationFactors,
    SparseStackedObservationFactors,
    TreeSparseStackedObservationFactors,
    np.ndarray,
]:
    conditional = np.asarray(
        [
            np.diag([0.2, 0.3, 0.4]),
            np.diag([0.5, 0.6, 0.7]),
        ],
        dtype=np.float64,
    )
    local_jacobian = np.zeros((2, 3, 7), dtype=np.float64)
    local_jacobian[0, :, :3] = np.eye(3)
    local_jacobian[1, :, :3] = 2.0 * np.eye(3)
    gauge_indices = np.asarray([0, 1], dtype=np.int64)
    block_0 = np.eye(7, dtype=np.float64) * 0.10
    block_1 = np.eye(7, dtype=np.float64) * 0.20
    cross = np.eye(7, dtype=np.float64) * 0.04
    gauge_prior = np.block([[block_0, cross], [cross, block_1]])
    dense_jacobian = np.zeros((2, 3, 14), dtype=np.float64)
    dense_jacobian[0, :, :7] = local_jacobian[0]
    dense_jacobian[1, :, 7:] = local_jacobian[1]
    gauge_row_marginal = np.asarray(
        [
            local_jacobian[0] @ block_0 @ local_jacobian[0].T,
            local_jacobian[1] @ block_1 @ local_jacobian[1].T,
        ]
    )
    marginal = conditional + gauge_row_marginal
    metadata = {
        "association_probability": np.ones(2),
        "prior_reliability": np.ones(2),
        "prior_nominal_probability": np.ones(2),
        "composite_weight": np.ones(2),
        "point_ids": np.asarray([10, 20], dtype=np.int64),
        "frame_indices": np.asarray([0, 1], dtype=np.int64),
        "view_ids": ("camera", "camera"),
        "factor_ids": ("factor-0", "factor-1"),
        "correlation_group_ids": ("group-0", "group-1"),
        "gauge_ids": ("gauge-0", "gauge-1"),
        "causal_frame_stop": 2,
    }
    dense = StackedObservationFactors(
        world_mean_m=np.zeros((2, 3), dtype=np.float64),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        gauge_jacobian=dense_jacobian,
        gauge_prior_covariance=gauge_prior,
        **metadata,
    )
    sparse = SparseStackedObservationFactors(
        world_mean_m=np.zeros((2, 3), dtype=np.float64),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        local_gauge_jacobian=local_jacobian,
        gauge_indices=gauge_indices,
        gauge_prior_covariance=gauge_prior,
        **metadata,
    )
    tree_prior = GaugeTreeSquareRootPriorV1.from_dense_covariance(
        gauge_ids=metadata["gauge_ids"],
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        joint_covariance=gauge_prior,
    )
    tree = bind_gauge_tree_prior(sparse, tree_prior)
    full_covariance = np.zeros((6, 6), dtype=np.float64)
    full_covariance[:3, :3] = conditional[0]
    full_covariance[3:, 3:] = conditional[1]
    full_covariance += dense_jacobian.reshape(6, 14) @ gauge_prior @ dense_jacobian.reshape(6, 14).T
    return dense, sparse, tree, full_covariance


@pytest.mark.parametrize("stack_index", [0, 1, 2])
def test_structured_actions_and_projection_match_dense_covariance(stack_index: int) -> None:
    dense, sparse, tree, full_covariance = _stacks()
    stacked = (dense, sparse, tree)[stack_index]
    vector = np.linspace(-0.3, 0.4, 6, dtype=np.float64).reshape(2, 3)
    query = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        observation_covariance_action(stacked, vector).reshape(-1),
        full_covariance @ vector.reshape(-1),
    )
    np.testing.assert_allclose(
        observation_covariance_quadratic(stacked, vector),
        vector.reshape(-1) @ full_covariance @ vector.reshape(-1),
    )
    projected = project_observation_covariance(stacked, query)
    assert isinstance(projected, ProjectedObservationCovariance)
    flat_query = query.reshape(2, 6)
    np.testing.assert_allclose(
        projected.marginal_covariance,
        flat_query @ full_covariance @ flat_query.T,
    )
    np.testing.assert_allclose(
        projected.marginal_covariance,
        projected.conditional_covariance + projected.gauge_covariance,
    )


def test_projection_preserves_cross_row_shared_gauge_covariance() -> None:
    _, sparse, _, full_covariance = _stacks()
    query = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )

    projected = project_observation_covariance(sparse, query)
    assert projected.gauge_covariance[0, 1] != 0.0
    np.testing.assert_allclose(projected.marginal_covariance[0, 1], full_covariance[0, 3])


def test_scalar_projection_and_input_validation() -> None:
    _, sparse, _, _ = _stacks()
    scalar = project_observation_covariance(
        sparse,
        np.ones((2, 3), dtype=np.float64),
    )
    assert scalar.query_dimension == 1
    assert scalar.scalar_variance == pytest.approx(scalar.marginal_covariance[0, 0])

    with pytest.raises(ValueError, match="component"):
        observation_covariance_action(
            sparse,
            np.ones((2, 3)),
            component="wrong",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="query_jacobian"):
        project_observation_covariance(sparse, np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        query = np.ones((1, 2, 3))
        query[0, 0, 0] = np.nan
        project_observation_covariance(sparse, query)
    with pytest.raises(TypeError, match="stacked must be"):
        project_observation_covariance(object(), np.ones((2, 3)))  # type: ignore[arg-type]
