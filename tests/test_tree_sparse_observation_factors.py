from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.observation_factors import ObservationFactor, ObservationFactorBundle
from prob4d.sim3 import Sim3
from prob4d.sparse_observation_factors import stack_sparse_observation_factors
from prob4d.tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    bind_gauge_tree_prior,
    stack_tree_sparse_observation_factors,
)


def _bundle(*, with_invalid_row: bool = False) -> ObservationFactorBundle:
    covariance_0 = np.eye(7, dtype=np.float64) * 2.0e-4
    covariance_1 = np.eye(7, dtype=np.float64) * 3.0e-4
    cross = np.eye(7, dtype=np.float64) * 5.0e-5
    joint = np.block(
        [
            [covariance_0, cross],
            [cross, covariance_1],
        ]
    )
    gauges = (
        GaugeEstimate("window-0", Sim3.identity(), covariance_0),
        GaugeEstimate("window-1", Sim3.identity(), covariance_1),
    )
    local_covariance = np.repeat(
        np.eye(3, dtype=np.float64)[None] * 1.0e-3,
        2,
        axis=0,
    )
    common = {
        "valid_mask": np.asarray([True, not with_invalid_row]),
        "local_covariance_m2": local_covariance,
        "association_probability": np.asarray([0.9, 0.8]),
        "prior_reliability": np.asarray([0.85, 0.75]),
        "prior_nominal_probability": 0.95,
        "composite_weight": 0.5,
        "causal_frame_stop": 6,
    }
    factors = (
        ObservationFactor(
            factor_id="factor-0",
            frame_index=2,
            view_id="camera-0",
            window_id="window-0",
            gauge_id="window-0",
            point_ids=np.asarray([10, 11], dtype=np.int64),
            points_local_m=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.2, 0.0, 1.1],
                ],
                dtype=np.float64,
            ),
            correlation_group_id="camera-0:frame-2",
            **common,
        ),
        ObservationFactor(
            factor_id="factor-1",
            frame_index=4,
            view_id="camera-0",
            window_id="window-1",
            gauge_id="window-1",
            point_ids=np.asarray([20, 21], dtype=np.int64),
            points_local_m=np.asarray(
                [
                    [0.1, 0.2, 1.2],
                    [0.3, 0.1, 1.3],
                ],
                dtype=np.float64,
            ),
            correlation_group_id="camera-0:frame-4",
            **common,
        ),
    )
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:explicit-gauge:camera-0",
        factors=factors,
        gauges=gauges,
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        causal_frame_stop=6,
        joint_gauge_covariance=joint,
        gauge_covariance_semantics="joint-cross-window",
    )


def _prior(bundle: ObservationFactorBundle) -> GaugeTreeSquareRootPriorV1:
    return GaugeTreeSquareRootPriorV1.from_dense_covariance(
        gauge_ids=tuple(gauge.window_id for gauge in bundle.gauges),
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        joint_covariance=bundle.joint_gauge_covariance,
    )


def test_tree_sparse_stack_matches_dense_prior_and_row_contracts() -> None:
    bundle = _bundle()
    dense_sparse = stack_sparse_observation_factors(bundle)
    tree_sparse = stack_tree_sparse_observation_factors(bundle, _prior(bundle))

    assert isinstance(tree_sparse, TreeSparseStackedObservationFactors)
    np.testing.assert_allclose(tree_sparse.world_mean_m, dense_sparse.world_mean_m)
    np.testing.assert_allclose(
        tree_sparse.conditional_world_covariance_m2,
        dense_sparse.conditional_world_covariance_m2,
    )
    np.testing.assert_allclose(
        tree_sparse.marginal_world_covariance_m2,
        dense_sparse.marginal_world_covariance_m2,
    )
    np.testing.assert_allclose(
        tree_sparse.dense_gauge_jacobian(),
        dense_sparse.dense_gauge_jacobian(),
    )
    np.testing.assert_allclose(
        tree_sparse.materialize_dense_gauge_prior(maximum_gauges=2),
        dense_sparse.gauge_prior_covariance,
    )
    np.testing.assert_allclose(
        tree_sparse.gauge_marginal_covariance_m2(),
        dense_sparse.gauge_marginal_covariance_m2(),
    )
    assert tree_sparse.gauge_ids == dense_sparse.gauge_ids
    assert (
        tree_sparse.gauge_prior_storage_nbytes == tree_sparse.gauge_tree_prior.factor_storage_nbytes
    )
    assert (
        tree_sparse.dense_gauge_prior_nbytes == tree_sparse.gauge_tree_prior.dense_covariance_nbytes
    )


def test_tree_sparse_matrix_free_actions_match_dense_linear_algebra() -> None:
    bundle = _bundle()
    dense_sparse = stack_sparse_observation_factors(bundle)
    tree_sparse = stack_tree_sparse_observation_factors(bundle, _prior(bundle))
    prior = dense_sparse.gauge_prior_covariance
    gauge_direction = np.linspace(-0.02, 0.03, 14, dtype=np.float64)

    np.testing.assert_allclose(
        tree_sparse.gauge_covariance_action(gauge_direction),
        prior @ gauge_direction,
    )
    np.testing.assert_allclose(
        tree_sparse.gauge_information_action(gauge_direction),
        np.linalg.solve(prior, gauge_direction),
    )

    row_direction = np.linspace(
        -0.03,
        0.04,
        tree_sparse.observation_count * 3,
        dtype=np.float64,
    ).reshape(tree_sparse.observation_count, 3)
    dense_design = dense_sparse.dense_gauge_jacobian()
    gauge_rhs = np.einsum(
        "mij,mi->j",
        dense_design,
        row_direction,
        optimize=True,
    )
    gauge_expected = np.einsum(
        "mij,j->mi",
        dense_design,
        prior @ gauge_rhs,
        optimize=True,
    )
    local_expected = np.einsum(
        "mij,mj->mi",
        tree_sparse.conditional_world_covariance_m2,
        row_direction,
        optimize=True,
    )
    np.testing.assert_allclose(
        tree_sparse.observation_gauge_covariance_action(row_direction),
        gauge_expected,
    )
    np.testing.assert_allclose(
        tree_sparse.marginal_observation_covariance_action(row_direction),
        gauge_expected + local_expected,
    )


def test_binding_reuses_immutable_rows_and_releases_dense_prior() -> None:
    bundle = _bundle()
    dense_sparse = stack_sparse_observation_factors(bundle)
    tree_sparse = bind_gauge_tree_prior(dense_sparse, _prior(bundle))

    assert tree_sparse.world_mean_m is dense_sparse.world_mean_m
    assert tree_sparse.local_gauge_jacobian is dense_sparse.local_gauge_jacobian
    assert tree_sparse.gauge_indices is dense_sparse.gauge_indices
    assert not hasattr(tree_sparse, "gauge_prior_covariance")
    with pytest.raises(ValueError):
        tree_sparse.local_gauge_jacobian[0, 0, 0] = 1.0


def test_tree_sparse_stack_preserves_include_invalid_semantics() -> None:
    bundle = _bundle(with_invalid_row=True)
    prior = _prior(bundle)
    default = stack_tree_sparse_observation_factors(bundle, prior)
    complete = stack_tree_sparse_observation_factors(
        bundle,
        prior,
        include_invalid=True,
    )

    assert default.observation_count == 2
    assert complete.observation_count == 4
    np.testing.assert_array_equal(default.point_ids, np.asarray([10, 20], dtype=np.int64))
    np.testing.assert_array_equal(
        complete.point_ids,
        np.asarray([10, 11, 20, 21], dtype=np.int64),
    )


def test_binding_rejects_wrong_gauge_order_and_wrong_covariance() -> None:
    bundle = _bundle()
    dense_sparse = stack_sparse_observation_factors(bundle)
    prior = _prior(bundle)
    wrong_order = replace(prior, gauge_ids=("window-1", "window-0"))

    with pytest.raises(ValueError, match="order does not match"):
        bind_gauge_tree_prior(dense_sparse, wrong_order)

    wrong_covariance = GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=prior.gauge_ids,
        parent_indices=prior.parent_indices,
        transition_matrices=prior.transition_matrices,
        innovation_covariances=prior.innovation_covariance_blocks() * 1.2,
    )
    with pytest.raises(ValueError):
        bind_gauge_tree_prior(dense_sparse, wrong_covariance)


def test_binding_rejects_postconstruction_row_marginal_tampering() -> None:
    bundle = _bundle()
    dense_sparse = stack_sparse_observation_factors(bundle)
    changed = dense_sparse.marginal_world_covariance_m2.copy()
    changed[0] += np.eye(3) * 1.0e-4
    changed.setflags(write=False)
    object.__setattr__(
        dense_sparse,
        "marginal_world_covariance_m2",
        changed,
    )

    with pytest.raises(ValueError, match="does not match the tree gauge prior"):
        bind_gauge_tree_prior(dense_sparse, _prior(bundle))


def test_tree_sparse_result_is_factory_built_and_slotted() -> None:
    with pytest.raises(TypeError, match="bind_gauge_tree_prior"):
        TreeSparseStackedObservationFactors()

    bundle = _bundle()
    tree_sparse = stack_tree_sparse_observation_factors(bundle, _prior(bundle))
    with pytest.raises((AttributeError, TypeError)):
        tree_sparse.new_attribute = 1  # type: ignore[misc]


def test_tree_sparse_contract_rejects_wrong_types() -> None:
    bundle = _bundle()
    dense_sparse = stack_sparse_observation_factors(bundle)

    with pytest.raises(TypeError, match="SparseStackedObservationFactors"):
        bind_gauge_tree_prior(object(), _prior(bundle))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GaugeTreeSquareRootPriorV1"):
        bind_gauge_tree_prior(dense_sparse, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ObservationFactorBundle"):
        stack_tree_sparse_observation_factors(  # type: ignore[arg-type]
            object(),
            _prior(bundle),
        )
