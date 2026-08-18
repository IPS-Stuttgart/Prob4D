from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.api.v2 import (
    ObservationGaussianOperator,
    build_observation_gaussian_operator,
    observation_gaussian_nll,
    observation_log_determinant,
    observation_precision_quadratic,
    solve_observation_covariance,
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
    full_covariance += (
        dense_jacobian.reshape(6, 14)
        @ gauge_prior
        @ dense_jacobian.reshape(6, 14).T
    )
    return dense, sparse, tree, full_covariance


@pytest.mark.parametrize(
    ("stack_index", "backend"),
    [
        (0, "dense-gauge-root-woodbury-v1"),
        (1, "sparse-gauge-root-woodbury-v1"),
        (2, "tree-block-information-v1"),
    ],
)
def test_structured_gaussian_matches_dense_reference(
    stack_index: int,
    backend: str,
) -> None:
    dense, sparse, tree, full_covariance = _stacks()
    stacked = (dense, sparse, tree)[stack_index]
    residual = np.linspace(-0.3, 0.4, 6, dtype=np.float64).reshape(2, 3)
    operator = build_observation_gaussian_operator(stacked)

    expected_solve = np.linalg.solve(full_covariance, residual.reshape(-1))
    expected_quadratic = float(residual.reshape(-1) @ expected_solve)
    sign, expected_log_determinant = np.linalg.slogdet(full_covariance)
    assert sign == 1.0
    expected_nll = 0.5 * (
        6 * np.log(2.0 * np.pi)
        + expected_log_determinant
        + expected_quadratic
    )

    assert isinstance(operator, ObservationGaussianOperator)
    assert operator.observation_count == 2
    assert operator.dimension == 6
    assert operator.factorization_backend == backend
    assert operator.factor_storage_nbytes > 0
    assert operator.dense_covariance_nbytes == full_covariance.nbytes
    assert operator.factor_storage_ratio_to_dense == pytest.approx(
        operator.factor_storage_nbytes / full_covariance.nbytes
    )
    np.testing.assert_allclose(operator.solve(residual).reshape(-1), expected_solve)
    assert operator.precision_quadratic(residual) == pytest.approx(expected_quadratic)
    assert operator.log_determinant == pytest.approx(expected_log_determinant)
    assert operator.gaussian_nll(residual) == pytest.approx(expected_nll)
    assert operator.gaussian_nll(residual, per_dimension=True) == pytest.approx(
        expected_nll / 6
    )


@pytest.mark.parametrize("stack_index", [0, 1, 2])
def test_batched_solve_and_convenience_functions(stack_index: int) -> None:
    dense, sparse, tree, full_covariance = _stacks()
    stacked = (dense, sparse, tree)[stack_index]
    right_hand_sides = np.asarray(
        [
            [[1.0, -0.5], [0.2, 0.7], [0.0, 0.1]],
            [[-0.3, 0.4], [0.8, -0.2], [0.5, 0.9]],
        ],
        dtype=np.float64,
    )
    expected = np.linalg.solve(full_covariance, right_hand_sides.reshape(6, 2))
    residual = right_hand_sides[:, :, 0]

    np.testing.assert_allclose(
        solve_observation_covariance(stacked, right_hand_sides).reshape(6, 2),
        expected,
    )
    expected_quadratic = float(residual.reshape(-1) @ expected[:, 0])
    assert observation_precision_quadratic(stacked, residual) == pytest.approx(
        expected_quadratic
    )
    assert observation_log_determinant(stacked) == pytest.approx(
        np.linalg.slogdet(full_covariance)[1]
    )
    assert observation_gaussian_nll(stacked, residual) == pytest.approx(
        build_observation_gaussian_operator(stacked).gaussian_nll(residual)
    )


@pytest.mark.parametrize("stack_index", [0, 1])
def test_operator_accepts_singular_gauge_covariance(stack_index: int) -> None:
    dense, sparse, _, _ = _stacks()
    stacked = (dense, sparse)[stack_index]
    gauge_prior = np.asarray(stacked.gauge_prior_covariance).copy()
    eigenvalues, eigenvectors = np.linalg.eigh(gauge_prior)
    eigenvalues[:7] = 0.0
    singular_prior = (eigenvectors * eigenvalues) @ eigenvectors.T
    singular_prior = 0.5 * (singular_prior + singular_prior.T)
    conditional = np.asarray(stacked.conditional_world_covariance_m2)
    dense_jacobian = np.asarray(dense.gauge_jacobian)
    marginal = conditional + np.asarray(
        [
            dense_jacobian[index]
            @ singular_prior
            @ dense_jacobian[index].T
            for index in range(2)
        ]
    )
    singular_stack = replace(
        stacked,
        gauge_prior_covariance=singular_prior,
        marginal_world_covariance_m2=marginal,
    )
    full_covariance = np.zeros((6, 6), dtype=np.float64)
    full_covariance[:3, :3] = conditional[0]
    full_covariance[3:, 3:] = conditional[1]
    full_covariance += (
        dense_jacobian.reshape(6, 14)
        @ singular_prior
        @ dense_jacobian.reshape(6, 14).T
    )
    residual = np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0

    operator = build_observation_gaussian_operator(singular_stack)
    np.testing.assert_allclose(
        operator.solve(residual).reshape(-1),
        np.linalg.solve(full_covariance, residual.reshape(-1)),
        atol=1e-11,
        rtol=1e-11,
    )
    assert operator.log_determinant == pytest.approx(
        np.linalg.slogdet(full_covariance)[1]
    )


@pytest.mark.parametrize("stack_index", [0, 1])
def test_operator_accepts_zero_gauge_covariance(stack_index: int) -> None:
    dense, sparse, _, _ = _stacks()
    stacked = (dense, sparse)[stack_index]
    conditional = np.asarray(stacked.conditional_world_covariance_m2)
    zero_prior = np.zeros_like(np.asarray(stacked.gauge_prior_covariance))
    zero_stack = replace(
        stacked,
        gauge_prior_covariance=zero_prior,
        marginal_world_covariance_m2=conditional,
    )
    residual = np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0
    full_covariance = np.zeros((6, 6), dtype=np.float64)
    full_covariance[:3, :3] = conditional[0]
    full_covariance[3:, 3:] = conditional[1]

    operator = build_observation_gaussian_operator(zero_stack)

    np.testing.assert_allclose(
        operator.solve(residual).reshape(-1),
        np.linalg.solve(full_covariance, residual.reshape(-1)),
    )
    assert operator.log_determinant == pytest.approx(
        np.linalg.slogdet(full_covariance)[1]
    )


def test_operator_fails_closed_for_singular_conditional_covariance() -> None:
    dense, _, _, _ = _stacks()
    conditional = np.asarray(dense.conditional_world_covariance_m2).copy()
    conditional[0, 0, 0] = 0.0
    gauge_jacobian = np.asarray(dense.gauge_jacobian)
    gauge_prior = np.asarray(dense.gauge_prior_covariance)
    marginal = conditional + np.asarray(
        [
            gauge_jacobian[index] @ gauge_prior @ gauge_jacobian[index].T
            for index in range(2)
        ]
    )
    singular_stack = replace(
        dense,
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
    )

    with pytest.raises(ValueError, match="strictly positive definite"):
        build_observation_gaussian_operator(singular_stack)


def test_operator_input_validation() -> None:
    _, sparse, _, _ = _stacks()
    operator = build_observation_gaussian_operator(sparse)

    with pytest.raises(ValueError, match="shape"):
        operator.solve(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        residual = np.ones((2, 3))
        residual[0, 0] = np.nan
        operator.solve(residual)
    with pytest.raises(ValueError, match="shape"):
        operator.precision_quadratic(np.ones((2, 3, 1)))
    with pytest.raises(TypeError, match="per_dimension"):
        operator.gaussian_nll(np.ones((2, 3)), per_dimension=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="stacked must be"):
        build_observation_gaussian_operator(object())  # type: ignore[arg-type]


def test_tree_information_factorization_handles_branching_gauge_tree() -> None:
    gauge_ids = ("root", "left", "right")
    parent_indices = np.asarray([-1, 0, 0], dtype=np.int64)
    transitions = np.zeros((3, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7) * 0.20
    transitions[2] = np.eye(7) * -0.15
    innovations = np.asarray(
        [
            np.eye(7) * 0.30,
            np.eye(7) * 0.20,
            np.eye(7) * 0.40,
        ]
    )
    prior = GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=gauge_ids,
        parent_indices=parent_indices,
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )
    gauge_covariance = prior.materialize_dense_covariance()
    conditional = np.asarray(
        [
            np.diag([0.2, 0.3, 0.4]),
            np.diag([0.5, 0.6, 0.7]),
            np.diag([0.8, 0.9, 1.0]),
        ]
    )
    local_jacobian = np.zeros((3, 3, 7), dtype=np.float64)
    for index, scale in enumerate((1.0, 1.5, 0.7)):
        local_jacobian[index, :, :3] = np.eye(3) * scale
    gauge_indices = np.arange(3, dtype=np.int64)
    dense_jacobian = np.zeros((3, 3, 21), dtype=np.float64)
    for index in range(3):
        dense_jacobian[index, :, 7 * index : 7 * (index + 1)] = local_jacobian[index]
    marginal = conditional + np.asarray(
        [
            local_jacobian[index]
            @ gauge_covariance[7 * index : 7 * (index + 1), 7 * index : 7 * (index + 1)]
            @ local_jacobian[index].T
            for index in range(3)
        ]
    )
    sparse = SparseStackedObservationFactors(
        world_mean_m=np.zeros((3, 3), dtype=np.float64),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        local_gauge_jacobian=local_jacobian,
        gauge_indices=gauge_indices,
        gauge_prior_covariance=gauge_covariance,
        association_probability=np.ones(3),
        prior_reliability=np.ones(3),
        prior_nominal_probability=np.ones(3),
        composite_weight=np.ones(3),
        point_ids=np.arange(3, dtype=np.int64),
        frame_indices=np.arange(3, dtype=np.int64),
        view_ids=("camera",) * 3,
        factor_ids=("factor-0", "factor-1", "factor-2"),
        correlation_group_ids=("group-0", "group-1", "group-2"),
        gauge_ids=gauge_ids,
        causal_frame_stop=3,
    )
    tree = bind_gauge_tree_prior(sparse, prior)
    full_covariance = np.zeros((9, 9), dtype=np.float64)
    for index in range(3):
        full_covariance[3 * index : 3 * (index + 1), 3 * index : 3 * (index + 1)] = (
            conditional[index]
        )
    full_covariance += (
        dense_jacobian.reshape(9, 21)
        @ gauge_covariance
        @ dense_jacobian.reshape(9, 21).T
    )
    residual = np.linspace(-0.4, 0.6, 9).reshape(3, 3)

    operator = build_observation_gaussian_operator(tree)
    expected_factor_storage = int(
        conditional.nbytes
        + 2 * prior.gauge_count * 7 * 7 * np.dtype(np.float64).itemsize
    )
    assert operator.factor_storage_nbytes == expected_factor_storage

    np.testing.assert_allclose(
        operator.solve(residual).reshape(-1),
        np.linalg.solve(full_covariance, residual.reshape(-1)),
        atol=1e-11,
        rtol=1e-11,
    )
    assert operator.log_determinant == pytest.approx(
        np.linalg.slogdet(full_covariance)[1]
    )
