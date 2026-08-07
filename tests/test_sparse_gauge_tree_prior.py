from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from prob4d.sparse_gauge_tree_prior import SparseGaugeTreePrior


def _random_prior(seed: int, gauge_count: int = 6) -> SparseGaugeTreePrior:
    generator = np.random.default_rng(seed)
    parents = np.full(gauge_count, -1, dtype=np.int64)
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    innovations = np.empty_like(transitions)
    for index in range(gauge_count):
        basis = generator.normal(size=(7, 7))
        innovations[index] = basis @ basis.T + np.eye(7) * 0.5
        if index:
            parents[index] = int(generator.integers(0, index))
            transitions[index] = (
                np.eye(7) * generator.uniform(0.45, 0.9)
                + generator.normal(scale=0.04, size=(7, 7))
            )
    return SparseGaugeTreePrior.from_components(
        window_ids=tuple(f"window-{index}" for index in range(gauge_count)),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )


def _sibling_prior(seed: int = 8) -> SparseGaugeTreePrior:
    generator = np.random.default_rng(seed)
    parents = np.asarray([-1, 0, 0], dtype=np.int64)
    transitions = np.zeros((3, 7, 7), dtype=np.float64)
    transitions[1:] = (
        np.eye(7)[None] * 0.65
        + generator.normal(scale=0.03, size=(2, 7, 7))
    )
    innovations = np.empty_like(transitions)
    for index in range(3):
        basis = generator.normal(size=(7, 7))
        innovations[index] = basis @ basis.T + np.eye(7) * 0.5
    return SparseGaugeTreePrior.from_components(
        window_ids=("root", "left", "right"),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )


def _dense_observation_design(
    jacobians: np.ndarray,
    gauge_indices: np.ndarray,
    gauge_count: int,
) -> np.ndarray:
    design = np.zeros(
        (3 * len(jacobians), 7 * gauge_count),
        dtype=np.float64,
    )
    for row, gauge_index in enumerate(gauge_indices):
        design[
            3 * row : 3 * (row + 1),
            7 * gauge_index : 7 * (gauge_index + 1),
        ] = jacobians[row]
    return design


def test_dense_square_root_and_covariance_are_exact() -> None:
    prior = _random_prior(1)
    root = prior.dense_square_root()
    covariance = prior.dense_covariance()

    np.testing.assert_allclose(
        covariance,
        root @ root.T,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(covariance, covariance.T, atol=0.0, rtol=0.0)


def test_dense_tree_admission_reconstructs_exact_covariance() -> None:
    prior = _random_prior(2)
    admitted = SparseGaugeTreePrior.from_dense_covariance(
        window_ids=prior.window_ids,
        parent_indices=prior.parent_indices,
        covariance=prior.dense_covariance(),
    )

    np.testing.assert_allclose(
        admitted.dense_covariance(),
        prior.dense_covariance(),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_array_equal(admitted.parent_indices, prior.parent_indices)


def test_joint_posterior_adapter_requires_and_preserves_tree_lineage() -> None:
    prior = _random_prior(3)
    parent_ids = tuple(
        None if index == 0 else prior.window_ids[int(prior.parent_indices[index])]
        for index in range(prior.gauge_count)
    )
    posterior = SimpleNamespace(
        window_ids=prior.window_ids,
        parent_window_ids=parent_ids,
        joint_covariance=prior.dense_covariance(),
        cross_window_covariance_preserved=True,
    )

    admitted = SparseGaugeTreePrior.from_joint_gauge_posterior(posterior)
    np.testing.assert_allclose(
        admitted.dense_covariance(),
        prior.dense_covariance(),
    )

    posterior.cross_window_covariance_preserved = False
    with pytest.raises(ValueError, match="complete cross-window covariance"):
        SparseGaugeTreePrior.from_joint_gauge_posterior(posterior)


def test_covariance_actions_match_dense_vector_and_matrix_products() -> None:
    prior = _random_prior(4)
    generator = np.random.default_rng(41)
    vector = generator.normal(size=prior.dense_dimension)
    matrix = generator.normal(size=(prior.dense_dimension, 5))
    covariance = prior.dense_covariance()

    np.testing.assert_allclose(prior.apply_covariance(vector), covariance @ vector)
    np.testing.assert_allclose(prior.apply_covariance(matrix), covariance @ matrix)
    np.testing.assert_allclose(
        prior.apply_covariance(vector.reshape(prior.gauge_count, 7)),
        (covariance @ vector).reshape(prior.gauge_count, 7),
    )


def test_information_actions_and_log_determinant_match_dense_algebra() -> None:
    prior = _random_prior(5)
    generator = np.random.default_rng(51)
    matrix = generator.normal(size=(prior.dense_dimension, 4))
    covariance = prior.dense_covariance()

    np.testing.assert_allclose(
        prior.apply_information(matrix),
        np.linalg.solve(covariance, matrix),
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        prior.apply_covariance(prior.apply_information(matrix)),
        matrix,
        atol=2e-10,
        rtol=2e-10,
    )
    sign, dense_log_determinant = np.linalg.slogdet(covariance)
    assert sign > 0.0
    assert prior.log_determinant() == pytest.approx(
        dense_log_determinant,
        abs=2e-10,
    )
    assert prior.supports_information_actions is True


def test_selected_marginal_and_cross_covariances_match_dense_blocks() -> None:
    prior = _random_prior(6)
    covariance = prior.dense_covariance()
    left = ("window-4", "window-1")
    right = (3, 0)
    left_indices = np.concatenate([np.arange(28, 35), np.arange(7, 14)])
    right_indices = np.concatenate([np.arange(21, 28), np.arange(0, 7)])

    np.testing.assert_allclose(
        prior.cross_covariance(left, right),
        covariance[np.ix_(left_indices, right_indices)],
    )
    marginal_indices = np.concatenate([np.arange(14, 21), np.arange(35, 42)])
    np.testing.assert_allclose(
        prior.marginal_covariance((2, "window-5")),
        covariance[np.ix_(marginal_indices, marginal_indices)],
    )


def test_sparse_observation_covariance_action_matches_dense_design() -> None:
    prior = _random_prior(7)
    generator = np.random.default_rng(71)
    observation_count = 13
    jacobians = generator.normal(size=(observation_count, 3, 7))
    gauge_indices = generator.integers(
        0,
        prior.gauge_count,
        size=observation_count,
    )
    value = generator.normal(size=(3 * observation_count, 3))
    design = _dense_observation_design(
        jacobians,
        gauge_indices,
        prior.gauge_count,
    )
    expected_covariance = design @ prior.dense_covariance() @ design.T

    np.testing.assert_allclose(
        prior.apply_observation_covariance(
            jacobians,
            gauge_indices,
            value,
        ),
        expected_covariance @ value,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        prior.observation_covariance(jacobians, gauge_indices),
        expected_covariance,
        atol=1e-10,
        rtol=1e-10,
    )


def test_dense_admission_rejects_correlated_sibling_innovations() -> None:
    prior = _sibling_prior()
    root = prior.dense_square_root()
    innovation_dependence = np.eye(prior.dense_dimension, dtype=np.float64)
    cross = np.eye(7, dtype=np.float64) * 0.15
    innovation_dependence[7:14, 14:21] = cross
    innovation_dependence[14:21, 7:14] = cross
    non_tree_covariance = root @ innovation_dependence @ root.T

    with pytest.raises(
        ValueError,
        match="not representable by the declared causal tree",
    ):
        SparseGaugeTreePrior.from_dense_covariance(
            window_ids=prior.window_ids,
            parent_indices=prior.parent_indices,
            covariance=non_tree_covariance,
        )


def test_semidefinite_prior_supports_covariance_but_fails_information_closed() -> None:
    parents = np.asarray([-1, 0], dtype=np.int64)
    transitions = np.zeros((2, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7)
    innovations = np.repeat(np.eye(7)[None], 2, axis=0)
    innovations[1, -1, -1] = 0.0
    prior = SparseGaugeTreePrior.from_components(
        window_ids=("root", "child"),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )

    value = np.ones(14, dtype=np.float64)
    assert np.all(np.isfinite(prior.apply_covariance(value)))
    assert prior.supports_information_actions is False
    with pytest.raises(ValueError, match="strictly positive-definite"):
        prior.apply_information(value)
    with pytest.raises(ValueError, match="strictly positive-definite"):
        prior.log_determinant()


def test_arrays_are_irreversibly_readonly_and_parents_fail_closed() -> None:
    prior = _random_prior(9)

    for value in (
        prior.parent_indices,
        prior.transition_matrices,
        prior.innovation_roots,
    ):
        assert value.flags.writeable is False
        with pytest.raises(ValueError):
            value.setflags(write=True)

    bad_parents = prior.parent_indices.copy()
    bad_parents[3] = 3
    with pytest.raises(ValueError, match="parent must precede"):
        SparseGaugeTreePrior(
            window_ids=prior.window_ids,
            parent_indices=bad_parents,
            transition_matrices=prior.transition_matrices,
            innovation_roots=prior.innovation_roots,
        )


def test_storage_is_linear_and_randomized_tree_parity_is_stable() -> None:
    prior = _random_prior(10, gauge_count=64)

    assert prior.retained_nbytes == 50_688
    assert prior.dense_covariance_nbytes == 1_605_632
    assert prior.dense_to_sparse_storage_ratio == pytest.approx(
        1_605_632 / 50_688
    )
    assert prior.dense_to_sparse_storage_ratio > 31.0

    for seed in range(20):
        candidate = _random_prior(1_000 + seed, gauge_count=2 + seed % 9)
        dense = candidate.dense_covariance()
        admitted = SparseGaugeTreePrior.from_dense_covariance(
            window_ids=candidate.window_ids,
            parent_indices=candidate.parent_indices,
            covariance=dense,
        )
        generator = np.random.default_rng(2_000 + seed)
        right_hand_side = generator.normal(
            size=(candidate.dense_dimension, 2)
        )
        np.testing.assert_allclose(
            admitted.apply_covariance(right_hand_side),
            dense @ right_hand_side,
            atol=2e-9,
            rtol=2e-9,
        )
