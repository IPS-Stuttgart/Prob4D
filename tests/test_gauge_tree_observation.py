from __future__ import annotations

import numpy as np
import pytest

from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _prior() -> GaugeTreeSquareRootPriorV1:
    parents = np.asarray([-1, 0, 0, 1, 1])
    transitions = np.zeros((5, 7, 7), dtype=np.float64)
    scales = np.empty_like(transitions)
    scales[0] = np.diag(np.linspace(0.05, 0.11, 7))
    for index in range(1, 5):
        transitions[index] = np.eye(7) * (0.7 + 0.02 * index)
        transitions[index, 4:, :3] = 0.01 * index
        scales[index] = np.diag(np.linspace(0.02, 0.04, 7) * (1.0 + 0.05 * index))
        scales[index, 3, 0] = 0.002
        scales[index, 6, 2] = -0.001
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=tuple(f"window-{index}" for index in range(5)),
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
    )


def test_diagonal_and_row_marginal_covariances_match_dense_blocks() -> None:
    prior = _prior()
    dense = prior.materialize_dense_covariance()
    diagonal = prior.diagonal_covariance_blocks()
    for index in range(prior.gauge_count):
        item = slice(7 * index, 7 * (index + 1))
        np.testing.assert_allclose(diagonal[index], dense[item, item])
    generator = np.random.default_rng(11)
    jacobian = generator.normal(size=(8, 3, 7))
    indices = np.asarray([0, 1, 1, 2, 3, 4, 0, 3])
    expected = np.stack(
        [jacobian[row] @ diagonal[index] @ jacobian[row].T for row, index in enumerate(indices)]
    )
    np.testing.assert_allclose(prior.row_marginal_covariance(jacobian, indices), expected)


def test_observation_covariance_action_matches_dense_design() -> None:
    prior = _prior()
    dense = prior.materialize_dense_covariance()
    generator = np.random.default_rng(13)
    row_count = 9
    jacobian = generator.normal(size=(row_count, 3, 7))
    indices = np.asarray([0, 1, 4, 2, 1, 3, 0, 4, 2])
    design = np.zeros((3 * row_count, prior.dimension), dtype=np.float64)
    for row, gauge_index in enumerate(indices):
        design[3 * row : 3 * (row + 1), 7 * gauge_index : 7 * (gauge_index + 1)] = jacobian[row]
    value = generator.normal(size=(row_count, 3, 4))
    expected = (design @ dense @ design.T @ value.reshape(3 * row_count, 4)).reshape(
        row_count, 3, 4
    )
    np.testing.assert_allclose(
        prior.observation_covariance_action(jacobian, indices, value),
        expected,
        atol=1e-12,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        prior.observation_covariance_action(jacobian, indices, value[:, :, 0]),
        expected[:, :, 0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_marginal_observation_action_adds_local_covariance_once() -> None:
    prior = _prior()
    generator = np.random.default_rng(17)
    jacobian = generator.normal(size=(6, 3, 7))
    indices = np.asarray([0, 1, 2, 3, 4, 1])
    local_scale = generator.normal(size=(6, 3, 3))
    local = np.einsum("mij,mkj->mik", local_scale, local_scale) + np.eye(3)[None] * 0.1
    value = generator.normal(size=(6, 3))
    expected = np.einsum("mij,mj->mi", local, value) + (
        prior.observation_covariance_action(jacobian, indices, value)
    )
    np.testing.assert_allclose(
        prior.marginal_observation_covariance_action(jacobian, indices, local, value),
        expected,
    )
    nonsymmetric = local.copy()
    nonsymmetric[0, 0, 1] += 0.2
    with pytest.raises(ValueError, match="symmetric"):
        prior.marginal_observation_covariance_action(jacobian, indices, nonsymmetric, value)
    indefinite = local.copy()
    indefinite[0] = -np.eye(3)
    with pytest.raises(ValueError, match="positive semidefinite"):
        prior.marginal_observation_covariance_action(jacobian, indices, indefinite, value)


def test_samples_are_deterministic_and_follow_tree_innovations() -> None:
    prior = _prior()
    first = prior.sample(seed=23, sample_count=4)
    second = prior.sample(seed=23, sample_count=4)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, prior.gauge_count, 7)
    innovations = np.asarray(prior.innovation_coordinates(first.transpose(1, 2, 0)))
    assert innovations.shape == (prior.gauge_count, 7, 4)
    assert np.all(np.isfinite(innovations))
