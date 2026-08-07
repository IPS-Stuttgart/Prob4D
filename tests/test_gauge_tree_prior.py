from __future__ import annotations

import numpy as np
import pytest

from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _prior(*, gauge_count: int = 5) -> GaugeTreeSquareRootPriorV1:
    gauge_ids = tuple(f"window-{index}" for index in range(gauge_count))
    parents = np.asarray([-1] + [(index - 1) // 2 for index in range(1, gauge_count)])
    transitions = np.zeros((gauge_count, 7, 7), dtype=np.float64)
    scales = np.empty_like(transitions)
    scales[0] = np.diag(np.linspace(0.05, 0.11, 7))
    for index in range(1, gauge_count):
        transition = np.eye(7) * (0.7 + 0.02 * index)
        transition[4:, :3] = 0.01 * index
        transitions[index] = transition
        scale = np.diag(np.linspace(0.02, 0.04, 7) * (1.0 + 0.05 * index))
        scale[3, 0] = 0.002
        scale[6, 2] = -0.001
        scales[index] = scale
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=gauge_ids,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
    )


def _explicit_dense(prior: GaugeTreeSquareRootPriorV1) -> np.ndarray:
    transform = np.zeros((prior.dimension, prior.dimension), dtype=np.float64)
    for column in range(prior.dimension):
        innovation = np.zeros((prior.gauge_count, 7), dtype=np.float64)
        innovation.reshape(-1)[column] = 1.0
        state = np.empty_like(innovation)
        state[0] = innovation[0]
        for child in range(1, prior.gauge_count):
            parent = int(prior.parent_indices[child])
            state[child] = (
                prior.transition_matrices[child] @ state[parent] + innovation[child]
            )
        transform[:, column] = state.reshape(-1)
    block_covariance = np.zeros((prior.dimension, prior.dimension), dtype=np.float64)
    for index, scale in enumerate(prior.innovation_scale_tril):
        item = slice(7 * index, 7 * (index + 1))
        block_covariance[item, item] = scale @ scale.T
    return transform @ block_covariance @ transform.T


def test_dense_materialization_matches_explicit_generative_tree() -> None:
    prior = _prior()
    np.testing.assert_allclose(
        prior.materialize_dense_covariance(),
        _explicit_dense(prior),
        atol=1e-14,
        rtol=1e-12,
    )


def test_covariance_and_information_actions_match_dense_algebra() -> None:
    prior = _prior()
    dense = prior.materialize_dense_covariance()
    vector = np.linspace(-0.4, 0.7, prior.dimension)
    matrix = np.column_stack((vector, vector[::-1], np.ones(prior.dimension)))
    np.testing.assert_allclose(prior.covariance_action(vector), dense @ vector)
    np.testing.assert_allclose(prior.covariance_action(matrix), dense @ matrix)
    np.testing.assert_allclose(
        prior.information_action(vector),
        np.linalg.solve(dense, vector),
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        prior.solve_information(prior.information_action(vector)),
        vector,
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        prior.solve_covariance(prior.covariance_action(vector)),
        vector,
        atol=2e-10,
        rtol=2e-10,
    )


def test_quadratic_and_log_determinant_match_dense_result() -> None:
    prior = _prior()
    dense = prior.materialize_dense_covariance()
    vector = np.linspace(-0.2, 0.3, prior.dimension)
    sign, log_determinant = np.linalg.slogdet(dense)
    assert sign == 1.0
    assert prior.information_quadratic(vector) == pytest.approx(
        float(vector @ np.linalg.solve(dense, vector)), rel=2e-10, abs=2e-10
    )
    assert prior.log_determinant_covariance() == pytest.approx(
        log_determinant, rel=2e-10, abs=2e-10
    )


def test_direct_transition_covariance_builder_matches_constructor() -> None:
    source = _prior()
    built = GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=source.gauge_ids,
        parent_indices=source.parent_indices,
        transition_matrices=source.transition_matrices,
        innovation_covariances=source.innovation_covariance_blocks(),
    )
    assert built.parent_gauge_ids == source.parent_gauge_ids
    np.testing.assert_allclose(
        built.materialize_dense_covariance(),
        source.materialize_dense_covariance(),
        atol=1e-14,
        rtol=1e-12,
    )


def test_root_only_prior_is_supported() -> None:
    prior = _prior(gauge_count=1)
    dense = prior.materialize_dense_covariance()
    vector = np.linspace(-0.2, 0.2, 7)
    assert prior.parent_gauge_ids == (None,)
    np.testing.assert_allclose(prior.covariance_action(vector), dense @ vector)
    recovered = GaugeTreeSquareRootPriorV1.from_dense_covariance(
        gauge_ids=prior.gauge_ids,
        parent_indices=prior.parent_indices,
        joint_covariance=dense,
    )
    np.testing.assert_allclose(
        recovered.materialize_dense_covariance(), dense, atol=1e-12, rtol=1e-10
    )


def test_dense_factorization_round_trip_and_digest_verification() -> None:
    source = _prior()
    dense = source.materialize_dense_covariance()
    recovered = GaugeTreeSquareRootPriorV1.from_dense_covariance(
        gauge_ids=source.gauge_ids,
        parent_indices=source.parent_indices,
        joint_covariance=dense,
    )
    np.testing.assert_allclose(
        recovered.materialize_dense_covariance(), dense, atol=1e-12, rtol=1e-10
    )
    np.testing.assert_allclose(
        recovered.transition_matrices,
        source.transition_matrices,
        atol=1e-12,
        rtol=1e-10,
    )
    recovered.verify_dense_covariance(dense, require_source_digest=True)
    tampered = dense.copy()
    tampered[0, 0] += 1e-8
    with pytest.raises(ValueError, match="bound source digest"):
        recovered.verify_dense_covariance(tampered, require_source_digest=True)


def test_dense_factorization_rejects_non_tree_structure() -> None:
    generator = np.random.default_rng(7)
    matrix = generator.normal(size=(28, 28))
    covariance = matrix @ matrix.T + np.eye(28)
    with pytest.raises(ValueError, match="not representable by the declared causal tree"):
        GaugeTreeSquareRootPriorV1.from_dense_covariance(
            gauge_ids=("a", "b", "c", "d"),
            parent_indices=np.asarray([-1, 0, 0, 1]),
            joint_covariance=covariance,
        )


def test_selected_and_cross_covariances_preserve_order() -> None:
    prior = _prior()
    dense = prior.materialize_dense_covariance()
    selected_ids = ("window-4", "window-0", "window-2")
    indices = np.concatenate([np.arange(7 * item, 7 * (item + 1)) for item in (4, 0, 2)])
    np.testing.assert_allclose(
        prior.selected_covariance(selected_ids),
        dense[np.ix_(indices, indices)],
    )
    left = np.concatenate([np.arange(7 * item, 7 * (item + 1)) for item in (3, 1)])
    right = np.concatenate([np.arange(7 * item, 7 * (item + 1)) for item in (4, 0)])
    np.testing.assert_allclose(
        prior.cross_covariance(("window-3", "window-1"), ("window-4", "window-0")),
        dense[np.ix_(left, right)],
    )
