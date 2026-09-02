"""Independent dense tests for the posterior rank--distortion theorem."""

from __future__ import annotations

import numpy as np
import pytest

from prob4d.posterior_rank_distortion import posterior_rank_distortion_frontier


class DenseInnovation:
    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.dimension = covariance.shape[0]
        self.observation_count = self.dimension // 3
        self.calls = 0

    def solve(self, value: object) -> np.ndarray:
        self.calls += 1
        raw = np.asarray(value, dtype=np.float64)
        return np.linalg.solve(
            self.covariance,
            raw.reshape(self.dimension, -1),
        ).reshape(raw.shape)


def model(seed: int = 17, rank: int = 7, qdim: int = 3):
    rng = np.random.default_rng(seed)
    dimension, state_dimension = 36, 8
    diagonal = np.diag(rng.uniform(0.3, 1.5, dimension))
    factor = rng.normal(size=(dimension, rank)) / np.sqrt(max(rank, 1))
    state_map = rng.normal(size=(dimension, state_dimension)) / 2
    query_map = rng.normal(size=(qdim, state_dimension))
    prior = query_map @ query_map.T
    cross = query_map @ state_map.T
    remainder = diagonal + state_map @ state_map.T
    innovation = remainder + factor @ factor.T
    return factor, prior, cross, remainder, innovation


def posterior(prior: np.ndarray, cross: np.ndarray, innovation: np.ndarray):
    gain = np.linalg.solve(innovation, cross.T).T
    return gain, prior - gain @ cross.T


def frontier(factor, prior, cross, innovation):
    return posterior_rank_distortion_frontier(
        factor.reshape(innovation.shape[0] // 3, 3, factor.shape[1]),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=DenseInnovation(innovation),
    )


def direct_trace_distortion(
    full_posterior: np.ndarray,
    reduced_posterior: np.ndarray,
) -> float:
    loss = full_posterior - reduced_posterior
    return float(np.trace(np.linalg.solve(full_posterior, loss)))


def test_frontier_matches_dense_reduced_posteriors_at_every_rank() -> None:
    factor, prior, cross, remainder, innovation = model(seed=21, rank=7, qdim=3)
    result = frontier(factor, prior, cross, innovation)
    _, full_posterior = posterior(prior, cross, innovation)

    assert result.original_rank == 7
    assert result.query_dimension == 3
    assert result.numerical_exact_rank == 3
    assert len(result.points) == 8
    np.testing.assert_array_less(-1e-12, result.generalized_eigenvalues)

    previous = np.inf
    for point in result.points:
        reduced_factor = point.compressed_factor_m.reshape(innovation.shape[0], -1)
        _, reduced_posterior = posterior(
            prior,
            cross,
            remainder + reduced_factor @ reduced_factor.T,
        )
        dense_distortion = direct_trace_distortion(full_posterior, reduced_posterior)
        np.testing.assert_allclose(
            point.audited_normalized_covariance_trace_loss,
            dense_distortion,
            atol=2e-11,
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            point.optimal_normalized_covariance_trace_loss,
            dense_distortion,
            atol=2e-11,
            rtol=2e-10,
        )
        assert point.maximum_normalized_covariance_contraction <= dense_distortion + 1e-12
        assert point.mean_shift_risk <= point.mean_shift_risk_upper_bound + 1e-12
        assert dense_distortion <= previous + 1e-11
        previous = dense_distortion

    for point in result.points[result.numerical_exact_rank :]:
        assert point.optimal_normalized_covariance_trace_loss < 1e-10
        assert point.exact_posterior


def _random_orthonormal_columns(
    rng: np.random.Generator,
    ambient: int,
    columns: int,
) -> np.ndarray:
    if columns == 0:
        return np.empty((ambient, 0))
    basis, _ = np.linalg.qr(rng.normal(size=(ambient, columns)), mode="reduced")
    return basis


def test_generalized_eigen_frontier_beats_sampled_same_rank_subspaces() -> None:
    factor, prior, cross, remainder, innovation = model(seed=9, rank=4, qdim=2)
    result = frontier(factor, prior, cross, innovation)
    _, full_posterior = posterior(prior, cross, innovation)
    rng = np.random.default_rng(20260902)

    for retained_rank in (1, 2, 3):
        optimum = result.point(retained_rank).audited_normalized_covariance_trace_loss
        for _ in range(300):
            projection = _random_orthonormal_columns(rng, 4, retained_rank)
            reduced = factor @ projection
            _, candidate_posterior = posterior(
                prior,
                cross,
                remainder + reduced @ reduced.T,
            )
            candidate = direct_trace_distortion(full_posterior, candidate_posterior)
            assert candidate >= optimum - 2e-10


def test_existing_euclidean_svd_order_is_not_globally_trace_optimal() -> None:
    factor, prior, cross, remainder, innovation = model(seed=93, rank=7, qdim=3)
    result = frontier(factor, prior, cross, innovation)
    optimum = result.point(1).audited_normalized_covariance_trace_loss

    solved_cross = np.linalg.solve(innovation, cross.T)
    full_posterior = prior - cross @ solved_cross
    posterior_root = np.linalg.cholesky(full_posterior)
    response = factor.T @ solved_cross
    normalized_response = np.linalg.solve(posterior_root, response.T).T
    left, _, _ = np.linalg.svd(normalized_response, full_matrices=True)
    svd_projection = left[:, :1]
    svd_factor = factor @ svd_projection
    _, svd_posterior = posterior(
        prior,
        cross,
        remainder + svd_factor @ svd_factor.T,
    )
    svd_distortion = direct_trace_distortion(full_posterior, svd_posterior)

    np.testing.assert_allclose(optimum, 0.3703045854595735, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(svd_distortion, 0.4632161428167537, rtol=2e-10, atol=2e-12)
    assert svd_distortion / optimum > 1.24


def test_exact_rank_recovers_zero_distortion_theorem() -> None:
    factor, prior, cross, _, innovation = model(seed=3, rank=14, qdim=5)
    result = frontier(factor, prior, cross, innovation)
    assert result.numerical_exact_rank == 5
    assert result.point(4).audited_normalized_covariance_trace_loss > 1e-5
    for retained_rank in range(5, 15):
        point = result.point(retained_rank)
        assert point.audited_normalized_covariance_trace_loss < 2e-10
        assert point.exact_posterior


def test_latent_and_query_reparameterization_preserve_the_frontier() -> None:
    factor, prior, cross, _, innovation = model(seed=41, rank=7, qdim=3)
    original = frontier(factor, prior, cross, innovation)

    latent, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(7, 7)))
    latent_changed = frontier(factor @ latent, prior, cross, innovation)
    np.testing.assert_allclose(
        original.generalized_eigenvalues,
        latent_changed.generalized_eigenvalues,
        atol=2e-11,
        rtol=2e-10,
    )
    for rank in range(8):
        first = original.point(rank).compressed_factor_m.reshape(36, rank)
        second = latent_changed.point(rank).compressed_factor_m.reshape(36, rank)
        np.testing.assert_allclose(first @ first.T, second @ second.T, atol=2e-10)

    transform = np.array([[2.0, 0.1, 0.3], [0.2, 0.7, 0.0], [0.0, 0.2, 3.0]])
    query_changed = frontier(
        factor,
        transform @ prior @ transform.T,
        transform @ cross,
        innovation,
    )
    np.testing.assert_allclose(
        original.generalized_eigenvalues,
        query_changed.generalized_eigenvalues,
        atol=2e-11,
        rtol=2e-10,
    )


def test_budget_selection_is_fail_closed_to_a_valid_rank() -> None:
    factor, prior, cross, _, innovation = model(seed=93, rank=7, qdim=3)
    result = frontier(factor, prior, cross, innovation)
    selected = result.minimum_rank_for_trace_budget(0.4)
    assert selected.retained_rank == 1
    assert selected.audited_normalized_covariance_trace_loss < 0.4
    exact = result.minimum_rank_for_trace_budget(0.0)
    assert exact.retained_rank >= result.numerical_exact_rank

    with pytest.raises(TypeError, match="budget"):
        result.minimum_rank_for_trace_budget(True)
    with pytest.raises(ValueError, match="budget"):
        result.minimum_rank_for_trace_budget(-0.1)


def test_zero_shared_rank_and_invalid_inputs() -> None:
    factor, prior, cross, _, innovation = model(seed=2, rank=0, qdim=2)
    result = frontier(factor, prior, cross, innovation)
    assert result.original_rank == 0
    assert result.point(0).exact_posterior

    factor, prior, cross, _, innovation = model(seed=2, rank=3, qdim=2)
    with pytest.raises(TypeError, match="numerical_relative_tolerance"):
        posterior_rank_distortion_frontier(
            factor.reshape(-1, 3, 3),
            prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=DenseInnovation(innovation),
            numerical_relative_tolerance=True,
        )
