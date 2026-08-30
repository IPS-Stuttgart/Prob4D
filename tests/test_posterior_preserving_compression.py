"""Independent dense references for the experimental compression theorem."""

from __future__ import annotations

import numpy as np
import pytest

from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)


class DenseInnovation:
    """Test-only oracle; production uses Prob4D's structured operator."""

    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = covariance.copy()
        self.dimension = covariance.shape[0]
        self.observation_count = self.dimension // 3
        self.calls = 0

    def solve(self, value: object) -> np.ndarray:
        self.calls += 1
        raw = np.asarray(value)
        return np.linalg.solve(
            self.covariance, raw.reshape(self.dimension, -1)
        ).reshape(raw.shape)


def model(seed: int = 17, rank: int = 7, qdim: int = 3):
    rng = np.random.default_rng(seed)
    dimension, state_dimension = 36, 8
    d = np.diag(rng.uniform(0.3, 1.5, dimension))
    u = rng.normal(size=(dimension, rank)) / np.sqrt(max(rank, 1))
    f = rng.normal(size=(dimension, state_dimension)) / 2
    g = rng.normal(size=(qdim, state_dimension))
    prior, cross = g @ g.T, g @ f.T
    a = d + f @ f.T
    s = a + u @ u.T
    return u, prior, cross, a, s


def compress(u, prior, cross, s, **kwargs):
    return compress_shared_factor_for_posterior(
        u.reshape(s.shape[0] // 3, 3, u.shape[1]),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=DenseInnovation(s),
        **kwargs,
    )


def posterior(prior, cross, s):
    gain = np.linalg.solve(s, cross.T).T
    return gain, prior - gain @ cross.T


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("rank,qdim", [(7, 1), (14, 3), (28, 5)])
def test_exact_gain_for_every_innovation_and_query_covariance(seed, rank, qdim):
    u, prior, cross, a, s = model(seed, rank, qdim)
    solver = DenseInnovation(s)
    result = compress_shared_factor_for_posterior(
        u.reshape(-1, 3, rank),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross.reshape(qdim, -1, 3),
        innovation_operator=solver,
    )
    reduced = result.compressed_factor_m.reshape(s.shape[0], -1)
    gain, covariance = posterior(prior, cross, s)
    new_gain, new_covariance = posterior(prior, cross, a + reduced @ reduced.T)
    assert solver.calls == 1
    assert not result.exact_fallback
    assert result.retained_rank == result.numerical_required_rank == qdim
    np.testing.assert_allclose(new_gain, gain, atol=2e-13, rtol=2e-11)
    np.testing.assert_allclose(new_covariance, covariance, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(
        result.latent_projection.T @ result.latent_projection,
        np.eye(qdim), atol=1e-12,
    )
    assert result.mean_shift_risk < 1e-20
    assert result.relative_covariance_error < 1e-20


def test_covariance_trace_and_marginal_preservation_can_break_posterior():
    d = np.diag([1.0, 1e-4, 1.0])
    f = np.array([[1.0], [1.0], [0.0]])
    u = np.array([[1000.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    a, prior, cross = d + f @ f.T, np.ones((1, 1)), f.T
    s = a + u @ u.T
    marginal_projection = np.array([[1.0, 0.0, 0.0]])
    naive = u[:, :1]
    np.testing.assert_array_equal(
        marginal_projection @ u @ u.T @ marginal_projection.T,
        marginal_projection @ naive @ naive.T @ marginal_projection.T,
    )
    assert np.sum(naive**2) / np.sum(u**2) > 0.999999
    full_gain, full_covariance = posterior(prior, cross, s)
    _, naive_covariance = posterior(prior, cross, a + naive @ naive.T)
    assert full_covariance[0, 0] / naive_covariance[0, 0] > 5000
    result = compress(u, prior, cross, s, maximum_rank=1)
    assert result.retained_rank == 1
    assert not result.exact_fallback
    reduced = result.compressed_factor_m.reshape(3, 1)
    new_gain, new_covariance = posterior(prior, cross, a + reduced @ reduced.T)
    np.testing.assert_allclose(new_gain, full_gain, atol=2e-12)
    np.testing.assert_allclose(new_covariance, full_covariance, atol=2e-12)


def test_rank_cap_returns_original_factor_not_an_inexact_truncation():
    u, prior, cross, _, s = model()
    result = compress(u, prior, cross, s, maximum_rank=2)
    assert result.exact_fallback
    assert result.retained_rank == 7
    assert result.reason == "no-parity-preserving-reduction-within-cap"
    assert result.compressed_factor_m.tobytes() == u.reshape(-1, 3, 7).tobytes()
    np.testing.assert_array_equal(result.latent_projection, np.eye(7))


def test_full_rank_is_necessary_for_a_full_query():
    u, prior, cross, _, s = model(rank=2, qdim=3)
    result = compress(u, prior, cross, s)
    assert result.exact_fallback
    assert result.reason == "full-rank-required"
    assert result.retained_rank == 2


def test_zero_cross_covariance_discards_every_shared_direction():
    u, prior, cross, _, s = model()
    result = compress(u, prior, np.zeros_like(cross), s, maximum_rank=0)
    assert not result.exact_fallback
    assert result.retained_rank == 0
    assert result.compressed_factor_m.shape == (12, 3, 0)


def test_no_shared_factor_has_a_well_defined_empty_result():
    u, prior, cross, _, s = model(rank=0)
    result = compress(u, prior, cross, s)
    assert result.original_rank == result.retained_rank == 0
    assert not result.exact_fallback
    assert result.reason == "no-shared-factor"


def test_rank_deficient_shared_factor_does_not_invent_information():
    u, prior, cross, a, _ = model(rank=1)
    u = np.column_stack([u, 2 * u, np.zeros_like(u), -u])
    s = a + u @ u.T
    result = compress(u, prior, cross, s)
    assert result.retained_rank == 1
    reduced = result.compressed_factor_m.reshape(36, 1)
    np.testing.assert_allclose(reduced @ reduced.T, u @ u.T, atol=1e-12)


def test_too_aggressive_numerical_rank_is_audited_and_repaired():
    u, prior, cross, _, s = model(rank=7, qdim=3)
    cross = cross.copy()
    cross[1:] *= 1e-3
    prior = 100.0 * np.eye(3)
    result = compress(u, prior, cross, s, rank_relative_tolerance=0.9)
    assert result.numerical_required_rank == 1
    assert result.retained_rank == 3
    capped = compress(
        u, prior, cross, s, rank_relative_tolerance=0.9, maximum_rank=1
    )
    assert capped.exact_fallback


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_query_units_and_invertible_query_basis_do_not_change_subspace(scale):
    u, prior, cross, _, s = model()
    original = compress(u, prior, cross, s)
    transform = scale * np.array([[2.0, 0.1, 0.3], [0.2, 0.7, 0.0], [0.0, 0.2, 3.0]])
    changed = compress(u, transform @ prior @ transform.T, transform @ cross, s)
    np.testing.assert_allclose(
        original.latent_projection @ original.latent_projection.T,
        changed.latent_projection @ changed.latent_projection.T,
        atol=1e-11,
    )


def test_latent_orthogonal_reparameterization_preserves_compressed_covariance():
    u, prior, cross, _, s = model()
    orthogonal, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(7, 7)))
    first = compress(u, prior, cross, s)
    second = compress(u @ orthogonal, prior, cross, s)
    a = first.compressed_factor_m.reshape(36, 3)
    b = second.compressed_factor_m.reshape(36, 3)
    np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-11)


def test_inputs_are_not_mutated_and_outputs_are_independent_readonly_copies():
    u, prior, cross, _, s = model()
    before = [x.copy() for x in (u, prior, cross)]
    result = compress(u, prior, cross, s)
    retained = result.compressed_factor_m.copy()
    for value, expected in zip((u, prior, cross), before, strict=True):
        np.testing.assert_array_equal(value, expected)
        value.fill(0)
    np.testing.assert_array_equal(result.compressed_factor_m, retained)
    assert not result.compressed_factor_m.flags.writeable
    assert not result.latent_projection.flags.writeable
    assert not np.shares_memory(result.compressed_factor_m, u)


@pytest.mark.parametrize("cap", [True, np.bool_(False), 1.5, "2"])
def test_noninteger_rank_caps_are_rejected(cap):
    u, prior, cross, _, s = model()
    with pytest.raises(TypeError, match="maximum_rank"):
        compress(u, prior, cross, s, maximum_rank=cap)


@pytest.mark.parametrize("name,value", [
    ("maximum_rank", -1),
    ("rank_relative_tolerance", -1),
    ("rank_relative_tolerance", 1.0),
    ("rank_relative_tolerance", np.nan),
    ("parity_relative_tolerance", 0.0),
    ("parity_relative_tolerance", np.inf),
])
def test_invalid_numeric_policy_is_rejected(name, value):
    u, prior, cross, _, s = model()
    with pytest.raises(ValueError):
        compress(u, prior, cross, s, **{name: value})


def test_inconsistent_covariance_blocks_fail_without_ridge():
    u, prior, cross, _, s = model()
    with pytest.raises(ValueError, match="full query posterior"):
        compress(u, prior, cross * 100, s)
    with pytest.raises(ValueError, match="innovation remainder"):
        compress(u * 100, prior, cross, s)
    with pytest.raises(ValueError, match="prior_query_covariance"):
        compress(u, -prior, cross, s)
    bad = prior.copy()
    bad[0, 1] += 1
    with pytest.raises(ValueError, match="symmetric"):
        compress(u, bad, cross, s)


@pytest.mark.parametrize("kind", ["nan", "complex"])
def test_invalid_factor_is_rejected(kind):
    u, prior, cross, _, s = model()
    if kind == "nan":
        u[0, 0] = np.nan
    else:
        u = u.astype(complex) + 1j
    with pytest.raises(ValueError):
        compress(u, prior, cross, s)


def test_solver_dimensions_and_return_shape_are_checked():
    u, prior, cross, _, s = model()
    with pytest.raises(ValueError, match="dimensions"):
        compress_shared_factor_for_posterior(
            u.reshape(-1, 3, 7), prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=DenseInnovation(np.eye(3)),
        )
    solver = DenseInnovation(s)
    solver.solve = lambda rhs: np.zeros((3, 3))
    with pytest.raises(ValueError, match="incorrect shape"):
        compress_shared_factor_for_posterior(
            u.reshape(-1, 3, 7), prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=solver,
        )


def test_changing_the_query_invalidates_the_original_compression():
    u, prior, cross, a, s = model(rank=7, qdim=3)
    result = compress(u, prior[:1, :1], cross[:1], s)
    reduced = result.compressed_factor_m.reshape(36, -1)
    _, full_covariance = posterior(prior[1:, 1:], cross[1:], s)
    _, changed_covariance = posterior(
        prior[1:, 1:], cross[1:], a + reduced @ reduced.T
    )
    assert np.linalg.norm(full_covariance - changed_covariance) > 1e-3


def test_nonzero_discarded_response_has_strictly_positive_covariance_loss():
    u, prior, cross, a, s = model(rank=7, qdim=3)
    v = np.eye(7)[:, :2]
    reduced = u @ v
    _, full_covariance = posterior(prior, cross, s)
    _, reduced_covariance = posterior(prior, cross, a + reduced @ reduced.T)
    loss = full_covariance - reduced_covariance
    assert np.linalg.eigvalsh(loss).min() > 1e-4
