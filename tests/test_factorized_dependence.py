"""Tests for block-local plus shared-low-rank Gaussian dependence."""

from __future__ import annotations

import numpy as np
import pytest

from prob4d.factorized_dependence import BlockSharedGaussianCovariance


def _model(
    *,
    groups: int = 6,
    dimension: int = 3,
    rank: int = 5,
    strength: float = 0.85,
    seed: int = 20260901,
) -> BlockSharedGaussianCovariance:
    rng = np.random.default_rng(seed)
    factors = rng.normal(scale=0.4, size=(groups, dimension, rank))
    factors[:, :, :dimension] += 1.5 * np.eye(dimension)[None, :, :]
    blocks = np.einsum("idr,ier->ide", factors, factors)
    return BlockSharedGaussianCovariance(blocks, factors, strength)


def test_dense_covariance_preserves_blocks_and_scales_cross_dependence():
    model = _model(groups=4, dimension=2, rank=3, strength=0.35)
    dense = model.dense_covariance()
    width = model.block_dimension
    for first in range(model.group_count):
        first_slice = slice(first * width, (first + 1) * width)
        np.testing.assert_allclose(
            dense[first_slice, first_slice],
            model.marginal_blocks[first],
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        for second in range(model.group_count):
            second_slice = slice(second * width, (second + 1) * width)
            np.testing.assert_allclose(
                dense[first_slice, second_slice],
                model.cross_covariance(first, second),
                atol=1.0e-14,
                rtol=1.0e-14,
            )


def test_factorized_solve_logdet_quadratic_and_nll_match_dense_algebra():
    model = _model()
    rng = np.random.default_rng(71)
    right = rng.normal(size=(model.dimension, 4))
    residual = rng.normal(size=model.dimension)
    dense = model.dense_covariance()

    np.testing.assert_allclose(
        model.solve(right),
        np.linalg.solve(dense, right),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(
        model.solve(residual),
        np.linalg.solve(dense, residual),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    sign, dense_logdet = np.linalg.slogdet(dense)
    assert sign > 0.0
    np.testing.assert_allclose(
        model.log_determinant(), dense_logdet, atol=2.0e-11, rtol=2.0e-11
    )
    dense_quadratic = float(residual @ np.linalg.solve(dense, residual))
    np.testing.assert_allclose(
        model.quadratic_form(residual),
        dense_quadratic,
        atol=2.0e-10,
        rtol=2.0e-11,
    )
    dense_nll = 0.5 * (
        model.dimension * np.log(2.0 * np.pi) + dense_logdet + dense_quadratic
    )
    np.testing.assert_allclose(model.gaussian_nll(residual), dense_nll, atol=2.0e-10)
    np.testing.assert_allclose(
        model.gaussian_nll(residual, normalized=True),
        dense_nll / model.dimension,
        atol=2.0e-11,
    )


def test_zero_strength_is_block_independent_and_exactly_solvable():
    model = _model(strength=0.0)
    dense = model.dense_covariance()
    width = model.block_dimension
    for first in range(model.group_count):
        for second in range(model.group_count):
            if first == second:
                continue
            first_slice = slice(first * width, (first + 1) * width)
            second_slice = slice(second * width, (second + 1) * width)
            np.testing.assert_array_equal(dense[first_slice, second_slice], 0.0)
    residual = np.arange(model.dimension, dtype=np.float64) / model.dimension
    np.testing.assert_allclose(
        model.solve(residual),
        np.linalg.solve(dense, residual),
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_full_shared_endpoint_retains_marginals_but_rejects_precision_operations():
    model = _model(strength=1.0)
    factor = model.shared_factors.reshape(model.dimension, model.latent_rank)
    np.testing.assert_allclose(model.dense_covariance(), factor @ factor.T, atol=1.0e-14)
    with pytest.raises(np.linalg.LinAlgError, match="strength=1"):
        model.log_determinant()
    with pytest.raises(np.linalg.LinAlgError, match="strength=1"):
        model.solve(np.zeros(model.dimension))


def test_sampling_recovers_the_dense_covariance():
    model = _model(groups=3, dimension=2, rank=4, strength=0.65, seed=18)
    samples = model.sample(np.random.default_rng(9182), 60_000).reshape(60_000, -1)
    empirical_mean = np.mean(samples, axis=0)
    empirical_covariance = np.cov(samples, rowvar=False, bias=True)
    dense = model.dense_covariance()
    scale = float(np.sqrt(np.max(np.diag(dense))))
    assert float(np.max(np.abs(empirical_mean))) < 0.025 * scale
    relative_frobenius_error = float(
        np.linalg.norm(empirical_covariance - dense) / np.linalg.norm(dense)
    )
    assert relative_frobenius_error < 0.025


def test_storage_report_for_paper_scale_is_more_than_six_hundred_fold_smaller():
    groups, dimension, rank = 2048, 3, 7
    factors = np.zeros((groups, dimension, rank), dtype=np.float64)
    factors[:, :, :dimension] = np.eye(dimension)[None, :, :]
    blocks = np.einsum("idr,ier->ide", factors, factors)
    model = BlockSharedGaussianCovariance(blocks, factors, 0.85)
    assert model.storage_bytes == 491_520
    assert model.dense_storage_bytes == 301_989_888
    np.testing.assert_allclose(model.storage_reduction_factor, 614.4)


def test_inputs_are_copied_and_outputs_are_read_only():
    factors = np.zeros((2, 2, 2), dtype=np.float64)
    factors[:, :, :] = np.eye(2)[None, :, :]
    blocks = np.einsum("idr,ier->ide", factors, factors)
    model = BlockSharedGaussianCovariance(blocks, factors, 0.5)
    factors[:] = 12.0
    blocks[:] = 12.0
    np.testing.assert_array_equal(model.marginal_blocks, np.tile(np.eye(2), (2, 1, 1)))
    np.testing.assert_array_equal(model.shared_factors, np.tile(np.eye(2), (2, 1, 1)))
    for value in (
        model.marginal_blocks,
        model.shared_factors,
        model.marginal_covariances,
        model.dense_covariance(),
        model.solve(np.ones(model.dimension)),
        model.sample(np.random.default_rng(1), 2),
    ):
        assert not value.flags.writeable


@pytest.mark.parametrize("strength", [-0.1, 1.1, np.nan, np.inf, True])
def test_invalid_strength_rejected(strength):
    factors = np.tile(np.eye(2)[None, :, :], (2, 1, 1))
    blocks = np.einsum("idr,ier->ide", factors, factors)
    with pytest.raises(ValueError):
        BlockSharedGaussianCovariance(blocks, factors, strength)


def test_mismatched_marginal_blocks_rejected():
    factors = np.tile(np.eye(2)[None, :, :], (2, 1, 1))
    blocks = np.einsum("idr,ier->ide", factors, factors)
    blocks[1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="reproduce every marginal"):
        BlockSharedGaussianCovariance(blocks, factors, 0.5)


def test_nonsymmetric_negative_and_singular_local_blocks_rejected():
    factors = np.tile(np.eye(2)[None, :, :], (2, 1, 1))
    blocks = np.einsum("idr,ier->ide", factors, factors)

    nonsymmetric = blocks.copy()
    nonsymmetric[0, 0, 1] = 0.2
    with pytest.raises(ValueError, match="symmetric"):
        BlockSharedGaussianCovariance(nonsymmetric, factors, 0.5)

    singular_factors = factors.copy()
    singular_factors[:, 1, :] = 0.0
    singular = np.einsum("idr,ier->ide", singular_factors, singular_factors)
    with pytest.raises(ValueError, match="positive definite"):
        BlockSharedGaussianCovariance(singular, singular_factors, 0.5)

    negative = blocks.copy()
    negative[0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        BlockSharedGaussianCovariance(negative, factors, 1.0)


def test_invalid_shapes_rhs_indices_and_sampling_rejected():
    model = _model(groups=2, dimension=2, rank=3)
    with pytest.raises(ValueError):
        BlockSharedGaussianCovariance(np.eye(2), np.ones((2, 2, 3)), 0.5)
    with pytest.raises(ValueError):
        BlockSharedGaussianCovariance(np.tile(np.eye(2), (2, 1, 1)), np.ones((2, 3, 2)), 0.5)
    with pytest.raises(ValueError):
        model.solve(np.ones(model.dimension + 1))
    with pytest.raises(ValueError):
        model.solve(np.ones((model.dimension + 1, 2)))
    with pytest.raises(ValueError):
        model.quadratic_form(np.ones(model.dimension + 1))
    with pytest.raises(IndexError):
        model.cross_covariance(-1, 0)
    with pytest.raises(IndexError):
        model.cross_covariance(0, model.group_count)
    with pytest.raises(TypeError):
        model.sample("not-a-generator", 2)
    for count in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            model.sample(np.random.default_rng(1), count)
