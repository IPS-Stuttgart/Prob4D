"""Tests for dense-to-factorized covariance endpoint conversion."""

from __future__ import annotations

import numpy as np
import pytest

from prob4d.factorized_dependence_adapters import (
    factorized_from_covariance_endpoints,
)


def _compatible_endpoints(
    *, groups: int = 5, dimension: int = 3, rank: int = 4, seed: int = 43
):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(groups, dimension, rank))
    factors[:, :, :dimension] += 2.0 * np.eye(dimension)[None, :, :]
    flat = factors.reshape(groups * dimension, rank)
    shared = flat @ flat.T
    local = np.zeros_like(shared)
    for group in range(groups):
        start = group * dimension
        stop = start + dimension
        local[start:stop, start:stop] = shared[start:stop, start:stop]
    return local, shared


def test_compatible_dense_endpoints_recover_covariance_and_dense_gaussian_algebra():
    local, shared = _compatible_endpoints()
    strength = 0.73
    model = factorized_from_covariance_endpoints(local, shared, 3, strength)
    expected = (1.0 - strength) * local + strength * shared
    np.testing.assert_allclose(
        model.dense_covariance(), expected, atol=2.0e-12, rtol=2.0e-12
    )
    residual = np.linspace(-0.4, 0.7, expected.shape[0])
    np.testing.assert_allclose(
        model.solve(residual),
        np.linalg.solve(expected, residual),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    sign, logdet = np.linalg.slogdet(expected)
    assert sign > 0.0
    np.testing.assert_allclose(model.log_determinant(), logdet, atol=2.0e-10)


def test_adapter_recovers_numerical_shared_rank():
    local, shared = _compatible_endpoints(groups=7, dimension=2, rank=3)
    model = factorized_from_covariance_endpoints(local, shared, 2, 0.85)
    assert model.latent_rank == 3
    assert model.group_count == 7
    assert model.block_dimension == 2


def test_scalar_diagonal_agreement_is_not_enough_for_block_preservation():
    local, shared = _compatible_endpoints(groups=2, dimension=2, rank=3)
    changed = shared.copy()
    changed[0, 1] += 0.2
    changed[1, 0] += 0.2
    assert np.array_equal(np.diag(changed), np.diag(local))
    with pytest.raises(ValueError, match="complete marginal block"):
        factorized_from_covariance_endpoints(local, changed, 2, 0.5)


def test_local_cross_group_dependence_is_rejected():
    local, shared = _compatible_endpoints(groups=3, dimension=2, rank=3)
    local[0, 2] = local[2, 0] = 0.01
    with pytest.raises(ValueError, match="block diagonal"):
        factorized_from_covariance_endpoints(local, shared, 2, 0.5)


def test_material_rank_truncation_is_rejected():
    local, shared = _compatible_endpoints(groups=3, dimension=2, rank=3)
    with pytest.raises(ValueError, match="rank truncation"):
        factorized_from_covariance_endpoints(
            local,
            shared,
            2,
            0.5,
            rank_rtol=0.99,
            rank_atol=0.0,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("shape", "matching shape"),
        ("dimension", "divisible"),
        ("local_nonsymmetric", "symmetric"),
        ("shared_nonsymmetric", "symmetric"),
        ("local_negative", "positive semidefinite"),
        ("shared_negative", "positive semidefinite"),
        ("zero_rank", "zero numerical rank"),
    ],
)
def test_invalid_dense_endpoints_fail_closed(mutation, match):
    local, shared = _compatible_endpoints(groups=3, dimension=2, rank=3)
    block_dimension = 2
    if mutation == "shape":
        shared = shared[:-1, :-1]
    elif mutation == "dimension":
        block_dimension = 4
    elif mutation == "local_nonsymmetric":
        local[0, 1] += 0.5
    elif mutation == "shared_nonsymmetric":
        shared[0, 1] += 0.5
    elif mutation == "local_negative":
        local[0, 0] = -10.0
    elif mutation == "shared_negative":
        shared[0, 0] = -10.0
    elif mutation == "zero_rank":
        local[:] = 0.0
        shared[:] = 0.0
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match=match):
        factorized_from_covariance_endpoints(
            local,
            shared,
            block_dimension,
            1.0 if mutation == "zero_rank" else 0.5,
        )


@pytest.mark.parametrize("block_dimension", [0, -1, 1.5, True])
def test_invalid_block_dimension_rejected(block_dimension):
    local, shared = _compatible_endpoints(groups=2, dimension=2, rank=3)
    with pytest.raises(ValueError, match="positive integer"):
        factorized_from_covariance_endpoints(
            local,
            shared,
            block_dimension,
            0.5,
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("covariance_rtol", -1.0),
        ("covariance_atol", np.nan),
        ("rank_rtol", np.inf),
        ("rank_atol", -1.0),
    ],
)
def test_invalid_tolerances_rejected(name, value):
    local, shared = _compatible_endpoints(groups=2, dimension=2, rank=3)
    keywords = {name: value}
    with pytest.raises(ValueError, match=name):
        factorized_from_covariance_endpoints(
            local,
            shared,
            2,
            0.5,
            **keywords,
        )
