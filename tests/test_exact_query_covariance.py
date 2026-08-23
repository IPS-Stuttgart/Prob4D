from __future__ import annotations

import numpy as np

from prob4d.exact_query_covariance import (
    decompose_shared_factor_for_exact_queries,
)


def _factor() -> np.ndarray:
    factor = np.zeros((2, 3, 4), dtype=np.float64)
    factor[0, 0, 0] = 2.0
    factor[0, 1, 1] = 3.0
    factor[1, 0, 2] = 4.0
    factor[1, 1, 3] = 5.0
    return factor


def _queries() -> dict[str, np.ndarray]:
    endpoint_pair = np.zeros((2, 2, 3), dtype=np.float64)
    endpoint_pair[0, 0, 0] = 1.0
    endpoint_pair[1, 1, 0] = 1.0
    endpoint_sum = np.zeros((2, 3), dtype=np.float64)
    endpoint_sum[0, 0] = 1.0
    endpoint_sum[1, 0] = 0.5
    return {
        "endpoint_pair": endpoint_pair,
        "endpoint_sum": endpoint_sum,
    }


def _flattened_factor(factor: np.ndarray) -> np.ndarray:
    return factor.reshape(factor.shape[0] * 3, factor.shape[2])


def _dense_covariance(factor: np.ndarray) -> np.ndarray:
    flattened = _flattened_factor(factor)
    return flattened @ flattened.T


def _query_factor(jacobian: np.ndarray, factor: np.ndarray) -> np.ndarray:
    normalized = jacobian[None] if jacobian.ndim == 2 else jacobian
    return np.einsum("qni,nir->qr", normalized, factor, optimize=True)


def test_minimum_subspace_preserves_registered_query_cross_covariances() -> None:
    factor = _factor()
    queries = _queries()

    result = decompose_shared_factor_for_exact_queries(factor, queries)

    assert result.decomposition_applied is True
    assert result.minimum_exact_query_rank == 2
    assert result.retained_query_rank == 2
    assert result.query_orthogonal_rank == 2
    assert result.strict_query_rank_reduction is True
    reconstructed = (
        _dense_covariance(result.query_coupled_factor_m)
        + _dense_covariance(result.query_orthogonal_factor_m)
    )
    assert np.allclose(reconstructed, _dense_covariance(factor))

    for first_jacobian in queries.values():
        full_first = _query_factor(first_jacobian, factor)
        coupled_first = _query_factor(
            first_jacobian,
            result.query_coupled_factor_m,
        )
        orthogonal_first = _query_factor(
            first_jacobian,
            result.query_orthogonal_factor_m,
        )
        assert np.allclose(orthogonal_first, 0.0, atol=1e-12)
        for second_jacobian in queries.values():
            full_second = _query_factor(second_jacobian, factor)
            coupled_second = _query_factor(
                second_jacobian,
                result.query_coupled_factor_m,
            )
            assert np.allclose(
                full_first @ full_second.T,
                coupled_first @ coupled_second.T,
                atol=1e-12,
                rtol=1e-12,
            )

    assert all(
        diagnostic.relative_covariance_error < 1e-12
        and diagnostic.relative_query_orthogonal_factor_norm < 1e-12
        for diagnostic in result.query_diagnostics
    )


def test_latent_rotation_and_query_reparameterization_preserve_components() -> None:
    factor = _factor()
    query = _queries()["endpoint_pair"]
    reference = decompose_shared_factor_for_exact_queries(
        factor,
        {"endpoint_pair": query},
    )

    generator = np.random.default_rng(20260823)
    rotation, _ = np.linalg.qr(generator.normal(size=(4, 4)))
    rotated_factor = np.einsum("nir,rs->nis", factor, rotation, optimize=True)
    rotated = decompose_shared_factor_for_exact_queries(
        rotated_factor,
        {"endpoint_pair": query},
    )

    reparameterization = np.array([[2.0, -1.0], [1.0, 1.0]])
    transformed_query = np.einsum(
        "ab,bni->ani",
        reparameterization,
        query,
        optimize=True,
    )
    transformed = decompose_shared_factor_for_exact_queries(
        factor,
        {"transformed": transformed_query},
    )

    assert rotated.minimum_exact_query_rank == reference.minimum_exact_query_rank
    assert transformed.minimum_exact_query_rank == reference.minimum_exact_query_rank
    assert np.allclose(
        _dense_covariance(rotated.query_coupled_factor_m),
        _dense_covariance(reference.query_coupled_factor_m),
        atol=1e-11,
        rtol=1e-11,
    )
    assert np.allclose(
        _dense_covariance(rotated.query_orthogonal_factor_m),
        _dense_covariance(reference.query_orthogonal_factor_m),
        atol=1e-11,
        rtol=1e-11,
    )
    assert np.allclose(
        _dense_covariance(transformed.query_coupled_factor_m),
        _dense_covariance(reference.query_coupled_factor_m),
        atol=1e-11,
        rtol=1e-11,
    )
    assert np.allclose(
        _dense_covariance(transformed.query_orthogonal_factor_m),
        _dense_covariance(reference.query_orthogonal_factor_m),
        atol=1e-11,
        rtol=1e-11,
    )


def test_rank_cap_fails_closed_to_the_untouched_full_factor() -> None:
    factor = _factor()
    result = decompose_shared_factor_for_exact_queries(
        factor,
        _queries(),
        maximum_query_rank=1,
    )

    assert result.decomposition_applied is False
    assert result.fallback_reason == "exact-query-rank-exceeds-cap"
    assert result.minimum_exact_query_rank == 2
    assert result.retained_query_rank == 4
    assert result.query_orthogonal_rank == 0
    assert np.array_equal(result.query_coupled_factor_m, factor)
    assert np.array_equal(result.query_coupled_projection, np.eye(4))
    assert result.query_orthogonal_factor_m.shape == (2, 3, 0)
    assert np.array_equal(
        _dense_covariance(result.query_coupled_factor_m),
        _dense_covariance(factor),
    )


def test_zero_latent_rank_is_supported_without_ambiguous_reshape() -> None:
    factor = np.empty((3, 3, 0), dtype=np.float64)
    query = np.ones((3, 3), dtype=np.float64)

    result = decompose_shared_factor_for_exact_queries(
        factor,
        {"zero": query},
        maximum_query_rank=0,
    )

    assert result.decomposition_applied is True
    assert result.minimum_exact_query_rank == 0
    assert result.retained_query_rank == 0
    assert result.query_orthogonal_rank == 0
    assert result.query_coupled_factor_m.shape == (3, 3, 0)
    assert result.query_orthogonal_factor_m.shape == (3, 3, 0)
    assert result.query_coupled_projection.shape == (0, 0)
    assert result.query_orthogonal_projection.shape == (0, 0)
    assert result.query_diagnostics[0].relative_covariance_error == 0.0
    assert result.query_diagnostics[0].relative_query_orthogonal_factor_norm == 0.0
    assert result.query_coupled_factor_m.flags.writeable is False
    assert result.query_orthogonal_factor_m.flags.writeable is False
