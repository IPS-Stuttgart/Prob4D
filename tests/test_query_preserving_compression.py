from __future__ import annotations

import numpy as np
import pytest

from prob4d.query_preserving_compression import (
    QueryPreservingCompressionPolicyV1,
    compress_shared_factor_for_queries,
)


def _two_mode_factor() -> np.ndarray:
    factor = np.zeros((2, 3, 2), dtype=np.float64)
    factor[0, 0, 0] = 10.0
    factor[1, 1, 1] = 1.0
    return factor


def test_query_weight_preserves_low_observation_energy_direction() -> None:
    factor = _two_mode_factor()
    query = np.zeros((2, 3), dtype=np.float64)
    query[1, 1] = 1.0
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.0,
        maximum_query_trace_loss_fraction=0.0,
        maximum_query_spectral_loss_fraction=0.0,
        maximum_rank=1,
        observation_weight=0.01,
        query_weights={"endpoint": 1.0},
    )

    result = compress_shared_factor_for_queries(
        factor,
        {"endpoint": query},
        policy=policy,
    )

    assert result.compression_applied is True
    assert result.retained_rank == 1
    projected = np.einsum(
        "ni,nir->r",
        query,
        result.compressed_factor_m,
        optimize=True,
    )
    original = np.einsum("ni,nir->r", query, factor, optimize=True)
    assert np.allclose(projected @ projected, original @ original)
    assert result.observation_trace_fraction == pytest.approx(1.0 / 101.0)
    assert result.query_diagnostics[0].trace_loss_fraction == 0.0
    assert result.query_diagnostics[0].spectral_loss_fraction == 0.0


def test_joint_observation_and_query_limits_require_both_modes() -> None:
    factor = _two_mode_factor()
    query = np.zeros((2, 3), dtype=np.float64)
    query[1, 1] = 1.0
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.99,
        maximum_query_trace_loss_fraction=0.0,
        maximum_query_spectral_loss_fraction=0.0,
        observation_weight=1.0,
        query_weights={"endpoint": 1.0},
    )

    result = compress_shared_factor_for_queries(
        factor,
        {"endpoint": query},
        policy=policy,
    )

    assert result.compression_applied is False
    assert result.fallback_reason == "full-rank-required"
    assert result.retained_rank == 2
    assert np.array_equal(result.compressed_factor_m, factor)
    assert np.array_equal(result.latent_projection, np.eye(2))


def test_rank_cap_fails_closed_to_exact_full_factor() -> None:
    factor = _two_mode_factor()
    queries = {
        "first": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "second": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=1.0,
        maximum_query_trace_loss_fraction=0.0,
        maximum_query_spectral_loss_fraction=0.0,
        maximum_rank=1,
    )

    result = compress_shared_factor_for_queries(factor, queries, policy=policy)

    assert result.compression_applied is False
    assert result.fallback_reason == "no-admissible-reduction-within-rank-cap"
    assert np.array_equal(result.compressed_factor_m, factor)
    full = factor.reshape(-1, 2) @ factor.reshape(-1, 2).T
    retained = result.compressed_factor_m.reshape(-1, 2)
    assert np.array_equal(retained @ retained.T, full)


def test_repeated_score_eigenspace_is_not_split() -> None:
    factor = np.zeros((1, 3, 2), dtype=np.float64)
    factor[0, 0, 0] = 1.0
    factor[0, 1, 1] = 1.0
    queries = {
        "xy": np.array(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        )
    }
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.5,
        maximum_query_trace_loss_fraction=0.5,
        maximum_query_spectral_loss_fraction=1.0,
        maximum_rank=1,
    )

    result = compress_shared_factor_for_queries(factor, queries, policy=policy)

    assert result.compression_applied is False
    assert result.fallback_reason == "no-admissible-reduction-within-rank-cap"
    assert result.retained_rank == 2


def test_zero_shared_covariance_reduces_safely_to_rank_zero() -> None:
    factor = np.zeros((3, 3, 4), dtype=np.float64)
    query = np.ones((3, 3), dtype=np.float64)
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=1.0,
        maximum_query_trace_loss_fraction=0.0,
        maximum_query_spectral_loss_fraction=0.0,
    )

    result = compress_shared_factor_for_queries(
        factor,
        {"zero": query},
        policy=policy,
    )

    assert result.compression_applied is True
    assert result.retained_rank == 0
    assert result.compressed_factor_m.shape == (3, 3, 0)
    assert result.observation_trace_fraction == 1.0


def test_result_owns_immutable_copies() -> None:
    factor = _two_mode_factor()
    query = np.zeros((2, 3), dtype=np.float64)
    query[0, 0] = 1.0
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.9,
        maximum_query_trace_loss_fraction=0.0,
        maximum_query_spectral_loss_fraction=0.0,
        maximum_rank=1,
    )
    result = compress_shared_factor_for_queries(
        factor,
        {"first": query},
        policy=policy,
    )
    factor[...] = 123.0
    query[...] = 123.0

    assert result.compressed_factor_m.flags.writeable is False
    assert result.latent_projection.flags.writeable is False
    assert result.score_eigenvalues.flags.writeable is False
    with pytest.raises(ValueError):
        result.compressed_factor_m[0, 0, 0] = 0.0


def test_validation_rejects_ambiguous_or_invalid_inputs() -> None:
    factor = _two_mode_factor()
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.9,
        maximum_query_trace_loss_fraction=0.1,
        maximum_query_spectral_loss_fraction=0.1,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        compress_shared_factor_for_queries(factor, {}, policy=policy)
    with pytest.raises(ValueError, match="must name exactly"):
        compress_shared_factor_for_queries(
            factor,
            {"q": np.ones((2, 3))},
            policy=QueryPreservingCompressionPolicyV1(
                minimum_observation_trace_fraction=0.9,
                maximum_query_trace_loss_fraction=0.1,
                maximum_query_spectral_loss_fraction=0.1,
                query_weights={"other": 1.0},
            ),
        )
    malformed = factor.copy()
    malformed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        compress_shared_factor_for_queries(
            malformed,
            {"q": np.ones((2, 3))},
            policy=policy,
        )


def test_summary_is_deterministic_and_json_compatible() -> None:
    factor = _two_mode_factor()
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.0,
        maximum_query_trace_loss_fraction=0.0,
        maximum_query_spectral_loss_fraction=0.0,
        maximum_rank=1,
        observation_weight=0.01,
        query_weights={"q": 1.0},
    )
    result = compress_shared_factor_for_queries(
        factor,
        {"q": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
        policy=policy,
    )
    summary = result.summary()
    assert summary["schema"] == "prob4d.query-preserving-compression"
    assert summary["version"] == 1
    assert summary["retained_rank"] == 1
    assert summary["query_diagnostics"][0]["name"] == "q"
    assert "claim_boundary" in summary


def test_policy_copies_and_freezes_query_weights() -> None:
    weights: dict[str, object] = {"q": 2.0}
    policy = QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.9,
        maximum_query_trace_loss_fraction=0.1,
        maximum_query_spectral_loss_fraction=0.1,
        query_weights=weights,
    )
    weights["q"] = 99.0
    assert policy.query_weights is not None
    assert policy.query_weights["q"] == 2.0
    with pytest.raises(TypeError):
        policy.query_weights["q"] = 3.0  # type: ignore[index]
