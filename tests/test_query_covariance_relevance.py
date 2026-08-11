from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.query_covariance_relevance import (
    QUERY_COVARIANCE_RELEVANCE_CLAIM_BOUNDARY,
    QUERY_COVARIANCE_RELEVANCE_SCHEMA,
    QUERY_COVARIANCE_RELEVANCE_VERSION,
    QueryCovarianceProjectionV1,
    project_joint_covariance_to_query,
)


def test_scalar_query_has_analytic_shared_fraction() -> None:
    jacobian = np.array([[1.0, 0.0, 0.0]])
    local = np.diag([1.0, 2.0, 3.0])[None, ...]
    factor = np.array([[[2.0], [0.0], [0.0]]])

    result = project_joint_covariance_to_query(jacobian, local, factor)

    np.testing.assert_allclose(result.conditional_covariance, [[1.0]])
    np.testing.assert_allclose(result.shared_query_factor, [[2.0]])
    np.testing.assert_allclose(result.shared_covariance, [[4.0]])
    np.testing.assert_allclose(result.total_covariance, [[5.0]])
    assert result.shared_trace_fraction == pytest.approx(0.8)
    assert result.shared_frobenius_fraction == pytest.approx(0.8)
    assert result.coordinate_shared_fractions == pytest.approx((0.8,))
    assert result.minimum_directional_shared_fraction == pytest.approx(0.8)
    assert result.mean_directional_shared_fraction == pytest.approx(0.8)
    assert result.maximum_directional_shared_fraction == pytest.approx(0.8)


def test_multirow_query_retains_shared_cross_coordinate_covariance() -> None:
    jacobian = np.zeros((2, 2, 3), dtype=np.float64)
    jacobian[0, 0, 0] = 1.0
    jacobian[1, 1, 0] = 1.0
    local = np.repeat(np.eye(3)[None, ...], 2, axis=0)
    factor = np.zeros((2, 3, 1), dtype=np.float64)
    factor[:, 0, 0] = 1.0

    result = project_joint_covariance_to_query(jacobian, local, factor)

    np.testing.assert_allclose(result.conditional_covariance, np.eye(2))
    np.testing.assert_allclose(result.shared_covariance, np.ones((2, 2)))
    np.testing.assert_allclose(result.total_covariance, [[2.0, 1.0], [1.0, 2.0]])
    assert result.shared_trace_fraction == pytest.approx(0.5)
    assert result.shared_effective_rank == 1
    assert result.total_effective_rank == 2
    assert result.minimum_directional_shared_fraction == pytest.approx(0.0)
    assert result.mean_directional_shared_fraction == pytest.approx(1.0 / 3.0)
    assert result.maximum_directional_shared_fraction == pytest.approx(2.0 / 3.0)


def test_projection_matches_dense_covariance_oracle() -> None:
    generator = np.random.default_rng(17)
    sample_count = 5
    query_dimension = 3
    rank = 4
    jacobian = generator.normal(size=(query_dimension, sample_count, 3))
    roots = generator.normal(size=(sample_count, 3, 3))
    local = np.einsum("nij,nkj->nik", roots, roots) + 0.1 * np.eye(3)[None, ...]
    factor = generator.normal(size=(sample_count, 3, rank))

    result = project_joint_covariance_to_query(jacobian, local, factor)

    dense_jacobian = jacobian.reshape(query_dimension, 3 * sample_count)
    dense_local = np.zeros((3 * sample_count, 3 * sample_count), dtype=np.float64)
    for index in range(sample_count):
        start = 3 * index
        dense_local[start : start + 3, start : start + 3] = local[index]
    dense_factor = factor.reshape(3 * sample_count, rank)
    expected_conditional = dense_jacobian @ dense_local @ dense_jacobian.T
    expected_shared = dense_jacobian @ dense_factor @ dense_factor.T @ dense_jacobian.T

    np.testing.assert_allclose(result.conditional_covariance, expected_conditional)
    np.testing.assert_allclose(result.shared_covariance, expected_shared)
    np.testing.assert_allclose(
        result.total_covariance,
        expected_conditional + expected_shared,
    )


def test_low_rank_basis_rotation_preserves_query_relevance() -> None:
    generator = np.random.default_rng(23)
    jacobian = generator.normal(size=(3, 4, 3))
    local = np.repeat(np.eye(3)[None, ...], 4, axis=0)
    factor = generator.normal(size=(4, 3, 3))
    orthogonal, _ = np.linalg.qr(generator.normal(size=(3, 3)))

    original = project_joint_covariance_to_query(jacobian, local, factor)
    rotated = project_joint_covariance_to_query(
        jacobian,
        local,
        np.einsum("nir,rs->nis", factor, orthogonal),
    )

    np.testing.assert_allclose(original.shared_covariance, rotated.shared_covariance)
    np.testing.assert_allclose(original.total_covariance, rotated.total_covariance)
    assert original.shared_trace_fraction == pytest.approx(rotated.shared_trace_fraction)
    assert original.maximum_directional_shared_fraction == pytest.approx(
        rotated.maximum_directional_shared_fraction
    )


def test_query_scaling_preserves_fractional_relevance() -> None:
    generator = np.random.default_rng(29)
    jacobian = generator.normal(size=(2, 3, 3))
    local = np.repeat(np.eye(3)[None, ...], 3, axis=0)
    factor = generator.normal(size=(3, 3, 2))

    original = project_joint_covariance_to_query(jacobian, local, factor)
    scaled = project_joint_covariance_to_query(7.0 * jacobian, local, factor)

    assert original.shared_trace_fraction == pytest.approx(scaled.shared_trace_fraction)
    assert original.shared_frobenius_fraction == pytest.approx(
        scaled.shared_frobenius_fraction
    )
    assert original.maximum_directional_shared_fraction == pytest.approx(
        scaled.maximum_directional_shared_fraction
    )


def test_zero_shared_factor_reports_zero_relevance() -> None:
    jacobian = np.eye(3)[None, ...].reshape(3, 1, 3)
    local = np.eye(3)[None, ...]
    factor = np.empty((1, 3, 0), dtype=np.float64)

    result = project_joint_covariance_to_query(jacobian, local, factor)

    np.testing.assert_array_equal(result.shared_covariance, np.zeros((3, 3)))
    assert result.shared_trace_fraction == 0.0
    assert result.shared_frobenius_fraction == 0.0
    assert result.coordinate_shared_fractions == (0.0, 0.0, 0.0)
    assert result.shared_effective_rank == 0
    assert result.minimum_directional_shared_fraction == 0.0
    assert result.maximum_directional_shared_fraction == 0.0


def test_zero_query_variance_is_explicitly_unidentified() -> None:
    jacobian = np.zeros((2, 1, 3), dtype=np.float64)
    local = np.zeros((1, 3, 3), dtype=np.float64)
    factor = np.zeros((1, 3, 1), dtype=np.float64)

    result = project_joint_covariance_to_query(jacobian, local, factor)

    assert result.total_trace == 0.0
    assert result.shared_trace_fraction is None
    assert result.shared_frobenius_fraction is None
    assert result.coordinate_shared_fractions == (None, None)
    assert result.active_query_dimension == 0
    assert result.minimum_directional_shared_fraction is None
    assert result.maximum_directional_shared_fraction is None


def test_projection_defensively_owns_immutable_arrays() -> None:
    jacobian = np.array([[1.0, 0.0, 0.0]])
    local = np.eye(3)[None, ...]
    factor = np.ones((1, 3, 1), dtype=np.float64)

    result = project_joint_covariance_to_query(jacobian, local, factor)
    local[...] = 9.0
    factor[...] = 9.0

    np.testing.assert_allclose(result.conditional_covariance, [[1.0]])
    np.testing.assert_allclose(result.shared_covariance, [[1.0]])
    for value in (
        result.conditional_covariance,
        result.shared_query_factor,
        result.shared_covariance,
        result.total_covariance,
    ):
        assert not value.flags.writeable


def test_summary_is_compact_and_states_the_scientific_boundary() -> None:
    result = project_joint_covariance_to_query(
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3)[None, ...],
        np.empty((1, 3, 0), dtype=np.float64),
    )

    summary = result.summary()

    assert summary["schema"] == QUERY_COVARIANCE_RELEVANCE_SCHEMA
    assert summary["version"] == QUERY_COVARIANCE_RELEVANCE_VERSION
    assert summary["claim_boundary"] == QUERY_COVARIANCE_RELEVANCE_CLAIM_BOUNDARY
    assert "conditional_covariance" not in summary
    assert "total_covariance" not in summary


def test_direct_construction_recomputes_all_derived_fields() -> None:
    projected = project_joint_covariance_to_query(
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3)[None, ...],
        np.ones((1, 3, 1), dtype=np.float64),
    )

    reconstructed = QueryCovarianceProjectionV1(
        conditional_covariance=projected.conditional_covariance,
        shared_query_factor=projected.shared_query_factor,
        observation_count=projected.observation_count,
    )

    assert reconstructed.summary() == projected.summary()
    np.testing.assert_array_equal(
        reconstructed.total_covariance,
        projected.total_covariance,
    )


def test_constructor_rejects_coercive_or_invalid_inputs() -> None:
    result = project_joint_covariance_to_query(
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3)[None, ...],
        np.ones((1, 3, 1), dtype=np.float64),
    )

    with pytest.raises(TypeError, match="observation_count"):
        replace(result, observation_count=True)
    with pytest.raises(ValueError, match="relative_rank_tolerance"):
        replace(result, relative_rank_tolerance=np.nan)

    asymmetric = result.conditional_covariance.copy()
    asymmetric[0, 0] = np.nan
    with pytest.raises(ValueError, match="conditional_covariance must be finite"):
        replace(result, conditional_covariance=asymmetric)


@pytest.mark.parametrize(
    ("jacobian", "local", "factor", "message"),
    [
        (np.ones((2, 2)), np.eye(3)[None, ...], np.ones((1, 3, 1)), "query_jacobian"),
        (
            np.ones((1, 1, 3)),
            np.ones((1, 2, 2)),
            np.ones((1, 3, 1)),
            "local_covariance_m2",
        ),
        (
            np.ones((1, 1, 3)),
            np.eye(3)[None, ...],
            np.ones((2, 3, 1)),
            "low_rank_factor_m",
        ),
    ],
)
def test_shape_mismatches_fail_closed(
    jacobian: np.ndarray,
    local: np.ndarray,
    factor: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_joint_covariance_to_query(jacobian, local, factor)


def test_invalid_covariance_and_nonfinite_inputs_fail_closed() -> None:
    jacobian = np.ones((1, 1, 3), dtype=np.float64)
    factor = np.ones((1, 3, 1), dtype=np.float64)

    asymmetric = np.eye(3)[None, ...]
    asymmetric[0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetric"):
        project_joint_covariance_to_query(jacobian, asymmetric, factor)

    indefinite = np.diag([1.0, 1.0, -0.1])[None, ...]
    with pytest.raises(ValueError, match="positive semidefinite"):
        project_joint_covariance_to_query(jacobian, indefinite, factor)

    nonfinite_jacobian = jacobian.copy()
    nonfinite_jacobian[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="query_jacobian must be finite"):
        project_joint_covariance_to_query(nonfinite_jacobian, np.eye(3)[None, ...], factor)

    nonfinite_factor = factor.copy()
    nonfinite_factor[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="low_rank_factor_m must be finite"):
        project_joint_covariance_to_query(jacobian, np.eye(3)[None, ...], nonfinite_factor)


@pytest.mark.parametrize("value", [True, -0.1, 1.0, np.nan, "1e-10"])
def test_invalid_rank_tolerance_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="relative_rank_tolerance"):
        project_joint_covariance_to_query(
            np.ones((1, 1, 3)),
            np.eye(3)[None, ...],
            np.ones((1, 3, 1)),
            relative_rank_tolerance=value,  # type: ignore[arg-type]
        )
