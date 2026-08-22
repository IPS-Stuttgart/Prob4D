from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.api.v2 import (
    ObservationGaussianOperator,
    StackedObservationFactors,
    build_observation_gaussian_operator,
)
from prob4d.query_posterior import (
    LowRankUpdatedObservationGaussianOperator,
    augment_observation_gaussian_operator,
    condition_gaussian_query,
)


def _base_operator() -> tuple[ObservationGaussianOperator, np.ndarray]:
    conditional = np.asarray(
        [
            np.diag([0.3, 0.4, 0.5]),
            np.diag([0.6, 0.7, 0.8]),
        ],
        dtype=np.float64,
    )
    stacked = StackedObservationFactors(
        world_mean_m=np.zeros((2, 3), dtype=np.float64),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=conditional,
        gauge_jacobian=np.zeros((2, 3, 7), dtype=np.float64),
        gauge_prior_covariance=np.zeros((7, 7), dtype=np.float64),
        association_probability=np.ones(2, dtype=np.float64),
        prior_reliability=np.ones(2, dtype=np.float64),
        prior_nominal_probability=np.ones(2, dtype=np.float64),
        composite_weight=np.ones(2, dtype=np.float64),
        point_ids=np.asarray([10, 20], dtype=np.int64),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        view_ids=("camera", "camera"),
        factor_ids=("factor-0", "factor-1"),
        correlation_group_ids=("group-0", "group-1"),
        gauge_ids=("gauge-0",),
        causal_frame_stop=2,
    )
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[:3, :3] = conditional[0]
    covariance[3:, 3:] = conditional[1]
    return build_observation_gaussian_operator(stacked), covariance


def test_low_rank_updated_operator_matches_dense_reference() -> None:
    base, base_covariance = _base_operator()
    factor = np.asarray(
        [
            [[0.4, -0.1], [0.2, 0.3], [-0.2, 0.1]],
            [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]],
        ],
        dtype=np.float64,
    )
    retained_factor = factor.copy()
    operator = augment_observation_gaussian_operator(base, factor)
    factor.fill(0.0)

    dense_factor = retained_factor.reshape(6, 2)
    covariance = base_covariance + dense_factor @ dense_factor.T
    right_hand_sides = np.linspace(-0.4, 0.7, 18).reshape(2, 3, 3)
    expected_solve = np.linalg.solve(
        covariance,
        right_hand_sides.reshape(6, 3),
    )
    residual = right_hand_sides[:, :, 0]
    expected_quadratic = float(residual.reshape(-1) @ expected_solve[:, 0])
    expected_nll = 0.5 * (
        6 * math.log(2.0 * math.pi)
        + np.linalg.slogdet(covariance)[1]
        + expected_quadratic
    )

    assert isinstance(operator, LowRankUpdatedObservationGaussianOperator)
    assert operator.base_operator is base
    assert operator.observation_count == 2
    assert operator.dimension == 6
    assert operator.update_rank == 2
    assert operator.factorization_backend.endswith("+low-rank-woodbury-v1")
    assert operator.factor_storage_nbytes > base.factor_storage_nbytes
    assert operator.dense_covariance_nbytes == covariance.nbytes
    np.testing.assert_allclose(operator.low_rank_factor, retained_factor)
    assert not operator.low_rank_factor.flags.writeable
    np.testing.assert_allclose(
        operator.solve(right_hand_sides).reshape(6, 3),
        expected_solve,
        atol=1e-12,
        rtol=1e-12,
    )
    assert operator.log_determinant == pytest.approx(
        np.linalg.slogdet(covariance)[1]
    )
    assert operator.precision_quadratic(residual) == pytest.approx(
        expected_quadratic
    )
    assert operator.gaussian_nll(residual) == pytest.approx(expected_nll)


def test_rank_one_factor_and_operator_validation() -> None:
    base, base_covariance = _base_operator()
    factor = np.linspace(-0.2, 0.3, 6).reshape(2, 3)
    operator = augment_observation_gaussian_operator(base, factor)
    covariance = base_covariance + np.outer(factor.reshape(-1), factor.reshape(-1))
    residual = np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0

    assert operator.update_rank == 1
    np.testing.assert_allclose(
        operator.solve(residual).reshape(-1),
        np.linalg.solve(covariance, residual.reshape(-1)),
    )

    with pytest.raises(TypeError, match="base must be"):
        LowRankUpdatedObservationGaussianOperator(object(), factor)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        augment_observation_gaussian_operator(base, np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        invalid = factor.copy()
        invalid[0, 0] = np.nan
        augment_observation_gaussian_operator(base, invalid)
    with pytest.raises(ValueError, match="shape"):
        operator.solve(np.ones((2, 2)))
    with pytest.raises(TypeError, match="per_dimension"):
        operator.gaussian_nll(residual, per_dimension=1)  # type: ignore[arg-type]


def test_query_posterior_matches_dense_linear_gaussian_update() -> None:
    base, observation_covariance = _base_operator()
    rng = np.random.default_rng(20260823)
    state_factor = rng.normal(size=(4, 4))
    state_covariance = state_factor @ state_factor.T + 0.3 * np.eye(4)
    state_mean = rng.normal(size=4)
    observation_matrix = rng.normal(size=(6, 4))
    query_matrix = rng.normal(size=(2, 4))
    observation = rng.normal(size=6)

    prior_root = np.linalg.cholesky(state_covariance)
    innovation_factor = (observation_matrix @ prior_root).reshape(2, 3, 4)
    innovation_operator = augment_observation_gaussian_operator(
        base,
        innovation_factor,
    )
    innovation = observation - observation_matrix @ state_mean
    innovation_covariance = (
        observation_covariance
        + observation_matrix @ state_covariance @ observation_matrix.T
    )
    cross_covariance = query_matrix @ state_covariance @ observation_matrix.T
    prior_query_mean = query_matrix @ state_mean
    prior_query_covariance = query_matrix @ state_covariance @ query_matrix.T

    result = condition_gaussian_query(
        prior_mean=prior_query_mean,
        prior_covariance=prior_query_covariance,
        innovation=innovation.reshape(2, 3),
        query_observation_cross_covariance=cross_covariance,
        innovation_operator=innovation_operator,
    )

    innovation_response = np.linalg.solve(innovation_covariance, innovation)
    expected_shift = cross_covariance @ innovation_response
    expected_reduction = (
        cross_covariance
        @ np.linalg.solve(innovation_covariance, cross_covariance.T)
    )
    expected_nll = 0.5 * (
        6 * math.log(2.0 * math.pi)
        + np.linalg.slogdet(innovation_covariance)[1]
        + innovation @ innovation_response
    )

    assert result.query_dimension == 2
    assert result.observation_dimension == 6
    np.testing.assert_allclose(result.mean_shift, expected_shift)
    np.testing.assert_allclose(result.covariance_reduction, expected_reduction)
    np.testing.assert_allclose(
        result.posterior_mean,
        prior_query_mean + expected_shift,
    )
    np.testing.assert_allclose(
        result.posterior_covariance,
        prior_query_covariance - expected_reduction,
    )
    assert result.innovation_precision_quadratic == pytest.approx(
        innovation @ innovation_response
    )
    assert result.innovation_log_determinant == pytest.approx(
        np.linalg.slogdet(innovation_covariance)[1]
    )
    assert result.innovation_negative_log_likelihood == pytest.approx(expected_nll)


def test_scalar_query_accepts_row_structured_cross_covariance() -> None:
    base, covariance = _base_operator()
    innovation = np.linspace(-0.2, 0.4, 6).reshape(2, 3)
    cross_covariance = np.linspace(-0.03, 0.04, 6).reshape(2, 3)
    prior_variance = 1.5

    result = condition_gaussian_query(
        prior_mean=np.asarray([0.2]),
        prior_covariance=np.asarray([[prior_variance]]),
        innovation=innovation,
        query_observation_cross_covariance=cross_covariance,
        innovation_operator=base,
    )
    dense_cross = cross_covariance.reshape(1, 6)
    expected_shift = dense_cross @ np.linalg.solve(covariance, innovation.reshape(-1))
    expected_reduction = dense_cross @ np.linalg.solve(covariance, dense_cross.T)

    np.testing.assert_allclose(result.mean_shift, expected_shift)
    np.testing.assert_allclose(result.covariance_reduction, expected_reduction)
    assert not result.prior_mean.flags.writeable
    assert not result.posterior_covariance.flags.writeable
    with pytest.raises(ValueError):
        result.posterior_mean[0] = 0.0


def test_query_conditioning_fails_closed_for_inconsistent_moments() -> None:
    base, _ = _base_operator()
    with pytest.raises(ValueError, match="inconsistent"):
        condition_gaussian_query(
            prior_mean=np.asarray([0.0]),
            prior_covariance=np.asarray([[0.01]]),
            innovation=np.zeros((2, 3)),
            query_observation_cross_covariance=np.ones((2, 3)),
            innovation_operator=base,
        )


def test_query_conditioning_input_validation() -> None:
    base, _ = _base_operator()
    with pytest.raises(TypeError, match="innovation_operator"):
        condition_gaussian_query(
            prior_mean=np.asarray([0.0]),
            prior_covariance=np.asarray([[1.0]]),
            innovation=np.zeros((2, 3)),
            query_observation_cross_covariance=np.zeros((2, 3)),
            innovation_operator=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="prior_covariance shape"):
        condition_gaussian_query(
            prior_mean=np.zeros(2),
            prior_covariance=np.eye(1),
            innovation=np.zeros((2, 3)),
            query_observation_cross_covariance=np.zeros((2, 6)),
            innovation_operator=base,
        )
    with pytest.raises(ValueError, match="innovation must have shape"):
        condition_gaussian_query(
            prior_mean=np.asarray([0.0]),
            prior_covariance=np.asarray([[1.0]]),
            innovation=np.zeros(6),
            query_observation_cross_covariance=np.zeros((2, 3)),
            innovation_operator=base,
        )
    with pytest.raises(ValueError, match="cross_covariance must have shape"):
        condition_gaussian_query(
            prior_mean=np.asarray([0.0]),
            prior_covariance=np.asarray([[1.0]]),
            innovation=np.zeros((2, 3)),
            query_observation_cross_covariance=np.zeros((1, 2)),
            innovation_operator=base,
        )
