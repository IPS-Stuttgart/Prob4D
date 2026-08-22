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
    augment_observation_gaussian_operator,
    condition_gaussian_query,
)


def _correlated_base_operator() -> tuple[ObservationGaussianOperator, np.ndarray]:
    conditional = np.asarray(
        [
            np.diag([0.30, 0.40, 0.50]),
            np.diag([0.60, 0.70, 0.80]),
        ],
        dtype=np.float64,
    )
    gauge_jacobian = np.zeros((2, 3, 7), dtype=np.float64)
    gauge_jacobian[0, :, :3] = np.asarray(
        [
            [1.00, 0.10, 0.00],
            [0.00, 0.80, -0.10],
            [0.05, 0.00, 0.60],
        ]
    )
    gauge_jacobian[1, :, :3] = np.asarray(
        [
            [0.50, 0.00, 0.20],
            [-0.10, 0.70, 0.00],
            [0.00, 0.15, 0.90],
        ]
    )
    gauge_prior_covariance = np.zeros((7, 7), dtype=np.float64)
    gauge_prior_covariance[:3, :3] = np.asarray(
        [
            [0.120, 0.020, -0.010],
            [0.020, 0.090, 0.015],
            [-0.010, 0.015, 0.070],
        ]
    )
    gauge_row_covariance = np.asarray(
        [
            jacobian @ gauge_prior_covariance @ jacobian.T
            for jacobian in gauge_jacobian
        ]
    )
    stacked = StackedObservationFactors(
        world_mean_m=np.zeros((2, 3), dtype=np.float64),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=conditional + gauge_row_covariance,
        gauge_jacobian=gauge_jacobian,
        gauge_prior_covariance=gauge_prior_covariance,
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
    flattened_jacobian = gauge_jacobian.reshape(6, 7)
    covariance += (
        flattened_jacobian
        @ gauge_prior_covariance
        @ flattened_jacobian.T
    )
    return build_observation_gaussian_operator(stacked), covariance


def test_query_conditioning_preserves_correlated_provider_covariance() -> None:
    base, provider_covariance = _correlated_base_operator()
    assert np.linalg.norm(provider_covariance[:3, 3:]) > 0.0

    rng = np.random.default_rng(20260823)
    state_factor = rng.normal(size=(4, 4))
    state_covariance = state_factor @ state_factor.T + 0.25 * np.eye(4)
    state_mean = rng.normal(size=4)
    observation_matrix = rng.normal(size=(6, 4))
    query_matrix = rng.normal(size=(2, 4))
    observation = rng.normal(size=6)

    prior_root = np.linalg.cholesky(state_covariance)
    innovation_operator = augment_observation_gaussian_operator(
        base,
        (observation_matrix @ prior_root).reshape(2, 3, 4),
    )
    innovation = observation - observation_matrix @ state_mean
    innovation_covariance = (
        provider_covariance
        + observation_matrix @ state_covariance @ observation_matrix.T
    )
    cross_covariance = query_matrix @ state_covariance @ observation_matrix.T
    prior_query_mean = query_matrix @ state_mean
    prior_query_covariance = query_matrix @ state_covariance @ query_matrix.T

    batched_rhs = np.concatenate(
        (
            innovation.reshape(2, 3, 1),
            cross_covariance.T.reshape(2, 3, 2),
        ),
        axis=2,
    )
    np.testing.assert_allclose(
        innovation_operator.solve(batched_rhs).reshape(6, 3),
        np.linalg.solve(innovation_covariance, batched_rhs.reshape(6, 3)),
        atol=1e-11,
        rtol=1e-11,
    )
    assert innovation_operator.log_determinant == pytest.approx(
        np.linalg.slogdet(innovation_covariance)[1]
    )

    result = condition_gaussian_query(
        prior_mean=prior_query_mean,
        prior_covariance=prior_query_covariance,
        innovation=innovation.reshape(2, 3),
        query_observation_cross_covariance=cross_covariance,
        innovation_operator=innovation_operator,
    )
    innovation_response = np.linalg.solve(innovation_covariance, innovation)
    expected_reduction = (
        cross_covariance
        @ np.linalg.solve(innovation_covariance, cross_covariance.T)
    )
    expected_nll = 0.5 * (
        6 * math.log(2.0 * math.pi)
        + np.linalg.slogdet(innovation_covariance)[1]
        + innovation @ innovation_response
    )

    np.testing.assert_allclose(
        result.posterior_mean,
        prior_query_mean + cross_covariance @ innovation_response,
    )
    np.testing.assert_allclose(
        result.posterior_covariance,
        prior_query_covariance - expected_reduction,
    )
    assert result.innovation_negative_log_likelihood == pytest.approx(expected_nll)
