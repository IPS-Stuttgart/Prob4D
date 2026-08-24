from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.api.v2 import (
    StackedObservationFactors,
    build_observation_gaussian_operator,
    project_observation_covariance,
)

_CONDITIONAL_X_VARIANCE = 0.25
_SHARED_GAUGE_VARIANCE = 0.75


def _shared_gauge_stack(observation_count: int) -> StackedObservationFactors:
    if observation_count < 1:
        raise ValueError("observation_count must be positive")

    conditional_row = np.diag([_CONDITIONAL_X_VARIANCE, 1.0, 1.0])
    conditional = np.repeat(conditional_row[None, :, :], observation_count, axis=0)
    gauge_jacobian = np.zeros((observation_count, 3, 7), dtype=np.float64)
    gauge_jacobian[:, 0, 0] = 1.0
    gauge_prior = np.zeros((7, 7), dtype=np.float64)
    gauge_prior[0, 0] = _SHARED_GAUGE_VARIANCE
    gauge_row_covariance = np.asarray(
        [jacobian @ gauge_prior @ jacobian.T for jacobian in gauge_jacobian]
    )
    marginal = conditional + gauge_row_covariance

    return StackedObservationFactors(
        world_mean_m=np.zeros((observation_count, 3), dtype=np.float64),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=marginal,
        gauge_jacobian=gauge_jacobian,
        gauge_prior_covariance=gauge_prior,
        association_probability=np.ones(observation_count),
        prior_reliability=np.ones(observation_count),
        prior_nominal_probability=np.ones(observation_count),
        composite_weight=np.ones(observation_count),
        point_ids=np.arange(1, observation_count + 1, dtype=np.int64),
        frame_indices=np.arange(observation_count, dtype=np.int64),
        view_ids=tuple("camera" for _ in range(observation_count)),
        factor_ids=tuple(f"factor-{index}" for index in range(observation_count)),
        correlation_group_ids=tuple("shared-provider" for _ in range(observation_count)),
        gauge_ids=("shared-gauge",),
        causal_frame_stop=observation_count,
    )


def _dense_covariance(stacked: StackedObservationFactors) -> np.ndarray:
    count = len(stacked.world_mean_m)
    conditional = np.zeros((3 * count, 3 * count), dtype=np.float64)
    for index, block in enumerate(stacked.conditional_world_covariance_m2):
        conditional[3 * index : 3 * index + 3, 3 * index : 3 * index + 3] = block
    jacobian = np.asarray(stacked.gauge_jacobian).reshape(3 * count, -1)
    return conditional + jacobian @ stacked.gauge_prior_covariance @ jacobian.T


def _dense_gaussian_nll(residual: np.ndarray, covariance: np.ndarray) -> float:
    flattened = residual.reshape(-1)
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign == 1.0
    quadratic = float(flattened @ np.linalg.solve(covariance, flattened))
    return 0.5 * (
        flattened.size * math.log(2.0 * math.pi) + log_determinant + quadratic
    )


def _rowwise_marginal_nll(
    residual: np.ndarray,
    marginal_covariance: np.ndarray,
) -> float:
    total = 0.0
    for row, covariance in zip(residual, marginal_covariance, strict=True):
        sign, log_determinant = np.linalg.slogdet(covariance)
        assert sign == 1.0
        quadratic = float(row @ np.linalg.solve(covariance, row))
        total += 0.5 * (3 * math.log(2.0 * math.pi) + log_determinant + quadratic)
    return total


def test_explicit_shared_latent_matches_collapsed_joint_gaussian() -> None:
    stacked = _shared_gauge_stack(2)
    residual = np.asarray(
        [[0.7, -0.2, 0.1], [-0.4, 0.3, -0.1]],
        dtype=np.float64,
    )
    dense_covariance = _dense_covariance(stacked)

    structured = build_observation_gaussian_operator(stacked).gaussian_nll(residual)
    collapsed = _dense_gaussian_nll(residual, dense_covariance)

    assert structured == pytest.approx(collapsed, abs=1e-12, rel=1e-12)


def test_rowwise_marginal_scores_discard_the_shared_dependence_direction() -> None:
    stacked = _shared_gauge_stack(2)
    common_mode = np.zeros((2, 3), dtype=np.float64)
    common_mode[:, 0] = 1.0
    contrast_mode = np.zeros((2, 3), dtype=np.float64)
    contrast_mode[:, 0] = np.asarray([1.0, -1.0])

    rowwise_common = _rowwise_marginal_nll(
        common_mode,
        stacked.marginal_world_covariance_m2,
    )
    rowwise_contrast = _rowwise_marginal_nll(
        contrast_mode,
        stacked.marginal_world_covariance_m2,
    )
    operator = build_observation_gaussian_operator(stacked)
    joint_common = operator.gaussian_nll(common_mode)
    joint_contrast = operator.gaussian_nll(contrast_mode)

    assert rowwise_common == pytest.approx(rowwise_contrast)
    assert joint_common < rowwise_common < joint_contrast
    assert joint_contrast - joint_common > 3.0


def test_physical_queries_preserve_shared_mode_cancellation_and_amplification() -> None:
    stacked = _shared_gauge_stack(2)
    query = np.asarray(
        [
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    projected = project_observation_covariance(stacked, query)

    np.testing.assert_allclose(
        projected.conditional_covariance,
        np.diag([0.5 * _CONDITIONAL_X_VARIANCE, 2.0 * _CONDITIONAL_X_VARIANCE]),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        projected.gauge_covariance,
        np.diag([_SHARED_GAUGE_VARIANCE, 0.0]),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        projected.marginal_covariance,
        projected.conditional_covariance + projected.gauge_covariance,
        atol=1e-12,
        rtol=1e-12,
    )


def test_structured_factor_storage_scales_subquadratically_in_observation_count() -> None:
    small = build_observation_gaussian_operator(_shared_gauge_stack(2))
    large = build_observation_gaussian_operator(_shared_gauge_stack(128))

    assert large.factor_storage_nbytes > small.factor_storage_nbytes
    assert large.factor_storage_ratio_to_dense < small.factor_storage_ratio_to_dense
    assert large.factor_storage_ratio_to_dense < 0.01
