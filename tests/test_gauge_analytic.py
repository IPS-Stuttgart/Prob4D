from __future__ import annotations

import numpy as np
import pytest

from prob4d.gauge import RelativeGaugeConstraint, SequentialGaugeEstimator
from prob4d.gauge_analytic import (
    ANALYTIC_GAUGE_PROPAGATION_METHOD,
    AnalyticSequentialGaugeEstimatorV2,
    analytic_sim3_inverse_jacobian,
    compose_sim3_with_covariance_analytic,
    invert_sim3_with_covariance_analytic,
)
from prob4d.sim3 import Sim3

_CENTRAL_DIFFERENCE_RELATIVE_STEP = float(np.cbrt(np.finfo(np.float64).eps))


def _central_jacobian(function, vector: np.ndarray) -> np.ndarray:
    baseline = np.asarray(function(vector), dtype=np.float64)
    result = np.empty((baseline.size, vector.size), dtype=np.float64)
    for index in range(vector.size):
        step = _CENTRAL_DIFFERENCE_RELATIVE_STEP * max(
            1.0,
            abs(float(vector[index])),
        )
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        result[:, index] = (function(plus) - function(minus)) / (2.0 * step)
    return result


@pytest.mark.parametrize("seed", range(12))
def test_analytic_inverse_jacobian_matches_central_difference(seed: int) -> None:
    generator = np.random.default_rng(seed)
    rotation = generator.normal(size=3)
    rotation *= generator.uniform(0.0, 2.7) / max(np.linalg.norm(rotation), 1e-12)
    vector = np.concatenate(
        ([generator.uniform(-2.0, 2.0)], rotation, generator.normal(size=3))
    )
    transform = Sim3.from_vector(vector)

    analytic = analytic_sim3_inverse_jacobian(transform)
    numerical = _central_jacobian(
        lambda value: Sim3.from_vector(value).inverse().as_vector(),
        vector,
    )

    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-8)


def test_analytic_covariance_propagation_matches_central_difference() -> None:
    first = Sim3.from_vector(np.array([0.3, 0.2, -0.1, 0.15, 1.0, -2.0, 0.5]))
    second = Sim3.from_vector(np.array([-0.4, -0.1, 0.25, 0.05, -0.2, 0.4, 1.2]))
    first_covariance = np.diag(np.linspace(0.01, 0.07, 7))
    second_covariance = np.diag(np.linspace(0.02, 0.08, 7))

    output, covariance = compose_sim3_with_covariance_analytic(
        first,
        first_covariance,
        second,
        second_covariance,
    )
    first_jacobian = _central_jacobian(
        lambda value: Sim3.from_vector(value).compose(second).as_vector(),
        first.as_vector(),
    )
    second_jacobian = _central_jacobian(
        lambda value: first.compose(Sim3.from_vector(value)).as_vector(),
        second.as_vector(),
    )
    expected = (
        first_jacobian @ first_covariance @ first_jacobian.T
        + second_jacobian @ second_covariance @ second_jacobian.T
    )

    np.testing.assert_allclose(output.as_vector(), first.compose(second).as_vector())
    np.testing.assert_allclose(covariance, expected, rtol=2e-6, atol=2e-8)

    inverse, inverse_covariance = invert_sim3_with_covariance_analytic(
        first,
        first_covariance,
    )
    inverse_jacobian = _central_jacobian(
        lambda value: Sim3.from_vector(value).inverse().as_vector(),
        first.as_vector(),
    )
    np.testing.assert_allclose(inverse.as_vector(), first.inverse().as_vector())
    np.testing.assert_allclose(
        inverse_covariance,
        inverse_jacobian @ first_covariance @ inverse_jacobian.T,
        rtol=2e-6,
        atol=2e-8,
    )


def test_analytic_estimator_preserves_legacy_transform_semantics() -> None:
    first_relative = Sim3.from_vector(
        np.array([0.05, 0.1, -0.05, 0.02, 0.3, -0.1, 0.2])
    )
    second_relative = Sim3.from_vector(
        np.array([-0.02, -0.04, 0.08, 0.03, -0.2, 0.4, 0.1])
    )
    constraints = [
        RelativeGaugeConstraint(
            "w0",
            "w1",
            first_relative,
            np.diag(np.linspace(1e-4, 7e-4, 7)),
        ),
        RelativeGaugeConstraint(
            "w1",
            "w2",
            second_relative,
            np.diag(np.linspace(2e-4, 8e-4, 7)),
        ),
    ]
    initial_covariance = np.diag(np.linspace(1e-5, 7e-5, 7))

    legacy = SequentialGaugeEstimator().estimate(
        ["w0", "w1", "w2"],
        constraints,
        initial_covariance=initial_covariance,
    )
    analytic_estimator = AnalyticSequentialGaugeEstimatorV2()
    analytic = analytic_estimator.estimate(
        ["w0", "w1", "w2"],
        constraints,
        initial_covariance=initial_covariance,
    )

    assert analytic_estimator.jacobian_method == ANALYTIC_GAUGE_PROPAGATION_METHOD
    for window_id in ("w0", "w1", "w2"):
        np.testing.assert_allclose(
            analytic[window_id].global_from_local.as_vector(),
            legacy[window_id].global_from_local.as_vector(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            analytic[window_id].covariance,
            legacy[window_id].covariance,
            rtol=3e-5,
            atol=2e-9,
        )


def test_analytic_gauge_propagation_fails_closed() -> None:
    branch_cut = Sim3.from_vector(np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="branch cut"):
        analytic_sim3_inverse_jacobian(branch_cut)
    with pytest.raises(ValueError, match="positive semidefinite"):
        invert_sim3_with_covariance_analytic(
            Sim3.identity(),
            np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.1]),
        )
    with pytest.raises(TypeError, match="genuine integer"):
        AnalyticSequentialGaugeEstimatorV2(covariance_intersection_grid_size=True)
