"""Coordinate/metric invariance tests for the query kernel, not factor fitting."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from prob4d.observable_gauge import ObservableGaugeFactor
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
)


@pytest.fixture
def partial_factor() -> ObservableGaugeFactor:
    # Analytic factor: x-axis twist is unobserved. The fixture deliberately
    # isolates the query algebra from point-cloud fitting and provider code.
    identity = np.eye(7)
    basis = np.delete(identity, 1, axis=1)
    return cast(
        ObservableGaugeFactor,
        SimpleNamespace(
            rank=6,
            information_matrix=10.0 * basis @ basis.T,
            observable_basis=basis,
            nullspace_basis=identity[:, 1:2],
        ),
    )


def _query() -> np.ndarray:
    jacobian = np.zeros((2, 7))
    jacobian[0, 0] = 10.0
    jacobian[1, 1] = 1.0
    return jacobian


def _evaluate(factor, jacobian, metric=None, prior=None):
    return evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7) if prior is None else prior,
        query_jacobian_local=jacobian,
        query_metric=metric,
    )


def _assert_unresolved(report):
    assert report.direct_observability_fraction == pytest.approx(100.0 / 101.0)
    assert report.metric_variance_reduction_fraction == pytest.approx(
        1000.0 / 1111.0
    )
    assert report.worst_supported_variance_ratio == pytest.approx(1.0)
    assert not report.gauge_invariant_query
    decision = QueryObservabilityGate(0.8, 0.8, 0.5).evaluate(report)
    assert not decision.admitted
    assert decision.reason_codes == (
        "excessive-worst-direction-variance-ratio",
    )


@pytest.mark.parametrize(
    "exponent",
    [-100, -20, -9, -6, -3, 0, 3, 9, 20, 100],
)
def test_uniform_unit_invariance(partial_factor, exponent):
    scale = 10.0**exponent
    _assert_unresolved(
        _evaluate(
            partial_factor,
            scale * _query(),
            np.eye(2) / scale**2,
        )
    )


@pytest.mark.parametrize(
    "exponent",
    [-100, -20, -6, -3, 0, 3, 6, 20, 100],
)
def test_mixed_unit_invariance(partial_factor, exponent):
    scale = 10.0**exponent
    transform = np.diag([1.0, scale])
    metric = np.diag([1.0, 1.0 / scale**2])
    _assert_unresolved(_evaluate(partial_factor, transform @ _query(), metric))


@pytest.mark.parametrize(
    "exponent",
    [-200, -40, -20, 0, 20, 40, 200],
)
def test_global_metric_scale_does_not_change_ratios(partial_factor, exponent):
    _assert_unresolved(
        _evaluate(
            partial_factor,
            _query(),
            10.0**exponent * np.eye(2),
        )
    )


@pytest.mark.parametrize("seed", list(range(8)))
def test_invertible_output_reparameterization(partial_factor, seed):
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    right, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    transform = left @ np.diag([0.2, 3.0]) @ right
    inverse = np.linalg.inv(transform)
    _assert_unresolved(
        _evaluate(
            partial_factor,
            transform @ _query(),
            inverse.T @ inverse,
        )
    )


def test_zero_query_remains_locally_insensitive(partial_factor):
    report = _evaluate(partial_factor, np.zeros((2, 7)))
    assert report.gauge_invariant_query
    assert report.direct_observability_fraction == 1.0
    assert report.nullspace_sensitivity_fraction == 0.0
    assert report.metric_variance_reduction_fraction == 0.0
    assert report.worst_supported_variance_ratio == 0.0


def test_tiny_nonzero_query_is_not_declared_invariant(partial_factor):
    jacobian = np.zeros((1, 7))
    jacobian[0, 1] = 1e-100
    report = _evaluate(partial_factor, jacobian)
    assert not report.gauge_invariant_query
    assert report.direct_observability_fraction == 0.0
    assert report.nullspace_sensitivity_fraction == 1.0
    assert report.worst_supported_variance_ratio == pytest.approx(1.0)


def test_prior_correlation_does_not_create_direct_support(partial_factor):
    prior = np.eye(7)
    prior[0, 1] = prior[1, 0] = 0.9
    jacobian = np.zeros((1, 7))
    jacobian[0, 1] = 1.0
    report = _evaluate(partial_factor, jacobian, prior=prior)
    assert report.direct_observability_fraction == 0.0
    assert report.metric_variance_reduction_fraction == pytest.approx(
        0.81 * 10.0 / 11.0
    )
    assert report.worst_supported_variance_ratio == pytest.approx(
        1.0 - 0.81 * 10.0 / 11.0
    )


def test_redundant_query_output_keeps_supported_variance(partial_factor):
    jacobian = np.zeros((2, 7))
    jacobian[:, 0] = [1.0, 2.0]
    report = _evaluate(partial_factor, jacobian)
    assert report.direct_observability_fraction == pytest.approx(1.0)
    assert report.metric_variance_reduction_fraction == pytest.approx(10.0 / 11.0)
    assert report.worst_supported_variance_ratio == pytest.approx(1.0 / 11.0)


def test_report_covariances_keep_caller_units(partial_factor):
    base = _evaluate(partial_factor, _query())
    transform = np.array([[2.0, 0.2], [0.0, 0.1]])
    inverse = np.linalg.inv(transform)
    changed = _evaluate(
        partial_factor,
        transform @ _query(),
        inverse.T @ inverse,
    )
    np.testing.assert_allclose(
        changed.prior_query_covariance,
        transform @ base.prior_query_covariance @ transform.T,
    )
    np.testing.assert_allclose(
        changed.posterior_query_covariance,
        transform @ base.posterior_query_covariance @ transform.T,
    )


def test_metric_shape_rejected_explicitly(partial_factor):
    with pytest.raises(ValueError, match="query_metric must match"):
        _evaluate(partial_factor, _query(), np.eye(3))


@pytest.mark.parametrize(
    "point,expected_direct,expected_ratio,expected_admitted",
    [
        ([1.0, 0.0, 0.0], 1.0, 1.0 / 11.0, True),
        ([0.0, 5.0, 0.0], 53.0 / 78.0, 276.0 / 286.0, False),
    ],
)
def test_existing_point_controls_preserved(
    partial_factor,
    point,
    expected_direct,
    expected_ratio,
    expected_admitted,
):
    x, y, z = point
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    jacobian = np.column_stack([point, -skew, np.eye(3)])
    report = _evaluate(partial_factor, jacobian)
    assert report.direct_observability_fraction == pytest.approx(expected_direct)
    assert report.worst_supported_variance_ratio == pytest.approx(expected_ratio)
    assert (
        QueryObservabilityGate(0.8, 0.8, 0.5).evaluate(report).admitted
        == expected_admitted
    )
