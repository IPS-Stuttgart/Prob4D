"""Integration with the existing local query API, not a recreated estimator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from prob4d.axial_query_certificate import (
    AxialRotationOrbit,
    certify_shared_orbit_advantage,
)
from prob4d.axial_query_study import analytic_local_control
from prob4d.observable_gauge import ObservableGaugeFactor
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
    point_position_query_jacobian,
)
from prob4d.sim3 import Sim3


def partial_factor(origin: np.ndarray, axis: np.ndarray) -> ObservableGaugeFactor:
    null = np.concatenate(([0.0], axis, np.zeros(3)))[:, None]
    projection = np.eye(7) - null @ null.T
    values, vectors = np.linalg.eigh(projection)
    observed = vectors[:, values > 0.5]
    return cast(
        ObservableGaugeFactor,
        SimpleNamespace(
            rank=6,
            observable_basis=observed,
            nullspace_basis=null,
            information_matrix=10.0 * observed @ observed.T,
            chart=SimpleNamespace(
                linearization=Sim3.identity(),
                reference_centroid=origin,
                cloud_scale=1.0,
            ),
        ),
    )


def test_actual_local_gate_accepts_a_finite_angle_ambiguous_scalar_query() -> None:
    model = AxialRotationOrbit(np.zeros(3), np.array([0.0, 0.0, 1.0]), "shared-test-gauge")
    factor = partial_factor(model.origin, model.axis)
    point = np.array([1.0, 0.0, 0.0])
    point_jacobian = point_position_query_jacobian(factor, point)
    scalar_jacobian = point_jacobian[0:1, :]
    report = evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7),
        query_jacobian_local=scalar_jacobian,
    )
    gate = QueryObservabilityGate(
        minimum_direct_observability_fraction=0.8,
        minimum_metric_variance_reduction_fraction=0.8,
        maximum_worst_supported_variance_ratio=0.5,
    )
    assert report.direct_observability_fraction == pytest.approx(1.0)
    assert report.metric_variance_reduction_fraction == pytest.approx(10.0 / 11.0)
    assert report.worst_supported_variance_ratio == pytest.approx(1.0 / 11.0)
    assert gate.evaluate(report).admitted
    # An affine loss, not an arbitrary unbounded nonlinear readout:
    # fallback - candidate = 0.25 + cos(theta), whose minimum is -0.75.
    fallback = model.affine_query(point[None], np.zeros((1, 3)), offset=4.0)
    candidate = model.affine_query(point[None], [[-1.0, 0.0, 0.0]], offset=3.75)
    certificate = certify_shared_orbit_advantage(
        fallback_loss=fallback,
        candidate_loss=candidate,
        scope_admitted=True,
    )
    assert not certificate.admitted
    assert certificate.lower_advantage == pytest.approx(-0.75)


def test_study_reference_algebra_matches_existing_api_in_random_frames() -> None:
    rng = np.random.default_rng(55)
    for _ in range(40):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        model = AxialRotationOrbit(rng.normal(size=3), axis, "shared-test-gauge")
        radial = rng.normal(size=3)
        radial -= axis * float(axis @ radial)
        factor = partial_factor(model.origin, model.axis)
        normal = radial / np.linalg.norm(radial)
        jacobian = normal[None, :] @ point_position_query_jacobian(
            factor, model.origin + radial
        )
        actual = evaluate_query_observability(
            factor,
            prior_covariance_local=np.eye(7),
            query_jacobian_local=jacobian,
        )
        reference = analytic_local_control(model, radial)
        for name, value in reference.items():
            assert getattr(actual, name) == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("unobserved_variance", [1.0, 1e6, 1e12])
def test_nullspace_variance_inflation_does_not_fix_zero_query_derivative(
    unobserved_variance: float,
) -> None:
    model = AxialRotationOrbit(np.zeros(3), np.array([0.0, 0.0, 1.0]), "shared-test-gauge")
    factor = partial_factor(model.origin, model.axis)
    prior = np.eye(7)
    prior[3, 3] = unobserved_variance
    jacobian = point_position_query_jacobian(factor, np.array([1.0, 0.0, 0.0]))[0:1]
    report = evaluate_query_observability(
        factor, prior_covariance_local=prior, query_jacobian_local=jacobian
    )
    assert jacobian[0, 3] == 0.0
    assert report.direct_observability_fraction == pytest.approx(1.0)
    assert report.prior_query_covariance[0, 0] == pytest.approx(2.0)
    assert report.posterior_query_covariance[0, 0] == pytest.approx(2.0 / 11.0)
