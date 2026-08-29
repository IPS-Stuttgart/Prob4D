"""Connect the finite-angle control to the existing *local* observability gate.

These tests do not change the gate or promote this conditional kernel to the
claim-bearing exporter. They expose why a local certificate needs a separate
finite-angle closure check before it is interpreted globally.
"""

import numpy as np

from prob4d.axial_gauge_moments import AxialGaugeOrbit, CircularMoments2
from prob4d.observable_gauge import CentroidGaugeChart, ObservableGaugeFactor
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
    point_position_query_jacobian,
)
from prob4d.sim3 import Sim3


def _z_axis_factor():
    basis = np.eye(7)
    return ObservableGaugeFactor(
        chart=CentroidGaugeChart(
            linearization=Sim3(scale=1.0, rotation=np.eye(3), translation=np.zeros(3)),
            source_centroid=np.zeros(3),
            cloud_scale=0.1,
        ),
        observable_basis=basis[:, [0, 1, 2, 4, 5, 6]],
        nullspace_basis=basis[:, 3:4],
        observable_information=999.0 * np.eye(6),
        normalized_geometry_spectrum=np.array([1.0] * 6 + [0.0]),
        rank_threshold=1e-8,
        residual_rms=0.01,
        residual_variance=0.0001,
        inlier_fraction=1.0,
        num_correspondences=48,
        covariance_method="iid_observable_information_v1",
    )


def test_locally_admitted_radial_query_has_finite_unobserved_orbit_variance():
    factor = _z_axis_factor()
    point = np.array([0.1, 0.0, 0.0])
    jacobian = point_position_query_jacobian(factor, point)[0:1]
    report = evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7),
        query_jacobian_local=jacobian,
    )
    gate = QueryObservabilityGate(0.99, 0.9, 0.1)
    assert gate.evaluate(report).admitted
    np.testing.assert_allclose(report.direct_observability_fraction, 1.0)
    np.testing.assert_allclose(report.posterior_query_covariance, [[0.00002]])
    np.testing.assert_array_equal(jacobian @ factor.nullspace_basis, [[0.0]])

    # This isolates the axial conditional contribution. It is NOT an exact
    # replacement for every term of the complete seven-dimensional posterior.
    orbit = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    query = orbit.point_moments(
        point[None, :], CircularMoments2.wrapped_normal(0.0, 1.0)
    ).project([[1.0, 0.0, 0.0]])
    assert query.covariance[0, 0] > 0.0019
    np.testing.assert_allclose(query.full_orbit_bounds, [[-0.1, 0.1]])


def test_full_position_gate_still_detects_the_first_order_tangential_nullspace():
    factor = _z_axis_factor()
    jacobian = point_position_query_jacobian(factor, np.array([0.1, 0.0, 0.0]))
    report = evaluate_query_observability(
        factor,
        prior_covariance_local=np.eye(7),
        query_jacobian_local=jacobian,
    )
    assert not QueryObservabilityGate(0.99, 0.9, 0.1).evaluate(report).admitted
    assert report.nullspace_sensitivity_fraction > 0.0
