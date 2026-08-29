"""Parity with the existing observable factor and query projection APIs."""

import numpy as np

from prob4d.conditional_gauge_design import (
    ConditionalGaugeSession,
    CorrelatedGaugeDesign,
    GaussianGaugeBelief,
)
from prob4d.observable_gauge import CentroidGaugeChart, ObservableGaugeFactor
from prob4d.query_observability import evaluate_query_observability, point_position_query_jacobian
from prob4d.sim3 import Sim3


def test_existing_partial_factor_and_query_api_parity():
    chart = CentroidGaugeChart(
        Sim3(scale=1.0, rotation=np.eye(3), translation=np.zeros(3)), np.zeros(3), 0.1
    )
    factor = ObservableGaugeFactor(
        chart=chart,
        observable_basis=np.eye(7)[:, [0, 2, 3, 4, 5, 6]],
        nullspace_basis=np.eye(7)[:, [1]],
        observable_information=10 * np.eye(6),
        normalized_geometry_spectrum=np.array([1.0] * 6 + [0.0]),
        rank_threshold=1e-8,
        residual_rms=0.01,
        residual_variance=0.0001,
        inlier_fraction=1.0,
        num_correspondences=16,
        covariance_method="iid_observable_information_v1",
    )
    model = CorrelatedGaugeDesign(
        "one-exact-centroid-chart", "known-observable-factor-covariance",
        ("original", "exact-source-replay"), (6, 6),
        np.vstack((factor.observable_basis.T, factor.observable_basis.T)),
        np.tile(factor.observable_covariance, (2, 2)),
    )
    prior = GaussianGaugeBelief(model.chart_id, np.arange(7) / 10, np.eye(7))
    session = ConditionalGaugeSession(model, prior)
    query = point_position_query_jacobian(factor, np.array([0.01, 0.02, 0.03]))
    existing_report = evaluate_query_observability(
        factor, prior_covariance_local=prior.covariance, query_jacobian_local=query
    )
    preview = session.preview_query("original", query)
    np.testing.assert_allclose(preview.posterior_metric_variance,
                               np.trace(existing_report.posterior_query_covariance), rtol=1e-12)
    expected = factor.fuse_local_gaussian(prior.mean, prior.covariance)
    actual = session.assimilate("original", np.zeros(6))
    np.testing.assert_allclose(actual.mean, expected.mean_local, rtol=1e-12)
    np.testing.assert_allclose(actual.covariance, expected.covariance_local, rtol=1e-12)
    assert session.assimilate("exact-source-replay", np.zeros(6)) is actual
