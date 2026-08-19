from __future__ import annotations

import numpy as np

from prob4d.gauge import RelativeGaugeConstraint, SequentialGaugeEstimator
from prob4d.gauge_analytic import AnalyticSequentialGaugeEstimatorV2
from prob4d.sim3 import Sim3


def _relative_constraint(
    reference_id: str,
    moving_id: str,
    reference: Sim3,
    moving: Sim3,
    covariance_scale: float,
) -> RelativeGaugeConstraint:
    return RelativeGaugeConstraint(
        reference_id=reference_id,
        moving_id=moving_id,
        reference_from_moving=reference.inverse().compose(moving),
        covariance=np.diag(np.linspace(1e-5, 7e-5, 7) * covariance_scale),
    )


def test_sequential_estimators_share_multi_parent_and_reverse_edge_control_flow() -> None:
    gauges = {
        "w0": Sim3.identity(),
        "w1": Sim3.from_vector(
            np.array([0.05, 0.12, -0.04, 0.03, 0.3, -0.1, 0.2])
        ),
        "w2": Sim3.from_vector(
            np.array([-0.03, -0.07, 0.09, 0.02, -0.2, 0.4, 0.1])
        ),
        "w3": Sim3.from_vector(
            np.array([0.02, 0.04, 0.03, -0.08, 0.1, 0.2, -0.3])
        ),
    }
    constraints = [
        _relative_constraint("w0", "w1", gauges["w0"], gauges["w1"], 1.0),
        _relative_constraint("w1", "w2", gauges["w1"], gauges["w2"], 2.0),
        _relative_constraint("w2", "w0", gauges["w2"], gauges["w0"], 3.0),
        _relative_constraint("w2", "w3", gauges["w2"], gauges["w3"], 4.0),
        _relative_constraint("w3", "w1", gauges["w3"], gauges["w1"], 5.0),
    ]
    initial_covariance = np.diag(np.linspace(1e-6, 7e-6, 7))
    ordered_ids = list(gauges)

    legacy = SequentialGaugeEstimator(covariance_intersection_grid_size=23).estimate(
        ordered_ids,
        constraints,
        initial_covariance=initial_covariance,
    )
    analytic = AnalyticSequentialGaugeEstimatorV2(
        covariance_intersection_grid_size=23
    ).estimate(
        ordered_ids,
        constraints,
        initial_covariance=initial_covariance,
    )

    for window_id, expected in gauges.items():
        np.testing.assert_allclose(
            legacy[window_id].global_from_local.as_vector(),
            expected.as_vector(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            analytic[window_id].global_from_local.as_vector(),
            legacy[window_id].global_from_local.as_vector(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            analytic[window_id].covariance,
            legacy[window_id].covariance,
            rtol=5e-5,
            atol=2e-9,
        )
