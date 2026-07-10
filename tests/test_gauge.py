import numpy as np

from prob4d.gauge import (
    FixedLagGaugeSmoother,
    GaugeCovarianceCalibration,
    RelativeGaugeConstraint,
    ScaleAnchor,
    SequentialGaugeEstimator,
    constraint_cost,
)
from prob4d.sim3 import Sim3


def constraint(
    reference_id: str,
    moving_id: str,
    reference_gauge: Sim3,
    moving_gauge: Sim3,
    noise: np.ndarray | None = None,
    covariance_scale: float = 1e-3,
) -> RelativeGaugeConstraint:
    relative = reference_gauge.inverse().compose(moving_gauge)
    if noise is not None:
        relative = relative.compose(Sim3.from_vector(noise))
    return RelativeGaugeConstraint(
        reference_id,
        moving_id,
        relative,
        np.eye(7) * covariance_scale,
    )


def test_sequential_gauge_estimator_recovers_exact_chain() -> None:
    truth = {
        "w0": Sim3.identity(),
        "w1": Sim3.from_vector(np.array([0.02, 0.01, -0.02, 0.01, 0.3, 0.0, 0.1])),
        "w2": Sim3.from_vector(np.array([0.04, 0.02, -0.03, 0.02, 0.7, 0.1, 0.2])),
    }
    constraints = [
        constraint("w0", "w1", truth["w0"], truth["w1"]),
        constraint("w1", "w2", truth["w1"], truth["w2"]),
    ]

    estimates = SequentialGaugeEstimator().estimate(["w0", "w1", "w2"], constraints)

    for window_id in truth:
        np.testing.assert_allclose(
            estimates[window_id].global_from_local.as_vector(),
            truth[window_id].as_vector(),
            atol=1e-8,
        )


def test_fixed_lag_smoothing_reduces_constraint_cost() -> None:
    truth = {
        "w0": Sim3.identity(),
        "w1": Sim3.from_vector(np.array([0.0, 0.01, 0.0, 0.0, 1.0, 0.0, 0.0])),
        "w2": Sim3.from_vector(np.array([0.0, 0.02, 0.0, 0.0, 2.0, 0.0, 0.0])),
        "w3": Sim3.from_vector(np.array([0.0, 0.03, 0.0, 0.0, 3.0, 0.0, 0.0])),
    }
    constraints = [
        constraint("w0", "w1", truth["w0"], truth["w1"], np.array([0.02, 0, 0, 0, 0.05, 0, 0])),
        constraint("w1", "w2", truth["w1"], truth["w2"], np.array([-0.01, 0, 0, 0, -0.03, 0, 0])),
        constraint("w2", "w3", truth["w2"], truth["w3"], np.array([0.015, 0, 0, 0, 0.04, 0, 0])),
        constraint("w0", "w2", truth["w0"], truth["w2"], np.array([-0.005, 0, 0, 0, 0.01, 0, 0])),
        constraint("w1", "w3", truth["w1"], truth["w3"], np.array([0.0, 0, 0, 0, -0.01, 0, 0])),
    ]
    ordered = list(truth)
    initial = SequentialGaugeEstimator().estimate(ordered, constraints)
    before = {key: value.global_from_local for key, value in initial.items()}

    smoothed = FixedLagGaugeSmoother(lag=3).smooth(ordered, initial, constraints)
    after = {key: value.global_from_local for key, value in smoothed.items()}

    assert constraint_cost(constraints, after) < constraint_cost(constraints, before)


def test_sparse_scale_anchor_reduces_chain_scale_drift() -> None:
    identity = Sim3.identity()
    noisy_step = Sim3.from_vector(np.array([0.03, 0, 0, 0, 1.0, 0, 0]))
    constraints = [
        RelativeGaugeConstraint("w0", "w1", noisy_step, np.eye(7) * 1e-2),
        RelativeGaugeConstraint("w1", "w2", noisy_step, np.eye(7) * 1e-2),
    ]
    initial = SequentialGaugeEstimator().estimate(["w0", "w1", "w2"], constraints)
    initial_error = abs(np.log(initial["w2"].global_from_local.scale))

    smoothed = FixedLagGaugeSmoother(lag=3).smooth(
        ["w0", "w1", "w2"],
        initial,
        constraints,
        scale_anchors=[ScaleAnchor("w2", identity.scale, 0.005)],
    )

    final_error = abs(np.log(smoothed["w2"].global_from_local.scale))
    assert final_error < initial_error * 0.2


def test_gauge_covariance_calibration_fits_blockwise_inflation() -> None:
    errors = np.tile(np.array([2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]), (2, 1))
    covariances = np.tile(np.eye(7), (2, 1, 1))

    calibration = GaugeCovarianceCalibration.fit(errors, covariances, trim_quantile=1.0)

    assert calibration.count == 2
    assert calibration.scale == 4.0
    assert calibration.rotation == 9.0
    assert calibration.translation == 16.0
    np.testing.assert_allclose(
        calibration.apply(np.eye(7)),
        np.diag([4.0, 9.0, 9.0, 9.0, 16.0, 16.0, 16.0]),
    )
