from itertools import permutations

import numpy as np

from prob4d.gauge import GaugeEstimate, RelativeGaugeConstraint
from prob4d.marginalized_gauge import MarginalizedFixedLagGaugeSmoother
from prob4d.sim3 import Sim3


def _chain(
    count: int = 5,
    prior_variance: float = 0.5,
    edge_variance: float = 0.2,
    translation: float = 0.1,
):
    window_ids = [f"w{index}" for index in range(count)]
    relative = Sim3(translation=np.array([translation, 0.0, 0.0]))
    transforms = {window_ids[0]: Sim3.identity()}
    for index in range(1, count):
        transforms[window_ids[index]] = transforms[window_ids[index - 1]].compose(relative)
    estimates = {
        window_id: GaugeEstimate(
            window_id,
            transforms[window_id],
            np.eye(7) * (prior_variance + index * edge_variance),
        )
        for index, window_id in enumerate(window_ids)
    }
    constraints = [
        RelativeGaugeConstraint(
            window_ids[index],
            window_ids[index + 1],
            relative,
            np.eye(7) * edge_variance,
            num_correspondences=100,
        )
        for index in range(count - 1)
    ]
    return window_ids, estimates, constraints


def test_chain_matches_full_batch_marginals() -> None:
    window_ids, estimates, constraints = _chain()
    reference = MarginalizedFixedLagGaugeSmoother(
        lag=len(window_ids), damping=1e-10, tolerance=1e-10
    ).smooth(window_ids, estimates, constraints)
    actual = MarginalizedFixedLagGaugeSmoother(
        lag=2, damping=1e-10, tolerance=1e-10
    ).smooth(window_ids, estimates, constraints)

    for window_id in window_ids:
        np.testing.assert_allclose(
            actual[window_id].global_from_local.as_vector(),
            reference[window_id].global_from_local.as_vector(),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual[window_id].covariance,
            reference[window_id].covariance,
            rtol=2e-5,
            atol=2e-7,
        )


def test_chain_retains_expired_uncertainty() -> None:
    prior_variance = 0.5
    edge_variance = 0.2
    window_ids, estimates, constraints = _chain(
        prior_variance=prior_variance,
        edge_variance=edge_variance,
        translation=0.0,
    )
    actual = MarginalizedFixedLagGaugeSmoother(
        lag=2, damping=1e-10, tolerance=1e-10
    ).smooth(window_ids, estimates, constraints)

    for index, window_id in enumerate(window_ids):
        np.testing.assert_allclose(
            np.diag(actual[window_id].covariance),
            prior_variance + index * edge_variance,
            rtol=2e-5,
            atol=2e-7,
        )


def test_constraint_order_does_not_change_result() -> None:
    window_ids, estimates, constraints = _chain(count=4)
    reference = MarginalizedFixedLagGaugeSmoother(lag=2).smooth(
        window_ids, estimates, constraints
    )

    for ordering in permutations(constraints):
        actual = MarginalizedFixedLagGaugeSmoother(lag=2).smooth(
            window_ids, estimates, list(ordering)
        )
        for window_id in window_ids:
            np.testing.assert_allclose(
                actual[window_id].global_from_local.as_vector(),
                reference[window_id].global_from_local.as_vector(),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                actual[window_id].covariance,
                reference[window_id].covariance,
                atol=1e-9,
            )
