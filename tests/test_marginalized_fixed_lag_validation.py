import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate, RelativeGaugeConstraint
from prob4d.marginalized_gauge import MarginalizedFixedLagGaugeSmoother
from prob4d.sim3 import Sim3


def _inputs():
    window_ids = ["w0", "w1", "w2", "w3"]
    estimates = {
        window_id: GaugeEstimate(window_id, Sim3.identity(), np.eye(7))
        for window_id in window_ids
    }
    constraints = [
        RelativeGaugeConstraint(
            window_ids[index],
            window_ids[index + 1],
            Sim3.identity(),
            np.eye(7),
        )
        for index in range(len(window_ids) - 1)
    ]
    return window_ids, estimates, constraints


def test_rejects_factor_arriving_after_marginalization() -> None:
    window_ids, estimates, constraints = _inputs()
    constraints.append(
        RelativeGaugeConstraint(
            window_ids[0],
            window_ids[2],
            Sim3.identity(),
            np.eye(7),
        )
    )

    with pytest.raises(ValueError, match="span reaches beyond"):
        MarginalizedFixedLagGaugeSmoother(lag=2).smooth(
            window_ids, estimates, constraints
        )


def test_rejects_invalid_factor_covariance() -> None:
    window_ids, estimates, constraints = _inputs()
    covariance = np.eye(7)
    covariance[-1, -1] = -0.1
    constraints[0] = RelativeGaugeConstraint(
        window_ids[0],
        window_ids[1],
        Sim3.identity(),
        covariance,
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        MarginalizedFixedLagGaugeSmoother(lag=2).smooth(
            window_ids, estimates, constraints
        )
