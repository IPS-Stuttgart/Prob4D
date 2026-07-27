import numpy as np
import pytest

from prob4d.gauge import (
    GaugeAnchor,
    GaugeCovarianceCalibration,
    GaugeEstimate,
    PointAnchor,
    RelativeGaugeConstraint,
    ScaleAnchor,
)
from prob4d.sim3 import Sim3


def _indefinite(dimension: int) -> np.ndarray:
    covariance = np.eye(dimension)
    covariance[-1, -1] = -0.1
    return covariance


@pytest.mark.parametrize(
    "factory",
    [
        lambda covariance: RelativeGaugeConstraint(
            "reference", "moving", Sim3.identity(), covariance
        ),
        lambda covariance: GaugeEstimate("window", Sim3.identity(), covariance),
        lambda covariance: GaugeAnchor("window", Sim3.identity(), covariance),
    ],
)
def test_gauge_contracts_reject_indefinite_covariance(factory) -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        factory(_indefinite(7))


def test_gauge_contracts_reject_asymmetric_and_nonfinite_covariance() -> None:
    asymmetric = np.eye(7)
    asymmetric[0, 1] = 0.2
    with pytest.raises(ValueError, match="symmetric"):
        GaugeEstimate("window", Sim3.identity(), asymmetric)

    nonfinite = np.eye(7)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        GaugeAnchor("window", Sim3.identity(), nonfinite)


def test_gauge_covariance_is_defensively_copied_and_read_only() -> None:
    source = np.eye(7)
    estimate = GaugeEstimate("window", Sim3.identity(), source)
    source[0, 0] = 9.0

    assert estimate.covariance[0, 0] == 1.0
    assert not estimate.covariance.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        estimate.covariance[0, 0] = 2.0


def test_relative_constraint_validates_metadata() -> None:
    with pytest.raises(ValueError, match="distinct"):
        RelativeGaugeConstraint("window", "window", Sim3.identity(), np.eye(7))
    with pytest.raises(ValueError, match="residual_rms"):
        RelativeGaugeConstraint(
            "reference",
            "moving",
            Sim3.identity(),
            np.eye(7),
            residual_rms=np.nan,
        )
    with pytest.raises(ValueError, match="num_correspondences"):
        RelativeGaugeConstraint(
            "reference",
            "moving",
            Sim3.identity(),
            np.eye(7),
            num_correspondences=-1,
        )


def test_point_anchor_copies_coordinates_and_covariance() -> None:
    local = np.array([1.0, 2.0, 3.0])
    global_point = np.array([4.0, 5.0, 6.0])
    covariance = np.eye(3)
    anchor = PointAnchor("window", local, global_point, covariance)
    local[:] = 0.0
    global_point[:] = 0.0
    covariance[:] = 0.0

    np.testing.assert_array_equal(anchor.local_point, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(anchor.global_point, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(anchor.covariance, np.eye(3))
    assert not anchor.local_point.flags.writeable
    assert not anchor.global_point.flags.writeable
    assert not anchor.covariance.flags.writeable


def test_anchor_scalar_contracts_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        ScaleAnchor("", 1.0, 0.1)
    with pytest.raises(ValueError, match="finite"):
        ScaleAnchor("window", np.nan, 0.1)
    with pytest.raises(ValueError, match="finite"):
        PointAnchor("window", np.array([np.nan, 0.0, 0.0]), np.zeros(3), np.eye(3))


def test_covariance_calibration_fails_closed_on_invalid_input() -> None:
    calibration = GaugeCovarianceCalibration(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        calibration.apply(_indefinite(7))

    covariances = np.stack([np.eye(7), _indefinite(7)])
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaugeCovarianceCalibration.fit(np.zeros((2, 7)), covariances)


def test_numerical_negative_noise_is_clipped_without_adding_a_floor() -> None:
    covariance = np.eye(7)
    covariance[-1, -1] = -1e-14
    estimate = GaugeEstimate("window", Sim3.identity(), covariance)

    assert estimate.covariance[-1, -1] >= 0.0
    assert np.min(np.linalg.eigvalsh(estimate.covariance)) >= -1e-15
