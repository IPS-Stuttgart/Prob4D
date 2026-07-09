import numpy as np

from prob4d.data import PredictionWindow
from prob4d.uncertainty import DepthDisagreementModel


def make_window() -> PredictionWindow:
    point_map = np.zeros((2, 2, 3, 3))
    point_map[..., 2] = np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]] * 2)
    return PredictionWindow(
        "window",
        np.array([0, 1]),
        point_map,
        np.ones((2, 2, 3), dtype=bool),
    )


def test_depth_model_grows_along_ray_uncertainty() -> None:
    covariance = DepthDisagreementModel().predict(make_window())

    assert covariance.parallel_variance[0, 0, 2] > covariance.parallel_variance[0, 0, 0]
    assert covariance.parallel_variance[0, 0, 2] > covariance.lateral_variance[0, 0, 2]
    np.testing.assert_allclose(
        covariance.matrices()[0, 0, 0, :2, :2], np.eye(2) * covariance.lateral_variance[0, 0, 0]
    )


def test_calibration_recovers_known_variance_scale() -> None:
    generator = np.random.default_rng(2)
    window = make_window()
    model = DepthDisagreementModel()
    covariance = model.predict(window)
    errors = np.zeros_like(window.point_map)
    errors[..., 2] = generator.normal(
        scale=np.sqrt(4.0 * covariance.parallel_variance), size=window.shape
    )
    errors[..., 0] = generator.normal(
        scale=np.sqrt(2.0 * covariance.lateral_variance), size=window.shape
    )
    errors[..., 1] = generator.normal(
        scale=np.sqrt(2.0 * covariance.lateral_variance), size=window.shape
    )

    calibrated, report = model.calibrate(errors, covariance, trim_quantile=1.0)

    assert calibrated.parallel_scale > 1.0
    assert calibrated.lateral_scale > 1.0
    assert report.count == np.prod(window.shape)
