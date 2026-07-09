import numpy as np

from prob4d.data import PredictionWindow
from prob4d.fusion import (
    fuse_gaussians_covariance_intersection,
    fuse_gaussians_independent,
    fuse_windows,
)
from prob4d.sim3 import Sim3
from prob4d.uncertainty import StructuredCovariance


def test_covariance_intersection_is_conservative_for_equal_inputs() -> None:
    first_mean = np.array([[0.0, 0.0, 0.0]])
    second_mean = np.array([[2.0, 0.0, 0.0]])
    covariance = np.eye(3)[None]

    independent_mean, independent_covariance = fuse_gaussians_independent(
        first_mean, covariance, second_mean, covariance
    )
    ci_mean, ci_covariance, weight = fuse_gaussians_covariance_intersection(
        first_mean, covariance, second_mean, covariance
    )

    np.testing.assert_allclose(independent_mean, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(independent_covariance, 0.5 * covariance)
    np.testing.assert_allclose(ci_mean, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(ci_covariance, covariance)
    np.testing.assert_allclose(weight, [0.5])


def test_pointwise_covariance_intersection_remains_available() -> None:
    means = np.zeros((2, 3))
    first_covariance = np.stack([np.eye(3), 4.0 * np.eye(3)])
    second_covariance = np.stack([4.0 * np.eye(3), np.eye(3)])

    _, _, weights = fuse_gaussians_covariance_intersection(
        means,
        first_covariance,
        means,
        second_covariance,
        weight_mode="pointwise",
    )

    assert weights[0] > 0.9
    assert weights[1] < 0.1


def make_window(window_id: str, frames: list[int], offset: float) -> PredictionWindow:
    points = np.zeros((len(frames), 1, 2, 3))
    points[..., 0] = np.asarray(frames)[:, None, None] + offset
    return PredictionWindow(
        window_id,
        np.asarray(frames),
        points,
        np.ones((len(frames), 1, 2), dtype=bool),
    )


def make_uncertainty(window: PredictionWindow, variance: float) -> StructuredCovariance:
    rays = np.zeros_like(window.point_map)
    rays[..., 2] = 1.0
    return StructuredCovariance(
        rays,
        np.full(window.shape, variance),
        np.full(window.shape, variance),
    )


def test_fuse_windows_transforms_and_combines_overlap() -> None:
    first = make_window("first", [0, 1], 0.0)
    second = make_window("second", [1, 2], -10.0)
    gauges = {
        "first": Sim3.identity(),
        "second": Sim3(translation=np.array([10.0, 0.0, 0.0])),
    }
    uncertainties = {
        "first": make_uncertainty(first, 1.0),
        "second": make_uncertainty(second, 1.0),
    }

    result = fuse_windows([first, second], gauges, uncertainties, method="covariance_intersection")

    np.testing.assert_array_equal(result.frame_indices, [0, 1, 2])
    np.testing.assert_allclose(result.point_map[:, 0, 0, 0], [0.0, 1.0, 2.0])
    assert result.contributors[1, 0, 0] == 2
    np.testing.assert_allclose(result.point_covariance[1, 0, 0], np.eye(3))
