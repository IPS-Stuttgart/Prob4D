from itertools import permutations

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


def test_fuse_windows_propagates_gauge_translation_uncertainty_to_points_only() -> None:
    window = make_window("window", [0, 1], 0.0)
    window = PredictionWindow(
        window.window_id,
        window.frame_indices,
        window.point_map,
        window.valid_mask,
        scene_flow=np.ones_like(window.point_map),
        deform_mask=window.valid_mask,
    )
    gauge_covariance = np.zeros((7, 7))
    gauge_covariance[4, 4] = 4.0

    result = fuse_windows(
        [window],
        {window.window_id: Sim3.identity()},
        {window.window_id: make_uncertainty(window, 1.0)},
        method="uniform",
        gauge_covariances={window.window_id: gauge_covariance},
    )

    np.testing.assert_allclose(result.point_covariance[..., 0, 0], 5.0)
    np.testing.assert_allclose(result.point_covariance[..., 1, 1], 1.0)
    np.testing.assert_allclose(
        result.flow_covariance, np.broadcast_to(np.eye(3), result.flow_covariance.shape)
    )


def test_gauge_covariance_propagation_matches_parameter_finite_difference() -> None:
    transform = Sim3.from_vector(np.array([0.2, 0.5, -0.3, 0.2, 1.0, 2.0, -1.0]))
    point = np.array([2.0, -0.5, 3.0])
    covariance = np.diag([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    window = PredictionWindow(
        "window",
        np.array([0]),
        point.reshape(1, 1, 1, 3),
        np.ones((1, 1, 1), dtype=bool),
    )

    result = fuse_windows(
        [window],
        {window.window_id: transform},
        {window.window_id: make_uncertainty(window, 1e-12)},
        method="uniform",
        gauge_covariances={window.window_id: covariance},
    )

    vector = transform.as_vector()
    jacobian = np.empty((3, 7))
    baseline = transform.transform_points(point)
    for index in range(7):
        perturbed = vector.copy()
        perturbed[index] += 1e-6
        jacobian[:, index] = (Sim3.from_vector(perturbed).transform_points(point) - baseline) / 1e-6
    expected = jacobian @ covariance @ jacobian.T
    np.testing.assert_allclose(result.point_covariance[0, 0, 0], expected, rtol=2e-6, atol=2e-6)



def test_covariance_intersection_fusion_is_invariant_to_window_input_order() -> None:
    windows = [
        make_window("window-c", [0, 1], 0.25),
        make_window("window-a", [0, 1], -0.10),
        make_window("window-b", [0, 1], 0.05),
    ]
    gauges = {window.window_id: Sim3.identity() for window in windows}
    uncertainties = {
        "window-a": make_uncertainty(windows[1], 0.8),
        "window-b": make_uncertainty(windows[2], 1.2),
        "window-c": make_uncertainty(windows[0], 2.0),
    }

    reference = fuse_windows(
        windows,
        gauges,
        uncertainties,
        method="covariance_intersection",
    )
    for ordering in permutations(windows):
        candidate = fuse_windows(
            list(ordering),
            gauges,
            uncertainties,
            method="covariance_intersection",
        )
        np.testing.assert_array_equal(candidate.frame_indices, reference.frame_indices)
        np.testing.assert_array_equal(candidate.valid_mask, reference.valid_mask)
        np.testing.assert_array_equal(candidate.contributors, reference.contributors)
        np.testing.assert_allclose(candidate.point_map, reference.point_map)
        np.testing.assert_allclose(
            candidate.point_covariance,
            reference.point_covariance,
        )
