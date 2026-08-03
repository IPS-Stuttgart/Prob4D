from itertools import permutations

import numpy as np
import pytest

import prob4d.fusion as fusion_module
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


@pytest.mark.parametrize("method", ["uniform", "precision", "covariance_intersection"])
def test_fuse_windows_spatial_tiling_preserves_all_fusion_methods(method: str) -> None:
    frames = [0]
    windows = [
        make_window("first", frames, -0.5),
        make_window("second", frames, 0.25),
        make_window("third", frames, 1.0),
    ]
    windows = [
        PredictionWindow(
            window.window_id,
            window.frame_indices,
            np.tile(window.point_map, (1, 2, 3, 1)),
            np.asarray(
                [
                    [
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                    ]
                ]
            ),
        )
        for window in windows
    ]
    windows[1] = PredictionWindow(
        windows[1].window_id,
        windows[1].frame_indices,
        windows[1].point_map,
        np.asarray(
            [
                [
                    [True, True, True, False, False, False],
                    [True, True, True, False, False, False],
                ]
            ]
        ),
    )
    windows[2] = PredictionWindow(
        windows[2].window_id,
        windows[2].frame_indices,
        windows[2].point_map,
        np.asarray(
            [
                [
                    [True, False, True, False, True, False],
                    [True, False, True, False, True, False],
                ]
            ]
        ),
    )
    gauges = {window.window_id: Sim3.identity() for window in windows}
    uncertainties = {
        window.window_id: make_uncertainty(window, float(index + 1))
        for index, window in enumerate(windows)
    }

    untiled = fuse_windows(
        windows,
        gauges,
        uncertainties,
        method=method,
        fusion_tile_size=10_000,
    )
    tiled = fuse_windows(
        windows,
        gauges,
        uncertainties,
        method=method,
        fusion_tile_size=2,
    )

    np.testing.assert_array_equal(tiled.valid_mask, untiled.valid_mask)
    np.testing.assert_array_equal(tiled.contributors, untiled.contributors)
    np.testing.assert_allclose(tiled.point_map, untiled.point_map, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        tiled.point_covariance,
        untiled.point_covariance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_covariance_intersection_tiles_reuse_global_pattern_weights() -> None:
    points = np.zeros((1, 1, 6, 3))
    first = PredictionWindow(
        "first",
        np.asarray([0]),
        points,
        np.ones((1, 1, 6), dtype=bool),
    )
    second_points = points.copy()
    second_points[..., 0] = 10.0
    second = PredictionWindow(
        "second",
        np.asarray([0]),
        second_points,
        np.ones((1, 1, 6), dtype=bool),
    )
    rays = np.zeros_like(points)
    rays[..., 2] = 1.0
    first_variance = np.asarray([[[1.0, 1.0, 1.0, 100.0, 100.0, 100.0]]])
    second_variance = first_variance[..., ::-1].copy()
    uncertainties = {
        "first": StructuredCovariance(rays, first_variance, first_variance),
        "second": StructuredCovariance(rays, second_variance, second_variance),
    }
    gauges = {"first": Sim3.identity(), "second": Sim3.identity()}

    global_batch = fuse_windows(
        [first, second],
        gauges,
        uncertainties,
        method="covariance_intersection",
        fusion_tile_size=6,
    )
    one_pixel_tiles = fuse_windows(
        [first, second],
        gauges,
        uncertainties,
        method="covariance_intersection",
        fusion_tile_size=1,
    )

    np.testing.assert_allclose(one_pixel_tiles.point_map, global_batch.point_map)
    np.testing.assert_allclose(
        one_pixel_tiles.point_covariance,
        global_batch.point_covariance,
    )
    assert np.all(one_pixel_tiles.point_map[..., 0] > 0.0)
    assert np.all(one_pixel_tiles.point_map[..., 0] < 10.0)


def test_fuse_windows_rejects_nonpositive_tile_size() -> None:
    window = make_window("window", [0], 0.0)

    with pytest.raises(ValueError, match="fusion_tile_size must be positive"):
        fuse_windows(
            [window],
            {window.window_id: Sim3.identity()},
            {window.window_id: make_uncertainty(window, 1.0)},
            method="uniform",
            fusion_tile_size=0,
        )


def test_structured_covariance_expansion_is_bounded_by_sample_or_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point_count = 5_000
    points = np.zeros((1, 1, point_count, 3))
    mask = np.ones((1, 1, point_count), dtype=bool)
    first = PredictionWindow("first", np.asarray([0]), points, mask)
    second_points = points.copy()
    second_points[..., 0] = 1.0
    second = PredictionWindow("second", np.asarray([0]), second_points, mask)
    rays = np.zeros_like(points)
    rays[..., 2] = 1.0
    uncertainty = StructuredCovariance(
        rays,
        np.ones((1, 1, point_count)),
        np.ones((1, 1, point_count)),
    )
    observed_sizes: list[int] = []
    original = fusion_module._structured_covariance_rows

    def tracked_rows(
        structured: StructuredCovariance,
        transform: Sim3,
        local_index: int,
        flat_indices: np.ndarray,
    ) -> np.ndarray:
        observed_sizes.append(int(flat_indices.size))
        return original(structured, transform, local_index, flat_indices)

    monkeypatch.setattr(fusion_module, "_structured_covariance_rows", tracked_rows)
    result = fuse_windows(
        [first, second],
        {"first": Sim3.identity(), "second": Sim3.identity()},
        {"first": uncertainty, "second": uncertainty},
        method="covariance_intersection",
        fusion_tile_size=64,
    )

    assert result.point_map.shape == (1, 1, point_count, 3)
    assert observed_sizes
    assert max(observed_sizes) == 4_096
    assert point_count not in observed_sizes
