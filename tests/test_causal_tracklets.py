import numpy as np
import pytest

from prob4d.causal_tracklets import (
    build_causal_scene_flow_tracklets,
    tracklets_to_observation_factors,
)
from prob4d.data import PredictionWindow
from prob4d.uncertainty import StructuredCovariance


def moving_window(*, frames: int = 4) -> PredictionWindow:
    height = 3
    width = 6
    point_map = np.zeros((frames, height, width, 3), dtype=np.float64)
    for frame in range(frames):
        rows, columns = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        point_map[frame, ..., 0] = columns - frame
        point_map[frame, ..., 1] = rows
        point_map[frame, ..., 2] = 1.0
    valid = np.ones((frames, height, width), dtype=bool)
    flow = np.zeros_like(point_map)
    deform = np.ones_like(valid)
    return PredictionWindow(
        window_id="window",
        frame_indices=np.arange(frames),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
    )


def test_scene_flow_tracklets_preserve_identity_across_frames() -> None:
    tracklets, report = build_causal_scene_flow_tracklets(
        moving_window(),
        causal_frame_stop=4,
        seed_stride=2,
        search_radius_pixels=1,
        maximum_step_error_local=0.2,
        association_sigma_local=0.1,
        minimum_link_probability=0.01,
        minimum_track_length=2,
    )

    first = np.flatnonzero(tracklets.track_ids == 0)
    np.testing.assert_array_equal(tracklets.frame_indices[first], [0, 1, 2, 3])
    np.testing.assert_array_equal(tracklets.columns[first], [0, 1, 2, 3])
    np.testing.assert_allclose(tracklets.association_probability[first], 1.0)
    assert report.seed_count == 6
    assert report.retained_track_count == 6
    assert tracklets.summary()["maximum_track_length"] == 4


def test_tracklets_do_not_read_post_cutoff_prediction_payloads() -> None:
    original = moving_window(frames=5)
    changed_points = original.point_map.copy()
    changed_valid = original.valid_mask.copy()
    changed_flow = original.scene_flow.copy()
    changed_deform = original.deform_mask.copy()
    changed_points[3:] += 10_000.0
    changed_valid[3:] = False
    changed_flow[3:] = 10_000.0
    changed_deform[3:] = False
    changed = PredictionWindow(
        window_id=original.window_id,
        frame_indices=original.frame_indices,
        point_map=changed_points,
        valid_mask=changed_valid,
        scene_flow=changed_flow,
        deform_mask=changed_deform,
    )

    settings = {
        "causal_frame_stop": 3,
        "seed_stride": 2,
        "search_radius_pixels": 1,
        "maximum_step_error_local": 0.2,
        "association_sigma_local": 0.1,
        "minimum_link_probability": 0.01,
        "minimum_track_length": 2,
    }
    first, first_report = build_causal_scene_flow_tracklets(
        original,
        **settings,
    )
    second, second_report = build_causal_scene_flow_tracklets(
        changed,
        **settings,
    )

    for name in (
        "track_ids",
        "frame_indices",
        "local_frame_indices",
        "rows",
        "columns",
        "points_local",
        "link_probability",
        "association_probability",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first_report.to_dict() == second_report.to_dict()


def test_collision_resolution_keeps_one_deterministic_track() -> None:
    point_map = np.zeros((2, 1, 3, 3), dtype=np.float64)
    point_map[0, 0, 0, 0] = 0.0
    point_map[0, 0, 2, 0] = 2.0
    point_map[1, 0, 1, 0] = 1.0
    point_map[..., 2] = 1.0
    valid = np.zeros((2, 1, 3), dtype=bool)
    valid[0, 0, [0, 2]] = True
    valid[1, 0, 1] = True
    flow = np.zeros_like(point_map)
    flow[0, 0, 0, 0] = 1.0
    flow[0, 0, 2, 0] = -1.0
    deform = np.zeros_like(valid)
    deform[0, 0, [0, 2]] = True
    window = PredictionWindow(
        window_id="collision",
        frame_indices=np.array([0, 1]),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
    )

    tracklets, report = build_causal_scene_flow_tracklets(
        window,
        causal_frame_stop=2,
        seed_stride=2,
        search_radius_pixels=1,
        maximum_step_error_local=0.1,
        association_sigma_local=0.05,
        minimum_link_probability=0.01,
        minimum_track_length=2,
    )

    assert tracklets.track_count == 1
    np.testing.assert_array_equal(tracklets.columns, [0, 1])
    assert report.collision_rejections == 1
    assert report.dropped_short_tracks == 1


def test_factor_conversion_keeps_association_and_reliability_separate() -> None:
    point_map = np.zeros((2, 1, 1, 3), dtype=np.float64)
    point_map[..., 2] = 1.0
    point_map[1, 0, 0, 0] = 0.05
    valid = np.ones((2, 1, 1), dtype=bool)
    flow = np.zeros_like(point_map)
    deform = np.ones_like(valid)
    window = PredictionWindow(
        window_id="single",
        frame_indices=np.array([4, 5]),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
    )
    tracklets, _ = build_causal_scene_flow_tracklets(
        window,
        causal_frame_stop=6,
        seed_stride=1,
        search_radius_pixels=0,
        maximum_step_error_local=0.2,
        association_sigma_local=0.1,
        minimum_link_probability=0.01,
        minimum_track_length=2,
    )

    rays = np.zeros_like(point_map)
    rays[..., 2] = 1.0
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.full(window.shape, 0.04),
        lateral_variance=np.full(window.shape, 0.01),
    )
    reliability = np.full(window.shape, 0.25)
    factors = tracklets_to_observation_factors(
        tracklets,
        covariance,
        view_id="camera0",
        prior_reliability=reliability,
        effective_samples_per_group=0.5,
    )

    assert len(factors) == 2
    assert factors[1].association_probability[0] < 1.0
    np.testing.assert_allclose(factors[1].prior_reliability, 0.25)
    assert factors[1].composite_weight == 0.5
    assert factors[0].point_ids[0] == factors[1].point_ids[0]
    np.testing.assert_allclose(
        factors[0].local_covariance_m2[0],
        np.diag([0.01, 0.01, 0.04]),
    )


def test_tracklet_builder_requires_scene_flow() -> None:
    point_map = np.zeros((2, 1, 1, 3), dtype=np.float64)
    point_map[..., 2] = 1.0
    window = PredictionWindow(
        window_id="no-flow",
        frame_indices=np.array([0, 1]),
        point_map=point_map,
        valid_mask=np.ones((2, 1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="scene_flow"):
        build_causal_scene_flow_tracklets(
            window,
            causal_frame_stop=2,
        )
