import numpy as np
import pytest

from prob4d.causal_tracklets import (
    CausalTrackletReport,
    CausalTrackletSet,
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


def minimal_tracklet_kwargs() -> dict[str, object]:
    return {
        "window_id": "window",
        "causal_frame_stop": 2,
        "source_shape": (2, 1, 1),
        "seed_frame_index": 0,
        "track_ids": np.array([0, 0], dtype=np.int64),
        "frame_indices": np.array([0, 1], dtype=np.int64),
        "local_frame_indices": np.array([0, 1], dtype=np.int64),
        "rows": np.array([0, 0], dtype=np.int64),
        "columns": np.array([0, 0], dtype=np.int64),
        "points_local": np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        "link_probability": np.array([1.0, 0.5]),
        "association_probability": np.array([1.0, 0.5]),
    }


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
    assert report.target_deform_mask_policy == "allow"
    assert tracklets.metadata["target_deform_mask_policy"] == "allow"
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
    first, first_report = build_causal_scene_flow_tracklets(original, **settings)
    second, second_report = build_causal_scene_flow_tracklets(changed, **settings)

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


def test_target_deform_mask_policy_is_explicit_and_audited() -> None:
    point_map = np.zeros((2, 1, 2, 3), dtype=np.float64)
    point_map[..., 0] = np.array([[[0.0, 1.0]], [[0.0, 1.0]]])
    point_map[..., 2] = 1.0
    valid = np.ones((2, 1, 2), dtype=bool)
    deform = np.ones_like(valid)
    deform[1, 0, 1] = False
    window = PredictionWindow(
        window_id="target-mask",
        frame_indices=np.array([0, 1]),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=np.zeros_like(point_map),
        deform_mask=deform,
    )
    settings = {
        "causal_frame_stop": 2,
        "seed_stride": 1,
        "search_radius_pixels": 0,
        "maximum_step_error_local": 0.1,
        "minimum_link_probability": 0.01,
        "minimum_track_length": 2,
    }

    allowed, allowed_report = build_causal_scene_flow_tracklets(window, **settings)
    required, required_report = build_causal_scene_flow_tracklets(
        window,
        target_deform_mask_policy="require",
        **settings,
    )

    assert allowed.track_count == 2
    assert allowed_report.terminated_target_mask == 0
    assert allowed_report.target_deform_mask_policy == "allow"
    assert allowed.metadata["target_deform_mask_policy"] == "allow"
    assert required.track_count == 1
    assert required_report.terminated_target_mask == 1
    assert required_report.target_deform_mask_policy == "require"


def test_tracklet_contract_rejects_scalar_and_array_coercion_aliases() -> None:
    kwargs = minimal_tracklet_kwargs()
    with pytest.raises(ValueError, match="window_id"):
        CausalTrackletSet(**{**kwargs, "window_id": 1})
    with pytest.raises(ValueError, match="causal_frame_stop"):
        CausalTrackletSet(**{**kwargs, "causal_frame_stop": True})
    with pytest.raises(ValueError, match=r"source_shape\[1\]"):
        CausalTrackletSet(**{**kwargs, "source_shape": (2, 1.0, 1)})
    with pytest.raises(ValueError, match="track_ids"):
        CausalTrackletSet(
            **{**kwargs, "track_ids": np.array([0.0, 0.0], dtype=np.float64)}
        )
    with pytest.raises(ValueError, match="link_probability"):
        CausalTrackletSet(
            **{**kwargs, "link_probability": np.array([True, False])}
        )


def test_builder_and_report_reject_boolean_numeric_aliases() -> None:
    with pytest.raises(ValueError, match="seed_stride"):
        build_causal_scene_flow_tracklets(
            moving_window(),
            causal_frame_stop=4,
            seed_stride=True,
        )
    with pytest.raises(ValueError, match="maximum_step_error_local"):
        build_causal_scene_flow_tracklets(
            moving_window(),
            causal_frame_stop=4,
            maximum_step_error_local=False,
        )
    report_kwargs = {
        "seed_count": 1,
        "retained_track_count": 1,
        "observation_count": 2,
        "dropped_short_tracks": 0,
        "terminated_invalid_source": 0,
        "terminated_no_candidate": 0,
        "terminated_target_mask": 0,
        "terminated_step_error": 0,
        "terminated_low_probability": 0,
        "collision_rejections": 0,
        "seed_stride": 1,
        "search_radius_pixels": 1,
        "maximum_step_error_local": 0.1,
        "association_sigma_local": 0.05,
        "minimum_link_probability": 0.1,
        "minimum_track_length": 2,
    }
    with pytest.raises(ValueError, match="seed_count"):
        CausalTrackletReport(**{**report_kwargs, "seed_count": True})


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

    with pytest.raises(ValueError, match="view_id"):
        tracklets_to_observation_factors(tracklets, covariance, view_id=1)
    with pytest.raises(ValueError, match="prior_nominal_probability"):
        tracklets_to_observation_factors(
            tracklets,
            covariance,
            view_id="camera0",
            prior_nominal_probability=True,
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
        build_causal_scene_flow_tracklets(window, causal_frame_stop=2)
