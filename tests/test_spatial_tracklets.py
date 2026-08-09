import numpy as np
import pytest

from prob4d.camera_panel_support import (
    CameraPanelSupportPolicyV1,
    evaluate_camera_panel_tracklet_support,
)
from prob4d.causal_tracklets import CausalTrackletSet
from prob4d.data import PredictionWindow
from prob4d.spatial_tracklets import (
    build_spatially_stratified_scene_flow_tracklets,
    seed_cell_ids_by_track,
    select_spatial_tracklet_seeds,
    spatial_tracklets_to_observation_factors,
)
from prob4d.uncertainty import StructuredCovariance


def moving_window(*, frames: int = 4, height: int = 4, width: int = 8) -> PredictionWindow:
    point_map = np.zeros((frames, height, width, 3), dtype=np.float64)
    rows, columns = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )
    for frame in range(frames):
        point_map[frame, ..., 0] = columns - frame
        point_map[frame, ..., 1] = rows
        point_map[frame, ..., 2] = 1.0
    valid = np.ones((frames, height, width), dtype=bool)
    return PredictionWindow(
        window_id="window",
        frame_indices=np.arange(frames),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=np.zeros_like(point_map),
        deform_mask=np.ones_like(valid),
    )


def panel_tracklets(cell_ids: tuple[int, ...], *, window_id: str) -> CausalTrackletSet:
    track_count = len(cell_ids)
    track_ids = np.repeat(np.arange(track_count, dtype=np.int64), 2)
    frame_indices = np.tile(np.array([0, 1], dtype=np.int64), track_count)
    local_indices = frame_indices.copy()
    rows = np.repeat(np.arange(track_count, dtype=np.int64), 2)
    columns = np.zeros(track_count * 2, dtype=np.int64)
    points = np.zeros((track_count * 2, 3), dtype=np.float64)
    points[:, 2] = 1.0
    return CausalTrackletSet(
        window_id=window_id,
        causal_frame_stop=2,
        source_shape=(2, max(track_count, 1), 1),
        seed_frame_index=0,
        track_ids=track_ids,
        frame_indices=frame_indices,
        local_frame_indices=local_indices,
        rows=rows,
        columns=columns,
        points_local=points,
        link_probability=np.ones(track_count * 2),
        association_probability=np.ones(track_count * 2),
        metadata={
            "seed_cell_grid_shape": [2, 2],
            "seed_cell_ids_by_track": list(cell_ids),
        },
    )


def test_spatial_selection_represents_every_occupied_cell() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    mask[1, 6] = True
    mask[6, 1] = True
    mask[6, 6] = True

    regular = select_spatial_tracklet_seeds(
        mask,
        seed_stride=4,
        seed_selection_policy="regular-grid",
        cell_grid_rows=2,
        cell_grid_columns=2,
    )
    stratified = select_spatial_tracklet_seeds(
        mask,
        seed_stride=4,
        seed_selection_policy="spatial-stratified",
        cell_grid_rows=2,
        cell_grid_columns=2,
        maximum_seeds_per_cell=1,
    )

    assert regular.seed_count == 1
    assert regular.selected_cell_count == 1
    assert stratified.seed_count == 4
    assert stratified.selected_cell_count == 4
    assert stratified.occupied_cell_count == 4
    np.testing.assert_array_equal(stratified.cell_occupancy, [1, 1, 1, 1])


def test_spatial_selection_is_deterministic_and_caps_each_cell() -> None:
    mask = np.ones((8, 8), dtype=bool)
    first = select_spatial_tracklet_seeds(
        mask,
        seed_stride=2,
        cell_grid_rows=2,
        cell_grid_columns=2,
        maximum_seeds_per_cell=2,
    )
    second = select_spatial_tracklet_seeds(
        mask.copy(),
        seed_stride=2,
        cell_grid_rows=2,
        cell_grid_columns=2,
        maximum_seeds_per_cell=2,
    )

    np.testing.assert_array_equal(first.rows, second.rows)
    np.testing.assert_array_equal(first.columns, second.columns)
    np.testing.assert_array_equal(first.cell_ids, second.cell_ids)
    assert first.seed_count == 8
    np.testing.assert_array_equal(first.cell_occupancy, [2, 2, 2, 2])
    assert not first.rows.flags.writeable


def test_spatial_tracklets_retain_cell_lineage_and_causal_cutoff() -> None:
    original = moving_window(frames=5)
    changed_points = original.point_map.copy()
    changed_valid = original.valid_mask.copy()
    changed_flow = original.scene_flow.copy()
    changed_deform = original.deform_mask.copy()
    changed_points[3:] += 1000.0
    changed_valid[3:] = False
    changed_flow[3:] = 1000.0
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
        "cell_grid_rows": 2,
        "cell_grid_columns": 4,
        "maximum_seeds_per_cell": 1,
        "search_radius_pixels": 1,
        "maximum_step_error_local": 0.2,
        "association_sigma_local": 0.1,
        "minimum_link_probability": 0.01,
        "minimum_track_length": 2,
    }

    first, first_report = build_spatially_stratified_scene_flow_tracklets(
        original,
        **settings,
    )
    second, second_report = build_spatially_stratified_scene_flow_tracklets(
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
    np.testing.assert_array_equal(
        seed_cell_ids_by_track(first),
        seed_cell_ids_by_track(second),
    )
    assert first_report.to_dict() == second_report.to_dict()
    assert first_report.seed_selection.selected_cell_count == 8
    assert first_report.retained_seed_cell_count >= 4
    assert first.metadata["seed_selection_policy"] == "spatial-stratified"


def test_spatial_factor_conversion_preserves_cell_groups_and_frame_budget() -> None:
    window = moving_window(frames=2, height=2, width=6)
    tracklets, _ = build_spatially_stratified_scene_flow_tracklets(
        window,
        causal_frame_stop=2,
        seed_stride=2,
        cell_grid_rows=1,
        cell_grid_columns=2,
        maximum_seeds_per_cell=1,
        search_radius_pixels=1,
        maximum_step_error_local=0.2,
        association_sigma_local=0.1,
        minimum_link_probability=0.01,
        minimum_track_length=2,
    )
    rays = np.zeros_like(window.point_map)
    rays[..., 2] = 1.0
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.full(window.shape, 0.04),
        lateral_variance=np.full(window.shape, 0.01),
    )

    spatial = spatial_tracklets_to_observation_factors(
        tracklets,
        covariance,
        view_id="camera-0",
        effective_samples_per_frame=0.5,
    )
    framewise = spatial_tracklets_to_observation_factors(
        tracklets,
        covariance,
        view_id="camera-0",
        correlation_group_mode="frame",
        effective_samples_per_frame=0.5,
    )

    assert len(spatial) == 4
    assert len(framewise) == 2
    assert all("seed-cell" in factor.correlation_group_id for factor in spatial)
    assert {factor.point_ids[0] for factor in spatial[:2]} == {0, 1}
    assert all(factor.composite_weight == 0.25 for factor in spatial)
    assert all(factor.composite_weight == 0.25 for factor in framewise)
    for frame_index in (0, 1):
        spatial_mass = sum(
            factor.composite_weight * len(factor.point_ids)
            for factor in spatial
            if factor.frame_index == frame_index
        )
        framewise_mass = sum(
            factor.composite_weight * len(factor.point_ids)
            for factor in framewise
            if factor.frame_index == frame_index
        )
        assert spatial_mass == pytest.approx(0.5)
        assert spatial_mass == pytest.approx(framewise_mass)


def test_camera_panel_support_requires_multiple_spatially_supported_views() -> None:
    first = panel_tracklets((0, 1, 2), window_id="view-a")
    second = panel_tracklets((2, 3), window_id="view-b")
    declared = ("camera-a", "camera-b")
    policy = CameraPanelSupportPolicyV1(
        minimum_view_count=2,
        minimum_seed_cell_count_per_view=2,
        minimum_supported_frame_fraction=1.0,
        require_all_declared_views=True,
    )

    report = evaluate_camera_panel_tracklet_support(
        {"camera-a": first, "camera-b": second},
        declared_view_ids=declared,
        panel_id="panel",
        required_frame_indices=(0, 1),
        policy=policy,
        metadata={"source_only": True},
    )

    assert report.support_feasible
    assert report.declared_view_ids == declared
    assert report.supported_frame_count == 2
    assert report.frame_results[0].spatially_supported_view_ids == (
        "camera-a",
        "camera-b",
    )
    assert dict(report.frame_results[0].seed_cell_counts_by_view) == {
        "camera-a": 3,
        "camera-b": 2,
    }
    assert len(report.camera_panel_support_id or "") == 64
    replay = evaluate_camera_panel_tracklet_support(
        {"camera-b": second, "camera-a": first},
        declared_view_ids=declared,
        panel_id="panel",
        required_frame_indices=(0, 1),
        policy=policy,
        metadata={"source_only": True},
    )
    assert replay.camera_panel_support_id == report.camera_panel_support_id

    support_negative = evaluate_camera_panel_tracklet_support(
        {
            "camera-a": first,
            "camera-b": panel_tracklets((0,), window_id="view-b-limited"),
        },
        declared_view_ids=declared,
        panel_id="panel-limited",
        required_frame_indices=(0, 1),
        policy=policy,
    )
    assert not support_negative.support_feasible
    assert support_negative.frame_results[0].reason_codes == (
        "insufficient-spatially-supported-views",
    )


def test_camera_panel_support_cannot_drop_a_declared_view() -> None:
    first = panel_tracklets((0, 1, 2), window_id="view-a")
    declared = ("camera-a", "camera-b")
    policy = CameraPanelSupportPolicyV1(
        minimum_view_count=1,
        minimum_seed_cell_count_per_view=2,
        minimum_supported_frame_fraction=1.0,
        require_all_declared_views=True,
    )

    report = evaluate_camera_panel_tracklet_support(
        {"camera-a": first},
        declared_view_ids=declared,
        panel_id="missing-camera-b",
        required_frame_indices=(0, 1),
        policy=policy,
    )

    assert not report.support_feasible
    assert report.declared_view_ids == declared
    assert report.frame_results[0].contributing_view_ids == ("camera-a",)
    assert report.frame_results[0].reason_codes == ("missing-declared-view",)

    with pytest.raises(ValueError, match="outside declared_view_ids"):
        evaluate_camera_panel_tracklet_support(
            {"camera-c": first},
            declared_view_ids=("camera-a",),
            panel_id="unknown-camera",
            required_frame_indices=(0, 1),
            policy=CameraPanelSupportPolicyV1(
                minimum_view_count=1,
                minimum_seed_cell_count_per_view=1,
            ),
        )


def test_spatial_contracts_reject_coercive_aliases() -> None:
    with pytest.raises(ValueError, match="seed_mask"):
        select_spatial_tracklet_seeds(np.ones((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="seed_stride"):
        select_spatial_tracklet_seeds(
            np.ones((2, 2), dtype=bool),
            seed_stride=True,
        )
    with pytest.raises(ValueError, match="seed_selection_policy"):
        select_spatial_tracklet_seeds(
            np.ones((2, 2), dtype=bool),
            seed_selection_policy="adaptive",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="minimum_view_count"):
        CameraPanelSupportPolicyV1(minimum_view_count=True)
    with pytest.raises(ValueError, match="required_frame_indices"):
        evaluate_camera_panel_tracklet_support(
            {"camera-a": panel_tracklets((0,), window_id="view-a")},
            declared_view_ids=("camera-a",),
            panel_id="panel",
            required_frame_indices=(1, 0),
            policy=CameraPanelSupportPolicyV1(
                minimum_view_count=1,
                minimum_seed_cell_count_per_view=1,
            ),
        )
