import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from prob4d.phystwin import (
    CoverResizeCrop,
    PhysTwinCase,
    directed_nearest_distances,
    point_set_metrics,
    sample_vector_field_nearest,
)


def make_case(tmp_path: Path) -> PhysTwinCase:
    root = tmp_path / "case"
    (root / "depth" / "0").mkdir(parents=True)
    (root / "mask").mkdir()
    (root / "color").mkdir()
    metadata = {
        "intrinsics": [[[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]],
        "WH": [4, 3],
        "frame_num": 2,
        "fps": 30,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (root / "calibrate.pkl").open("wb") as handle:
        pickle.dump(np.eye(4)[None], handle)
    depth_mm = np.full((3, 4), 1000.0)
    np.save(root / "depth" / "0" / "0.npy", depth_mm)
    np.save(root / "depth" / "0" / "1.npy", depth_mm)
    mask = np.ones((3, 4), dtype=bool)
    masks = {
        frame: {0: {"object": mask, "controller": np.zeros_like(mask)}}
        for frame in range(2)
    }
    with (root / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    return PhysTwinCase.from_directory(root)


def test_cover_crop_coordinate_round_trip() -> None:
    crop = CoverResizeCrop.from_shapes(480, 848, 320, 640)
    source = np.array([[0.0, 0.0], [424.0, 240.0], [847.0, 479.0]])

    recovered = crop.target_to_source(crop.source_to_target(source))

    np.testing.assert_allclose(recovered, source, atol=1e-12)
    assert crop.resized_height == 362
    assert crop.resized_width == 640
    assert crop.crop_row == 21


def test_phystwin_metric_point_map_and_projection_round_trip(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    crop = CoverResizeCrop.from_shapes(3, 4, 3, 4)

    points, valid = case.metric_point_map(0, 0, crop)
    pixels, depth = case.project_world(points[valid], 0)

    assert np.all(valid)
    assert np.allclose(depth, 1.0)
    source_grid = crop.target_source_grid()[valid]
    np.testing.assert_allclose(pixels, source_grid, atol=1e-7)


def test_phystwin_metric_truth_preserves_absolute_frames(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    crop = CoverResizeCrop.from_shapes(3, 4, 3, 4)

    truth = case.metric_truth(np.array([1]), 0, crop)

    np.testing.assert_array_equal(truth.frame_indices, [1])
    assert truth.point_map.shape == (1, 3, 4, 3)
    assert np.all(truth.valid_mask)


def test_nearest_field_sampling_reports_out_of_bounds() -> None:
    rows, columns = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    field = np.stack((columns, rows, np.ones_like(rows)), axis=-1).astype(float)

    sampled, active = sample_vector_field_nearest(
        field,
        np.array([[1.1, 0.2], [-2.0, 0.0], [2.0, 1.0], [np.nan, 0.0]]),
    )

    np.testing.assert_allclose(sampled[[0, 2]], [[1.0, 0.0, 1.0], [2.0, 1.0, 1.0]])
    np.testing.assert_array_equal(active, [True, False, True, False])


def test_point_set_metrics_are_symmetric_and_metric() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = source + np.array([0.01, 0.0, 0.0])

    distances = directed_nearest_distances(source, target)
    metrics = point_set_metrics(source, target)

    np.testing.assert_allclose(distances, 0.01)
    assert metrics.symmetric_mean_m == pytest.approx(0.01)
