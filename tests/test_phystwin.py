import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from prob4d.phystwin import (
    CoverResizeCrop,
    PhysTwinCase,
    directed_nearest_distances,
    nearest_neighbor_indices,
    point_set_metrics,
    sample_vector_field_nearest,
)


def mark_pickle_execution(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class UnsafePicklePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return mark_pickle_execution, (str(self.marker),)


def write_case(
    tmp_path: Path,
    *,
    intrinsics: object | None = None,
    camera_to_world: object | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> Path:
    root = tmp_path / "case"
    (root / "depth" / "0").mkdir(parents=True)
    (root / "mask").mkdir()
    (root / "color").mkdir()
    metadata: dict[str, object] = {
        "intrinsics": (
            [[[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]]
            if intrinsics is None
            else intrinsics
        ),
        "WH": [4, 3],
        "frame_num": 2,
        "fps": 30,
    }
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    calibration = np.eye(4)[None] if camera_to_world is None else camera_to_world
    with (root / "calibrate.pkl").open("wb") as handle:
        pickle.dump(calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    depth_mm = np.full((3, 4), 1000.0)
    np.save(root / "depth" / "0" / "0.npy", depth_mm)
    np.save(root / "depth" / "0" / "1.npy", depth_mm)
    mask = np.ones((3, 4), dtype=bool)
    masks = {
        frame: {0: {"object": mask, "controller": np.zeros_like(mask)}}
        for frame in range(2)
    }
    with (root / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return root


def make_case(tmp_path: Path) -> PhysTwinCase:
    return PhysTwinCase.from_directory(write_case(tmp_path))


def test_cover_crop_coordinate_round_trip() -> None:
    crop = CoverResizeCrop.from_shapes(480, 848, 320, 640)
    source = np.array([[0.0, 0.0], [424.0, 240.0], [847.0, 479.0]])

    recovered = crop.target_to_source(crop.source_to_target(source))

    np.testing.assert_allclose(recovered, source, atol=1e-12)
    assert crop.resized_height == 362
    assert crop.resized_width == 640
    assert crop.crop_row == 21


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_cover_crop_rejects_coercive_or_invalid_dimensions(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CoverResizeCrop.from_shapes(invalid, 848, 320, 640)  # type: ignore[arg-type]


def test_phystwin_metric_point_map_and_projection_round_trip(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    crop = CoverResizeCrop.from_shapes(3, 4, 3, 4)

    points, valid = case.metric_point_map(0, 0, crop)
    pixels, depth = case.project_world(points[valid], 0)

    assert np.all(valid)
    assert np.allclose(depth, 1.0)
    source_grid = crop.target_source_grid()[valid]
    np.testing.assert_allclose(pixels, source_grid, atol=1e-7)
    assert not case.intrinsics.flags.writeable
    assert not case.camera_to_world.flags.writeable


def test_phystwin_projection_preserves_nonfinite_track_placeholders(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    points = np.array([[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]])

    pixels, depth = case.project_world(points, 0)

    np.testing.assert_allclose(pixels[0], [1.5, 1.0])
    assert depth[0] == pytest.approx(1.0)
    assert np.all(np.isnan(pixels[1]))
    assert np.isnan(depth[1])


def test_phystwin_metric_truth_preserves_absolute_frames(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    crop = CoverResizeCrop.from_shapes(3, 4, 3, 4)

    truth = case.metric_truth(np.array([1]), 0, crop)

    np.testing.assert_array_equal(truth.frame_indices, [1])
    assert truth.point_map.shape == (1, 3, 4, 3)
    assert np.all(truth.valid_mask)


def test_phystwin_rejects_nonintegral_frame_indices(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    crop = CoverResizeCrop.from_shapes(3, 4, 3, 4)

    with pytest.raises(ValueError, match="must contain integers"):
        case.metric_truth(np.array([1.0]), 0, crop)


def test_phystwin_rejects_invalid_camera_geometry(tmp_path: Path) -> None:
    transform = np.eye(4)[None]
    transform[0, 0, 0] = 2.0
    root = write_case(tmp_path, camera_to_world=transform)

    with pytest.raises(ValueError, match="rotations must be orthonormal"):
        PhysTwinCase.from_directory(root)


def test_phystwin_rejects_nonfinite_camera_transforms(tmp_path: Path) -> None:
    transform = np.eye(4)[None]
    transform[0, 0, 3] = np.nan
    root = write_case(tmp_path, camera_to_world=transform)

    with pytest.raises(ValueError, match="camera transforms must be finite"):
        PhysTwinCase.from_directory(root)


def test_phystwin_rejects_executable_legacy_pickle(tmp_path: Path) -> None:
    root = write_case(tmp_path)
    marker = tmp_path / "pickle-executed"
    with (root / "calibrate.pkl").open("wb") as handle:
        pickle.dump(UnsafePicklePayload(marker), handle)

    with pytest.raises(ValueError, match="invalid or unsafe PhysTwin calibration pickle"):
        PhysTwinCase.from_directory(root)
    assert not marker.exists()


def test_phystwin_rejects_nonfinite_json_constants(tmp_path: Path) -> None:
    root = write_case(tmp_path)
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        '{"intrinsics": [[[2, 0, 1.5], [0, 2, 1], [0, 0, 1]]], '
        '"WH": [4, 3], "frame_num": 2, "fps": NaN}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        PhysTwinCase.from_directory(root)


def test_phystwin_rejects_duplicate_metadata_keys(tmp_path: Path) -> None:
    root = write_case(tmp_path)
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        '{"intrinsics": [[[2, 0, 1.5], [0, 2, 1], [0, 0, 1]]], '
        '"WH": [4, 3], "frame_num": 2, "fps": 30, "fps": 60}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: fps"):
        PhysTwinCase.from_directory(root)


def test_nearest_field_sampling_reports_out_of_bounds() -> None:
    rows, columns = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    field = np.stack((columns, rows, np.ones_like(rows)), axis=-1).astype(float)

    sampled, active = sample_vector_field_nearest(
        field,
        np.array([[1.1, 0.2], [-2.0, 0.0], [2.0, 1.0], [np.nan, 0.0]]),
    )

    np.testing.assert_allclose(sampled[[0, 2]], [[1.0, 0.0, 1.0], [2.0, 1.0, 1.0]])
    np.testing.assert_array_equal(active, [True, False, True, False])


def test_nearest_neighbor_rejects_invalid_chunk_sizes() -> None:
    points = np.zeros((1, 3))

    for invalid in (True, 0, -1, 1.5):
        with pytest.raises((TypeError, ValueError)):
            nearest_neighbor_indices(points, points, chunk_size=invalid)  # type: ignore[arg-type]

    indices, distances = nearest_neighbor_indices(
        points,
        points,
        chunk_size=np.int64(1),
    )
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_array_equal(distances, [0.0])


def test_nearest_neighbor_rejects_nonfinite_points() -> None:
    finite = np.zeros((1, 3))
    nonfinite = np.array([[np.nan, 0.0, 0.0]])

    with pytest.raises(ValueError, match="point sets must be finite"):
        nearest_neighbor_indices(nonfinite, finite)
    with pytest.raises(ValueError, match="point sets must be finite"):
        nearest_neighbor_indices(finite, nonfinite)


def test_point_set_metrics_are_symmetric_and_metric() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = source + np.array([0.01, 0.0, 0.0])

    distances = directed_nearest_distances(source, target)
    metrics = point_set_metrics(source, target)

    np.testing.assert_allclose(distances, 0.01)
    assert metrics.symmetric_mean_m == pytest.approx(0.01)
