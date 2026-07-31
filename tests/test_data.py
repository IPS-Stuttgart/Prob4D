from pathlib import Path

import numpy as np
import pytest

from prob4d.data import PredictionWindow


def make_window() -> PredictionWindow:
    points = np.arange(2 * 3 * 4 * 3, dtype=float).reshape(2, 3, 4, 3)
    return PredictionWindow(
        window_id="w0",
        frame_indices=np.array([10, 11]),
        point_map=points,
        valid_mask=np.ones((2, 3, 4), dtype=bool),
        scene_flow=np.ones_like(points),
        deform_mask=np.ones((2, 3, 4), dtype=bool),
    )


def test_prediction_window_round_trip(tmp_path: Path) -> None:
    expected = make_window()
    path = tmp_path / "window.npz"

    expected.to_npz(path)
    actual = PredictionWindow.from_npz(path)

    assert actual.window_id == expected.window_id
    np.testing.assert_array_equal(actual.frame_indices, expected.frame_indices)
    np.testing.assert_allclose(actual.point_map, expected.point_map)
    np.testing.assert_allclose(actual.scene_flow, expected.scene_flow)


def test_upstream_file_requires_explicit_start_frame(tmp_path: Path) -> None:
    path = tmp_path / "upstream.npz"
    np.savez(
        path,
        point_map=np.zeros((2, 1, 1, 3)),
        valid_mask=np.ones((2, 1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="start_frame"):
        PredictionWindow.from_npz(path)

    window = PredictionWindow.from_npz(path, start_frame=7)
    np.testing.assert_array_equal(window.frame_indices, [7, 8])


def test_window_rejects_misaligned_metadata() -> None:
    with pytest.raises(ValueError, match="time dimension"):
        PredictionWindow(
            window_id="bad",
            frame_indices=np.array([0, 1]),
            point_map=np.zeros((3, 1, 1, 3)),
            valid_mask=np.ones((3, 1, 1), dtype=bool),
        )


def test_window_rejects_negative_absolute_frame_ids() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PredictionWindow(
            window_id="bad",
            frame_indices=np.array([-1, 0]),
            point_map=np.ones((2, 1, 1, 3)),
            valid_mask=np.ones((2, 1, 1), dtype=bool),
        )


def test_window_rejects_deformation_outside_valid_geometry() -> None:
    valid = np.array([[[True, False]]])
    deform = np.array([[[False, True]]])
    with pytest.raises(ValueError, match="subset"):
        PredictionWindow(
            window_id="bad",
            frame_indices=np.array([0]),
            point_map=np.ones((1, 1, 2, 3)),
            valid_mask=valid,
            scene_flow=np.ones((1, 1, 2, 3)),
            deform_mask=deform,
        )


def test_window_rejects_nonfinite_active_scene_flow() -> None:
    flow = np.ones((1, 1, 1, 3))
    flow[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="scene_flow"):
        PredictionWindow(
            window_id="bad",
            frame_indices=np.array([0]),
            point_map=np.ones((1, 1, 1, 3)),
            valid_mask=np.ones((1, 1, 1), dtype=bool),
            scene_flow=flow,
            deform_mask=np.ones((1, 1, 1), dtype=bool),
        )


@pytest.mark.parametrize("bad_value", [0.0, np.nan])
def test_window_rejects_invalid_active_ray_directions(bad_value: float) -> None:
    rays = np.ones((1, 1, 1, 3))
    rays[0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="ray_directions"):
        PredictionWindow(
            window_id="bad",
            frame_indices=np.array([0]),
            point_map=np.ones((1, 1, 1, 3)),
            valid_mask=np.ones((1, 1, 1), dtype=bool),
            ray_directions=rays,
        )


def test_explicit_float32_storage_halves_dense_vectors_and_keeps_ray_parity(
    tmp_path: Path,
) -> None:
    points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    points[..., 2] += 1.0
    flow = np.ones_like(points)
    valid = np.ones((2, 3, 4), dtype=bool)
    deform = np.ones_like(valid)
    legacy = PredictionWindow(
        "legacy",
        np.array([0, 1]),
        points,
        valid,
        scene_flow=flow,
        deform_mask=deform,
    )
    compact = PredictionWindow(
        "compact",
        np.array([0, 1]),
        points,
        valid,
        scene_flow=flow,
        deform_mask=deform,
        dense_storage_dtype="float32",
    )

    assert compact.point_map.dtype == np.float32
    assert compact.scene_flow.dtype == np.float32
    assert legacy.point_map.dtype == np.float64
    assert compact.dense_vector_storage_bytes * 2 == legacy.dense_vector_storage_bytes
    np.testing.assert_array_equal(
        compact.rays_at(1, dtype=np.float64),
        legacy.rays_at(1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        compact.rays(dtype=np.float64)[1],
        compact.rays_at(1, dtype=np.float64),
    )
    with pytest.raises(ValueError):
        compact.point_map[0, 0, 0, 0] = 1.0

    path = tmp_path / "compact.npz"
    compact.to_npz(path)
    restored = PredictionWindow.from_npz(
        path,
        dense_storage_dtype="float32",
    )
    assert restored.point_map.dtype == np.float32
    assert restored.scene_flow.dtype == np.float32
    np.testing.assert_array_equal(restored.point_map, compact.point_map)


def test_window_rejects_unknown_dense_storage_dtype() -> None:
    with pytest.raises(ValueError, match="dense_storage_dtype"):
        PredictionWindow(
            window_id="bad-storage",
            frame_indices=np.array([0]),
            point_map=np.ones((1, 1, 1, 3)),
            valid_mask=np.ones((1, 1, 1), dtype=bool),
            dense_storage_dtype="float16",  # type: ignore[arg-type]
        )
