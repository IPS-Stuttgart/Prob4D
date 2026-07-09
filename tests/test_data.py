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

