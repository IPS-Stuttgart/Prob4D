from __future__ import annotations

import numpy as np
import pytest

from prob4d.data import PredictionWindow


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray([3, 4], dtype=np.int64)
    points = np.asarray(
        [
            [[[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]]],
            [[[1.5, 0.0, 1.0], [0.0, 2.5, 1.0]]],
        ]
    )
    valid = np.ones(points.shape[:-1], dtype=bool)
    return frames, points, valid


def test_prediction_window_defensively_copies_and_freezes_arrays() -> None:
    frames, points, valid = _inputs()
    window = PredictionWindow(
        window_id="window-a",
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
    )

    frames[0] = 99
    points[0, 0, 0] = 99.0
    valid[0, 0, 0] = False

    np.testing.assert_array_equal(window.frame_indices, [3, 4])
    np.testing.assert_allclose(window.point_map[0, 0, 0], [1.0, 0.0, 1.0])
    assert bool(window.valid_mask[0, 0, 0])
    assert not window.frame_indices.flags.writeable
    assert not window.point_map.flags.writeable
    assert not window.valid_mask.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        window.point_map[0, 0, 0, 0] = 2.0


def test_prediction_window_normalizes_and_freezes_optional_arrays() -> None:
    frames, points, valid = _inputs()
    flow = np.ones_like(points)
    deform = valid.copy()
    rays = 2.0 * points
    window = PredictionWindow(
        window_id="window-a",
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
        ray_directions=rays,
    )

    np.testing.assert_allclose(
        np.linalg.norm(window.ray_directions[window.valid_mask], axis=-1),
        1.0,
    )
    assert not window.scene_flow.flags.writeable
    assert not window.deform_mask.flags.writeable
    assert not window.ray_directions.flags.writeable

    flow[...] = 7.0
    rays[...] = 0.0
    np.testing.assert_allclose(window.scene_flow, 1.0)
    assert np.all(np.linalg.norm(window.ray_directions[window.valid_mask], axis=-1) > 0.0)


def test_prediction_window_rejects_nonfinite_active_optional_values() -> None:
    frames, points, valid = _inputs()
    flow = np.ones_like(points)
    flow[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="active scene_flow"):
        PredictionWindow(
            window_id="window-a",
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            scene_flow=flow,
            deform_mask=valid,
        )

    rays = points.copy()
    rays[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="ray directions must be nonzero"):
        PredictionWindow(
            window_id="window-a",
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            ray_directions=rays,
        )
