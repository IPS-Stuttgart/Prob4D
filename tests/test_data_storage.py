from pathlib import Path

import numpy as np
import pytest

from prob4d.data import (
    PREDICTION_WINDOW_NPZ_SCHEMA,
    PREDICTION_WINDOW_NPZ_VERSION,
    PredictionWindow,
)


def _window(dtype: str = "float64") -> PredictionWindow:
    points = np.array(
        [[[[1.0000000000000002, 2.0000000000000004, 3.0000000000000004]]]],
        dtype=np.float64,
    )
    flow = points / 7.0
    rays = np.array([[[[0.1, 0.2, 0.3]]]], dtype=np.float64)
    return PredictionWindow(
        window_id="w",
        frame_indices=np.array([3]),
        point_map=points,
        valid_mask=np.ones((1, 1, 1), dtype=bool),
        scene_flow=flow,
        deform_mask=np.ones((1, 1, 1), dtype=bool),
        ray_directions=rays,
        dense_storage_dtype=dtype,
    )


def test_versioned_round_trip_preserves_float64_precision(tmp_path: Path) -> None:
    expected = _window()
    path = tmp_path / "window.npz"
    expected.to_npz(path)

    with np.load(path, allow_pickle=False) as payload:
        assert payload["schema_name"].item() == PREDICTION_WINDOW_NPZ_SCHEMA
        assert payload["schema_version"].item() == PREDICTION_WINDOW_NPZ_VERSION
        assert payload["dense_storage_dtype"].item() == "float64"
        assert payload["point_map"].dtype == np.float64

    actual = PredictionWindow.from_npz(path)
    assert actual.dense_storage_dtype == "float64"
    assert actual.point_map.dtype == np.float64
    np.testing.assert_array_equal(actual.point_map, expected.point_map)
    np.testing.assert_array_equal(actual.scene_flow, expected.scene_flow)


def test_compact_serialization_requires_explicit_dtype(tmp_path: Path) -> None:
    expected = _window()
    path = tmp_path / "compact.npz"
    expected.to_npz(path, storage_dtype="float32")

    restored = PredictionWindow.from_npz(path)
    assert restored.dense_storage_dtype == "float32"
    assert restored.point_map.dtype == np.float32
    np.testing.assert_array_equal(
        restored.point_map,
        expected.point_map.astype(np.float32),
    )

    promoted = PredictionWindow.from_npz(path, dense_storage_dtype="float64")
    assert promoted.point_map.dtype == np.float64
    np.testing.assert_array_equal(
        promoted.point_map,
        restored.point_map.astype(np.float64),
    )


def test_legacy_archive_keeps_historical_float64_load_default(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        window_id=np.asarray("legacy"),
        frame_indices=np.asarray([4]),
        point_map=np.ones((1, 1, 1, 3), dtype=np.float32),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )

    restored = PredictionWindow.from_npz(path)
    assert restored.point_map.dtype == np.float64
    assert restored.dense_storage_dtype == "float64"


def test_versioned_archive_rejects_dtype_metadata_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        schema_name=np.asarray(PREDICTION_WINDOW_NPZ_SCHEMA),
        schema_version=np.asarray(PREDICTION_WINDOW_NPZ_VERSION, dtype=np.int64),
        dense_storage_dtype=np.asarray("float64"),
        window_id=np.asarray("bad"),
        frame_indices=np.asarray([0]),
        point_map=np.ones((1, 1, 1, 3), dtype=np.float32),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="dtype disagrees"):
        PredictionWindow.from_npz(path)


def test_versioned_archive_rejects_partial_metadata(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez(
        path,
        schema_name=np.asarray(PREDICTION_WINDOW_NPZ_SCHEMA),
        point_map=np.ones((1, 1, 1, 3), dtype=np.float64),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="incomplete storage metadata"):
        PredictionWindow.from_npz(path, start_frame=0)
