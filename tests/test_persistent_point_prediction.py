from __future__ import annotations

import numpy as np
import pytest

from prob4d.persistent_point_prediction import (
    PERSISTENT_POINT_WINDOW_NPZ_SCHEMA,
    PERSISTENT_POINT_WINDOW_NPZ_VERSION,
    PersistentPointPredictionWindow,
)


def _window() -> PersistentPointPredictionWindow:
    trajectory = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return PersistentPointPredictionWindow(
        window_id="pointworld-window-0000",
        frame_indices=np.asarray([4, 5, 6], dtype=np.int64),
        point_ids=np.asarray([10, 20], dtype=np.int64),
        point_trajectory=trajectory,
        valid_mask=np.ones((3, 2), dtype=bool),
        context_frame_count=1,
        uncertainty=np.zeros((3, 2, 1), dtype=np.float32),
        uncertainty_semantics=("pointworld-normalized-relative-log-variance-v1"),
        storage_dtype="float32",
    )


def test_roundtrip_preserves_persistent_identity_and_precision(tmp_path) -> None:
    window = _window()
    output = tmp_path / "persistent.npz"
    window.to_npz(output)
    loaded = PersistentPointPredictionWindow.from_npz(output)

    assert loaded.summary() == window.summary()
    assert np.array_equal(loaded.frame_indices, [4, 5, 6])
    assert np.array_equal(loaded.point_ids, [10, 20])
    assert loaded.point_trajectory.dtype == np.dtype(np.float32)
    assert loaded.prediction_frame_indices.tolist() == [5, 6]
    assert loaded.local_index(5) == 1
    assert loaded.common_frames(window).tolist() == [4, 5, 6]

    for array in (
        loaded.frame_indices,
        loaded.point_ids,
        loaded.point_trajectory,
        loaded.valid_mask,
        loaded.uncertainty,
    ):
        assert array is not None
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_contract_rejects_ambiguous_identity_and_uncertainty() -> None:
    base = _window()
    with pytest.raises(ValueError, match="strictly increasing"):
        PersistentPointPredictionWindow(
            window_id=base.window_id,
            frame_indices=base.frame_indices,
            point_ids=np.asarray([10, 10], dtype=np.int64),
            point_trajectory=base.point_trajectory,
            valid_mask=base.valid_mask,
            context_frame_count=1,
            uncertainty=base.uncertainty,
            uncertainty_semantics=base.uncertainty_semantics,
            storage_dtype="float32",
        )
    invalid_first = np.ones((3, 2), dtype=bool)
    invalid_first[0, 1] = False
    with pytest.raises(ValueError, match="first context frame"):
        PersistentPointPredictionWindow(
            window_id=base.window_id,
            frame_indices=base.frame_indices,
            point_ids=base.point_ids,
            point_trajectory=base.point_trajectory,
            valid_mask=invalid_first,
            context_frame_count=1,
            storage_dtype="float32",
        )
    with pytest.raises(ValueError, match="describe present uncertainty"):
        PersistentPointPredictionWindow(
            window_id=base.window_id,
            frame_indices=base.frame_indices,
            point_ids=base.point_ids,
            point_trajectory=base.point_trajectory,
            valid_mask=base.valid_mask,
            context_frame_count=1,
            uncertainty=base.uncertainty,
            uncertainty_semantics="absent",
            storage_dtype="float32",
        )


def test_archive_rejects_schema_and_field_drift(tmp_path) -> None:
    base = _window()
    output = tmp_path / "bad.npz"
    np.savez_compressed(
        output,
        schema_name=np.asarray(PERSISTENT_POINT_WINDOW_NPZ_SCHEMA),
        schema_version=np.asarray(
            PERSISTENT_POINT_WINDOW_NPZ_VERSION,
            dtype=np.int64,
        ),
        storage_dtype=np.asarray("float32"),
        window_id=np.asarray(base.window_id),
        frame_indices=base.frame_indices,
        point_ids=base.point_ids,
        point_trajectory=base.point_trajectory,
        valid_mask=base.valid_mask,
        context_frame_count=np.asarray(1, dtype=np.int64),
        point_identity_semantics=np.asarray(base.point_identity_semantics),
        trajectory_semantics=np.asarray(base.trajectory_semantics),
        uncertainty_semantics=np.asarray(base.uncertainty_semantics),
        uncertainty=base.uncertainty,
        unexpected=np.asarray(1),
    )
    with pytest.raises(ValueError, match="fields changed"):
        PersistentPointPredictionWindow.from_npz(output)
