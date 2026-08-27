from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.persistent_point_prediction import (
    POINTWORLD_POINT_IDENTITY_SEMANTICS,
    POINTWORLD_REPORTED_UNCERTAINTY_SEMANTICS,
    PersistentPointPredictionWindow,
    persistent_point_window_from_pointworld,
)


def _pointworld_window() -> PersistentPointPredictionWindow:
    positions = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3) / 10.0
    valid = np.ones((3, 4), dtype=bool)
    log_variance = np.full((3, 4, 1), -4.0, dtype=np.float32)
    return persistent_point_window_from_pointworld(
        window_id="pointworld-window-000",
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        scene_positions=positions,
        scene_valid_mask=valid,
        reported_log_variance=log_variance,
        normalization_id="a" * 64,
    )


def test_pointworld_adapter_preserves_persistent_axis_and_roundtrips(
    tmp_path: Path,
) -> None:
    window = _pointworld_window()

    assert window.shape == (3, 4)
    assert window.storage_dtype == "float32"
    assert window.point_identity_semantics == POINTWORLD_POINT_IDENTITY_SEMANTICS
    assert (
        window.reported_uncertainty_semantics
        == POINTWORLD_REPORTED_UNCERTAINTY_SEMANTICS
    )
    assert window.reported_uncertainty_reference_id == "a" * 64
    assert len(np.unique(window.point_ids)) == window.point_count
    assert not window.point_ids.flags.writeable
    assert not window.point_positions.flags.writeable

    archive = tmp_path / "window.npz"
    window.to_npz(archive)
    restored = PersistentPointPredictionWindow.from_npz(archive)

    assert restored.window_id == window.window_id
    assert restored.position_semantics == window.position_semantics
    assert np.array_equal(restored.frame_indices, window.frame_indices)
    assert np.array_equal(restored.source_point_indices, window.source_point_indices)
    assert np.array_equal(restored.point_ids, window.point_ids)
    assert np.array_equal(restored.point_positions, window.point_positions)
    assert np.array_equal(restored.reported_log_variance, window.reported_log_variance)


def test_window_scoped_ids_do_not_assert_cross_window_identity() -> None:
    first = _pointworld_window()
    second = persistent_point_window_from_pointworld(
        window_id="pointworld-window-001",
        frame_indices=first.frame_indices,
        scene_positions=first.point_positions,
        scene_valid_mask=first.valid_mask,
        reported_log_variance=first.reported_log_variance,
        normalization_id="a" * 64,
    )

    assert np.array_equal(first.source_point_indices, second.source_point_indices)
    assert not np.any(np.isin(first.point_ids, second.point_ids))


def test_invalid_entries_may_be_nonfinite_but_valid_entries_may_not() -> None:
    window = _pointworld_window()
    positions = np.array(window.point_positions, copy=True)
    valid = np.array(window.valid_mask, copy=True)
    valid[1, 2] = False
    positions[1, 2] = np.nan

    accepted = persistent_point_window_from_pointworld(
        window_id="invalid-row-is-masked",
        frame_indices=window.frame_indices,
        scene_positions=positions,
        scene_valid_mask=valid,
        reported_log_variance=window.reported_log_variance,
        normalization_id="b" * 64,
    )
    assert not accepted.valid_mask[1, 2]

    valid[1, 2] = True
    with pytest.raises(ValueError, match="valid point positions must be finite"):
        persistent_point_window_from_pointworld(
            window_id="invalid-row-is-active",
            frame_indices=window.frame_indices,
            scene_positions=positions,
            scene_valid_mask=valid,
            reported_log_variance=window.reported_log_variance,
            normalization_id="b" * 64,
        )


def test_pointworld_adapter_rejects_vector_log_variance() -> None:
    window = _pointworld_window()
    with pytest.raises(ValueError, match=r"must have shape \(T, N, 1\)"):
        persistent_point_window_from_pointworld(
            window_id="wrong-logvar-shape",
            frame_indices=window.frame_indices,
            scene_positions=window.point_positions,
            scene_valid_mask=window.valid_mask,
            reported_log_variance=np.zeros((3, 4, 3), dtype=np.float32),
            normalization_id="c" * 64,
        )


def test_uncertainty_fields_are_all_or_none() -> None:
    window = _pointworld_window()
    with pytest.raises(ValueError, match="must either all be present or all be absent"):
        PersistentPointPredictionWindow(
            window_id="incomplete-uncertainty",
            frame_indices=window.frame_indices,
            source_point_indices=window.source_point_indices,
            point_ids=window.point_ids,
            point_positions=window.point_positions,
            valid_mask=window.valid_mask,
            position_semantics=window.position_semantics,
            point_identity_semantics=window.point_identity_semantics,
            reported_log_variance=window.reported_log_variance,
            storage_dtype="float32",
        )


def test_source_indices_must_be_genuine_unique_integers() -> None:
    window = _pointworld_window()
    with pytest.raises(TypeError, match="genuine integers"):
        persistent_point_window_from_pointworld(
            window_id="float-source-indices",
            frame_indices=window.frame_indices,
            scene_positions=window.point_positions,
            scene_valid_mask=window.valid_mask,
            reported_log_variance=window.reported_log_variance,
            normalization_id="d" * 64,
            source_point_indices=np.asarray([0.0, 1.0, 2.0, 3.0]),
        )

    with pytest.raises(ValueError, match="source_point_indices must be unique"):
        persistent_point_window_from_pointworld(
            window_id="duplicate-source-indices",
            frame_indices=window.frame_indices,
            scene_positions=window.point_positions,
            scene_valid_mask=window.valid_mask,
            reported_log_variance=window.reported_log_variance,
            normalization_id="d" * 64,
            source_point_indices=np.asarray([0, 1, 1, 3], dtype=np.int64),
        )


def test_archive_rejects_unknown_fields(tmp_path: Path) -> None:
    window = _pointworld_window()
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        schema_name=np.asarray("prob4d.persistent-point-prediction-window-npz"),
        schema_version=np.asarray(1, dtype=np.int64),
        storage_dtype=np.asarray("float32"),
        window_id=np.asarray(window.window_id),
        frame_indices=window.frame_indices,
        source_point_indices=window.source_point_indices,
        point_ids=window.point_ids,
        point_positions=window.point_positions,
        valid_mask=window.valid_mask,
        position_semantics=np.asarray(window.position_semantics),
        point_identity_semantics=np.asarray(window.point_identity_semantics),
        reported_log_variance=window.reported_log_variance,
        reported_uncertainty_semantics=np.asarray(
            window.reported_uncertainty_semantics
        ),
        reported_uncertainty_reference_id=np.asarray(
            window.reported_uncertainty_reference_id
        ),
        unexpected=np.asarray(1),
    )

    with pytest.raises(ValueError, match=r"extra=\['unexpected'\]"):
        PersistentPointPredictionWindow.from_npz(path)
