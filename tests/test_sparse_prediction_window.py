from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.sparse_prediction_window import (
    SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA,
    SparsePredictionWindow,
)


def _window(*, metadata: dict[str, object] | None = None) -> SparsePredictionWindow:
    position = np.asarray(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.1, 0.0, 1.0], [1.0, 0.2, 1.0]],
            [[0.2, 0.0, 1.0], [1.0, 0.4, 1.0]],
        ],
        dtype=np.float32,
    )
    uncertainty = np.full((3, 2, 1), -4.0, dtype=np.float32)
    uncertainty_valid = np.ones((3, 2), dtype=bool)
    uncertainty_valid[0] = False
    return SparsePredictionWindow(
        window_id="window-0001",
        frame_indices=np.asarray([7, 8, 9], dtype=np.int64),
        point_ids=np.asarray([3, 11], dtype=np.int64),
        position=position,
        valid_mask=np.ones((3, 2), dtype=bool),
        coordinate_semantics="metric-baxter-base",
        identity_semantics="persistent-within-window-source-order-v1",
        uncertainty_semantics="provider-native-log-variance-v1",
        provider_uncertainty=uncertainty,
        uncertainty_valid_mask=uncertainty_valid,
        dense_storage_dtype="float32",
        metadata={"z": 1, "a": [True, "value"]} if metadata is None else metadata,
    )


def test_sparse_window_round_trip_is_no_clobber_and_immutable(tmp_path: Path) -> None:
    expected = _window()
    path = tmp_path / "sparse-window.npz"
    expected.to_npz(path)
    actual = SparsePredictionWindow.from_npz(path)

    assert actual.content_id == expected.content_id
    assert actual.summary() == expected.summary()
    np.testing.assert_array_equal(actual.frame_indices, expected.frame_indices)
    np.testing.assert_array_equal(actual.point_ids, expected.point_ids)
    np.testing.assert_allclose(actual.position, expected.position)
    np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)
    np.testing.assert_allclose(
        actual.provider_uncertainty,
        expected.provider_uncertainty,
    )
    assert actual.position.flags.writeable is False
    assert actual.valid_mask.flags.writeable is False
    assert actual.provider_uncertainty is not None
    assert actual.provider_uncertainty.flags.writeable is False

    with pytest.raises(FileExistsError):
        expected.to_npz(path)
    assert SparsePredictionWindow.from_npz(path).content_id == expected.content_id


def test_sparse_window_content_id_uses_canonical_metadata_order() -> None:
    left = _window(metadata={"first": 1, "second": {"x": 2, "y": 3}})
    right = _window(metadata={"second": {"y": 3, "x": 2}, "first": 1})
    assert left.content_id == right.content_id


def test_sparse_window_requires_unique_seed_valid_point_ids() -> None:
    valid = np.ones((3, 2), dtype=bool)
    valid[0, 1] = False
    with pytest.raises(ValueError, match="seed frame"):
        SparsePredictionWindow(
            window_id="window",
            frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
            point_ids=np.asarray([0, 1], dtype=np.int64),
            position=np.zeros((3, 2, 3), dtype=np.float32),
            valid_mask=valid,
            coordinate_semantics="metric",
            identity_semantics="persistent-within-window",
            dense_storage_dtype="float32",
        )

    with pytest.raises(ValueError, match="unique"):
        SparsePredictionWindow(
            window_id="window",
            frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
            point_ids=np.asarray([4, 4], dtype=np.int64),
            position=np.zeros((3, 2, 3), dtype=np.float32),
            valid_mask=np.ones((3, 2), dtype=bool),
            coordinate_semantics="metric",
            identity_semantics="persistent-within-window",
            dense_storage_dtype="float32",
        )


def test_sparse_window_separates_uncertainty_presence_and_semantics() -> None:
    with pytest.raises(ValueError, match="describe present"):
        SparsePredictionWindow(
            window_id="window",
            frame_indices=np.asarray([0, 1], dtype=np.int64),
            point_ids=np.asarray([0], dtype=np.int64),
            position=np.zeros((2, 1, 3), dtype=np.float32),
            valid_mask=np.ones((2, 1), dtype=bool),
            coordinate_semantics="metric",
            identity_semantics="persistent-within-window",
            uncertainty_semantics="absent",
            provider_uncertainty=np.zeros((2, 1, 1), dtype=np.float32),
            uncertainty_valid_mask=np.ones((2, 1), dtype=bool),
            dense_storage_dtype="float32",
        )

    with pytest.raises(ValueError, match="present together"):
        SparsePredictionWindow(
            window_id="window",
            frame_indices=np.asarray([0, 1], dtype=np.int64),
            point_ids=np.asarray([0], dtype=np.int64),
            position=np.zeros((2, 1, 3), dtype=np.float32),
            valid_mask=np.ones((2, 1), dtype=bool),
            coordinate_semantics="metric",
            identity_semantics="persistent-within-window",
            uncertainty_semantics="provider-native",
            provider_uncertainty=np.zeros((2, 1, 1), dtype=np.float32),
            uncertainty_valid_mask=None,
            dense_storage_dtype="float32",
        )


def test_sparse_window_rejects_silent_dtype_conversion() -> None:
    with pytest.raises(ValueError, match="dtype must match"):
        SparsePredictionWindow(
            window_id="window",
            frame_indices=np.asarray([0, 1], dtype=np.int64),
            point_ids=np.asarray([0], dtype=np.int64),
            position=np.zeros((2, 1, 3), dtype=np.float64),
            valid_mask=np.ones((2, 1), dtype=bool),
            coordinate_semantics="metric",
            identity_semantics="persistent-within-window",
            dense_storage_dtype="float32",
        )


def test_sparse_window_archive_rejects_unknown_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    _window().to_npz(source)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    assert str(payload["schema_name"].item()) == SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA
    payload["unexpected"] = np.asarray(1, dtype=np.int64)
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **payload)
    with pytest.raises(ValueError, match="fields changed"):
        SparsePredictionWindow.from_npz(tampered)


def test_sparse_window_archive_rejects_declared_dtype_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    _window().to_npz(source)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["position"] = payload["position"].astype(np.float64)
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **payload)
    with pytest.raises(ValueError, match="disagrees"):
        SparsePredictionWindow.from_npz(tampered)
