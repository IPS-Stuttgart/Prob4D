from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from prob4d.sintel_uncertainty import _resize_masked_bilinear, load_sintel_truth


def _corner_problem() -> tuple[np.ndarray, np.ndarray]:
    values = np.full((1, 2, 2, 3), 10.0, dtype=np.float32)
    values[0, 1, 1] = np.array([0.0, 0.0, 0.0])
    mask = np.array([[[True, True], [True, False]]])
    return values, mask


def test_mask_normalized_resize_prevents_zero_fill_leakage() -> None:
    values, mask = _corner_problem()

    resized, resized_mask = _resize_masked_bilinear(
        values,
        mask,
        (3, 3),
        minimum_support=0.5,
    )

    assert resized_mask[0, 1, 1]
    np.testing.assert_allclose(resized[0, 1, 1], [10.0, 10.0, 10.0])
    assert not resized_mask[0, 2, 2]
    np.testing.assert_array_equal(resized[0, 2, 2], np.zeros(3))


def test_mask_normalized_resize_applies_declared_support_threshold() -> None:
    values, mask = _corner_problem()

    resized, resized_mask = _resize_masked_bilinear(
        values,
        mask,
        (3, 3),
        minimum_support=0.8,
    )

    assert not resized_mask[0, 1, 1]
    np.testing.assert_array_equal(resized[0, 1, 1], np.zeros(3))


def test_mask_normalized_resize_same_shape_preserves_valid_values() -> None:
    values, mask = _corner_problem()

    resized, resized_mask = _resize_masked_bilinear(values, mask, (2, 2))

    np.testing.assert_array_equal(resized_mask, mask)
    np.testing.assert_array_equal(resized[mask], values[mask])
    np.testing.assert_array_equal(resized[~mask], np.zeros((1, 3)))


@pytest.mark.parametrize("value", [True, 0.0, 1.1, np.nan, "0.5"])
def test_mask_normalized_resize_rejects_invalid_support(value: object) -> None:
    values, mask = _corner_problem()
    with pytest.raises((TypeError, ValueError), match="minimum_support"):
        _resize_masked_bilinear(
            values,
            mask,
            (3, 3),
            minimum_support=value,  # type: ignore[arg-type]
        )


def test_load_sintel_truth_uses_mask_normalized_resize(tmp_path: Path) -> None:
    path = tmp_path / "truth.hdf5"
    points = np.zeros((1, 2, 2, 3), dtype=np.float32)
    points[..., 2] = 10.0
    points[0, 1, 1] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    mask = np.ones((1, 2, 2), dtype=np.uint8)
    poses = np.eye(4, dtype=np.float64)[None, ...]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("point_map", data=points)
        handle.create_dataset("valid_mask", data=mask)
        handle.create_dataset("camera_pose", data=poses)

    truth = load_sintel_truth(
        path,
        output_shape=(3, 3),
        minimum_resize_support=0.5,
    )

    assert truth.valid_mask[0, 1, 1]
    np.testing.assert_allclose(truth.point_map[0, 1, 1], [0.0, 0.0, 10.0])
    assert not truth.valid_mask[0, 2, 2]
    assert np.all(np.isfinite(truth.point_map))
