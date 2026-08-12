from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.sintel_uncertainty import (
    _relative_camera_poses,
    _resize_masked_bilinear,
    _validated_rigid_camera_poses,
    load_sintel_truth,
)


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
    h5py = pytest.importorskip("h5py")
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


def _rigid_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def test_rigid_camera_pose_validation_and_analytic_relative_transform() -> None:
    first_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    first = _rigid_pose(first_rotation, np.array([1.0, 2.0, 3.0]))
    delta = _rigid_pose(np.eye(3), np.array([0.5, -0.25, 1.0]))
    poses = _validated_rigid_camera_poses(
        np.stack([first, first @ delta]),
        expected_frames=2,
    )

    relative = _relative_camera_poses(poses)

    np.testing.assert_allclose(relative[0], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(relative[1], delta, atol=1e-12)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonfinite", "finite"),
        ("last-row", "homogeneous"),
        ("scaled", "orthogonal"),
        ("reflection", "determinant"),
    ],
)
def test_rigid_camera_pose_validation_rejects_invalid_geometry(
    mutation: str,
    message: str,
) -> None:
    poses = np.eye(4, dtype=np.float64)[None, ...]
    if mutation == "nonfinite":
        poses[0, 0, 0] = np.nan
    elif mutation == "last-row":
        poses[0, 3, 0] = 1.0
    elif mutation == "scaled":
        poses[0, 0, 0] = 2.0
    else:
        poses[0, 0, 0] = -1.0

    with pytest.raises(ValueError, match=message):
        _validated_rigid_camera_poses(poses, expected_frames=1)


def test_rigid_camera_pose_validation_rejects_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match="camera_pose must have shape"):
        _validated_rigid_camera_poses(
            np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0),
            expected_frames=1,
        )


@pytest.mark.parametrize("max_depth", [True, 0.0, -1.0, np.nan, "70"])
def test_load_sintel_truth_rejects_invalid_max_depth(
    tmp_path: Path,
    max_depth: object,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "truth.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("point_map", data=np.zeros((1, 1, 1, 3)))
        handle.create_dataset("valid_mask", data=np.ones((1, 1, 1)))
        handle.create_dataset("camera_pose", data=np.eye(4)[None, ...])

    with pytest.raises((TypeError, ValueError), match="max_depth"):
        load_sintel_truth(path, output_shape=(1, 1), max_depth=max_depth)


def test_load_sintel_truth_rejects_pose_frame_mismatch(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "truth.hdf5"
    points = np.zeros((1, 1, 1, 3), dtype=np.float32)
    points[..., 2] = 1.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("point_map", data=points)
        handle.create_dataset("valid_mask", data=np.ones((1, 1, 1)))
        handle.create_dataset(
            "camera_pose",
            data=np.repeat(np.eye(4)[None, ...], 2, axis=0),
        )

    with pytest.raises(ValueError, match="camera_pose must have shape"):
        load_sintel_truth(path, output_shape=(1, 1))
