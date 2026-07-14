from pathlib import Path

import numpy as np
import pytest

from prob4d.vggt_baseline import (
    Sample,
    canonicalize_to_first_camera,
    prediction_path,
    read_samples,
    select_partition,
)


def test_read_samples_and_partition(tmp_path: Path) -> None:
    (tmp_path / "filename_list.txt").write_text(
        "scene_a/a.mp4 scene_a/a.hdf5\n\nscene_b/b.mp4 scene_b/b.hdf5\n",
        encoding="utf-8",
    )

    samples = read_samples(tmp_path)

    assert samples == [
        Sample(Path("scene_a/a.mp4"), Path("scene_a/a.hdf5")),
        Sample(Path("scene_b/b.mp4"), Path("scene_b/b.hdf5")),
    ]
    assert select_partition(samples, 0, 2) == samples[:1]
    assert select_partition(samples, 1, 2) == samples[1:]


def test_select_partition_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError, match="0 <= index < count"):
        select_partition([], 2, 2)


def test_canonicalize_to_first_camera() -> None:
    points = np.array([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]])
    extrinsics = np.zeros((2, 3, 4))
    extrinsics[:, :3, :3] = np.eye(3)
    extrinsics[0, :, 3] = [10.0, 20.0, 30.0]

    actual = canonicalize_to_first_camera(points, extrinsics)

    np.testing.assert_allclose(actual, points + [10.0, 20.0, 30.0])


def test_prediction_path_preserves_sample_directory(tmp_path: Path) -> None:
    sample = Sample(Path("scene/video.mp4"), Path("scene/data.hdf5"))

    assert prediction_path(tmp_path, sample, "world_points") == (
        tmp_path / "world_points/scene/video.npz"
    )
