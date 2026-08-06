import subprocess
from pathlib import Path

import numpy as np
import pytest

from prob4d.vggt_baseline import (
    Sample,
    canonicalize_to_first_camera,
    main,
    prediction_path,
    read_samples,
    require_clean_git_checkout,
    select_partition,
    write_prediction_archive,
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


def _initialize_git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "prob4d@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Prob4D test"],
        cwd=path,
        check=True,
    )
    (path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_integrity_bound_vggt_requires_clean_source_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "vggt"
    _initialize_git_repository(repository)

    require_clean_git_checkout(repository)

    (repository / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean source checkout"):
        require_clean_git_checkout(repository)

    subprocess.run(
        ["git", "checkout", "--", "source.py"],
        cwd=repository,
        check=True,
    )
    (repository / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean source checkout"):
        require_clean_git_checkout(repository)


def test_resume_verifies_cached_predictions_without_loading_vggt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "predictions"
    vggt_root = tmp_path / "vggt"
    checkpoint = tmp_path / "vggt.pt"
    sample = Sample(Path("scene/video.mp4"), Path("scene/data.hdf5"))

    dataset_root.mkdir()
    (dataset_root / "filename_list.txt").write_text(
        f"{sample.video_path.as_posix()} {sample.data_path.as_posix()}\n",
        encoding="utf-8",
    )
    video = dataset_root / sample.video_path
    video.parent.mkdir(parents=True)
    video.write_bytes(b"cached-video")
    checkpoint.write_bytes(b"exact-checkpoint")
    _initialize_git_repository(vggt_root)

    points = np.ones((2, 1, 1, 3), dtype=np.float32)
    extrinsics = np.zeros((2, 3, 4), dtype=np.float32)
    extrinsics[:, :3, :3] = np.eye(3, dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    for representation in ("world_points", "depth_unprojected"):
        write_prediction_archive(
            prediction_path(output_root, sample, representation),
            point_map=points,
            camera_extrinsics=extrinsics,
            camera_intrinsics=intrinsics,
        )

    def fail_to_load_model(*args: object, **kwargs: object) -> object:
        raise AssertionError("resume verification must not load VGGT")

    monkeypatch.setattr("prob4d.vggt_baseline.load_vggt", fail_to_load_model)

    assert (
        main(
            [
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
                "--vggt-root",
                str(vggt_root),
                "--checkpoint",
                str(checkpoint),
                "--resume",
            ]
        )
        == 0
    )
    assert (output_root / "run-part-00.json").is_file()
