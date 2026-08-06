from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one replacement anchor in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


baseline = Path("src/prob4d/vggt_baseline.py")
replace_once(
    baseline,
    '''def git_commit(repository: Path) -> str:
    """Return the exact baseline implementation revision."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


''',
    '''def git_commit(repository: Path) -> str:
    """Return the exact VGGT source revision."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_git_checkout(repository: Path) -> None:
    """Reject integrity-bound execution from modified or untracked VGGT source."""

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("integrity-bound VGGT export requires a clean source checkout")


''',
)
replace_once(
    baseline,
    '''    model, image_loader, pose_converter, unprojector = load_vggt(
        args.vggt_root,
        args.checkpoint,
        args.checkpoint_revision,
        args.device,
    )
    vggt_revision = git_commit(args.vggt_root)
    loader_sha256 = file_sha256(Path(__file__).resolve())
''',
    '''    vggt_revision = git_commit(args.vggt_root)
    if integrity_bound:
        require_clean_git_checkout(args.vggt_root)
    loader_sha256 = file_sha256(Path(__file__).resolve())
    model_bundle: tuple[Any, Any, Any, Any] | None = None
''',
)
replace_once(
    baseline,
    '''        else:
            print(f"[{sample_index}/{len(samples)}] infer {sample.video_path}", flush=True)
            with tempfile.TemporaryDirectory(prefix="prob4d-vggt-") as temporary:
                frame_paths = extract_video_frames(
                    args.dataset_root / sample.video_path, Path(temporary)
                )
                arrays = infer_sample(
                    model=model,
                    load_and_preprocess_images=image_loader,
                    pose_encoding_to_extri_intri=pose_converter,
                    unproject_depth_map_to_point_map=unprojector,
                    frame_paths=frame_paths,
                    device=args.device,
                    preprocess_mode=args.preprocess_mode,
                )
''',
    '''        else:
            print(f"[{sample_index}/{len(samples)}] infer {sample.video_path}", flush=True)
            if model_bundle is None:
                model_bundle = load_vggt(
                    args.vggt_root,
                    args.checkpoint,
                    args.checkpoint_revision,
                    args.device,
                )
            model, image_loader, pose_converter, unprojector = model_bundle
            with tempfile.TemporaryDirectory(prefix="prob4d-vggt-") as temporary:
                frame_paths = extract_video_frames(
                    args.dataset_root / sample.video_path, Path(temporary)
                )
                arrays = infer_sample(
                    model=model,
                    load_and_preprocess_images=image_loader,
                    pose_encoding_to_extri_intri=pose_converter,
                    unproject_depth_map_to_point_map=unprojector,
                    frame_paths=frame_paths,
                    device=args.device,
                    preprocess_mode=args.preprocess_mode,
                )
''',
)

tests = Path("tests/test_vggt_baseline.py")
replace_once(
    tests,
    "from pathlib import Path\n",
    "from pathlib import Path\nimport subprocess\n",
)
replace_once(
    tests,
    '''from prob4d.vggt_baseline import (
    Sample,
    canonicalize_to_first_camera,
    prediction_path,
    read_samples,
    select_partition,
)
''',
    '''from prob4d.vggt_baseline import (
    Sample,
    canonicalize_to_first_camera,
    main,
    prediction_path,
    read_samples,
    require_clean_git_checkout,
    select_partition,
    write_prediction_archive,
)
''',
)
with tests.open("a", encoding="utf-8") as stream:
    stream.write(
        '''


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
    (path / "source.py").write_text("VALUE = 1\\n", encoding="utf-8")
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

    (repository / "source.py").write_text("VALUE = 2\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean source checkout"):
        require_clean_git_checkout(repository)

    subprocess.run(
        ["git", "checkout", "--", "source.py"],
        cwd=repository,
        check=True,
    )
    (repository / "untracked.py").write_text("VALUE = 3\\n", encoding="utf-8")
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
        f"{sample.video_path.as_posix()} {sample.data_path.as_posix()}\\n",
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
'''
    )

docs = Path("docs/provider-neutral-predictions.md")
replace_once(
    docs,
    '''run record additionally binds the exact VGGT checkout, executed Prob4D loader
module bytes, preprocessing mode, input-video bytes, and every cached prediction
archive.
''',
    '''run record additionally binds the exact VGGT checkout, executed Prob4D loader
module bytes, preprocessing mode, input-video bytes, and every cached prediction
archive. Claim-bearing export rejects modified or untracked files in the VGGT
checkout. With `--resume`, complete cached samples are validated and sealed
without importing Torch or loading the VGGT model.
''',
)

Path(".github/patches/vggt-clean-source/apply.py").unlink()
