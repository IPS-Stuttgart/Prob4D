from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    load_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)
from prob4d.vggt_baseline import write_prediction_archive
from prob4d.vggt_integrity import (
    build_run_record,
    build_sample_record,
    describe_prediction_archive,
    load_vggt_run_metadata,
    relative_member,
    save_vggt_run_metadata,
)
from prob4d.vggt_provider_adapter import import_vggt_prediction_manifest


def _write_vggt_archive(
    path: Path,
    *,
    offset: float,
    extrinsic_shift: float = 0.0,
) -> None:
    points = np.zeros((2, 2, 3, 3), dtype=np.float16)
    points[..., 0] = offset
    points[..., 2] = 1.0
    points[0, 0, 0] = np.nan
    extrinsics = np.zeros((2, 3, 4), dtype=np.float32)
    extrinsics[:, :3, :3] = np.eye(3, dtype=np.float32)
    extrinsics[0, 0, 3] = extrinsic_shift
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        point_map=points,
        camera_extrinsics=extrinsics,
        camera_intrinsics=intrinsics,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    dataset_root = tmp_path / "dataset"
    prediction_root = tmp_path / "predictions"
    sample_id = "scene/video.mp4"
    video = dataset_root / sample_id
    video.parent.mkdir(parents=True)
    video.write_bytes(b"exact-video-bytes")

    members = []
    for index, representation in enumerate(("world_points", "depth_unprojected")):
        path = prediction_root / representation / "scene/video.npz"
        _write_vggt_archive(path, offset=float(index))
        members.append(
            describe_prediction_archive(
                path,
                representation=representation,
                relative_path=relative_member(
                    path,
                    root=prediction_root,
                    name="prediction path",
                ),
            )
        )
    sample = build_sample_record(
        sample_id=sample_id,
        input_video_path=video,
        representations=members,
    )
    run = build_run_record(
        vggt_commit="a" * 40,
        loader_module_sha256="b" * 64,
        checkpoint="/models/vggt.pt",
        checkpoint_sha256="c" * 64,
        checkpoint_revision=None,
        preprocess_mode="crop",
        partition_index=0,
        partition_count=1,
        samples=[sample],
        dataset_root=dataset_root,
        output_root=prediction_root,
        elapsed_seconds=1.25,
    )
    metadata_path = prediction_root / "run-part-00.json"
    save_vggt_run_metadata(metadata_path, run)
    return metadata_path, dataset_root, prediction_root, sample_id


def test_vggt_run_metadata_is_path_and_timing_independent(tmp_path: Path) -> None:
    metadata_path, _, _, _ = _fixture(tmp_path)
    first = load_vggt_run_metadata(metadata_path)
    second = build_run_record(
        vggt_commit=str(first["vggt_commit"]),
        loader_module_sha256=str(first["loader_module_sha256"]),
        checkpoint=str(first["checkpoint"]),
        checkpoint_sha256=str(first["checkpoint_sha256"]),
        checkpoint_revision=None,
        preprocess_mode=str(first["preprocess_mode"]),
        partition_index=0,
        partition_count=1,
        samples=first["samples"],
        dataset_root=tmp_path / "moved-dataset",
        output_root=tmp_path / "moved-predictions",
        elapsed_seconds=99.0,
    )
    assert second["run_id"] == first["run_id"]


def test_legacy_unpinned_metadata_is_not_provider_neutral(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "method": "VGGT-1B",
                "checkpoint": "facebook/VGGT-1B",
                "checkpoint_sha256": None,
                "integrity_bound": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy or unpinned"):
        load_vggt_run_metadata(path)


def test_import_vggt_writes_shared_dependent_causal_payloads(tmp_path: Path) -> None:
    metadata_path, dataset_root, prediction_root, sample_id = _fixture(tmp_path)
    output = tmp_path / "bundle/provider.json"

    manifest = import_vggt_prediction_manifest(
        metadata_path,
        output,
        sequence_id="sequence-a",
        sample_id=sample_id,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        frame_start=10,
    )

    assert manifest.provider_family == "VGGT"
    assert manifest.coordinate_semantics == "sequence-local-sim3"
    assert len(manifest.payloads) == 2
    assert manifest.payloads[0].dependence_group_ids == (manifest.payloads[1].dependence_group_ids)
    assert manifest.payloads[0].stochastic_member_id == (manifest.payloads[1].stochastic_member_id)
    assert all(not payload.is_causally_admitted(11) for payload in manifest.payloads)
    assert all(payload.is_causally_admitted(12) for payload in manifest.payloads)

    loaded = load_prediction_provider_manifest(output)
    assert loaded.artifact_id == manifest.artifact_id
    _, report = verify_prediction_provider_manifest(output, causal_frame_stop=12)
    assert report["verified_payload_count"] == 2
    assert report["admitted_payload_count"] == 2

    first_payload = output.parent / loaded.payloads[0].path
    window = PredictionWindow.from_npz(first_payload, dense_storage_dtype="float32")
    assert window.frame_indices.tolist() == [10, 11]
    assert not window.valid_mask[0, 0, 0]
    assert np.all(window.point_map[~window.valid_mask] == 0.0)


def test_import_rejects_tampered_cached_prediction(tmp_path: Path) -> None:
    metadata_path, dataset_root, prediction_root, sample_id = _fixture(tmp_path)
    source = prediction_root / "world_points/scene/video.npz"
    source.write_bytes(source.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="byte count mismatch"):
        import_vggt_prediction_manifest(
            metadata_path,
            tmp_path / "bundle/provider.json",
            sequence_id="sequence-a",
            sample_id=sample_id,
            dataset_root=dataset_root,
            prediction_root=prediction_root,
        )


def test_import_rejects_mixed_camera_runs(tmp_path: Path) -> None:
    metadata_path, dataset_root, prediction_root, sample_id = _fixture(tmp_path)
    path = prediction_root / "depth_unprojected/scene/video.npz"
    _write_vggt_archive(path, offset=1.0, extrinsic_shift=2.0)

    run = json.loads(metadata_path.read_text(encoding="utf-8"))
    member = next(
        item
        for item in run["samples"][0]["representations"]
        if item["representation"] == "depth_unprojected"
    )
    replacement = describe_prediction_archive(
        path,
        representation="depth_unprojected",
        relative_path=member["path"],
    )
    sample = build_sample_record(
        sample_id=sample_id,
        input_video_path=dataset_root / sample_id,
        representations=[
            item if item["representation"] == "world_points" else replacement
            for item in run["samples"][0]["representations"]
        ],
    )
    revised = build_run_record(
        vggt_commit=run["vggt_commit"],
        loader_module_sha256=run["loader_module_sha256"],
        checkpoint=run["checkpoint"],
        checkpoint_sha256=run["checkpoint_sha256"],
        checkpoint_revision=None,
        preprocess_mode=run["preprocess_mode"],
        partition_index=0,
        partition_count=1,
        samples=[sample],
        dataset_root=dataset_root,
        output_root=prediction_root,
        elapsed_seconds=2.0,
    )
    revised_path = prediction_root / "run-part-01.json"
    save_vggt_run_metadata(revised_path, revised)

    with pytest.raises(ValueError, match="camera extrinsics"):
        import_vggt_prediction_manifest(
            revised_path,
            tmp_path / "bundle/provider.json",
            sequence_id="sequence-a",
            sample_id=sample_id,
            dataset_root=dataset_root,
            prediction_root=prediction_root,
        )


def test_atomic_vggt_archive_writer_is_idempotent_and_refuses_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prediction.npz"
    points = np.ones((1, 1, 1, 3), dtype=np.float32)
    extrinsics = np.zeros((1, 3, 4), dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)[None]

    write_prediction_archive(
        path,
        point_map=points,
        camera_extrinsics=extrinsics,
        camera_intrinsics=intrinsics,
    )
    original = path.read_bytes()
    write_prediction_archive(
        path,
        point_map=points,
        camera_extrinsics=extrinsics,
        camera_intrinsics=intrinsics,
    )
    assert path.read_bytes() == original

    with pytest.raises(ValueError, match="refusing to replace"):
        write_prediction_archive(
            path,
            point_map=points + 1.0,
            camera_extrinsics=extrinsics,
            camera_intrinsics=intrinsics,
        )
