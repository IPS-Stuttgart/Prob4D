from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.cut3r_provider_adapter import import_cut3r_online_prediction_manifest
from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    load_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)


def _write_frame(
    root: Path,
    index: int,
    *,
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    pose: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> None:
    stem = f"{index:06d}"
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "conf").mkdir(parents=True, exist_ok=True)
    (root / "camera").mkdir(parents=True, exist_ok=True)
    if depth is None:
        depth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    if confidence is None:
        confidence = np.asarray([[2.0, 1.0], [3.0, 4.0]], dtype=np.float32)
    if pose is None:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(index)
    if intrinsics is None:
        intrinsics = np.asarray(
            [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    np.save(root / "depth" / f"{stem}.npy", depth)
    np.save(root / "conf" / f"{stem}.npy", confidence)
    np.savez_compressed(
        root / "camera" / f"{stem}.npz",
        pose=pose,
        intrinsics=intrinsics,
    )


def _import(root: Path, output: Path, **kwargs: object):
    arguments: dict[str, object] = {
        "sequence_id": "sequence-a",
        "cut3r_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "input_video_sha256": "c" * 64,
        "input_video_byte_count": 1234,
        "frame_start": 10,
        "confidence_threshold": 1.5,
    }
    arguments.update(kwargs)
    return import_cut3r_online_prediction_manifest(root, output, **arguments)


def test_import_builds_world_points_and_prefix_lineage(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_frame(source, 0)
    _write_frame(source, 1)
    output = tmp_path / "bundle" / "provider.json"

    manifest = _import(source, output)

    assert manifest.provider_family == "CUT3R-online"
    assert manifest.coordinate_semantics == "sequence-local-sim3"
    assert manifest.metadata["execution_mode"] == "recurrent-online"
    assert manifest.metadata["metric_scale_claimed"] is False
    assert len(manifest.payloads) == 1
    payload = manifest.payloads[0]
    assert payload.output_frame_ids == (10, 11)
    assert payload.frame_lineage[0].source_frame_start == 10
    assert payload.frame_lineage[0].source_frame_stop_exclusive == 11
    assert payload.frame_lineage[1].source_frame_start == 10
    assert payload.frame_lineage[1].source_frame_stop_exclusive == 12
    assert not payload.is_causally_admitted(11)
    assert payload.is_causally_admitted(12)

    loaded = load_prediction_provider_manifest(output)
    assert loaded.artifact_id == manifest.artifact_id
    _, report = verify_prediction_provider_manifest(output, causal_frame_stop=12)
    assert report["verified_payload_count"] == 1
    assert report["admitted_payload_count"] == 1

    window = PredictionWindow.from_npz(
        output.parent / payload.path,
        dense_storage_dtype="float32",
    )
    assert window.frame_indices.tolist() == [10, 11]
    assert window.valid_mask[0].tolist() == [[True, False], [True, True]]
    expected_first = np.asarray([-0.25, -0.25, 1.0])
    np.testing.assert_allclose(window.point_map[0, 0, 0], expected_first)
    np.testing.assert_allclose(
        window.point_map[1, 0, 0],
        expected_first + np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.all(window.point_map[~window.valid_mask] == 0.0)


def test_identical_import_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_frame(source, 0)
    output = tmp_path / "bundle" / "provider.json"

    first = _import(source, output)
    first_bytes = output.read_bytes()
    second = _import(source, output)

    assert second.artifact_id == first.artifact_id
    assert output.read_bytes() == first_bytes


def test_import_rejects_noncontiguous_frame_stems(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_frame(source, 0)
    _write_frame(source, 2)

    with pytest.raises(ValueError, match="contiguous from zero"):
        _import(source, tmp_path / "bundle/provider.json")


def test_import_rejects_mismatched_member_sets(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_frame(source, 0)
    (source / "conf" / "000000.npy").unlink()
    (source / "conf").mkdir(exist_ok=True)
    np.save(source / "conf" / "000001.npy", np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="frame sets disagree|contiguous from zero"):
        _import(source, tmp_path / "bundle/provider.json")


def test_import_rejects_nonrigid_camera(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = 2.0
    _write_frame(source, 0, pose=pose)

    with pytest.raises(ValueError, match="orthonormal"):
        _import(source, tmp_path / "bundle/provider.json")


def test_import_rejects_empty_selected_support(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_frame(
        source,
        0,
        confidence=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="no point above"):
        _import(source, tmp_path / "bundle/provider.json")


def test_import_rejects_boolean_threshold(tmp_path: Path) -> None:
    source = tmp_path / "cut3r"
    _write_frame(source, 0)

    with pytest.raises(TypeError, match="confidence_threshold"):
        _import(
            source,
            tmp_path / "bundle/provider.json",
            confidence_threshold=True,
        )
