from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    load_prediction_provider_manifest,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window(path: Path, *, window_id: str = "w0", frames=(0, 1, 2)) -> PredictionWindow:
    point_map = np.zeros((len(frames), 2, 3, 3), dtype=np.float32)
    point_map[..., 2] = 1.0
    valid = np.ones(point_map.shape[:-1], dtype=bool)
    flow = np.zeros_like(point_map)
    window = PredictionWindow(
        window_id=window_id,
        frame_indices=np.asarray(frames, dtype=np.int64),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=valid,
        dense_storage_dtype="float32",
    )
    window.to_npz(path)
    return window


def _descriptor(path: Path, *, relative: str = "window.npz") -> PredictionPayloadDescriptorV1:
    return PredictionPayloadDescriptorV1(
        product_role="independent-window",
        window_id="w0",
        path=relative,
        sha256=_sha(path),
        byte_count=path.stat().st_size,
        view_id="camera-0",
        stochastic_member_id="seed-member-0",
        dependence_group_ids=("model:shared", "seed:0"),
        dense_storage_dtype="float32",
        has_scene_flow=True,
        has_ray_directions=False,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=frame,
                source_frame_start=0,
                source_frame_stop_exclusive=3,
                contributor_ids=("w0",),
            )
            for frame in (0, 1, 2)
        ),
    )


def _manifest(descriptor: PredictionPayloadDescriptorV1) -> PredictionProviderManifestV1:
    return PredictionProviderManifestV1(
        sequence_id="case-a",
        provider_family="Example4D",
        provider_repository="example/provider",
        provider_revision="a" * 40,
        provider_run_id="b" * 64,
        model_set_id="c" * 64,
        loader_id="d" * 64,
        coordinate_semantics="window-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="forward-point-displacement",
        ray_semantics="absent",
        payloads=(descriptor,),
        metadata={
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    )


def test_roundtrip_and_payload_verification(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _window(payload)
    manifest = _manifest(_descriptor(payload))
    manifest_path = tmp_path / "provider.json"
    save_prediction_provider_manifest(manifest_path, manifest)

    loaded = load_prediction_provider_manifest(manifest_path)
    assert loaded.artifact_id == manifest.artifact_id
    _, report = verify_prediction_provider_manifest(
        manifest_path,
        causal_frame_stop=3,
    )
    assert report["verified_payload_count"] == 1
    assert report["admitted_payload_count"] == 1
    assert report["payloads_verified"] is True


def test_paths_are_retrieval_metadata_for_portable_identity(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _window(payload)
    first = _descriptor(payload, relative="a/window.npz")
    second = _descriptor(payload, relative="b/window.npz")
    assert first.payload_id == second.payload_id
    assert _manifest(first).artifact_id == _manifest(second).artifact_id


def test_causal_admission_uses_source_dependencies_not_output_id(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _window(payload)
    descriptor = _descriptor(payload)
    future_dependency = replace(
        descriptor,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=frame,
                source_frame_start=0,
                source_frame_stop_exclusive=8,
                contributor_ids=("w0",),
            )
            for frame in descriptor.output_frame_ids
        ),
        payload_id=None,
    )
    assert future_dependency.is_causally_admitted(7) is False
    assert future_dependency.is_causally_admitted(8) is True


def test_tampered_payload_fails_before_array_use(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _window(payload)
    manifest_path = tmp_path / "provider.json"
    save_prediction_provider_manifest(manifest_path, _manifest(_descriptor(payload)))
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="byte count mismatch"):
        verify_prediction_provider_manifest(manifest_path)


def test_strict_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_prediction_provider_manifest(path)


def test_payload_rejects_boolean_byte_count(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _window(payload)
    descriptor = _descriptor(payload).to_record()
    descriptor["byte_count"] = True
    with pytest.raises(ValueError, match="byte_count"):
        PredictionPayloadDescriptorV1.from_record(descriptor)


def test_saved_manifest_is_append_safe_and_idempotent(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _window(payload)
    path = tmp_path / "provider.json"
    manifest = _manifest(_descriptor(payload))
    save_prediction_provider_manifest(path, manifest)
    save_prediction_provider_manifest(path, manifest)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["sequence_id"] = "changed"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_prediction_provider_manifest(changed_path)
