from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.provider_bundle as provider_bundle
from prob4d.data import PredictionWindow
from prob4d.provider_bundle import (
    COORDINATE_SEMANTICS,
    FRAME_INDEX_SEMANTICS,
    LEGACY_ARCHIVE_SCHEMA,
    PROVIDER_INGEST_SPEC_SCHEMA,
    PROVIDER_INGEST_SPEC_VERSION,
    SOURCE_LINEAGE_SEMANTICS,
    VERSIONED_ARCHIVE_SCHEMA,
    build_motioncrafter_provider_window_bundle,
    build_provider_window_bundle,
    load_provider_window_bundle,
    verify_provider_window_bundle,
    write_provider_window_bundle,
)


def _write_window(
    path: Path,
    *,
    window_id: str,
    frame_indices: tuple[int, ...],
    versioned: bool = True,
    with_flow: bool = True,
) -> None:
    frames = np.asarray(frame_indices, dtype=np.int64)
    point_map = np.zeros((len(frames), 2, 3, 3), dtype=np.float32)
    point_map[..., 2] = 1.0
    point_map[..., 0] = np.arange(len(frames), dtype=np.float32)[:, None, None]
    valid_mask = np.ones(point_map.shape[:-1], dtype=bool)
    scene_flow = np.full_like(point_map, 0.05) if with_flow else None
    deform_mask = np.ones_like(valid_mask) if with_flow else None
    window = PredictionWindow(
        window_id=window_id,
        frame_indices=frames,
        point_map=point_map,
        valid_mask=valid_mask,
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        dense_storage_dtype="float32",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if versioned:
        window.to_npz(path)
        return
    payload: dict[str, np.ndarray] = {
        "window_id": np.asarray(window_id),
        "frame_indices": frames,
        "point_map": point_map,
        "valid_mask": valid_mask,
    }
    if scene_flow is not None:
        payload["scene_flow"] = scene_flow
        payload["deform_mask"] = deform_mask
    np.savez_compressed(path, **payload)


def _spec(*, allow_legacy: bool = False) -> dict[str, object]:
    return {
        "schema_name": PROVIDER_INGEST_SPEC_SCHEMA,
        "schema_version": PROVIDER_INGEST_SPEC_VERSION,
        "provider_name": "Example4D",
        "provider_version": "1.2.3",
        "implementation_identity": "git:" + "a" * 40,
        "model_set_identity": "sha256:" + "b" * 64,
        "source_identity": "sha256:" + "c" * 64,
        "coordinate_semantics": COORDINATE_SEMANTICS,
        "frame_index_semantics": FRAME_INDEX_SEMANTICS,
        "source_lineage_semantics": SOURCE_LINEAGE_SEMANTICS,
        "allow_legacy_window_archives": allow_legacy,
        "windows": [
            {
                "window_id": "window_b",
                "path": "windows/window_b.npz",
                "source_frame_start": 1,
                "source_frame_stop_exclusive": 4,
            },
            {
                "window_id": "window_a",
                "path": "windows/window_a.npz",
                "source_frame_start": 0,
                "source_frame_stop_exclusive": 3,
            },
        ],
        "metadata": {"adapter": "example", "fold": 0},
    }


def _write_spec(root: Path, value: dict[str, object]) -> Path:
    path = root / "provider-spec.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_provider_bundle_round_trip_and_payload_verification(tmp_path: Path) -> None:
    _write_window(
        tmp_path / "windows/window_a.npz",
        window_id="window_a",
        frame_indices=(0, 1, 2),
    )
    _write_window(
        tmp_path / "windows/window_b.npz",
        window_id="window_b",
        frame_indices=(1, 2, 3),
    )
    bundle = build_provider_window_bundle(_write_spec(tmp_path, _spec()))

    assert [item.window_id for item in bundle.windows] == ["window_a", "window_b"]
    assert all(item.archive_schema == VERSIONED_ARCHIVE_SCHEMA for item in bundle.windows)
    assert bundle.capabilities == ("point-map", "scene-flow")

    manifest = tmp_path / "provider-bundle.json"
    write_provider_window_bundle(bundle, manifest)
    write_provider_window_bundle(bundle, manifest)
    loaded = load_provider_window_bundle(manifest)
    assert loaded == bundle
    assert loaded.bundle_id == bundle.bundle_id

    verification = verify_provider_window_bundle(loaded, payload_root=tmp_path)
    assert verification["bundle_id"] == bundle.bundle_id
    assert verification["verified_window_count"] == 2


def test_payload_tamper_is_detected(tmp_path: Path) -> None:
    _write_window(
        tmp_path / "windows/window_a.npz",
        window_id="window_a",
        frame_indices=(0, 1, 2),
    )
    _write_window(
        tmp_path / "windows/window_b.npz",
        window_id="window_b",
        frame_indices=(1, 2, 3),
    )
    bundle = build_provider_window_bundle(_write_spec(tmp_path, _spec()))
    with (tmp_path / "windows/window_a.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="byte count mismatch|SHA-256 mismatch"):
        verify_provider_window_bundle(bundle, payload_root=tmp_path)


def test_legacy_archives_require_explicit_admission(tmp_path: Path) -> None:
    _write_window(
        tmp_path / "windows/window_a.npz",
        window_id="window_a",
        frame_indices=(0, 1, 2),
        versioned=False,
    )
    _write_window(
        tmp_path / "windows/window_b.npz",
        window_id="window_b",
        frame_indices=(1, 2, 3),
        versioned=False,
    )
    with pytest.raises(ValueError, match="allow_legacy_window_archives"):
        build_provider_window_bundle(_write_spec(tmp_path, _spec()))

    bundle = build_provider_window_bundle(_write_spec(tmp_path, _spec(allow_legacy=True)))
    assert all(item.archive_schema == LEGACY_ARCHIVE_SCHEMA for item in bundle.windows)
    verify_provider_window_bundle(bundle, payload_root=tmp_path)


def test_window_identity_and_source_interval_fail_closed(tmp_path: Path) -> None:
    _write_window(
        tmp_path / "windows/window_a.npz",
        window_id="different",
        frame_indices=(0, 1, 2),
    )
    _write_window(
        tmp_path / "windows/window_b.npz",
        window_id="window_b",
        frame_indices=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="window_id differs"):
        build_provider_window_bundle(_write_spec(tmp_path, _spec()))

    _write_window(
        tmp_path / "windows/window_a.npz",
        window_id="window_a",
        frame_indices=(0, 1, 2),
    )
    value = _spec()
    value["windows"][0]["source_frame_stop_exclusive"] = 3
    with pytest.raises(ValueError, match="inside the complete source interval"):
        build_provider_window_bundle(_write_spec(tmp_path, value))


def test_strict_json_and_safe_path_validation(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_name":"a","schema_name":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid strict JSON"):
        build_provider_window_bundle(duplicate)

    value = _spec()
    value["windows"][0]["path"] = "../window_b.npz"
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        build_provider_window_bundle(_write_spec(tmp_path, value))

    value = _spec()
    value["windows"][0]["source_frame_start"] = True
    with pytest.raises(TypeError, match="genuine integer"):
        build_provider_window_bundle(_write_spec(tmp_path, value))


def test_bundle_identity_and_closed_schema_are_verified(tmp_path: Path) -> None:
    _write_window(
        tmp_path / "windows/window_a.npz",
        window_id="window_a",
        frame_indices=(0, 1, 2),
    )
    _write_window(
        tmp_path / "windows/window_b.npz",
        window_id="window_b",
        frame_indices=(1, 2, 3),
    )
    bundle = build_provider_window_bundle(_write_spec(tmp_path, _spec()))
    manifest = tmp_path / "provider-bundle.json"
    write_provider_window_bundle(bundle, manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["provider_version"] = "changed"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_provider_window_bundle(manifest)

    value = bundle.to_dict()
    value["unknown"] = 1
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        load_provider_window_bundle(manifest)


def test_motioncrafter_integrity_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_window(
        tmp_path / "windows/window_0000.npz",
        window_id="window_0000",
        frame_indices=(10, 11, 12),
        versioned=False,
    )
    commit = "d" * 40
    model_set = "e" * 64
    source = "f" * 64
    run_spec = "1" * 64
    manifest = {
        "format_version": 1,
        "motioncrafter_commit": commit,
        "config": {
            "model_type": "determ",
            "model_source_set_sha256": model_set,
        },
        "stochastic_seed_schedule": {"policy": "derived-per-call"},
        "overlap_windows": [
            {
                "window_id": "window_0000",
                "path": "windows/window_0000.npz",
                "start_frame": 10,
                "stop_frame": 13,
            }
        ],
        "artifact_integrity": {
            "run_spec_sha256": run_spec,
            "run_spec": {"input_video": {"sha256": source}},
        },
    }
    manifest_path = tmp_path / "predictions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        provider_bundle,
        "_verify_motioncrafter_manifest",
        lambda path: {"integrity_bound": True, "manifest_path": str(path)},
    )

    bundle = build_motioncrafter_provider_window_bundle(manifest_path)
    assert bundle.provider_name == "MotionCrafter"
    assert bundle.provider_version == f"determ@{commit[:12]}"
    assert bundle.implementation_identity == f"git:{commit}"
    assert bundle.model_set_identity == f"sha256:{model_set}"
    assert bundle.source_identity == f"sha256:{source}"
    assert bundle.metadata["motioncrafter_run_spec_sha256"] == run_spec
    verify_provider_window_bundle(bundle, payload_root=tmp_path)


def test_symlink_payloads_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.npz"
    _write_window(
        outside,
        window_id="window_a",
        frame_indices=(0, 1, 2),
    )
    windows = tmp_path / "windows"
    windows.mkdir()
    try:
        (windows / "window_a.npz").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    _write_window(
        windows / "window_b.npz",
        window_id="window_b",
        frame_indices=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="contains a symlink"):
        build_provider_window_bundle(_write_spec(tmp_path, _spec()))
