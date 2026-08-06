from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import prob4d.motioncrafter_integrity as motioncrafter_integrity
import prob4d.prediction_provider_manifest as provider_manifest
from prob4d.data import PredictionWindow
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    import_motioncrafter_prediction_manifest,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_window(path: Path) -> None:
    points = np.zeros((3, 2, 3, 3), dtype=np.float32)
    points[..., 2] = 1.0
    valid = np.ones(points.shape[:-1], dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    PredictionWindow(
        window_id="window-0",
        frame_indices=np.asarray((0, 1, 2), dtype=np.int64),
        point_map=points,
        valid_mask=valid,
        dense_storage_dtype="float32",
    ).to_npz(path)


def _manifest(payload: Path) -> PredictionProviderManifestV1:
    descriptor = PredictionPayloadDescriptorV1(
        product_role="independent-window",
        window_id="window-0",
        path=payload.name,
        sha256=_sha256(payload),
        byte_count=payload.stat().st_size,
        view_id="camera-0",
        stochastic_member_id="seed-0",
        dependence_group_ids=("model:shared", "seed:0"),
        dense_storage_dtype="float32",
        has_scene_flow=False,
        has_ray_directions=False,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=frame,
                source_frame_start=0,
                source_frame_stop_exclusive=3,
                contributor_ids=("window-0",),
            )
            for frame in (0, 1, 2)
        ),
    )
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
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=(descriptor,),
        metadata={"uses_truth": False},
    )


def test_active_writer_lock_fails_closed(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _write_window(payload)
    destination = tmp_path / "provider.json"
    lock = destination.with_name(f".{destination.name}.lock")
    lock.write_text("active\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="writer is already active"):
        save_prediction_provider_manifest(destination, _manifest(payload))
    assert not destination.exists()
    assert lock.read_text(encoding="utf-8") == "active\n"


def test_symlink_manifest_destination_is_rejected(tmp_path: Path) -> None:
    payload = tmp_path / "window.npz"
    _write_window(payload)
    target = tmp_path / "target.json"
    target.write_text("do-not-replace\n", encoding="utf-8")
    destination = tmp_path / "provider.json"
    try:
        destination.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="symbolic link"):
        save_prediction_provider_manifest(destination, _manifest(payload))
    assert target.read_text(encoding="utf-8") == "do-not-replace\n"


def test_payload_mutation_during_verification_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "window.npz"
    _write_window(payload)
    manifest_path = tmp_path / "provider.json"
    save_prediction_provider_manifest(manifest_path, _manifest(payload))
    original = provider_manifest.PredictionWindow.from_npz

    def mutating_loader(path: str | Path, *args: object, **kwargs: object) -> PredictionWindow:
        window = original(path, *args, **kwargs)
        member = Path(path)
        member.write_bytes(member.read_bytes() + b"tamper")
        return window

    monkeypatch.setattr(
        provider_manifest.PredictionWindow,
        "from_npz",
        staticmethod(mutating_loader),
    )
    with pytest.raises(ValueError, match="changed during verification"):
        verify_prediction_provider_manifest(manifest_path)


def test_manifest_mutation_during_verification_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "window.npz"
    _write_window(payload)
    manifest_path = tmp_path / "provider.json"
    save_prediction_provider_manifest(manifest_path, _manifest(payload))
    original = provider_manifest.load_prediction_provider_manifest

    def mutating_loader(path: str | Path) -> PredictionProviderManifestV1:
        manifest = original(path)
        member = Path(path)
        member.write_bytes(member.read_bytes() + b" ")
        return manifest

    monkeypatch.setattr(
        provider_manifest,
        "load_prediction_provider_manifest",
        mutating_loader,
    )
    with pytest.raises(ValueError, match="manifest changed during verification"):
        verify_prediction_provider_manifest(manifest_path)


def test_motioncrafter_manifest_mutation_during_import_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "predictions.json"
    source.write_text("{}\n", encoding="utf-8")

    def mutating_verifier(path: str | Path, *, verify_hashes: bool) -> dict[str, object]:
        assert verify_hashes is True
        member = Path(path)
        member.write_text('{"changed": true}\n', encoding="utf-8")
        return {"integrity_bound": True}

    monkeypatch.setattr(
        motioncrafter_integrity,
        "verify_motioncrafter_prediction_manifest",
        mutating_verifier,
    )
    with pytest.raises(ValueError, match="changed during import"):
        import_motioncrafter_prediction_manifest(
            source,
            tmp_path / "provider.json",
            sequence_id="case-a",
        )
