from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prob4d import (
    _atomic_file,
    _heldout_promotion_common,
    common_mode_stress,
    fresh_provider_readiness,
    material_identity_mixture,
    prediction_provider_manifest,
    query_covariance_preservation,
    source_provider_competence,
    vggt_provider_adapter,
)
from prob4d._atomic_file import atomic_write_bytes
from prob4d.data import PredictionWindow
from prob4d.vggt_baseline import write_prediction_archive


def test_no_clobber_publication_has_exactly_one_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "evidence.json"
    barrier = threading.Barrier(2)
    original_link = _atomic_file.os.link

    def synchronized_link(source: object, target: object) -> None:
        barrier.wait(timeout=5.0)
        original_link(source, target)

    monkeypatch.setattr(_atomic_file.os, "link", synchronized_link)

    def write(payload: bytes) -> Exception | None:
        try:
            atomic_write_bytes(destination, payload, overwrite=False)
        except Exception as error:  # pragma: no branch - asserted below
            return error
        return None

    payloads = (b'{"writer": 1}\n', b'{"writer": 2}\n')
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(write, payloads))

    assert sum(outcome is None for outcome in outcomes) == 1
    errors = [outcome for outcome in outcomes if outcome is not None]
    assert len(errors) == 1
    assert isinstance(errors[0], FileExistsError)
    assert destination.read_bytes() in payloads
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_explicit_overwrite_replaces_complete_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"
    atomic_write_bytes(destination, b"first", overwrite=False)
    atomic_write_bytes(destination, b"second", overwrite=True)
    assert destination.read_bytes() == b"second"


def test_atomic_writer_rejects_non_boolean_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overwrite must be a Boolean"):
        atomic_write_bytes(
            tmp_path / "artifact.bin",
            b"payload",
            overwrite=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "writer",
    [
        fresh_provider_readiness._atomic_write_json,
        source_provider_competence._atomic_write_json,
        query_covariance_preservation._atomic_write_json,
        _heldout_promotion_common._atomic_write_json,
    ],
)
def test_claim_bearing_json_writers_preserve_no_clobber(
    tmp_path: Path,
    writer: Any,
) -> None:
    destination = tmp_path / "report.json"
    writer(destination, {"value": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        writer(destination, {"value": 2}, overwrite=False)
    writer(destination, {"value": 2}, overwrite=True)
    assert destination.read_text(encoding="utf-8") == '{\n  "value": 2\n}\n'


def test_other_claim_bearing_writers_use_shared_no_clobber(tmp_path: Path) -> None:
    report = tmp_path / "common-mode.json"
    common_mode_stress._atomic_write_json(report, {"value": 1})
    with pytest.raises(FileExistsError):
        common_mode_stress._atomic_write_json(report, {"value": 2})

    manifest = tmp_path / "provider.json"
    prediction_provider_manifest._atomic_write_text(manifest, "first\n")
    with pytest.raises(FileExistsError):
        prediction_provider_manifest._atomic_write_text(manifest, "second\n")


class _MixtureRecord:
    def __init__(self, value: int) -> None:
        self.value = value

    def to_record(self) -> dict[str, int]:
        return {"value": self.value}


def test_material_identity_writer_is_race_safe_and_overwritable(tmp_path: Path) -> None:
    destination = tmp_path / "mixture.json"
    material_identity_mixture.write_material_identity_mixture(
        destination,
        _MixtureRecord(1),  # type: ignore[arg-type]
    )
    with pytest.raises(FileExistsError):
        material_identity_mixture.write_material_identity_mixture(
            destination,
            _MixtureRecord(2),  # type: ignore[arg-type]
        )
    material_identity_mixture.write_material_identity_mixture(
        destination,
        _MixtureRecord(2),  # type: ignore[arg-type]
        overwrite=True,
    )
    assert destination.read_text(encoding="utf-8") == '{"value":2}\n'


def _window(value: float) -> PredictionWindow:
    return PredictionWindow(
        window_id="window-0",
        frame_indices=np.array([0], dtype=np.int64),
        point_map=np.full((1, 1, 1, 3), value, dtype=np.float32),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
        dense_storage_dtype="float32",
    )


def test_vggt_window_publication_is_idempotent_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "window.npz"
    vggt_provider_adapter._write_window_atomically(destination, _window(1.0))
    vggt_provider_adapter._write_window_atomically(destination, _window(1.0))
    with pytest.raises(ValueError, match="different canonical VGGT payload"):
        vggt_provider_adapter._write_window_atomically(destination, _window(2.0))


def test_vggt_prediction_archive_publication_is_idempotent_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "prediction.npz"
    points = np.ones((1, 1, 1, 3), dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None, ...]
    intrinsics = np.eye(3, dtype=np.float32)[None, ...]

    write_prediction_archive(
        destination,
        point_map=points,
        camera_extrinsics=extrinsics,
        camera_intrinsics=intrinsics,
    )
    write_prediction_archive(
        destination,
        point_map=points.copy(),
        camera_extrinsics=extrinsics.copy(),
        camera_intrinsics=intrinsics.copy(),
    )
    with pytest.raises(ValueError, match="different VGGT prediction"):
        write_prediction_archive(
            destination,
            point_map=points * 2.0,
            camera_extrinsics=extrinsics,
            camera_intrinsics=intrinsics,
        )
