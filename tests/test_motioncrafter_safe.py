from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prob4d.io import load_prediction_bundle
from prob4d.motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    MotionCrafterRunConfig,
)
from prob4d.motioncrafter_integrity import verify_motioncrafter_prediction_manifest
from prob4d.motioncrafter_runner import SafeMotionCrafterRunner


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeAdapter:
    calls: list[tuple[int, int, int, int]] = []
    fail_after: int | None = None

    def __init__(self, config: MotionCrafterRunConfig) -> None:
        self.config = config

    def read_video(self, _: Path | None = None) -> np.ndarray:
        return np.zeros((30, 3, 64, 64), dtype=np.float32)

    def infer(
        self,
        frames: np.ndarray,
        *,
        window_size: int,
        overlap: int,
        seed: int,
    ) -> tuple[_FakeTensor, _FakeTensor]:
        type(self).calls.append((len(frames), window_size, overlap, seed))
        if type(self).fail_after is not None and len(type(self).calls) > type(self).fail_after:
            raise RuntimeError("injected interruption")
        count = int(frames.shape[0])
        points = np.zeros((count, 2, 2, 3), dtype=np.float32)
        points[..., 2] = 1.0
        valid = np.ones((count, 2, 2), dtype=bool)
        return _FakeTensor(points), _FakeTensor(valid)

    @staticmethod
    def _arrays(results: tuple[_FakeTensor, _FakeTensor]) -> dict[str, np.ndarray]:
        points, valid = results
        return {
            "point_map": points.numpy().astype(np.float32),
            "valid_mask": valid.numpy().astype(bool),
        }


def _provenance(*, clean: bool = True) -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "clean": clean,
        "status_sha256": "b" * 64,
        "status_entry_count": 0 if clean else 1,
    }


def _config(tmp_path: Path, *, output_name: str = "output") -> MotionCrafterRunConfig:
    video = tmp_path / f"{output_name}.mp4"
    video.write_bytes(b"synthetic video identity")
    return MotionCrafterRunConfig(
        upstream_root=tmp_path / "upstream",
        video_path=video,
        output_directory=tmp_path / output_name,
        window_size=25,
        overlap=8,
        seed=71,
        seed_policy=MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    )


def _runner(
    config: MotionCrafterRunConfig,
    *,
    clean: bool = True,
    adapter_factory: Any = _FakeAdapter,
) -> SafeMotionCrafterRunner:
    return SafeMotionCrafterRunner(
        config,
        adapter_factory=adapter_factory,
        provenance_provider=lambda _: _provenance(clean=clean),
    )


def test_safe_motioncrafter_run_binds_and_loads_every_artifact(tmp_path: Path) -> None:
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    manifest_path = _runner(_config(tmp_path)).run()

    verification = verify_motioncrafter_prediction_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = load_prediction_bundle(manifest_path)

    assert verification["integrity_bound"] is True
    assert verification["hashes_verified"] is True
    assert verification["member_count"] == 4
    assert len(manifest["artifact_integrity"]["members"]) == 4
    assert len(bundle.overlap_windows) == 2
    assert len(_FakeAdapter.calls) == 4


def test_safe_motioncrafter_verification_rejects_corruption(tmp_path: Path) -> None:
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    manifest_path = _runner(_config(tmp_path)).run()
    member = manifest_path.parent / "baseline_disjoint.npz"
    member.write_bytes(member.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="byte count mismatch"):
        verify_motioncrafter_prediction_manifest(manifest_path)
    with pytest.raises(ValueError, match="byte count mismatch"):
        load_prediction_bundle(manifest_path)


def test_safe_motioncrafter_resume_skips_hash_validated_products(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = 2
    with pytest.raises(RuntimeError, match="injected interruption"):
        _runner(config).run()
    assert len(_FakeAdapter.calls) == 3

    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    manifest_path = _runner(config).run(resume=True)

    assert manifest_path.is_file()
    assert len(_FakeAdapter.calls) == 2
    assert verify_motioncrafter_prediction_manifest(manifest_path)["member_count"] == 4


def test_safe_motioncrafter_refuses_nonempty_output_and_dirty_upstream(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_directory.mkdir()
    (config.output_directory / "unrelated.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        _runner(config).run()

    dirty_config = _config(tmp_path, output_name="dirty")
    with pytest.raises(ValueError, match="dirty"):
        _runner(dirty_config, clean=False).run()


def test_safe_motioncrafter_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    manifest_path = _runner(_config(tmp_path)).run()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["overlap_windows"][0]["path"] = "../outside.npz"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        verify_motioncrafter_prediction_manifest(manifest_path)


def test_completed_resume_verifies_without_loading_motioncrafter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    manifest_path = _runner(config).run()

    def forbidden_adapter(_: MotionCrafterRunConfig) -> Any:
        raise AssertionError("MotionCrafter must not load for a completed resume")

    resumed = _runner(config, adapter_factory=forbidden_adapter).run(resume=True)
    assert resumed == manifest_path


def test_safe_motioncrafter_resume_rejects_unbound_output_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = 2
    with pytest.raises(RuntimeError, match="injected interruption"):
        _runner(config).run()
    (config.output_directory / "stale.txt").write_text("unbound", encoding="utf-8")

    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    with pytest.raises(ValueError, match="unbound files"):
        _runner(config).run(resume=True)


def test_safe_motioncrafter_rejects_manifest_config_drift(tmp_path: Path) -> None:
    _FakeAdapter.calls = []
    _FakeAdapter.fail_after = None
    manifest_path = _runner(_config(tmp_path)).run()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["config"]["overlap"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="seed schedule|bound run spec"):
        verify_motioncrafter_prediction_manifest(manifest_path)
