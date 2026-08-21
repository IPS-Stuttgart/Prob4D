from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "science" / "build_cut3r_deform360_source_freeze.py"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cut3r_deform360_source_freeze", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_repository(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", path, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (path / "identity.txt").write_text("identity\n", encoding="utf-8")
    subprocess.run(["git", "-C", path, "add", "identity.txt"], check=True)
    subprocess.run(["git", "-C", path, "commit", "-qm", "identity"], check=True)
    return subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _selection(protocol: dict[str, Any]) -> dict[str, Any]:
    source = [
        {
            "object_id": item["object_id"],
            "episode_id": item["episode_id"],
            "stratum": item["stratum"],
        }
        for item in protocol["source_groups"]
    ]
    target = [dict(item) for item in protocol["forbidden_target_groups"]]
    return {
        "selection_artifact_sha256": protocol["source_dataset"]["selection_artifact_sha256"],
        "selection_sha256": protocol["source_dataset"]["selection_sha256"],
        "selection": {"calibration": source, "confirmation": target},
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _camera_transform(center: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(center)
    return transform


def _episode(
    processed_root: Path,
    group: dict[str, Any],
    *,
    missing_camera: str | None = None,
) -> None:
    episode = processed_root / group["object_id"] / f"episode_{group['episode_id']:04d}"
    episode.mkdir(parents=True)
    camera_centers = {
        "cam-a": (2.0, 0.0, 0.2),
        "cam-b": (0.0, 2.0, 0.1),
        "cam-c": (-2.0, 0.0, 0.0),
        "cam-d": (0.0, -2.0, -0.1),
        "cam-e": (1.4, 1.4, 0.3),
    }
    intrinsics = {
        camera: np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        for camera in camera_centers
    }
    extrinsics = {camera: _camera_transform(center) for camera, center in camera_centers.items()}
    np.save(episode / "undistorted_intrinsics.npy", intrinsics, allow_pickle=True)
    np.save(episode / "extrinsics.npy", extrinsics, allow_pickle=True)
    for camera in camera_centers:
        camera_dir = episode / camera
        camera_dir.mkdir()
        if camera == missing_camera:
            continue
        (camera_dir / "undistorted.mp4").write_bytes(
            (f"video:{group['group_id']}:{camera}\n" * 11).encode()
        )
        (camera_dir / "aligned_timestamps.txt").write_text(
            "".join(f"{index}\n" for index in range(64)),
            encoding="utf-8",
        )
        _write_json(camera_dir / "alignment.json", {"camera": camera, "frames": 64})
        _write_json(camera_dir / "metadata.json", {"camera": camera, "verified": True})


def _fixture(tmp_path: Path, *, common_camera_count: int = 5) -> dict[str, Path]:
    package_root = Path(__file__).parents[1]
    protocol = json.loads(
        (package_root / "protocols" / "cut3r_deform360_source_v1.json").read_text(encoding="utf-8")
    )
    repository = tmp_path / "prob4d"
    cut3r = tmp_path / "cut3r"
    protocol["provider"]["revision"] = _git_repository(cut3r)
    _git_repository(repository)
    checkpoint = tmp_path / protocol["provider"]["checkpoint_filename"]
    checkpoint.write_bytes(b"checkpoint-bytes")
    wheel = tmp_path / "prob4d-0.5.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")

    selection = _selection(protocol)
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    protocol["source_dataset"]["selection_file_sha256"] = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)

    processed_root = tmp_path / "processed"
    missing_camera = None if common_camera_count == 5 else "cam-e"
    for index, group in enumerate(protocol["source_groups"]):
        _episode(
            processed_root,
            group,
            missing_camera=missing_camera if index == 0 else None,
        )
    return {
        "repository": repository,
        "protocol": protocol_path,
        "selection": selection_path,
        "processed_root": processed_root,
        "cut3r": cut3r,
        "checkpoint": checkpoint,
        "wheel": wheel,
        "output": tmp_path / "output",
    }


def test_builds_group_aware_source_freeze_without_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    opened: list[Path] = []
    original = module._regular_snapshot

    def recording_snapshot(path: Path, *, name: str):
        opened.append(path)
        return original(path, name=name)

    monkeypatch.setattr(module, "_regular_snapshot", recording_snapshot)
    result = module.build_source_freeze(
        repository=fixture["repository"],
        protocol_path=fixture["protocol"],
        selection_path=fixture["selection"],
        processed_root=fixture["processed_root"],
        cut3r_checkout=fixture["cut3r"],
        checkpoint_path=fixture["checkpoint"],
        prob4d_wheel=fixture["wheel"],
        output_directory=fixture["output"],
    )

    assert result["decision"] == module.SUPPORT_PASS
    assert result["source_group_count"] == 10
    assert result["forbidden_target_group_count"] == 12
    assert len(result["camera_panel"]["selected_cameras"]) == 4
    assert len(result["source_cases"]) == 40
    forbidden = {item["object_id"] for item in result["forbidden_target_groups"]}
    assert not any(any(object_id in path.parts for object_id in forbidden) for path in opened)

    spec = json.loads(
        (fixture["output"] / "cut3r-comparison-spec.json").read_text(encoding="utf-8")
    )
    assert len(spec["groups"]) == 10
    assert {role: len(groups) for role, groups in spec["group_roles"].items()} == {
        "development": 2,
        "calibration": 4,
        "source_evaluation": 4,
    }
    assert all(len(group["cases"]) == 4 for group in spec["groups"])
    assert all(
        case["frame_stop_exclusive"] == 58 and case["evaluation_frame_start"] == 24
        for group in spec["groups"]
        for case in group["cases"]
    )


def test_retains_support_negative_without_comparison_spec(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path, common_camera_count=4)
    protocol = json.loads(fixture["protocol"].read_text(encoding="utf-8"))
    protocol["camera_panel"]["panel_size"] = 5
    protocol["camera_panel"]["minimum_common_supported_cameras"] = 5
    _write_json(fixture["protocol"], protocol)

    result = module.build_source_freeze(
        repository=fixture["repository"],
        protocol_path=fixture["protocol"],
        selection_path=fixture["selection"],
        processed_root=fixture["processed_root"],
        cut3r_checkout=fixture["cut3r"],
        checkpoint_path=fixture["checkpoint"],
        prob4d_wheel=fixture["wheel"],
        output_directory=fixture["output"],
    )

    assert result["decision"] == module.SUPPORT_NEGATIVE
    assert result["camera_panel"] is None
    assert not (fixture["output"] / "cut3r-comparison-spec.json").exists()
    assert (fixture["output"] / "cut3r-deform360-source-freeze.json").is_file()


def test_rejects_selection_roster_drift(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    selection["selection"]["calibration"][0]["episode_id"] += 1
    _write_json(fixture["selection"], selection)
    protocol = json.loads(fixture["protocol"].read_text(encoding="utf-8"))
    protocol["source_dataset"]["selection_file_sha256"] = hashlib.sha256(
        fixture["selection"].read_bytes()
    ).hexdigest()
    _write_json(fixture["protocol"], protocol)

    with pytest.raises(ValueError, match="source roster differs"):
        module.build_source_freeze(
            repository=fixture["repository"],
            protocol_path=fixture["protocol"],
            selection_path=fixture["selection"],
            processed_root=fixture["processed_root"],
            cut3r_checkout=fixture["cut3r"],
            checkpoint_path=fixture["checkpoint"],
            prob4d_wheel=fixture["wheel"],
            output_directory=fixture["output"],
        )


def test_publication_is_no_clobber(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    arguments = {
        "repository": fixture["repository"],
        "protocol_path": fixture["protocol"],
        "selection_path": fixture["selection"],
        "processed_root": fixture["processed_root"],
        "cut3r_checkout": fixture["cut3r"],
        "checkpoint_path": fixture["checkpoint"],
        "prob4d_wheel": fixture["wheel"],
        "output_directory": fixture["output"],
    }
    first = module.build_source_freeze(**arguments)
    second = module.build_source_freeze(**arguments)
    assert first == second

    freeze_path = fixture["output"] / "cut3r-deform360-source-freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different retained bytes"):
        module.build_source_freeze(**arguments)


def test_rejects_source_role_drift_from_frozen_hash_rule(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    protocol = json.loads(fixture["protocol"].read_text(encoding="utf-8"))
    first = protocol["source_groups"][0]
    second = protocol["source_groups"][1]
    first["role"], second["role"] = second["role"], first["role"]
    _write_json(fixture["protocol"], protocol)

    with pytest.raises(ValueError, match="source role assignment differs"):
        module.build_source_freeze(
            repository=fixture["repository"],
            protocol_path=fixture["protocol"],
            selection_path=fixture["selection"],
            processed_root=fixture["processed_root"],
            cut3r_checkout=fixture["cut3r"],
            checkpoint_path=fixture["checkpoint"],
            prob4d_wheel=fixture["wheel"],
            output_directory=fixture["output"],
        )
