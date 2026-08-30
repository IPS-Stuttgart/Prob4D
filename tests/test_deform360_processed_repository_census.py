from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_deform360_source_bundle.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deform360_processed_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_processed_census_is_metadata_only_and_groups_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_module()
    source = tmp_path / "source"
    processed = source / "processed"
    for object_id in ("001-rope", "008-pink-cloth"):
        for episode_id in ("episode_0", "episode_1"):
            episode = processed / object_id / episode_id
            tracking = episode / "tracking"
            tracking.mkdir(parents=True)
            (episode / "aligned_timestamps.txt").write_text("do not read", encoding="utf-8")
            (episode / "cameras.tar").write_bytes(b"camera payload")
            (tracking / "vel.npy").write_bytes(b"velocity payload")
    forbidden = processed / "001-rope" / "target-secret"
    forbidden.mkdir(parents=True)
    (forbidden / "outcome.json").write_text("secret", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    (external / "must-not-read.bin").write_bytes(b"secret")
    (processed / "008-pink-cloth" / "episode_1" / "external-link").symlink_to(
        external,
        target_is_directory=True,
    )

    original_open = builtins.open

    def guarded_open(file: object, *args: object, **kwargs: object):
        path = Path(file) if isinstance(file, (str, os.PathLike)) else None
        if path is not None and path.is_relative_to(source):
            raise AssertionError(f"dataset content open attempted: {path}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    decision, census = audit.build_processed_repository_census(
        source,
        forbidden_tokens=("target",),
        max_entries=1_000,
        max_depth=8,
    )

    assert decision == "processed-census-present"
    assert census["object_count"] == 2
    assert census["episode_count"] == 4
    assert census["global_extension_counts"] == {".npy": 4, ".tar": 4, ".txt": 4}
    assert census["global_basename_counts"] == {
        "aligned_timestamps.txt": 4,
        "cameras.tar": 4,
        "vel.npy": 4,
    }
    assert len(census["layout_signatures"]) == 2
    assert census["forbidden_token_counts"] == {"target": 1}
    serialized = repr(census)
    assert "outcome.json" not in serialized
    assert "must-not-read.bin" not in serialized
    linked_episode = next(
        episode
        for episode in census["episodes"]
        if episode["object_id"] == "008-pink-cloth"
        and episode["episode_id"] == "episode_1"
    )
    assert linked_episode["counts"]["symlink"] == 1
    assert census["execution_boundary"] == {
        "dataset_file_contents_opened": False,
        "symlinks_followed": False,
        "dataset_mutated": False,
    }


def test_processed_census_rejects_missing_or_linked_root(tmp_path: Path) -> None:
    audit = _load_module()
    missing_decision, _ = audit.build_processed_repository_census(
        tmp_path / "missing",
        forbidden_tokens=(),
        max_entries=10,
        max_depth=4,
    )
    assert missing_decision == "processed-root-missing"

    physical = tmp_path / "physical"
    physical.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "processed").symlink_to(physical, target_is_directory=True)
    linked_decision, linked = audit.build_processed_repository_census(
        source,
        forbidden_tokens=(),
        max_entries=10,
        max_depth=4,
    )
    assert linked_decision == "processed-root-symlink-rejected"
    assert "symlink" in linked["error"]
