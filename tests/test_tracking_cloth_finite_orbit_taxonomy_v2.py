from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_tracking_cloth_finite_orbit_real_v1.py"
PROTOCOL_V1 = ROOT / "protocols/tracking-cloth-finite-orbit-real-v1.json"
PROTOCOL_V2 = ROOT / "protocols/tracking-cloth-finite-orbit-real-v2.json"


def _load_module() -> ModuleType:
    name = "tracking_cloth_finite_orbit_taxonomy_v2"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _publisher_roster(root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(root / f"denim_A2_shake_{index:02d}.csv" for index in range(32))
    paths.extend(root / f"denim_A2_twist_{index:02d}.csv" for index in range(32))
    paths.extend(root / f"cotton_A2_half_lay_{index:02d}.csv" for index in range(8))
    paths.extend(root / f"cotton_A2_full_lay_{index:02d}.csv" for index in range(8))
    paths.extend(root / f"polyester_A2_hitting_{index:02d}.csv" for index in range(4))
    paths.extend(root / f"wool_A2_self_collision_{index:02d}.csv" for index in range(36))
    assert len(paths) == 120
    return paths


def test_v2_taxonomy_recovers_publisher_documented_64_56_partition(
    tmp_path: Path,
) -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))

    recordings, classification = module._classify_recordings(
        tmp_path,
        _publisher_roster(tmp_path),
        protocol,
    )

    assert classification == {
        "mode": "declared-aliases",
        "counts": {"collision": 56, "shake": 32, "twist": 32},
    }
    assert sum(recording.label != "collision" for recording in recordings) == 64
    assert sum(recording.label == "collision" for recording in recordings) == 56


def test_v1_taxonomy_reproduces_pre_outcome_classification_failure(
    tmp_path: Path,
) -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL_V1.read_text(encoding="utf-8"))

    with pytest.raises(
        RuntimeError,
        match=r"alias counts were .*'collision': 36.*",
    ):
        module._classify_recordings(
            tmp_path,
            _publisher_roster(tmp_path),
            protocol,
        )


def test_v2_changes_only_taxonomy_and_protocol_identity() -> None:
    v1 = json.loads(PROTOCOL_V1.read_text(encoding="utf-8"))
    v2 = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))

    assert v2["protocol_id"] == "tracking-cloth-finite-orbit-real-v2"
    assert v2["supersedes_failed_protocol_id"] == v1["protocol_id"]
    assert v2["geometry"] == v1["geometry"]
    assert v2["controlled_factor"] == v1["controlled_factor"]
    assert v2["inference"] == v1["inference"]
    assert v2["registered_criteria"] == v1["registered_criteria"]
    assert v2["claim_boundary"] == v1["claim_boundary"]
    assert v2["dataset"]["expected_source_files"] == 64
    assert v2["dataset"]["expected_target_files"] == 56
    assert v2["taxonomy_provenance"]["csv_contents_opened_before_v2_freeze"] is False
    assert v2["taxonomy_provenance"]["scientific_outcomes_opened_before_v2_freeze"] is False
