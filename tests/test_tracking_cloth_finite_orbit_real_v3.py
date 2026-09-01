from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_tracking_cloth_finite_orbit_real_v3.py"
PROTOCOL = ROOT / "protocols" / "tracking-cloth-finite-orbit-real-v3.json"


def _load_module():
    name = "tracking_cloth_finite_orbit_real_v3_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_binds_header_only_support_qualification_and_target_roster() -> None:
    module = _load_module()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["protocol_id"] == module.PROTOCOL_ID
    qualification = protocol["support_qualification"]
    assert qualification["result_id"] == (
        "5f130c8643c18346ad55e7d5997deefadc20481d14c8024e641ce811f23119e0"
    )
    assert qualification["target_trajectory_values_parsed"] is False
    assert qualification["supported_target_group_count"] == 15
    assert qualification["unsupported_target_group_count"] == 27
    assert protocol["marker_support"]["selected_marker_triplet"] == ["1", "20", "5"]
    assert protocol["scientific_settings"]["target_frames_per_recording"] == 128
    assert protocol["dataset"]["target_group_ids"] == [
        module._stable_id(path) for path in protocol["dataset"]["target_relative_paths"]
    ]


def test_v3_configures_exact_target_roster_and_preserves_parent_criteria() -> None:
    module = _load_module()
    protocol = module._load_protocol(PROTOCOL)
    v2 = module._load_v2()
    module._configure_v2(v2, protocol)
    recordings = [
        SimpleNamespace(relative_path=path) for path in protocol["dataset"]["target_relative_paths"]
    ] + [SimpleNamespace(relative_path="tracking_dataset/Self-collisions/excluded.csv")]
    eligible, excluded = v2._subset(recordings, source=False)
    assert [row.relative_path for row in eligible] == protocol["dataset"]["target_relative_paths"]
    assert len(excluded) == 1
    parent = v2._load_parent(protocol, PROTOCOL)
    effective = v2._effective_parent(parent)
    assert effective["dataset"]["expected_target_files"] == 15
    assert effective["geometry"]["target_frames_per_recording"] == 128
    assert effective["geometry"]["minimum_target_cases"] == 1000
    assert effective["controlled_factor"] == parent["controlled_factor"]
    assert effective["inference"] == parent["inference"]
    assert effective["registered_criteria"] == parent["registered_criteria"]
