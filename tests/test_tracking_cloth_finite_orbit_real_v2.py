from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_tracking_cloth_finite_orbit_real_v2.py"
PROTOCOL = ROOT / "protocols" / "tracking-cloth-finite-orbit-real-v2.json"


def _load_module():
    name = "tracking_cloth_finite_orbit_real_v2_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _recording(material: str, size: str, phase: str, index: int):
    motion = "shake" if phase == "source" else "collision"
    return SimpleNamespace(relative_path=f"tracking_dataset/{material}_{size}_{motion}_{index}.csv")


def test_frozen_protocol_identity_and_parent_binding() -> None:
    module = _load_module()
    protocol = module._load_protocol(PROTOCOL)
    parent = module._load_parent(protocol, PROTOCOL)
    assert protocol["protocol_id"] == module.PROTOCOL_ID
    assert protocol["dataset"]["expected_source_files"] == 24
    assert protocol["dataset"]["expected_target_files"] == 42
    assert parent["protocol_id"] == "tracking-cloth-finite-orbit-real-v1"
    effective = module._effective_parent(parent)
    assert effective["dataset"]["expected_source_files"] == 24
    assert effective["dataset"]["expected_target_files"] == 42
    assert effective["geometry"]["maximum_common_marker_count"] == 20
    assert effective["controlled_factor"] == parent["controlled_factor"]
    assert effective["inference"] == parent["inference"]
    assert effective["registered_criteria"] == parent["registered_criteria"]


def test_protocol_rejects_any_target_side_retuning(tmp_path: Path) -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed = copy.deepcopy(protocol)
    changed["information_order"]["target_side_retuning_allowed"] = True
    changed.pop("protocol_id")
    changed["protocol_id"] = module._sha256(changed)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        module._load_protocol(path)


def test_roster_filter_is_material_and_size_explicit() -> None:
    module = _load_module()
    source = []
    target = []
    for material in ("cotton", "denim", "wool", "polyester"):
        for size in ("A2", "A3"):
            source.extend(_recording(material, size, "source", index) for index in range(8))
    for material in ("cotton", "denim", "wool", "polyester"):
        target.extend(_recording(material, "A2", "target", index) for index in range(14))
    eligible_source, excluded_source = module._subset(source, source=True)
    eligible_target, excluded_target = module._subset(target, source=False)
    assert len(eligible_source) == 24
    assert len(excluded_source) == 40
    assert len(eligible_target) == 42
    assert len(excluded_target) == 14
    assert {module._material_and_size(row.relative_path) for row in eligible_source} == {
        ("cotton", "A2"),
        ("denim", "A2"),
        ("wool", "A2"),
    }


def test_original_v1_implementation_is_loaded_and_only_parser_hooks_change() -> None:
    module = _load_module()
    base = module._load_base()
    assert base._read_markers is module.read_motive_markers
    assert base._common_marker_names is not None
    assert base._evaluate_recording.__module__ == "tracking_cloth_finite_orbit_v1_frozen"
