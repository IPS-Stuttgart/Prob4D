from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_tracking_cloth_augmented_headers_v2.py"


def module():
    spec = importlib.util.spec_from_file_location("augmented_header_audit_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_free_hanging_motion_family_comes_from_public_filename() -> None:
    audit = module()
    assert audit._material_size_category(
        "tracking_dataset/Free-hanging/cotton_A2_shake_fast_hands.csv"
    ) == ("cotton", "A2", "shake")
    assert audit._material_size_category(
        "tracking_dataset/Free-hanging/wool_A3_twist_slow.csv"
    ) == ("wool", "A3", "twist")


def test_other_public_motion_directories_remain_unambiguous() -> None:
    audit = module()
    assert audit._material_size_category(
        "tracking_dataset/Hitting/polyester_A2_hitting.csv"
    ) == ("polyester", "A2", "hitting")
    assert audit._material_size_category(
        "tracking_dataset/Tablecloth/denim_A2_full_lay.csv"
    ) == ("denim", "A2", "tablecloth")
    assert audit._material_size_category(
        "tracking_dataset/Self-collisions/cotton_A2_trial.csv"
    ) == ("cotton", "A2", "self-collision")


def test_wrapper_binds_the_exact_reviewed_v1_implementation() -> None:
    audit = module()
    base = audit._load_base()
    assert base.RESULT_SCHEMA == "prob4d.tracking-cloth-augmented-header-audit-result.v1"
