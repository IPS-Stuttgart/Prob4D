"""Regression tests for the DOT R04-R10 multiview support audit."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_dot_r04_r10_camera_support.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dot_camera_support_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dot_archive_uses_shared_cam001_3d_carrier() -> None:
    module = _load_module()
    assert module.SHARED_3D_CAMERA == "cam001"
    assert (
        module.LAYOUT_CENSUS_ID
        == "74f090a99d6740ac3388c43493531ea5168291e4c5a709fb74344e45b46b4f19"
    )
    two_d, three_d = module.coordinate_members("R04", 7, "cam005")
    assert two_d == "R04/coordinates/2d/frame000007_cam005.txt"
    assert three_d == "R04/coordinates/3d/frame000007_cam001.txt"


def test_audit_rejects_silent_2d_3d_row_truncation() -> None:
    module = _load_module()
    source = inspect.getsource(module.audit)
    assert "camera-specific 2-D and shared 3-D row counts differ" in source
    assert "count = min(len(rows_2d), len(rows_3d))" not in source


def test_result_exposes_coordinate_layout_provenance() -> None:
    module = _load_module()
    source = inspect.getsource(module.audit)
    assert '"shared_3d_camera_label": SHARED_3D_CAMERA' in source
    assert '"layout_census_id": LAYOUT_CENSUS_ID' in source
    assert '"shared_3d_member": three_name' in source
