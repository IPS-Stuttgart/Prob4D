from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_tracking_cloth_augmented_rod_source_v1.py"


def module():
    spec = importlib.util.spec_from_file_location("augmented_rod_source_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_rigid_pair_is_ranked_first_by_all_metrics() -> None:
    audit = module()
    frames = 100
    angle = np.linspace(0.0, 1.0, frames)
    rigid_a = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(frames)))
    rigid_b = rigid_a + np.array([0.0, 0.0, 2.0])
    deforming = np.column_stack((2.0 * np.cos(angle), np.zeros(frames), angle))
    coordinates = 100.0 * np.stack((rigid_a, rigid_b, deforming), axis=1)
    rows = audit._pair_rows(
        coordinates,
        ("21", "22", "cloth"),
        minimum_valid_fraction=0.8,
        minimum_distance=25.0,
    )
    for metric in ("relative_90_spread", "relative_mad", "combined_stability"):
        rank, _ = audit._rank(rows, metric, ["21", "22"], 3)
        assert rank == 1


def test_missing_and_short_pairs_fail_support() -> None:
    audit = module()
    coordinates = np.zeros((10, 3, 3), dtype=np.float64)
    coordinates[:, 1, 0] = 0.001
    coordinates[:, 2, 0] = 0.002
    coordinates[:5, 2] = np.nan
    try:
        audit._pair_rows(
            coordinates,
            ("a", "b", "c"),
            minimum_valid_fraction=0.8,
            minimum_distance=25.0,
        )
    except RuntimeError as error:
        assert "no candidate marker pairs" in str(error)
    else:
        raise AssertionError("unsupported pairs must fail closed")
