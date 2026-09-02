from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d._tracking_cloth_approximate_orbit_data import (
    Recording,
    _split,
    _stable_id,
)
from prob4d._tracking_cloth_approximate_orbit_evaluation import (
    _method_metrics,
    _pair_evidence,
)
from prob4d._tracking_cloth_approximate_orbit_io import (
    CALIBRATION_SCHEMA,
    _load_calibration,
    _load_protocol,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "tracking-cloth-approximate-orbit-tube-v1.json"


def test_protocol_is_content_addressed() -> None:
    value = _load_protocol(PROTOCOL)
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id")
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == supplied
    assert value["split"]["expected_calibration_groups"] == 18
    assert value["split"]["expected_target_groups"] == 9


def test_within_material_split_is_disjoint_and_balanced() -> None:
    protocol = _load_protocol(PROTOCOL)
    cohort = []
    for material in ("cotton", "denim", "wool"):
        for index in range(9):
            relative = f"tracking_dataset/Self-collisions/{material}_A2_case_{index}.csv"
            cohort.append(
                Recording(
                    path=Path(relative),
                    relative_path=relative,
                    group_id=_stable_id(relative),
                    material=material,
                )
            )
    calibration, target = _split(cohort, protocol)
    assert len(calibration) == 18
    assert len(target) == 9
    assert not ({row.group_id for row in calibration} & {row.group_id for row in target})
    for material in ("cotton", "denim", "wool"):
        assert sum(row.material == material for row in calibration) == 6
        assert sum(row.material == material for row in target) == 3


def test_pair_evidence_removes_unresolved_angle_from_orbit_score() -> None:
    previous = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [20.0, 0.0, 25.0],
        ]
    )
    current = np.array(
        [
            [10.0, 5.0, -2.0],
            [10.0, 5.0, 98.0],
            [10.0, 25.0, 23.0],
        ]
    )
    evidence = _pair_evidence(previous, current)
    assert evidence is not None
    assert evidence.orbit_score_mm == pytest.approx(0.0, abs=1e-12)
    assert evidence.point_score_mm == pytest.approx(np.sqrt(800.0))
    assert evidence.query_true_mm == pytest.approx(-25.0)
    assert evidence.query_center_mm == pytest.approx(-25.0)


def test_pair_evidence_orbit_distance_tracks_axial_and_radial_drift() -> None:
    previous = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [20.0, 0.0, 25.0],
        ]
    )
    current = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [23.0, 0.0, 29.0],
        ]
    )
    evidence = _pair_evidence(previous, current)
    assert evidence is not None
    assert evidence.orbit_score_mm == pytest.approx(5.0)
    assert evidence.query_true_mm - evidence.query_center_mm == pytest.approx(4.0)


def test_method_metrics_abstains_when_interval_crosses_threshold() -> None:
    truth = np.array([-2.0, 3.0, 10.0])
    center = np.array([-1.0, 1.0, 8.0])
    exact = _method_metrics(truth, center, 0.0)
    wide = _method_metrics(truth, center, 2.0)
    assert exact["accepted_count"] == 3
    assert wide["accepted_count"] == 1
    assert wide["harmful_accepted_count"] == 0


def test_calibration_loader_rejects_target_access(tmp_path: Path) -> None:
    protocol = _load_protocol(PROTOCOL)
    value = {
        "schema": CALIBRATION_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "information_order": {"target_trajectory_values_parsed": True},
    }
    value["calibration_id"] = _sha256(value)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="accessed target"):
        _load_calibration(path, protocol)
