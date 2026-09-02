from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "science"
    / "run_tracking_cloth_continuous_risk_calibrated_so2_v2.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "tracking_cloth_continuous_risk_v2",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_partition_is_material_balanced_and_disjoint() -> None:
    module = _module()
    records = []
    for material in ("cotton", "denim", "wool", "polyester"):
        for index in range(8):
            records.append(
                module.RecordingRef(
                    path=Path(f"{material}-{index}.csv"),
                    relative_path=f"A2/{material}/{index}.csv",
                    label="shake" if index < 4 else "twist",
                    scenario=None,
                    material=material,
                    size="A2",
                )
            )
    protocol = {
        "source_partition": {
            "salt": "unit-test",
            "selection_per_material": 2,
            "support_calibration_per_material": 3,
            "risk_calibration_per_material": 3,
        }
    }
    first = module._source_partitions(records, protocol)
    second = module._source_partitions(list(reversed(records)), protocol)
    for role, expected in (
        ("selection", 8),
        ("support_calibration", 12),
        ("risk_calibration", 12),
    ):
        assert len(first[role]) == expected
        assert [item.group_id for item in first[role]] == [
            item.group_id for item in second[role]
        ]
    groups = [
        {item.group_id for item in first[role]}
        for role in first
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_constant_angular_velocity_candidate_and_zero_velocity_fallback() -> None:
    module = _module()

    def point(angle: float) -> np.ndarray:
        return np.array(
            [0.5, math.cos(angle), math.sin(angle)],
            dtype=float,
        )

    anchors = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )
    earlier = np.vstack((anchors, point(0.0)))
    previous = np.vstack((anchors, point(0.2)))
    current = np.vstack((anchors, point(0.4)))
    candidate, fallback, axis, increment = module._predict_probe(
        earlier,
        previous,
        current,
        "test-orbit",
    )
    np.testing.assert_allclose(candidate, point(0.4), atol=1e-12)
    np.testing.assert_allclose(fallback, point(0.2), atol=1e-12)
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-12)
    assert math.isclose(increment, 0.2, abs_tol=1e-12)


def test_continuous_case_bound_rejects_the_complete_circle() -> None:
    module = _module()
    angle = 0.4
    fallback = np.array([0.5, 1.0, 0.0])
    candidate = np.array([0.5, math.cos(angle), math.sin(angle)])
    case = module.Case(
        normalized_support_score=0.0,
        representative_mm=candidate,
        fallback_mm=fallback,
        truth_mm=candidate,
        origin_mm=np.zeros(3),
        axis=np.array([1.0, 0.0, 0.0]),
        radial_scale_mm=1.0,
        angular_increment_rad=angle,
        gauge_id="test-orbit",
    )
    value = module._case_values(
        case,
        support_threshold=0.05,
        angle_normalizer_rad=1.0,
        numerical_slack=1e-12,
    )
    assert value["actual_advantage"] > 0.0
    assert value["base_lower_advantage"] > 0.0
    assert value["support_admitted"]
    assert not value["full_circle_admitted"]
