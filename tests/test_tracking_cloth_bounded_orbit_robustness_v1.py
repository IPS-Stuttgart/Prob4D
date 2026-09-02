from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_tracking_cloth_bounded_orbit_robustness_v1.py"
PROTOCOL = ROOT / "protocols" / "tracking-cloth-bounded-orbit-robustness-v1.json"


def _load_module():
    name = "tracking_cloth_bounded_orbit_robustness_v1_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_binds_parent_result_and_secondary_information_order() -> None:
    module = _load_module()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["protocol_id"] == module.PROTOCOL_ID
    assert protocol["parent_result"]["result_id"] == module.PARENT_RESULT_ID
    assert protocol["information_order"][
        "parent_target_outcomes_were_opened_before_this_secondary_study"
    ]
    assert protocol["information_order"]["target_side_retuning_allowed"] is False
    assert protocol["information_order"]["self_collision_target_trajectories_opened"] is False


def test_hash_direction_is_deterministic_and_unit_length() -> None:
    module = _load_module()
    first = module._hash_direction(1, "recording", 17, 0.02, 4, "probe")
    second = module._hash_direction(1, "recording", 17, 0.02, 4, "probe")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first), 1.0, atol=1e-15)


def test_synthetic_group_preserves_outer_containment_and_zero_false_acceptance() -> None:
    module = _load_module()
    protocol = module._load_protocol(PROTOCOL)
    cases = []
    for frame_index, radius in enumerate((0.2, 0.5, 0.8, 1.1)):
        cases.append(
            {
                "frame_index": frame_index,
                "anchor_a": np.array([0.0, 0.0, 0.0]),
                "anchor_b": np.array([2.0, 0.0, 0.0]),
                "probe": np.array([0.7, radius, 0.0]),
                "anchor_distance": 2.0,
                "axial_coordinate": 0.7,
                "radius": radius,
                "true_width": 2.0 * radius,
            }
        )
    metadata = {
        "group_id": "synthetic",
        "relative_path": "synthetic.csv",
        "sampled_case_count": len(cases),
        "available_rows": len(cases),
        "unit_scale_to_mm": 1.0,
    }
    result = module._evaluate_group(
        cases,
        metadata,
        {"q25": 0.6, "q50": 1.2, "q75": 1.8},
        protocol,
    )
    for noise in result["noise_levels"]:
        assert noise["outer_orbit_containment"] == 1.0
        assert noise["maximum_outer_width_violation_mm"] <= 0.0
        assert all(row["outer_false_accept"] == 0.0 for row in noise["thresholds"])
    zero = result["noise_levels"][0]
    assert zero["plugin_orbit_containment"] == 1.0
    assert zero["plugin_width_ratio_mean"] == 1.0
    assert zero["outer_width_ratio_mean"] == 1.0
