from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_tracking_cloth_finite_orbit_robustness_v1.py"
PROTOCOL = ROOT / "protocols" / "tracking-cloth-finite-orbit-robustness-v1.json"


def _module():
    name = "tracking_cloth_finite_orbit_robustness_v1_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_marks_result_as_secondary_and_source_calibrated() -> None:
    module = _module()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["protocol_id"] == module.PROTOCOL_ID
    assert protocol["source_selection"]["target_side_retuning"] is False
    assert protocol["information_order"][
        "threshold_selected_from_source_trajectories_before_target_loading"
    ] is True
    assert protocol["claim_boundary"]["fresh_target_holdout"] is False
    assert protocol["claim_boundary"]["posthoc_secondary_analysis"] is True


def test_strict_local_motive_parser_reads_registered_triplet(tmp_path: Path) -> None:
    module = _module()
    rows = [
        ["Format Version", "1.23"],
        ["Take Name", "fixture"],
        ["Frame", "Time", "Marker", "Marker", "Marker", "Marker", "Marker", "Marker", "Marker", "Marker", "Marker"],
        ["", "", "1", "1", "1", "20", "20", "20", "5", "5", "5"],
        ["", "", "101", "101", "101", "120", "120", "120", "105", "105", "105"],
        ["", "", "Position", "Position", "Position", "Position", "Position", "Position", "Position", "Position", "Position"],
        ["Frame", "Time", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
        ["0", "0.0", "0", "0", "0", "1", "0", "0", "0.5", "0.2", "0.1"],
        ["1", "0.1", "0", "0", "0", "1", "0", "0", "0.5", "0.3", "0.1"],
    ]
    path = tmp_path / "fixture.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    frames, markers = module._parse_recording(path, ["1", "20", "5"])
    np.testing.assert_array_equal(frames, [0, 1])
    np.testing.assert_allclose(markers["20"], [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(markers["5"][1], [0.5, 0.3, 0.1])


def test_closed_form_and_dense_orbit_range_agree() -> None:
    module = _module()
    case = module._case(
        "fixture",
        "fixture.csv",
        4,
        np.array([-0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.1]),
        0.01,
    )
    assert case is not None
    closed, _, _ = module._estimated_orbit_range(case, 0.02, 3, "radial", None)
    dense, _, _ = module._estimated_orbit_range(case, 0.02, 3, "radial", 4096)
    assert abs(closed - dense) < 1e-6


def test_noiseless_axis_accepts_invariant_and_rejects_radial() -> None:
    module = _module()
    case = module._case(
        "fixture",
        "fixture.csv",
        1,
        np.array([-0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.1]),
        0.01,
    )
    assert case is not None
    rows = module._case_records([case], 0.0, 4, 16, 0.01, None)
    metrics = module._group_metrics(rows)[0]
    assert metrics["invariant_acceptance"] == 1.0
    assert metrics["radial_rejection"] == 1.0
    assert metrics["harmful_accepted_radial_fraction_all"] == 0.0
    assert metrics["guarded_rmse_fraction"] < metrics["local_rmse_fraction"]
