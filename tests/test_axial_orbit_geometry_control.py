from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "axial-orbit-geometric-error-control-v1.json"
STUDY = ROOT / "scripts" / "science" / "run_axial_orbit_geometric_error_control.py"


def _load_study():
    name = "axial_orbit_geometric_error_control_test"
    spec = importlib.util.spec_from_file_location(name, STUDY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_protocol_and_reduced_deterministic_control() -> None:
    module = _load_study()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["schema"] == (
        "prob4d.axial-orbit-geometric-error-control.v1"
    )
    assert protocol["case_count"] == 5000
    assert protocol["point_count_range"] == [3, 20]
    assert protocol["query_dimension_range"] == [1, 8]

    reduced = json.loads(json.dumps(protocol))
    reduced["case_count"] = 1000
    reduced["registered_checks"]["minimum_nonzero_ratio_count"] = 990
    first = module.build_report(reduced)
    second = module.build_report(reduced)
    assert first["decision"] == "passed"
    assert all(first["registered_checks"].values())
    assert first["registered_checks"] == second["registered_checks"]
    assert first["maximum_operator_bound_violation"] == 0.0
    assert first["actual_to_bound_ratio_quantiles"] == second[
        "actual_to_bound_ratio_quantiles"
    ]
    assert first["coefficient_bound_quantiles"] == second[
        "coefficient_bound_quantiles"
    ]
    assert first["actual_to_bound_ratio_quantiles"]["q100"] <= 1.0 + 1e-12
    assert first["actual_to_bound_ratio_quantiles"]["q100"] >= 0.01
