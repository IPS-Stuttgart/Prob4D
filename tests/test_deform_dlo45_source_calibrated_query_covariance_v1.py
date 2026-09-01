from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_deform_dlo45_source_calibrated_query_covariance_v1.py"
PROTOCOL = ROOT / "protocols/deform-dlo45-source-calibrated-query-covariance-v1.json"
REQUEST = (
    ROOT / "protocols/execution_requests/deform_dlo45_source_calibrated_query_covariance_v1.json"
)
WORKFLOW = ROOT / ".github/workflows/deform-dlo45-source-calibrated-query-covariance-v1.yml"
EVIDENCE = ROOT / "evidence/deform-dlo45-source-calibrated-query-covariance-v1"


def _module():
    name = "dlo45_source_calibrated_query_covariance"
    script_path = str(SCRIPT.parent)
    sys.path.insert(0, script_path)
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(script_path)
    return module


def test_registered_protocol_and_request_are_content_addressed() -> None:
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    module.validate_protocol(protocol)
    module.validate_request(request, protocol)
    assert request["new_data_collection_authorized"] is False
    assert protocol["information_boundary"]["target_split_previously_opened"] is True


def test_equal_group_weighted_quantile_gives_each_trajectory_equal_mass() -> None:
    module = _module()
    values = {
        "long": [1.0] * 100,
        "short": [10.0],
    }
    assert module.equal_group_weighted_quantile(values, 0.5) == 1.0
    assert module.equal_group_weighted_quantile(values, 0.75) == 10.0


def test_scalar_covariance_inflation_preserves_mean_error_and_repairs_nees() -> None:
    module = _module()
    error = np.array([1.0, -2.0, 0.5])
    covariance = np.diag([0.5, 2.0, 0.25])
    inflation = 2.0
    raw = module.mahalanobis(error, covariance)
    calibrated = module.mahalanobis(error, inflation * covariance)
    assert math.isclose(calibrated, raw / inflation, rel_tol=0.0, abs_tol=1e-12)


def test_retained_evidence_records_strong_positive_without_new_data() -> None:
    calibration = json.loads((EVIDENCE / "source/calibration.json").read_text(encoding="utf-8"))
    result = json.loads((EVIDENCE / "evaluation/result.json").read_text(encoding="utf-8"))
    assert calibration["calibration_id"] == (
        "c1efc8d6fe2ec27d63083ee29b4a677faf1c37b26cd4239b54e4cd7a90a34fcd"
    )
    assert result["result_id"] == (
        "383b78a1a66a02f0f54dffa11e68303cd2ded8c6cf05877a9c7c213fcd92aca4"
    )
    assert result["decision"] == "source-calibrated-strong-positive"
    assert result["information_boundary"]["new_data_collected"] is False
    assert all(result["criteria"].values())


def test_workflow_is_hosted_source_first_and_reproduces_retained_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04" in text
    assert "repository: roahmlab/DEFORM" in text
    assert "b73b8b8ecc033caefa693fab7898741d4e6dbeff" in text
    assert "Freeze covariance on 112 training trajectories only" in text
    assert "Apply frozen covariance to the existing 28-file evaluation split" in text
    assert text.index("Freeze covariance on 112 training trajectories only") < text.index(
        "Apply frozen covariance to the existing 28-file evaluation split"
    )
    assert "Compare reproduced metrics with retained evidence" in text
    assert "new_data_collected" in text
    assert "self-hosted" not in text
    assert "gpuserver4090" not in text
    assert "git push" not in text
    assert "contents: write" not in text
    assert "secrets." not in text
