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


def _module():
    name = "dlo45_source_calibrated_query_covariance"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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


def test_workflow_is_file_triggered_source_first_and_no_new_data() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "science/deform-dlo45-source-calibrated-covariance-v1" in text
    assert "deform_dlo45_source_calibrated_query_covariance_v1.json" in text
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "needs: [contract, source-calibration]" in text
    assert "new_data_collection_authorized" in text
    assert "source_calibration_authorized" in text
    assert "existing_eval_reanalysis_authorized" in text
    assert "git push" not in text
    assert "contents: write" not in text
    assert "secrets." not in text
