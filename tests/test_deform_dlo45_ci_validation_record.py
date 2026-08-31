from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = (
    ROOT
    / "evidence/deform-dlo45-query-observability-heldout-v1/ci-validation.json"
)
RESULT_ID = "1ac8cd083b39877888ea0eb2f4b9400ca89eda09436f25f5f0a6f43b154b1007"
EXPECTED_CHECKS = {
    "ruff-check-and-format",
    "observable-gauge-tests",
    "query-observability-tests",
    "axial-query-certificate-tests",
    "github-action-pin-tests",
    "trusted-self-hosted-policy-tests",
    "evidence-identity-and-information-order",
    "active-versus-archived-workflow-boundary",
}


def test_final_validation_record_is_present_and_successful() -> None:
    with RECORD_PATH.open(encoding="utf-8") as stream:
        record = json.load(stream)

    assert record["schema"] == "prob4d.deform-dlo45-pr-validation"
    assert record["schema_version"] == 1
    assert record["decision"] == "pass"
    assert record["frozen_heldout_result_id"] == RESULT_ID
    assert record["post_open_retuning_permitted"] is False
    assert set(record["checks"]) == EXPECTED_CHECKS
    assert isinstance(record["github_run_id"], str)
    assert record["github_run_id"].isdigit()
    assert len(record["validated_source_revision"]) == 40
