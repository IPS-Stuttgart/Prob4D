from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cut3r-exact-retry-dispatch.yml"
REQUEST = (
    ROOT / "protocols" / "dispatch_requests" / "cut3r_deform360_source_freeze_exact_retry_v1.json"
)
HISTORICAL_EXECUTION_SHA = "8b923e8cd67ca65f09312cffe305e36852f36fbb"
RETAINED_REQUEST_ID = "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
REQUIRED_RUNNER_LABELS = [
    "self-hosted",
    "Linux",
    "X64",
    "nvidia-smi",
    "data-prob4d-deform360-source-v1",
    "prob4d-cut3r",
]


def test_dispatch_request_is_exact_and_target_closed() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert request["schema"] == ("prob4d.cut3r-source-freeze-exact-retry-dispatch-request")
    assert request["schema_version"] == 1
    assert request["repository"] == "IPS-Stuttgart/Prob4D"
    assert request["issue_number"] == 49
    assert request["target_workflow"] == "cut3r-source-freeze-auto-v2.yml"
    assert request["target_ref"] == "main"
    assert request["historical_execution_sha"] == HISTORICAL_EXECUTION_SHA
    assert request["retained_request_id"] == RETAINED_REQUEST_ID
    assert request["retained_request_path"] == (
        "protocols/execution_requests/cut3r_deform360_source_freeze_v2.json"
    )
    assert request["superseded_workflow_run_id"] == 32621813949
    assert request["superseded_run_expected_head_sha"] == (HISTORICAL_EXECUTION_SHA)
    assert request["superseded_run_expected_event"] == "push"
    assert request["superseded_execute_job_name"] == (
        "Freeze retained source inputs from trusted merged main"
    )
    assert request["required_runner_labels"] == REQUIRED_RUNNER_LABELS
    assert request["scientific_inputs_changed"] is False
    assert request["source_outcomes_opened"] is False
    assert request["target_outcomes_opened"] is False
    assert "changes no scientific input" in request["claim_boundary"]


def test_dispatch_workflow_is_hosted_main_bound_and_one_shot() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "\n  pull_request:" in text
    assert "\n  push:" in text
    assert "branches: [main]" in text
    assert "pull_request_target:" not in text
    assert "workflow_dispatch:" not in text
    assert ("protocols/dispatch_requests/cut3r_deform360_source_freeze_exact_retry_v1.json") in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_AFTER" = "$EXPECTED_SHA"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted" not in text
    assert "persist-credentials: false" in text


def test_dispatch_workflow_fails_closed_before_exact_retry() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "authorize-retry" in text
    assert '--execution-revision "$HISTORICAL_EXECUTION_SHA"' in text
    assert '--expected-request-id "$RETAINED_REQUEST_ID"' in text
    assert HISTORICAL_EXECUTION_SHA in text
    assert RETAINED_REQUEST_ID in text
    assert "32621813949" in text
    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, "
        "data-prob4d-deform360-source-v1, prob4d-cut3r]"
    ) in text
    assert "/actions/runs/{stale_run_id}/jobs?per_page=100" in text
    assert "/actions/runs/{stale_run_id}/artifacts?per_page=100" in text
    assert 'f"{run_path}/cancel"' in text
    assert "already succeeded; refusing duplicate dispatch" in text
    assert "refusing to classify it as zero-evidence" in text
    assert "did not reach a terminal state; exact retry not dispatched" in text
    assert "actions: write" in text
    assert "issues: write" in text
    assert 'f"/repos/{repository}/actions/workflows/{target_workflow}/dispatches"' in text
    assert '"execution_sha": expected_head' in text
    assert '"request_id": os.environ["RETAINED_REQUEST_ID"]' in text
