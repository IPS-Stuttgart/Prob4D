from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-pcd-query-inventory.yml"


def test_workflow_is_exact_main_request_bound() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "\n  push:" in text
    assert "branches: [main]" in text
    assert "protocols/execution_requests/deform360_pcd_query_inventory_v1.json" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert 'test "$EVENT_AFTER" = "$(/usr/bin/git rev-parse HEAD)"' in text
    assert "/usr/bin/git diff --name-only" in text
    assert "validate-request" in text
    assert "source_protocol_git_blob_sha" in text


def test_self_hosted_job_is_read_only_and_on_gpuserver4090() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "environment: trusted-self-hosted-validation" in text
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in text
    assert 'test "$RUNNER_NAME" = "workstation1"' in text
    assert (
        "SOURCE_ROOT: /mnt/seagate10tb/florianpfaff/datasets/deform360/"
        "processed-repository/processed"
    ) in text
    assert "permissions:\n      contents: read" in text
    assert "persist-credentials: false" in text
    assert "secrets." not in text
    assert "git push" not in text


def test_workflow_keeps_payloads_arrays_and_targets_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'archive_member_payload_reads_authorized: "false"' in text
    assert 'numeric_array_reads_authorized: "false"' in text
    assert 'provider_predictions_authorized: "false"' in text
    assert 'physical_query_scoring_authorized: "false"' in text
    assert 'target_outcomes_authorized: "false"' in text
    assert 'dataset_mutation_authorized: "false"' in text
    assert "tar headers only" in text.lower()
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
