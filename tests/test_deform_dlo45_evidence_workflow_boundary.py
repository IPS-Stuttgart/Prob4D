from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_WORKFLOW = ROOT / ".github/workflows/deform-dlo45-evidence-integrity.yml"
ARCHIVE = (
    ROOT
    / "evidence/deform-dlo45-query-observability-heldout-v1/executed-workflows"
)
EXECUTION_WORKFLOWS = (
    "deform-dlo45-observability-source-v1.yml",
    "deform-dlo45-observability-official-mirror-v1.yml",
    "deform-dlo45-observability-hosted-v1.yml",
    "deform-dlo45-local-segment-source-v2.yml",
    "deform-dlo45-query-gate-source-v1.yml",
    "deform-dlo45-query-observability-heldout-v1.yml",
)
TEMPORARY_WORKFLOWS = (
    "archive-deform-dlo45-execution-workflows-once.yml",
    "cleanup-connector-probe-issue-once.yml",
    "deform-dlo45-mechanical-format-fix.yml",
    "deform-dlo45-pr-reconcile-once.yml",
)
TEMPORARY_TRIGGERS = (
    "archive_deform_dlo45_workflows_once.trigger",
    "cleanup_connector_probe_issue_once.trigger",
    "deform_dlo45_mechanical_format_fix.trigger",
    "deform_dlo45_pr_reconcile_once.trigger",
)


def test_completed_execution_workflows_are_inert_provenance() -> None:
    assert ACTIVE_WORKFLOW.is_file()
    for name in EXECUTION_WORKFLOWS:
        assert not (ROOT / ".github/workflows" / name).exists()
        assert (ARCHIVE / name).is_file()


def test_only_the_hosted_integrity_check_remains_active() -> None:
    workflow = ACTIVE_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "self-hosted" not in workflow
    assert "contents: read" in workflow
    assert "tests/test_deform_dlo45_query_observability_evidence.py" in workflow
    assert "tests/test_github_action_pins.py" in workflow


def test_no_one_shot_helper_remains_in_the_merge_tree() -> None:
    for name in TEMPORARY_WORKFLOWS:
        assert not (ROOT / ".github/workflows" / name).exists()
    for name in TEMPORARY_TRIGGERS:
        assert not (ROOT / "protocols/execution_requests" / name).exists()
