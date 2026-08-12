from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ecosystem-release-capsule.yml"


def test_repository_owned_integration_tests_trigger_ecosystem_capsule() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    pull_request, remainder = text.split("  push:\n", maxsplit=1)
    push, _ = remainder.split("  workflow_dispatch:\n", maxsplit=1)

    for path_filter in (
        '      - "integration_tests/**"',
        '      - "src/prob4d/public_api_manifest.py"',
        '      - "tests/test_ecosystem_capsule_workflow.py"',
    ):
        assert path_filter in pull_request
        assert path_filter in push
        assert text.count(path_filter) == 2


def test_workflow_materializes_and_reverifies_bound_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "THREE_REPOSITORY_EVIDENCE_OUTPUT" in text
    assert "prob4d-ecosystem-evidence" in text
    assert text.count('--evidence-root "${evidence_root}"') == 2
    assert "public-api-manifest.json" not in text
    assert "accepted_selection_artifact_id" in text
    assert "rejected_selection_artifact_id" in text
    assert "exact_fallback_identity" in text


def test_workflow_uploads_capsule_log_and_complete_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Upload capsule, log, and bound evidence" in text
    assert "${{ runner.temp }}/prob4d-ecosystem-evidence/" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 90" in text


def test_workflow_keeps_external_actions_and_checkouts_fail_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "actions/upload-artifact@v" not in text
    assert text.count("persist-credentials: false") == 3
    assert "permissions:\n  contents: read" in text
