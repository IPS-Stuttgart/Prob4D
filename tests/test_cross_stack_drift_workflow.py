from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cross-stack-drift.yml"


def test_drift_workflow_is_scheduled_manual_hosted_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "  schedule:\n" in text
    assert '    - cron: "17 3 * * 1"' in text
    assert "  workflow_dispatch:\n" in text
    assert "pull_request:" not in text
    assert "  push:\n" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "permissions:\n  contents: read" in text
    assert "secrets." not in text


def test_drift_workflow_resolves_current_main_without_claim_bearing_pins() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("          ref: main") == 3
    assert text.count("persist-credentials: false") == 3
    assert "BAYESIAN_PHYSTWIN_REF" not in text
    assert "CAUSAL4D_REF" not in text
    assert "resolved-revisions.txt" in text
    assert "git -C prob4d rev-parse HEAD" in text
    assert "git -C bayesian-phystwin rev-parse HEAD" in text
    assert "git -C causal4d rev-parse HEAD" in text


def test_drift_workflow_uses_only_isolated_installed_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("python -m build --wheel") == 3
    assert "PYTHONNOUSERSITE=1" in text
    assert "env -u PYTHONPATH" in text
    assert '"${RUNNER_TEMP}/cross-stack-drift/bin/python" -I -m pytest' in text
    assert "test_three_repository_metamorphic_v1.py" in text
    assert "python -m pip install -e" not in text


def test_drift_workflow_uses_immutable_action_pins_and_retains_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "actions/upload-artifact@v" not in text
    assert ": > cross-stack-drift.log" in text
    assert ": > cross-stack-drift.xml" in text
    assert ": > wheel-sha256.txt" in text
    assert "cross-stack-drift.log" in text
    assert "cross-stack-drift.xml" in text
    assert "wheel-sha256.txt" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text
