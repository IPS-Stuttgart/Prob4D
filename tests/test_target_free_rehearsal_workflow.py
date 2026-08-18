from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "target-free-rehearsal.yml"


def test_target_free_rehearsal_workflow_is_exact_head_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Target-free observation rehearsal" in text
    assert "permissions:\n  contents: read" in text
    assert "github.event.pull_request.head.sha || github.sha" in text
    assert "ref: ${{ env.REHEARSAL_SOURCE_REVISION }}" in text
    assert "persist-credentials: false" in text
    assert "continue-on-error" not in text
    assert "secrets." not in text
    assert "sudo " not in text


def test_target_free_rehearsal_workflow_uses_installed_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m build" in text
    assert "find dist -maxdepth 1 -name '*.whl'" in text
    assert '"$bin/prob4d" diagnostic target-free-rehearsal run' in text
    assert '"$bin/prob4d" diagnostic target-free-rehearsal verify' in text
    assert '"$bin/python" -m prob4d_independent_verifier' in text
    assert "target_free_rehearsal_receipt.json" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
