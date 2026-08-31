from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "dot-rope-pooled-result-finalization-v1.yml"


def test_finalization_workflow_is_request_bound_and_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: DOT rope pooled-result provenance finalization v1" in text
    assert "branches: [main]" in text
    assert "protocols/execution_requests/dot_rope_pooled_result_finalization_v1.json" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "pull_request_target:" not in text
    assert "github.event.pull_request.head.sha" not in text
    assert "requirements/ci/quality.txt" in text
    assert "python -m pip install -r requirements/ci/quality.txt" in text
    assert 'python -m pip install -e ".[dev]"' not in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "git push" not in text
    assert "secrets." not in text


def test_finalization_workflow_verifies_exact_source_artifact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Verify exact source artifact metadata" in text
    assert "/actions/artifacts/" in text
    assert "source artifact ID changed" in text
    assert "source artifact name changed" in text
    assert "source artifact digest changed" in text
    assert "source artifact workflow run changed" in text
    assert "actions: read" in text
    assert "github-token: ${{ github.token }}" in text


def test_finalization_workflow_is_metadata_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "finalize_dot_rope_pooled_result.py" in text
    assert "Finalize metadata without dataset access or rescoring" in text
    assert "scientific payload was not preserved" in text
    assert "No dataset was opened and no score was recomputed" in text
    assert "DATASET_ROOT" not in text
    assert "R01-10.zip" not in text
