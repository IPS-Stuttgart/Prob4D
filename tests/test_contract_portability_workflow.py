from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "contract-portability.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_contract_portability_workflow_is_read_only_and_consolidated() -> None:
    text = _workflow_text()
    assert "permissions:\n  contents: read" in text
    assert "pull_request_target" not in text
    assert "contents: write" not in text
    assert "installed-wheel-typing:" in text
    assert "atomic-publication-portability:" in text
    assert text.count("persist-credentials: false") == 2


def test_installed_typing_gate_rejects_any_and_invalid_calls() -> None:
    text = _workflow_text()
    assert "--disallow-any-expr" in text
    assert "--disallow-any-unimported" in text
    assert "consumer_v2.py" in text
    assert "consumer_v2_invalid.py" in text
    assert 'grep -F "[arg-type]" typing-invalid.txt' in text


def test_portability_matrix_covers_macos_and_windows() -> None:
    text = _workflow_text()
    assert "os: [macos-latest, windows-latest]" in text
    assert "tests/test_atomic_evidence_publication.py" in text
    assert "tests/test_data_storage.py" in text
    assert "tests/test_prediction_store.py" in text
