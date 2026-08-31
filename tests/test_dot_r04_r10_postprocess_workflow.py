from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-r04-r10-postprocess-v1.yml"


def _job(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"\n  {name}:")
    end = text.index(f"\n  {next_name}:", start) if next_name else len(text)
    return text[start:end]


def test_postprocessor_is_exact_run_bound_and_hosted_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    prefix = text.split("permissions:", maxsplit=1)[0]
    assert "\n  workflow_run:" in prefix
    assert 'workflows: ["DOT rope CUT3R held-out confirmation v1"]' in prefix
    assert "types: [completed]" in prefix
    assert 'TARGET_RUN_ID: "33434695566"' in text
    assert "TARGET_HEAD_SHA: 9e1b77b2e70685881db7f188a95a3a91443275e8" in text
    assert "run_attempt" in text
    assert '!= 2' in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "secrets." not in text
    assert "git push" not in text
    assert "contents: write" not in text


def test_recovery_requires_provider_seal_and_exact_archive_failure_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    inspect = _job(text, "inspect", "recover_evaluation")
    recover = _job(text, "recover_evaluation", "materialize")
    assert 'provider.get("conclusion") == "success"' in inspect
    assert '"Verify provider seal before downloading marker archive": "success"' in inspect
    assert '"Download and verify official R01-R10 marker archive": "failure"' in inspect
    assert '"Evaluate the frozen alpha and comparators": "skipped"' in inspect
    assert 'mode = "recover-evaluation"' in inspect
    assert "Reverify provider seal before any marker download" in recover
    assert "Accept-Encoding: identity" in recover
    assert "expected_bytes" in recover
    assert "ARCHIVE_MD5" in recover
    assert "verify_dot_rope_cut3r_heldout_result.py" in recover
    assert "r11" not in recover.lower()


def test_materializer_remains_strong_positive_only_and_emits_no_repository_write() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    materialize = _job(text, "materialize")
    assert "Independently verify decision and evidence identities" in materialize
    assert "SOURCE_RUN_ID: ${{ steps.source.outputs.run_id }}" in materialize
    assert "SOURCE_ARTIFACT_NAME: ${{ steps.source.outputs.name }}" in materialize
    assert "if: steps.verify.outputs.decision == 'heldout-strong-positive'" in materialize
    assert "materialize_dot_rope_query_selective_request.py" in materialize
    assert "run_dot_rope_query_selective_heldout.py" in materialize
    assert "validate-request" in materialize
    assert "test ! -e \"$R11_REQUEST\"" in materialize
    assert "rm -f -- \"$R11_REQUEST\"" in materialize
    assert "git status --porcelain" in materialize
    assert "git push" not in materialize
    assert "contents: write" not in materialize


def test_all_external_actions_are_commit_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        target = stripped.split()[1]
        if target.startswith("./"):
            continue
        revision = target.rsplit("@", 1)[1]
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)
