from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "dot-rope-cut3r-pooled-evaluation-hosted-v1.yml"


def test_hosted_pooled_evaluation_is_file_triggered_and_not_self_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: DOT rope hosted pooled CUT3R evaluation v1" in text
    assert "branches: [main]" in text
    assert ("protocols/execution_requests/dot_rope_cut3r_pooled_evaluation_hosted_v1.json") in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "git push" not in text
    assert "secrets." not in text


def test_hosted_pooled_evaluation_binds_official_dot_archive() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "R01-10.zip" in text
    assert "ca546ff5f22c0279123ccb18509858ee" in text
    assert "doi:10.13021/ORC2020/XXLVXM" in text
    assert "api/datasets/:persistentId/" in text
    assert "api/access/datafile" in text
    assert "official DOT R01-10.zip checksum changed" in text
    assert "md5sum --check --strict" in text
    assert 'test "$(stat -c %s "$archive")" = "$EXPECTED_BYTES"' in text


def test_hosted_pooled_evaluation_reuses_sealed_provider_and_audit_bound_request() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Download exact previously sealed provider bundle" in text
    assert "actions: read" in text
    assert "github-token: ${{ github.token }}" in text
    assert "evaluate_dot_rope_cut3r_pooled.py" in text
    assert "validate-request" in text
    assert "Evaluate without decoding normal-view pixels" in text
    assert "complete-source-evaluation-pooled-marker-support" in text
    assert "R01-R03 source markers were opened only after prediction sealing" in text
    assert "R04-R70 remained unopened" in text
