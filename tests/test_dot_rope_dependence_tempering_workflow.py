from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "dot-rope-dependence-tempering-source-v1.yml"


def test_source_calibration_is_single_file_triggered_and_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: DOT rope dependence tempering source v1" in text
    assert "branches: [main]" in text
    assert "protocols/execution_requests/dot_rope_dependence_tempering_source_v1.json" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "git push" not in text
    assert "secrets." not in text


def test_source_calibration_reuses_sealed_provider_and_official_archive() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "dot-rope-cut3r-sealed-provider-33329701704-1" not in text
    assert "provider_artifact_name" in text
    assert "Download exact sealed provider bundle" in text
    assert "actions: read" in text
    assert "R01-10.zip" in text
    assert "ca546ff5f22c0279123ccb18509858ee" in text
    assert "doi:10.13021/ORC2020/XXLVXM" in text
    assert "md5sum --check --strict" in text


def test_source_calibration_preserves_confirmation_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "calibrate_dot_rope_dependence_tempering.py" in text
    assert "source-dependence-strength-frozen" in text
    assert "confirmation_payloads_opened" in text
    assert "Only already-open R01-R03 source outcomes were used" in text
    assert "R04-R70 remained unopened" in text
