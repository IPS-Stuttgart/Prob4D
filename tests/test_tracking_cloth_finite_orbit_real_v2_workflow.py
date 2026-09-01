from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/tracking-cloth-finite-orbit-real-v2.yml"


def test_workflow_is_hosted_checksum_bound_and_terminal_result_preserving() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    prefix = text.split("permissions:", maxsplit=1)[0]
    assert "repair/tracking-cloth-finite-orbit-motive-parser-v1" in prefix
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted" not in text
    assert "10.5281/zenodo.14644526" in text
    assert "b4868b702f8a42b2ea1069d0f1a3b8f6" in text
    assert 'test "$count" -eq 120' in text
    assert "run_tracking_cloth_finite_orbit_real_v2.py" in text
    assert "target-marker-support-negative" in text
    assert "evaluated-real-geometry-passed" in text
    assert "evaluated-real-geometry-failed" in text
    assert "unsupported target group was replaced" in text
    assert "Remove raw public release before evidence upload" in text
    assert "Raw dataset payload appeared in evidence" in text


def test_workflow_binds_and_verifies_the_parser_only_predecessor_failure() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'PREDECESSOR_RUN_ID: "33361712662"' in text
    assert 'PREDECESSOR_ARTIFACT_ID: "9777455628"' in text
    assert "tracking-cloth-finite-orbit-real-v1-1" in text
    assert "sha256:88f0371a63bb00bc7c9619acefa518e3b5dfd6b473be0db8466482ea98de080d" in text
    assert "61b249cdedbc8b31e31b96f6692962090b42b6e5221c71bdbb507d43a013cb88" in text
    assert "fewer than three common 3-D marker groups: []" in text
    assert "source_seal.json" in text
    assert "result.json" in text
    assert "predecessor unexpectedly contains scientific evidence" in text


def test_workflow_verifies_source_seal_before_target_result_semantics() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    source_check = text.index('source_seal["target_header_opened"] is False')
    result_check = text.index('result["status"] in allowed')
    assert source_check < result_check
    assert 'result["protocol_id"] ==' in text
    assert 'support["unsupported_groups_replaced"] is False' in text
    assert 'aggregate["target_groups"] == 42' in text
    assert 'result["cohort"]["source_groups"] == 24' in text
    assert 'result["cohort"]["target_groups"] == 42' in text


def test_external_actions_are_pinned_to_full_commit_shas() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        target = stripped.split(maxsplit=1)[1]
        if target.startswith("./"):
            continue
        revision = target.rsplit("@", maxsplit=1)[1].split(maxsplit=1)[0]
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)
