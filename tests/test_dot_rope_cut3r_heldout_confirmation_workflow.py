from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "dot-rope-cut3r-heldout-confirmation-v1.yml"
)


def test_confirmation_workflow_is_single_request_triggered_and_stage_sealed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: DOT rope CUT3R held-out confirmation v1" in text
    assert "branches: [main]" in text
    assert "dot_rope_cut3r_heldout_confirmation_v1.json" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "normal-view images only" in text
    assert "after provider seal" in text
    assert "R11-R70 remain unopened" in text


def test_confirmation_provider_is_read_only_and_bound_to_gpuserver4090() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    provider = text[text.index("\n  provider:") : text.index("\n  evaluate:")]

    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in provider
    assert 'test "$RUNNER_NAME" = "workstation1"' in provider
    assert "environment: trusted-self-hosted-validation" in provider
    assert "permissions:\n      contents: read" in provider
    assert "contents: write" not in provider
    assert "secrets." not in provider
    assert "persist-credentials: false" in provider
    assert "weights_only=False" in provider
    assert "7ed9f6106fb063686990c874ede99876ebc939ab" in provider
    assert "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103" in text


def test_confirmation_marker_evaluation_is_hosted_and_frozen() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    evaluation = text[text.index("\n  evaluate:") : text.index("\n  publish:")]

    assert "runs-on: ubuntu-latest" in evaluation
    assert "actions: read" in evaluation
    assert "R04-R10 marker outcomes" in evaluation
    assert "ca546ff5f22c0279123ccb18509858ee" in text
    assert "md5sum --check --strict" in text
    assert "heldout-strong-positive" in evaluation
    assert "heldout-directional-positive" in evaluation
    assert "heldout-mixed-or-negative" in evaluation
    assert "heldout-support-negative" in evaluation
    assert "technical-failure" in evaluation
    assert "git push" not in text
    assert "secrets." not in text
