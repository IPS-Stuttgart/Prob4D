from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "dot-rope-cut3r-heldout-confirmation-v1.yml"


def test_confirmation_workflow_is_single_request_triggered_and_stage_sealed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: DOT rope CUT3R held-out confirmation v1" in text
    assert "branches: [main]" in text
    assert "dot_rope_cut3r_heldout_confirmation_gpuserver6000_v1.json" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "Prove the old gpuserver4090 run never opened confirmation data" in text
    assert 'ORIGINAL_RUN_ID: "33363832286"' in text
    assert "old confirmation provider may have started; recovery is forbidden" in text
    assert "normal-view images only" in text
    assert "after provider seal" in text
    assert "R11-R70 remained unopened" in text


def test_confirmation_provider_is_read_only_and_bound_to_gpuserver6000() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    provider = text[text.index("\n  provider:") : text.index("\n  evaluate:")]

    assert "runs-on: [self-hosted, gpuserver6000]" in provider
    assert 'test "$RUNNER_NAME" = "workstation2"' in provider
    assert "gpuserver4090]" not in provider
    assert "environment: trusted-self-hosted-validation" in provider
    assert "permissions:\n      contents: read" in provider
    assert "contents: write" not in provider
    assert "secrets." not in provider
    assert "persist-credentials: false" in provider
    assert "weights_only=False" in provider
    assert "7ed9f6106fb063686990c874ede99876ebc939ab" in provider
    assert "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103" in text
    assert "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf" in text
    assert "CUT3R_CHECKOUT" in provider
    assert "CUT3R_CHECKPOINT" in provider
    assert "CUT3R_PYTHON" in provider
    assert "prepare_cut3r_runtime.py" in provider
    assert "native RoPE requires an available nvcc compiler" in provider
    assert "provider-seal.json" in provider
    assert provider.index("Predict from marker-free normal-view images only") < provider.index(
        "Upload immutable provider seal before marker evaluation"
    )


def test_confirmation_marker_evaluation_is_hosted_and_frozen() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    evaluation = text[text.index("\n  evaluate:") : text.index("\n  publish:")]

    assert "runs-on: ubuntu-latest" in evaluation
    assert "actions: read" in evaluation
    verify_index = evaluation.index("Verify provider seal before downloading marker archive")
    download_index = evaluation.index("Download and verify official R01-R10 marker archive")
    assert verify_index < download_index
    assert "Evaluate the frozen alpha and comparators" in evaluation
    assert "ca546ff5f22c0279123ccb18509858ee" in text
    assert "md5sum --check --strict" in text
    assert "heldout-strong-positive" in evaluation
    assert "heldout-directional-positive" in evaluation
    assert "heldout-mixed-or-negative" in evaluation
    assert "heldout-support-negative" in evaluation
    assert "technical-failure" in evaluation
    assert "verify_dot_rope_cut3r_heldout_result.py" in evaluation
    assert "git push" not in text
    assert "secrets." not in text
