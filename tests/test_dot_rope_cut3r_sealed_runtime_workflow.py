from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "dot-rope-cut3r-sealed-runtime-v1.yml"


def _job(text: str, name: str, following: str | None = None) -> str:
    start = text.index(f"\n  {name}:")
    if following is None:
        return text[start:]
    return text[start : text.index(f"\n  {following}:", start)]


def test_workflow_is_request_bound_and_source_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "\n  pull_request:" in text
    assert "\n  push:" in text
    assert "branches: [main]" in text
    assert "protocols/execution_requests/dot_rope_cut3r_sealed_runtime_v1.json" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert "Expected only %s; changed paths were:" in text
    assert "validate-request" in text
    assert "R01-R03" in text
    assert "R04-R70 remained unopened" in text
    assert "secrets." not in text
    assert "git push" not in text


def test_provider_binds_sealed_cuda_runtime_without_rebuilding() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    provider = _job(text, "provider", "evaluate")

    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in provider
    assert 'test "$RUNNER_NAME" = "workstation1"' in provider
    assert "CUT3R_RUNTIME_ROOT: /home/github-runner/.cache/prob4d/cut3r-runtime-v1" in text
    assert (
        "CUT3R_RUNTIME_PYTHON: "
        "/home/github-runner/.cache/prob4d/cut3r-runtime-v1/venv/bin/python" in text
    )
    assert (
        "CUT3R_RUNTIME_CHECKPOINT: "
        "/home/github-runner/.cache/prob4d/cut3r-runtime-v1/"
        "cut3r_512_dpt_4_64.pth" in text
    )
    assert "CUT3R_REVISION: 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf" in text
    assert (
        "CUT3R_CHECKPOINT_SHA256: "
        "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103" in text
    )
    assert "from dust3r.model import ARCroco3DStereo" in provider
    assert "torch_cuda_version" in provider
    assert '"12.6"' in provider
    assert "dot_dataset_accessed" in provider
    assert "--expected-artifact-id" in provider
    assert "--build" not in provider
    assert "vars.CUT3R_CHECKOUT" not in text
    assert "vars.CUT3R_PYTHON" not in text
    assert "vars.CUT3R_CHECKPOINT" not in text


def test_prediction_and_marker_access_are_information_separated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    provider = _job(text, "provider", "evaluate")
    evaluate = _job(text, "evaluate", "summarize")

    assert "runtime-smoke" in provider
    assert "Execute marker-free normal-view predictions" in provider
    assert "\n              predict" in provider
    assert "\n              evaluate" not in provider
    assert "Download exact sealed provider bundle" in evaluate
    assert "Open source markers only after prediction seal and evaluate" in evaluate
    assert "\n              evaluate" in evaluate
    assert "\n              predict" not in evaluate
    assert "needs.provider.outputs.decision == 'sealed-provider-predictions'" in evaluate
    assert "R01-10.zip" in provider
    assert "R01-10.zip" in evaluate


def test_self_hosted_jobs_are_read_only_and_artifacts_are_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    provider = _job(text, "provider", "evaluate")
    evaluate = _job(text, "evaluate", "summarize")
    summarize = _job(text, "summarize")

    for job in (provider, evaluate):
        assert "permissions:\n      contents: read" in job
        assert "contents: write" not in job
        assert "issues: write" not in job
        assert "persist-credentials: false" in job

    assert "runs-on: ubuntu-latest" in summarize
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in text
    assert "if-no-files-found: error" in text
