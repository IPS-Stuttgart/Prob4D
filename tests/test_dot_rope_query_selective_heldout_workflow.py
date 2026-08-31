from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-rope-query-selective-heldout-v1.yml"
CONTRACT_WORKFLOW = ROOT / ".github/workflows/dot-rope-query-selective-heldout-contract-v1.yml"
REQUEST = "protocols/execution_requests/dot_rope_query_selective_heldout_v1.json"


def _job(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"\n  {name}:")
    end = text.index(f"\n  {next_name}:", start) if next_name else len(text)
    return text[start:end]


def test_pr_contract_is_separate_from_main_only_execution() -> None:
    execution = WORKFLOW.read_text(encoding="utf-8")
    contract = CONTRACT_WORKFLOW.read_text(encoding="utf-8")
    execution_prefix = execution.split("permissions:", maxsplit=1)[0]
    contract_prefix = contract.split("permissions:", maxsplit=1)[0]

    assert "\n  push:" in execution_prefix
    assert "\n  pull_request:" not in execution_prefix
    assert "branches: [main]" in execution_prefix
    assert REQUEST in execution_prefix
    assert "\n  contract:" not in execution

    assert "\n  pull_request:" in contract_prefix
    assert "\n  push:" not in contract_prefix
    assert "branches: [main]" in contract_prefix
    assert WORKFLOW.name in contract_prefix
    assert CONTRACT_WORKFLOW.name in contract_prefix
    assert "pull_request_target:" not in contract
    assert "github.event.pull_request.head.sha" not in contract
    assert "runs-on: ubuntu-latest" in contract
    assert "self-hosted" not in contract

    for text in (execution, contract):
        assert "permissions:\n  contents: read\n" in text
        assert "secrets." not in text
        assert "git push" not in text


def test_prerequisite_is_exact_artifact_bound_and_strong_positive() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    prerequisite = _job(text, "prerequisite", "provider")
    assert "runs-on: ubuntu-latest" in prerequisite
    assert "actions: read" in prerequisite
    assert "/actions/artifacts/{os.environ['ARTIFACT_ID']}" in prerequisite
    assert 'value.get("digest") != os.environ["EXPECTED_DIGEST"]' in prerequisite
    assert "verify_dot_rope_cut3r_heldout_result.py" in prerequisite
    assert 'verified["decision"] != "heldout-strong-positive"' in prerequisite
    assert "prerequisite evaluation identity changed" in prerequisite
    assert "prerequisite marker-support identity changed" in prerequisite


def test_only_provider_job_uses_the_protected_gpu_runner() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    provider = _job(text, "provider", "seal")
    seal = _job(text, "seal", "evaluate")
    evaluate = _job(text, "evaluate", "publish")
    selector = "runs-on: [self-hosted, Linux, X64, gpuserver4090]"
    assert selector in provider
    assert "environment: trusted-self-hosted-validation" in provider
    assert 'test "$RUNNER_NAME" = "workstation1"' in provider
    assert "ref: ${{ needs.authorize.outputs.head_sha }}" in provider
    assert "persist-credentials: false" in provider
    assert "contents: write" not in provider
    assert "issues: write" not in provider
    assert "runs-on: ubuntu-latest" in seal
    assert "runs-on: ubuntu-latest" in evaluate
    assert selector not in seal
    assert selector not in evaluate


def test_archive_and_information_boundaries_are_explicit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "R11-20.zip" in text
    assert "23ce3e7067465d3edabe20b4c7cfa388" in text
    assert "R21-30.zip" in text
    assert "8aee77f79d1aff6e1f3fd21886b251a0" in text
    provider = _job(text, "provider", "seal")
    seal = _job(text, "seal", "evaluate")
    evaluate = _job(text, "evaluate", "publish")
    assert "Predict R11-R30 from normal-view images only" in provider
    assert "two_dimensional_markers_opened" in provider
    assert "three_dimensional_markers_opened" in provider
    assert "Seal factor fits, admission, and predictions without 3-D access" in seal
    assert "three_dimensional_markers_opened" in seal
    assert "Score the already sealed predictions against 3-D outcomes" in evaluate
    assert "--prediction-seal" in evaluate
    assert "R31-R70 remained unopened" in text


def test_external_actions_are_pinned_by_full_commit_sha() -> None:
    for path in (WORKFLOW, CONTRACT_WORKFLOW):
        text = path.read_text(encoding="utf-8")
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
