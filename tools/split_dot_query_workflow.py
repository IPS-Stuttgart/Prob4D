#!/usr/bin/env python3
"""Stage a security-preserving split of the DOT query-selective workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

BRANCH = "science/dot-query-selective-r11-r30-v1"
EXPECTED_PARENT = "1b4b52ec5ebfc864870dca469046edad1a42ecaf"
EXECUTION = Path(".github/workflows/dot-rope-query-selective-heldout-v1.yml")
TEST = Path("tests/test_dot_rope_query_selective_heldout_workflow.py")
EXPECTED_EXECUTION_BLOB = "804645882535703a58f76362171e72c09fb053de"
EXPECTED_TEST_BLOB = "6a847fbed73b15c757305afd6ec2674f813f48dc"
STAGED_CONTRACT = Path("tools/dot-rope-query-selective-heldout-contract-v1.yml.payload")
STAGED_EXECUTION = Path("tools/dot-rope-query-selective-heldout-v1.yml.payload")
STAGED_TEST = Path("tools/test_dot_rope_query_selective_heldout_workflow.py.payload")


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    require(os.environ.get("GITHUB_REPOSITORY") == "IPS-Stuttgart/Prob4D", "wrong repository")
    require(os.environ.get("GITHUB_REF_NAME") == BRANCH, "wrong branch")
    require(os.environ.get("GITHUB_ACTOR") == "FlorianPfaff", "wrong actor")
    require(output("git", "rev-parse", "HEAD^") == EXPECTED_PARENT, "unexpected parent")
    require(output("git", "hash-object", str(EXECUTION)) == EXPECTED_EXECUTION_BLOB, "execution workflow drift")
    require(output("git", "hash-object", str(TEST)) == EXPECTED_TEST_BLOB, "workflow test drift")

    text = EXECUTION.read_text(encoding="utf-8")
    permissions_at = text.index("\npermissions:\n")
    jobs_at = text.index("\njobs:\n")
    contract_at = text.index("\n  contract:\n", jobs_at)
    authorize_at = text.index("\n  authorize:\n", contract_at)
    common = text[permissions_at + 1 : jobs_at + len("\njobs:\n")]
    contract_block = text[contract_at + 1 : authorize_at]
    contract_block = contract_block.replace(
        "    if: github.event_name == 'pull_request'\n", ""
    )
    contract_block = contract_block.replace(
        "          ref: ${{ github.event.pull_request.head.sha }}\n", ""
    )
    require("github.event.pull_request.head.sha" not in contract_block, "untrusted PR head ref remains")

    request_path = "protocols/execution_requests/dot_rope_query_selective_heldout_v1.json"
    execution = (
        "name: DOT rope query-selective held-out CUT3R v1\n\n"
        "on:\n"
        "  push:\n"
        "    branches: [main]\n"
        "    paths:\n"
        f"      - \"{request_path}\"\n\n"
        + common
        + text[authorize_at + 1 :]
    )
    require("pull_request:" not in execution.split("permissions:", 1)[0], "execution remains PR-triggered")
    require("\n  contract:\n" not in execution, "contract job remains in execution")

    contract = (
        "name: DOT rope query-selective held-out contract v1\n\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [main]\n"
        "    paths:\n"
        "      - \".github/workflows/dot-rope-query-selective-heldout-contract-v1.yml\"\n"
        "      - \".github/workflows/dot-rope-query-selective-heldout-v1.yml\"\n"
        "      - \"protocols/dot-rope-query-selective-heldout-v1.json\"\n"
        "      - \"scripts/science/run_dot_rope_query_selective_heldout.py\"\n"
        "      - \"tests/test_dot_rope_query_selective_heldout.py\"\n"
        "      - \"tests/test_dot_rope_query_selective_heldout_workflow.py\"\n"
        "      - \"tests/test_trusted_self_hosted_validation_policy.py\"\n"
        "      - \"tests/_trusted_self_hosted_validation_policy_base.py\"\n"
        "      - \"docs/dot-rope-query-selective-heldout-v1.md\"\n\n"
        "permissions:\n"
        "  contents: read\n\n"
        "concurrency:\n"
        "  group: dot-rope-query-selective-heldout-contract-v1-${{ github.ref }}-${{ github.sha }}\n"
        "  cancel-in-progress: false\n\n"
        "jobs:\n"
        + contract_block
    )
    require("\n  push:" not in contract.split("permissions:", 1)[0], "contract remains push-triggered")
    require("runs-on: ubuntu-latest" in contract, "hosted contract runner missing")
    require("self-hosted" not in contract, "self-hosted runner leaked into PR workflow")

    test = '''from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/dot-rope-query-selective-heldout-v1.yml"
CONTRACT_WORKFLOW = (
    ROOT / ".github/workflows/dot-rope-query-selective-heldout-contract-v1.yml"
)
REQUEST = "protocols/execution_requests/dot_rope_query_selective_heldout_v1.json"


def _job(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"\\n  {name}:")
    end = text.index(f"\\n  {next_name}:", start) if next_name else len(text)
    return text[start:end]


def test_pr_contract_is_separate_from_main_only_execution() -> None:
    execution = WORKFLOW.read_text(encoding="utf-8")
    contract = CONTRACT_WORKFLOW.read_text(encoding="utf-8")
    execution_prefix = execution.split("permissions:", maxsplit=1)[0]
    contract_prefix = contract.split("permissions:", maxsplit=1)[0]

    assert "\\n  push:" in execution_prefix
    assert "\\n  pull_request:" not in execution_prefix
    assert "branches: [main]" in execution_prefix
    assert REQUEST in execution_prefix
    assert "\\n  contract:" not in execution

    assert "\\n  pull_request:" in contract_prefix
    assert "\\n  push:" not in contract_prefix
    assert "branches: [main]" in contract_prefix
    assert WORKFLOW.name in contract_prefix
    assert CONTRACT_WORKFLOW.name in contract_prefix
    assert "pull_request_target:" not in contract
    assert "github.event.pull_request.head.sha" not in contract
    assert "runs-on: ubuntu-latest" in contract
    assert "self-hosted" not in contract

    for text in (execution, contract):
        assert "permissions:\\n  contents: read\\n" in text
        assert "secrets." not in text
        assert "git push" not in text


def test_prerequisite_is_exact_artifact_bound_and_strong_positive() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    prerequisite = _job(text, "prerequisite", "provider")
    assert "runs-on: ubuntu-latest" in prerequisite
    assert "actions: read" in prerequisite
    assert "/actions/artifacts/{os.environ['ARTIFACT_ID']}" in prerequisite
    assert "value.get(\\\"digest\\\") != os.environ[\\\"EXPECTED_DIGEST\\\"]" in prerequisite
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
'''

    for path in (STAGED_CONTRACT, STAGED_EXECUTION, STAGED_TEST):
        path.parent.mkdir(parents=True, exist_ok=True)
    STAGED_CONTRACT.write_text(contract, encoding="utf-8", newline="\n")
    STAGED_EXECUTION.write_text(execution, encoding="utf-8", newline="\n")
    STAGED_TEST.write_text(test, encoding="utf-8", newline="\n")

    subprocess.check_call(["git", "diff", "--check"])
    subprocess.check_call(["git", "config", "user.name", "github-actions[bot]"])
    subprocess.check_call([
        "git", "config", "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    ])
    subprocess.check_call(["git", "add", str(STAGED_CONTRACT), str(STAGED_EXECUTION), str(STAGED_TEST)])
    subprocess.check_call(["git", "commit", "-m", "Stage secure DOT workflow split [skip ci]"])
    subprocess.check_call(["git", "push", "origin", f"HEAD:{BRANCH}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
