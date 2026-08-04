from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "bpt-controlled-decisive-v1.yml"
)


def test_decisive_execution_uses_exact_revisions_and_runner_labels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "PROB4D_SOURCE_REVISION: "
        "aa8ffc6541011d044561e09870569a14ab3f586f"
    ) in text
    assert (
        "BPT_SOURCE_REVISION: "
        "59256a124c4df1d780b79d1c31d6c1d01e63d10f"
    ) in text
    assert (
        "BPT_BASE_REVISION: "
        "b2da5df5eddd5437d444b60b11130262d115e264"
    ) in text
    assert (
        "PROTOCOL_SHA256: "
        "921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111"
    ) in text
    assert (
        "RUNNER_PAYLOAD_SHA256: "
        "83af55e5744531110df5744031ab30b570bb5a0b9aa0bbb246961db783e166f5"
    ) in text
    assert (
        "VERIFIER_PAYLOAD_SHA256: "
        "4a206f1bf15b85e47cbe1f13c3095d7531b17c8d32c9ddb68f26ca0d099778ad"
    ) in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text


def test_decisive_execution_is_read_only_and_fail_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "\n  push:" not in text
    assert "${{ secrets." not in text
    assert "Exact private BayesianPhysTwin source is unavailable" in text
    assert "git -C \"${dst}\" cat-file -e" in text
    assert "merge-base --is-ancestor" in text
    assert "status --porcelain=v1" in text
    assert "sha256sum --check SHA256SUMS" in text


def test_decisive_execution_accepts_only_registered_outcomes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"${status}" -ne 0 && "${status}" -ne 3' in text
    assert "verify_prob4d_bpt_controlled_decisive_v1.py" in text
    assert "test_prob4d_bpt_controlled_decisive_verifier_v1.py" in text
    assert 'test -s "${EVIDENCE_ROOT}/result/calibration_trials.csv"' in text
    assert "retention-days: 90" in text


def test_decisive_execution_resolves_staged_exact_bpt_source() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "${RUNNER_WORKSPACE}/../BayesianPhysTwin/BayesianPhysTwin/"
        "staged-bpt-source"
    ) in text
    assert "resolution=runner_local_shared_object_store" in text
    assert "git clone --shared --no-checkout" in text
    assert "runner_ssh_identity" in text


def test_decisive_execution_uses_immutable_action_pins() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    ) in text
    assert (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
    ) in text
    assert (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    ) in text
