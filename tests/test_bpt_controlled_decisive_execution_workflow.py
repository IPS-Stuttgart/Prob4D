from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "bpt-controlled-decisive-v1.yml"


def test_decisive_execution_uses_exact_revisions_and_runner_labels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "PROB4D_SOURCE_REVISION: "
        "aa8ffc6541011d044561e09870569a14ab3f586f"
    ) in text
    assert (
        "BPT_SOURCE_REVISION: "
        "db0f0119a3a4220f5489566829846681e844627d"
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
        "REFERENCE_REPORT_ID: "
        "c592807d62e9f5121acf85747432574601264160de67b15e9a1c8e48a12cc040"
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
    assert "runner_local_complete_clone" in text
    assert "git clone --no-local --no-checkout" in text


def test_decisive_execution_uses_complete_source_not_encoded_payloads() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "RUNNER_PAYLOAD_SHA256" not in text
    assert "VERIFIER_PAYLOAD_SHA256" not in text
    assert "base64.b64decode" not in text
    assert "zlib.decompress" not in text
    assert "prob4d_bpt_controlled_decisive_core_v1.py" in text
    assert "run_prob4d_bpt_controlled_decisive_v1.py" in text
    assert "test_prob4d_bpt_controlled_decisive_v1.py" in text


def test_decisive_execution_accepts_only_registered_outcomes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"${status}" -ne 0 && "${status}" -ne 3' in text
    assert "registered decision criteria do not recompute" in text
    assert "overall decision does not match registered criteria" in text
    assert "all_rejections_exact_fallback" in text
    assert "fresh target rows differ from retained deterministic evidence" in text
    assert "retention-days: 90" in text


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
