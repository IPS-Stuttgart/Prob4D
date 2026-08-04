from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "bpt-controlled-decisive-v1.yml"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_decisive_execution_uses_exact_revisions_hashes_and_runner() -> None:
    text = _workflow_text()

    identities = {
        "PROB4D_SOURCE_REVISION": "aa8ffc6541011d044561e09870569a14ab3f586f",
        "BPT_SOURCE_REVISION": "76d4aba20dd386e1f8583e501781d702d7937566",
        "BPT_BASE_REVISION": "b2da5df5eddd5437d444b60b11130262d115e264",
        "PROTOCOL_SHA256": (
            "921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111"
        ),
        "RUNNER_PAYLOAD_SHA256": (
            "16beddf036d797ad16868a4b45596b11b2f9617ac6f39f609b5a1b9ce6de3a63"
        ),
        "VERIFIER_PAYLOAD_SHA256": (
            "1b07e9b9c0b3f31c1700d1e4f97ae43467ce01a05b9e04108b4b2c434efb5eda"
        ),
        "VERIFIER_PART00_SHA256": (
            "02fa5c69ef566444f91bad415147337ec929a5ed26b980309534ecef1733e937"
        ),
        "VERIFIER_PART01_SHA256": (
            "e37cf46fefe37c6e946775b99b08ab4b2cdc68442f35e27bcd74500fe3d216fc"
        ),
    }
    for name, value in identities.items():
        assert f"{name}: {value}" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text


def test_decisive_execution_is_read_only_and_fail_closed() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "\n  push:" not in text
    assert "${{ secrets." not in text
    assert "Exact private BayesianPhysTwin source is unavailable" in text
    assert "git -C \"${dst}\" cat-file -e" in text
    assert "merge-base --is-ancestor" in text
    assert "status --porcelain=v1" in text
    assert "base64.b64decode(runner_encoded, validate=True)" in text
    assert "sha256sum --check SHA256SUMS" in text


def test_decisive_execution_repairs_only_registered_transport_defect() -> None:
    text = _workflow_text()

    assert "repair_index = 7570" in text
    assert 'verifier_encoded[repair_index] != "s"' in text
    assert "delete_one_extraneous_base64_character" in text
    assert "frozen verifier transport bytes changed" in text
    assert "materialized-independent-verifier.py" in text
    assert "VERIFIER_PAYLOAD_SHA256" in text


def test_decisive_execution_accepts_only_registered_outcomes() -> None:
    text = _workflow_text()

    assert '"${status}" -ne 0 && "${status}" -ne 3' in text
    assert "calibration_trials.csv" in text
    assert 'test -s "${EVIDENCE_ROOT}/result/${required}"' in text
    assert "materialized-independent-verifier.py" in text
    assert "retention-days: 90" in text


def test_decisive_execution_resolves_exact_runner_local_bpt_source() -> None:
    text = _workflow_text()

    assert "${RUNNER_WORKSPACE}/../BayesianPhysTwin/BayesianPhysTwin" in text
    assert "resolution=runner_local_shared_object_store" in text
    assert "git clone --shared --no-checkout" in text
    assert "runner_ssh_identity" in text


def test_decisive_execution_uses_immutable_action_pins() -> None:
    text = _workflow_text()

    assert (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    ) in text
    assert (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
    ) in text
    assert (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    ) in text
