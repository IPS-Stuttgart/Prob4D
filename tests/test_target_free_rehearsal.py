from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.cli import main as prob4d_main
from prob4d.target_free_rehearsal import (
    TARGET_FREE_REHEARSAL_PROTOCOL_ID,
    run_target_free_rehearsal,
    verify_target_free_rehearsal,
)


def test_target_free_rehearsal_runs_and_replays_every_control(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rehearsal"
    receipt = run_target_free_rehearsal(output, source_revision="a" * 40)

    assert receipt["protocol_id"] == TARGET_FREE_REHEARSAL_PROTOCOL_ID
    assert receipt["positive_control"]["official_loader_status"] == "valid"
    assert receipt["positive_control"]["independent_verifier_status"] == "valid"
    assert receipt["positive_control"]["claim_bearing_loader"]["status"] == (
        "rejected-as-required"
    )
    assert len(receipt["negative_controls"]) == 5
    assert all(
        control["official_rejected"] and control["independent_rejected"]
        for control in receipt["negative_controls"]
    )
    assert receipt["target_access"] == {
        "source_suffix_payloads_opened": 0,
        "target_payloads_opened": 0,
        "target_outcomes_opened": 0,
        "scientific_evidence": False,
    }

    sealed = output / "target_free_rehearsal_receipt.json"
    assert verify_target_free_rehearsal(sealed) == receipt


def test_target_free_rehearsal_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "rehearsal"
    run_target_free_rehearsal(output, source_revision="b" * 40)
    with pytest.raises(FileExistsError):
        run_target_free_rehearsal(output, source_revision="b" * 40)


def test_target_free_rehearsal_rejects_receipt_tampering(tmp_path: Path) -> None:
    output = tmp_path / "rehearsal"
    run_target_free_rehearsal(output, source_revision="c" * 40)
    receipt_path = output / "target_free_rehearsal_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["target_access"]["target_payloads_opened"] = 1
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        verify_target_free_rehearsal(receipt_path)


def test_grouped_cli_runs_and_verifies_the_rehearsal(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "cli-rehearsal"
    assert (
        prob4d_main(
            [
                "diagnostic",
                "target-free-rehearsal",
                "run",
                str(output),
                "--source-revision",
                "d" * 40,
            ]
        )
        == 0
    )
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["target_access"]["target_outcomes_opened"] == 0

    assert (
        prob4d_main(
            [
                "diagnostic",
                "target-free-rehearsal",
                "verify",
                str(output / "target_free_rehearsal_receipt.json"),
            ]
        )
        == 0
    )
    verify_payload = json.loads(capsys.readouterr().out)
    assert verify_payload["receipt_id"] == run_payload["receipt_id"]
