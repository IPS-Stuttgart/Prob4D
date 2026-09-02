from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.information_contract_sealed import (
    evaluate_sealed_information_contract,
    generate_sealed_smoke,
)
from prob4d.information_contract_sealed_smoke import _deterministic_npz_bytes


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_case_payload(
    manifest_path: Path,
    case_index: int,
    mutate,
) -> Path:
    manifest = _load(manifest_path)
    case = manifest["cases"][case_index]
    payload = manifest_path.parent / case["payload"]
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    mutate(arrays)
    payload.write_bytes(_deterministic_npz_bytes(arrays))
    case["payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    _write(manifest_path, manifest)
    return payload


def test_sealed_smoke_separates_truth_and_submission(tmp_path: Path) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")
    first = evaluate_sealed_information_contract(challenge, submission)
    second = evaluate_sealed_information_contract(challenge, submission)

    assert first == second
    assert first["information_order"] == {
        "mode": "retrospective-open-target",
        "claim_class": "retrospective-diagnostic",
        "prospective_claim_eligible": False,
    }
    assert first["aggregate"]["contract"]["all_cases_pass"] is True
    assert first["aggregate"]["finite_query"]["status"] == "evaluated"
    assert (
        first["aggregate"]["finite_query"][
            "local_admits_finite_rejects_count"
        ]
        == 2
    )
    assert (
        first["aggregate"]["finite_query"]["equal_group_mean"][
            "finite_query_false_accept_fraction"
        ]
        == 0.0
    )

    challenge_manifest = _load(challenge)
    submission_manifest = _load(submission)
    for challenge_case, submission_case in zip(
        challenge_manifest["cases"],
        submission_manifest["cases"],
        strict=True,
    ):
        with np.load(
            challenge.parent / challenge_case["payload"],
            allow_pickle=False,
        ) as archive:
            assert "truth_xyz_m" in archive.files
            assert "prediction_mean_xyz_m" not in archive.files
            assert "selected_action" not in archive.files
        with np.load(
            submission.parent / submission_case["payload"],
            allow_pickle=False,
        ) as archive:
            assert "truth_xyz_m" not in archive.files
            assert "realized_action_loss" not in archive.files
            assert "prediction_mean_xyz_m" in archive.files
            assert "selected_action" in archive.files

    for case in first["cases"]:
        communication = case["metrics"]["communication"]
        assert communication["payload_file_bytes"] == case["payloads"][
            "submission"
        ]["size_bytes"]
        assert (
            communication["merged_evaluation_payload_file_bytes"]
            > communication["payload_file_bytes"]
        )


def test_submission_cannot_smuggle_target_truth(tmp_path: Path) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")

    def mutate(arrays):
        arrays["truth_xyz_m"] = np.zeros((3, 3), dtype=np.float64)

    _rewrite_case_payload(submission, 0, mutate)
    with pytest.raises(ValueError, match="owned by the other side"):
        evaluate_sealed_information_contract(challenge, submission)


def test_challenge_cannot_smuggle_provider_prediction(tmp_path: Path) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")

    def mutate(arrays):
        arrays["prediction_mean_xyz_m"] = np.zeros((3, 3), dtype=np.float64)

    _rewrite_case_payload(challenge, 0, mutate)
    with pytest.raises(ValueError, match="owned by the other side"):
        evaluate_sealed_information_contract(challenge, submission)


def test_finite_query_false_accept_fails_contract(tmp_path: Path) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")

    def mutate(arrays):
        admitted = np.asarray(arrays["finite_query_admitted"]).copy()
        admitted[0] = True
        arrays["finite_query_admitted"] = admitted

    _rewrite_case_payload(submission, 0, mutate)
    result = evaluate_sealed_information_contract(challenge, submission)
    case = result["cases"][0]
    assert case["metrics"]["finite_query"]["false_accept_count"] == 1
    assert (
        case["contract_checks"]["finite_query_admission_consistent"]
        is False
    )
    assert case["contract_pass"] is False
    assert result["aggregate"]["contract"]["all_cases_pass"] is False


def test_submission_roster_must_exactly_match_challenge(tmp_path: Path) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")
    manifest = _load(submission)
    manifest["cases"].pop()
    _write(submission, manifest)

    with pytest.raises(ValueError, match="case roster must exactly match"):
        evaluate_sealed_information_contract(challenge, submission)


def test_prospective_label_requires_sealed_information_order(
    tmp_path: Path,
) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")
    challenge_manifest = _load(challenge)
    challenge_manifest["dataset"]["information_order"] = (
        "prospective-sealed-target"
    )
    _write(challenge, challenge_manifest)

    with pytest.raises(ValueError, match="incompatible"):
        evaluate_sealed_information_contract(challenge, submission)

    submission_manifest = _load(submission)
    producer = submission_manifest["producer"]
    producer["submission_mode"] = "prospective-sealed"
    producer["prediction_sealed_before_truth"] = True
    _write(submission, submission_manifest)

    result = evaluate_sealed_information_contract(challenge, submission)
    assert result["information_order"]["prospective_claim_eligible"] is True
    assert result["information_order"]["claim_class"] == "prospective-heldout"


def test_payload_arrays_must_exactly_match_declared_tasks(
    tmp_path: Path,
) -> None:
    challenge, submission = generate_sealed_smoke(tmp_path / "sealed")

    def mutate(arrays):
        arrays.pop("reported_query_variance")

    _rewrite_case_payload(submission, 0, mutate)
    with pytest.raises(ValueError, match="does not match declared tasks"):
        evaluate_sealed_information_contract(challenge, submission)
