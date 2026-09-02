from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.information_contract_sealed as sealed


def _rewrite_payload(
    manifest_path: Path,
    case_index: int,
    mutate,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][case_index]
    payload = manifest_path.parent / case["payload"]
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    mutate(arrays)
    payload.write_bytes(sealed._deterministic_npz_bytes(arrays))
    case["payload_sha256"] = sealed._sha256_file(payload)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def test_sealed_smoke_is_deterministic_and_truth_separated(tmp_path: Path) -> None:
    challenge, submission = sealed.generate_sealed_smoke(tmp_path / "sealed")
    first = sealed.evaluate_sealed_information_contract(challenge, submission)
    second = sealed.evaluate_sealed_information_contract(challenge, submission)

    assert first == second
    assert first["schema_name"] == sealed.SEALED_RESULT_SCHEMA
    assert first["information_order"] == {
        "mode": "retrospective-open-target",
        "claim_class": "retrospective-diagnostic",
        "prospective_claim_eligible": False,
    }
    assert first["aggregate"]["contract"]["all_cases_pass"] is True
    assert first["aggregate"]["finite_query"][
        "local_admits_finite_rejects_count"
    ] == 2
    assert first["aggregate"]["finite_query"]["equal_group_mean"][
        "finite_query_false_accept_fraction"
    ] == 0.0

    challenge_manifest = json.loads(challenge.read_text(encoding="utf-8"))
    submission_manifest = json.loads(submission.read_text(encoding="utf-8"))
    for case in challenge_manifest["cases"]:
        with np.load(challenge.parent / case["payload"], allow_pickle=False) as archive:
            names = set(archive.files)
        assert names.issubset(sealed._CHALLENGE_ARRAYS)
        assert not names.intersection(sealed._SUBMISSION_ARRAYS)
    for case in submission_manifest["cases"]:
        with np.load(submission.parent / case["payload"], allow_pickle=False) as archive:
            names = set(archive.files)
        assert names.issubset(sealed._SUBMISSION_ARRAYS)
        assert not names.intersection(sealed._CHALLENGE_ARRAYS)

    for case in first["cases"]:
        communication = case["metrics"]["communication"]
        assert communication["payload_file_bytes"] == case["payloads"]["submission"][
            "size_bytes"
        ]
        assert "merged_evaluation_payload_file_bytes" in communication


def test_submission_cannot_smuggle_target_truth(tmp_path: Path) -> None:
    challenge, submission = sealed.generate_sealed_smoke(tmp_path / "sealed")
    challenge_manifest = json.loads(challenge.read_text(encoding="utf-8"))
    source = challenge.parent / challenge_manifest["cases"][0]["payload"]
    with np.load(source, allow_pickle=False) as archive:
        truth = np.array(archive["truth_xyz_m"], copy=True)

    def mutate(arrays):
        arrays["truth_xyz_m"] = truth

    _rewrite_payload(submission, 0, mutate)
    with pytest.raises(ValueError, match="owned by the other side"):
        sealed.evaluate_sealed_information_contract(challenge, submission)


def test_challenge_cannot_smuggle_provider_prediction(tmp_path: Path) -> None:
    challenge, submission = sealed.generate_sealed_smoke(tmp_path / "sealed")
    submission_manifest = json.loads(submission.read_text(encoding="utf-8"))
    source = submission.parent / submission_manifest["cases"][0]["payload"]
    with np.load(source, allow_pickle=False) as archive:
        prediction = np.array(archive["prediction_mean_xyz_m"], copy=True)

    def mutate(arrays):
        arrays["prediction_mean_xyz_m"] = prediction

    _rewrite_payload(challenge, 0, mutate)
    with pytest.raises(ValueError, match="owned by the other side"):
        sealed.evaluate_sealed_information_contract(challenge, submission)


def test_finite_query_false_acceptance_fails_contract(tmp_path: Path) -> None:
    challenge, submission = sealed.generate_sealed_smoke(tmp_path / "sealed")

    def mutate(arrays):
        arrays["finite_query_admitted"] = np.ones(3, dtype=np.bool_)

    _rewrite_payload(submission, 0, mutate)
    result = sealed.evaluate_sealed_information_contract(challenge, submission)
    case = result["cases"][0]
    assert case["metrics"]["finite_query"]["false_accept_count"] == 2
    assert case["contract_checks"]["finite_query_admission_consistent"] is False
    assert case["contract_pass"] is False
    assert result["aggregate"]["contract"]["all_cases_pass"] is False


def test_submission_roster_must_exactly_match_challenge(tmp_path: Path) -> None:
    challenge, submission = sealed.generate_sealed_smoke(tmp_path / "sealed")
    manifest = json.loads(submission.read_text(encoding="utf-8"))
    manifest["cases"].pop()
    submission.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="roster must exactly match"):
        sealed.evaluate_sealed_information_contract(challenge, submission)


def test_prospective_label_requires_compatible_seal_declaration(tmp_path: Path) -> None:
    challenge, submission = sealed.generate_sealed_smoke(tmp_path / "sealed")
    challenge_manifest = json.loads(challenge.read_text(encoding="utf-8"))
    challenge_manifest["dataset"]["information_order"] = "prospective-sealed-target"
    challenge.write_text(
        json.dumps(challenge_manifest, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="submission_mode"):
        sealed.evaluate_sealed_information_contract(challenge, submission)

    submission_manifest = json.loads(submission.read_text(encoding="utf-8"))
    submission_manifest["producer"]["submission_mode"] = "prospective-sealed"
    submission.write_text(
        json.dumps(submission_manifest, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires prediction_sealed_before_truth"):
        sealed.evaluate_sealed_information_contract(challenge, submission)


def test_cli_generation_and_replay_are_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "sealed"
    assert sealed.main(["smoke", str(root)]) == 0
    replay = tmp_path / "replay.json"
    assert sealed.main(
        [
            "evaluate",
            str(root / "challenge" / "challenge.json"),
            str(root / "submission" / "submission.json"),
            str(replay),
        ]
    ) == 0
    assert replay.read_bytes() == (root / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        sealed.main(["smoke", str(root)])
