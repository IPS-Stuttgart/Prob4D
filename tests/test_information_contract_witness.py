from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.information_contract_witness as witness
from prob4d.information_contract_witness_smoke import (
    _deterministic_npz,
    generate_witness_smoke,
)


def _rewrite_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    payload = _deterministic_npz(arrays)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_smoke_selects_known_axis_and_confirms_ranking_reversal(tmp_path: Path) -> None:
    result_path = generate_witness_smoke(tmp_path / "smoke")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected = json.loads((result_path.parent / "witness.json").read_text(encoding="utf-8"))

    np.testing.assert_array_equal(selected["query_vector"], [1.0, 0.0, 0.0])
    assert selected["source_max_normalized_error_ratio"] == pytest.approx(10.0)
    assert selected["held_outcomes_opened_during_selection"] is False
    assert result["target_query_reselection"] is False
    assert result["rankings"]["point_accuracy_winner"] == (
        "provider-a-accurate-overconfident"
    )
    assert result["rankings"]["selected_query_calibration_winner"] == (
        "provider-b-less-accurate-calibrated"
    )
    assert result["rankings"]["point_vs_query_calibration_ranking_reversal"] is True
    assert result["information_order"]["prospective_claim_eligible"] is False


def test_generation_and_replay_are_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_witness_smoke(first)
    generate_witness_smoke(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    replay = witness.evaluate_frozen_witness(first / "held.json", first / "witness.json")
    assert witness._canonical_bytes(replay) == (first / "result.json").read_bytes()


def test_equal_group_selection_is_invariant_to_nested_case_duplication() -> None:
    residual = np.array(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    covariance = np.repeat(np.eye(2)[None], 4, axis=0)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    empirical, reported = witness._equal_group_moments(residual, covariance, groups, 2)

    duplicate = np.concatenate((residual[:2], residual[:2], residual[2:]))
    duplicate_covariance = np.repeat(np.eye(2)[None], len(duplicate), axis=0)
    duplicate_groups = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    empirical_duplicate, reported_duplicate = witness._equal_group_moments(
        duplicate, duplicate_covariance, duplicate_groups, 2
    )

    np.testing.assert_array_equal(empirical, empirical_duplicate)
    np.testing.assert_array_equal(reported, reported_duplicate)


def test_registered_query_basis_restricts_the_search(tmp_path: Path) -> None:
    root = tmp_path / "restricted"
    generate_witness_smoke(root)
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    payload_path = root / source["payload"]
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["query_basis"] = np.array([[0.0], [1.0], [0.0]])
    source["payload_sha256"] = _rewrite_npz(payload_path, arrays)
    (root / "source.json").write_bytes(witness._canonical_bytes(source))

    selected = witness.select_falsification_witness(root / "source.json")
    np.testing.assert_array_equal(selected["query_vector"], [0.0, 1.0, 0.0])
    assert selected["source_max_normalized_error_ratio"] == pytest.approx(1.0)


def test_source_payload_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "tamper"
    generate_witness_smoke(root)
    path = root / "source-provider-a.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="source payload SHA-256 mismatch"):
        witness.select_falsification_witness(root / "source.json")


def test_held_manifest_cannot_switch_witness(tmp_path: Path) -> None:
    root = tmp_path / "mismatch"
    generate_witness_smoke(root)
    held = json.loads((root / "held.json").read_text(encoding="utf-8"))
    held["source_witness_id"] = "0" * 64
    (root / "held.json").write_bytes(witness._canonical_bytes(held))
    with pytest.raises(ValueError, match="not bound to the supplied witness"):
        witness.evaluate_frozen_witness(root / "held.json", root / "witness.json")


def test_nonpositive_covariance_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "covariance"
    generate_witness_smoke(root)
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    payload_path = root / source["payload"]
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["reported_covariance"][0, 0, 0] = -1.0
    source["payload_sha256"] = _rewrite_npz(payload_path, arrays)
    (root / "source.json").write_bytes(witness._canonical_bytes(source))
    with pytest.raises(ValueError, match="positive definite"):
        witness.select_falsification_witness(root / "source.json")


def test_witness_content_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "witness-tamper"
    generate_witness_smoke(root)
    value = json.loads((root / "witness.json").read_text(encoding="utf-8"))
    value["query_vector"][0] = 0.5
    (root / "witness.json").write_bytes(witness._canonical_bytes(value))
    with pytest.raises(ValueError, match="content ID"):
        witness.evaluate_frozen_witness(root / "held.json", root / "witness.json")


def test_cli_refuses_to_clobber_outputs(tmp_path: Path) -> None:
    root = tmp_path / "cli"
    assert witness.main(["smoke", str(root)]) == 0
    with pytest.raises(FileExistsError):
        witness.main(["smoke", str(root)])
    replay = tmp_path / "replay.json"
    assert witness.main(
        ["evaluate", str(root / "held.json"), str(root / "witness.json"), str(replay)]
    ) == 0
    with pytest.raises(FileExistsError):
        witness.main(
            ["evaluate", str(root / "held.json"), str(root / "witness.json"), str(replay)]
        )
