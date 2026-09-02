from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import prob4d.information_contract_benchmark as benchmark
from prob4d.information_contract_benchmark_smoke import _deterministic_npz_bytes


def _rewrite_payload(
    suite_path: Path,
    case_index: int,
    mutate,
) -> Path:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    case = suite["cases"][case_index]
    payload = suite_path.parent / case["payload"]
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(arrays)
    payload.write_bytes(_deterministic_npz_bytes(arrays))
    case["payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def test_smoke_suite_covers_all_axes_and_is_deterministic(tmp_path: Path) -> None:
    suite = benchmark.generate_smoke_suite(tmp_path / "suite")
    first = benchmark.evaluate_information_contract_suite(suite)
    second = benchmark.evaluate_information_contract_suite(suite)

    assert first == second
    assert first["schema_name"] == benchmark.RESULT_SCHEMA
    assert first["aggregate"]["case_count"] == 2
    assert first["aggregate"]["independent_group_count"] == 2
    assert first["aggregate"]["contract"]["all_cases_pass"] is True
    assert set(first["aggregate"]["declared_task_case_counts"]) == {
        "forecast",
        "calibration",
        "dependence",
        "query",
        "gauge",
        "fallback",
        "decision",
        "communication",
    }
    assert (
        first["aggregate"]["equal_group_mean"]["dependence_nll_gain_per_dimension"]
        > 0.0
    )
    assert first["aggregate"]["equal_group_mean"]["gauge_false_accept_fraction"] == 0.0
    assert first["aggregate"]["equal_group_mean"]["gauge_false_reject_fraction"] == 0.0
    assert first["aggregate"]["equal_group_mean"]["exact_fallback_fraction"] == 1.0

    by_id = {case["case_id"]: case for case in first["cases"]}
    assert by_id["admissible-shared-dependence"]["metrics"]["decision"][
        "expected_admitted"
    ] is True
    assert by_id["ambiguous-exact-fallback"]["metrics"]["decision"][
        "expected_admitted"
    ] is False
    assert all(case["contract_pass"] for case in first["cases"])


def test_payload_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    suite = benchmark.generate_smoke_suite(tmp_path / "suite")
    payload = suite.parent / "admissible-shared-dependence.npz"
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        benchmark.evaluate_information_contract_suite(suite)


def test_false_gauge_acceptance_is_reported_as_contract_failure(
    tmp_path: Path,
) -> None:
    suite = benchmark.generate_smoke_suite(tmp_path / "suite")

    def mutate(arrays):
        arrays["query_admitted"] = np.ones(3, dtype=np.bool_)

    _rewrite_payload(suite, 0, mutate)
    result = benchmark.evaluate_information_contract_suite(suite)
    case = next(
        item for item in result["cases"]
        if item["case_id"] == "admissible-shared-dependence"
    )
    assert case["metrics"]["gauge"]["false_accept_count"] == 1
    assert case["contract_checks"]["gauge_admission_consistent"] is False
    assert case["contract_pass"] is False
    assert result["aggregate"]["contract"]["all_cases_pass"] is False


def test_rejected_decision_must_return_registered_fallback(
    tmp_path: Path,
) -> None:
    suite = benchmark.generate_smoke_suite(tmp_path / "suite")

    def mutate(arrays):
        arrays["selected_action"] = np.array(0, dtype=np.int64)

    _rewrite_payload(suite, 1, mutate)
    result = benchmark.evaluate_information_contract_suite(suite)
    case = next(
        item for item in result["cases"]
        if item["case_id"] == "ambiguous-exact-fallback"
    )
    assert case["metrics"]["decision"]["expected_admitted"] is False
    assert case["contract_checks"]["decision_policy_consistent"] is False
    assert case["contract_pass"] is False


def test_unknown_payload_array_is_rejected(tmp_path: Path) -> None:
    suite = benchmark.generate_smoke_suite(tmp_path / "suite")

    def mutate(arrays):
        arrays["hidden_target_selection"] = np.ones(1)

    _rewrite_payload(suite, 0, mutate)
    with pytest.raises(ValueError, match="unregistered arrays"):
        benchmark.evaluate_information_contract_suite(suite)


def test_cli_smoke_and_replay_are_byte_identical(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    assert benchmark.main(["smoke", str(directory)]) == 0
    replay = tmp_path / "replay.json"
    assert benchmark.main(
        ["evaluate", str(directory / "suite.json"), str(replay)]
    ) == 0
    assert replay.read_bytes() == (directory / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.main(["smoke", str(directory)])
