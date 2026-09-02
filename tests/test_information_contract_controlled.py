from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.information_contract_controlled import evaluate_controlled_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/information_contract_v1/controlled_suite.json"


def test_controlled_suite_reproduces_all_anti_gaming_controls() -> None:
    result = evaluate_controlled_suite(SUITE)
    systems = {row["system_id"]: row for row in result["systems"]}

    assert result["completion"] == {
        "system_count": 7,
        "prediction_system_count": 5,
        "query_decision_system_count": 2,
        "communication_system_count": 1,
    }
    assert "overall_score" not in result
    assert "overall_rank" not in result
    assert result["cross_system"]["shared_bias_counterexample_count"] == 1
    assert all(value == 0 for value in result["information_boundary"].values())
    assert (
        systems["accurate-overconfident"]["prediction"]["probabilistic"]
        ["equal_unit_registered_coverage"]
        == 0.0
    )
    assert (
        systems["accurate-overconfident"]["prediction"]["probabilistic"]
        ["equal_unit_normalized_nees"]
        == pytest.approx(450.0)
    )
    assert (
        systems["joint-dependence-aware"]["prediction"]["probabilistic"]
        ["equal_unit_joint_gaussian_nll"]
        < systems["diagonal-same-mean-marginals"]["prediction"]
        ["probabilistic"]["equal_unit_joint_gaussian_nll"]
    )
    assert systems["ambiguity-aware-contract"]["query_decision"]["fallback_count"] == 1
    assert (
        systems["ambiguity-aware-contract"]["query_decision"]["query_identified_count"]
        == 1
    )
    assert (
        systems["ambiguity-aware-contract"]["query_decision"]["query_rejected_count"]
        == 1
    )
    assert (
        systems["unsupported-specificity"]["query_decision"]
        ["harmful_nonfallback_count"]
        == 1
    )
    assert (
        systems["ambiguity-aware-contract"]["communication"]
        ["equal_unit_compression_ratio"]
        == 17.0
    )


def test_controlled_suite_rejects_non_positive_definite_covariance(
    tmp_path: Path,
) -> None:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    suite["systems"][0]["prediction_cases"][0]["covariance"] = [
        [1.0, 2.0],
        [2.0, 1.0],
    ]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    with pytest.raises(ValueError, match="positive definite"):
        evaluate_controlled_suite(path)


def test_controlled_suite_rejects_quotient_mass_drift(tmp_path: Path) -> None:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    suite["systems"][3]["query_decision_cases"][0]["quotient_masses"] = {
        "unresolved-orbit": 0.9
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    with pytest.raises(ValueError, match="must sum to one"):
        evaluate_controlled_suite(path)
