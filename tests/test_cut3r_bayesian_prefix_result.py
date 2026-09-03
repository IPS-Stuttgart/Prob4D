from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/cut3r-bayesian-prefix-dev-v1"


@pytest.mark.parametrize(
    "name", ["result.json", "prediction-barrier.json", "numerical-verification.json"]
)
def test_published_evidence_content_identity(name: str) -> None:
    record = json.loads((EVIDENCE / name).read_text())
    identity = record.pop("artifact_id")
    assert identity == content_id(record)


def test_complete_denominator_and_honest_no_superiority_decision() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    verification = json.loads((EVIDENCE / "numerical-verification.json").read_text())
    barrier = json.loads((EVIDENCE / "prediction-barrier.json").read_text())
    assert result["scored_sequence_count"] == 3
    assert result["complete_denominator"] is True
    assert all(row["status"] == "ordinary_success" for row in barrier["cases"].values())
    assert verification["scored_rows"] == 28
    assert verification["maximum_metric_difference"] < 1e-11
    assert verification["result_id"] == result["artifact_id"]
    assert result["prediction_barrier_id"] == barrier["artifact_id"]
    assert barrier["future_3d_opened"] is False
    assert result["provider_inference_rerun"] is False
    assert result["protected_targets_accessed"] is False
    assert result["autopromotion"] is False
    aggregate = result["aggregate"]
    shared = aggregate["bayesian_shared"]["rmse_prefix_span"]
    assert shared < aggregate["cut3r_full_prefix_alignment"]["rmse_prefix_span"]
    assert shared > aggregate["last_residual"]["rmse_prefix_span"]
    assert (
        aggregate["bayesian_shared"]["nll_per_coordinate"]
        > aggregate["bayesian_iid"]["nll_per_coordinate"]
    )
