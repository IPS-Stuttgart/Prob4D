from __future__ import annotations

import copy
from pathlib import Path

import pytest

from prob4d.selection_evidence import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    DeploymentDecisionV1,
    MetricOrderV1,
    SelectionRuleV1,
    build_selection_evidence_bundle,
    load_selection_evidence,
    selection_evidence_from_dict,
)


def _bundle():
    candidates = (
        CandidateSpecV1(
            candidate_id="fallback",
            method_id="physical-fallback",
            complexity_rank=0,
            parameters={"visual_update": False},
        ),
        CandidateSpecV1(
            candidate_id="visual",
            method_id="persistent-explicit-joint-gauge",
            complexity_rank=1,
            parameters={"minimum_track_length": 3},
        ),
    )
    rows = (
        CalibrationMetricRowV1(
            group_id="object-a",
            candidate_id="fallback",
            metrics={"rmse_mm": 5.0},
        ),
        CalibrationMetricRowV1(
            group_id="object-a",
            candidate_id="visual",
            metrics={"rmse_mm": 2.0},
        ),
    )
    candidate_artifact_id = "a" * 64
    fallback_artifact_id = "b" * 64
    decisions = (
        DeploymentDecisionV1(
            group_id="target-a",
            candidate_id="visual",
            accepted=True,
            guard_name="target-blind-guard-v1",
            guard_value=0.1,
            candidate_artifact_id=candidate_artifact_id,
            fallback_artifact_id=fallback_artifact_id,
            deployed_artifact_id=candidate_artifact_id,
            reason="accepted",
        ),
    )
    return build_selection_evidence_bundle(
        experiment_id="claim-boundary-regression",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="c" * 40,
        candidates=candidates,
        calibration_rows=rows,
        selection_rule=SelectionRuleV1(
            primary=MetricOrderV1("rmse_mm", "minimize"),
        ),
        deployment_decisions=decisions,
    )


def test_tampered_claim_boundary_is_rejected() -> None:
    payload = copy.deepcopy(_bundle().to_dict())
    payload["claim_boundary"] = "weakened claim boundary"

    with pytest.raises(ValueError, match="claim_boundary mismatch"):
        selection_evidence_from_dict(payload)


def test_non_integer_schema_version_is_rejected() -> None:
    payload = copy.deepcopy(_bundle().to_dict())
    payload["schema_version"] = 2.0

    with pytest.raises(ValueError, match="unsupported selection evidence version"):
        selection_evidence_from_dict(payload)


def test_unreadable_evidence_is_reported_as_validation_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unreadable or invalid JSON"):
        load_selection_evidence(tmp_path / "missing.json")
