import json
from pathlib import Path
from typing import Any

import pytest

from prob4d.promotion_evidence import (
    PROMOTION_EVIDENCE_CARD_SCHEMA,
    build_promotion_evidence_card,
    load_promotion_evidence_card,
    promotion_evidence_card_from_dict,
    render_promotion_evidence_markdown,
    write_promotion_evidence_card,
)


def _arm(
    arm_id: str,
    role: str,
    provider: str | None,
    query: str,
    sensor: bool = False,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "role": role,
        "provider_method_id": provider,
        "query_method_id": query,
        "sensor_assisted": sensor,
        "metadata": {},
    }


def _lock() -> dict[str, Any]:
    arms = [
        _arm("fallback", "physical_fallback", None, "bpt-fallback"),
        _arm(
            "identity",
            "cross_window_identity_marginalized",
            "provider-identity",
            "bpt-identity",
        ),
    ]
    return {
        "schema_name": "prob4d.heldout-provider-promotion-lock",
        "schema_version": 1,
        "experiment_id": "heldout-v1",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "a" * 40,
        "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "bayesian_phystwin_revision": "b" * 40,
        "motioncrafter_revision": "c" * 40,
        "model_set_id": "d" * 64,
        "prediction_run_spec_id": "e" * 64,
        "provider_evaluation_manifest_sha256": "f" * 64,
        "frozen_artifact_ids": {"provider_configuration": "0" * 64},
        "development_group_ids": ["dev-1"],
        "calibration_group_ids": ["cal-1"],
        "target_group_ids": ["target-1", "target-2"],
        "arms": arms,
        "provider_reference_arm_id": "identity",
        "primary_query_arm_id": "identity",
        "bootstrap_resamples": 500,
        "bootstrap_seed": 1,
        "minimum_target_group_count": 2,
        "query_superiority_margin_mm": 0.1,
        "harmful_update_margin_mm": 0.0,
        "maximum_harmful_accepted_updates": 0,
        "maximum_worst_group_regression_mm": 0.0,
        "maximum_technical_failures": 0,
        "minimum_mean_accepted_coverage": 0.9,
        "metadata": {"split_semantics": "complete-object-session-v1"},
        "claim_boundary": "lock only",
        "promotion_lock_id": "1" * 64,
    }


def _report() -> dict[str, Any]:
    return {
        "schema_name": "prob4d.heldout-provider-promotion-report",
        "schema_version": 1,
        "promotion_lock_id": "1" * 64,
        "query_results_id": "2" * 64,
        "provider_report_sha256": "3" * 64,
        "provider_evaluation_manifest_sha256": "f" * 64,
        "provider_audit": {
            "reference_method": "provider-identity",
            "case_count": 2,
            "group_count": 2,
            "decision_policy_id": "provider-v1",
        },
        "provider_decision": {"overall_passed": True},
        "query_aggregate": {
            "identity": {
                "mean_query_rmse_mm": 4.0,
                "worst_group_regression_mm": -0.5,
                "accepted_update_count": 2,
                "rejected_update_count": 0,
                "harmful_accepted_update_count": 0,
                "technical_failure_count": 0,
                "mean_accepted_coverage": 0.94,
                "mean_accepted_width_mm": 1.2,
            }
        },
        "query_decision": {
            "overall_passed": True,
            "physical_fallback_arm_id": "fallback",
            "paired_bootstrap": {
                "mean": -1.0,
                "ci95_lower": -1.2,
                "ci95_upper": -0.8,
                "group_count": 2,
                "semantics": (
                    "paired-target-group-bootstrap-"
                    "candidate-minus-physical-fallback-v1"
                ),
            },
            "exact_fallback_failure_count": 0,
        },
        "overall_passed": True,
        "claim_boundary": "Only the frozen held-out provider and query gate.",
        "report_id": "4" * 64,
    }


def test_evidence_card_is_compact_content_addressed_and_round_trips(
    tmp_path: Path,
) -> None:
    card = build_promotion_evidence_card(_lock(), _report())
    assert card["schema_name"] == PROMOTION_EVIDENCE_CARD_SCHEMA
    assert card["repositories"]["prob4d"]["revision"] == "a" * 40
    assert card["cohort"]["target_group_count"] == 2
    assert card["guarded_query"]["paired_candidate_minus_fallback_mm"]["mean"] == -1.0
    assert card["guarded_query"]["harmful_accepted_update_count"] == 0

    json_path = tmp_path / "evidence.json"
    md_path = tmp_path / "evidence.md"
    write_promotion_evidence_card(card, json_path, md_path)
    assert load_promotion_evidence_card(json_path) == card
    assert "Overall decision: **PASS**" in md_path.read_text()
    with pytest.raises(FileExistsError):
        write_promotion_evidence_card(card, json_path, md_path)


def test_evidence_card_rejects_tampering() -> None:
    card = build_promotion_evidence_card(_lock(), _report())
    card["guarded_query"]["accepted_update_count"] = 99
    with pytest.raises(ValueError, match="ID mismatch"):
        promotion_evidence_card_from_dict(card)


def test_markdown_keeps_claim_boundary() -> None:
    card = build_promotion_evidence_card(_lock(), _report())
    rendered = render_promotion_evidence_markdown(card)
    assert card["claim_boundary"] in rendered
    assert "Exact fallback failures" in rendered


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_name":"a","schema_name":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_promotion_evidence_card(path)
