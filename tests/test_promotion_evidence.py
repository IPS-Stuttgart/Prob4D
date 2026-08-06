from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from prob4d.heldout_promotion import (
    PromotionQueryRowV1,
    build_query_results,
    evaluate_heldout_promotion,
    promotion_lock_from_config,
)
from prob4d.promotion_evidence import (
    PROMOTION_EVIDENCE_CARD_SCHEMA,
    build_promotion_evidence_card,
    load_promotion_evidence_card,
    promotion_evidence_card_from_dict,
    render_promotion_evidence_markdown,
    write_promotion_evidence_card,
)

_ROLES = (
    ("fallback", "physical_fallback", None, "bpt-fallback", False),
    (
        "framewise",
        "framewise_explicit_joint_gauge",
        "provider-framewise",
        "bpt-framewise",
        False,
    ),
    (
        "identity",
        "cross_window_identity_marginalized",
        "provider-identity",
        "bpt-identity",
        False,
    ),
    (
        "persistent",
        "persistent_explicit_joint_gauge",
        "provider-persistent",
        "bpt-persistent",
        False,
    ),
    (
        "rowwise",
        "rowwise_gauge_marginalized",
        "provider-rowwise",
        "bpt-rowwise",
        False,
    ),
    (
        "sensor",
        "sensor_assisted",
        "provider-sensor",
        "bpt-sensor",
        True,
    ),
    (
        "visual",
        "visual_baseline",
        "provider-visual",
        "bpt-visual",
        False,
    ),
)


def _config() -> dict[str, Any]:
    return {
        "experiment_id": "heldout-v1",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "a" * 40,
        "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "bayesian_phystwin_revision": "b" * 40,
        "motioncrafter_revision": "c" * 40,
        "model_set_id": "d" * 64,
        "prediction_run_spec_id": "e" * 64,
        "provider_evaluation_manifest_sha256": "f" * 64,
        "frozen_artifact_ids": {
            "provider_configuration": "0" * 64,
            "gauge_calibration": "1" * 64,
            "point_calibration": "2" * 64,
            "source_reliability_calibration": "3" * 64,
            "material_identity_calibration": "4" * 64,
            "selection_lock": "5" * 64,
            "bayesian_guard_configuration": "6" * 64,
        },
        "development_group_ids": ["dev-1"],
        "calibration_group_ids": ["cal-1"],
        "target_group_ids": ["target-1", "target-2"],
        "arms": [
            {
                "arm_id": arm_id,
                "role": role,
                "provider_method_id": provider_method,
                "query_method_id": query_method,
                "sensor_assisted": sensor_assisted,
                "metadata": {},
            }
            for (
                arm_id,
                role,
                provider_method,
                query_method,
                sensor_assisted,
            ) in _ROLES
        ],
        "provider_reference_arm_id": "visual",
        "primary_query_arm_id": "identity",
        "bootstrap_resamples": 500,
        "bootstrap_seed": 11,
        "minimum_target_group_count": 2,
        "query_superiority_margin_mm": 0.1,
        "harmful_update_margin_mm": 0.0,
        "maximum_harmful_accepted_updates": 0,
        "maximum_worst_group_regression_mm": 0.0,
        "maximum_technical_failures": 0,
        "minimum_mean_accepted_coverage": 0.9,
        "metadata": {"split_semantics": "complete-object-session-v1"},
    }


def _provider_report(lock: Any) -> dict[str, Any]:
    cases = [
        {
            "case_id": f"{group_id}-case",
            "group_id": group_id,
            "method_id": method_id,
        }
        for group_id in lock.target_group_ids
        for method_id in lock.provider_method_ids
    ]
    return {
        "schema_name": "prob4d.provider-evaluation-report",
        "schema_version": 3,
        "source_manifest_sha256": lock.provider_evaluation_manifest_sha256,
        "primary_mode": "metric",
        "primary_support": "common_across_registered_methods",
        "reference_method": lock.provider_reference_method_id,
        "bootstrap_resamples": lock.bootstrap_resamples,
        "bootstrap_seed": lock.bootstrap_seed,
        "legacy_artifacts_allowed": False,
        "cases": cases,
        "decision": {
            "policy_id": "provider-v1",
            "overall_passed": True,
            "rules": [],
        },
    }


def _report_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = promotion_lock_from_config(_config())
    rows = []
    for group_id in lock.target_group_ids:
        fallback_id = hashlib.sha256(f"fallback-{group_id}".encode()).hexdigest()
        rows.append(
            PromotionQueryRowV1(
                group_id=group_id,
                arm_id="fallback",
                query_rmse_mm=5.0,
                deployed_artifact_id=fallback_id,
                fallback_artifact_id=fallback_id,
                accepted=None,
                exact_fallback_reproduced=None,
                accepted_coverage=None,
                accepted_width_mm=None,
            )
        )
        for arm in lock.arms:
            if arm.role == "physical_fallback":
                continue
            rows.append(
                PromotionQueryRowV1(
                    group_id=group_id,
                    arm_id=arm.arm_id,
                    query_rmse_mm=4.0,
                    deployed_artifact_id=hashlib.sha256(
                        f"deployed-{group_id}-{arm.arm_id}".encode()
                    ).hexdigest(),
                    fallback_artifact_id=fallback_id,
                    accepted=True,
                    exact_fallback_reproduced=None,
                    accepted_coverage=0.94,
                    accepted_width_mm=1.2,
                )
            )
    query_results = build_query_results(lock, rows=tuple(rows))
    provider_report = _provider_report(lock)
    provider_bytes = json.dumps(provider_report, sort_keys=True).encode()
    report = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256=hashlib.sha256(provider_bytes).hexdigest(),
    )
    return lock.to_dict(), report.to_dict()


def _resign(card: dict[str, Any]) -> None:
    descriptor = {key: value for key, value in card.items() if key != "evidence_card_id"}
    payload = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    card["evidence_card_id"] = hashlib.sha256(payload).hexdigest()


def test_evidence_card_is_compact_content_addressed_and_round_trips(
    tmp_path: Path,
) -> None:
    lock, report = _report_pair()
    card = build_promotion_evidence_card(lock, report)

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


def test_evidence_card_rejects_identity_tampering() -> None:
    lock, report = _report_pair()
    card = build_promotion_evidence_card(lock, report)
    card["guarded_query"]["accepted_update_count"] = 99

    with pytest.raises(
        ValueError,
        match="accepted and rejected counts|ID mismatch",
    ):
        promotion_evidence_card_from_dict(card)


def test_evidence_card_rejects_resigned_semantic_tampering() -> None:
    lock, report = _report_pair()
    card = build_promotion_evidence_card(lock, report)
    card["status"] = "FAIL"
    _resign(card)

    with pytest.raises(ValueError, match="status disagrees"):
        promotion_evidence_card_from_dict(card)


def test_markdown_keeps_claim_boundary() -> None:
    lock, report = _report_pair()
    card = build_promotion_evidence_card(lock, report)
    rendered = render_promotion_evidence_markdown(card)

    assert card["claim_boundary"] in rendered
    assert "Exact fallback failures" in rendered


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_name":"a","schema_name":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_promotion_evidence_card(path)
