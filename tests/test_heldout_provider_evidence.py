from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from prob4d._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    promotion_lock_from_config,
    write_promotion_lock,
)
from prob4d._heldout_promotion_query import (
    PromotionQueryRowV1,
    build_query_results,
    write_query_results,
)
from prob4d._heldout_promotion_report import (
    HeldoutProviderPromotionReportV1,
    evaluate_heldout_promotion,
    write_promotion_report,
)
from prob4d.heldout_provider_evidence import (
    HeldoutProviderEvidenceV2,
    SelectedCandidateBindingV1,
    build_heldout_provider_evidence,
    heldout_provider_evidence_from_dict,
    load_heldout_provider_evidence,
    main,
    write_heldout_provider_evidence,
)
from prob4d.selection_evidence import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    DeploymentDecisionV1,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionEvidenceBundleV2,
    SelectionRuleV1,
    build_selection_evidence_bundle,
    write_selection_evidence,
)

ROLES = (
    ("fallback", "physical_fallback", None, "bpt-fallback", False),
    ("visual", "visual_baseline", "provider-visual", "bpt-visual", False),
    (
        "rowwise",
        "rowwise_gauge_marginalized",
        "provider-rowwise",
        "bpt-rowwise",
        False,
    ),
    (
        "framewise",
        "framewise_explicit_joint_gauge",
        "provider-framewise",
        "bpt-framewise",
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
        "identity",
        "cross_window_identity_marginalized",
        "provider-identity",
        "bpt-identity",
        False,
    ),
    ("sensor", "sensor_assisted", "provider-sensor", "bpt-sensor", True),
)


def _arm_values() -> list[dict[str, object]]:
    return [
        {
            "arm_id": arm_id,
            "role": role,
            "query_method_id": query_method,
            "provider_method_id": provider_method,
            "sensor_assisted": sensor_assisted,
            "metadata": {},
        }
        for arm_id, role, provider_method, query_method, sensor_assisted in ROLES
    ]


def _lock() -> HeldoutProviderPromotionLockV1:
    return promotion_lock_from_config(
        {
            "experiment_id": "integrated-heldout-evidence-v2",
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
            "development_group_ids": ["development-1"],
            "calibration_group_ids": ["calibration-1", "calibration-2"],
            "target_group_ids": ["target-1", "target-2", "target-3"],
            "arms": _arm_values(),
            "provider_reference_arm_id": "visual",
            "primary_query_arm_id": "identity",
            "bootstrap_resamples": 500,
            "bootstrap_seed": 17,
            "minimum_target_group_count": 3,
            "query_superiority_margin_mm": 0.0,
            "harmful_update_margin_mm": 0.0,
            "maximum_harmful_accepted_updates": 0,
            "maximum_worst_group_regression_mm": 0.0,
            "maximum_technical_failures": 0,
            "minimum_mean_accepted_coverage": 0.9,
            "metadata": {},
        }
    )


def _candidate_artifact(group_id: str) -> str:
    return hashlib.sha256(f"candidate-{group_id}".encode()).hexdigest()


def _fallback_artifact(group_id: str) -> str:
    return hashlib.sha256(f"fallback-{group_id}".encode()).hexdigest()


def _selection(
    lock: HeldoutProviderPromotionLockV1,
    *,
    calibration_groups: tuple[str, ...] | None = None,
) -> SelectionEvidenceBundleV2:
    groups = lock.calibration_group_ids if calibration_groups is None else calibration_groups
    candidates = (
        CandidateSpecV1(
            candidate_id="fallback",
            method_id="physical-fallback",
            complexity_rank=0,
        ),
        CandidateSpecV1(
            candidate_id="identity",
            method_id="provider-identity",
            complexity_rank=2,
            parameters={"minimum_track_length": 3},
        ),
    )
    rows: list[CalibrationMetricRowV1] = []
    for group_id in groups:
        rows.extend(
            (
                CalibrationMetricRowV1(
                    group_id=group_id,
                    candidate_id="fallback",
                    metrics={"rmse_mm": 5.0, "coverage": 1.0},
                ),
                CalibrationMetricRowV1(
                    group_id=group_id,
                    candidate_id="identity",
                    metrics={"rmse_mm": 2.0, "coverage": 0.95},
                ),
            )
        )
    rule = SelectionRuleV1(
        primary=MetricOrderV1("rmse_mm", "minimize"),
        constraints=(MetricConstraintV1("coverage", "at_least", 0.9),),
    )
    decisions = tuple(
        DeploymentDecisionV1(
            group_id=group_id,
            candidate_id="identity",
            accepted=group_id != "target-3",
            guard_name="source-frozen-regret-guard-v1",
            guard_value=0.1 if group_id != "target-3" else 0.9,
            candidate_artifact_id=_candidate_artifact(group_id),
            fallback_artifact_id=_fallback_artifact(group_id),
            deployed_artifact_id=(
                _candidate_artifact(group_id)
                if group_id != "target-3"
                else _fallback_artifact(group_id)
            ),
            reason="accepted" if group_id != "target-3" else "exact fallback",
        )
        for group_id in lock.target_group_ids
    )
    return build_selection_evidence_bundle(
        experiment_id=lock.experiment_id,
        source_repository=lock.source_repository,
        source_revision=lock.source_revision,
        candidates=candidates,
        calibration_rows=rows,
        selection_rule=rule,
        deployment_decisions=decisions,
    )


def _provider_report(lock: HeldoutProviderPromotionLockV1) -> dict[str, object]:
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
        "source_manifest": "/sealed/provider-manifest.json",
        "source_manifest_sha256": lock.provider_evaluation_manifest_sha256,
        "primary_mode": "metric",
        "primary_support": "common_across_registered_methods",
        "secondary_support": "native_per_method",
        "reference_method": lock.provider_reference_method_id,
        "bootstrap_resamples": lock.bootstrap_resamples,
        "bootstrap_seed": lock.bootstrap_seed,
        "legacy_artifacts_allowed": False,
        "evaluation_chunk_size": 4096,
        "manifest_metadata": {},
        "method_metadata": {},
        "cases": cases,
        "aggregate": {},
        "comparisons": {},
        "decision_policy": {},
        "decision": {
            "policy_id": "provider-gate-v1",
            "overall_passed": True,
            "group_count_passed": True,
            "rules": [],
        },
        "support_semantics": "common support",
        "claim_boundary": "provider only",
    }


def _provider_text(lock: HeldoutProviderPromotionLockV1) -> str:
    return json.dumps(
        _provider_report(lock),
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _query_rows(lock: HeldoutProviderPromotionLockV1) -> tuple[PromotionQueryRowV1, ...]:
    rows: list[PromotionQueryRowV1] = []
    for group_id in lock.target_group_ids:
        fallback_id = _fallback_artifact(group_id)
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
            primary = arm.arm_id == lock.primary_query_arm_id
            rejected = primary and group_id == "target-3"
            deployed = (
                _fallback_artifact(group_id)
                if rejected
                else (
                    _candidate_artifact(group_id)
                    if primary
                    else hashlib.sha256(
                        f"deployed-{group_id}-{arm.arm_id}".encode()
                    ).hexdigest()
                )
            )
            rows.append(
                PromotionQueryRowV1(
                    group_id=group_id,
                    arm_id=arm.arm_id,
                    query_rmse_mm=5.0 if rejected else (4.0 if primary else 4.5),
                    deployed_artifact_id=deployed,
                    fallback_artifact_id=fallback_id,
                    accepted=not rejected,
                    exact_fallback_reproduced=True if rejected else None,
                    accepted_coverage=None if rejected else 0.95,
                    accepted_width_mm=None if rejected else 1.0,
                )
            )
    return tuple(rows)


def _binding() -> SelectedCandidateBindingV1:
    return SelectedCandidateBindingV1(
        candidate_id="identity",
        arm_id="identity",
        method_role="provider",
        method_id="provider-identity",
    )


def _evidence() -> HeldoutProviderEvidenceV2:
    lock = _lock()
    selection = _selection(lock)
    query = build_query_results(lock, rows=_query_rows(lock))
    provider_text = _provider_text(lock)
    report = evaluate_heldout_promotion(
        lock,
        query,
        _provider_report(lock),
        provider_report_sha256=hashlib.sha256(provider_text.encode()).hexdigest(),
    )
    return build_heldout_provider_evidence(
        selection_evidence=selection,
        promotion_lock=lock,
        selected_candidate_binding=_binding(),
        provider_report_json=provider_text,
        query_results=query,
        promotion_report=report,
    )


def test_complete_evidence_replays_selection_target_and_bootstrap_plan() -> None:
    evidence = _evidence()
    replay = evidence.replay_report()

    assert replay.selected_candidate_id == "identity"
    assert replay.selected_arm_id == "identity"
    assert replay.candidate_order == ("identity", "fallback")
    assert replay.calibration_group_count == 2
    assert replay.target_group_count == 3
    assert replay.bootstrap_resamples == 500
    assert replay.bootstrap_seed == 17
    assert replay.accepted_update_count == 2
    assert replay.fallback_update_count == 1
    assert replay.exact_fallback_count == 1
    assert replay.provider_passed
    assert replay.query_passed
    assert replay.overall_passed
    assert len(replay.replay_id) == 64


def test_round_trip_is_self_contained_and_no_clobber(tmp_path: Path) -> None:
    evidence = _evidence()
    path = tmp_path / "heldout-evidence-v2.json"

    write_heldout_provider_evidence(evidence, path)
    loaded = load_heldout_provider_evidence(path)

    assert loaded.to_dict() == evidence.to_dict()
    with pytest.raises(FileExistsError):
        write_heldout_provider_evidence(evidence, path)


def test_exact_provider_report_bytes_are_bound() -> None:
    evidence = _evidence()
    payload = copy.deepcopy(evidence.to_dict())
    payload["provider_report_json"] += " "

    with pytest.raises(ValueError, match="exact provider-report bytes"):
        heldout_provider_evidence_from_dict(payload)


def test_calibration_roster_must_equal_frozen_lock() -> None:
    lock = _lock()
    query = build_query_results(lock, rows=_query_rows(lock))
    provider_text = _provider_text(lock)
    report = evaluate_heldout_promotion(
        lock,
        query,
        _provider_report(lock),
        provider_report_sha256=hashlib.sha256(provider_text.encode()).hexdigest(),
    )

    with pytest.raises(ValueError, match="calibration groups"):
        build_heldout_provider_evidence(
            selection_evidence=_selection(
                lock,
                calibration_groups=("other-1", "other-2"),
            ),
            promotion_lock=lock,
            selected_candidate_binding=_binding(),
            provider_report_json=provider_text,
            query_results=query,
            promotion_report=report,
        )


def test_selected_candidate_must_bind_primary_arm_method() -> None:
    evidence = _evidence()
    bad_binding = SelectedCandidateBindingV1(
        candidate_id="identity",
        arm_id="identity",
        method_role="query",
        method_id="provider-identity",
    )

    with pytest.raises(ValueError, match="bound promotion arm"):
        build_heldout_provider_evidence(
            selection_evidence=evidence.selection_evidence,
            promotion_lock=evidence.promotion_lock,
            selected_candidate_binding=bad_binding,
            provider_report_json=evidence.provider_report_json,
            query_results=evidence.query_results,
            promotion_report=evidence.promotion_report,
        )


def test_selection_and_query_artifact_decisions_must_match() -> None:
    evidence = _evidence()
    rows = list(evidence.query_results.rows)
    index = next(
        index
        for index, row in enumerate(rows)
        if row.group_id == "target-1" and row.arm_id == "identity"
    )
    rows[index] = replace(
        rows[index],
        deployed_artifact_id=hashlib.sha256(b"different-candidate").hexdigest(),
    )
    changed_query = build_query_results(evidence.promotion_lock, rows=rows)

    with pytest.raises(ValueError, match="deployed artifact mismatch"):
        build_heldout_provider_evidence(
            selection_evidence=evidence.selection_evidence,
            promotion_lock=evidence.promotion_lock,
            selected_candidate_binding=evidence.selected_candidate_binding,
            provider_report_json=evidence.provider_report_json,
            query_results=changed_query,
        )


def test_retained_promotion_report_must_equal_replay() -> None:
    evidence = _evidence()
    report = evidence.promotion_report
    changed = HeldoutProviderPromotionReportV1(
        promotion_lock_id=report.promotion_lock_id,
        query_results_id=report.query_results_id,
        provider_report_sha256=report.provider_report_sha256,
        provider_evaluation_manifest_sha256=(
            report.provider_evaluation_manifest_sha256
        ),
        provider_audit=report.provider_audit,
        provider_decision=report.provider_decision,
        query_aggregate=report.query_aggregate,
        query_decision=report.query_decision,
        overall_passed=not report.overall_passed,
    )

    with pytest.raises(ValueError, match="deterministic evidence replay"):
        build_heldout_provider_evidence(
            selection_evidence=evidence.selection_evidence,
            promotion_lock=evidence.promotion_lock,
            selected_candidate_binding=evidence.selected_candidate_binding,
            provider_report_json=evidence.provider_report_json,
            query_results=evidence.query_results,
            promotion_report=changed,
        )


def test_duplicate_keys_inside_embedded_provider_report_fail_closed() -> None:
    evidence = _evidence()

    with pytest.raises(ValueError, match="duplicate JSON key"):
        build_heldout_provider_evidence(
            selection_evidence=evidence.selection_evidence,
            promotion_lock=evidence.promotion_lock,
            selected_candidate_binding=evidence.selected_candidate_binding,
            provider_report_json='{"schema_name":"one","schema_name":"two"}',
            query_results=evidence.query_results,
            promotion_report=evidence.promotion_report,
        )


def test_pack_and_verify_cli(tmp_path: Path) -> None:
    evidence = _evidence()
    selection_path = tmp_path / "selection.json"
    lock_path = tmp_path / "lock.json"
    provider_path = tmp_path / "provider.json"
    query_path = tmp_path / "query.json"
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "evidence.json"

    write_selection_evidence(evidence.selection_evidence, selection_path)
    write_promotion_lock(evidence.promotion_lock, lock_path)
    provider_path.write_bytes(evidence.provider_report_json.encode("utf-8"))
    write_query_results(evidence.query_results, query_path)
    write_promotion_report(evidence.promotion_report, report_path)

    assert (
        main(
            [
                "pack",
                "--selection-evidence",
                str(selection_path),
                "--promotion-lock",
                str(lock_path),
                "--provider-report",
                str(provider_path),
                "--query-results",
                str(query_path),
                "--promotion-report",
                str(report_path),
                "--arm-id",
                "identity",
                "--method-role",
                "provider",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert main(["verify", str(output_path)]) == 0
    assert load_heldout_provider_evidence(output_path).evidence_id == evidence.evidence_id
