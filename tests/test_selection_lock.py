from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from prob4d.deployment_ledger import (
    append_deployment_decision,
    build_deployment_ledger,
    deployment_ledger_from_dict,
    load_deployment_ledger,
    write_deployment_ledger,
)
from prob4d.selection_evidence import DeploymentDecisionV1
from prob4d.selection_lock import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionRuleV1,
    build_selection_lock,
    load_selection_lock,
    selection_lock_from_dict,
    write_selection_lock,
)

SOURCE_REVISION = "a" * 40


def candidates() -> tuple[CandidateSpecV1, ...]:
    return (
        CandidateSpecV1(
            candidate_id="fallback",
            method_id="physical-fallback",
            complexity_rank=0,
            parameters={"visual_update": False},
        ),
        CandidateSpecV1(
            candidate_id="persistent-gauge",
            method_id="persistent-explicit-joint-gauge",
            complexity_rank=2,
            parameters={"minimum_track_length": 3, "guard": 0.25},
        ),
        CandidateSpecV1(
            candidate_id="overfit",
            method_id="persistent-high-capacity",
            complexity_rank=5,
            parameters={"rank": 16},
        ),
    )


def rows() -> tuple[CalibrationMetricRowV1, ...]:
    values = {
        ("object-a", "fallback"): (5.0, 0.0, 1.0),
        ("object-b", "fallback"): (5.4, 0.0, 1.0),
        ("object-a", "persistent-gauge"): (2.1, 0.0, 0.95),
        ("object-b", "persistent-gauge"): (2.3, 0.0, 0.96),
        ("object-a", "overfit"): (1.0, 0.0, 0.80),
        ("object-b", "overfit"): (1.1, 0.0, 0.82),
    }
    return tuple(
        CalibrationMetricRowV1(
            group_id=group_id,
            candidate_id=candidate_id,
            metrics={
                "rmse_mm": rmse,
                "harmful_updates": harmful,
                "coverage": coverage,
            },
        )
        for (group_id, candidate_id), (rmse, harmful, coverage) in values.items()
    )


def rule() -> SelectionRuleV1:
    return SelectionRuleV1(
        primary=MetricOrderV1("rmse_mm", "minimize"),
        tie_break_metrics=(MetricOrderV1("coverage", "maximize"),),
        constraints=(
            MetricConstraintV1("harmful_updates", "at_most", 0.0, "sum"),
            MetricConstraintV1("coverage", "at_least", 0.9),
        ),
    )


def selection_lock():
    return build_selection_lock(
        experiment_id="prob4d-bpt-real-provider-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision=SOURCE_REVISION,
        candidates=candidates(),
        calibration_rows=rows(),
        selection_rule=rule(),
        metadata={"split_registry_id": "f" * 64},
    )


def decision(
    group_id: str,
    *,
    accepted: bool,
    guard_value: float,
    candidate_id: str = "persistent-gauge",
) -> DeploymentDecisionV1:
    candidate_artifact_id = ("b" if group_id == "target-a" else "c") * 64
    fallback_artifact_id = ("d" if group_id == "target-a" else "e") * 64
    return DeploymentDecisionV1(
        group_id=group_id,
        candidate_id=candidate_id,
        accepted=accepted,
        guard_name="target-blind-residual-guard-v1",
        guard_value=guard_value,
        candidate_artifact_id=candidate_artifact_id,
        fallback_artifact_id=fallback_artifact_id,
        deployed_artifact_id=(candidate_artifact_id if accepted else fallback_artifact_id),
        reason="accepted" if accepted else "guard rejection; exact fallback",
    )


def test_selection_lock_exists_before_any_target_deployment() -> None:
    lock = selection_lock()

    assert lock.selection_order == (
        "persistent-gauge",
        "fallback",
        "overfit",
    )
    assert lock.selected_candidate_id == "persistent-gauge"
    assert "decisions" not in lock.to_dict()
    report = lock.replay_report()
    assert report.deployment_group_count == 0
    assert report.accepted_update_count == 0
    assert report.fallback_update_count == 0
    assert report.exact_fallback_count == 0


def test_selection_lock_builder_is_input_order_invariant() -> None:
    expected = selection_lock()
    rebuilt = build_selection_lock(
        experiment_id=expected.experiment_id,
        source_repository=expected.source_repository,
        source_revision=expected.source_revision,
        candidates=tuple(reversed(candidates())),
        calibration_rows=tuple(reversed(rows())),
        selection_rule=rule(),
        metadata={"split_registry_id": "f" * 64},
    )

    assert rebuilt.selection_lock_id == expected.selection_lock_id
    assert rebuilt.replay_report().replay_digest == expected.replay_report().replay_digest


def test_selection_lock_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    lock = selection_lock()
    path = tmp_path / "selection-lock.json"

    write_selection_lock(lock, path)
    loaded = load_selection_lock(path)
    assert loaded.to_dict() == lock.to_dict()

    selected_tamper = copy.deepcopy(lock.to_dict())
    selected_tamper["selected_candidate_id"] = "fallback"
    with pytest.raises(ValueError, match="selected_candidate_id"):
        selection_lock_from_dict(selected_tamper)

    boundary_tamper = copy.deepcopy(lock.to_dict())
    boundary_tamper["claim_boundary"] = "selection may use target outcomes"
    with pytest.raises(ValueError, match="claim_boundary"):
        selection_lock_from_dict(boundary_tamper)

    version_tamper = copy.deepcopy(lock.to_dict())
    version_tamper["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        selection_lock_from_dict(version_tamper)


def test_selection_lock_rejects_incomplete_calibration_matrix() -> None:
    with pytest.raises(ValueError, match="complete group-by-candidate matrix"):
        build_selection_lock(
            experiment_id="experiment",
            source_repository="IPS-Stuttgart/Prob4D",
            source_revision=SOURCE_REVISION,
            candidates=candidates(),
            calibration_rows=rows()[:-1],
            selection_rule=rule(),
        )


def test_deployment_ledger_forms_an_immutable_hash_chain() -> None:
    lock = selection_lock()
    root = build_deployment_ledger(lock, metadata={"target_split": "sealed-v1"})
    first = append_deployment_decision(
        root,
        decision("target-a", accepted=True, guard_value=0.1),
    )
    second = append_deployment_decision(
        first,
        decision("target-b", accepted=False, guard_value=0.8),
    )

    assert root.decisions == ()
    assert root.previous_ledger_id is None
    assert first.previous_ledger_id == root.deployment_ledger_id
    assert second.previous_ledger_id == first.deployment_ledger_id
    assert len(first.decisions) == 1
    assert len(second.decisions) == 2
    assert second.accepted_update_count == 1
    assert second.fallback_update_count == 1
    assert second.exact_fallback_count == 1
    assert second.selection_lock_id == lock.selection_lock_id


def test_ledger_append_rejects_duplicate_group_and_candidate_drift() -> None:
    root = build_deployment_ledger(selection_lock())
    first = append_deployment_decision(
        root,
        decision("target-a", accepted=True, guard_value=0.1),
    )

    with pytest.raises(ValueError, match="already appended"):
        append_deployment_decision(
            first,
            decision("target-a", accepted=False, guard_value=2.0),
        )
    with pytest.raises(ValueError, match="locked candidate"):
        append_deployment_decision(
            first,
            decision(
                "target-b",
                accepted=True,
                guard_value=0.1,
                candidate_id="fallback",
            ),
        )


def test_deployment_ledger_round_trip_and_chain_tamper_rejection(
    tmp_path: Path,
) -> None:
    ledger = append_deployment_decision(
        append_deployment_decision(
            build_deployment_ledger(selection_lock()),
            decision("target-a", accepted=True, guard_value=0.1),
        ),
        decision("target-b", accepted=False, guard_value=0.8),
    )
    path = tmp_path / "deployment-ledger.json"

    write_deployment_ledger(ledger, path)
    loaded = load_deployment_ledger(path)
    assert loaded.to_dict() == ledger.to_dict()

    payload = copy.deepcopy(ledger.to_dict())
    payload["previous_ledger_id"] = "0" * 64
    with pytest.raises(ValueError, match="previous_ledger_id"):
        deployment_ledger_from_dict(payload)


def test_target_decisions_never_change_the_selection_lock_identity() -> None:
    lock = selection_lock()
    ledger = build_deployment_ledger(lock)
    original_lock_id = lock.selection_lock_id

    ledger = append_deployment_decision(
        ledger,
        decision("target-a", accepted=False, guard_value=99.0),
    )
    ledger = append_deployment_decision(
        ledger,
        decision("target-b", accepted=True, guard_value=-5.0),
    )

    assert lock.selection_lock_id == original_lock_id
    assert ledger.selection_lock_id == original_lock_id
    assert ledger.deployment_ledger_id != original_lock_id


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_name": "one", "schema_name": "two"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_selection_lock(path)


def test_written_lock_contains_complete_calibration_matrix(tmp_path: Path) -> None:
    lock = selection_lock()
    path = tmp_path / "selection-lock.json"
    write_selection_lock(lock, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["calibration_rows"]) == 6
    assert payload["selection_lock_id"] == lock.selection_lock_id
    assert payload["selection_order"] == [
        "persistent-gauge",
        "fallback",
        "overfit",
    ]
