from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from prob4d.selection_evidence import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    DeploymentDecisionV1,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionRuleV1,
    build_selection_evidence_bundle,
    load_selection_evidence,
    selection_evidence_from_dict,
    write_selection_evidence,
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


def decision(
    group_id: str,
    *,
    accepted: bool,
    guard_value: float,
) -> DeploymentDecisionV1:
    candidate_artifact_id = ("b" if group_id == "target-a" else "c") * 64
    fallback_artifact_id = ("d" if group_id == "target-a" else "e") * 64
    return DeploymentDecisionV1(
        group_id=group_id,
        candidate_id="persistent-gauge",
        accepted=accepted,
        guard_name="target-blind-residual-guard-v1",
        guard_value=guard_value,
        candidate_artifact_id=candidate_artifact_id,
        fallback_artifact_id=fallback_artifact_id,
        deployed_artifact_id=(
            candidate_artifact_id if accepted else fallback_artifact_id
        ),
        reason="accepted" if accepted else "guard rejection; exact fallback",
    )


def bundle():
    return build_selection_evidence_bundle(
        experiment_id="prob4d-bpt-real-provider-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision=SOURCE_REVISION,
        candidates=candidates(),
        calibration_rows=rows(),
        selection_rule=rule(),
        deployment_decisions=(
            decision("target-a", accepted=True, guard_value=0.1),
            decision("target-b", accepted=False, guard_value=0.8),
        ),
        metadata={"split_registry_id": "f" * 64},
    )


def test_bundle_replays_complete_order_and_exact_fallback() -> None:
    evidence = bundle()

    assert evidence.selection_order == (
        "persistent-gauge",
        "fallback",
        "overfit",
    )
    assert evidence.selected_candidate_id == "persistent-gauge"
    report = evidence.replay_report()
    assert report.accepted_update_count == 1
    assert report.fallback_update_count == 1
    assert report.exact_fallback_count == 1
    assert len(report.replay_digest) == 64


def test_builder_is_input_order_invariant() -> None:
    expected = bundle()
    rebuilt = build_selection_evidence_bundle(
        experiment_id=expected.experiment_id,
        source_repository=expected.source_repository,
        source_revision=expected.source_revision,
        candidates=tuple(reversed(candidates())),
        calibration_rows=tuple(reversed(rows())),
        selection_rule=rule(),
        deployment_decisions=(
            decision("target-b", accepted=False, guard_value=0.8),
            decision("target-a", accepted=True, guard_value=0.1),
        ),
        metadata={"split_registry_id": "f" * 64},
    )

    assert rebuilt.artifact_id == expected.artifact_id
    assert rebuilt.replay_report().replay_digest == expected.replay_report().replay_digest


def test_round_trip_replays_without_experiment_code(tmp_path: Path) -> None:
    evidence = bundle()
    path = tmp_path / "selection-evidence.json"

    write_selection_evidence(evidence, path)
    loaded = load_selection_evidence(path)

    assert loaded.artifact_id == evidence.artifact_id
    assert loaded.to_dict() == evidence.to_dict()


def test_tampered_selected_candidate_is_rejected() -> None:
    payload = copy.deepcopy(bundle().to_dict())
    payload["selected_candidate_id"] = "fallback"

    with pytest.raises(ValueError, match="selected_candidate_id"):
        selection_evidence_from_dict(payload)


def test_incomplete_calibration_matrix_is_rejected() -> None:
    incomplete = rows()[:-1]

    with pytest.raises(ValueError, match="complete group-by-candidate matrix"):
        build_selection_evidence_bundle(
            experiment_id="experiment",
            source_repository="IPS-Stuttgart/Prob4D",
            source_revision=SOURCE_REVISION,
            candidates=candidates(),
            calibration_rows=incomplete,
            selection_rule=rule(),
            deployment_decisions=(
                decision("target-a", accepted=True, guard_value=0.1),
            ),
        )


def test_rejected_update_must_reproduce_exact_fallback() -> None:
    with pytest.raises(ValueError, match="declared fallback"):
        DeploymentDecisionV1(
            group_id="target-a",
            candidate_id="persistent-gauge",
            accepted=False,
            guard_name="guard",
            guard_value=1.0,
            candidate_artifact_id="b" * 64,
            fallback_artifact_id="c" * 64,
            deployed_artifact_id="d" * 64,
            reason="rejected",
        )


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_name": "one", "schema_name": "two"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_selection_evidence(path)


def test_nested_metadata_is_immutable() -> None:
    evidence = bundle()

    with pytest.raises(TypeError, match="immutable"):
        evidence.metadata["new"] = "value"  # type: ignore[index]


def test_deployment_guard_rows_cannot_change_calibration_selection() -> None:
    first = bundle()
    second = build_selection_evidence_bundle(
        experiment_id=first.experiment_id,
        source_repository=first.source_repository,
        source_revision=first.source_revision,
        candidates=candidates(),
        calibration_rows=rows(),
        selection_rule=rule(),
        deployment_decisions=(
            decision("target-a", accepted=False, guard_value=99.0),
            decision("target-b", accepted=True, guard_value=-5.0),
        ),
        metadata={"split_registry_id": "f" * 64},
    )

    assert second.selection_order == first.selection_order
    assert second.selected_candidate_id == first.selected_candidate_id
    assert second.artifact_id != first.artifact_id


def test_unknown_top_level_field_is_rejected() -> None:
    payload = copy.deepcopy(bundle().to_dict())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown"):
        selection_evidence_from_dict(payload)


def test_written_json_is_canonical_and_contains_every_calibration_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    write_selection_evidence(bundle(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["calibration_rows"]) == 6
    assert payload["selection_order"] == [
        "persistent-gauge",
        "fallback",
        "overfit",
    ]
    assert payload["artifact_id"] == bundle().artifact_id
