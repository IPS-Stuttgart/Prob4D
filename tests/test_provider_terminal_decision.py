from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.provider_terminal_decision import (
    CLASSIFICATION_BATCH_INCOMPATIBLE,
    CLASSIFICATION_COMPLETED_POSITIVE,
    CLASSIFICATION_SCIENTIFIC_NEGATIVE,
    CLASSIFICATION_SUPPORT_NEGATIVE,
    CLASSIFICATION_TECHNICAL_FAILURE,
    ProviderTerminalDecisionV1,
    build_provider_terminal_decision,
    load_provider_terminal_decision,
    main,
    write_provider_terminal_decision,
)

INFRASTRUCTURE_FORBIDDEN = (
    "provider-competence",
    "provider-calibration",
    "bayesian-phystwin-benefit",
    "causal4d-intervention-benefit",
    "deployment-safety",
    "state-of-the-art",
)


def _batch_negative() -> ProviderTerminalDecisionV1:
    return ProviderTerminalDecisionV1(
        protocol_id="provider-v7-source",
        provider_manifest_id="a" * 64,
        classification=CLASSIFICATION_BATCH_INCOMPATIBLE,
        failed_stage="source-batch-preflight",
        source_payloads_accessed=True,
        source_outcomes_accessed=False,
        target_payloads_accessed=False,
        target_outcomes_accessed=False,
        rerun_authorized=False,
        successor_protocol_required=True,
        evidence_ids=("b" * 64,),
        authorized_inferences=(),
        forbidden_inferences=INFRASTRUCTURE_FORBIDDEN,
        summary="The admitted source payloads cannot form the frozen scorer batch.",
        metadata={"future_prediction_payloads_opened": 0},
    )


def test_infrastructure_negative_roundtrips_and_is_idempotent(tmp_path: Path) -> None:
    decision = _batch_negative()
    path = tmp_path / "terminal.json"

    write_provider_terminal_decision(path, decision)
    write_provider_terminal_decision(path, decision)
    loaded = load_provider_terminal_decision(path)

    assert loaded == decision
    assert loaded.scientific_result is False
    assert loaded.authorized_inferences == ()
    assert loaded.target_outcomes_accessed is False


def test_infrastructure_negative_cannot_authorize_provider_competence() -> None:
    with pytest.raises(ValueError, match="authorize no scientific inference"):
        ProviderTerminalDecisionV1(
            protocol_id="provider-v7-source",
            provider_manifest_id="a" * 64,
            classification=CLASSIFICATION_TECHNICAL_FAILURE,
            failed_stage="source-scorer",
            source_payloads_accessed=True,
            source_outcomes_accessed=False,
            target_payloads_accessed=False,
            target_outcomes_accessed=False,
            rerun_authorized=False,
            successor_protocol_required=True,
            evidence_ids=("b" * 64,),
            authorized_inferences=("provider-competence",),
            forbidden_inferences=INFRASTRUCTURE_FORBIDDEN,
            summary="The scorer terminated before an outcome was computed.",
        )


def test_infrastructure_negative_requires_complete_nonclaim_boundary() -> None:
    with pytest.raises(ValueError, match="omits forbidden claims"):
        ProviderTerminalDecisionV1(
            protocol_id="provider-v7-source",
            provider_manifest_id="a" * 64,
            classification=CLASSIFICATION_BATCH_INCOMPATIBLE,
            failed_stage="source-batch-preflight",
            source_payloads_accessed=True,
            source_outcomes_accessed=False,
            target_payloads_accessed=False,
            target_outcomes_accessed=False,
            rerun_authorized=False,
            successor_protocol_required=True,
            evidence_ids=("b" * 64,),
            authorized_inferences=(),
            forbidden_inferences=("provider-competence",),
            summary="The admitted payloads are incompatible.",
        )


def test_support_negative_must_precede_payload_access() -> None:
    with pytest.raises(ValueError, match="precede payload access"):
        ProviderTerminalDecisionV1(
            protocol_id="provider-v7-support",
            provider_manifest_id="a" * 64,
            classification=CLASSIFICATION_SUPPORT_NEGATIVE,
            failed_stage="support-feasibility",
            source_payloads_accessed=True,
            source_outcomes_accessed=False,
            target_payloads_accessed=False,
            target_outcomes_accessed=False,
            rerun_authorized=False,
            successor_protocol_required=True,
            evidence_ids=("b" * 64,),
            authorized_inferences=("provider-support-negative",),
            forbidden_inferences=INFRASTRUCTURE_FORBIDDEN,
            summary="The frozen stream roster is not geometrically supported.",
        )


def test_scientific_negative_requires_outcome_evidence() -> None:
    with pytest.raises(ValueError, match="requires opened outcome evidence"):
        ProviderTerminalDecisionV1(
            protocol_id="provider-v7-source",
            provider_manifest_id="a" * 64,
            classification=CLASSIFICATION_SCIENTIFIC_NEGATIVE,
            failed_stage="source-competence",
            source_payloads_accessed=True,
            source_outcomes_accessed=False,
            target_payloads_accessed=False,
            target_outcomes_accessed=False,
            rerun_authorized=False,
            successor_protocol_required=False,
            evidence_ids=("b" * 64,),
            authorized_inferences=("provider-source-negative",),
            forbidden_inferences=("provider-target-competence",),
            summary="No outcome evidence was actually opened.",
        )


def test_opened_target_result_cannot_authorize_rerun() -> None:
    with pytest.raises(ValueError, match="cannot authorize a rerun"):
        ProviderTerminalDecisionV1(
            protocol_id="provider-v7-target",
            provider_manifest_id="a" * 64,
            classification=CLASSIFICATION_COMPLETED_POSITIVE,
            failed_stage=None,
            source_payloads_accessed=True,
            source_outcomes_accessed=True,
            target_payloads_accessed=True,
            target_outcomes_accessed=True,
            rerun_authorized=True,
            successor_protocol_required=False,
            evidence_ids=("b" * 64,),
            authorized_inferences=("provider-target-competence",),
            forbidden_inferences=("deployment-safety", "state-of-the-art"),
            summary="The frozen target gate passed.",
        )


def test_build_from_spec_and_cli_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    decision = _batch_negative()
    specification = decision.to_record(include_artifact_id=False)
    specification_path = tmp_path / "specification.json"
    specification_path.write_text(
        json.dumps(specification, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rebuilt = build_provider_terminal_decision(specification_path)
    assert rebuilt.artifact_id == decision.artifact_id

    output = tmp_path / "terminal.json"
    assert main(["build", str(specification_path), str(output)]) == 0
    build_summary = json.loads(capsys.readouterr().out)
    assert build_summary["scientific_result"] is False
    assert build_summary["authorized_inferences"] == []

    assert main(["verify", str(output)]) == 0
    verify_summary = json.loads(capsys.readouterr().out)
    assert verify_summary["artifact_id"] == decision.artifact_id


def test_tampered_decision_fails_content_identity(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    write_provider_terminal_decision(path, _batch_negative())
    record = json.loads(path.read_text(encoding="utf-8"))
    record["summary"] = "changed after sealing"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact ID mismatch"):
        load_provider_terminal_decision(path)
