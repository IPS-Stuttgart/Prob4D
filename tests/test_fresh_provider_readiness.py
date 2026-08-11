from __future__ import annotations

import json
from dataclasses import replace

import pytest

from prob4d.fresh_provider_readiness import (
    FreshProviderCohortLockV1,
    FreshProviderReadinessRequestV1,
    ReadinessGateV1,
    authorize_fresh_provider_target,
    evaluate_fresh_provider_readiness,
    load_fresh_provider_readiness_decision,
    load_fresh_provider_readiness_request,
    load_fresh_provider_target_authorization,
    source_competence_gates,
    unevaluated_gate,
    write_fresh_provider_readiness_decision,
    write_fresh_provider_readiness_request,
    write_fresh_provider_target_authorization,
)
from prob4d.source_provider_competence import (
    SourceProviderCompetencePolicyV1,
    SourceProviderCompetenceReportV1,
    SourceProviderGroupResultV1,
)

GATE_NAMES = (
    "support-feasibility",
    "source-mean",
    "identity-reliability",
    "gauge-dependence",
    "point-covariance",
    "query-relevance",
    "exact-fallback",
)


def _lock() -> FreshProviderCohortLockV1:
    return FreshProviderCohortLockV1(
        protocol_id="fresh-provider-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        provider_repository="example/provider",
        provider_revision="b" * 40,
        model_set_id="1" * 64,
        loader_id="2" * 64,
        cohort_binding_id="3" * 64,
        promotion_lock_id="4" * 64,
        query_definition_id="5" * 64,
        fallback_identity_id="6" * 64,
        development_group_ids=("dev-1", "dev-2"),
        calibration_group_ids=("cal-1", "cal-2"),
        target_group_ids=("target-1", "target-2"),
        confirmation_group_ids=("confirm-1",),
    )


def _pass(name: str, digit: str) -> ReadinessGateV1:
    return ReadinessGateV1(
        gate_name=name,
        status="pass",
        evidence_id=digit * 64,
    )


def _request(gates: tuple[ReadinessGateV1, ...]) -> FreshProviderReadinessRequestV1:
    return FreshProviderReadinessRequestV1(cohort_lock=_lock(), gates=gates)


def _all_pass() -> tuple[ReadinessGateV1, ...]:
    return tuple(_pass(name, str(index + 1)) for index, name in enumerate(GATE_NAMES))


def _terminal_request(
    index: int,
    *,
    status: str = "fail",
) -> FreshProviderReadinessRequestV1:
    gates = []
    for position, name in enumerate(GATE_NAMES):
        if position < index:
            gates.append(_pass(name, str(position + 1)))
        elif position == index:
            gates.append(
                ReadinessGateV1(
                    gate_name=name,
                    status=status,
                    evidence_id="f" * 64,
                    reason_codes=("registered-failure",),
                )
            )
        else:
            gates.append(unevaluated_gate(name))
    return _request(tuple(gates))


def test_all_pass_authorizes_exactly_one_target_evaluation() -> None:
    decision = evaluate_fresh_provider_readiness(_request(_all_pass()))

    assert decision.classification == "ready-for-one-target-evaluation"
    assert decision.authorize_target_evaluation
    assert not decision.authorize_point_uncertainty_development
    assert decision.target_evaluation_budget == 1

    authorization = authorize_fresh_provider_target(decision)
    assert authorization.target_group_ids == ("target-1", "target-2")
    assert authorization.target_evaluation_budget == 1


@pytest.mark.parametrize(
    ("index", "classification"),
    (
        (0, "support-negative"),
        (1, "source-mean-negative"),
        (2, "identity-or-association-negative"),
        (3, "gauge-or-dependence-negative"),
        (4, "point-covariance-localized"),
        (5, "query-irrelevant-or-nonidentifiable"),
        (6, "technical-failure"),
    ),
)
def test_each_gate_has_one_terminal_classification(
    index: int,
    classification: str,
) -> None:
    decision = evaluate_fresh_provider_readiness(_terminal_request(index))

    assert decision.classification == classification
    assert decision.authorize_point_uncertainty_development == (
        classification == "point-covariance-localized"
    )
    assert not decision.authorize_target_evaluation
    assert decision.target_evaluation_budget == 0


@pytest.mark.parametrize("index", range(len(GATE_NAMES)))
def test_technical_failure_stops_at_exact_stage(index: int) -> None:
    decision = evaluate_fresh_provider_readiness(
        _terminal_request(index, status="technical-failure")
    )

    assert decision.classification == "technical-failure"
    assert decision.terminal_gate == GATE_NAMES[index]


def test_downstream_evidence_after_terminal_failure_is_rejected() -> None:
    gates = list(_terminal_request(1).gates)
    gates[2] = _pass("identity-reliability", "9")
    with pytest.raises(ValueError, match="must remain not-evaluated"):
        evaluate_fresh_provider_readiness(_request(tuple(gates)))


def test_missing_gate_after_pass_is_not_silently_interpreted() -> None:
    gates = list(_all_pass())
    gates[3] = unevaluated_gate("gauge-dependence")
    with pytest.raises(ValueError, match="unevaluated"):
        evaluate_fresh_provider_readiness(_request(tuple(gates)))


def test_rosters_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        replace(_lock(), target_group_ids=("dev-1", "target-2"))


def _source_report(*, point_rmse: float = 0.9, association: float = 0.95):
    policy = SourceProviderCompetencePolicyV1(
        minimum_evaluable_groups=2,
        maximum_technical_failures=0,
        permitted_technical_failure_codes=(),
        maximum_mean_proper_score_delta=0.0,
        maximum_mean_point_rmse_ratio=1.0,
        maximum_mean_endpoint_rmse_ratio=1.0,
        maximum_worst_group_point_rmse_ratio=1.1,
        maximum_mean_absolute_drift_slope_m_per_frame=0.02,
        maximum_mean_seam_rmse_m=0.03,
        minimum_mean_quality_group_pass_fraction=0.5,
        minimum_mean_association_precision=0.9,
        minimum_mean_identity_retention=0.8,
        minimum_mean_support_retention=0.85,
        minimum_identity_group_pass_fraction=0.5,
    )

    def group(group_id: str) -> SourceProviderGroupResultV1:
        return SourceProviderGroupResultV1(
            group_id=group_id,
            candidate_proper_score=9.0,
            baseline_proper_score=10.0,
            candidate_point_rmse_m=point_rmse,
            baseline_point_rmse_m=1.0,
            candidate_endpoint_rmse_m=0.8,
            baseline_endpoint_rmse_m=1.0,
            absolute_drift_slope_m_per_frame=0.01,
            seam_rmse_m=0.02,
            association_precision=association,
            identity_retention=0.9,
            support_retention=0.95,
        )

    return SourceProviderCompetenceReportV1(
        provider_manifest_id="a" * 64,
        cohort_binding_id="b" * 64,
        group_definition="complete-object-v1",
        policy=policy,
        groups=(group("source-a"), group("source-b")),
    )


def test_source_report_adapter_does_not_evaluate_identity_after_mean_failure() -> None:
    mean_gate, identity_gate = source_competence_gates(_source_report(point_rmse=1.4))

    assert mean_gate.status == "fail"
    assert identity_gate.status == "not-evaluated"


def test_source_report_adapter_retains_identity_failure() -> None:
    mean_gate, identity_gate = source_competence_gates(_source_report(association=0.5))

    assert mean_gate.status == "pass"
    assert identity_gate.status == "fail"


def test_round_trip_replays_request_decision_and_authorization(tmp_path) -> None:
    request = _request(_all_pass())
    decision = evaluate_fresh_provider_readiness(request)
    authorization = authorize_fresh_provider_target(decision)
    request_path = tmp_path / "request.json"
    decision_path = tmp_path / "decision.json"
    authorization_path = tmp_path / "authorization.json"

    write_fresh_provider_readiness_request(request_path, request)
    write_fresh_provider_readiness_decision(decision_path, decision)
    write_fresh_provider_target_authorization(authorization_path, authorization)

    assert load_fresh_provider_readiness_request(request_path).to_dict() == request.to_dict()
    assert load_fresh_provider_readiness_decision(decision_path).to_dict() == decision.to_dict()
    assert (
        load_fresh_provider_target_authorization(authorization_path).to_dict()
        == authorization.to_dict()
    )

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload["target_evaluation_budget"] = 0
    decision_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization changed"):
        load_fresh_provider_readiness_decision(decision_path)
