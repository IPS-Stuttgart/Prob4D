from __future__ import annotations

import json

from prob4d.fresh_provider_conformance import (
    CONFORMANCE_CASES,
    build_fresh_provider_conformance_request,
    main,
    run_fresh_provider_conformance,
)


def test_corpus_covers_every_readiness_classification() -> None:
    expected = {
        "support-negative",
        "source-mean-negative",
        "identity-or-association-negative",
        "gauge-or-dependence-negative",
        "point-covariance-localized",
        "query-irrelevant-or-nonidentifiable",
        "ready-for-one-target-evaluation",
        "technical-failure",
    }

    assert {case.expected_classification for case in CONFORMANCE_CASES} == expected


def test_corpus_is_deterministic_and_passes() -> None:
    first = run_fresh_provider_conformance()
    second = run_fresh_provider_conformance()

    assert first.all_passed
    assert first.to_dict() == second.to_dict()
    assert first.fresh_provider_conformance_id == second.fresh_provider_conformance_id
    assert all(result.passed for result in first.results)


def test_only_clean_case_receives_target_authorization() -> None:
    report = run_fresh_provider_conformance()
    authorized = [result for result in report.results if result.authorization_id is not None]

    assert len(authorized) == 1
    assert authorized[0].case_id == "ready-for-one-target-evaluation"
    assert authorized[0].authorize_target_evaluation
    assert authorized[0].target_evaluation_budget == 1
    assert all(
        not result.authorize_target_evaluation and result.target_evaluation_budget == 0
        for result in report.results
        if result.case_id != "ready-for-one-target-evaluation"
    )


def test_only_point_covariance_case_authorizes_model_development() -> None:
    report = run_fresh_provider_conformance()
    authorized = [
        result
        for result in report.results
        if result.authorize_point_uncertainty_development
    ]

    assert [result.case_id for result in authorized] == ["point-covariance-localized"]


def test_later_gates_are_unevaluated_after_terminal_fixture() -> None:
    case = next(case for case in CONFORMANCE_CASES if case.case_id == "source-mean-negative")
    request = build_fresh_provider_conformance_request(case)

    statuses = [gate.status for gate in request.gates]
    assert statuses[:2] == ["pass", "fail"]
    assert statuses[2:] == ["not-evaluated"] * 5


def test_module_cli_emits_replay_complete_report(capsys) -> None:
    assert main(["--compact"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["all_passed"] is True
    assert payload["metadata"]["target_payloads_opened"] is False
    assert payload["metadata"]["target_outcomes_opened"] is False
    assert len(payload["results"]) == len(CONFORMANCE_CASES)
