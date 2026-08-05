from __future__ import annotations

from pathlib import Path

from prob4d.heldout_promotion import (
    HeldoutProviderPromotionReportV1,
    diagnose_heldout_promotion,
    load_promotion_diagnosis,
    main,
    write_promotion_diagnosis,
    write_promotion_report,
)


def _provider_decision(
    *,
    group_count_passed: bool = True,
    rules: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rule_values = [] if rules is None else rules
    return {
        "policy_id": "provider-gate-v1",
        "minimum_group_count": 3,
        "observed_group_count": 3 if group_count_passed else 2,
        "group_count_passed": group_count_passed,
        "rules": rule_values,
        "overall_passed": group_count_passed
        and all(rule.get("passed") is True for rule in rule_values),
    }


def _query_decision(**overrides: object) -> dict[str, object]:
    decision: dict[str, object] = {
        "minimum_target_group_count": 3,
        "observed_target_group_count": 3,
        "target_group_count_passed": True,
        "paired_bootstrap": {"ci95_upper": -1.0},
        "query_superiority_margin_mm": 0.25,
        "query_superiority_passed": True,
        "maximum_harmful_accepted_updates": 0,
        "observed_harmful_accepted_updates": 0,
        "harmful_accepted_updates_passed": True,
        "maximum_worst_group_regression_mm": 0.0,
        "observed_worst_group_regression_mm": -0.5,
        "worst_group_regression_passed": True,
        "maximum_technical_failures": 0,
        "observed_technical_failures": 0,
        "technical_failures_passed": True,
        "minimum_mean_accepted_coverage": 0.9,
        "observed_mean_accepted_coverage": 0.95,
        "accepted_coverage_passed": True,
        "exact_fallback_failure_count": 0,
        "exact_fallback_passed": True,
    }
    decision.update(overrides)
    gate_names = (
        "target_group_count_passed",
        "query_superiority_passed",
        "harmful_accepted_updates_passed",
        "worst_group_regression_passed",
        "technical_failures_passed",
        "accepted_coverage_passed",
        "exact_fallback_passed",
    )
    decision["overall_passed"] = all(decision[name] is True for name in gate_names)
    return decision


def _report(
    *,
    provider_decision: dict[str, object] | None = None,
    query_decision: dict[str, object] | None = None,
) -> HeldoutProviderPromotionReportV1:
    provider = _provider_decision() if provider_decision is None else provider_decision
    query = _query_decision() if query_decision is None else query_decision
    return HeldoutProviderPromotionReportV1(
        promotion_lock_id="a" * 64,
        query_results_id="b" * 64,
        provider_report_sha256="c" * 64,
        provider_evaluation_manifest_sha256="d" * 64,
        provider_audit={"structural_validation_passed": True},
        provider_decision=provider,
        query_aggregate={"identity": {}},
        query_decision=query,
        overall_passed=(
            provider["overall_passed"] is True
            and query["overall_passed"] is True
        ),
    )


def _failed_rule(rule_id: str, metric: str) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "candidate_method": "candidate",
        "metric": metric,
        "metric_path": f"metric.metrics.{metric}",
        "decision_bound": "ci95_upper",
        "decision_bound_value": 0.2,
        "pass_threshold": 0.0,
        "passed": False,
    }


def test_passing_diagnosis_round_trips(tmp_path: Path) -> None:
    diagnosis = diagnose_heldout_promotion(_report())
    output = tmp_path / "diagnosis.json"
    write_promotion_diagnosis(diagnosis, output)

    assert diagnosis.overall_passed is True
    assert diagnosis.boundary_ids == ("promotion_ready",)
    assert diagnosis.failed_provider_rule_ids == ()
    assert diagnosis.failed_query_gate_ids == ()
    assert load_promotion_diagnosis(output) == diagnosis


def test_failed_provider_rules_are_grouped_by_metric_family() -> None:
    provider = _provider_decision(
        rules=[
            _failed_rule("seam", "seam_error"),
            _failed_rule("identity", "association_retention"),
            _failed_rule("coverage", "coverage_95"),
        ]
    )

    diagnosis = diagnose_heldout_promotion(_report(provider_decision=provider))

    assert diagnosis.overall_passed is False
    assert diagnosis.failed_provider_rule_ids == ("seam", "identity", "coverage")
    assert diagnosis.boundary_ids == (
        "gauge_consistency",
        "identity_persistence",
        "uncertainty_calibration",
    )


def test_failed_guarded_query_localizes_safety_and_identifiability() -> None:
    query = _query_decision(
        paired_bootstrap={"ci95_upper": 0.5},
        query_superiority_passed=False,
        observed_harmful_accepted_updates=2,
        harmful_accepted_updates_passed=False,
        observed_worst_group_regression_mm=0.5,
        worst_group_regression_passed=False,
    )

    diagnosis = diagnose_heldout_promotion(_report(query_decision=query))

    assert diagnosis.failed_query_gate_ids == (
        "harmful_accepted_updates_passed",
        "worst_group_regression_passed",
        "query_superiority_passed",
    )
    assert diagnosis.boundary_ids == (
        "guard_calibration",
        "object_session_transfer",
        "query_identifiability_or_physical_model_discrepancy",
    )


def test_diagnose_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    report_path = tmp_path / "promotion-report.json"
    diagnosis_path = tmp_path / "promotion-diagnosis.json"
    markdown_path = tmp_path / "promotion-diagnosis.md"
    write_promotion_report(_report(), report_path)

    assert (
        main(
            [
                "diagnose",
                str(report_path),
                "--output",
                str(diagnosis_path),
                "--markdown",
                str(markdown_path),
            ]
        )
        == 0
    )
    assert load_promotion_diagnosis(diagnosis_path).overall_passed is True
    assert "promotion_ready" in markdown_path.read_text(encoding="utf-8")
