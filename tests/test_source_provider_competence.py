from __future__ import annotations

import json

import pytest

from prob4d.source_provider_competence import (
    SourceProviderCompetencePolicyV1,
    SourceProviderCompetenceReportV1,
    SourceProviderGroupResultV1,
    load_source_provider_competence,
    write_source_provider_competence,
)


def _policy(**overrides: object) -> SourceProviderCompetencePolicyV1:
    values: dict[str, object] = {
        "minimum_evaluable_groups": 2,
        "maximum_technical_failures": 0,
        "permitted_technical_failure_codes": (),
        "maximum_mean_proper_score_delta": 0.0,
        "maximum_mean_point_rmse_ratio": 1.0,
        "maximum_mean_endpoint_rmse_ratio": 1.0,
        "maximum_worst_group_point_rmse_ratio": 1.1,
        "maximum_mean_absolute_drift_slope_m_per_frame": 0.02,
        "maximum_mean_seam_rmse_m": 0.03,
        "minimum_mean_quality_group_pass_fraction": 0.5,
        "minimum_mean_association_precision": 0.9,
        "minimum_mean_identity_retention": 0.8,
        "minimum_mean_support_retention": 0.85,
        "minimum_identity_group_pass_fraction": 0.5,
    }
    values.update(overrides)
    return SourceProviderCompetencePolicyV1(**values)


def _group(group_id: str, **overrides: object) -> SourceProviderGroupResultV1:
    values: dict[str, object] = {
        "group_id": group_id,
        "candidate_proper_score": 9.0,
        "baseline_proper_score": 10.0,
        "candidate_point_rmse_m": 0.9,
        "baseline_point_rmse_m": 1.0,
        "candidate_endpoint_rmse_m": 0.8,
        "baseline_endpoint_rmse_m": 1.0,
        "absolute_drift_slope_m_per_frame": 0.01,
        "seam_rmse_m": 0.02,
        "association_precision": 0.95,
        "identity_retention": 0.9,
        "support_retention": 0.95,
    }
    values.update(overrides)
    return SourceProviderGroupResultV1(**values)


def _failure(group_id: str, code: str) -> SourceProviderGroupResultV1:
    return SourceProviderGroupResultV1(
        group_id=group_id,
        candidate_proper_score=None,
        baseline_proper_score=None,
        candidate_point_rmse_m=None,
        baseline_point_rmse_m=None,
        candidate_endpoint_rmse_m=None,
        baseline_endpoint_rmse_m=None,
        absolute_drift_slope_m_per_frame=None,
        seam_rmse_m=None,
        association_precision=None,
        identity_retention=None,
        support_retention=None,
        technical_failure_code=code,
    )


def _report(
    groups: tuple[SourceProviderGroupResultV1, ...],
    *,
    policy: SourceProviderCompetencePolicyV1 | None = None,
) -> SourceProviderCompetenceReportV1:
    return SourceProviderCompetenceReportV1(
        provider_manifest_id="1" * 64,
        cohort_binding_id="2" * 64,
        group_definition="complete-physical-object-v1",
        policy=_policy() if policy is None else policy,
        groups=groups,
        metadata={"split": "source-only"},
    )


def test_passing_report_separates_mean_and_identity_gates() -> None:
    report = _report((_group("object-b"), _group("object-a")))

    assert tuple(group.group_id for group in report.groups) == ("object-a", "object-b")
    assert report.mean_quality_status == "pass"
    assert report.identity_reliability_status == "pass"
    assert report.source_competence_pass
    assert report.mean_point_rmse_ratio == pytest.approx(0.9)
    assert report.mean_endpoint_rmse_ratio == pytest.approx(0.8)
    assert report.mean_quality_group_pass_fraction == 1.0
    assert report.identity_group_pass_fraction == 1.0
    assert len(report.source_provider_competence_id) == 64


def test_mean_failure_is_not_relabelled_as_covariance_failure() -> None:
    report = _report(
        (
            _group("object-a", candidate_point_rmse_m=1.4),
            _group("object-b", candidate_point_rmse_m=1.3),
        )
    )

    assert report.mean_quality_status == "fail"
    assert "mean-point-rmse-regression" in report.mean_quality_reasons
    assert report.identity_reliability_status == "pass"
    assert not report.source_competence_pass


def test_identity_failure_remains_separate_from_mean_quality() -> None:
    report = _report(
        (
            _group("object-a", association_precision=0.5),
            _group("object-b", identity_retention=0.4),
        )
    )

    assert report.mean_quality_status == "pass"
    assert report.identity_reliability_status == "fail"
    assert "association-precision-below-minimum" in report.identity_reliability_reasons
    assert "identity-retention-below-minimum" in report.identity_reliability_reasons


def test_permitted_technical_failure_is_retained_but_not_scored() -> None:
    policy = _policy(
        maximum_technical_failures=1,
        permitted_technical_failure_codes=("gpu-oom",),
    )
    report = _report(
        (_group("object-a"), _group("object-b"), _failure("object-c", "gpu-oom")),
        policy=policy,
    )

    assert report.group_count == 3
    assert report.technical_failure_count == 1
    assert report.evaluable_group_count == 2
    assert report.technical_integrity_pass
    assert report.source_competence_pass


def test_unpermitted_technical_failure_is_a_technical_terminal() -> None:
    report = _report(
        (_group("object-a"), _group("object-b"), _failure("object-c", "gpu-oom"))
    )

    assert report.mean_quality_status == "technical-failure"
    assert report.identity_reliability_status == "technical-failure"
    assert "unpermitted-technical-failure-code" in report.mean_quality_reasons
    assert not report.source_competence_pass


def test_target_opening_is_rejected() -> None:
    with pytest.raises(ValueError, match="target closed"):
        SourceProviderCompetenceReportV1(
            provider_manifest_id="1" * 64,
            cohort_binding_id="2" * 64,
            group_definition="complete-physical-object-v1",
            policy=_policy(),
            groups=(_group("object-a"), _group("object-b")),
            target_payloads_opened=True,
        )


def test_technical_failure_group_cannot_mix_metrics() -> None:
    with pytest.raises(ValueError, match="must not contain scored metrics"):
        _group("object-a", technical_failure_code="gpu-oom")


def test_round_trip_replays_all_derived_fields(tmp_path) -> None:
    report = _report((_group("object-a"), _group("object-b")))
    path = tmp_path / "report.json"
    write_source_provider_competence(path, report)

    loaded = load_source_provider_competence(path)
    assert loaded.to_dict() == report.to_dict()
    with pytest.raises(FileExistsError):
        write_source_provider_competence(path, report)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean_point_rmse_ratio"] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="derived fields changed"):
        load_source_provider_competence(path)
