from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.joint_covariance_metrics import (
    JOINT_COVARIANCE_CLAIM_BOUNDARY,
    JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA,
    JOINT_COVARIANCE_DIAGNOSTIC_VERSION,
)
from prob4d.source_covariance_localization import (
    SourceCovarianceLocalizationPolicyV1,
    load_source_covariance_localization,
    localize_source_covariance,
    write_source_covariance_localization,
)
from prob4d.source_provider_competence import (
    SourceProviderCompetencePolicyV1,
    SourceProviderCompetenceReportV1,
    SourceProviderGroupResultV1,
)


def _source_policy() -> SourceProviderCompetencePolicyV1:
    return SourceProviderCompetencePolicyV1(
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


def _source_group(group_id: str, **changes: object) -> SourceProviderGroupResultV1:
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
    values.update(changes)
    return SourceProviderGroupResultV1(**values)  # type: ignore[arg-type]


def _source_report(**group_a_changes: object) -> SourceProviderCompetenceReportV1:
    return SourceProviderCompetenceReportV1(
        provider_manifest_id="1" * 64,
        cohort_binding_id="2" * 64,
        group_definition="complete-object-v1",
        policy=_source_policy(),
        groups=(
            _source_group("object-a", **group_a_changes),
            _source_group("object-b"),
        ),
    )


def _policy() -> SourceCovarianceLocalizationPolicyV1:
    return SourceCovarianceLocalizationPolicyV1(
        minimum_group_count=2,
        normalized_nees_lower=0.5,
        normalized_nees_upper=1.5,
        minimum_joint_pass_fraction=1.0,
        shared_energy_lower=0.5,
        shared_energy_upper=1.5,
        minimum_shared_pass_fraction=1.0,
        conditional_energy_lower=0.5,
        conditional_energy_upper=1.5,
        minimum_conditional_pass_fraction=1.0,
    )


def _joint(
    *,
    shared_a: float = 1.0,
    shared_b: float = 1.0,
    conditional_a: float = 1.0,
    conditional_b: float = 1.0,
    nees_a: float = 1.0,
    nees_b: float = 1.0,
) -> dict[str, object]:
    return {
        "schema_name": JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA,
        "schema_version": JOINT_COVARIANCE_DIAGNOSTIC_VERSION,
        "source_path": "/sealed/source.npz",
        "source_sha256": "3" * 64,
        "evaluation": {
            "groups": [
                {
                    "factor_group_id": "object-a",
                    "normalized_nees": nees_a,
                    "shared_subspace_normalized_energy": shared_a,
                    "conditional_subspace_normalized_energy": conditional_a,
                },
                {
                    "factor_group_id": "object-b",
                    "normalized_nees": nees_b,
                    "shared_subspace_normalized_energy": shared_b,
                    "conditional_subspace_normalized_energy": conditional_b,
                },
            ]
        },
        "claim_boundary": JOINT_COVARIANCE_CLAIM_BOUNDARY,
    }


def _localize(report: SourceProviderCompetenceReportV1, joint: dict[str, object]):
    return localize_source_covariance(
        report,
        joint,
        joint_diagnostic_sha256="4" * 64,
        policy=_policy(),
    )


def test_shared_failure_redirects_without_authorizing_point_model() -> None:
    result = _localize(_source_report(), _joint(shared_a=3.0))

    assert result.classification == "gauge-or-dependence-negative"
    assert not result.authorize_point_uncertainty_development
    gauge, point = result.readiness_gates()
    assert gauge.status == "fail"
    assert point.status == "not-evaluated"


def test_conditional_failure_is_the_only_point_model_authorization() -> None:
    result = _localize(_source_report(), _joint(conditional_b=3.0))

    assert result.classification == "point-covariance-localized"
    assert result.authorize_point_uncertainty_development
    gauge, point = result.readiness_gates()
    assert gauge.status == "pass"
    assert point.status == "fail"
    assert point.evidence_id == result.source_covariance_localization_id


def test_adequate_subspaces_pass_both_covariance_gates() -> None:
    result = _localize(_source_report(), _joint())

    assert result.classification == "covariance-adequate"
    assert not result.authorize_point_uncertainty_development
    gauge, point = result.readiness_gates()
    assert gauge.status == point.status == "pass"


def test_mean_or_identity_failure_stops_before_covariance_gates() -> None:
    mean_negative = _localize(
        _source_report(candidate_point_rmse_m=2.0),
        _joint(),
    )
    assert mean_negative.classification == "source-mean-negative"
    assert all(gate.status == "not-evaluated" for gate in mean_negative.readiness_gates())

    identity_negative = _localize(
        _source_report(association_precision=0.1),
        _joint(),
    )
    assert identity_negative.classification == "identity-or-association-negative"
    assert all(gate.status == "not-evaluated" for gate in identity_negative.readiness_gates())


def test_joint_groups_must_match_exact_evaluable_source_units() -> None:
    joint = _joint()
    groups = joint["evaluation"]["groups"]  # type: ignore[index]
    groups[1]["factor_group_id"] = "object-c"  # type: ignore[index]
    with pytest.raises(ValueError, match="do not match evaluable source"):
        _localize(_source_report(), joint)


def test_localization_round_trip_replays_derived_decision(tmp_path: Path) -> None:
    result = _localize(_source_report(), _joint(conditional_a=2.0))
    path = tmp_path / "localization.json"
    write_source_covariance_localization(path, result)
    assert load_source_covariance_localization(path).to_dict() == result.to_dict()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authorize_point_uncertainty_development"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch|derived fields changed"):
        load_source_covariance_localization(path)
