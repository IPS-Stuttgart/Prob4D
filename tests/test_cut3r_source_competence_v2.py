from __future__ import annotations

from copy import deepcopy

import pytest

from prob4d.cut3r_comparison import build_cut3r_comparison_lock
from prob4d.cut3r_source_competence import build_cut3r_source_competence_lock
from prob4d.cut3r_source_competence_v2 import (
    CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS,
    CUT3R_SOURCE_COMPETENCE_V2_RECORDS_SCHEMA,
    build_cut3r_source_competence_v2_lock,
    build_cut3r_source_competence_v2_report,
    source_competence_gates_v2,
    validate_cut3r_source_competence_v2_report,
    write_cut3r_source_competence_v2_lock,
    write_cut3r_source_competence_v2_report,
)


def _case(case_id: str, digest_character: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "input_video_sha256": digest_character * 64,
        "input_video_byte_count": 100,
        "frame_start": 0,
        "frame_stop_exclusive": 2,
        "evaluation_frame_start": 0,
        "evaluation_frame_stop_exclusive": 2,
    }


def _comparison() -> dict[str, object]:
    return build_cut3r_comparison_lock(
        {
            "protocol_name": "common-support-test-v2",
            "provider_revision": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "prob4d_revision": "c" * 40,
            "prob4d_distribution_sha256": "d" * 64,
            "window_size": 2,
            "overlap": 1,
            "confidence_threshold": 1.0,
            "storage_dtype": "float32",
            "random_seeds": [7, 11],
            "groups": [
                {"group_id": "dev", "cases": [_case("dev-case", "1")]},
                {"group_id": "cal", "cases": [_case("cal-case", "2")]},
                {
                    "group_id": "source-a",
                    "cases": [_case("source-a-case", "3")],
                },
                {
                    "group_id": "source-b",
                    "cases": [_case("source-b-case", "4")],
                },
            ],
            "group_roles": {
                "development": ["dev"],
                "calibration": ["cal"],
                "source_evaluation": ["source-a", "source-b"],
            },
            "include_revisit_diagnostic": False,
        }
    )


def _source_policy(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "minimum_evaluable_groups": 2,
        "maximum_technical_failures": 0,
        "permitted_technical_failure_codes": [],
        "maximum_mean_proper_score_delta": 0.0,
        "maximum_mean_point_rmse_ratio": 1.0,
        "maximum_mean_endpoint_rmse_ratio": 1.0,
        "maximum_worst_group_point_rmse_ratio": 1.2,
        "maximum_mean_absolute_drift_slope_m_per_frame": 0.01,
        "maximum_mean_seam_rmse_m": 0.02,
        "minimum_mean_quality_group_pass_fraction": 1.0,
        "minimum_mean_association_precision": 0.7,
        "minimum_mean_identity_retention": 0.7,
        "minimum_mean_support_retention": 0.7,
        "minimum_identity_group_pass_fraction": 1.0,
    }
    result.update(overrides)
    return result


def _source_lock(
    comparison: dict[str, object],
    **policy_overrides: object,
) -> dict[str, object]:
    return build_cut3r_source_competence_lock(
        comparison,
        {
            "contrast_id": "prob4d-fusion-value",
            "candidate_provider_manifest_id": "e" * 64,
            "baseline_provider_manifest_id": "f" * 64,
            "cohort_binding_id": "1" * 64,
            "group_definition": "complete-object-v2-test",
            "record_definition_sha256": "2" * 64,
            "policy": _source_policy(**policy_overrides),
        },
    )


def _paired_policy(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "maximum_mean_seam_rmse_ratio": 1.0,
        "maximum_worst_group_seam_rmse_ratio": 1.2,
        "maximum_mean_absolute_drift_slope_ratio": 1.0,
        "maximum_worst_group_absolute_drift_slope_ratio": 1.2,
        "minimum_mean_association_precision_delta": -0.02,
        "minimum_worst_group_association_precision_delta": -0.05,
        "minimum_mean_identity_retention_delta": -0.02,
        "minimum_worst_group_identity_retention_delta": -0.05,
        "minimum_mean_support_retention_delta": -0.02,
        "minimum_worst_group_support_retention_delta": -0.05,
        "minimum_paired_quality_group_pass_fraction": 1.0,
        "minimum_paired_identity_group_pass_fraction": 1.0,
    }
    result.update(overrides)
    return result


def _v2_lock(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    **paired_overrides: object,
) -> dict[str, object]:
    return build_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        {
            "source_competence_policy": source_lock["policy"],
            "common_support_definition_sha256": "3" * 64,
            "proper_score_semantics": (
                CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS
            ),
            "paired_policy": _paired_policy(**paired_overrides),
            "require_complete_source_roster": True,
        },
    )


def _metric_support(frame: int) -> dict[str, object]:
    return {
        "point_support_sha256": ("4" if frame == 0 else "5") * 64,
        "point_support_count": 8,
        "endpoint_support_sha256": ("6" if frame == 0 else "7") * 64,
        "endpoint_support_count": 1,
        "proper_score_support_sha256": ("8" if frame == 0 else "9") * 64,
        "proper_score_dimension": 24,
        "proper_score_semantics": (
            CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS
        ),
        "seam_support_sha256": "a" * 64 if frame == 0 else None,
        "seam_support_count": 4 if frame == 0 else 0,
    }


def _record(
    group: str,
    case: str,
    frame: int,
    seed: int,
    arm: str,
) -> dict[str, object]:
    candidate = arm == "restarted-prob4d-fused"
    return {
        "group_id": group,
        "case_id": case,
        "frame_index": frame,
        "random_seed": seed,
        "arm_id": arm,
        "point_error_m": 0.8 if candidate else 1.0,
        "endpoint_error_m": 0.7 if candidate else 1.0,
        "proper_score": 8.0 if candidate else 10.0,
        "seam_error_m": 0.01 if frame == 0 else None,
        "association_correct_count": 9 if candidate else 8,
        "association_predicted_count": 10,
        "identity_retained_count": 9 if candidate else 8,
        "identity_reference_count": 10,
        "support_retained_count": 9 if candidate else 8,
        "support_reference_count": 10,
        "metric_support": _metric_support(frame),
    }


def _records(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    v2_lock: dict[str, object],
) -> dict[str, object]:
    rows = []
    for group, case in (
        ("source-a", "source-a-case"),
        ("source-b", "source-b-case"),
    ):
        for frame in range(2):
            for seed in (7, 11):
                for arm in ("restarted-prob4d-fused", "restarted-newest"):
                    rows.append(_record(group, case, frame, seed, arm))
    return {
        "schema": CUT3R_SOURCE_COMPETENCE_V2_RECORDS_SCHEMA,
        "schema_version": 2,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": v2_lock[
            "common_support_definition_sha256"
        ],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": [],
        "records": rows,
    }


def _setup() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    comparison = _comparison()
    source_lock = _source_lock(comparison)
    v2_lock = _v2_lock(comparison, source_lock)
    return comparison, source_lock, v2_lock, _records(comparison, source_lock, v2_lock)


def test_common_support_records_build_passing_paired_report() -> None:
    comparison, source_lock, v2_lock, records = _setup()

    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )

    assert report["source_competence_pass"]
    assert report["mean_quality_status"] == "pass"
    assert report["identity_reliability_status"] == "pass"
    assert report["aggregate"]["mean_seam_rmse_ratio"] == pytest.approx(1.0)
    assert report["groups"][0]["candidate"]["association_precision"] == pytest.approx(
        0.9
    )
    assert report["groups"][0]["baseline"]["association_precision"] == pytest.approx(
        0.8
    )
    assert report["v1_report"]["source_competence_pass"]


def test_point_support_mismatch_fails_before_scoring() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    baseline = next(
        row
        for row in records["records"]
        if row["arm_id"] == "restarted-newest"
    )
    baseline["metric_support"]["point_support_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="different exact metric support"):
        build_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
        )


def test_proper_score_dimension_mismatch_fails_before_scoring() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    baseline = next(
        row
        for row in records["records"]
        if row["arm_id"] == "restarted-newest"
    )
    baseline["metric_support"]["proper_score_dimension"] = 21

    with pytest.raises(ValueError, match="different exact metric support"):
        build_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
        )


def test_arm_specific_proper_score_semantics_are_rejected() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    records["records"][0]["metric_support"]["proper_score_semantics"] = (
        "arm-specific-predicted-covariance-v1"
    )

    with pytest.raises(ValueError, match="arm-neutral fixed-scale semantics"):
        build_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
        )


def test_absent_seam_requires_zero_count_and_null_digest() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    row = next(item for item in records["records"] if item["frame_index"] == 1)
    row["metric_support"]["seam_support_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="zero count and null digest"):
        build_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
        )


def test_paired_seam_regression_fails_even_when_v1_absolute_gate_passes() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    for row in records["records"]:
        if row["arm_id"] == "restarted-prob4d-fused" and row["frame_index"] == 0:
            row["seam_error_m"] = 0.019

    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )

    assert report["v1_report"]["mean_quality_status"] == "pass"
    assert report["mean_quality_status"] == "fail"
    assert "mean-seam-rmse-ratio-exceeded" in report["mean_quality_reasons"]
    assert not report["source_competence_pass"]


def test_paired_identity_regression_fails_after_mean_pass() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    for row in records["records"]:
        if row["arm_id"] == "restarted-prob4d-fused":
            row["association_correct_count"] = 7
            row["identity_retained_count"] = 7
            row["support_retained_count"] = 7
        else:
            row["association_correct_count"] = 9
            row["identity_retained_count"] = 9
            row["support_retained_count"] = 9

    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )

    assert report["v1_report"]["identity_reliability_status"] == "pass"
    assert report["mean_quality_status"] == "pass"
    assert report["identity_reliability_status"] == "fail"
    assert "mean-identity-retention-delta-below-minimum" in report[
        "identity_reliability_reasons"
    ]


def test_complete_roster_lock_rejects_permissive_v1_policy() -> None:
    comparison = _comparison()
    source_lock = _source_lock(
        comparison,
        minimum_evaluable_groups=1,
        maximum_technical_failures=1,
        permitted_technical_failure_codes=["gpu-oom"],
    )

    with pytest.raises(ValueError, match="complete source roster requires"):
        _v2_lock(comparison, source_lock)


def test_target_opening_is_rejected() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    records["target_outcomes_opened"] = True

    with pytest.raises(ValueError, match="may not open target outcomes"):
        build_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
        )


def test_readiness_adapter_does_not_evaluate_identity_after_mean_failure() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    for row in records["records"]:
        if row["arm_id"] == "restarted-prob4d-fused" and row["frame_index"] == 0:
            row["seam_error_m"] = 0.019
    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )

    mean_gate, identity_gate = source_competence_gates_v2(report)

    assert mean_gate.status == "fail"
    assert identity_gate.status == "not-evaluated"
    assert identity_gate.evidence_id is None


def test_readiness_adapter_rejects_report_identity_tampering() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )
    report["aggregate"]["mean_seam_rmse_ratio"] = 0.5

    with pytest.raises(ValueError, match="content identity changed"):
        source_competence_gates_v2(report)


def test_report_replay_rejects_tampering() -> None:
    comparison, source_lock, v2_lock, records = _setup()
    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )
    tampered = deepcopy(report)
    tampered["aggregate"]["mean_seam_rmse_ratio"] = 0.5

    with pytest.raises(ValueError, match="does not match the bound records"):
        validate_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
            tampered,
        )


def test_identical_lock_and_report_publication_is_idempotent(tmp_path) -> None:
    comparison, source_lock, v2_lock, records = _setup()
    report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )
    lock_path = tmp_path / "common-support-lock.json"
    report_path = tmp_path / "source-competence-v2.json"

    write_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        lock_path,
        v2_lock,
    )
    write_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        lock_path,
        v2_lock,
    )
    write_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
        report_path,
        report,
    )
    write_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
        report_path,
        report,
    )

    assert lock_path.is_file()
    assert report_path.is_file()
