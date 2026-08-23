from __future__ import annotations

from copy import deepcopy

import pytest

from prob4d.cut3r_comparison import build_cut3r_comparison_lock
from prob4d.cut3r_recurrent_state_recovery import (
    build_cut3r_recurrent_state_recovery_report,
)
from prob4d.cut3r_source_competence import build_cut3r_source_competence_lock
from prob4d.cut3r_source_competence_v2 import (
    CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS,
    CUT3R_SOURCE_COMPETENCE_V2_RECORDS_SCHEMA,
    build_cut3r_source_competence_v2_lock,
    build_cut3r_source_competence_v2_report,
)
from prob4d.prediction_cli import main as prediction_main


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
            "protocol_name": "recurrent-state-recovery-test-v1",
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


def _source_policy() -> dict[str, object]:
    return {
        "minimum_evaluable_groups": 2,
        "maximum_technical_failures": 0,
        "permitted_technical_failure_codes": [],
        "maximum_mean_proper_score_delta": 0.0,
        "maximum_mean_point_rmse_ratio": 1.0,
        "maximum_mean_endpoint_rmse_ratio": 1.0,
        "maximum_worst_group_point_rmse_ratio": 1.2,
        "maximum_mean_absolute_drift_slope_m_per_frame": 2.0,
        "maximum_mean_seam_rmse_m": 2.0,
        "minimum_mean_quality_group_pass_fraction": 1.0,
        "minimum_mean_association_precision": 0.5,
        "minimum_mean_identity_retention": 0.5,
        "minimum_mean_support_retention": 0.5,
        "minimum_identity_group_pass_fraction": 1.0,
    }


def _source_lock(
    comparison: dict[str, object],
    *,
    contrast_id: str,
    digest_character: str,
) -> dict[str, object]:
    return build_cut3r_source_competence_lock(
        comparison,
        {
            "contrast_id": contrast_id,
            "candidate_provider_manifest_id": digest_character * 64,
            "baseline_provider_manifest_id": "f" * 64,
            "cohort_binding_id": "1" * 64,
            "group_definition": "complete-object-recovery-test",
            "record_definition_sha256": (
                "2" if contrast_id == "prob4d-fusion-value" else "3"
            )
            * 64,
            "policy": _source_policy(),
        },
    )


def _paired_policy() -> dict[str, object]:
    return {
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


def _v2_lock(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    digest_character: str,
) -> dict[str, object]:
    return build_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        {
            "source_competence_policy": source_lock["policy"],
            "common_support_definition_sha256": digest_character * 64,
            "proper_score_semantics": (
                CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS
            ),
            "paired_policy": _paired_policy(),
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


def _arm_values(arm: str, frame: int) -> tuple[float, float, float, float | None, int]:
    if arm == "restarted-newest":
        return (
            1.0 if frame == 0 else 2.0,
            1.0,
            10.0,
            1.0 if frame == 0 else None,
            8,
        )
    if arm == "restarted-prob4d-fused":
        return (
            1.0 if frame == 0 else 1.7,
            0.8,
            8.0,
            0.8 if frame == 0 else None,
            9,
        )
    if arm == "native-continuous":
        return (
            1.0 if frame == 0 else 1.4,
            0.6,
            6.0,
            0.6 if frame == 0 else None,
            9,
        )
    raise AssertionError(arm)


def _record(
    group: str,
    case: str,
    frame: int,
    seed: int,
    arm: str,
) -> dict[str, object]:
    point, endpoint, score, seam, retained = _arm_values(arm, frame)
    return {
        "group_id": group,
        "case_id": case,
        "frame_index": frame,
        "random_seed": seed,
        "arm_id": arm,
        "point_error_m": point,
        "endpoint_error_m": endpoint,
        "proper_score": score,
        "seam_error_m": seam,
        "association_correct_count": retained,
        "association_predicted_count": 10,
        "identity_retained_count": retained,
        "identity_reference_count": 10,
        "support_retained_count": retained,
        "support_reference_count": 10,
        "metric_support": _metric_support(frame),
    }


def _records(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    v2_lock: dict[str, object],
    *,
    candidate_arm: str,
) -> dict[str, object]:
    rows = []
    for group, case in (
        ("source-a", "source-a-case"),
        ("source-b", "source-b-case"),
    ):
        for frame in range(2):
            for seed in (7, 11):
                for arm in (candidate_arm, "restarted-newest"):
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


def _setup() -> tuple[object, ...]:
    comparison = _comparison()
    fusion_source = _source_lock(
        comparison,
        contrast_id="prob4d-fusion-value",
        digest_character="e",
    )
    recurrence_source = _source_lock(
        comparison,
        contrast_id="provider-recurrence-value",
        digest_character="d",
    )
    fusion_support = _v2_lock(comparison, fusion_source, "b")
    recurrence_support = _v2_lock(comparison, recurrence_source, "c")
    fusion_records = _records(
        comparison,
        fusion_source,
        fusion_support,
        candidate_arm="restarted-prob4d-fused",
    )
    recurrence_records = _records(
        comparison,
        recurrence_source,
        recurrence_support,
        candidate_arm="native-continuous",
    )
    fusion_report = build_cut3r_source_competence_v2_report(
        comparison,
        fusion_source,
        fusion_support,
        fusion_records,
    )
    recurrence_report = build_cut3r_source_competence_v2_report(
        comparison,
        recurrence_source,
        recurrence_support,
        recurrence_records,
    )
    specification = {
        "bootstrap_seed": 20260823,
        "bootstrap_replicates": 200,
        "confidence_level": 0.95,
        "minimum_recurrence_gap_by_metric": {
            "point_rmse_m": 1e-12,
            "endpoint_rmse_m": 1e-12,
            "proper_score": 1e-12,
            "seam_rmse_m": 1e-12,
            "absolute_drift_slope_m_per_frame": 1e-12,
        },
        "minimum_valid_bootstrap_fraction": 0.8,
    }
    return (
        comparison,
        fusion_source,
        fusion_support,
        fusion_records,
        fusion_report,
        recurrence_source,
        recurrence_support,
        recurrence_records,
        recurrence_report,
        specification,
    )


def test_recovery_report_quantifies_three_arm_mechanism() -> None:
    report = build_cut3r_recurrent_state_recovery_report(*_setup())

    assert report["evidence"]["byte_identical_restarted_newest_rows"]
    assert report["evidence"][
        "common_three_arm_metric_support_for_evaluable_groups"
    ]
    for metric in (
        "endpoint_rmse_m",
        "proper_score",
        "seam_rmse_m",
        "absolute_drift_slope_m_per_frame",
    ):
        result = report["aggregate"][metric]
        assert result["status"] == "defined"
        assert result["recovery_fraction"] == pytest.approx(0.5)
        assert result["bootstrap_interval"]["lower"] == pytest.approx(0.5)
        assert result["bootstrap_interval"]["upper"] == pytest.approx(0.5)
    assert 0.0 < report["aggregate"]["point_rmse_m"]["recovery_fraction"] < 1.0
    assert report["target_access"] == "forbidden"
    assert report["descriptive_only"]


def test_recovery_report_is_deterministic() -> None:
    inputs = _setup()
    assert build_cut3r_recurrent_state_recovery_report(
        *inputs
    ) == build_cut3r_recurrent_state_recovery_report(*inputs)


def test_recovery_rejects_different_restarted_newest_rows() -> None:
    inputs = list(_setup())
    recurrence_records = deepcopy(inputs[7])
    newest = next(
        row
        for row in recurrence_records["records"]
        if row["arm_id"] == "restarted-newest"
    )
    newest["proper_score"] = 10.1
    inputs[7] = recurrence_records
    inputs[8] = build_cut3r_source_competence_v2_report(
        inputs[0],
        inputs[5],
        inputs[6],
        recurrence_records,
    )

    with pytest.raises(ValueError, match="byte-identical restarted-newest rows"):
        build_cut3r_recurrent_state_recovery_report(*inputs)


def test_recovery_is_undefined_when_native_does_not_outperform_restart() -> None:
    inputs = list(_setup())
    recurrence_records = deepcopy(inputs[7])
    for row in recurrence_records["records"]:
        if row["arm_id"] == "native-continuous":
            row["endpoint_error_m"] = 1.2
    inputs[7] = recurrence_records
    inputs[8] = build_cut3r_source_competence_v2_report(
        inputs[0],
        inputs[5],
        inputs[6],
        recurrence_records,
    )

    report = build_cut3r_recurrent_state_recovery_report(*inputs)
    endpoint = report["aggregate"]["endpoint_rmse_m"]
    assert endpoint["status"] == "undefined-native-not-better"
    assert endpoint["recovery_fraction"] is None
    assert endpoint["bootstrap_interval"]["interval_status"] == "not-applicable"


def test_prediction_cli_dispatches_cut3r_recovery_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        prediction_main(["cut3r-recovery", "--help"])

    assert caught.value.code == 0
    assert "recurrent-state" in capsys.readouterr().out
