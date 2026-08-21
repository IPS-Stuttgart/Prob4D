from __future__ import annotations

import pytest

from prob4d.cut3r_comparison import build_cut3r_comparison_lock
from prob4d.cut3r_source_competence import (
    CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
    build_cut3r_source_competence_lock,
    build_cut3r_source_competence_report,
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
            "protocol_name": "seam-support-test-v1",
            "provider_revision": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "prob4d_revision": "c" * 40,
            "prob4d_distribution_sha256": "d" * 64,
            "window_size": 2,
            "overlap": 1,
            "confidence_threshold": 1.0,
            "storage_dtype": "float32",
            "random_seeds": [7],
            "groups": [
                {"group_id": "dev", "cases": [_case("dev-case", "1")]},
                {"group_id": "cal", "cases": [_case("cal-case", "2")]},
                {"group_id": "source", "cases": [_case("source-case", "3")]},
            ],
            "group_roles": {
                "development": ["dev"],
                "calibration": ["cal"],
                "source_evaluation": ["source"],
            },
            "include_revisit_diagnostic": False,
        }
    )


def _lock(comparison: dict[str, object]) -> dict[str, object]:
    return build_cut3r_source_competence_lock(
        comparison,
        {
            "contrast_id": "prob4d-fusion-value",
            "candidate_provider_manifest_id": "e" * 64,
            "baseline_provider_manifest_id": "f" * 64,
            "cohort_binding_id": "4" * 64,
            "group_definition": "complete-object-v1",
            "record_definition_sha256": "5" * 64,
            "policy": {
                "minimum_evaluable_groups": 1,
                "maximum_technical_failures": 0,
                "permitted_technical_failure_codes": [],
                "maximum_mean_proper_score_delta": 0.0,
                "maximum_mean_point_rmse_ratio": 1.0,
                "maximum_mean_endpoint_rmse_ratio": 1.0,
                "maximum_worst_group_point_rmse_ratio": 1.0,
                "maximum_mean_absolute_drift_slope_m_per_frame": 1.0,
                "maximum_mean_seam_rmse_m": 1.0,
                "minimum_mean_quality_group_pass_fraction": 1.0,
                "minimum_mean_association_precision": 0.0,
                "minimum_mean_identity_retention": 0.0,
                "minimum_mean_support_retention": 0.0,
                "minimum_identity_group_pass_fraction": 1.0,
            },
        },
    )


def _record(frame: int, arm: str, seam_error_m: float | None) -> dict[str, object]:
    candidate = arm == "restarted-prob4d-fused"
    return {
        "group_id": "source",
        "case_id": "source-case",
        "frame_index": frame,
        "random_seed": 7,
        "arm_id": arm,
        "point_error_m": 0.5 if candidate else 1.0,
        "endpoint_error_m": 0.5 if candidate else 1.0,
        "proper_score": 0.5 if candidate else 1.0,
        "seam_error_m": seam_error_m,
        "association_correct_count": 1,
        "association_predicted_count": 1,
        "identity_retained_count": 1,
        "identity_reference_count": 1,
        "support_retained_count": 1,
        "support_reference_count": 1,
    }


def test_paired_arms_must_use_the_same_seam_observation_support() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    records = {
        "schema": CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
        "schema_version": 1,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": lock["source_competence_lock_id"],
        "record_definition_sha256": "5" * 64,
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": [],
        "records": [
            _record(0, "restarted-prob4d-fused", 0.01),
            _record(0, "restarted-newest", None),
            _record(1, "restarted-prob4d-fused", None),
            _record(1, "restarted-newest", 0.01),
        ],
    }

    with pytest.raises(ValueError, match="arm-neutral seam observation support"):
        build_cut3r_source_competence_report(comparison, lock, records)
