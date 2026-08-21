from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from prob4d.cut3r_comparison import (
    build_cut3r_comparison_lock,
    write_cut3r_comparison_lock,
)
from prob4d.cut3r_source_competence import (
    CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
    build_cut3r_source_competence_lock,
    build_cut3r_source_competence_report,
    main,
    validate_cut3r_source_competence_lock,
    write_cut3r_source_competence_lock,
    write_cut3r_source_competence_report,
)


def _case(
    case_id: str,
    *,
    evaluation_frame_stop_exclusive: int = 2,
    digest_character: str = "1",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "input_video_sha256": digest_character * 64,
        "input_video_byte_count": 100,
        "frame_start": 0,
        "frame_stop_exclusive": evaluation_frame_stop_exclusive,
        "evaluation_frame_start": 0,
        "evaluation_frame_stop_exclusive": evaluation_frame_stop_exclusive,
    }


def _comparison(*, include_revisit_diagnostic: bool = False) -> dict[str, object]:
    return build_cut3r_comparison_lock(
        {
            "protocol_name": "test-v1",
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
                {"group_id": "dev", "cases": [_case("dev-case")]},
                {"group_id": "cal", "cases": [_case("cal-case")]},
                {
                    "group_id": "source-a",
                    "cases": [_case("source-a-case", digest_character="2")],
                },
                {
                    "group_id": "source-b",
                    "cases": [_case("source-b-case", digest_character="3")],
                },
            ],
            "group_roles": {
                "development": ["dev"],
                "calibration": ["cal"],
                "source_evaluation": ["source-a", "source-b"],
            },
            "include_revisit_diagnostic": include_revisit_diagnostic,
        }
    )


def _policy(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "minimum_evaluable_groups": 2,
        "maximum_technical_failures": 0,
        "permitted_technical_failure_codes": [],
        "maximum_mean_proper_score_delta": 0.0,
        "maximum_mean_point_rmse_ratio": 1.0,
        "maximum_mean_endpoint_rmse_ratio": 1.0,
        "maximum_worst_group_point_rmse_ratio": 1.1,
        "maximum_mean_absolute_drift_slope_m_per_frame": 0.01,
        "maximum_mean_seam_rmse_m": 0.02,
        "minimum_mean_quality_group_pass_fraction": 1.0,
        "minimum_mean_association_precision": 0.9,
        "minimum_mean_identity_retention": 0.8,
        "minimum_mean_support_retention": 0.8,
        "minimum_identity_group_pass_fraction": 1.0,
    }
    value.update(overrides)
    return value


def _specification(
    **policy_overrides: object,
) -> dict[str, object]:
    return {
        "contrast_id": "prob4d-fusion-value",
        "candidate_provider_manifest_id": "e" * 64,
        "baseline_provider_manifest_id": "f" * 64,
        "cohort_binding_id": "1" * 64,
        "group_definition": "complete-object-v1",
        "record_definition_sha256": "2" * 64,
        "policy": _policy(**policy_overrides),
    }


def _lock(comparison: dict[str, object], **policy_overrides: object) -> dict[str, object]:
    return build_cut3r_source_competence_lock(
        comparison,
        _specification(**policy_overrides),
    )


def _record(
    group: str,
    case: str,
    frame: int,
    seed: int,
    arm: str,
    *,
    candidate_point_error_m: float = 0.8,
    baseline_point_error_m: float = 1.0,
) -> dict[str, object]:
    candidate = arm == "restarted-prob4d-fused"
    return {
        "group_id": group,
        "case_id": case,
        "frame_index": frame,
        "random_seed": seed,
        "arm_id": arm,
        "point_error_m": (
            candidate_point_error_m if candidate else baseline_point_error_m
        ),
        "endpoint_error_m": 0.7 if candidate else 1.0,
        "proper_score": 8.0 if candidate else 10.0,
        "seam_error_m": 0.01 if frame == 0 else None,
        "association_correct_count": 9 if candidate else 8,
        "association_predicted_count": 10,
        "identity_retained_count": 9 if candidate else 8,
        "identity_reference_count": 10,
        "support_retained_count": 9 if candidate else 8,
        "support_reference_count": 10,
    }


def _records(comparison: dict[str, object], lock: dict[str, object]) -> dict[str, object]:
    rows = []
    for group, case in (("source-a", "source-a-case"), ("source-b", "source-b-case")):
        for frame in range(2):
            for seed in (7, 11):
                for arm in ("restarted-prob4d-fused", "restarted-newest"):
                    rows.append(_record(group, case, frame, seed, arm))
    return {
        "schema": CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
        "schema_version": 1,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": lock["source_competence_lock_id"],
        "record_definition_sha256": "2" * 64,
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": [],
        "records": rows,
    }


def _rehash_lock(lock: dict[str, object]) -> None:
    unsigned = deepcopy(lock)
    unsigned.pop("source_competence_lock_id")
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    lock["source_competence_lock_id"] = hashlib.sha256(encoded).hexdigest()


def test_complete_paired_records_build_passing_report() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    report = build_cut3r_source_competence_report(
        comparison,
        lock,
        _records(comparison, lock),
    )

    assert report.source_competence_pass
    assert report.mean_point_rmse_ratio == pytest.approx(0.8)
    assert report.mean_endpoint_rmse_ratio == pytest.approx(0.7)
    assert report.group_count == 2
    assert report.metadata["contrast_id"] == "prob4d-fusion-value"
    assert report.metadata["baseline_provider_manifest_id"] == "f" * 64


def test_missing_frame_arm_record_fails_closed() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    records = _records(comparison, lock)
    records["records"].pop()

    with pytest.raises(ValueError, match="complete paired roster"):
        build_cut3r_source_competence_report(comparison, lock, records)


def test_target_opening_is_rejected() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    records = _records(comparison, lock)
    records["target_outcomes_opened"] = True

    with pytest.raises(ValueError, match="may not open target outcomes"):
        build_cut3r_source_competence_report(comparison, lock, records)


def test_paired_reference_rosters_must_match() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    records = _records(comparison, lock)
    records["records"][0]["support_reference_count"] = 9

    with pytest.raises(ValueError, match="arm-neutral support_reference_count"):
        build_cut3r_source_competence_report(comparison, lock, records)


def test_predeclared_group_failure_is_retained() -> None:
    comparison = _comparison()
    lock = _lock(
        comparison,
        minimum_evaluable_groups=1,
        maximum_technical_failures=1,
        permitted_technical_failure_codes=["gpu-oom"],
    )
    records = _records(comparison, lock)
    records["records"] = [
        item for item in records["records"] if item["group_id"] != "source-b"
    ]
    records["group_failures"] = [
        {
            "group_id": "source-b",
            "technical_failure_code": "gpu-oom",
            "metadata": {},
        }
    ]

    report = build_cut3r_source_competence_report(comparison, lock, records)

    assert report.source_competence_pass
    assert report.technical_failure_count == 1
    assert report.groups[1].technical_failure_code == "gpu-oom"


def test_lock_replay_rejects_impossible_minimum_group_count() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    lock["policy"]["minimum_evaluable_groups"] = 3
    _rehash_lock(lock)

    with pytest.raises(ValueError, match="exceeds the frozen source-evaluation roster"):
        validate_cut3r_source_competence_lock(comparison, lock)


def test_lock_replay_rejects_candidate_baseline_identity_collapse() -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    lock["baseline_provider_manifest_id"] = lock["candidate_provider_manifest_id"]
    _rehash_lock(lock)

    with pytest.raises(ValueError, match="must be distinct"):
        validate_cut3r_source_competence_lock(comparison, lock)


def test_noncausal_revisit_contrast_cannot_be_frozen() -> None:
    comparison = _comparison(include_revisit_diagnostic=True)
    specification = _specification()
    specification["contrast_id"] = "noncausal-revisit-upper-bound"

    with pytest.raises(ValueError, match="enabled claim-eligible contrast"):
        build_cut3r_source_competence_lock(comparison, specification)


def test_equal_case_weighting_prevents_long_case_domination() -> None:
    comparison = build_cut3r_comparison_lock(
        {
            "protocol_name": "equal-case-test-v1",
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
                {"group_id": "dev", "cases": [_case("dev-case")]},
                {"group_id": "cal", "cases": [_case("cal-case")]},
                {
                    "group_id": "source",
                    "cases": [
                        _case("short", evaluation_frame_stop_exclusive=2),
                        _case(
                            "long",
                            evaluation_frame_stop_exclusive=4,
                            digest_character="2",
                        ),
                    ],
                },
            ],
            "group_roles": {
                "development": ["dev"],
                "calibration": ["cal"],
                "source_evaluation": ["source"],
            },
            "include_revisit_diagnostic": False,
        }
    )
    lock = build_cut3r_source_competence_lock(
        comparison,
        _specification(
            minimum_evaluable_groups=1,
            maximum_mean_point_rmse_ratio=2.0,
            maximum_mean_endpoint_rmse_ratio=2.0,
            maximum_worst_group_point_rmse_ratio=2.0,
        ),
    )
    rows = []
    for case, frame_count, candidate_error in (("short", 2, 2.0), ("long", 4, 0.0)):
        for frame in range(frame_count):
            for arm in ("restarted-prob4d-fused", "restarted-newest"):
                rows.append(
                    _record(
                        "source",
                        case,
                        frame,
                        7,
                        arm,
                        candidate_point_error_m=candidate_error,
                    )
                )
    records = {
        "schema": CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
        "schema_version": 1,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": lock["source_competence_lock_id"],
        "record_definition_sha256": "2" * 64,
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": [],
        "records": rows,
    }

    report = build_cut3r_source_competence_report(comparison, lock, records)

    assert report.groups[0].candidate_point_rmse_m == pytest.approx(2.0**0.5)
    assert report.groups[0].baseline_point_rmse_m == pytest.approx(1.0)


def test_require_pass_writes_valid_negative_before_exit_three(tmp_path: Path) -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    records = _records(comparison, lock)
    for record in records["records"]:
        if record["arm_id"] == "restarted-prob4d-fused":
            record["point_error_m"] = 2.0
    comparison_path = tmp_path / "comparison.json"
    lock_path = tmp_path / "lock.json"
    records_path = tmp_path / "records.json"
    report_path = tmp_path / "report.json"
    write_cut3r_comparison_lock(comparison_path, comparison)
    write_cut3r_source_competence_lock(comparison, lock_path, lock)
    records_path.write_text(json.dumps(records), encoding="utf-8")

    status = main(
        [
            "report",
            str(comparison_path),
            str(lock_path),
            str(records_path),
            "--output",
            str(report_path),
            "--require-pass",
        ]
    )

    assert status == 3
    assert report_path.is_file()


def test_lock_and_report_publication_are_idempotent(tmp_path: Path) -> None:
    comparison = _comparison()
    lock = _lock(comparison)
    records = _records(comparison, lock)
    lock_path = tmp_path / "lock.json"
    report_path = tmp_path / "report.json"

    assert write_cut3r_source_competence_lock(comparison, lock_path, lock) == lock
    assert write_cut3r_source_competence_lock(comparison, lock_path, lock) == lock
    report = build_cut3r_source_competence_report(comparison, lock, records)
    assert write_cut3r_source_competence_report(
        comparison,
        lock,
        records,
        report_path,
        report,
    ).to_dict() == report.to_dict()
    assert write_cut3r_source_competence_report(
        comparison,
        lock,
        records,
        report_path,
        report,
    ).to_dict() == report.to_dict()
