from __future__ import annotations

import json
from copy import deepcopy

import pytest

from prob4d.cut3r_comparison import build_cut3r_comparison_lock
from prob4d.cut3r_source_competence import build_cut3r_source_competence_lock
from prob4d.cut3r_source_competence_audit import (
    CUT3R_PROPER_SCORE_REFERENCE_FIT_SCOPE,
    build_cut3r_metric_support_manifest,
    build_cut3r_source_competence_audit_lock,
    build_cut3r_source_competence_support_audit_report,
    metric_support_from_manifest_entry,
    source_competence_gates_audited,
    validate_cut3r_source_competence_support_audit_report,
    verify_cut3r_source_competence_audit_reference,
    write_cut3r_metric_support_manifest,
    write_cut3r_source_competence_audit_lock,
    write_cut3r_source_competence_support_audit_report,
)
from prob4d.cut3r_source_competence_audit import (
    main as audit_main,
)
from prob4d.cut3r_source_competence_v2 import (
    CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS,
    CUT3R_SOURCE_COMPETENCE_V2_RECORDS_SCHEMA,
    build_cut3r_source_competence_v2_lock,
    build_cut3r_source_competence_v2_report,
)

REFERENCE_BYTES = b'{"schema":"fixed-scale-reference","scale_m":0.01}\n'


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
            "protocol_name": "support-audit-test-v1",
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
        "maximum_mean_absolute_drift_slope_m_per_frame": 0.01,
        "maximum_mean_seam_rmse_m": 0.02,
        "minimum_mean_quality_group_pass_fraction": 1.0,
        "minimum_mean_association_precision": 0.7,
        "minimum_mean_identity_retention": 0.7,
        "minimum_mean_support_retention": 0.7,
        "minimum_identity_group_pass_fraction": 1.0,
    }


def _source_lock(comparison: dict[str, object]) -> dict[str, object]:
    return build_cut3r_source_competence_lock(
        comparison,
        {
            "contrast_id": "prob4d-fusion-value",
            "candidate_provider_manifest_id": "e" * 64,
            "baseline_provider_manifest_id": "f" * 64,
            "cohort_binding_id": "1" * 64,
            "group_definition": "complete-object-support-audit-test",
            "record_definition_sha256": "2" * 64,
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
) -> dict[str, object]:
    return build_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        {
            "source_competence_policy": source_lock["policy"],
            "common_support_definition_sha256": "3" * 64,
            "proper_score_semantics": (CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS),
            "paired_policy": _paired_policy(),
            "require_complete_source_roster": True,
        },
    )


def _audit_lock(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    v2_lock: dict[str, object],
) -> dict[str, object]:
    return build_cut3r_source_competence_audit_lock(
        comparison,
        source_lock,
        v2_lock,
        {
            "common_support_definition_sha256": "3" * 64,
            "proper_score_reference_artifact_id": "a" * 64,
            "proper_score_reference_fit_scope": (CUT3R_PROPER_SCORE_REFERENCE_FIT_SCOPE),
            "proper_score_semantics": (CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS),
            "require_complete_manifest_roster": True,
        },
        REFERENCE_BYTES,
    )


def _entry(group: str, case: str, frame: int, seed: int) -> dict[str, object]:
    point_rows = [
        [group, case, frame, 100, "registered-world"],
        [group, case, frame, 101, "registered-world"],
    ]
    return {
        "group_id": group,
        "case_id": case,
        "frame_index": frame,
        "random_seed": seed,
        "point_rows": point_rows,
        "endpoint_rows": [[group, case, frame, "distal", 101, "registered-world"]],
        "proper_score_rows": [
            [group, case, frame, point_id, axis, "registered-world"]
            for point_id in (100, 101)
            for axis in (0, 1, 2)
        ],
        "seam_rows": (
            [
                [
                    group,
                    case,
                    frame,
                    "window-left",
                    "window-right",
                    100,
                    "registered-world",
                ]
            ]
            if frame == 0
            else []
        ),
    }


def _manifest_input() -> dict[str, object]:
    entries = []
    for group, case in (
        ("source-a", "source-a-case"),
        ("source-b", "source-b-case"),
    ):
        for frame in range(2):
            for seed in (7, 11):
                entries.append(_entry(group, case, frame, seed))
    return {
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "entries": entries,
    }


def _manifest(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    v2_lock: dict[str, object],
    audit_lock: dict[str, object],
    specification: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_cut3r_metric_support_manifest(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
        _manifest_input() if specification is None else specification,
    )


def _record(
    group: str,
    case: str,
    frame: int,
    seed: int,
    arm: str,
    support: dict[str, object],
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
        "metric_support": support,
    }


def _records(
    comparison: dict[str, object],
    source_lock: dict[str, object],
    v2_lock: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    support_by_key = {
        (
            entry["group_id"],
            entry["case_id"],
            entry["frame_index"],
            entry["random_seed"],
        ): metric_support_from_manifest_entry(entry)
        for entry in manifest["entries"]
    }
    rows = []
    for group, case in (
        ("source-a", "source-a-case"),
        ("source-b", "source-b-case"),
    ):
        for frame in range(2):
            for seed in (7, 11):
                for arm in ("restarted-prob4d-fused", "restarted-newest"):
                    rows.append(
                        _record(
                            group,
                            case,
                            frame,
                            seed,
                            arm,
                            support_by_key[(group, case, frame, seed)],
                        )
                    )
    return {
        "schema": CUT3R_SOURCE_COMPETENCE_V2_RECORDS_SCHEMA,
        "schema_version": 2,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": v2_lock["common_support_definition_sha256"],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": [],
        "records": rows,
    }


def _setup() -> tuple[dict[str, object], ...]:
    comparison = _comparison()
    source_lock = _source_lock(comparison)
    v2_lock = _v2_lock(comparison, source_lock)
    audit_lock = _audit_lock(comparison, source_lock, v2_lock)
    manifest = _manifest(comparison, source_lock, v2_lock, audit_lock)
    records = _records(comparison, source_lock, v2_lock, manifest)
    v2_report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )
    return comparison, source_lock, v2_lock, audit_lock, manifest, records, v2_report


def test_audit_builds_passing_claim_bearing_receipt() -> None:
    comparison, source_lock, v2_lock, audit_lock, manifest, records, v2_report = _setup()

    report = build_cut3r_source_competence_support_audit_report(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
        records,
        manifest,
        v2_report,
    )
    mean_gate, identity_gate = source_competence_gates_audited(report)

    assert report["verified_pair_count"] == 8
    assert report["support_manifest_status"] == "pass"
    assert report["proper_score_reference_binding_status"] == "pass"
    assert report["audited_source_competence_pass"]
    assert mean_gate.status == "pass"
    assert identity_gate.status == "pass"
    assert mean_gate.evidence_id == report["source_competence_support_audit_report_id"]
    assert (
        mean_gate.metadata["metric_support_manifest_id"] == manifest["metric_support_manifest_id"]
    )


def test_independent_rows_reject_colluding_supplied_hashes() -> None:
    comparison, source_lock, v2_lock, audit_lock, manifest, records, _ = _setup()
    for row in records["records"]:
        row["metric_support"]["point_support_sha256"] = "b" * 64
    v2_report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )

    with pytest.raises(ValueError, match="independently reconstructed metric support"):
        build_cut3r_source_competence_support_audit_report(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            records,
            manifest,
            v2_report,
        )


def test_row_reordering_is_detected_even_with_identical_row_set() -> None:
    comparison, source_lock, v2_lock, audit_lock, _, records, v2_report = _setup()
    specification = _manifest_input()
    specification["entries"][0]["point_rows"].reverse()
    reordered = _manifest(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
        specification,
    )

    with pytest.raises(ValueError, match="independently reconstructed metric support"):
        build_cut3r_source_competence_support_audit_report(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            records,
            reordered,
            v2_report,
        )


def test_exact_proper_score_reference_bytes_are_verified() -> None:
    comparison, source_lock, v2_lock, audit_lock, *_ = _setup()

    verify_cut3r_source_competence_audit_reference(audit_lock, REFERENCE_BYTES)
    with pytest.raises(ValueError, match="reference bytes do not match"):
        verify_cut3r_source_competence_audit_reference(
            audit_lock,
            REFERENCE_BYTES + b"tampered",
        )

    rebuilt = _audit_lock(comparison, source_lock, v2_lock)
    assert rebuilt == audit_lock


def test_manifest_rejects_duplicate_support_rows() -> None:
    comparison = _comparison()
    source_lock = _source_lock(comparison)
    v2_lock = _v2_lock(comparison, source_lock)
    audit_lock = _audit_lock(comparison, source_lock, v2_lock)
    specification = _manifest_input()
    first = specification["entries"][0]
    first["point_rows"].append(deepcopy(first["point_rows"][0]))

    with pytest.raises(ValueError, match="duplicate canonical rows"):
        _manifest(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            specification,
        )


def test_manifest_requires_three_consistent_score_axes_per_point() -> None:
    comparison = _comparison()
    source_lock = _source_lock(comparison)
    v2_lock = _v2_lock(comparison, source_lock)
    audit_lock = _audit_lock(comparison, source_lock, v2_lock)
    specification = _manifest_input()
    specification["entries"][0]["proper_score_rows"].pop()

    with pytest.raises(ValueError, match="three distinct axes"):
        _manifest(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            specification,
        )


def test_complete_manifest_roster_rejects_missing_pair() -> None:
    comparison = _comparison()
    source_lock = _source_lock(comparison)
    v2_lock = _v2_lock(comparison, source_lock)
    audit_lock = _audit_lock(comparison, source_lock, v2_lock)
    specification = _manifest_input()
    specification["entries"].pop()

    with pytest.raises(ValueError, match="frozen complete roster"):
        _manifest(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            specification,
        )


def test_manifest_rejects_target_access() -> None:
    comparison = _comparison()
    source_lock = _source_lock(comparison)
    v2_lock = _v2_lock(comparison, source_lock)
    audit_lock = _audit_lock(comparison, source_lock, v2_lock)
    specification = _manifest_input()
    specification["target_payloads_opened"] = True

    with pytest.raises(ValueError, match="may not open target data"):
        _manifest(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            specification,
        )


def test_valid_negative_v2_result_remains_auditable() -> None:
    comparison, source_lock, v2_lock, audit_lock, manifest, records, _ = _setup()
    for row in records["records"]:
        if row["arm_id"] == "restarted-prob4d-fused" and row["frame_index"] == 0:
            row["seam_error_m"] = 0.019
    v2_report = build_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
    )

    report = build_cut3r_source_competence_support_audit_report(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
        records,
        manifest,
        v2_report,
    )
    mean_gate, identity_gate = source_competence_gates_audited(report)

    assert not report["audited_source_competence_pass"]
    assert report["support_manifest_status"] == "pass"
    assert mean_gate.status == "fail"
    assert identity_gate.status == "not-evaluated"


def test_report_replay_rejects_tampering() -> None:
    comparison, source_lock, v2_lock, audit_lock, manifest, records, v2_report = _setup()
    report = build_cut3r_source_competence_support_audit_report(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
        records,
        manifest,
        v2_report,
    )
    tampered = deepcopy(report)
    tampered["verified_pair_count"] = 7

    with pytest.raises(ValueError, match="content identity changed"):
        validate_cut3r_source_competence_support_audit_report(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            records,
            manifest,
            v2_report,
            tampered,
        )


def test_publication_is_idempotent_and_no_clobber(tmp_path) -> None:
    comparison, source_lock, v2_lock, audit_lock, manifest, records, v2_report = _setup()
    report = build_cut3r_source_competence_support_audit_report(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
        records,
        manifest,
        v2_report,
    )
    lock_path = tmp_path / "support-audit-lock.json"
    manifest_path = tmp_path / "support-manifest.json"
    report_path = tmp_path / "support-audit-report.json"

    for _ in range(2):
        write_cut3r_source_competence_audit_lock(
            comparison,
            source_lock,
            v2_lock,
            lock_path,
            audit_lock,
            REFERENCE_BYTES,
        )
        write_cut3r_metric_support_manifest(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            manifest_path,
            manifest,
        )
        write_cut3r_source_competence_support_audit_report(
            comparison,
            source_lock,
            v2_lock,
            audit_lock,
            records,
            manifest,
            v2_report,
            report_path,
            report,
        )

    assert lock_path.is_file()
    assert manifest_path.is_file()
    assert report_path.is_file()


def test_cli_roundtrip(tmp_path, capsys) -> None:
    comparison, source_lock, v2_lock, _, _, _, _ = _setup()
    comparison_path = tmp_path / "comparison.json"
    source_path = tmp_path / "source-lock.json"
    v2_lock_path = tmp_path / "v2-lock.json"
    audit_spec_path = tmp_path / "audit-spec.json"
    reference_path = tmp_path / "score-reference.json"
    audit_lock_path = tmp_path / "audit-lock.json"
    manifest_input_path = tmp_path / "manifest-input.json"
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.json"
    v2_report_path = tmp_path / "v2-report.json"
    audit_report_path = tmp_path / "audit-report.json"

    def dump(path, value) -> None:
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    dump(comparison_path, comparison)
    dump(source_path, source_lock)
    dump(v2_lock_path, v2_lock)
    dump(
        audit_spec_path,
        {
            "common_support_definition_sha256": "3" * 64,
            "proper_score_reference_artifact_id": "a" * 64,
            "proper_score_reference_fit_scope": (CUT3R_PROPER_SCORE_REFERENCE_FIT_SCOPE),
            "proper_score_semantics": (CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS),
            "require_complete_manifest_roster": True,
        },
    )
    reference_path.write_bytes(REFERENCE_BYTES)
    assert (
        audit_main(
            [
                "freeze",
                str(comparison_path),
                str(source_path),
                str(v2_lock_path),
                str(audit_spec_path),
                str(reference_path),
                "--output",
                str(audit_lock_path),
            ]
        )
        == 0
    )
    dump(manifest_input_path, _manifest_input())
    assert (
        audit_main(
            [
                "manifest",
                str(comparison_path),
                str(source_path),
                str(v2_lock_path),
                str(audit_lock_path),
                str(manifest_input_path),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())
    records = _records(comparison, source_lock, v2_lock, manifest)
    v2_report = build_cut3r_source_competence_v2_report(comparison, source_lock, v2_lock, records)
    dump(records_path, records)
    dump(v2_report_path, v2_report)
    assert (
        audit_main(
            [
                "report",
                str(comparison_path),
                str(source_path),
                str(v2_lock_path),
                str(audit_lock_path),
                str(records_path),
                str(manifest_path),
                str(v2_report_path),
                "--output",
                str(audit_report_path),
                "--require-pass",
            ]
        )
        == 0
    )
    assert (
        audit_main(
            [
                "gates",
                str(comparison_path),
                str(source_path),
                str(v2_lock_path),
                str(audit_lock_path),
                str(records_path),
                str(manifest_path),
                str(v2_report_path),
                str(audit_report_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out.splitlines()[-1]
    gates = json.loads(output)
    assert gates["source_mean"]["status"] == "pass"
    assert gates["identity_reliability"]["status"] == "pass"
