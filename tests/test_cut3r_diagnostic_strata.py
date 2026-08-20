from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from prob4d.cut3r_comparison import (
    build_cut3r_comparison_lock,
    write_cut3r_comparison_lock,
)
from prob4d.cut3r_diagnostic_strata import (
    CUT3R_DIAGNOSTIC_STRATA,
    CUT3R_STRATA_LOCK_SCHEMA,
    CUT3R_STRATA_REPORT_SCHEMA,
    build_cut3r_diagnostic_strata_lock,
    build_cut3r_diagnostic_strata_report,
    cut3r_diagnostic_strata_summary,
    load_cut3r_diagnostic_strata_lock,
    load_cut3r_diagnostic_strata_report,
    main,
    validate_cut3r_diagnostic_strata_lock,
    validate_cut3r_diagnostic_strata_report,
    write_cut3r_diagnostic_strata_lock,
    write_cut3r_diagnostic_strata_report,
)


def _case(
    case_id: str,
    digest_character: str,
    *,
    evaluation_stop: int,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input_video_sha256": digest_character * 64,
        "input_video_byte_count": 1000,
        "frame_start": 0,
        "frame_stop_exclusive": evaluation_stop,
        "evaluation_frame_start": 0,
        "evaluation_frame_stop_exclusive": evaluation_stop,
    }


def _comparison_lock() -> dict[str, Any]:
    return build_cut3r_comparison_lock(
        {
            "protocol_name": "cut3r-strata-test-v1",
            "provider_revision": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "prob4d_revision": "c" * 40,
            "prob4d_distribution_sha256": "d" * 64,
            "window_size": 25,
            "overlap": 8,
            "confidence_threshold": 1.5,
            "storage_dtype": "float32",
            "random_seeds": [7, 11],
            "groups": [
                {
                    "group_id": "development",
                    "cases": [_case("development-case", "1", evaluation_stop=1)],
                },
                {
                    "group_id": "calibration",
                    "cases": [_case("calibration-case", "2", evaluation_stop=1)],
                },
                {
                    "group_id": "source-a",
                    "cases": [
                        _case("source-a-long", "3", evaluation_stop=100),
                        _case("source-a-short", "5", evaluation_stop=1),
                    ],
                },
                {
                    "group_id": "source-b",
                    "cases": [_case("source-b-case", "4", evaluation_stop=1)],
                },
            ],
            "group_roles": {
                "development": ["development"],
                "calibration": ["calibration"],
                "source_evaluation": ["source-a", "source-b"],
            },
            "include_revisit_diagnostic": False,
        }
    )


def _strata_specification() -> dict[str, Any]:
    edges = {
        "absolute-prefix-age": [0, 10],
        "frames-since-restart-boundary": [0, 10],
        "occlusion-reappearance-gap": [0, 2],
        "normalized-image-motion": [0, 0.1],
        "viewpoint-rotation-novelty": [0, 15],
        "metric-anchor-conditioning": [0, 2],
    }
    return {
        "record_definition_sha256": "e" * 64,
        "minimum_evaluable_groups_per_bin": 2,
        "metric_names": ["proper-score", "point-error-m"],
        "strata": [
            {
                "stratum_id": stratum_id,
                "feature_name": feature_name,
                "unit": unit,
                "bin_edges": edges[stratum_id],
                "value_source": value_source,
                "uses_truth": False,
                "uses_downstream_physical_innovation": False,
                "uses_target_outcomes": False,
                "selection_role": "reporting-only",
            }
            for stratum_id, feature_name, unit, value_source in CUT3R_DIAGNOSTIC_STRATA
        ],
    }


def _features(frame_index: int) -> dict[str, float]:
    values = {
        feature_name: 0.0
        for _, feature_name, _, _ in CUT3R_DIAGNOSTIC_STRATA
    }
    values["frames_since_sequence_start"] = float(frame_index)
    values["frames_since_restart_boundary"] = float(frame_index % 17)
    return values


def _records(comparison_lock: dict[str, Any], strata_lock: dict[str, Any]) -> dict[str, Any]:
    arm_values = {
        "native-continuous": {
            "source-a-long": {"point-error-m": 0.8, "proper-score": 0.8},
            "source-a-short": {"point-error-m": 5.8, "proper-score": 5.8},
            "source-b-case": {"point-error-m": 4.0, "proper-score": 4.0},
        },
        "restarted-newest": {
            "source-a-long": {"point-error-m": 1.0, "proper-score": 1.0},
            "source-a-short": {"point-error-m": 5.0, "proper-score": 5.0},
            "source-b-case": {"point-error-m": 3.0, "proper-score": 3.0},
        },
        "restarted-prob4d-fused": {
            "source-a-long": {"point-error-m": 0.5, "proper-score": 0.5},
            "source-a-short": {"point-error-m": 4.5, "proper-score": 4.5},
            "source-b-case": {"point-error-m": 2.0, "proper-score": 2.0},
        },
    }
    records: list[dict[str, Any]] = []
    for group_id, case_id, frame_count in (
        ("source-a", "source-a-long", 100),
        ("source-a", "source-a-short", 1),
        ("source-b", "source-b-case", 1),
    ):
        for frame_index in range(frame_count):
            for random_seed in comparison_lock["random_seeds"]:
                for arm_id, by_case in arm_values.items():
                    records.append(
                        {
                            "group_id": group_id,
                            "case_id": case_id,
                            "frame_index": frame_index,
                            "random_seed": random_seed,
                            "arm_id": arm_id,
                            "features": _features(frame_index),
                            "metrics": by_case[case_id],
                        }
                    )
    return {
        "schema": "prob4d.cut3r-diagnostic-records",
        "schema_version": 1,
        "comparison_lock_id": comparison_lock["lock_id"],
        "strata_lock_id": strata_lock["strata_lock_id"],
        "record_definition_sha256": strata_lock["record_definition_sha256"],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "records": records,
    }


def _first_bin(report: dict[str, Any], stratum_id: str) -> dict[str, Any]:
    stratum = next(
        item for item in report["strata_results"] if item["stratum_id"] == stratum_id
    )
    return stratum["bins"][0]


def test_strata_lock_is_bound_source_only_and_reporting_only() -> None:
    comparison = _comparison_lock()
    first = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    second = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())

    assert first == second
    assert first["schema"] == CUT3R_STRATA_LOCK_SCHEMA
    assert first["comparison_lock_id"] == comparison["lock_id"]
    assert first["source_evaluation_groups"] == ["source-a", "source-b"]
    assert first["selection_role"] == "reporting-only"
    assert first["target_access"] == "forbidden"
    assert [item["stratum_id"] for item in first["strata"]] == [
        item[0] for item in CUT3R_DIAGNOSTIC_STRATA
    ]
    assert validate_cut3r_diagnostic_strata_lock(comparison, first) == first


def test_strata_lock_rejects_outcome_features_and_underpowered_requirement() -> None:
    comparison = _comparison_lock()
    specification = _strata_specification()
    specification["strata"][0]["uses_truth"] = True
    with pytest.raises(ValueError, match="uses_truth must be false"):
        build_cut3r_diagnostic_strata_lock(comparison, specification)

    specification = _strata_specification()
    specification["minimum_evaluable_groups_per_bin"] = 3
    with pytest.raises(ValueError, match="exceeds the frozen source-evaluation roster"):
        build_cut3r_diagnostic_strata_lock(comparison, specification)


def test_report_uses_equal_group_mass_and_paired_contrasts() -> None:
    comparison = _comparison_lock()
    lock = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    records = _records(comparison, lock)
    report = build_cut3r_diagnostic_strata_report(comparison, lock, records)

    assert report["schema"] == CUT3R_STRATA_REPORT_SCHEMA
    assert report["record_count"] == 612
    assert report["nested_observation_count"] == 204
    assert report["target_outcomes_opened"] is False
    bin_result = _first_bin(report, "normalized-image-motion")
    assert bin_result["evaluable_group_count"] == 2
    assert bin_result["nested_observation_count"] == 204
    assert bin_result["meets_minimum_evaluable_groups"] is True

    arm_results = {item["arm_id"]: item for item in bin_result["arm_results"]}
    # Source A first averages its two cases: (1 + 5) / 2 = 3. Source B is 3.
    # Long source-A case rows therefore cannot dominate either aggregation stage.
    assert arm_results["restarted-newest"]["equal_group_mean_metrics"][
        "point-error-m"
    ] == pytest.approx(3.0)
    group_results = {item["group_id"]: item for item in bin_result["group_results"]}
    assert group_results["source-a"]["evaluable_case_count"] == 2
    assert group_results["source-a"]["evaluable_seed_count"] == 2
    contrasts = {item["contrast_id"]: item for item in bin_result["contrast_results"]}
    assert contrasts["prob4d-fusion-value"]["equal_group_mean_delta_metrics"][
        "point-error-m"
    ] == pytest.approx(-0.75)
    assert contrasts["provider-recurrence-value"]["equal_group_mean_delta_metrics"][
        "point-error-m"
    ] == pytest.approx(0.65)

    empty_bin = report["strata_results"][3]["bins"][1]
    assert empty_bin["evaluable_group_count"] == 0
    assert empty_bin["meets_minimum_evaluable_groups"] is False
    assert empty_bin["arm_results"] == []

    assert validate_cut3r_diagnostic_strata_report(
        comparison,
        lock,
        records,
        report,
    ) == report
    summary = cut3r_diagnostic_strata_summary(report)
    assert summary["random_seed_count"] == 2
    assert summary["stratum_count"] == 6
    assert summary["populated_bin_count"] == 8
    assert summary["adequately_supported_bin_count"] == 6
    assert summary["selection_role"] == "reporting-only"


def test_report_rejects_arm_dependent_stratification_and_target_access() -> None:
    comparison = _comparison_lock()
    lock = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    records = _records(comparison, lock)
    records["records"][1]["features"]["normalized_image_motion"] = 0.2
    with pytest.raises(ValueError, match="identical across arms and seeds"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)

    records = _records(comparison, lock)
    records["target_outcomes_opened"] = True
    with pytest.raises(ValueError, match="may not open target outcomes"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)


def test_report_rejects_unpaired_common_support() -> None:
    comparison = _comparison_lock()
    lock = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    records = _records(comparison, lock)
    records["records"].pop()
    with pytest.raises(ValueError, match="paired common support"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)


def test_report_requires_the_complete_frozen_seed_roster() -> None:
    comparison = _comparison_lock()
    lock = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    records = _records(comparison, lock)
    records["records"] = [
        record
        for record in records["records"]
        if not (
            record["group_id"] == "source-b"
            and record["frame_index"] == 0
            and record["random_seed"] == 11
        )
    ]
    with pytest.raises(ValueError, match="complete frozen random-seed roster"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)



def test_records_bind_the_frozen_definition_and_schedule() -> None:
    comparison = _comparison_lock()
    lock = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    records = _records(comparison, lock)

    records["record_definition_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different frozen record definition"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)

    records = _records(comparison, lock)
    records["records"][0]["features"]["frames_since_sequence_start"] = 1.0
    with pytest.raises(ValueError, match="frozen source frame index"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)

    records = _records(comparison, lock)
    target = next(
        record
        for record in records["records"]
        if record["frame_index"] == 18
    )
    target["features"]["frames_since_restart_boundary"] = 18.0
    with pytest.raises(ValueError, match="frozen window schedule"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)


def test_report_rejects_omitted_frozen_frames() -> None:
    comparison = _comparison_lock()
    lock = build_cut3r_diagnostic_strata_lock(comparison, _strata_specification())
    records = _records(comparison, lock)
    records["records"] = [
        record
        for record in records["records"]
        if not (
            record["group_id"] == "source-a"
            and record["case_id"] == "source-a-long"
            and record["frame_index"] == 99
        )
    ]
    with pytest.raises(ValueError, match="retain every frozen evaluation frame"):
        build_cut3r_diagnostic_strata_report(comparison, lock, records)


def test_checked_example_builds_against_quality_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    comparison = build_cut3r_comparison_lock(
        json.loads(
            (
                root / "tests/fixtures/cut3r-strata-comparison-spec.json"
            ).read_text(encoding="utf-8")
        )
    )
    specification = json.loads(
        (
            root / "docs/examples/cut3r-diagnostic-strata-spec.json"
        ).read_text(encoding="utf-8")
    )

    lock = build_cut3r_diagnostic_strata_lock(comparison, specification)
    assert len(lock["source_evaluation_groups"]) == 4
    assert lock["minimum_evaluable_groups_per_bin"] == 4

def test_round_trip_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    comparison = _comparison_lock()
    comparison_path = tmp_path / "comparison.json"
    strata_specification_path = tmp_path / "strata-specification.json"
    strata_lock_path = tmp_path / "strata-lock.json"
    records_path = tmp_path / "records.json"
    report_path = tmp_path / "report.json"

    write_cut3r_comparison_lock(comparison_path, comparison)
    strata_specification_path.write_text(
        json.dumps(_strata_specification(), sort_keys=True),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "freeze",
                str(comparison_path),
                str(strata_specification_path),
                "--output",
                str(strata_lock_path),
            ]
        )
        == 0
    )
    strata_lock_id = capsys.readouterr().out.strip()
    lock = load_cut3r_diagnostic_strata_lock(comparison, strata_lock_path)
    assert strata_lock_id == lock["strata_lock_id"]
    assert write_cut3r_diagnostic_strata_lock(comparison, strata_lock_path, lock) == lock

    records = _records(comparison, lock)
    records_path.write_text(json.dumps(records, sort_keys=True), encoding="utf-8")
    assert (
        main(
            [
                "report",
                str(comparison_path),
                str(strata_lock_path),
                str(records_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report_id = capsys.readouterr().out.strip()
    report = load_cut3r_diagnostic_strata_report(
        comparison,
        lock,
        records,
        report_path,
    )
    assert report_id == report["report_id"]
    assert (
        write_cut3r_diagnostic_strata_report(
            comparison,
            lock,
            records,
            report_path,
            report,
        )
        == report
    )

    assert (
        main(
            [
                "verify-report",
                str(comparison_path),
                str(strata_lock_path),
                str(records_path),
                str(report_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == report_id

    tampered = deepcopy(report)
    tampered["record_count"] += 1
    with pytest.raises(ValueError, match="does not match the bound records"):
        validate_cut3r_diagnostic_strata_report(
            comparison,
            lock,
            records,
            tampered,
        )
