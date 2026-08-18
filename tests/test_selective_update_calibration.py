from __future__ import annotations

import json
from dataclasses import replace

import pytest

from prob4d.selective_update_calibration import (
    SelectiveUpdateCalibrationV1,
    SelectiveUpdateGroupV1,
    SelectiveUpdateThresholdsV1,
    build_selective_update_calibration,
    load_selective_update_calibration,
    main,
    write_selective_update_calibration,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def thresholds() -> SelectiveUpdateThresholdsV1:
    return SelectiveUpdateThresholdsV1(
        nominal_coverage=0.9,
        maximum_accepted_coverage_shortfall=0.05,
        maximum_complete_policy_coverage_shortfall=0.02,
        maximum_selection_coverage_drop=0.05,
        maximum_mean_width_ratio_vs_fallback=1.1,
        minimum_mean_score_advantage_vs_fallback=0.05,
        harmful_score_margin=0.0,
        maximum_harmful_accepted_fraction=0.0,
        minimum_accepted_group_count=2,
        maximum_worst_group_coverage_shortfall=0.05,
    )


def row(
    group_id: str,
    *,
    accepted: bool,
    candidate_coverage: float,
    candidate_width: float,
    candidate_score: float,
    fallback_coverage: float = 0.9,
    fallback_width: float = 10.0,
    fallback_score: float = 1.0,
) -> SelectiveUpdateGroupV1:
    if accepted:
        deployed_coverage = candidate_coverage
        deployed_width = candidate_width
        deployed_score = candidate_score
    else:
        deployed_coverage = fallback_coverage
        deployed_width = fallback_width
        deployed_score = fallback_score
    return SelectiveUpdateGroupV1(
        group_id=group_id,
        accepted=accepted,
        candidate_coverage=candidate_coverage,
        candidate_width=candidate_width,
        candidate_score=candidate_score,
        fallback_coverage=fallback_coverage,
        fallback_width=fallback_width,
        fallback_score=fallback_score,
        deployed_coverage=deployed_coverage,
        deployed_width=deployed_width,
        deployed_score=deployed_score,
        metadata={"source_partition": "validation"},
    )


def passing_rows() -> tuple[SelectiveUpdateGroupV1, ...]:
    return (
        row(
            "validation-a",
            accepted=True,
            candidate_coverage=0.92,
            candidate_width=9.0,
            candidate_score=0.70,
        ),
        row(
            "validation-b",
            accepted=True,
            candidate_coverage=0.90,
            candidate_width=9.5,
            candidate_score=0.75,
        ),
        row(
            "validation-c",
            accepted=False,
            candidate_coverage=0.80,
            candidate_width=8.0,
            candidate_score=1.30,
        ),
        row(
            "validation-d",
            accepted=False,
            candidate_coverage=0.82,
            candidate_width=8.5,
            candidate_score=1.20,
        ),
    )


def build(
    rows: tuple[SelectiveUpdateGroupV1, ...] | None = None,
) -> SelectiveUpdateCalibrationV1:
    selected_rows = passing_rows() if rows is None else rows
    return build_selective_update_calibration(
        protocol_id="selective-source-validation-v1",
        query_definition="equal-object guarded physical query",
        score_definition="group-mean Gaussian negative log likelihood",
        width_unit="mm",
        group_definition="complete physical object or acquisition session",
        selection_lock_id=DIGEST_A,
        guard_artifact_id=DIGEST_B,
        candidate_artifact_id=DIGEST_C,
        fallback_artifact_id=DIGEST_D,
        guard_fit_group_ids=("fit-a", "fit-b"),
        guard_calibration_group_ids=("cal-a", "cal-b"),
        rows=selected_rows,
        thresholds=thresholds(),
        metadata={"uses_target_outcomes": False},
    )


def test_selective_update_certificate_passes_and_separates_all_populations() -> None:
    artifact = build()
    report = artifact.report

    assert report.passed is True
    assert report.group_count == 4
    assert report.accepted_group_count == 2
    assert report.rejected_group_count == 2
    assert report.accepted_mean_coverage == pytest.approx(0.91)
    assert report.candidate_mean_coverage == pytest.approx(0.86)
    assert report.deployed_mean_coverage == pytest.approx(0.905)
    assert report.selection_coverage_drop == pytest.approx(-0.05)
    assert report.accepted_score_advantage_vs_fallback == pytest.approx(0.275)
    assert report.deployed_score_advantage_vs_fallback == pytest.approx(0.1375)
    assert report.accepted_to_fallback_width_ratio == pytest.approx(0.925)
    assert report.harmful_accepted_count == 0
    assert all(report.criteria.values())


def test_selection_induced_undercoverage_fails_even_when_complete_policy_is_calibrated() -> None:
    rows = [
        row(
            "validation-a",
            accepted=True,
            candidate_coverage=0.50,
            candidate_width=5.0,
            candidate_score=0.70,
        ),
        row(
            "validation-b",
            accepted=True,
            candidate_coverage=0.55,
            candidate_width=5.0,
            candidate_score=0.70,
        ),
    ]
    rows.extend(
        row(
            f"validation-{index}",
            accepted=False,
            candidate_coverage=1.0,
            candidate_width=12.0,
            candidate_score=1.30,
            fallback_coverage=1.0,
        )
        for index in range(2, 10)
    )
    artifact = build(tuple(rows))

    assert artifact.report.deployed_mean_coverage == pytest.approx(0.905)
    assert artifact.report.accepted_mean_coverage == pytest.approx(0.525)
    assert artifact.report.selection_coverage_drop == pytest.approx(0.38)
    assert artifact.report.criteria["complete_policy_coverage"] is True
    assert artifact.report.criteria["accepted_coverage"] is False
    assert artifact.report.criteria["selection_coverage_drop"] is False
    assert artifact.report.passed is False


def test_harmful_accepted_update_is_detected() -> None:
    rows = list(passing_rows())
    rows[0] = row(
        "validation-a",
        accepted=True,
        candidate_coverage=0.92,
        candidate_width=9.0,
        candidate_score=1.20,
    )
    artifact = build(tuple(rows))

    assert artifact.report.harmful_accepted_count == 1
    assert artifact.report.harmful_accepted_fraction == pytest.approx(0.5)
    assert artifact.report.criteria["harmful_accepted_fraction"] is False
    assert artifact.report.passed is False


def test_deployed_metrics_must_match_candidate_or_fallback() -> None:
    with pytest.raises(ValueError, match="deployed metrics"):
        SelectiveUpdateGroupV1(
            group_id="validation-a",
            accepted=False,
            candidate_coverage=0.8,
            candidate_width=8.0,
            candidate_score=1.2,
            fallback_coverage=0.9,
            fallback_width=10.0,
            fallback_score=1.0,
            deployed_coverage=0.9,
            deployed_width=10.0,
            deployed_score=0.9,
        )


def test_split_overlap_is_rejected() -> None:
    artifact = build()
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            artifact,
            guard_calibration_group_ids=("cal-a", "fit-a"),
        )


def test_artifact_is_permutation_invariant() -> None:
    first = build(passing_rows())
    second = build(tuple(reversed(passing_rows())))

    assert first.artifact_id == second.artifact_id
    assert first.to_dict() == second.to_dict()


def test_round_trip_tamper_rejection_and_no_clobber(tmp_path) -> None:
    artifact = build()
    path = tmp_path / "selective-update-calibration.json"
    write_selective_update_calibration(artifact, path)

    loaded = load_selective_update_calibration(path)
    assert loaded.to_dict() == artifact.to_dict()
    write_selective_update_calibration(artifact, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report"]["deployed_mean_coverage"] -= 0.1
    with pytest.raises(ValueError, match="deterministic replay"):
        SelectiveUpdateCalibrationV1.from_dict(payload)

    different = replace(artifact, metadata={"variant": "different"})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_selective_update_calibration(different, path)


def test_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path) -> None:
    artifact = build()
    path = tmp_path / "selective-update-calibration.json"
    write_selective_update_calibration(artifact, path)
    original = path.read_text(encoding="utf-8")

    schema_line = f'  "schema_name": "{artifact.to_dict()["schema_name"]}",'
    path.write_text(
        original.replace(schema_line, f"{schema_line}\n{schema_line}", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_selective_update_calibration(path)

    path.write_text(
        original.replace('  "protocol_id":', '  "poison": NaN,\n  "protocol_id":', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_selective_update_calibration(path)


def test_cli_build_and_verify(tmp_path, capsys) -> None:
    artifact = build()
    raw = artifact.to_dict()
    for key in (
        "schema_name",
        "schema_version",
        "score_direction",
        "evidence_partition",
        "uses_target_outcomes",
        "validation_group_ids",
        "report",
        "claim_boundary",
        "artifact_id",
    ):
        raw.pop(key)
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "artifact.json"
    input_path.write_text(json.dumps(raw), encoding="utf-8")

    assert main(["build", str(input_path), "--output", str(output_path), "--require-pass"]) == 0
    assert capsys.readouterr().out.strip() == artifact.artifact_id
    assert main(["verify", str(output_path), "--require-pass"]) == 0
    assert capsys.readouterr().out.strip() == artifact.artifact_id
