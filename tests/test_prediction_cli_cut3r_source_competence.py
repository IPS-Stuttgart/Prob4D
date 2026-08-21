from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prob4d.cut3r_comparison import (
    build_cut3r_comparison_lock,
    write_cut3r_comparison_lock,
)
from prob4d.prediction_cli import main


def _case(case_id: str, digest_character: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input_video_sha256": digest_character * 64,
        "input_video_byte_count": 1000,
        "frame_start": 0,
        "frame_stop_exclusive": 2,
        "evaluation_frame_start": 0,
        "evaluation_frame_stop_exclusive": 2,
    }


def _comparison_lock() -> dict[str, Any]:
    return build_cut3r_comparison_lock(
        {
            "protocol_name": "cut3r-source-competence-cli-test-v1",
            "provider_revision": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "prob4d_revision": "c" * 40,
            "prob4d_distribution_sha256": "d" * 64,
            "window_size": 25,
            "overlap": 8,
            "confidence_threshold": 1.5,
            "storage_dtype": "float32",
            "random_seeds": [7],
            "groups": [
                {"group_id": "development", "cases": [_case("dev", "1")]},
                {"group_id": "calibration", "cases": [_case("cal", "2")]},
                {"group_id": "source", "cases": [_case("source", "3")]},
            ],
            "group_roles": {
                "development": ["development"],
                "calibration": ["calibration"],
                "source_evaluation": ["source"],
            },
            "include_revisit_diagnostic": False,
        }
    )


def _specification() -> dict[str, Any]:
    return {
        "contrast_id": "prob4d-fusion-value",
        "candidate_provider_manifest_id": "e" * 64,
        "baseline_provider_manifest_id": "f" * 64,
        "cohort_binding_id": "1" * 64,
        "group_definition": "complete-object-v1",
        "record_definition_sha256": "2" * 64,
        "policy": {
            "minimum_evaluable_groups": 1,
            "maximum_technical_failures": 0,
            "permitted_technical_failure_codes": [],
            "maximum_mean_proper_score_delta": 0.0,
            "maximum_mean_point_rmse_ratio": 1.0,
            "maximum_mean_endpoint_rmse_ratio": 1.0,
            "maximum_worst_group_point_rmse_ratio": 1.2,
            "maximum_mean_absolute_drift_slope_m_per_frame": 0.001,
            "maximum_mean_seam_rmse_m": 0.01,
            "minimum_mean_quality_group_pass_fraction": 0.75,
            "minimum_mean_association_precision": 0.9,
            "minimum_mean_identity_retention": 0.8,
            "minimum_mean_support_retention": 0.8,
            "minimum_identity_group_pass_fraction": 0.75,
        },
    }


def test_prediction_cli_dispatches_cut3r_source_competence_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["cut3r-source-competence", "--help"])

    assert caught.value.code == 0
    assert "freeze" in capsys.readouterr().out


def test_prediction_cli_dispatches_cut3r_source_competence_freeze(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    comparison_path = tmp_path / "comparison.json"
    specification_path = tmp_path / "specification.json"
    output_path = tmp_path / "source-competence-lock.json"
    write_cut3r_comparison_lock(comparison_path, _comparison_lock())
    specification_path.write_text(
        json.dumps(_specification(), sort_keys=True),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "cut3r-source-competence",
                "freeze",
                str(comparison_path),
                str(specification_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    lock_id = capsys.readouterr().out.strip()
    assert len(lock_id) == 64
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "source_competence_lock_id"
    ] == lock_id
