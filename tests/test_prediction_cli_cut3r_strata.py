from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prob4d.cut3r_comparison import (
    build_cut3r_comparison_lock,
    write_cut3r_comparison_lock,
)
from prob4d.cut3r_diagnostic_strata import CUT3R_DIAGNOSTIC_STRATA
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
            "protocol_name": "cut3r-strata-cli-test-v1",
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


def _strata_specification() -> dict[str, Any]:
    return {
        "record_definition_sha256": "e" * 64,
        "minimum_evaluable_groups_per_bin": 1,
        "metric_names": ["point-error-m"],
        "strata": [
            {
                "stratum_id": stratum_id,
                "feature_name": feature_name,
                "unit": unit,
                "bin_edges": [0, 1],
                "value_source": value_source,
                "uses_truth": False,
                "uses_downstream_physical_innovation": False,
                "uses_target_outcomes": False,
                "selection_role": "reporting-only",
            }
            for stratum_id, feature_name, unit, value_source in CUT3R_DIAGNOSTIC_STRATA
        ],
    }


def test_prediction_cli_dispatches_cut3r_strata_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["cut3r-strata", "--help"])
    assert caught.value.code == 0
    assert "freeze" in capsys.readouterr().out


def test_prediction_cli_dispatches_cut3r_strata_freeze(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    comparison_path = tmp_path / "comparison.json"
    specification_path = tmp_path / "strata-specification.json"
    output_path = tmp_path / "strata-lock.json"
    write_cut3r_comparison_lock(comparison_path, _comparison_lock())
    specification_path.write_text(
        json.dumps(_strata_specification(), sort_keys=True),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "cut3r-strata",
                "freeze",
                str(comparison_path),
                str(specification_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    strata_lock_id = capsys.readouterr().out.strip()
    assert len(strata_lock_id) == 64
    assert json.loads(output_path.read_text(encoding="utf-8"))["strata_lock_id"] == (
        strata_lock_id
    )
