from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prob4d.prediction_cli import main
from prob4d.semantic_compatibility import build_semantic_compatibility_manifest


def _case(case_id: str, digest_character: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input_video_sha256": digest_character * 64,
        "input_video_byte_count": 1000,
        "frame_start": 0,
        "frame_stop_exclusive": 50,
        "evaluation_frame_start": 10,
        "evaluation_frame_stop_exclusive": 40,
    }


def _cut3r_specification() -> dict[str, Any]:
    return {
        "protocol_name": "cli-dispatch-test",
        "provider_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "prob4d_revision": "c" * 40,
        "prob4d_distribution_sha256": "d" * 64,
        "window_size": 25,
        "overlap": 8,
        "confidence_threshold": 0.0,
        "storage_dtype": "float32",
        "random_seeds": [7],
        "groups": [
            {"group_id": "development", "cases": [_case("case-a", "1")]},
            {"group_id": "calibration", "cases": [_case("case-b", "2")]},
            {"group_id": "source-evaluation", "cases": [_case("case-c", "3")]},
        ],
        "group_roles": {
            "development": ["development"],
            "calibration": ["calibration"],
            "source_evaluation": ["source-evaluation"],
        },
        "include_revisit_diagnostic": False,
    }


def test_prediction_cli_dispatches_semantic_compatibility(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["compatibility", "print"]) == 0
    assert json.loads(capsys.readouterr().out) == build_semantic_compatibility_manifest()


def test_prediction_cli_dispatches_cut3r_comparison(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    specification = tmp_path / "specification.json"
    lock = tmp_path / "lock.json"
    specification.write_text(
        json.dumps(_cut3r_specification(), sort_keys=True),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "cut3r-comparison",
                "build",
                str(specification),
                "--output",
                str(lock),
            ]
        )
        == 0
    )
    lock_id = capsys.readouterr().out.strip()
    assert len(lock_id) == 64
    assert main(["cut3r-comparison", "verify", str(lock)]) == 0
    assert capsys.readouterr().out.strip() == lock_id
