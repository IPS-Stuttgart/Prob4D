from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prob4d.material_identity_cli import main
from prob4d.material_identity_mixture import load_material_identity_mixture
from prob4d.material_identity_weight_calibration import (
    load_material_identity_weight_calibration,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _calibration_config() -> dict[str, object]:
    examples: list[dict[str, object]] = []
    for group in range(4):
        for example, score in enumerate((0.2, 0.8)):
            null_id = _sha(f"group-{group}-example-{example}-null")
            linked_id = _sha(f"group-{group}-example-{example}-linked")
            examples.append(
                {
                    "example_id": f"group-{group}-example-{example}",
                    "group_id": f"group-{group}",
                    "candidate_ids": [null_id, linked_id],
                    "candidate_kinds": ["null", "linked"],
                    "features": [[0.0], [score]],
                    "true_candidate_id": linked_id if score > 0.5 else null_id,
                    "metadata": {},
                }
            )
    return {
        "feature_names": ["source_score"],
        "feature_schema_id": "a" * 64,
        "association_rule_id": "b" * 64,
        "tracklet_producer_revision": "c" * 40,
        "association_revision": "d" * 40,
        "label_definition": "source material-label identity",
        "group_definition": "complete physical object or acquisition session",
        "cross_fit_fold_count": 2,
        "ridge": 0.05,
        "maximum_iterations": 100,
        "convergence_tolerance": 1e-10,
        "examples": examples,
        "metadata": {"source_only": True},
        "uses_target_outcomes": False,
    }


def _mixture_config() -> dict[str, object]:
    return {
        "target_endpoint": {"window_id": "window-2", "track_id": 3},
        "window_order": ["window-0", "window-1", "window-2"],
        "causal_frame_stop": 75,
        "association_rule_id": "b" * 64,
        "feature_schema_id": "a" * 64,
        "tracklet_producer_revision": "c" * 40,
        "association_revision": "d" * 40,
        "feature_names": ["source_score"],
        "candidates": [
            {
                "source_endpoint": None,
                "association_result_id": None,
                "source_score": None,
                "features": [0.0],
                "metadata": {"fallback": True},
            },
            {
                "source_endpoint": {"window_id": "window-1", "track_id": 7},
                "association_result_id": "e" * 64,
                "source_score": 0.9,
                "features": [0.9],
                "metadata": {"source_only": True},
            },
        ],
        "metadata": {"claim_bearing": False},
    }


def test_fit_validate_and_apply_calibration_cli(tmp_path: Path, capsys) -> None:
    calibration_config = tmp_path / "calibration-config.json"
    calibration_path = tmp_path / "calibration.json"
    mixture_config = tmp_path / "mixture-config.json"
    mixture_path = tmp_path / "mixture.json"
    calibration_config.write_text(json.dumps(_calibration_config()), encoding="utf-8")
    mixture_config.write_text(json.dumps(_mixture_config()), encoding="utf-8")

    assert main(
        [
            "fit-calibration",
            str(calibration_config),
            "--output",
            str(calibration_path),
        ]
    ) == 0
    fit_summary = json.loads(capsys.readouterr().out)
    assert fit_summary["report"]["group_count"] == 4
    assert fit_summary["report"]["log_loss_advantage_vs_uniform"] > 0.0

    assert main(["validate-calibration", str(calibration_path)]) == 0
    assert json.loads(capsys.readouterr().out) == fit_summary

    assert main(
        [
            "calibrate-mixture",
            str(calibration_path),
            str(mixture_config),
            "--output",
            str(mixture_path),
        ]
    ) == 0
    mixture_summary = json.loads(capsys.readouterr().out)
    calibration = load_material_identity_weight_calibration(calibration_path)
    mixture = load_material_identity_mixture(mixture_path)
    assert mixture.calibration_id == calibration.artifact_id
    assert mixture_summary["probabilities"][1] > mixture_summary["probabilities"][0]


def test_documented_calibration_and_mixture_examples(tmp_path: Path, capsys) -> None:
    calibration_path = tmp_path / "calibration.json"
    mixture_path = tmp_path / "mixture.json"

    assert main(
        [
            "fit-calibration",
            "docs/examples/material-identity-weight-calibration-input.json",
            "--output",
            str(calibration_path),
        ]
    ) == 0
    calibration_summary = json.loads(capsys.readouterr().out)
    assert calibration_summary["report"]["group_count"] == 4
    assert calibration_summary["report"]["cross_fitted_top1_accuracy"] == 1.0

    assert main(
        [
            "calibrate-mixture",
            str(calibration_path),
            "docs/examples/material-identity-calibrated-mixture-config.json",
            "--output",
            str(mixture_path),
        ]
    ) == 0
    mixture_summary = json.loads(capsys.readouterr().out)
    assert len(mixture_summary["probabilities"]) == 3
    assert sum(mixture_summary["probabilities"]) == pytest.approx(1.0)
    assert max(mixture_summary["probabilities"]) > 0.8
