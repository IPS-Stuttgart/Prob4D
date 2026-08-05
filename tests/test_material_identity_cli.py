from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.material_identity_cli import main, mixture_from_config
from prob4d.material_identity_mixture import load_material_identity_mixture
from prob4d.material_identity_stream import (
    create_material_identity_stream,
    write_material_identity_stream,
)


def _config() -> dict[str, object]:
    return {
        "target_endpoint": {"window_id": "window-2", "track_id": 3},
        "window_order": ["window-0", "window-1", "window-2"],
        "causal_frame_stop": 75,
        "association_rule_id": "a" * 64,
        "calibration_id": "b" * 64,
        "tracklet_producer_revision": "c" * 40,
        "association_revision": "d" * 40,
        "candidates": [
            {
                "source_endpoint": None,
                "association_result_id": None,
                "source_score": None,
                "calibrated_log_weight": 0.0,
                "metadata": {"fallback": True},
            },
            {
                "source_endpoint": {"window_id": "window-1", "track_id": 7},
                "association_result_id": "e" * 64,
                "source_score": 0.8,
                "calibrated_log_weight": float(np.log(3.0)),
                "metadata": {"source_only": True},
            },
        ],
        "metadata": {"claim_bearing": False},
    }


def test_documented_mixture_configuration_is_valid() -> None:
    config = json.loads(
        Path("docs/examples/material-identity-mixture-config.json").read_text(
            encoding="utf-8"
        )
    )

    mixture = mixture_from_config(config)

    assert mixture.target_endpoint.window_id == "window-002"
    assert mixture.null_probability == pytest.approx(0.25)


def test_build_and_validate_mixture_cli(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "mixture-config.json"
    mixture_path = tmp_path / "mixture.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")

    assert main(["build-mixture", str(config_path), "--output", str(mixture_path)]) == 0
    build_summary = json.loads(capsys.readouterr().out)
    assert build_summary["null_probability"] == 0.25

    assert main(["validate-mixture", str(mixture_path)]) == 0
    validate_summary = json.loads(capsys.readouterr().out)
    assert validate_summary == build_summary
    assert load_material_identity_mixture(mixture_path).mixture_id == build_summary["mixture_id"]


def test_marginalize_and_moment_match_cli(tmp_path: Path, capsys) -> None:
    mixture = mixture_from_config(_config())
    config_path = tmp_path / "mixture-config.json"
    mixture_path = tmp_path / "mixture.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    assert main(["build-mixture", str(config_path), "--output", str(mixture_path)]) == 0
    capsys.readouterr()

    likelihood_path = tmp_path / "likelihoods.json"
    likelihood_path.write_text(
        json.dumps(
            {
                "candidate_ids": list(mixture.candidate_ids),
                "log_likelihoods": [0.0, float(np.log(2.0))],
                "likelihood_power": 1.0,
            }
        ),
        encoding="utf-8",
    )
    assert main(["marginalize", str(mixture_path), str(likelihood_path)]) == 0
    marginalized = json.loads(capsys.readouterr().out)
    assert marginalized["posterior_probabilities"] == pytest.approx(
        [1.0 / 7.0, 6.0 / 7.0]
    )

    hypotheses_path = tmp_path / "hypotheses.json"
    hypotheses_path.write_text(
        json.dumps(
            {
                "candidate_ids": list(mixture.candidate_ids),
                "means": [[0.0], [2.0]],
                "covariances": [[[1.0]], [[1.0]]],
                "probabilities": [0.25, 0.75],
            }
        ),
        encoding="utf-8",
    )
    assert main(["moment-match", str(mixture_path), str(hypotheses_path)]) == 0
    matched = json.loads(capsys.readouterr().out)
    assert matched["mean"] == [1.5]
    assert matched["within_hypothesis_covariance"] == [[1.0]]
    assert matched["between_hypothesis_covariance"] == [[0.75]]
    assert matched["covariance"] == [[1.75]]


def test_validate_stream_cli(tmp_path: Path, capsys) -> None:
    stream = create_material_identity_stream(
        sequence_id="sequence-1",
        case_id="case-1",
        stream_id="camera0",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        root_window_id="window-0",
    )
    stream_path = tmp_path / "stream.json"
    write_material_identity_stream(stream, stream_path)

    assert main(["validate-stream", str(stream_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["artifact_id"] == stream.artifact_id
    assert summary["admitted_window_ids"] == ["window-0"]
    assert summary["update_count"] == 0
