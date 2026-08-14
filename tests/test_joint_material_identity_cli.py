import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.joint_material_identity import load_joint_material_identity_posterior
from prob4d.material_identity_cli import main
from prob4d.material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
    write_material_identity_mixture,
)

RULE_ID = "a" * 64
CALIBRATION_ID = "b" * 64
TRACKLET_REVISION = "c" * 40
ASSOCIATION_REVISION = "d" * 40
RESULT_ID = "e" * 64


def write_mixture(path: Path, *, target_track: int) -> None:
    mixture = MaterialIdentityMixtureV1(
        target_endpoint=LocalTrackEndpoint("window-1", target_track),
        window_order=("window-0", "window-1"),
        causal_frame_stop=22,
        association_rule_id=RULE_ID,
        calibration_id=CALIBRATION_ID,
        tracklet_producer_revision=TRACKLET_REVISION,
        association_revision=ASSOCIATION_REVISION,
        candidates=(
            MaterialIdentityCandidateV1(
                source_endpoint=None,
                association_result_id=None,
                source_score=None,
                calibrated_log_weight=0.0,
            ),
            MaterialIdentityCandidateV1(
                source_endpoint=LocalTrackEndpoint("window-0", 0),
                association_result_id=RESULT_ID,
                source_score=0.9,
                calibrated_log_weight=np.log(9.0),
            ),
        ),
    )
    write_material_identity_mixture(path, mixture)


def test_build_validate_and_marginalize_joint_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_mixture(tmp_path / "first.json", target_track=0)
    write_mixture(tmp_path / "second.json", target_track=1)
    config = {
        "window_order": ["window-0", "window-1"],
        "mixture_paths": ["first.json", "second.json"],
        "maximum_joint_assignments": 100,
        "metadata": {"purpose": "source-only"},
    }
    config_path = tmp_path / "joint-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "joint.json"

    assert main(["build-joint", str(config_path), "--output", str(output)]) == 0
    build_summary = json.loads(capsys.readouterr().out)
    assert build_summary["feasible_assignment_count"] == 3
    assert build_summary["rejected_assignment_count"] == 1
    posterior = load_joint_material_identity_posterior(output)

    assert main(["validate-joint", str(output)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["posterior_id"] == posterior.posterior_id

    likelihoods = {
        "assignment_ids": list(posterior.assignment_ids),
        "log_likelihoods": [-5.0, -1.0, -3.0],
        "likelihood_power": 1.0,
    }
    likelihood_path = tmp_path / "likelihoods.json"
    likelihood_path.write_text(json.dumps(likelihoods), encoding="utf-8")
    assert main(["marginalize-joint", str(output), str(likelihood_path)]) == 0
    marginal = json.loads(capsys.readouterr().out)
    assert marginal["assignment_ids"] == list(posterior.assignment_ids)
    assert len(marginal["marginals"]) == 2
    assert sum(marginal["posterior_probabilities"]) == pytest.approx(1.0)


def test_joint_cli_rejects_unconfined_and_duplicate_paths(tmp_path: Path) -> None:
    write_mixture(tmp_path / "first.json", target_track=0)
    config_path = tmp_path / "joint-config.json"
    config = {
        "window_order": ["window-0", "window-1"],
        "mixture_paths": ["../outside.json"],
        "maximum_joint_assignments": 100,
        "metadata": {},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="confined relative path"):
        main(
            [
                "build-joint",
                str(config_path),
                "--output",
                str(tmp_path / "joint.json"),
            ]
        )

    config["mixture_paths"] = ["first.json", "first.json"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        main(
            [
                "build-joint",
                str(config_path),
                "--output",
                str(tmp_path / "joint.json"),
            ]
        )
