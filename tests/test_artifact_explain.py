from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.artifact_explain import explain_artifact, main, render_text
from prob4d.cli import main as grouped_main
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.gauge_tree_prior_artifact import write_gauge_tree_prior_artifact
from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)


def _observation() -> ObservationBeliefExportV1:
    return ObservationBeliefExportV1(
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0",),
        factor_names=(),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]]),
        frame_ids=np.asarray([8]),
        entity_ids=np.asarray([0]),
        view_indices=np.asarray([0]),
        window_indices=np.asarray([0]),
        correlation_group_ids=np.asarray([0]),
        factor_group_ids=np.asarray([0]),
        prior_reliability=np.asarray([0.9]),
        association_probability=np.asarray([1.0]),
        local_covariance_m2=np.eye(3)[None] * 1e-4,
        low_rank_factor_m=np.zeros((1, 3, 0)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([0.9]),
        group_composite_weight=np.asarray([1.0]),
        metadata={"causal_source": "prefix only"},
    )


def _gauge_prior() -> GaugeTreeSquareRootPriorV1:
    transitions = np.zeros((2, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7) * 0.8
    scales = np.repeat(np.eye(7, dtype=np.float64)[None], 2, axis=0) * 0.1
    return GaugeTreeSquareRootPriorV1(
        gauge_ids=("gauge-0", "gauge-1"),
        parent_indices=np.asarray([-1, 0]),
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        source_joint_covariance_sha256="c" * 64,
    )


def test_explain_observation_uses_strict_loader(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    artifact = _observation()
    save_observation_belief_export(path, artifact)

    explanation = explain_artifact(path, require_strict=True)

    assert explanation["status"] == "valid"
    assert explanation["artifact_kind"] == "observation-belief-v1"
    assert explanation["identity"]["artifact_id"] == artifact.artifact_id
    assert explanation["context"]["case_id"] == "case-1"
    assert explanation["summary"]["observation_count"] == 1
    assert "Members:" not in render_text(explanation)


def test_explain_observation_can_include_array_inventory(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, _observation())

    explanation = explain_artifact(path, include_arrays=True)

    members = {member["name"]: member for member in explanation["members"]}
    assert members["mean_xyz_m"]["shape"] == [1, 3]
    assert members["mean_xyz_m"]["dtype"] == "<f8"
    assert members["descriptor_json"]["shape"] == []


def test_explain_gauge_tree_prior_uses_sidecar_validation(tmp_path: Path) -> None:
    path = tmp_path / "prior.json"
    loaded = write_gauge_tree_prior_artifact(_gauge_prior(), path)

    explanation = explain_artifact(path, require_strict=True)

    assert explanation["status"] == "valid"
    assert explanation["artifact_kind"] == "gauge-tree-prior-artifact-v1"
    assert explanation["identity"]["artifact_id"] == loaded.manifest.artifact_id
    assert explanation["summary"]["gauge_count"] == 2


def test_unknown_json_is_never_reported_as_valid(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text(
        json.dumps(
            {
                "schema": "example.unknown",
                "schema_version": 1,
                "artifact_id": "d" * 64,
                "case_id": "case-unknown",
            }
        ),
        encoding="utf-8",
    )

    explanation = explain_artifact(path)

    assert explanation["status"] == "structural-only"
    assert explanation["validation_scope"] == "strict-JSON-object-syntax-only"
    assert explanation["identity"]["artifact_id"] == "d" * 64
    assert explanation["warnings"]


def test_require_strict_rejects_unknown_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"schema":"example.unknown","schema_version":1}', encoding="utf-8")

    assert main([str(path), "--require-strict"]) == 2
    assert "no registered strict loader" in capsys.readouterr().err


def test_json_parser_rejects_duplicate_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")

    assert main([str(path)]) == 2
    assert "duplicate JSON object key" in capsys.readouterr().err


def test_npz_descriptor_error_is_not_hidden(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "duplicate-descriptor.npz"
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            '{"schema_name":"first","schema_name":"second"}'
        ),
    )

    assert main([str(path)]) == 2
    assert "duplicate JSON object key" in capsys.readouterr().err


def test_grouped_cli_prints_machine_readable_explanation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, _observation())

    assert grouped_main(["artifact", "explain", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["artifact_kind"] == "observation-belief-v1"


def test_grouped_cli_help_is_registered(capsys: pytest.CaptureFixture[str]) -> None:
    assert grouped_main(["artifact", "--help"]) == 0
    output = capsys.readouterr().out
    assert "explain" in output
