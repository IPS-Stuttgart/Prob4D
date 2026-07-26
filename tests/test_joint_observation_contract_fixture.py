from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from prob4d.observation_validation import load_observation_belief_export

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "prob4d_joint_observation_v1.json"
)


def _artifact() -> tuple[ObservationBeliefExportV1, str]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    descriptor = payload["descriptor"]
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    artifact = ObservationBeliefExportV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **arrays,
    )
    return artifact, payload["expected_artifact_id"]


def test_joint_gauge_fixture_is_the_portable_producer_contract(
    tmp_path: Path,
) -> None:
    artifact, expected_artifact_id = _artifact()

    assert artifact.artifact_id == expected_artifact_id
    assert np.array_equal(np.unique(artifact.factor_group_ids), np.asarray([0]))
    assert artifact.metadata["gauge_posterior"]["model"] == (
        "sequential_joint_spanning_tree_v1"
    )
    assert artifact.metadata[
        "joint_cross_window_gauge_covariance_represented"
    ] is True

    cross_covariance = (
        artifact.low_rank_factor_m[0] @ artifact.low_rank_factor_m[2].T
    )
    assert cross_covariance[0, 0] != 0.0

    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, artifact)
    restored = load_observation_belief_export(path)
    assert restored.artifact_id == expected_artifact_id
    np.testing.assert_array_equal(
        restored.low_rank_factor_m,
        artifact.low_rank_factor_m,
    )
