import json

import numpy as np
import pytest

from prob4d.causal_tracklets import CausalTrackletSet
from prob4d.cross_window_tracklet_evidence import (
    CausalTrackletArtifactV1,
    associate_cross_window_tracklets_joint_gauge,
    joint_gauge_residual_covariance_m2,
    tracklet_content_id,
)
from prob4d.cross_window_tracklets import CrossWindowAssociationConfig
from prob4d.sim3 import Sim3

SOURCE_REVISION = "b" * 40
ASSOCIATION_REVISION = "c" * 40


def make_tracklets(
    window_id: str,
    tracks: list[np.ndarray],
    *,
    metadata: dict[str, object] | None = None,
) -> CausalTrackletSet:
    frame_count = tracks[0].shape[0]
    assert all(track.shape == (frame_count, 3) for track in tracks)
    frames = np.arange(1, frame_count + 1, dtype=np.int64)
    track_ids: list[int] = []
    frame_indices: list[int] = []
    local_indices: list[int] = []
    points: list[np.ndarray] = []
    for track_id, track in enumerate(tracks):
        for local_index, (frame, point) in enumerate(zip(frames, track, strict=True)):
            track_ids.append(track_id)
            frame_indices.append(int(frame))
            local_indices.append(local_index)
            points.append(point)
    count = len(track_ids)
    return CausalTrackletSet(
        window_id=window_id,
        causal_frame_stop=int(frames[-1]) + 1,
        source_shape=(frame_count, 1, 1),
        seed_frame_index=int(frames[0]),
        track_ids=np.asarray(track_ids, dtype=np.int64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        local_frame_indices=np.asarray(local_indices, dtype=np.int64),
        rows=np.zeros(count, dtype=np.int64),
        columns=np.zeros(count, dtype=np.int64),
        points_local=np.asarray(points, dtype=np.float64),
        link_probability=np.ones(count, dtype=np.float64),
        association_probability=np.ones(count, dtype=np.float64),
        metadata=metadata or {"test": True},
    )


def make_artifact(
    tracklets: CausalTrackletSet,
    *,
    manifest_digit: str,
    source_revision: str = SOURCE_REVISION,
) -> CausalTrackletArtifactV1:
    return CausalTrackletArtifactV1(
        tracklets=tracklets,
        prediction_manifest_id=manifest_digit * 64,
        source_revision=source_revision,
        builder_configuration={
            "seed_stride": 8,
            "nested": {"values": [1, 2, 3]},
        },
    )


def translation_joint_covariance(cross_covariance: float) -> np.ndarray:
    covariance = np.zeros((14, 14), dtype=np.float64)
    for axis in range(3):
        left = 4 + axis
        right = 11 + axis
        covariance[left, left] = 0.01
        covariance[right, right] = 0.01
        covariance[left, right] = cross_covariance
        covariance[right, left] = cross_covariance
    return covariance


def zero_covariance(tracklets: CausalTrackletSet) -> np.ndarray:
    return np.zeros((tracklets.observation_count, 3, 3), dtype=np.float64)


def test_tracklet_artifact_binds_complete_content_and_lineage() -> None:
    tracks = [np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float64)]
    first = make_artifact(
        make_tracklets("left", tracks, metadata={"nested": {"tag": [1, 2]}}),
        manifest_digit="a",
    )
    repeated = make_artifact(
        make_tracklets("left", tracks, metadata={"nested": {"tag": [1, 2]}}),
        manifest_digit="a",
    )
    changed_points = make_artifact(
        make_tracklets("left", [tracks[0] + np.array([0.001, 0.0, 0.0])]),
        manifest_digit="a",
    )
    changed_manifest = make_artifact(
        make_tracklets("left", tracks, metadata={"nested": {"tag": [1, 2]}}),
        manifest_digit="c",
    )

    assert first.tracklet_set_id == repeated.tracklet_set_id
    assert first.artifact_id == repeated.artifact_id
    assert first.tracklet_set_id != changed_points.tracklet_set_id
    assert first.artifact_id != changed_points.artifact_id
    assert first.tracklet_set_id == changed_manifest.tracklet_set_id
    assert first.artifact_id != changed_manifest.artifact_id
    assert tracklet_content_id(first.tracklets) == first.tracklet_set_id
    assert json.loads(json.dumps(first.to_dict()))["artifact_id"] == first.artifact_id
    with pytest.raises(TypeError, match="immutable"):
        first.builder_configuration["nested"]["values"].append(4)  # type: ignore[index,union-attr]


def test_tracklet_artifact_rejects_noncanonical_provenance() -> None:
    tracklets = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    with pytest.raises(TypeError, match="prediction_manifest_id"):
        CausalTrackletArtifactV1(
            tracklets=tracklets,
            prediction_manifest_id=True,  # type: ignore[arg-type]
            source_revision=SOURCE_REVISION,
        )
    with pytest.raises(ValueError, match="source_revision"):
        CausalTrackletArtifactV1(
            tracklets=tracklets,
            prediction_manifest_id="a" * 64,
            source_revision="B" * 40,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        CausalTrackletArtifactV1(
            tracklets=tracklets,
            prediction_manifest_id="a" * 64,
            source_revision=SOURCE_REVISION,
            builder_configuration={1: "non-string-key"},  # type: ignore[dict-item]
        )


def test_joint_gauge_residual_covariance_uses_cross_window_blocks() -> None:
    points = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]])
    local = np.zeros((2, 3, 3), dtype=np.float64)
    independent = joint_gauge_residual_covariance_m2(
        points,
        points,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        left_conditional_local_covariance_m2=local,
        right_conditional_local_covariance_m2=local,
        joint_gauge_covariance=translation_joint_covariance(0.0),
        left_gauge_index=0,
        right_gauge_index=1,
    )
    positively_correlated = joint_gauge_residual_covariance_m2(
        points,
        points,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        left_conditional_local_covariance_m2=local,
        right_conditional_local_covariance_m2=local,
        joint_gauge_covariance=translation_joint_covariance(0.009),
        left_gauge_index=0,
        right_gauge_index=1,
    )
    negatively_correlated = joint_gauge_residual_covariance_m2(
        points,
        points,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        left_conditional_local_covariance_m2=local,
        right_conditional_local_covariance_m2=local,
        joint_gauge_covariance=translation_joint_covariance(-0.009),
        left_gauge_index=0,
        right_gauge_index=1,
    )

    assert np.allclose(independent, 0.02 * np.eye(3)[None])
    assert np.allclose(positively_correlated, 0.002 * np.eye(3)[None])
    assert np.allclose(negatively_correlated, 0.038 * np.eye(3)[None])
    assert not positively_correlated.flags.writeable


def test_joint_gauge_association_uses_dependence_and_binds_evidence() -> None:
    left_tracklets = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right_tracklets = make_tracklets(
        "right",
        [np.array([[0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])],
    )
    left = make_artifact(left_tracklets, manifest_digit="a")
    right = make_artifact(right_tracklets, manifest_digit="c")
    config = CrossWindowAssociationConfig(
        covariance_floor_m2=1e-12,
        maximum_weighted_rms_m=0.2,
        maximum_shared_frame_distance_m=0.2,
        minimum_compatibility_score=0.0,
        minimum_score_margin=0.0,
    )
    settings = {
        "left_global_from_local": Sim3.identity(),
        "right_global_from_local": Sim3.identity(),
        "left_conditional_local_covariance_m2": zero_covariance(left_tracklets),
        "right_conditional_local_covariance_m2": zero_covariance(right_tracklets),
        "gauge_ids": ("left", "right"),
        "left_gauge_id": "left",
        "right_gauge_id": "right",
        "association_revision": ASSOCIATION_REVISION,
        "configuration": config,
    }
    independent = associate_cross_window_tracklets_joint_gauge(
        left,
        right,
        joint_gauge_covariance=translation_joint_covariance(0.0),
        candidate_chunk_size=1,
        **settings,
    )
    positively_correlated = associate_cross_window_tracklets_joint_gauge(
        left,
        right,
        joint_gauge_covariance=translation_joint_covariance(0.009),
        candidate_chunk_size=128,
        **settings,
    )

    assert independent.accepted_pairs == ((0, 0),)
    assert positively_correlated.accepted_pairs == ((0, 0),)
    assert independent.association.candidates[0].used_covariance
    assert positively_correlated.association.candidates[0].used_covariance
    assert (
        independent.association.candidates[0].compatibility_score
        == positively_correlated.association.candidates[0].compatibility_score
    )
    assert (
        independent.association.candidates[0].normalized_rms
        < positively_correlated.association.candidates[0].normalized_rms
    )
    assert independent.result_id != positively_correlated.result_id
    payload = json.loads(json.dumps(positively_correlated.to_dict()))
    assert payload["result_id"] == positively_correlated.result_id
    assert payload["left_tracklet_artifact_id"] == left.artifact_id
    assert payload["right_tracklet_artifact_id"] == right.artifact_id
    assert payload["dependence_semantics"] == "joint-cross-window-gauge-v1"
    assert payload["ranking_semantics"] == (
        "isotropic-geometric-mutual-best-joint-gauge-diagnostic-v1"
    )
    assert payload["conditional_point_cross_covariance_semantics"] == (
        "assumed-zero-unavailable-v1"
    )
    assert payload["tracklet_producer_revision"] == SOURCE_REVISION
    assert payload["association_revision"] == ASSOCIATION_REVISION


def test_covariance_inflation_cannot_improve_mutual_best_rank() -> None:
    left_tracklets = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right_tracklets = make_tracklets(
        "right",
        [
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
        ],
    )
    left = make_artifact(left_tracklets, manifest_digit="a")
    right = make_artifact(right_tracklets, manifest_digit="c")
    right_covariance = zero_covariance(right_tracklets)
    right_covariance[2:] = 1.0 * np.eye(3)
    result = associate_cross_window_tracklets_joint_gauge(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        left_conditional_local_covariance_m2=zero_covariance(left_tracklets),
        right_conditional_local_covariance_m2=right_covariance,
        gauge_ids=("left", "right"),
        joint_gauge_covariance=np.zeros((14, 14), dtype=np.float64),
        left_gauge_id="left",
        right_gauge_id="right",
        association_revision=ASSOCIATION_REVISION,
        configuration=CrossWindowAssociationConfig(
            covariance_floor_m2=1e-12,
            maximum_weighted_rms_m=0.05,
            maximum_shared_frame_distance_m=0.05,
            minimum_compatibility_score=0.0,
            minimum_score_margin=0.0,
        ),
    )

    assert result.accepted_pairs == ((0, 0),)
    precise, inflated = result.association.candidates
    assert precise.compatibility_score == inflated.compatibility_score
    assert precise.weighted_rms_m == inflated.weighted_rms_m
    assert precise.normalized_rms > inflated.normalized_rms


def test_joint_gauge_evidence_is_execution_chunk_invariant() -> None:
    left_tracklets = make_tracklets(
        "left",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        ],
    )
    right_tracklets = make_tracklets(
        "right",
        [
            np.array([[1.01, 0.0, 1.0], [1.01, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
        ],
    )
    left = make_artifact(left_tracklets, manifest_digit="a")
    right = make_artifact(right_tracklets, manifest_digit="c")
    settings = {
        "left_global_from_local": Sim3.identity(),
        "right_global_from_local": Sim3.identity(),
        "left_conditional_local_covariance_m2": zero_covariance(left_tracklets),
        "right_conditional_local_covariance_m2": zero_covariance(right_tracklets),
        "gauge_ids": ("left", "right"),
        "joint_gauge_covariance": translation_joint_covariance(0.0),
        "left_gauge_id": "left",
        "right_gauge_id": "right",
        "association_revision": ASSOCIATION_REVISION,
        "configuration": CrossWindowAssociationConfig(
            covariance_floor_m2=1e-12,
            maximum_shared_frame_distance_m=0.1,
            maximum_weighted_rms_m=0.05,
            minimum_score_margin=0.0,
        ),
    }

    small = associate_cross_window_tracklets_joint_gauge(
        left,
        right,
        candidate_chunk_size=1,
        **settings,
    )
    large = associate_cross_window_tracklets_joint_gauge(
        left,
        right,
        candidate_chunk_size=128,
        **settings,
    )

    assert small.accepted_pairs == ((0, 1), (1, 0))
    assert small.to_dict() == large.to_dict()
    assert small.result_id == large.result_id


def test_joint_gauge_association_fails_closed_on_lineage_and_covariance() -> None:
    left_tracklets = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right_tracklets = make_tracklets(
        "right",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    left = make_artifact(left_tracklets, manifest_digit="a")
    mismatched_right = make_artifact(
        right_tracklets,
        manifest_digit="c",
        source_revision="d" * 40,
    )
    settings = {
        "left_global_from_local": Sim3.identity(),
        "right_global_from_local": Sim3.identity(),
        "left_conditional_local_covariance_m2": zero_covariance(left_tracklets),
        "right_conditional_local_covariance_m2": zero_covariance(right_tracklets),
        "gauge_ids": ("left", "right"),
        "joint_gauge_covariance": translation_joint_covariance(0.0),
        "left_gauge_id": "left",
        "right_gauge_id": "right",
        "association_revision": ASSOCIATION_REVISION,
    }
    with pytest.raises(ValueError, match="exact source revision"):
        associate_cross_window_tracklets_joint_gauge(
            left,
            mismatched_right,
            **settings,
        )

    right = make_artifact(right_tracklets, manifest_digit="c")
    invalid = translation_joint_covariance(0.0)
    invalid[4, 4] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        associate_cross_window_tracklets_joint_gauge(
            left,
            right,
            **(settings | {"joint_gauge_covariance": invalid}),
        )
    with pytest.raises(ValueError, match="left_gauge_id"):
        associate_cross_window_tracklets_joint_gauge(
            left,
            right,
            **(settings | {"left_gauge_id": "wrong"}),
        )
    with pytest.raises(TypeError, match="canonical tuple"):
        associate_cross_window_tracklets_joint_gauge(
            left,
            right,
            **(settings | {"gauge_ids": {"left", "right"}}),
        )
    with pytest.raises(TypeError, match="association_revision"):
        associate_cross_window_tracklets_joint_gauge(
            left,
            right,
            **(settings | {"association_revision": True}),
        )
