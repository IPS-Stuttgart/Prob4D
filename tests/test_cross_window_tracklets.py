import json
from dataclasses import replace

import numpy as np
import pytest

from prob4d.causal_tracklets import CausalTrackletSet
from prob4d.cross_window_tracklets import (
    CrossWindowAssociationCandidate,
    CrossWindowAssociationConfig,
    CrossWindowAssociationLink,
    associate_cross_window_tracklets,
)
from prob4d.sim3 import Sim3


def make_tracklets(
    window_id: str,
    tracks: list[np.ndarray],
    *,
    frame_start: int = 1,
) -> CausalTrackletSet:
    frame_count = tracks[0].shape[0]
    assert all(track.shape == (frame_count, 3) for track in tracks)
    frames = np.arange(frame_start, frame_start + frame_count, dtype=np.int64)
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
        metadata={"test": True},
    )


def test_cross_window_association_recovers_swapped_window_local_ids() -> None:
    left = make_tracklets(
        "left",
        [
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [1.1, 0.0, 1.0], [1.2, 0.0, 1.0]]),
        ],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[-1.0, 0.0, 1.0], [-0.9, 0.0, 1.0], [-0.8, 0.0, 1.0]]),
            np.array([[-2.0, 0.0, 1.0], [-1.9, 0.0, 1.0], [-1.8, 0.0, 1.0]]),
        ],
    )

    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3(translation=np.array([2.0, 0.0, 0.0])),
        configuration=CrossWindowAssociationConfig(
            isotropic_distance_scale_m=0.05,
            maximum_weighted_rms_m=0.05,
            minimum_compatibility_score=0.5,
            minimum_score_margin=0.2,
        ),
    )

    assert result.accepted_pairs == ((0, 1), (1, 0))
    assert result.unmatched_left_track_ids == ()
    assert result.unmatched_right_track_ids == ()
    assert all(link.compatibility_score == pytest.approx(1.0) for link in result.links)
    assert result.descriptor()["schema_version"] == 2


def test_cross_window_association_rejects_an_ambiguous_mutual_best() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
            np.array([[-0.01, 0.0, 1.0], [-0.01, 0.0, 1.0]]),
        ],
    )

    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=CrossWindowAssociationConfig(
            isotropic_distance_scale_m=0.1,
            maximum_weighted_rms_m=0.1,
            minimum_score_margin=0.05,
        ),
    )

    assert result.links == ()
    assert result.ambiguous_mutual_best_count == 1
    assert result.unmatched_left_track_ids == (0,)
    assert result.unmatched_right_track_ids == (0, 1)


def test_cross_window_association_uses_global_covariance_when_supplied() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right = make_tracklets(
        "right",
        [np.array([[0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])],
    )
    tight = np.repeat((1e-4 * np.eye(3))[None], 2, axis=0)
    loose = np.repeat((0.1 * np.eye(3))[None], 2, axis=0)
    config = CrossWindowAssociationConfig(
        maximum_weighted_rms_m=0.2,
        maximum_shared_frame_distance_m=0.2,
        minimum_compatibility_score=0.0,
        minimum_score_margin=0.0,
    )

    tight_result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=config,
        left_global_covariance_m2=tight,
        right_global_covariance_m2=tight,
    )
    loose_result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=config,
        left_global_covariance_m2=loose,
        right_global_covariance_m2=loose,
    )

    tight_candidate = tight_result.candidates[0]
    loose_candidate = loose_result.candidates[0]
    assert tight_candidate.used_covariance
    assert loose_candidate.used_covariance
    assert loose_candidate.compatibility_score > tight_candidate.compatibility_score
    assert loose_candidate.normalized_rms < tight_candidate.normalized_rms


def test_covariance_score_uses_reduced_mahalanobis_rms() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])],
    )
    right = make_tracklets(
        "right",
        [np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])],
    )
    half_identity = np.repeat((0.5 * np.eye(3))[None], 2, axis=0)
    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=CrossWindowAssociationConfig(
            covariance_floor_m2=1e-15,
            maximum_weighted_rms_m=2.0,
            maximum_shared_frame_distance_m=2.0,
            minimum_compatibility_score=0.0,
            minimum_score_margin=0.0,
        ),
        left_global_covariance_m2=half_identity,
        right_global_covariance_m2=half_identity,
    )

    assert result.candidates[0].normalized_rms == pytest.approx(1.0)


def test_nonshared_causal_suffix_does_not_change_association() -> None:
    left_short = make_tracklets(
        "left-short",
        [np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])],
    )
    left_long = make_tracklets(
        "left-long",
        [
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [10_000.0, 0.0, 1.0],
                ]
            )
        ],
    )
    right = make_tracklets(
        "right",
        [np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])],
    )
    settings = {
        "right": right,
        "left_global_from_local": Sim3.identity(),
        "right_global_from_local": Sim3.identity(),
    }

    short_result = associate_cross_window_tracklets(left_short, **settings)
    long_result = associate_cross_window_tracklets(left_long, **settings)

    assert short_result.candidates[0].shared_frame_indices == (1, 2)
    assert long_result.candidates[0].shared_frame_indices == (1, 2)
    assert short_result.candidates[0].compatibility_score == pytest.approx(
        long_result.candidates[0].compatibility_score
    )
    assert short_result.candidates[0].weighted_rms_m == pytest.approx(
        long_result.candidates[0].weighted_rms_m
    )


def test_cross_window_configuration_rejects_boolean_numeric_aliases() -> None:
    with pytest.raises(ValueError, match="minimum_shared_frames"):
        CrossWindowAssociationConfig(minimum_shared_frames=True)
    with pytest.raises(ValueError, match="minimum_score_margin"):
        CrossWindowAssociationConfig(minimum_score_margin=False)
    with pytest.raises(ValueError, match="maximum_spatial_candidate_pairs"):
        CrossWindowAssociationConfig(maximum_spatial_candidate_pairs=True)


def test_candidate_and_link_contracts_reject_inconsistent_direct_construction() -> None:
    candidate_kwargs = {
        "left_track_id": 0,
        "right_track_id": 1,
        "shared_frame_indices": (1, 2),
        "effective_support": 2.0,
        "weighted_rms_m": 0.01,
        "maximum_distance_m": 0.02,
        "normalized_rms": 0.5,
        "compatibility_score": 0.8,
        "used_covariance": False,
    }
    with pytest.raises(ValueError, match="left_track_id"):
        CrossWindowAssociationCandidate(**{**candidate_kwargs, "left_track_id": True})
    with pytest.raises(ValueError, match="strictly increasing"):
        CrossWindowAssociationCandidate(**{**candidate_kwargs, "shared_frame_indices": (2, 1)})
    with pytest.raises(ValueError, match="weighted_rms_m"):
        CrossWindowAssociationCandidate(
            **{
                **candidate_kwargs,
                "weighted_rms_m": 0.03,
                "maximum_distance_m": 0.02,
            }
        )
    with pytest.raises(ValueError, match="used_covariance"):
        CrossWindowAssociationCandidate(**{**candidate_kwargs, "used_covariance": 1})

    with pytest.raises(ValueError, match="score margins"):
        CrossWindowAssociationLink(
            left_track_id=0,
            right_track_id=1,
            shared_frame_indices=(1, 2),
            compatibility_score=0.5,
            left_score_margin=0.6,
            right_score_margin=0.4,
        )


def test_result_contract_rejects_inconsistent_manual_accounting() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right = make_tracklets(
        "right",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
    )

    with pytest.raises(ValueError, match="evaluated_track_pair_count"):
        replace(result, evaluated_track_pair_count=result.evaluated_track_pair_count + 1)
    with pytest.raises(ValueError, match="unmatched_left_track_ids"):
        replace(result, unmatched_left_track_ids=(0,))
    with pytest.raises(ValueError, match="sorted unique"):
        replace(result, links=(result.links[0], result.links[0]))


def test_evaluated_pair_count_excludes_insufficient_overlap_pairs() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
        frame_start=1,
    )
    right = make_tracklets(
        "right",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
        frame_start=2,
    )

    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=CrossWindowAssociationConfig(minimum_shared_frames=2),
    )

    assert result.spatial_candidate_pair_count == 1
    assert result.insufficient_shared_frame_pair_count == 1
    assert result.evaluated_track_pair_count == 0
    assert result.candidates == ()


def test_spatial_gate_avoids_exhaustive_distractor_pairs() -> None:
    left = make_tracklets(
        "left",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[10.0, 0.0, 1.0], [10.0, 0.0, 1.0]]),
        ],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
            np.array([[10.01, 0.0, 1.0], [10.01, 0.0, 1.0]]),
            np.array([[100.0, 0.0, 1.0], [100.0, 0.0, 1.0]]),
        ],
    )

    result = associate_cross_window_tracklets(
        left,
        right,
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=CrossWindowAssociationConfig(
            maximum_shared_frame_distance_m=0.1,
            maximum_weighted_rms_m=0.05,
        ),
        candidate_chunk_size=1,
    )

    assert result.possible_track_pair_count == 6
    assert result.spatial_candidate_pair_count == 2
    assert result.spatially_rejected_pair_count == 4
    assert result.evaluated_track_pair_count == 2
    assert result.accepted_pairs == ((0, 0), (1, 1))


def test_two_axis_chunking_is_result_and_identity_invariant() -> None:
    left = make_tracklets(
        "left",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            np.array([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        ],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[2.01, 0.0, 1.0], [2.01, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
            np.array([[1.01, 0.0, 1.0], [1.01, 0.0, 1.0]]),
        ],
    )
    settings = {
        "left_global_from_local": Sim3.identity(),
        "right_global_from_local": Sim3.identity(),
        "configuration": CrossWindowAssociationConfig(
            maximum_shared_frame_distance_m=0.1,
            maximum_weighted_rms_m=0.05,
        ),
    }

    small = associate_cross_window_tracklets(
        left,
        right,
        candidate_chunk_size=1,
        **settings,
    )
    large = associate_cross_window_tracklets(
        left,
        right,
        candidate_chunk_size=128,
        **settings,
    )

    assert small.accepted_pairs == ((0, 1), (1, 2), (2, 0))
    assert small.to_dict() == large.to_dict()
    assert small.result_id == large.result_id
    assert json.loads(json.dumps(small.to_dict()))["result_id"] == small.result_id


def test_cross_window_association_rejects_one_sided_or_invalid_covariance() -> None:
    left = make_tracklets(
        "left",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    right = make_tracklets(
        "right",
        [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])],
    )
    covariance = np.repeat(np.eye(3)[None], 2, axis=0)

    with pytest.raises(ValueError, match="both windows or neither"):
        associate_cross_window_tracklets(
            left,
            right,
            left_global_from_local=Sim3.identity(),
            right_global_from_local=Sim3.identity(),
            left_global_covariance_m2=covariance,
        )

    invalid = covariance.copy()
    invalid[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        associate_cross_window_tracklets(
            left,
            right,
            left_global_from_local=Sim3.identity(),
            right_global_from_local=Sim3.identity(),
            left_global_covariance_m2=invalid,
            right_global_covariance_m2=covariance,
        )


def test_spatial_candidate_cap_fails_closed() -> None:
    left = make_tracklets(
        "left",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
        ],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 1.0], [0.01, 0.0, 1.0]]),
        ],
    )

    with pytest.raises(ValueError, match="maximum_spatial_candidate_pairs"):
        associate_cross_window_tracklets(
            left,
            right,
            left_global_from_local=Sim3.identity(),
            right_global_from_local=Sim3.identity(),
            configuration=CrossWindowAssociationConfig(
                maximum_shared_frame_distance_m=0.1,
                maximum_weighted_rms_m=0.05,
                maximum_spatial_candidate_pairs=3,
            ),
            candidate_chunk_size=1,
        )


def test_exhaustive_candidate_mode_is_also_capped() -> None:
    left = make_tracklets(
        "left",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        ],
    )
    right = make_tracklets(
        "right",
        [
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        ],
    )

    with pytest.raises(ValueError, match="maximum_spatial_candidate_pairs"):
        associate_cross_window_tracklets(
            left,
            right,
            left_global_from_local=Sim3.identity(),
            right_global_from_local=Sim3.identity(),
            configuration=CrossWindowAssociationConfig(
                maximum_shared_frame_distance_m=None,
                maximum_spatial_candidate_pairs=3,
            ),
        )
