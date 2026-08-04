import json
from dataclasses import replace

import numpy as np
import pytest

from prob4d.causal_tracklets import CausalTrackletSet
from prob4d.cross_window_tracklets import (
    CrossWindowAssociationConfig,
    associate_cross_window_tracklets,
)
from prob4d.material_identity_stream import (
    MaterialIdentityHypothesisStreamV1,
    append_material_identity_update,
    association_summary_from_result,
    create_material_identity_stream,
    load_material_identity_stream,
    write_material_identity_stream,
)
from prob4d.sim3 import Sim3


def make_tracklets(window_id: str, x_values: list[float]) -> CausalTrackletSet:
    frames = np.array([1, 2], dtype=np.int64)
    track_ids: list[int] = []
    frame_indices: list[int] = []
    local_indices: list[int] = []
    points: list[list[float]] = []
    for track_id, x_value in enumerate(x_values):
        for local_index, frame in enumerate(frames):
            track_ids.append(track_id)
            frame_indices.append(int(frame))
            local_indices.append(local_index)
            points.append([x_value, 0.0, 1.0])
    count = len(track_ids)
    return CausalTrackletSet(
        window_id=window_id,
        causal_frame_stop=3,
        source_shape=(2, 1, 1),
        seed_frame_index=1,
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


def associate(source: str, target: str, source_x: list[float], target_x: list[float]):
    return associate_cross_window_tracklets(
        make_tracklets(source, source_x),
        make_tracklets(target, target_x),
        left_global_from_local=Sim3.identity(),
        right_global_from_local=Sim3.identity(),
        configuration=CrossWindowAssociationConfig(
            isotropic_distance_scale_m=0.05,
            maximum_weighted_rms_m=0.1,
            maximum_shared_frame_distance_m=0.2,
            minimum_compatibility_score=0.1,
            minimum_score_margin=0.05,
        ),
    )


def root_stream() -> MaterialIdentityHypothesisStreamV1:
    return create_material_identity_stream(
        sequence_id="sequence",
        case_id="case",
        stream_id="camera0",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="2d6df37",
        root_window_id="w0",
        metadata={"claim_bearing": False},
    )


def test_append_preserves_local_ids_and_pairwise_gate_semantics() -> None:
    result = associate("w0", "w1", [0.0, 1.0], [0.01, 1.01])
    stream = append_material_identity_update(
        root_stream(),
        [result],
        target_window_id="w1",
    )

    assert stream.admitted_window_ids == ("w0", "w1")
    assert stream.hypothesis_count == 2
    summary = stream.updates[0].associations[0]
    assert [
        (value.source_track_id, value.target_track_id)
        for value in summary.hypotheses
    ] == [(0, 0), (1, 1)]
    assert all(value.selected_by_pairwise_gate for value in summary.hypotheses)
    assert summary.unmatched_source_track_ids == ()
    assert summary.unmatched_target_track_ids == ()
    assert stream.root_window_id == "w0"
    assert "global" not in json.dumps(stream.to_record()).lower()


def test_ambiguous_alternatives_are_retained_without_forced_identity() -> None:
    result = associate("w0", "w1", [0.0], [0.01, -0.01])
    stream = append_material_identity_update(
        root_stream(),
        [result],
        target_window_id="w1",
    )

    summary = stream.updates[0].associations[0]
    assert len(summary.hypotheses) == 2
    assert summary.selected_hypotheses == ()
    assert summary.unmatched_source_track_ids == (0,)
    assert summary.unmatched_target_track_ids == (0, 1)
    assert summary.ambiguous_mutual_best_count == 1


def test_association_summary_requires_directed_source_to_new_target_orientation() -> None:
    result = associate("w0", "w1", [0.0], [0.0])

    with pytest.raises(ValueError, match="right window"):
        association_summary_from_result(result, target_window_id="w0")


def test_append_rejects_future_sources_repeated_targets_and_duplicate_sources() -> None:
    first = append_material_identity_update(
        root_stream(),
        [associate("w0", "w1", [0.0], [0.0])],
        target_window_id="w1",
    )

    with pytest.raises(ValueError, match="already admitted"):
        append_material_identity_update(
            first,
            [associate("w0", "w1", [0.0], [0.0])],
            target_window_id="w1",
        )
    with pytest.raises(ValueError, match="non-admitted source"):
        append_material_identity_update(
            first,
            [associate("future", "w2", [0.0], [0.0])],
            target_window_id="w2",
        )
    duplicate = associate("w0", "w2", [0.0], [0.0])
    with pytest.raises(ValueError, match="repeat a source"):
        append_material_identity_update(
            first,
            [duplicate, duplicate],
            target_window_id="w2",
        )


def test_multi_parent_update_is_sorted_acyclic_and_append_invariant() -> None:
    first = append_material_identity_update(
        root_stream(),
        [associate("w0", "w1", [0.0], [0.0])],
        target_window_id="w1",
    )
    second = append_material_identity_update(
        first,
        [
            associate("w1", "w2", [0.0], [0.0]),
            associate("w0", "w2", [0.0], [0.0]),
        ],
        target_window_id="w2",
    )

    assert second.admitted_window_ids == ("w0", "w1", "w2")
    assert second.updates[1].source_window_ids == ("w0", "w1")
    assert second.updates[0] == first.updates[0]
    assert second.updates[1].previous_update_id == first.updates[0].update_id
    assert second.artifact_id != first.artifact_id


def test_summary_update_and_stream_tampering_fail_closed() -> None:
    stream = append_material_identity_update(
        root_stream(),
        [associate("w0", "w1", [0.0], [0.0])],
        target_window_id="w1",
    )
    summary = stream.updates[0].associations[0]
    with pytest.raises(ValueError, match="evaluated_track_pair_count"):
        replace(
            summary,
            evaluated_track_pair_count=summary.evaluated_track_pair_count + 1,
            summary_id=None,
        )
    with pytest.raises(ValueError, match="hash chain"):
        replace(
            stream,
            updates=(
                replace(
                    stream.updates[0],
                    previous_update_id="0" * 64,
                    update_id=None,
                ),
            ),
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        replace(stream, artifact_id="0" * 64)


def test_round_trip_revalidates_nested_ids_and_metadata(tmp_path) -> None:
    stream = append_material_identity_update(
        root_stream(),
        [associate("w0", "w1", [0.0], [0.0])],
        target_window_id="w1",
    )
    path = write_material_identity_stream(stream, tmp_path / "stream.json")
    loaded = load_material_identity_stream(path)

    assert loaded == stream
    assert loaded.artifact_id == stream.artifact_id
    with pytest.raises(TypeError):
        loaded.metadata["new"] = True


def test_loader_rejects_unknown_fields_duplicate_keys_and_tampered_ids(tmp_path) -> None:
    stream = append_material_identity_update(
        root_stream(),
        [associate("w0", "w1", [0.0], [0.0])],
        target_window_id="w1",
    )
    path = write_material_identity_stream(stream, tmp_path / "stream.json")
    record = json.loads(path.read_text())
    record["unknown"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unknown"):
        load_material_identity_stream(path)

    valid = json.dumps(stream.to_record())
    path.write_text('{"sequence_id":"duplicate",' + valid[1:])
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_material_identity_stream(path)

    record = stream.to_record()
    record["updates"][0]["update_id"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="update ID mismatch"):
        load_material_identity_stream(path)


def test_root_contract_rejects_coercion_aliases() -> None:
    with pytest.raises(ValueError, match="sequence_id"):
        create_material_identity_stream(
            sequence_id=1,
            case_id="case",
            stream_id="camera0",
            source_repository="IPS-Stuttgart/Prob4D",
            source_revision="revision",
            root_window_id="w0",
        )
