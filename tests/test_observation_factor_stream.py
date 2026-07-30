from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate
from prob4d.observation_factor_stream import (
    OBSERVATION_FACTOR_STREAM_SCHEMA,
    append_observation_factor_bundle,
    load_observation_factor_stream,
    write_observation_factor_stream,
)
from prob4d.observation_factors import (
    ObservationFactor,
    ObservationFactorBundle,
    write_observation_factor_bundle,
)
from prob4d.sim3 import Sim3


def _write_delta_bundle(
    root: Path,
    *,
    name: str,
    frame_index: int,
    causal_frame_stop: int,
    point_id: int,
) -> Path:
    gauge = GaugeEstimate(
        "window-0",
        Sim3.identity(),
        np.eye(7, dtype=np.float64) * 1e-4,
    )
    factor = ObservationFactor(
        factor_id=f"{name}:frame-{frame_index}",
        frame_index=frame_index,
        view_id="camera-0",
        window_id="window-0",
        gauge_id="window-0",
        point_ids=np.asarray([point_id], dtype=np.int64),
        points_local_m=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        valid_mask=np.asarray([True]),
        local_covariance_m2=np.asarray([np.eye(3) * 1e-3]),
        association_probability=np.asarray([0.9]),
        prior_reliability=np.asarray([0.8]),
        prior_nominal_probability=0.95,
        composite_weight=1.0,
        correlation_group_id=f"camera-0:frame-{frame_index}",
        causal_frame_stop=causal_frame_stop,
    )
    bundle = ObservationFactorBundle(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tracklets:camera-0",
        factors=(factor,),
        gauges=(gauge,),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        causal_frame_stop=causal_frame_stop,
        joint_gauge_covariance=gauge.covariance,
        gauge_covariance_semantics="joint-cross-window",
    )
    manifest, _ = write_observation_factor_bundle(bundle, root / f"{name}.json")
    return manifest


def test_stream_roundtrip_binds_disjoint_bundle_updates(tmp_path: Path) -> None:
    stream_path = tmp_path / "stream.json"
    first = _write_delta_bundle(
        tmp_path,
        name="update-0",
        frame_index=1,
        causal_frame_stop=3,
        point_id=7,
    )
    second = _write_delta_bundle(
        tmp_path,
        name="update-1",
        frame_index=4,
        causal_frame_stop=6,
        point_id=7,
    )

    stream = append_observation_factor_bundle(
        None,
        first,
        stream_manifest_path=stream_path,
        admitted_frame_start=0,
        metadata={"protocol": "prefix-updates-v1"},
    )
    stream = append_observation_factor_bundle(
        stream,
        second,
        stream_manifest_path=stream_path,
    )
    write_observation_factor_stream(stream, stream_path)
    loaded = load_observation_factor_stream(stream_path)
    record = json.loads(stream_path.read_text(encoding="utf-8"))

    assert record["schema"] == OBSERVATION_FACTOR_STREAM_SCHEMA
    assert loaded.artifact_id == stream.artifact_id
    assert loaded.admitted_frame_start == 0
    assert loaded.causal_frame_stop == 6
    assert loaded.factor_count == 2
    assert loaded.observation_count == 2
    assert loaded.updates[0].previous_update_id is None
    assert loaded.updates[1].previous_update_id == loaded.updates[0].update_id
    assert loaded.updates[0].persistent_identity_count == 1
    assert loaded.updates[0].bundle_manifest_path == "update-0.json"
    assert loaded.updates[1].bundle_manifest_path == "update-1.json"


def test_stream_rejects_reintroduced_frame(tmp_path: Path) -> None:
    stream_path = tmp_path / "stream.json"
    first = _write_delta_bundle(
        tmp_path,
        name="update-0",
        frame_index=1,
        causal_frame_stop=3,
        point_id=7,
    )
    overlapping = _write_delta_bundle(
        tmp_path,
        name="overlap",
        frame_index=2,
        causal_frame_stop=5,
        point_id=7,
    )
    stream = append_observation_factor_bundle(
        None,
        first,
        stream_manifest_path=stream_path,
        admitted_frame_start=0,
    )

    with pytest.raises(ValueError, match="already admitted frame"):
        append_observation_factor_bundle(
            stream,
            overlapping,
            stream_manifest_path=stream_path,
        )


def test_stream_rejects_changed_bundle_after_sealing(tmp_path: Path) -> None:
    stream_path = tmp_path / "stream.json"
    bundle = _write_delta_bundle(
        tmp_path,
        name="update-0",
        frame_index=1,
        causal_frame_stop=3,
        point_id=7,
    )
    stream = append_observation_factor_bundle(
        None,
        bundle,
        stream_manifest_path=stream_path,
        admitted_frame_start=0,
    )
    write_observation_factor_stream(stream, stream_path)
    bundle.write_text(bundle.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no longer matches its bundle"):
        load_observation_factor_stream(stream_path)


def test_stream_rejects_path_traversal_even_when_identity_is_unchanged(
    tmp_path: Path,
) -> None:
    stream_path = tmp_path / "stream.json"
    bundle = _write_delta_bundle(
        tmp_path,
        name="update-0",
        frame_index=1,
        causal_frame_stop=3,
        point_id=7,
    )
    stream = append_observation_factor_bundle(
        None,
        bundle,
        stream_manifest_path=stream_path,
        admitted_frame_start=0,
    )
    write_observation_factor_stream(stream, stream_path)
    record = json.loads(stream_path.read_text(encoding="utf-8"))
    record["updates"][0]["bundle_manifest_path"] = "../update-0.json"
    stream_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        load_observation_factor_stream(stream_path)


def test_stream_rejects_broken_update_chain(tmp_path: Path) -> None:
    stream_path = tmp_path / "stream.json"
    first = _write_delta_bundle(
        tmp_path,
        name="update-0",
        frame_index=1,
        causal_frame_stop=3,
        point_id=7,
    )
    second = _write_delta_bundle(
        tmp_path,
        name="update-1",
        frame_index=4,
        causal_frame_stop=6,
        point_id=7,
    )
    stream = append_observation_factor_bundle(
        None,
        first,
        stream_manifest_path=stream_path,
        admitted_frame_start=0,
    )
    stream = append_observation_factor_bundle(
        stream,
        second,
        stream_manifest_path=stream_path,
    )
    write_observation_factor_stream(stream, stream_path)
    record = json.loads(stream_path.read_text(encoding="utf-8"))
    record["updates"][1]["previous_update_id"] = "0" * 64
    stream_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError):
        load_observation_factor_stream(stream_path, validate_bundles=False)
