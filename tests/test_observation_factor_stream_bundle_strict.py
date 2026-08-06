from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge import GaugeEstimate
from prob4d.observation_factor_stream import append_observation_factor_bundle
from prob4d.observation_factors import (
    ObservationFactor,
    ObservationFactorBundle,
    write_observation_factor_bundle,
)
from prob4d.sim3 import Sim3


def _write_delta_bundle(root: Path) -> Path:
    gauge = GaugeEstimate(
        "window-0",
        Sim3.identity(),
        np.eye(7, dtype=np.float64) * 1e-4,
    )
    factor = ObservationFactor(
        factor_id="update-0:frame-1",
        frame_index=1,
        view_id="camera-0",
        window_id="window-0",
        gauge_id="window-0",
        point_ids=np.asarray([7], dtype=np.int64),
        points_local_m=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        valid_mask=np.asarray([True]),
        local_covariance_m2=np.asarray([np.eye(3) * 1e-3]),
        association_probability=np.asarray([0.9]),
        prior_reliability=np.asarray([0.8]),
        prior_nominal_probability=0.95,
        composite_weight=1.0,
        correlation_group_id="camera-0:frame-1",
        causal_frame_stop=3,
    )
    bundle = ObservationFactorBundle(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tracklets:camera-0",
        factors=(factor,),
        gauges=(gauge,),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        causal_frame_stop=3,
        joint_gauge_covariance=gauge.covariance,
        gauge_covariance_semantics="joint-cross-window",
    )
    manifest, _ = write_observation_factor_bundle(bundle, root / "update-0.json")
    return manifest


def _append(manifest: Path, stream_path: Path) -> None:
    append_observation_factor_bundle(
        None,
        manifest,
        stream_manifest_path=stream_path,
        admitted_frame_start=0,
    )


def test_bundle_manifest_rejects_coercive_schema_version(tmp_path: Path) -> None:
    manifest = _write_delta_bundle(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["schema_version"] = "4"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        _append(manifest, tmp_path / "stream.json")


def test_bundle_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = _write_delta_bundle(tmp_path)
    text = manifest.read_text(encoding="utf-8").replace(
        '  "schema_version": 4,',
        '  "schema_version": 4,\n  "schema_version": 4,',
        1,
    )
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _append(manifest, tmp_path / "stream.json")


@pytest.mark.parametrize("allow_pickle", [0, True, "false"])
def test_bundle_manifest_requires_literal_false_pickle_policy(
    tmp_path: Path,
    allow_pickle: object,
) -> None:
    manifest = _write_delta_bundle(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["allow_pickle"] = allow_pickle
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="must disable pickle"):
        _append(manifest, tmp_path / "stream.json")


def test_bundle_manifest_rejects_noncanonical_gauge_order(tmp_path: Path) -> None:
    manifest = _write_delta_bundle(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["gauge_covariance"]["ordered_gauge_ids"] = [1]
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain nonempty strings"):
        _append(manifest, tmp_path / "stream.json")


def test_bundle_manifest_requires_complete_covariance_record(tmp_path: Path) -> None:
    manifest = _write_delta_bundle(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    del record["gauge_covariance"]["diagonal_blocks_match_gauge_marginals"]
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="gauge_covariance fields changed"):
        _append(manifest, tmp_path / "stream.json")
