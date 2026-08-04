from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prob4d.calibration import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    load_gauge_covariance_calibration,
    load_point_uncertainty_calibration,
    save_gauge_covariance_calibration,
    save_point_uncertainty_calibration,
)
from prob4d.gauge import GaugeEstimate
from prob4d.observation_factors import (
    ObservationFactor,
    ObservationFactorBundle,
    load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from prob4d.sim3 import Sim3

_PROVENANCE = {
    "calibration_case_ids": ("scene-a", "scene-b"),
    "source_repository": "FlorianPfaff/Prob4D",
    "source_revision": "a" * 40,
    "motioncrafter_revision": "b" * 40,
    "model_identifier": "motioncrafter-base@checkpoint-sha256:cafebabe",
    "covariance_method": "held_out_scene_family_v1",
    "image_resolution": (384, 640),
    "window_size": 16,
    "window_overlap": 8,
    "covariance_cluster_size": 32,
    "input_artifact_sha256": ("c" * 64,),
    "metadata": {"split": {"kind": "scene-family", "fold": 1}},
}


def _gauge_calibration() -> GaugeCovarianceCalibrationV1:
    return GaugeCovarianceCalibrationV1(
        scale=4.0,
        rotation=2.0,
        translation=3.0,
        count=12,
        trim_quantile=0.99,
        **_PROVENANCE,
    )


def _point_calibration() -> PointUncertaintyCalibrationV1:
    return PointUncertaintyCalibrationV1(
        parallel_floor=1e-4,
        parallel_depth_coefficient=1e-3,
        lateral_floor=2e-4,
        lateral_depth_coefficient=2e-3,
        disagreement_gain=0.5,
        parallel_scale=1.5,
        lateral_scale=1.25,
        count=12,
        trim_quantile=0.99,
        parallel_scale_update=1.1,
        lateral_scale_update=1.2,
        parallel_normalized_mse=0.9,
        lateral_normalized_mse=1.1,
        **_PROVENANCE,
    )


@pytest.mark.parametrize(
    ("artifact", "changes", "message"),
    [
        (_gauge_calibration, {"count": True}, "count must be an integer"),
        (_gauge_calibration, {"count": 12.0}, "count must be an integer"),
        (_gauge_calibration, {"scale": "4.0"}, "scale must be a JSON number"),
        (
            _gauge_calibration,
            {"source_revision": 123},
            "source_revision must be a string",
        ),
        (
            _gauge_calibration,
            {"calibration_case_ids": (1, "1")},
            "calibration_case_ids must contain nonempty strings",
        ),
        (
            _gauge_calibration,
            {"image_resolution": (384.0, 640)},
            "image_resolution item must be an integer",
        ),
        (
            _gauge_calibration,
            {"window_size": True},
            "window_size must be an integer",
        ),
        (
            _gauge_calibration,
            {"input_artifact_sha256": (123,)},
            "input_artifact_sha256 must contain nonempty strings",
        ),
        (
            _gauge_calibration,
            {"metadata": {1: "integer", "1": "string"}},
            "metadata must be finite JSON data",
        ),
        (_point_calibration, {"count": "12"}, "count must be an integer"),
        (
            _point_calibration,
            {"trim_quantile": False},
            "trim_quantile must be a JSON number",
        ),
    ],
)
def test_calibration_constructors_reject_scalar_coercion(
    artifact: Callable[[], Any],
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(artifact(), **changes)


def test_valid_gauge_calibration_identity_is_unchanged(tmp_path: Path) -> None:
    artifact = _gauge_calibration()

    assert artifact.artifact_id == (
        "ba5de02ee348fc413e4dc5a9c558aa2231db06c910cc165a94748215ae75f2fe"
    )
    path = tmp_path / "gauge.json"
    save_gauge_covariance_calibration(artifact, path)
    assert load_gauge_covariance_calibration(path) == artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(version=True), "version must be an integer"),
        (
            lambda value: value["calibration"].update(count=12.0),
            "count must be an integer",
        ),
        (
            lambda value: value.update(artifact_id=123),
            "artifact_id must be a string",
        ),
        (
            lambda value: value.update(unsigned_extra="not content addressed"),
            "calibration artifact fields changed",
        ),
    ],
)
def test_gauge_loader_rejects_noncanonical_payloads(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = tmp_path / "gauge.json"
    save_gauge_covariance_calibration(_gauge_calibration(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_gauge_covariance_calibration(path)


def test_point_loader_rejects_noncanonical_payload(tmp_path: Path) -> None:
    path = tmp_path / "point.json"
    save_point_uncertainty_calibration(_point_calibration(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["window_overlap"] = "8"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="window_overlap must be an integer"):
        load_point_uncertainty_calibration(path)


def test_calibration_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "gauge.json"
    save_gauge_covariance_calibration(_gauge_calibration(), path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '  "version": 1\n',
        '  "version": 1,\n  "version": 1\n',
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key 'version'"):
        load_gauge_covariance_calibration(path)


def _factor() -> ObservationFactor:
    return ObservationFactor(
        factor_id="factor-0",
        frame_index=4,
        view_id="camera-0",
        window_id="window-0",
        gauge_id="window-0",
        point_ids=np.asarray([11, 12]),
        points_local_m=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        valid_mask=np.asarray([True, True]),
        local_covariance_m2=np.tile(np.eye(3) * 0.01, (2, 1, 1)),
        association_probability=np.asarray([0.9, 0.6]),
        prior_reliability=np.asarray([0.7, 0.4]),
        prior_nominal_probability=0.8,
        composite_weight=0.5,
        correlation_group_id="backbone-window-0",
        causal_frame_stop=9,
        ray_directions_local=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    )


def _factor_bundle() -> ObservationFactorBundle:
    gauge = GaugeEstimate("window-0", Sim3.identity(), np.eye(7) * 1e-4)
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        factors=(_factor(),),
        gauges=(gauge,),
        source_revision="0123456789abcdef",
        causal_frame_stop=9,
        case_id="double-stretch-sloth",
        stream_id="prob4d:camera-points",
        metadata={"producer": "unit-test", "metric": True},
        joint_gauge_covariance=gauge.covariance,
        gauge_covariance_semantics="joint-cross-window",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(schema_version=4.0),
            "schema_version must be an integer",
        ),
        (
            lambda value: value.update(causal_frame_stop="9"),
            "causal_frame_stop must be an integer",
        ),
        (
            lambda value: value["payload"].update(allow_pickle=0),
            "payload.allow_pickle must be the literal Boolean false",
        ),
        (
            lambda value: value["factors"][0].update(frame_index=True),
            "frame_index must be an integer",
        ),
        (
            lambda value: value["factors"][0].update(
                prior_nominal_probability="0.8"
            ),
            "prior_nominal_probability must be a JSON number",
        ),
        (
            lambda value: value["gauge_covariance"].update(
                ordered_gauge_ids=[1]
            ),
            "ordered_gauge_ids must contain nonempty strings",
        ),
        (
            lambda value: value.update(unsigned_extra="ignored before"),
            "schema-v4 manifest fields changed",
        ),
    ],
)
def test_schema_v4_loader_rejects_scalar_coercion(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    manifest, _ = write_observation_factor_bundle(
        _factor_bundle(),
        tmp_path / "factors.json",
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(record)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_observation_factor_bundle(manifest)


def test_schema_v4_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest, _ = write_observation_factor_bundle(
        _factor_bundle(),
        tmp_path / "factors.json",
    )
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        '  "schema_version": 4,\n',
        '  "schema_version": 4,\n  "schema_version": 4,\n',
        1,
    )
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key 'schema_version'"):
        load_observation_factor_bundle(manifest)
