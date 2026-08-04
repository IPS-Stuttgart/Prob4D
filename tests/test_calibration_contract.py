from __future__ import annotations

import json
from pathlib import Path

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
from prob4d.data import PredictionWindow
from prob4d.uncertainty import DepthDisagreementModel

PROVENANCE = {
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


def test_gauge_calibration_is_content_addressed_and_round_trips(tmp_path: Path) -> None:
    artifact = GaugeCovarianceCalibrationV1(
        scale=4.0,
        rotation=2.0,
        translation=3.0,
        count=12,
        trim_quantile=0.99,
        **PROVENANCE,
    )
    covariance = np.eye(7)
    inflated = artifact.apply(covariance)
    np.testing.assert_allclose(np.diag(inflated), [4.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])

    path = tmp_path / "gauge-calibration.json"
    save_gauge_covariance_calibration(artifact, path)
    loaded = load_gauge_covariance_calibration(path)

    assert loaded == artifact
    assert loaded.artifact_id == artifact.artifact_id
    assert len(loaded.artifact_id) == 64
    with pytest.raises(TypeError):
        loaded.metadata["new"] = True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["scale"] = 9.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id"):
        load_gauge_covariance_calibration(path)


def test_gauge_calibration_fit_records_provenance() -> None:
    errors = np.ones((10, 7))
    covariances = np.repeat(np.eye(7)[None], 10, axis=0)
    artifact = GaugeCovarianceCalibrationV1.fit(
        errors,
        covariances,
        trim_quantile=1.0,
        **PROVENANCE,
    )

    assert artifact.count == 10
    assert artifact.scale == pytest.approx(1.0)
    assert artifact.rotation == pytest.approx(1.0)
    assert artifact.translation == pytest.approx(1.0)


def _window() -> PredictionWindow:
    point_map = np.zeros((2, 2, 3, 3))
    point_map[..., 2] = 2.0
    return PredictionWindow(
        "window",
        np.array([0, 1]),
        point_map,
        np.ones((2, 2, 3), dtype=bool),
    )


def test_point_calibration_is_reusable_model_and_round_trips(tmp_path: Path) -> None:
    generator = np.random.default_rng(4)
    window = _window()
    base = DepthDisagreementModel()
    covariance = base.predict(window)
    errors = np.zeros_like(window.point_map)
    errors[..., 2] = generator.normal(
        scale=np.sqrt(3.0 * covariance.parallel_variance),
        size=window.shape,
    )
    errors[..., 0] = generator.normal(
        scale=np.sqrt(2.0 * covariance.lateral_variance),
        size=window.shape,
    )
    errors[..., 1] = generator.normal(
        scale=np.sqrt(2.0 * covariance.lateral_variance),
        size=window.shape,
    )
    artifact = PointUncertaintyCalibrationV1.fit(
        base,
        errors,
        covariance,
        trim_quantile=1.0,
        **PROVENANCE,
    )

    assert artifact.count == np.prod(window.shape)
    assert artifact.model.parallel_scale == artifact.parallel_scale
    assert artifact.model.lateral_scale == artifact.lateral_scale

    path = tmp_path / "point-calibration.json"
    save_point_uncertainty_calibration(artifact, path)
    loaded = load_point_uncertainty_calibration(path)
    assert loaded == artifact
    np.testing.assert_allclose(
        loaded.model.predict(window).parallel_variance,
        artifact.model.predict(window).parallel_variance,
    )
