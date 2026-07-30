from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prob4d.alignment import DENSE_ALIGNMENT_COVARIANCE_METHOD
from prob4d.calibration import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
)
from prob4d.calibration_compatibility import (
    MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA,
    MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1,
    MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V2,
    POINT_UNCERTAINTY_COVARIANCE_METHOD,
    CalibrationCompatibilityError,
    assert_calibration_pair_compatible,
    calibration_compatibility_mismatches,
    load_prediction_calibration_target,
    motioncrafter_model_identifier,
)
from prob4d.motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
)
from prob4d.uncertainty import CalibrationReport, DepthDisagreementModel


def _manifest() -> dict[str, object]:
    return {
        "format_version": 1,
        "motioncrafter_commit": "b" * 40,
        "config": {
            "model_type": "determ",
            "unet_path": "TencentARC/MotionCrafter",
            "vae_path": "TencentARC/MotionCrafter",
            "num_inference_steps": 5,
            "guidance_scale": 1.0,
            "decode_chunk_size": 25,
            "low_memory_usage": False,
            "seed": 42,
            "frame_stride": 1,
            "height": 384,
            "width": 640,
            "window_size": 16,
            "overlap": 8,
        },
    }


def _target(tmp_path: Path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(_manifest(), indent=2) + "\n", encoding="utf-8")
    return load_prediction_calibration_target(path)


def _provenance(target, *, covariance_method: str) -> dict[str, object]:
    return {
        "calibration_case_ids": ("scene-a", "scene-b"),
        "source_repository": target.source_repository,
        "source_revision": "a" * 40,
        "motioncrafter_revision": target.motioncrafter_revision,
        "model_identifier": target.model_identifier,
        "covariance_method": covariance_method,
        "image_resolution": target.image_resolution,
        "window_size": target.window_size,
        "window_overlap": target.window_overlap,
        "covariance_cluster_size": target.covariance_cluster_size,
        "input_artifact_sha256": ("c" * 64,),
    }


def _gauge(target) -> GaugeCovarianceCalibrationV1:
    return GaugeCovarianceCalibrationV1(
        scale=1.0,
        rotation=1.0,
        translation=1.0,
        count=8,
        trim_quantile=0.99,
        **_provenance(
            target,
            covariance_method=DENSE_ALIGNMENT_COVARIANCE_METHOD,
        ),
    )


def _point(target) -> PointUncertaintyCalibrationV1:
    return PointUncertaintyCalibrationV1.from_model(
        DepthDisagreementModel(),
        CalibrationReport(
            count=8,
            parallel_scale_update=1.0,
            lateral_scale_update=1.0,
            parallel_normalized_mse=1.0,
            lateral_normalized_mse=1.0,
        ),
        trim_quantile=0.99,
        **_provenance(
            target,
            covariance_method=POINT_UNCERTAINTY_COVARIANCE_METHOD,
        ),
    )


def test_model_identifier_is_canonical_and_sensitive_to_model_settings() -> None:
    first = _manifest()
    reordered = {
        "config": dict(reversed(list(first["config"].items()))),
        "motioncrafter_commit": first["motioncrafter_commit"],
        "format_version": 1,
    }
    assert motioncrafter_model_identifier(first) == motioncrafter_model_identifier(
        reordered
    )

    changed = json.loads(json.dumps(first))
    changed["config"]["unet_path"] = "other/checkpoint"
    assert motioncrafter_model_identifier(first) != motioncrafter_model_identifier(
        changed
    )


def test_model_identifier_preserves_legacy_seed_semantics_and_binds_new_policy() -> None:
    implicit_legacy = _manifest()
    explicit_legacy = json.loads(json.dumps(implicit_legacy))
    explicit_legacy["config"]["seed_policy"] = (
        MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON
    )
    derived = json.loads(json.dumps(implicit_legacy))
    derived["config"]["seed_policy"] = MOTIONCRAFTER_SEED_POLICY_DERIVED_PER_CALL

    legacy_identifier = motioncrafter_model_identifier(implicit_legacy)
    assert MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA == (
        MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V1
    )
    assert legacy_identifier == motioncrafter_model_identifier(explicit_legacy)
    assert legacy_identifier.startswith(f"{MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA}:")
    assert motioncrafter_model_identifier(derived).startswith(
        f"{MOTIONCRAFTER_MODEL_IDENTIFIER_SCHEMA_V2}:"
    )
    assert motioncrafter_model_identifier(derived) != legacy_identifier

    invalid = json.loads(json.dumps(implicit_legacy))
    invalid["config"]["seed_policy"] = "unknown"
    with pytest.raises(ValueError, match="seed policy"):
        motioncrafter_model_identifier(invalid)


def test_calibration_pair_matches_prediction_manifest(tmp_path: Path) -> None:
    target = _target(tmp_path)
    gauge = _gauge(target)
    point = _point(target)

    assert_calibration_pair_compatible(gauge, point, target)
    assert target.gauge_covariance_method == DENSE_ALIGNMENT_COVARIANCE_METHOD
    assert target.point_covariance_method == POINT_UNCERTAINTY_COVARIANCE_METHOD
    assert len(target.manifest_sha256) == 64


def test_calibration_mismatch_fails_with_field_level_diagnostics(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    incompatible = replace(
        _gauge(target),
        motioncrafter_revision="d" * 40,
        image_resolution=(320, 640),
    )

    mismatches = calibration_compatibility_mismatches(
        incompatible,
        target,
        expected_covariance_method=target.gauge_covariance_method,
    )
    assert any(item.startswith("motioncrafter_revision:") for item in mismatches)
    assert any(item.startswith("image_resolution:") for item in mismatches)
    with pytest.raises(
        CalibrationCompatibilityError,
        match="motioncrafter_revision",
    ):
        assert_calibration_pair_compatible(incompatible, _point(target), target)


def test_manifest_validation_precedes_payload_access(tmp_path: Path) -> None:
    manifest = _manifest()
    del manifest["config"]["window_size"]
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution or window settings"):
        load_prediction_calibration_target(path)
