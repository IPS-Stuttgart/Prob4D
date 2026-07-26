from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import prob4d.provider_v1 as provider
from prob4d.calibration import (
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
)
from prob4d.observation_contract import ObservationBeliefExportV1
from prob4d.uncertainty import CalibrationReport, DepthDisagreementModel


PROVENANCE = {
    "calibration_case_ids": ("scene-a",),
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
}


def _gauge_calibration() -> GaugeCovarianceCalibrationV1:
    return GaugeCovarianceCalibrationV1(
        scale=1.0,
        rotation=1.0,
        translation=1.0,
        count=8,
        trim_quantile=0.99,
        **PROVENANCE,
    )


def _point_calibration() -> PointUncertaintyCalibrationV1:
    report = CalibrationReport(
        count=8,
        parallel_scale_update=1.0,
        lateral_scale_update=1.0,
        parallel_normalized_mse=1.0,
        lateral_normalized_mse=1.0,
    )
    return PointUncertaintyCalibrationV1.from_model(
        DepthDisagreementModel(),
        report,
        trim_quantile=0.99,
        **PROVENANCE,
    )


def _observation() -> ObservationBeliefExportV1:
    return ObservationBeliefExportV1(
        case_id="case-a",
        stream_id="prob4d:test",
        causal_frame_stop=2,
        view_names=("camera0",),
        window_names=("window0",),
        factor_names=(),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="d" * 64,
        declared_frame_ids=np.asarray([0]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]]),
        frame_ids=np.asarray([0]),
        entity_ids=np.asarray([0]),
        view_indices=np.asarray([0]),
        window_indices=np.asarray([0]),
        correlation_group_ids=np.asarray([0]),
        factor_group_ids=np.asarray([0]),
        prior_reliability=np.asarray([1.0]),
        association_probability=np.asarray([1.0]),
        local_covariance_m2=np.asarray([np.eye(3)]),
        low_rank_factor_m=np.empty((1, 3, 0)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([1.0]),
        group_composite_weight=np.asarray([1.0]),
        metadata={"existing": True},
    )


def test_provider_v1_exposes_versioned_contracts() -> None:
    assert provider.PROVIDER_API_VERSION == 1
    assert provider.PROB4D_PROVIDER_API_VERSION == 1
    assert provider.PROB4D_CAUSAL_STREAM_CONTRACT_VERSION == 2
    assert provider.OBSERVATION_BELIEF_SCHEMA == "phys4d.observation_belief"
    assert provider.OBSERVATION_BELIEF_VERSION == 1
    assert provider.OBSERVATION_FACTOR_SCHEMA_VERSION == 3
    assert provider.GAUGE_COVARIANCE_CALIBRATION_VERSION == 1
    assert provider.POINT_UNCERTAINTY_CALIBRATION_VERSION == 1
    assert callable(provider.load_observation_belief_export)
    assert callable(provider.load_gauge_covariance_calibration)
    assert callable(provider.load_point_uncertainty_calibration)
    manifest = provider.prob4d_provider_manifest(provider_revision="a" * 40)
    assert manifest["provider_api_version"] == provider.PROVIDER_API_VERSION
    assert "versioned_python_provider_api" in manifest["capabilities"]
    assert "versioned_causal_stream_contract" in manifest["capabilities"]


def test_select_causal_source_forwards_exact_boundary(monkeypatch) -> None:
    sentinel = object()
    anchor = object()
    captured = {}

    def fake_select(manifest_path, *, causal_frame_stop, metric_anchor):
        captured.update(
            manifest_path=manifest_path,
            causal_frame_stop=causal_frame_stop,
            metric_anchor=metric_anchor,
        )
        return sentinel

    monkeypatch.setattr(provider, "select_causal_overlap_windows", fake_select)
    result = provider.select_causal_source(
        Path("predictions.json"),
        causal_frame_stop=134,
        metric_anchor=anchor,
    )

    assert result is sentinel
    assert captured == {
        "manifest_path": Path("predictions.json"),
        "causal_frame_stop": 134,
        "metric_anchor": anchor,
    }


def test_export_observation_belief_forwards_and_binds_stable_parameters(
    monkeypatch,
) -> None:
    raw = object()
    bound = object()
    anchor = object()
    model = object()
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured.update(manifest_path=manifest_path, **kwargs)
        return raw

    def fake_bind(artifact, *, metric_anchor):
        captured.update(bound_artifact=artifact, bound_anchor=metric_anchor)
        return bound

    monkeypatch.setattr(provider, "build_prob4d_observation_belief", fake_export)
    monkeypatch.setattr(provider, "bind_causal_stream_contract_v2", fake_bind)
    result = provider.export_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=134,
        metric_anchor=anchor,
        pixel_stride=8,
        effective_samples_per_group=32.0,
        minimum_prior_reliability=0.1,
        gauge_mode="sequential",
        fixed_lag=5,
        allow_approximate_fixed_lag_covariance=False,
        max_gauge_rank=48,
        minimum_retained_gauge_trace=0.997,
        view_name="left-camera",
        source_revision="a" * 40,
        uncertainty_model=model,
    )

    assert result is bound
    assert captured == {
        "manifest_path": "predictions.json",
        "case_id": "case-a",
        "causal_frame_stop": 134,
        "metric_anchor": anchor,
        "pixel_stride": 8,
        "effective_samples_per_group": 32.0,
        "minimum_prior_reliability": 0.1,
        "gauge_mode": "sequential",
        "fixed_lag": 5,
        "allow_approximate_fixed_lag_covariance": False,
        "max_gauge_rank": 48,
        "minimum_retained_gauge_trace": 0.997,
        "view_name": "left-camera",
        "source_revision": "a" * 40,
        "uncertainty_model": model,
        "bound_artifact": raw,
        "bound_anchor": anchor,
    }


def test_fixed_lag_export_is_not_labelled_as_strict_stream(monkeypatch) -> None:
    raw = object()
    anchor = object()
    monkeypatch.setattr(
        provider,
        "build_prob4d_observation_belief",
        lambda *args, **kwargs: raw,
    )

    def fail_bind(*args, **kwargs):
        raise AssertionError("fixed-lag output must not receive stream contract v2")

    monkeypatch.setattr(provider, "bind_causal_stream_contract_v2", fail_bind)
    result = provider.export_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=134,
        metric_anchor=anchor,
        gauge_mode="fixed_lag",
        allow_approximate_fixed_lag_covariance=True,
    )
    assert result is raw


def test_claim_bearing_export_requires_both_calibrations() -> None:
    with pytest.raises(ValueError, match="require both"):
        provider.export_observation_belief(
            "predictions.json",
            case_id="case-a",
            causal_frame_stop=134,
            metric_anchor=object(),
            gauge_covariance_calibration=_gauge_calibration(),
            allow_uncalibrated_exploratory_covariance=False,
        )


def test_calibrated_export_records_artifact_ids_before_stream_binding(monkeypatch) -> None:
    original = _observation()
    gauge = _gauge_calibration()
    point = _point_calibration()
    monkeypatch.setattr(
        provider,
        "build_prob4d_observation_belief",
        lambda *args, **kwargs: original,
    )

    def fake_bind(artifact, *, metric_anchor):
        calibration = artifact.metadata["covariance_calibration"]
        assert calibration["status"] == "calibrated"
        assert calibration["gauge_artifact_id"] == gauge.artifact_id
        assert calibration["point_artifact_id"] == point.artifact_id
        assert calibration["alignment_count"] == 0
        assert calibration["covariance_fallback_counts"] == {}
        return artifact

    monkeypatch.setattr(provider, "bind_causal_stream_contract_v2", fake_bind)
    exported = provider.export_calibrated_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=2,
        metric_anchor=object(),
        gauge_covariance_calibration=gauge,
        point_uncertainty_calibration=point,
    )

    assert exported.artifact_id != original.artifact_id
