from __future__ import annotations

from pathlib import Path

import pytest

import prob4d.provider_v2 as provider
from prob4d.covariance_root import current_covariance_root_mode


def test_provider_v2_exposes_safe_capabilities() -> None:
    assert provider.PROVIDER_API_VERSION == 2
    assert provider.PROB4D_PROVIDER_API_VERSION == 2
    manifest = provider.prob4d_provider_manifest(provider_revision="a" * 40)
    assert manifest["provider_api_version"] == 2
    assert "strict_prediction_calibration_compatibility" in manifest["capabilities"]
    assert (
        "canonical_repeated_eigenspace_covariance_root" in manifest["capabilities"]
    )
    assert manifest["metadata"]["python_import_boundary"] == "prob4d.provider_v2"
    assert "canonical basis" in manifest["metadata"]["covariance_root_semantics"]
    assert manifest["limitations"]["uncalibrated_export_is_default"] is False


def test_exploratory_export_is_explicit_and_context_local(monkeypatch) -> None:
    sentinel = object()
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured.update(
            manifest_path=manifest_path,
            root_mode=current_covariance_root_mode(),
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(provider._v1, "export_observation_belief", fake_export)
    result = provider.export_exploratory_observation_belief(
        Path("predictions.json"),
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        sampling_mode="information_stratified",
        allow_pointwise_covariance_fallback=True,
    )

    assert result is sentinel
    assert captured["root_mode"] == "canonical_eigenspaces"
    assert current_covariance_root_mode() == "legacy_eigenvectors"
    assert captured["allow_uncalibrated_exploratory_covariance"] is True
    assert captured["allow_pointwise_covariance_fallback"] is True
    assert captured["sampling_mode"] == "information_stratified"


def test_exploratory_export_can_reproduce_legacy_root_basis(monkeypatch) -> None:
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured["root_mode"] = current_covariance_root_mode()
        return object()

    monkeypatch.setattr(provider._v1, "export_observation_belief", fake_export)
    provider.export_exploratory_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        gauge_root_mode="legacy_eigenvectors",
    )
    assert captured["root_mode"] == "legacy_eigenvectors"


def test_calibrated_export_validates_before_delegating(monkeypatch) -> None:
    sentinel = object()
    target = object()
    gauge = object()
    point = object()
    calls = []

    def fake_target(manifest_path, **kwargs):
        calls.append(("target", manifest_path, kwargs))
        return target

    def fake_assert(gauge_calibration, point_calibration, supplied_target):
        calls.append(
            (
                "assert",
                gauge_calibration,
                point_calibration,
                supplied_target,
            )
        )

    def fake_export(manifest_path, **kwargs):
        calls.append(
            (
                "export",
                manifest_path,
                kwargs,
                current_covariance_root_mode(),
            )
        )
        return sentinel

    monkeypatch.setattr(provider, "load_prediction_calibration_target", fake_target)
    monkeypatch.setattr(provider, "assert_calibration_pair_compatible", fake_assert)
    monkeypatch.setattr(
        provider._v1,
        "export_calibrated_observation_belief",
        fake_export,
    )

    result = provider.export_calibrated_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        gauge_covariance_calibration=gauge,
        point_uncertainty_calibration=point,
        source_revision="a" * 40,
    )

    assert result is sentinel
    assert [call[0] for call in calls] == ["target", "assert", "export"]
    export_kwargs = calls[-1][2]
    assert calls[-1][3] == "canonical_eigenspaces"
    assert current_covariance_root_mode() == "legacy_eigenvectors"
    assert export_kwargs["gauge_mode"] == "sequential"
    assert export_kwargs["allow_pointwise_covariance_fallback"] is False
    assert export_kwargs["source_revision"] == "a" * 40


def test_calibrated_export_requires_explicit_source_revision() -> None:
    with pytest.raises(TypeError, match="source_revision"):
        provider.export_calibrated_observation_belief(
            "predictions.json",
            case_id="case-a",
            causal_frame_stop=10,
            metric_anchor=object(),
            gauge_covariance_calibration=object(),
            point_uncertainty_calibration=object(),
        )
