from __future__ import annotations

from pathlib import Path

import prob4d.provider_v1 as provider


def test_provider_v1_exposes_versioned_contracts() -> None:
    assert provider.PROVIDER_API_VERSION == 1
    assert provider.OBSERVATION_BELIEF_SCHEMA == "phys4d.observation_belief"
    assert provider.OBSERVATION_BELIEF_VERSION == 1
    assert provider.OBSERVATION_FACTOR_SCHEMA_VERSION == 3


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


def test_export_observation_belief_forwards_stable_parameters(monkeypatch) -> None:
    sentinel = object()
    anchor = object()
    model = object()
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured.update(manifest_path=manifest_path, **kwargs)
        return sentinel

    monkeypatch.setattr(provider, "build_prob4d_observation_belief", fake_export)
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
        view_name="left-camera",
        source_revision="a" * 40,
        uncertainty_model=model,
    )

    assert result is sentinel
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
        "view_name": "left-camera",
        "source_revision": "a" * 40,
        "uncertainty_model": model,
    }
