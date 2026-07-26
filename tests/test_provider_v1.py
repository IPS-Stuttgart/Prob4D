from __future__ import annotations

from pathlib import Path

import prob4d.provider_v1 as provider


def test_provider_v1_exposes_versioned_contracts() -> None:
    assert provider.PROVIDER_API_VERSION == 1
    assert provider.PROB4D_PROVIDER_API_VERSION == 1
    assert provider.PROB4D_CAUSAL_STREAM_CONTRACT_VERSION == 2
    assert provider.OBSERVATION_BELIEF_SCHEMA == "phys4d.observation_belief"
    assert provider.OBSERVATION_BELIEF_VERSION == 1
    assert provider.OBSERVATION_FACTOR_SCHEMA_VERSION == 3
    assert callable(provider.load_observation_belief_export)
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
