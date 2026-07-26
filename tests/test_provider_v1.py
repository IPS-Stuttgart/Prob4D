from __future__ import annotations

from typing import cast

import prob4d.provider_v1 as provider
from prob4d._metric_gauge_anchor import MetricGaugeAnchor
from prob4d.observation_contract import ObservationBeliefExportV1


def test_provider_v1_declares_current_joint_covariance_contract() -> None:
    manifest = provider.prob4d_provider_manifest(provider_revision="a" * 40)

    assert provider.PROVIDER_API_VERSION == 1
    assert provider.OBSERVATION_BELIEF_VERSION == 1
    assert provider.OBSERVATION_FACTOR_SCHEMA_VERSION == 3
    assert "joint_cross_window_sim3_gauge_covariance" in manifest["capabilities"]
    assert manifest["limitations"][
        "joint_cross_window_gauge_covariance_in_observation_belief_v1"
    ] is True


def test_provider_v1_forwards_joint_gauge_options(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = cast(ObservationBeliefExportV1, object())

    def fake_build(manifest_path, **kwargs):
        captured["manifest_path"] = manifest_path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(provider, "build_prob4d_observation_belief", fake_build)
    anchor = cast(MetricGaugeAnchor, object())

    result = provider.export_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=42,
        metric_anchor=anchor,
        max_gauge_rank=21,
        minimum_retained_gauge_trace=0.995,
    )

    assert result is sentinel
    assert captured["manifest_path"] == "predictions.json"
    assert captured["gauge_mode"] == "sequential"
    assert captured["max_gauge_rank"] == 21
    assert captured["minimum_retained_gauge_trace"] == 0.995
    assert captured["allow_approximate_fixed_lag_covariance"] is False
