from __future__ import annotations

from pathlib import Path

import prob4d.provider_v2_cli as cli


def test_calibrated_cli_requires_and_forwards_claim_bearing_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    anchor = object()
    gauge = object()
    point = object()
    artifact = object()
    captured = {}

    monkeypatch.setattr(cli, "load_metric_gauge_anchor", lambda path: anchor)
    monkeypatch.setattr(cli, "load_gauge_covariance_calibration", lambda path: gauge)
    monkeypatch.setattr(cli, "load_point_uncertainty_calibration", lambda path: point)

    def fake_export(manifest_path, **kwargs):
        captured.update(manifest_path=manifest_path, **kwargs)
        return artifact

    def fake_publish(args, supplied_artifact):
        captured["output"] = args.output_npz
        captured["published_artifact"] = supplied_artifact
        return 0

    monkeypatch.setattr(cli, "export_calibrated_observation_belief", fake_export)
    monkeypatch.setattr(cli, "_publish_artifact", fake_publish)

    output = tmp_path / "observation.npz"
    assert (
        cli.main_calibrated(
            [
                "predictions.json",
                str(output),
                "--case-id",
                "case-a",
                "--causal-frame-stop",
                "134",
                "--metric-gauge-anchor",
                "anchor.json",
                "--gauge-covariance-calibration",
                "gauge.json",
                "--point-uncertainty-calibration",
                "point.json",
                "--source-revision",
                "a" * 40,
                "--sampling-mode",
                "information_stratified",
            ]
        )
        == 0
    )

    assert captured["manifest_path"] == Path("predictions.json")
    assert captured["metric_anchor"] is anchor
    assert captured["gauge_covariance_calibration"] is gauge
    assert captured["point_uncertainty_calibration"] is point
    assert captured["source_revision"] == "a" * 40
    assert captured["sampling_mode"] == "information_stratified"
    assert captured["output"] == output
    assert captured["published_artifact"] is artifact


def test_exploratory_cli_keeps_controls_explicit(monkeypatch, tmp_path: Path) -> None:
    anchor = object()
    artifact = object()
    captured = {}

    monkeypatch.setattr(cli, "load_metric_gauge_anchor", lambda path: anchor)

    def fake_export(manifest_path, **kwargs):
        captured.update(manifest_path=manifest_path, **kwargs)
        return artifact

    monkeypatch.setattr(cli, "export_exploratory_observation_belief", fake_export)
    monkeypatch.setattr(cli, "_publish_artifact", lambda args, value: 0)

    assert (
        cli.main_exploratory(
            [
                "predictions.json",
                str(tmp_path / "observation.npz"),
                "--case-id",
                "case-a",
                "--causal-frame-stop",
                "134",
                "--metric-gauge-anchor",
                "anchor.json",
                "--gauge-mode",
                "fixed_lag",
                "--allow-approximate-fixed-lag-covariance",
                "--gauge-root-mode",
                "legacy_eigenvectors",
                "--allow-pointwise-covariance-fallback",
            ]
        )
        == 0
    )

    assert captured["metric_anchor"] is anchor
    assert captured["gauge_mode"] == "fixed_lag"
    assert captured["allow_approximate_fixed_lag_covariance"] is True
    assert captured["gauge_root_mode"] == "legacy_eigenvectors"
    assert captured["allow_pointwise_covariance_fallback"] is True
    assert captured["gauge_covariance_calibration"] is None
    assert captured["point_uncertainty_calibration"] is None
