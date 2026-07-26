from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prob4d.provider_manifest import prob4d_provider_manifest
from prob4d.provider_manifest_cli import main


def test_provider_manifest_declares_covariance_boundary() -> None:
    manifest = prob4d_provider_manifest(provider_revision="a" * 40)

    assert manifest["provider_revision"] == "a" * 40
    assert manifest["artifact_schema_versions"] == {
        "GaugeCovarianceCalibrationV1": 1,
        "MetricGaugeAnchor": 1,
        "ObservationBeliefV1": 1,
        "ObservationFactorBundle": 3,
        "PointUncertaintyCalibrationV1": 1,
        "Prob4DCausalObservationStream": 2,
    }
    assert "joint_cross_window_sim3_gauge_covariance" in manifest["capabilities"]
    assert "content_addressed_covariance_calibration" in manifest["capabilities"]
    assert "content_addressed_metric_gauge_anchor" in manifest["capabilities"]
    assert "metric_anchor_covariance_propagation" in manifest["capabilities"]
    assert "versioned_causal_stream_contract" in manifest["capabilities"]
    assert manifest["metadata"]["observation_stream_contract_version"] == 2
    assert manifest["limitations"][
        "joint_cross_window_gauge_covariance_in_observation_belief_v1"
    ] is True
    assert manifest["limitations"][
        "fixed_lag_boundary_covariance_exactness_claim"
    ] is False
    assert manifest["limitations"][
        "provider_pointwise_covariance_fallback_default"
    ] is False
    descriptor = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["manifest_id"] == expected


def test_provider_manifest_cli_writes_exact_payload(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "provider.json"
    assert main(["--provider-revision", "b" * 40, "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed) == json.loads(output.read_text(encoding="utf-8"))
