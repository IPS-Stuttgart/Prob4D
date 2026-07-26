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
        "MetricGaugeAnchor": 2,
        "ObservationBeliefV1": 1,
        "ObservationFactorBundle": 3,
        "Prob4DObservationContract": 2,
    }
    assert "joint_cross_window_sim3_gauge_covariance" in manifest["capabilities"]
    assert "content_addressed_metric_gauge_anchor" in manifest["capabilities"]
    assert manifest["limitations"][
        "joint_cross_window_gauge_covariance_in_observation_belief_v1"
    ] is True
    assert manifest["limitations"][
        "fixed_lag_boundary_covariance_exactness_claim"
    ] is False
    assert manifest["metadata"]["prob4d_observation_contract_version"] == 2
    assert (
        manifest["metadata"]["observation_belief_covariance_layout"]
        == "joint_sim3_tree_root_v1"
    )
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
