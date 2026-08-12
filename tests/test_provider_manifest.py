from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import prob4d
from prob4d.provider_manifest_cli import main
from prob4d.provider_v2 import prob4d_provider_manifest


def test_provider_manifest_declares_current_v2_boundary() -> None:
    manifest = prob4d_provider_manifest(provider_revision="a" * 40)

    assert manifest["provider_version"] == prob4d.__version__
    assert manifest["provider_revision"] == "a" * 40
    assert manifest["provider_api_version"] == 2
    assert manifest["artifact_schema_versions"]["ObservationFactorBundle"] == 4
    assert manifest["artifact_schema_versions"]["ObservationFactorStreamV1"] == 1
    assert "joint_cross_window_sim3_gauge_covariance" in manifest["capabilities"]
    assert "runtime_revision_attestation" in manifest["capabilities"]
    assert "strict_claim_bearing_observation_loading" in manifest["capabilities"]
    assert "provider_attested_observation_artifacts" in manifest["capabilities"]
    assert manifest["metadata"]["python_import_boundary"] == "prob4d.provider_v2"
    assert manifest["limitations"]["uncalibrated_export_is_default"] is False

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


def test_provider_manifest_cli_writes_exact_v2_payload(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "provider.json"
    assert main(["--provider-revision", "b" * 40, "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert printed == persisted
    assert printed["provider_api_version"] == 2
    assert printed["metadata"]["python_import_boundary"] == "prob4d.provider_v2"


def test_provider_manifest_cli_no_longer_selects_provider_v1() -> None:
    with pytest.raises(SystemExit):
        main(["--api-version", "1"])
