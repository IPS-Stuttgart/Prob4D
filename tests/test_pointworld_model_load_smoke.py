from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_pointworld_model_load_smoke.py"
PROTOCOL = ROOT / "protocols" / "pointworld-model-load-smoke-v1.json"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pointworld_model_load_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(module: ModuleType, blob_sha: str) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    request: dict[str, object] = {
        "schema": module.REQUEST_SCHEMA,
        "schema_version": 1,
        "issue_number": 333,
        "profile": module.PROFILE,
        "source_protocol_path": "protocols/pointworld-model-load-smoke-v1.json",
        "source_protocol_git_blob_sha": blob_sha,
        "execution_authorized": True,
        "dataset_access_authorized": False,
        "prediction_execution_authorized": False,
        "provider_residuals_authorized": False,
        "target_outcomes_authorized": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    request["request_id"] = module.canonical_id(request)
    return request


def test_frozen_protocol_is_valid_and_target_closed() -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validated = module.validate_protocol(protocol)

    assert validated["pointworld_revision"] == "05484826dfef74cbe278a3974179a5a16705d35d"
    assert validated["dinov3_revision"] == "54694f7627fd815f62a5dcc82944ffa6153bbb76"
    assert validated["dinov3_weights_sha256"] == (
        "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
    )
    assert validated["dinov3_weights_size_bytes"] == 1_213_050_671
    assert validated["dataset_access_authorized"] is False
    assert validated["prediction_execution_authorized"] is False
    assert validated["target_outcomes_authorized"] is False


def test_request_identity_binds_protocol_blob() -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    blob_sha = "a" * 40
    request = _request(module, blob_sha)

    assert (
        module.validate_request(
            request,
            protocol,
            source_protocol_git_blob_sha=blob_sha,
        )
        == request["request_id"]
    )

    with pytest.raises(ValueError, match="checked protocol blob"):
        module.validate_request(
            request,
            protocol,
            source_protocol_git_blob_sha="b" * 40,
        )


def test_request_rejects_prediction_or_target_authorization() -> None:
    module = _load_module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for field in ("prediction_execution_authorized", "target_outcomes_authorized"):
        request = _request(module, "a" * 40)
        request[field] = True
        identity = dict(request)
        identity.pop("request_id")
        request["request_id"] = module.canonical_id(identity)
        with pytest.raises(ValueError, match=f"{field} must remain false"):
            module.validate_request(
                request,
                protocol,
                source_protocol_git_blob_sha="a" * 40,
            )


def test_preflight_failure_is_sanitized_and_machine_readable(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "preflight.json"
    secret_root = tmp_path / "private-staging-root"
    status = module.run_preflight(
        protocol_path=PROTOCOL,
        pointworld_checkout=secret_root / "code" / "PointWorld",
        checkpoint=secret_root / "models" / "model-best.pt",
        dinov3_weights=secret_root / "models" / "dinov3.pth",
        request_id="c" * 64,
        prob4d_revision="d" * 40,
        output=output,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert status == 3
    assert result["decision"] == "fail"
    assert result["failure_stage"] == "preflight"
    assert str(secret_root) not in json.dumps(result)
    assert result["dataset_opened"] is False
    assert result["prediction_executed"] is False
    assert result["target_outcomes_opened"] is False


def test_file_hash_is_content_exact(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "asset.bin"
    path.write_bytes(b"prob4d-pointworld-smoke")
    assert module._sha256_file(path) == (
        "8609f7e445ae1bba7cb22180ed9f39b88e87387516c37bfcc08ae9044f33c156"
    )
