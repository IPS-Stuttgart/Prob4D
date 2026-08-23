from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "scripts" / "science" / "run_cut3r_source_freeze_execution.py"
REQUEST_PATH = ROOT / "protocols" / "execution_requests" / "cut3r_deform360_source_freeze_v2.json"


def _driver() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "cut3r_source_freeze_execution_driver",
        DRIVER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical_id(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_checked_in_v2_request_is_content_addressed_and_target_closed() -> None:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    identity = dict(request)
    request_id = identity.pop("request_id")

    assert request_id == _canonical_id(identity)
    assert request["schema_version"] == 2
    assert request["authorization_mode"] == ("merged-main-read-only-self-hosted-v1")
    assert request["source_group_count"] == 10
    assert request["forbidden_target_group_count"] == 12
    assert request["repository_write_token_on_self_hosted"] is False
    assert request["environment_approval_required"] is False
    assert request["source_rgb_frames_decoded"] is False
    assert request["source_prediction_payloads_opened"] is False
    assert request["source_residuals_or_truth_opened"] is False
    assert request["target_payloads_opened"] is False
    assert request["target_outcomes_opened"] is False
    assert request["comparison_execution_authorized"] is False


def test_driver_validates_checked_in_request_against_exact_protocol_blob() -> None:
    module = _driver()
    request = module.validate_request(
        REQUEST_PATH,
        repository=ROOT,
        require_clean=False,
    )

    assert request["request_id"] == (
        "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
    )


def test_driver_rejects_target_access_and_request_identity_drift(
    tmp_path: Path,
) -> None:
    module = _driver()
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    request["target_outcomes_opened"] = True
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="target-closed boundary"):
        module.validate_request(path, repository=ROOT, require_clean=False)

    request["target_outcomes_opened"] = False
    request["claim_boundary"] = "changed"
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="request ID mismatch"):
        module.validate_request(path, repository=ROOT, require_clean=False)


def test_driver_rejects_protocol_blob_drift(tmp_path: Path) -> None:
    module = _driver()
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    request["source_protocol_path"] = protocol.relative_to(tmp_path).as_posix()
    identity = dict(request)
    identity.pop("request_id")
    request["request_id"] = _canonical_id(identity)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Prob4D tests"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(ValueError, match="protocol path changed"):
        module.validate_request(
            request_path,
            repository=tmp_path,
            require_clean=False,
        )


def test_log_sanitization_prefers_longer_paths() -> None:
    module = _driver()
    text = "/data/cut3r/checkpoint /data/cut3r"
    sanitized = module._sanitize_log(
        text,
        {
            "/data/cut3r": "<CUT3R>",
            "/data/cut3r/checkpoint": "<CHECKPOINT>",
        },
    )

    assert sanitized == "<CHECKPOINT> <CUT3R>"
