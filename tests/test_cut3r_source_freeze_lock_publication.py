from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "publish_cut3r_source_freeze_lock.py"
REQUEST = (
    ROOT
    / "protocols"
    / "publication_requests"
    / "cut3r_deform360_source_freeze_v2.json"
)
SOURCE_REQUEST_ID = (
    "8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e"
)


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "publish_cut3r_source_freeze_lock",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_fixture(root: Path) -> dict[str, object]:
    evidence = root / "evidence"
    payload = evidence / "cut3r-deform360-source-freeze"
    execution = {
        "request_id": SOURCE_REQUEST_ID,
        "decision": "source-support-freeze-ready",
        "freeze_artifact_id": "a" * 64,
        "source_group_count": 10,
        "forbidden_target_group_count": 12,
        "source_rgb_frames_decoded": False,
        "source_prediction_payloads_opened": False,
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "comparison_execution_authorized": False,
    }
    freeze = {
        "artifact_id": "a" * 64,
        "decision": "source-support-freeze-ready",
        "source_group_count": 10,
        "forbidden_target_group_count": 12,
        "information_boundary": {
            "source_rgb_frames_decoded": False,
            "source_prediction_payloads_opened": False,
            "source_residuals_or_truth_opened": False,
            "source_future_geometry_opened": False,
            "target_payloads_opened": False,
            "target_outcomes_opened": False,
            "downstream_physical_innovations_opened": False,
        },
    }
    _write_json(payload / "execution-summary.json", execution)
    _write_json(payload / "cut3r-deform360-source-freeze.json", freeze)
    _write_json(payload / "cut3r-comparison-spec.json", {"schema": "spec"})
    _write_json(payload / "cut3r-comparison-lock.json", {"schema": "lock"})
    _write_json(payload / "cut3r-comparison-summary.json", {"schema": "summary"})

    rows = []
    for path in sorted(evidence.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(evidence).as_posix()}")
    (evidence / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return execution


def test_checked_in_publication_request_is_target_closed() -> None:
    module = _module()
    request = module.validate_request(REQUEST)

    assert request["publication_request_id"] == SOURCE_REQUEST_ID
    assert request["source_request_id"] == SOURCE_REQUEST_ID
    assert request["artifact_name_prefix"].startswith(
        "cut3r-source-freeze-v2-8f3c9fba12f8-"
    )
    assert "issue_number" not in request
    assert request["source_group_count"] == 10
    assert request["forbidden_target_group_count"] == 12
    assert request["source_rgb_frames_decoded"] is False
    assert request["source_residuals_or_truth_opened"] is False
    assert request["target_payloads_opened"] is False
    assert request["target_outcomes_opened"] is False
    assert set(request["output_paths"]) == set(module.OUTPUT_NAMES)


def test_artifact_validation_and_exact_publication(tmp_path: Path) -> None:
    module = _module()
    artifact_root = tmp_path / "artifact"
    execution = _artifact_fixture(artifact_root)
    request = module.validate_request(REQUEST)

    files, measured_execution = module.validate_artifact(
        artifact_root,
        request=request,
    )
    assert measured_execution == execution

    repository = tmp_path / "repository"
    artifact = {
        "id": 123,
        "name": "cut3r-source-freeze-v2-8f3c9fba12f8-123-1",
        "workflow_run": {"id": 456},
    }
    receipt = module._publish_exact_files(
        repository_root=repository,
        request=request,
        files=files,
        artifact=artifact,
        execution=measured_execution,
    )

    assert receipt["source_workflow_artifact_id"] == 123
    assert receipt["source_workflow_run_id"] == 456
    for key, relative in request["output_paths"].items():
        destination = repository / relative
        assert destination.read_bytes() == files[key].read_bytes()


def test_artifact_rejects_target_access(tmp_path: Path) -> None:
    module = _module()
    artifact_root = tmp_path / "artifact"
    _artifact_fixture(artifact_root)
    execution_path = next(artifact_root.rglob("execution-summary.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["target_outcomes_opened"] = True
    _write_json(execution_path, execution)

    with pytest.raises(ValueError, match="checksum mismatch|target-closed"):
        module.validate_artifact(
            artifact_root,
            request=module.validate_request(REQUEST),
        )


def test_safe_zip_extraction_rejects_escape(tmp_path: Path) -> None:
    module = _module()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as destination:
        destination.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        module._safe_extract_zip(archive.read_bytes(), tmp_path / "output")


def test_publication_refuses_different_existing_lock(tmp_path: Path) -> None:
    module = _module()
    artifact_root = tmp_path / "artifact"
    execution = _artifact_fixture(artifact_root)
    request = module.validate_request(REQUEST)
    files, _ = module.validate_artifact(artifact_root, request=request)
    repository = tmp_path / "repository"
    first_path = repository / request["output_paths"]["source_freeze"]
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text("different\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="different lock bytes"):
        module._publish_exact_files(
            repository_root=repository,
            request=request,
            files=files,
            artifact={
                "id": 1,
                "name": "artifact",
                "workflow_run": {"id": 2},
            },
            execution=execution,
        )
