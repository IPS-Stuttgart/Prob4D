from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prob4d.cut3r_source_comparison_verifier import (
    content_id,
    validate_case_artifact,
    validate_shard_artifact,
    write_custody_receipt,
)

PLAN_ID = "a" * 64


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(case_root: Path, manifest: dict[str, object]) -> None:
    unsigned = dict(manifest)
    unsigned.pop("artifact_id", None)
    manifest["artifact_id"] = content_id(unsigned)
    (case_root / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _member_records(case_root: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(case_root.rglob("*")):
        if not path.is_file() or path.name == "case_manifest.json":
            continue
        payload = path.read_bytes()
        result.append(
            {
                "path": path.relative_to(case_root).as_posix(),
                "sha256": _digest(payload),
                "byte_count": len(payload),
            }
        )
    return result


def _case(
    output_root: Path,
    case_id: str = "case-00",
    *,
    role: str = "development",
    status: str = "ordinary-success",
) -> Path:
    case_root = output_root / "cases" / case_id
    member = case_root / "predictions" / "result.bin"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"prediction")
    failure: str | None = None
    progress = {
        "source_rgb_frames_decoded": True,
        "cut3r_inference_executed": True,
        "source_predictions_written": True,
    }
    if status == "retained-technical-failure":
        failure = "RuntimeError: bounded synthetic failure"
        progress["cut3r_inference_executed"] = False
        progress["source_predictions_written"] = False
    manifest: dict[str, object] = {
        "schema": "prob4d.cut3r-source-comparison-case",
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "case_id": case_id,
        "group_id": "group-00",
        "role": role,
        "status": status,
        "elapsed_seconds": 1.25,
        "failure": failure,
        "members": _member_records(case_root),
        **progress,
        "source_residuals_or_truth_opened": False,
        "candidate_reference_file_contents_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }
    _write_manifest(case_root, manifest)
    return case_root


def _shard_report(
    output_root: Path,
    case_manifest: dict[str, object],
    *,
    scope: str = "development-smoke",
) -> Path:
    success = int(case_manifest["status"] == "ordinary-success")
    failure = int(case_manifest["status"] == "retained-technical-failure")
    report: dict[str, object] = {
        "schema": "prob4d.cut3r-source-comparison-shard",
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "scope": scope,
        "shard_index": 0,
        "shard_count": 2,
        "case_count": 1,
        "ordinary_success_count": success,
        "retained_technical_failure_count": failure,
        "case_artifact_ids": [case_manifest["artifact_id"]],
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
    }
    report["artifact_id"] = content_id(report)
    path = output_root / "shards" / "smoke.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_validates_complete_case_and_smoke_custody(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    case = validate_case_artifact(
        case_root,
        expected_plan_id=PLAN_ID,
        require_success=True,
    )
    report = _shard_report(tmp_path, case)

    receipt = validate_shard_artifact(
        tmp_path,
        report,
        expected_plan_id=PLAN_ID,
        require_success=True,
    )
    receipt_path = tmp_path / "custody" / "smoke.json"
    write_custody_receipt(receipt_path, receipt)
    write_custody_receipt(receipt_path, receipt)

    assert receipt["decision"] == "source-comparison-custody-valid"
    assert receipt["case_ids"] == ["case-00"]
    assert receipt["decoded_source_frames_retained"] is False
    assert receipt_path.is_file()


def test_rejects_member_byte_tampering(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    (case_root / "predictions" / "result.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_case_artifact(case_root)


def test_rejects_undeclared_members(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    (case_root / "unregistered.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="member roster mismatch"):
        validate_case_artifact(case_root)


def test_rejects_decoded_source_frames_even_when_declared(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    decoded = case_root / "decoded" / "000000.png"
    decoded.parent.mkdir()
    decoded.write_bytes(b"source-rgb")
    manifest = json.loads((case_root / "case_manifest.json").read_text(encoding="utf-8"))
    manifest["members"] = _member_records(case_root)
    _write_manifest(case_root, manifest)

    with pytest.raises(ValueError, match="decoded source frame retained"):
        validate_case_artifact(case_root)


def test_rejects_invalid_case_content_identity(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    manifest_path = case_root / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="content identity"):
        validate_case_artifact(case_root)


def test_smoke_rejects_retained_technical_failure(tmp_path: Path) -> None:
    case_root = _case(tmp_path, status="retained-technical-failure")
    case = validate_case_artifact(case_root, require_success=False)
    report = _shard_report(tmp_path, case)

    with pytest.raises(ValueError, match="technical failures"):
        validate_shard_artifact(tmp_path, report, require_success=True)

    receipt = validate_shard_artifact(tmp_path, report, require_success=False)
    assert receipt["retained_technical_failure_count"] == 1
