from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from prob4d.cut3r_source_comparison_plan import (
    AMENDED_EXECUTION_PLAN_VERSION,
    AMENDMENT_IMPLEMENTATION_FILES,
    IMPLEMENTATION_FILES,
    PROVIDER_FILES,
    _content_id,
    build_amended_execution_plan,
    build_execution_plan,
    validate_execution_plan,
)
from prob4d.cut3r_source_comparison_verifier import path_identity_sha256

ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _provider(tmp_path: Path) -> tuple[Path, Path, str, str]:
    checkout = tmp_path / "cut3r"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "tests@example.invalid")
    _git(checkout, "config", "user.name", "Prob4D tests")
    for relative in PROVIDER_FILES:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {relative}\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "synthetic provider")
    revision = _git(checkout, "rev-parse", "HEAD")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"synthetic checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return checkout, checkpoint, revision, digest


def _implementation_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "prob4d"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Prob4D tests")
    for relative in (*IMPLEMENTATION_FILES, *AMENDMENT_IMPLEMENTATION_FILES):
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "synthetic implementation")
    return repository


def _preflight(path: Path, *, revision: str, checkpoint_sha256: str) -> dict[str, object]:
    cases = [
        {
            "case_id": f"case-{index:02d}",
            "group_id": f"group-{index // 4:02d}",
            "role": (
                "development"
                if index < 8
                else "calibration"
                if index < 24
                else "source_evaluation"
            ),
            "relative_video_path": f"source/case-{index:02d}/video.mp4",
            "video_sha256": f"{index:02x}".ljust(64, "a"),
            "video_byte_count": 1000 + index,
        }
        for index in range(40)
    ]
    report: dict[str, object] = {
        "schema": "prob4d.cut3r-deform360-source-comparison-preflight",
        "schema_version": 1,
        "decision": "source-comparison-preflight-ready",
        "resolved_case_count": 40,
        "resolved_group_count": 10,
        "source_freeze_id": "b" * 64,
        "comparison_lock_id": "c" * 64,
        "cases": cases,
        "cut3r": {
            "checkout_revision": revision,
            "checkpoint_filename": "model.pth",
            "checkpoint_sha256": checkpoint_sha256,
        },
        "source_rgb_frames_decoded": False,
        "cut3r_inference_executed": False,
        "source_prediction_payloads_opened": False,
        "source_residuals_or_truth_opened": False,
        "candidate_reference_file_contents_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "comparison_execution_authorized": False,
    }
    report["artifact_id"] = _content_id(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _parent_plan_and_failure(
    tmp_path: Path,
    *,
    repository: Path,
    preflight_path: Path,
    checkout: Path,
    checkpoint: Path,
) -> tuple[Path, Path, dict[str, object]]:
    parent_plan = build_execution_plan(
        repository=repository,
        preflight_path=preflight_path,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
    )
    parent_plan["provider"]["callable"] = "src.dust3r.inference.inference"
    unsigned_plan = dict(parent_plan)
    unsigned_plan.pop("plan_id")
    parent_plan["plan_id"] = _content_id(unsigned_plan)
    parent_path = tmp_path / "parent-plan.json"
    parent_path.write_text(json.dumps(parent_plan), encoding="utf-8")
    result: dict[str, object] = {
        "schema": "prob4d.cut3r-source-comparison-smoke-result",
        "schema_version": 1,
        "decision": "pre-science-technical-failure-no-retry",
        "execution_plan_id": parent_plan["plan_id"],
        "case_id_sha256": hashlib.sha256(b"case-00").hexdigest(),
        "attempt_count": 1,
        "ordinary_success_count": 0,
        "retained_technical_failure_count": 1,
        "output_file_count_after_failure": 0,
        "retry_authorized": False,
        "retry_performed": False,
        "failure": {"terminal_stage": "initialize-cut3r-runtime"},
        "information_boundary": {
            "source_rgb_frames_decoded": False,
            "cut3r_inference_executed": False,
            "source_predictions_written": False,
            "source_residuals_or_truth_opened": False,
            "target_payloads_opened": False,
            "target_outcomes_opened": False,
        },
    }
    result["artifact_id"] = _content_id(result)
    result_path = tmp_path / "parent-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return parent_path, result_path, parent_plan


def test_plan_binds_source_bytes_and_frozen_execution_semantics(tmp_path: Path) -> None:
    repository = _implementation_repository(tmp_path)
    checkout, checkpoint, revision, checkpoint_sha = _provider(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    preflight = _preflight(
        preflight_path,
        revision=revision,
        checkpoint_sha256=checkpoint_sha,
    )

    plan = build_execution_plan(
        repository=repository,
        preflight_path=preflight_path,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
    )
    validated = validate_execution_plan(
        plan,
        repository=repository,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
    )

    assert validated["preflight_artifact_id"] == preflight["artifact_id"]
    assert validated["method"]["window_schedule"] == [
        {"start": 0, "stop": 25, "window_id": "window-000000-000025"},
        {"start": 17, "stop": 42, "window_id": "window-000017-000042"},
        {"start": 33, "stop": 58, "window_id": "window-000033-000058"},
    ]
    assert validated["method"]["fused_mean"].startswith("decoded-uniform")
    assert validated["method"]["control_uses_same_windows_gauges_and_uncertainty"] is True
    assert validated["information_boundary"]["target_outcomes_opened"] is False


def test_plan_rejects_boundary_mutation(tmp_path: Path) -> None:
    repository = _implementation_repository(tmp_path)
    checkout, checkpoint, revision, checkpoint_sha = _provider(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    _preflight(preflight_path, revision=revision, checkpoint_sha256=checkpoint_sha)
    plan = build_execution_plan(
        repository=repository,
        preflight_path=preflight_path,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
    )
    plan["information_boundary"]["target_outcomes_opened"] = True
    unsigned = dict(plan)
    unsigned.pop("plan_id")
    plan["plan_id"] = _content_id(unsigned)

    with pytest.raises(ValueError, match="target_outcomes_opened"):
        validate_execution_plan(plan)


def test_amended_plan_registers_one_different_smoke_without_method_change(
    tmp_path: Path,
) -> None:
    repository = _implementation_repository(tmp_path)
    checkout, checkpoint, revision, checkpoint_sha = _provider(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    _preflight(preflight_path, revision=revision, checkpoint_sha256=checkpoint_sha)
    parent_path, result_path, parent_plan = _parent_plan_and_failure(
        tmp_path,
        repository=repository,
        preflight_path=preflight_path,
        checkout=checkout,
        checkpoint=checkpoint,
    )
    smoke_output_root = tmp_path / "registered-smoke-output"
    smoke_attempt_ledger = tmp_path / "attempts" / "smoke-attempt.json"

    plan = build_amended_execution_plan(
        repository=repository,
        preflight_path=preflight_path,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
        parent_plan_path=parent_path,
        parent_smoke_result_path=result_path,
        smoke_output_root=smoke_output_root,
        smoke_attempt_ledger=smoke_attempt_ledger,
    )
    validated = validate_execution_plan(plan)

    assert validated["schema_version"] == AMENDED_EXECUTION_PLAN_VERSION
    assert validated["method"] == parent_plan["method"]
    assert validated["cases"] == parent_plan["cases"]
    assert validated["execution"]["smoke_policy"]["registered_case_id"] == "case-01"
    assert validated["execution"]["smoke_policy"]["attempt_limit"] == 1
    assert validated["execution"]["smoke_policy"][
        "registered_output_root_path_sha256"
    ] == path_identity_sha256(smoke_output_root)
    assert validated["execution"]["smoke_policy"][
        "registered_attempt_ledger_path_sha256"
    ] == path_identity_sha256(smoke_attempt_ledger)
    assert validated["amendment"]["prior_retry_authorized"] is False
    assert validated["amendment"]["method_changed"] is False


def test_amended_plan_rejects_reuse_of_failed_case(tmp_path: Path) -> None:
    repository = _implementation_repository(tmp_path)
    checkout, checkpoint, revision, checkpoint_sha = _provider(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    _preflight(preflight_path, revision=revision, checkpoint_sha256=checkpoint_sha)
    parent_path, result_path, _ = _parent_plan_and_failure(
        tmp_path,
        repository=repository,
        preflight_path=preflight_path,
        checkout=checkout,
        checkpoint=checkpoint,
    )
    plan = build_amended_execution_plan(
        repository=repository,
        preflight_path=preflight_path,
        cut3r_checkout=checkout,
        checkpoint=checkpoint,
        parent_plan_path=parent_path,
        parent_smoke_result_path=result_path,
        smoke_output_root=tmp_path / "registered-smoke-output",
        smoke_attempt_ledger=tmp_path / "attempts" / "smoke-attempt.json",
    )
    plan["execution"]["smoke_policy"]["registered_case_id_sha256"] = plan["amendment"][
        "prior_case_id_sha256"
    ]
    unsigned = dict(plan)
    unsigned.pop("plan_id")
    plan["plan_id"] = _content_id(unsigned)

    with pytest.raises(ValueError, match="reuse the failed case|hash is invalid"):
        validate_execution_plan(plan)
