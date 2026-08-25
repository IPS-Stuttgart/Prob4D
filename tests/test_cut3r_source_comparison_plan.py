from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from prob4d.cut3r_source_comparison_plan import (
    IMPLEMENTATION_FILES,
    PROVIDER_FILES,
    _content_id,
    build_execution_plan,
    validate_execution_plan,
)

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
    for relative in IMPLEMENTATION_FILES:
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
