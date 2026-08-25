from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "cut3r_source_comparison_smoke_retry_v2.json"
)
FIRST_SUMMARY = ROOT / "evidence" / "cut3r-source-comparison-smoke-v1" / "summary.json"
HELPER = ROOT / "scripts" / "ci" / "cut3r_source_comparison_smoke_retry_v2.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "cut3r-source-comparison-smoke-retry-v2.yml"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return cast(dict[str, Any], value)


def test_authorization_is_content_addressed_and_one_shot() -> None:
    authorization = _load(AUTHORIZATION)
    recorded = authorization.pop("authorization_id")
    assert recorded == hashlib.sha256(_canonical_json(authorization)).hexdigest()
    assert authorization["schema"] == (
        "prob4d.cut3r-source-comparison-smoke-retry-authorization"
    )
    assert authorization["schema_version"] == 1
    assert authorization["decision"] == (
        "one-exact-zero-progress-smoke-replacement-authorized"
    )
    limits = authorization["authorization"]
    assert limits == {
        "bayesian_phystwin_authorized": False,
        "causal4d_authorized": False,
        "full_source_shards_authorized": False,
        "maximum_replacement_attempts": 1,
        "source_truth_or_residuals_authorized": False,
        "target_access_authorized": False,
    }


def test_authorization_matches_the_retained_zero_progress_failure() -> None:
    authorization = _load(AUTHORIZATION)
    summary = _load(FIRST_SUMMARY)
    first = authorization["first_smoke"]
    assert summary["artifact_id"] == first["artifact_id"]
    assert summary["execution_plan_id"] == first["plan_id"]
    assert summary["frozen_implementation_revision"] == first["implementation_revision"]
    assert summary["attempt_count"] == 1
    assert summary["ordinary_success_count"] == 0
    assert summary["retained_technical_failure_count"] == 1
    assert summary["output_file_count_after_failure"] == 0
    assert summary["failure"] == first["failure"]
    assert summary["retry_authorized"] is False
    assert summary["retry_performed"] is False
    for field, value in first["zero_progress"].items():
        assert summary["information_boundary"][field] is value


def test_helper_replays_the_exact_authorization() -> None:
    subprocess.run(
        (
            sys.executable,
            str(HELPER),
            "validate",
            "--repository",
            str(ROOT),
            "--authorization",
            str(AUTHORIZATION),
            "--first-summary",
            str(FIRST_SUMMARY),
        ),
        check=True,
        capture_output=True,
        text=True,
    )


def test_workflow_is_hosted_authorized_and_read_only_on_workstation2() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "\n  pull_request:" in text
    assert "\n  issue_comment:" in text
    assert "pull_request_target:" not in text
    assert "github.event.issue.number == 49" in text
    assert "github.actor == 'FlorianPfaff'" in text
    assert "github.event.comment.user.login == 'FlorianPfaff'" in text
    assert "runs-on: [self-hosted, host-workstation2]" in text
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert 'test "$RUNNER_OS" = "Linux"' in text
    assert 'test "$RUNNER_ARCH" = "X64"' in text
    assert "physical_gpu_index=1" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "permissions:\n      actions: read\n      contents: read\n      issues: read" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "sudo " not in text
    assert "safe.directory=*" not in text
    assert "persist-credentials: false" in text


def test_workflow_binds_plan_custody_and_information_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "adapt-plan" in text
    assert "dust3r.inference.inference" not in text
    assert "verify_cut3r_source_comparison_artifacts.py shard" in text
    assert "--allow-technical-failures" in text
    assert "ordinary-success-development-smoke" in text
    assert "full source shards" in text
    assert "BayesianPhysTwin" in text
    assert "Causal4D" in text
    assert "Reject a duplicate retained attempt" in text
    assert "a retained replacement artifact already exists" in text
    assert "a terminal replacement receipt already exists" in text
