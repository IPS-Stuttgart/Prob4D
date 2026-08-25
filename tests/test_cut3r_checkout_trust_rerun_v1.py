from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cut3r-checkout-trust-rerun-v1.yml"
EXACT_COMMAND = (
    "/prob4d-repair-cut3r-checkout-trust-and-rerun-v1 "
    "32771242880 9551181122 "
    "d1eff3af637eb297e72334693b1c51723f4eb9487a6cf6a8957d130bc34b9721"
)


def _job(text: str, name: str, next_name: str | None) -> str:
    start = text.index(f"\n  {name}:")
    if next_name is None:
        return text[start:]
    return text[start : text.index(f"\n  {next_name}:", start + 1)]


def test_repair_is_exact_issue_bound_and_not_pull_request_privileged() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "\n  issue_comment:" in text
    assert "\n  workflow_run:" in text
    assert "pull_request_target:" not in text
    assert EXACT_COMMAND in text
    assert "github.event.issue.number == 49" in text
    assert "github.actor == 'FlorianPfaff'" in text
    assert "github.event.comment.user.login == 'FlorianPfaff'" in text
    assert "ref: ${{ github.sha }}" in text
    assert "persist-credentials: false" in text
    assert "secrets." not in text
    assert "git push" not in text


def test_authorization_binds_the_exact_pre_provider_failure_capsule() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        'FAILED_RUN_ID: "32771242880"',
        'FAILED_RUN_ATTEMPT: "2"',
        "FAILED_HEAD_SHA: 30f85ecd6c2395b333f723db47a18f6677d7c58f",
        'FAILED_EXECUTE_JOB_ID: "97702294765"',
        'FAILURE_ARTIFACT_ID: "9551181122"',
        'FAILURE_ARTIFACT_SIZE: "4384"',
        (
            "FAILURE_ARTIFACT_DIGEST: "
            "sha256:d1eff3af637eb297e72334693b1c51723"
            "f4eb9487a6cf6a8957d130bc34b9721"
        ),
        "Run one target-closed source freeze",
        "failed run job roster mismatch",
        "failed run artifact count mismatch",
    )
    for value in required:
        assert value in text


def test_prepare_uses_scoped_probe_and_temporary_system_trust_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    prepare = _job(text, "prepare", "rerun")
    cleanup = _job(text, "cleanup", "cleanup_report")

    assert "runs-on: [self-hosted, host-workstation2]" in prepare
    assert 'test "$RUNNER_NAME_VALUE" = "workstation2"' in prepare
    assert 'test "$RUNNER_OS_VALUE" = "Linux"' in prepare
    assert 'test "$RUNNER_ARCH_VALUE" = "X64"' in prepare
    assert '-c safe.directory="$CUT3R_CHECKOUT"' in prepare
    assert "git config --system --add safe.directory" in prepare
    assert "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf" in text
    assert "status --porcelain=v1 --untracked-files=all" in prepare
    assert "issues: write" not in prepare
    assert "actions: write" not in prepare

    assert "runs-on: [self-hosted, host-workstation2]" in cleanup
    assert "--fixed-value" in cleanup
    assert "--unset-all safe.directory" in cleanup
    assert "issues: write" not in cleanup
    assert "actions: write" not in cleanup


def test_only_hosted_job_can_request_the_exact_job_rerun() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    rerun = _job(text, "rerun", "cleanup")

    assert "runs-on: ubuntu-latest" in rerun
    assert "actions: write" in rerun
    assert 'f"/repos/{repository}/actions/jobs/{job_id}/rerun"' in rerun
    assert '"enable_debug_logging": False' in rerun
    assert "Only the failed retained-data job" in rerun


def test_cleanup_is_bound_to_the_same_run_after_attempt_three() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    cleanup = _job(text, "cleanup", "cleanup_report")
    report = _job(text, "cleanup_report", None)

    for segment in (cleanup, report):
        assert "github.event.workflow_run.id == 32771242880" in segment
        assert "github.event.workflow_run.run_attempt >= 3" in segment
        assert "github.event.workflow_run.status == 'completed'" in segment
    assert "runs-on: ubuntu-latest" in report
    assert "issues: write" in report
    assert "No retained path value is published" in report
