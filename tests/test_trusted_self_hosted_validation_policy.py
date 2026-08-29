from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
TRUSTED_WORKFLOW = WORKFLOW_ROOT / "trusted-self-hosted-validation.yml"
SOURCE_FREEZE_EXECUTION_WORKFLOW = WORKFLOW_ROOT / "cut3r-source-freeze-execution.yml"
SOURCE_FREEZE_AUTO_V2_WORKFLOW = WORKFLOW_ROOT / "cut3r-source-freeze-auto-v2.yml"
CUT3R_CHECKOUT_TRUST_RERUN_WORKFLOW = WORKFLOW_ROOT / "cut3r-checkout-trust-rerun-v1.yml"
CUT3R_SOURCE_COMPARISON_V2_WORKFLOW = WORKFLOW_ROOT / "cut3r-source-comparison-v2.yml"
POINTWORLD_MODEL_LOAD_SMOKE_WORKFLOW = WORKFLOW_ROOT / "pointworld-model-load-smoke.yml"
CUT3R_RUNNER_SELECTOR = (
    "runs-on: [self-hosted, Linux, X64, nvidia-smi, "
    + "data-prob4d-deform360-source-v1, prob4d-cut3r]"
)
CUT3R_AUTO_V2_RUNNER_SELECTOR = "runs-on: [self-hosted, host-workstation2]"
TRUSTED_SELF_HOSTED_WORKFLOWS = (
    TRUSTED_WORKFLOW,
    SOURCE_FREEZE_EXECUTION_WORKFLOW,
    SOURCE_FREEZE_AUTO_V2_WORKFLOW,
    CUT3R_CHECKOUT_TRUST_RERUN_WORKFLOW,
    POINTWORLD_MODEL_LOAD_SMOKE_WORKFLOW,
)
REMOVED_TEMPORARY_WORKFLOWS = (
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory.yml",
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory-launch.yml",
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory-priority.yml",
)


def _workflow_files() -> tuple[Path, ...]:
    return tuple(sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(WORKFLOW_ROOT.glob("*.yaml")))


def _uses_self_hosted_runner(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("runs-on:"):
            continue
        indentation = len(line) - len(stripped)
        if "self-hosted" in stripped.partition(":")[2]:
            return True
        for continuation in lines[index + 1 :]:
            continuation_stripped = continuation.lstrip()
            if not continuation_stripped:
                continue
            continuation_indentation = len(continuation) - len(continuation_stripped)
            if continuation_indentation <= indentation:
                break
            if "self-hosted" in continuation:
                return True
    return False


def test_only_reviewed_protected_workflows_can_use_self_hosted_runners() -> None:
    assert all(path.is_file() for path in TRUSTED_SELF_HOSTED_WORKFLOWS)
    offenders = []
    for path in _workflow_files():
        if path in TRUSTED_SELF_HOSTED_WORKFLOWS:
            continue
        text = path.read_text(encoding="utf-8")
        if _uses_self_hosted_runner(text):
            offenders.append(path.relative_to(ROOT).as_posix())
        assert "ci/self-hosted-" not in text, path
        assert "SELF_HOSTED_RECOVERY" not in text, path
    assert offenders == []


def test_trusted_workflow_is_manual_main_bound_and_exact_sha_authorized() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "\n  pull_request:" not in text
    assert "pull_request_target:" not in text
    assert "DISPATCH_REF: ${{ github.ref }}" in text
    assert "refs/heads/main" in text
    assert "environment: trusted-self-hosted-validation" in text
    assert CUT3R_RUNNER_SELECTOR in text
    assert "[0-9a-f]{40}" in text
    assert "only same-repository pull requests are admitted" in text
    assert "pull request base must be main" in text
    assert "actual_head_sha != expected_head_sha" in text
    assert "ref: ${{ needs.authorize.outputs.head_sha }}" in text
    assert "persist-credentials: false" in text


def test_trusted_workflow_uses_immutable_workspace_output_and_absolute_tools() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n  pull-requests: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "secrets." not in text
    assert 'echo "root=$root" >> "$GITHUB_OUTPUT"' in text
    assert "steps.workspace.outputs.root" in text
    assert "TRUSTED_ROOT" not in text
    assert "GITHUB_PATH" not in text
    assert '"$root/venv/bin/python"' in text
    assert "$(/usr/bin/git rev-parse HEAD)" in text
    assert '/usr/bin/rm -rf -- "$root"' in text
    assert "/usr/bin/git reset --hard HEAD" in text
    assert "/usr/bin/git clean -ffdx" in text
    assert "/usr/bin/git restore --worktree -- src/prob4d.egg-info" in text
    assert "/usr/bin/rm -rf build dist ./*.egg-info src/*.egg-info" not in text


def test_privileged_profiles_are_fixed_and_reports_bind_exact_source() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "full-validation" in text
    assert "production-memory" in text
    assert "cut3r-deform360-source-freeze" in text
    assert "build_cut3r_deform360_source_freeze.py" in text
    assert "DEFORM360_PROCESSED_ROOT" in text
    assert "status -ne 0 && $status -ne 3" in text
    assert "--frames 25 --height 320 --width 640 --contributors 3" in text
    assert "--include-flow" in text
    assert ('report["repository_revision"] != os.environ["EXPECTED_HEAD_SHA"]') in text
    assert "git push" not in text


def test_source_freeze_execution_is_merged_main_bound_and_target_closed() -> None:
    text = SOURCE_FREEZE_EXECUTION_WORKFLOW.read_text(encoding="utf-8")

    assert "\n  push:" in text
    assert "branches: [main]" in text
    assert ("protocols/execution_requests/cut3r_deform360_source_freeze_v1.json") in text
    assert "pull_request_target:" not in text
    assert "github.event_name == 'push'" in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert "source_protocol_git_blob_sha" in text
    assert "execution request ID mismatch" in text
    assert "environment: trusted-self-hosted-validation" in text
    assert CUT3R_RUNNER_SELECTOR in text
    assert "ref: ${{ needs.authorize.outputs.head_sha }}" in text
    assert "persist-credentials: false" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "secrets." not in text
    assert "comparison_execution_authorized" in text
    assert '"source_rgb_frames_decoded"' in text
    assert '"target_payloads_opened"' in text
    assert '"target_outcomes_opened"' in text
    assert "status -ne 0 && $status -ne 3" in text
    assert "git push" not in text


def test_source_freeze_execution_keeps_write_permission_off_self_hosted_job() -> None:
    text = SOURCE_FREEZE_EXECUTION_WORKFLOW.read_text(encoding="utf-8")

    execute_start = text.index("\n  execute:")
    report_start = text.index("\n  report:")
    execute = text[execute_start:report_start]
    report = text[report_start:]

    assert "permissions:\n      contents: read" in execute
    assert "issues: write" not in execute
    assert "contents: write" not in execute
    assert "permissions:\n      contents: read\n      issues: write" in report
    assert "runs-on: ubuntu-latest" in report


def test_source_freeze_execution_binds_the_builders_canonical_identity() -> None:
    text = SOURCE_FREEZE_EXECUTION_WORKFLOW.read_text(encoding="utf-8")
    execute_start = text.index("\n  execute:")
    report_start = text.index("\n  report:")
    execute = text[execute_start:report_start]

    assert 'freeze["artifact_id"]' not in execute
    assert "freeze['artifact_id']" not in execute
    assert 'freeze["source_freeze_id"]' in execute
    assert 'identity.pop("source_freeze_id")' in execute
    assert "hashlib.sha256(canonical).hexdigest()" in execute
    assert '"freeze_artifact_id": source_freeze_id' in execute
    assert "freeze_artifact_id={source_freeze_id}" in execute


def test_auto_v2_source_freeze_supports_exact_retry_and_bounded_queue() -> None:
    text = SOURCE_FREEZE_AUTO_V2_WORKFLOW.read_text(encoding="utf-8")

    assert "\n  push:" in text
    assert "\n  workflow_dispatch:" in text
    assert "execution_sha:" in text
    assert "request_id:" in text
    assert "branches: [main]" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_AFTER" = "$EXPECTED_SHA"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "authorize-retry" in text
    assert '--execution-revision "$EXECUTION_SHA"' in text
    assert '--expected-request-id "$EXPECTED_REQUEST_ID"' in text
    assert "ref: ${{ github.sha }}" in text
    assert ('git worktree add --detach "$execution_repository" "$EXPECTED_HEAD_SHA"') in text
    assert '--repository "$execution_repository"' in text
    assert '--request "$execution_repository/$REQUEST_PATH"' in text
    assert '"$build_python" -m build \\' in text
    assert '"$execution_repository"' in text
    assert "scripts/science/run_cut3r_source_freeze_execution.py execute \\" in text
    assert '"$execution_repository/scripts/science' not in text
    assert "control_plane_sha=$CURRENT_CONTROL_PLANE_SHA" in text
    assert "ls-files \\" in text
    assert "--error-unmatch -- src/prob4d.egg-info" in text
    assert '/usr/bin/git worktree remove --force "$execution_repository"' in text
    assert CUT3R_AUTO_V2_RUNNER_SELECTOR in text
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert 'test "$RUNNER_OS" = "Linux"' in text
    assert 'test "$RUNNER_ARCH" = "X64"' in text
    assert "command -v nvidia-smi" in text
    assert "check-variables" in text
    assert "missing repository-variable names" in text
    assert "Bound self-hosted runner acceptance wait" in text
    assert 'RUNNER_ACCEPTANCE_TIMEOUT_SECONDS: "1200"' in text
    assert "/actions/runs/{run_id}/jobs" in text
    assert "/actions/runs/{run_id}/cancel" in text
    assert "actions: write" in text

    execute_start = text.index("\n  execute:")
    watchdog_start = text.index("\n  watchdog:")
    report_start = text.index("\n  report:")
    execute = text[execute_start:watchdog_start]
    watchdog = text[watchdog_start:report_start]
    report = text[report_start:]

    assert "success() &&" in execute
    assert "github.event_name == 'workflow_dispatch'" in execute
    assert "needs: [contract, authorize, preflight, announce]" in execute
    assert "permissions:\n      contents: read" in execute
    assert "issues: write" not in execute
    assert "actions: write" not in execute
    assert "contents: write" not in execute
    assert "pull-requests: write" not in execute
    assert "persist-credentials: false" in execute
    assert "secrets." not in execute
    assert "repository_write_token_on_self_hosted=false" in execute
    assert "environment_approval_required=false" in execute
    assert '/usr/bin/python3 - "$root/evidence"' in execute
    assert 'python - "$root/evidence"' not in execute
    assert "git push" not in execute

    assert "runs-on: ubuntu-latest" in watchdog
    assert "actions: write" in watchdog
    assert "issues: write" in watchdog
    assert "runner acceptance timeout" in watchdog
    assert "always() &&" in report
    assert "github.event_name == 'workflow_dispatch'" in report
    assert "runs-on: ubuntu-latest" in report
    assert "issues: write" in report


def test_checkout_trust_rerun_is_exact_temporary_and_hosted_dispatched() -> None:
    text = CUT3R_CHECKOUT_TRUST_RERUN_WORKFLOW.read_text(encoding="utf-8")

    assert "\n  issue_comment:" in text
    assert "\n  workflow_run:" in text
    assert "pull_request_target:" not in text
    assert CUT3R_AUTO_V2_RUNNER_SELECTOR in text
    assert 'test "$RUNNER_NAME_VALUE" = "workstation2"' in text
    assert '-c safe.directory="$CUT3R_CHECKOUT"' in text
    assert "git config --system --add safe.directory" in text
    assert "--fixed-value" in text
    assert "--unset-all safe.directory" in text
    assert 'FAILED_RUN_ID: "32771242880"' in text
    assert 'FAILED_EXECUTE_JOB_ID: "97702294765"' in text
    assert 'FAILURE_ARTIFACT_ID: "9551181122"' in text
    assert "actions/jobs/{job_id}/rerun" in text
    assert "git push" not in text
    assert "secrets." not in text

    prepare_start = text.index("\n  prepare:")
    rerun_start = text.index("\n  rerun:")
    cleanup_start = text.index("\n  cleanup:")
    cleanup_report_start = text.index("\n  cleanup_report:")
    prepare = text[prepare_start:rerun_start]
    rerun = text[rerun_start:cleanup_start]
    cleanup = text[cleanup_start:cleanup_report_start]
    cleanup_report = text[cleanup_report_start:]

    assert "permissions:\n      contents: read" in prepare
    assert "actions: write" not in prepare
    assert "issues: write" not in prepare
    assert "runs-on: ubuntu-latest" in rerun
    assert "actions: write" in rerun
    assert "issues: write" in rerun
    assert "permissions:\n      contents: read" in cleanup
    assert "actions: write" not in cleanup
    assert "issues: write" not in cleanup
    assert "runs-on: ubuntu-latest" in cleanup_report
    assert "issues: write" in cleanup_report


def test_pointworld_model_load_smoke_is_main_request_bound_and_target_closed() -> None:
    text = POINTWORLD_MODEL_LOAD_SMOKE_WORKFLOW.read_text(encoding="utf-8")

    assert "\n  push:" in text
    assert "branches: [main]" in text
    assert "protocols/execution_requests/pointworld_model_load_smoke_v1.json" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert "validate-request" in text
    assert "source_protocol_git_blob_sha" in text
    assert "environment: trusted-self-hosted-validation" in text
    assert CUT3R_AUTO_V2_RUNNER_SELECTOR in text
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert 'test "$RUNNER_OS" = "Linux"' in text
    assert 'test "$RUNNER_ARCH" = "X64"' in text
    assert "command -v nvidia-smi" in text
    assert "persist-credentials: false" in text
    assert "dataset_access_authorized" in text
    assert "prediction_execution_authorized" in text
    assert "provider_residuals_authorized" in text
    assert "target_outcomes_authorized" in text
    assert "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035" not in text
    assert "git push" not in text
    assert "secrets." not in text

    execute_start = text.index("\n  execute:")
    report_start = text.index("\n  report:")
    execute = text[execute_start:report_start]
    report = text[report_start:]

    assert "permissions:\n      contents: read" in execute
    assert "issues: write" not in execute
    assert "contents: write" not in execute
    assert "prediction_executed" in execute
    assert "dataset_opened" in execute
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in execute
    assert "permissions:\n      contents: read\n      issues: write" in report
    assert "runs-on: ubuntu-latest" in report


def test_source_comparison_v2_issue_trigger_is_terminally_removed() -> None:
    assert not CUT3R_SOURCE_COMPARISON_V2_WORKFLOW.exists()


def test_full_validation_tracks_the_05_cleanup_surface() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "src/prob4d/api/v1.py" not in text
    assert "src/prob4d/_provider_export_core.py" in text
    assert "src/prob4d/api/v2.py" in text
    assert "src/prob4d/provider_v1.py" in text
    assert "src/prob4d/public_api_manifest.py" in text


def test_completed_temporary_inventory_workflows_are_removed() -> None:
    for path in REMOVED_TEMPORARY_WORKFLOWS:
        assert not path.exists(), path.relative_to(ROOT).as_posix()
