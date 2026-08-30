from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-source-bundle-audit.yml"


def test_workflow_is_single_request_main_bound_and_metadata_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "\n  push:" in text
    assert "branches: [main]" in text
    assert "protocols/execution_requests/deform360_source_bundle_audit_v1.json" in text
    assert "pull_request_target:" not in text
    assert 'test "$EVENT_REF" = "refs/heads/main"' in text
    assert 'test "$EVENT_FORCED" = "false"' in text
    assert 'test "$EVENT_DELETED" = "false"' in text
    assert "validate-request" in text
    assert "source_protocol_git_blob_sha" in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "runs-on: [self-hosted, gpuserver4090]" in text
    assert 'test "$RUNNER_OS" = "Linux"' in text
    assert 'test "$RUNNER_ARCH" = "X64"' in text
    assert "persist-credentials: false" in text
    assert "metadata_access_authorized" in text
    assert "file_content_reads_authorized" in text
    assert "prediction_execution_authorized" in text
    assert "provider_residuals_authorized" in text
    assert "target_payloads_authorized" in text
    assert "target_outcomes_authorized" in text
    assert "dataset_mutation_authorized" in text
    assert "git push" not in text
    assert "secrets." not in text


def test_self_hosted_job_has_read_only_repository_access_and_sanitized_output() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    execute_start = text.index("\n  execute:")
    report_start = text.index("\n  report:")
    execute = text[execute_start:report_start]
    report = text[report_start:]

    assert "permissions:\n      contents: read" in execute
    assert "issues: write" not in execute
    assert "contents: write" not in execute
    assert "pull-requests: write" not in execute
    assert "GITHUB_TOKEN" not in execute
    assert "/usr/bin/python3 scripts/science/audit_deform360_source_bundle.py" in execute
    assert '--output "$root/evidence/result.json"' in execute
    assert '--summary "$root/evidence/summary.md"' in execute
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in execute
    assert 'test "${{ steps.finalize.outputs.decision }}" = "source-bundle-present"' in execute
    assert '/usr/bin/rm -rf -- "$root"' in execute

    assert "permissions:\n      contents: read\n      issues: write" in report
    assert "runs-on: ubuntu-latest" in report
    assert "GITHUB_TOKEN: ${{ github.token }}" in report


def test_source_path_and_forbidden_boundaries_are_literal() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert ("SOURCE_ROOT: /home/florianpfaff/deform360-fresh-source-processed-v1-1a3f9b1") in text
    assert "dataset_file_contents_opened" in text
    assert "follow symlinks" in text
    assert "target_payloads_opened" in text
    assert "target_outcomes_opened" in text
    assert "dataset_mutated" in text
