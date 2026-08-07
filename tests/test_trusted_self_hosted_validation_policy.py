from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
TRUSTED_WORKFLOW = WORKFLOW_ROOT / "trusted-self-hosted-validation.yml"
REMOVED_TEMPORARY_WORKFLOWS = (
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory.yml",
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory-launch.yml",
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory-priority.yml",
)


def _workflow_files() -> tuple[Path, ...]:
    return tuple(
        sorted(WORKFLOW_ROOT.glob("*.yml"))
        + sorted(WORKFLOW_ROOT.glob("*.yaml"))
    )


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


def test_only_the_protected_manual_workflow_can_use_self_hosted_runners() -> None:
    assert TRUSTED_WORKFLOW.is_file()
    offenders = []
    for path in _workflow_files():
        if path == TRUSTED_WORKFLOW:
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
    assert 'DISPATCH_REF: ${{ github.ref }}' in text
    assert "refs/heads/main" in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "[0-9a-f]{40}" in text
    assert "only same-repository pull requests are admitted" in text
    assert "pull request base must be main" in text
    assert "actual_head_sha != expected_head_sha" in text
    assert 'ref: ${{ needs.authorize.outputs.head_sha }}' in text
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
    assert '$(/usr/bin/git rev-parse HEAD)' in text
    assert '/usr/bin/rm -rf -- "$root"' in text
    assert "/usr/bin/git reset --hard HEAD" in text
    assert "/usr/bin/git clean -ffdx" in text
    assert "/usr/bin/git restore --worktree -- src/prob4d.egg-info" in text
    assert "/usr/bin/rm -rf build dist ./*.egg-info src/*.egg-info" not in text


def test_privileged_profiles_are_fixed_and_reports_bind_exact_source() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "full-validation" in text
    assert "production-memory" in text
    assert "--frames 25 --height 320 --width 640 --contributors 3" in text
    assert "--include-flow" in text
    assert 'report["repository_revision"] != os.environ["EXPECTED_HEAD_SHA"]' in text
    assert "git push" not in text


def test_completed_temporary_inventory_workflows_are_removed() -> None:
    for path in REMOVED_TEMPORARY_WORKFLOWS:
        assert not path.exists(), path.relative_to(ROOT).as_posix()
