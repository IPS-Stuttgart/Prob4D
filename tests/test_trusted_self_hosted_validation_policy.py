from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
TRUSTED_WORKFLOW = WORKFLOW_ROOT / "trusted-self-hosted-validation.yml"
REMOVED_TEMPORARY_WORKFLOWS = (
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory.yml",
    WORKFLOW_ROOT / "issue-49-protected-cohort-inventory-launch.yml",
)


def _workflow_files() -> tuple[Path, ...]:
    return tuple(
        sorted(WORKFLOW_ROOT.glob("*.yml"))
        + sorted(WORKFLOW_ROOT.glob("*.yaml"))
    )


def test_only_the_protected_manual_workflow_can_use_self_hosted_runners() -> None:
    assert TRUSTED_WORKFLOW.is_file()
    offenders = []
    for path in _workflow_files():
        if path == TRUSTED_WORKFLOW:
            continue
        text = path.read_text(encoding="utf-8")
        if "self-hosted" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_trusted_workflow_is_manual_main_bound_and_exact_sha_authorized() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "\n  pull_request:" not in text
    assert "pull_request_target:" not in text
    assert 'DISPATCH_REF: ${{ github.ref }}' in text
    assert 'refs/heads/main' in text
    assert 'environment: trusted-self-hosted-validation' in text
    assert 'runs-on: [self-hosted, Linux, X64, nvidia-smi]' in text
    assert '^ [0-9a-f]{40}$' not in text
    assert '^[0-9a-f]{40}$' in text
    assert '$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/pulls/$REQUESTED_PR' in text
    assert 'pull.get("state") != "open"' in text
    assert 'pull.get("base", {}).get("ref") != "main"' in text
    assert 'head_repository.get("full_name") != repository' in text
    assert 'actual_sha != requested_sha' in text
    assert 'ref: ${{ needs.authorize.outputs.head_sha }}' in text
    assert 'persist-credentials: false' in text


def test_trusted_workflow_is_read_only_isolated_and_cleans_up() -> None:
    text = TRUSTED_WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "secrets." not in text
    assert 'HOME=$root/home' in text
    assert 'XDG_CACHE_HOME=$root/cache' in text
    assert 'PIP_CACHE_DIR=$root/cache/pip' in text
    assert 'TMPDIR=$root/tmp' in text
    assert 'git reset --hard HEAD' in text
    assert 'git clean -ffdx' in text
    assert 'prob4d.trusted-exact-head-validation' in text


def test_completed_temporary_inventory_workflows_are_removed() -> None:
    for path in REMOVED_TEMPORARY_WORKFLOWS:
        assert not path.exists(), path.relative_to(ROOT).as_posix()
