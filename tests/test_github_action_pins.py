from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
_USES_LINE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<value>[^#\s]+)\s*(?:#\s*(?P<comment>.+))?$"
)
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSION_COMMENT = re.compile(r"^v[0-9]+(?:\.[0-9]+){0,2}(?:[-+][A-Za-z0-9.-]+)?$")


def _pin_errors(text: str, *, source: str) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _USES_LINE.match(line)
        if match is None:
            continue
        value = match.group("value").strip("\"'")
        if value.startswith(("./", "docker://")):
            continue
        if "@" not in value:
            errors.append(
                f"{source}:{line_number}: external action has no ref: {value}"
            )
            continue
        action, ref = value.rsplit("@", 1)
        if "/" not in action or not _FULL_COMMIT.fullmatch(ref):
            errors.append(
                f"{source}:{line_number}: external action must use a full "
                f"lowercase commit SHA: {value}"
            )
        comment = (match.group("comment") or "").strip().split(maxsplit=1)[0:1]
        if not comment or _VERSION_COMMENT.fullmatch(comment[0]) is None:
            errors.append(
                f"{source}:{line_number}: immutable action pin needs a version "
                f"annotation such as '# v7': {value}"
            )
    return errors


def _checkout_credential_errors(text: str, *, source: str) -> list[str]:
    lines = text.splitlines()
    errors: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("- uses: actions/checkout@"):
            continue
        indentation = len(line) - len(stripped)
        block: list[str] = []
        for candidate in lines[index + 1 :]:
            candidate_stripped = candidate.lstrip()
            candidate_indentation = len(candidate) - len(candidate_stripped)
            if candidate_stripped.startswith("- ") and candidate_indentation <= indentation:
                break
            block.append(candidate)
        if not any("persist-credentials: false" in value for value in block):
            errors.append(
                f"{source}:{index + 1}: checkout must set persist-credentials: false"
            )
    return errors


def _piped_run_block_errors(text: str, *, source: str) -> list[str]:
    """Reject diagnostic pipelines that can hide a failing producer behind tee."""

    block_run = re.compile(r"^(?P<indent>\s*)run:\s*(?:[|>][-+]?)\s*$")
    inline_run = re.compile(r"^\s*run:\s*(?P<command>.+)$")
    lines = text.splitlines()
    errors: list[str] = []
    for index, line in enumerate(lines):
        block_match = block_run.match(line)
        if block_match is not None:
            indentation = len(block_match.group("indent"))
            block: list[str] = []
            for candidate in lines[index + 1 :]:
                if not candidate.strip():
                    block.append(candidate)
                    continue
                candidate_stripped = candidate.lstrip()
                candidate_indentation = len(candidate) - len(candidate_stripped)
                if candidate_indentation <= indentation:
                    break
                block.append(candidate)
            command = "\n".join(block)
        else:
            inline_match = inline_run.match(line)
            if inline_match is None:
                continue
            command = inline_match.group("command")
        if "| tee" not in command:
            continue
        if "set -o pipefail" in command or "set -euo pipefail" in command:
            continue
        errors.append(
            f"{source}:{index + 1}: a command piped through tee must enable "
            "pipefail so diagnostics cannot mask a failing producer"
        )
    return errors


def test_every_external_github_action_is_immutably_pinned() -> None:
    workflow_files = sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(
        WORKFLOW_ROOT.glob("*.yaml")
    )
    assert workflow_files, "no GitHub Actions workflows found"

    errors: list[str] = []
    for path in workflow_files:
        errors.extend(
            _pin_errors(
                path.read_text(encoding="utf-8"),
                source=path.relative_to(ROOT).as_posix(),
            )
        )
    assert not errors, "\n".join(errors)


def test_checkout_actions_disable_persisted_credentials() -> None:
    errors: list[str] = []
    for path in sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(
        WORKFLOW_ROOT.glob("*.yaml")
    ):
        errors.extend(
            _checkout_credential_errors(
                path.read_text(encoding="utf-8"),
                source=path.relative_to(ROOT).as_posix(),
            )
        )
    assert not errors, "\n".join(errors)


def test_piped_workflow_commands_enable_pipefail() -> None:
    errors: list[str] = []
    for path in sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(
        WORKFLOW_ROOT.glob("*.yaml")
    ):
        errors.extend(
            _piped_run_block_errors(
                path.read_text(encoding="utf-8"),
                source=path.relative_to(ROOT).as_posix(),
            )
        )
    assert not errors, "\n".join(errors)


def test_pipefail_policy_rejects_masked_diagnostic_pipelines() -> None:
    unsafe_inline = "run: command 2>&1 | tee diagnostics.txt\n"
    safe_inline = (
        "run: set -o pipefail; command 2>&1 | tee diagnostics.txt\n"
    )
    unsafe_block = "run: |\n  command 2>&1 | tee diagnostics.txt\n"
    safe_block = (
        "run: |\n"
        "  set -o pipefail\n"
        "  command 2>&1 | tee diagnostics.txt\n"
    )

    assert len(
        _piped_run_block_errors(unsafe_inline, source="fixture.yml")
    ) == 1
    assert _piped_run_block_errors(safe_inline, source="fixture.yml") == []
    assert len(
        _piped_run_block_errors(unsafe_block, source="fixture.yml")
    ) == 1
    assert _piped_run_block_errors(safe_block, source="fixture.yml") == []


def test_pin_policy_rejects_tags_branches_and_short_shas() -> None:
    text = "\n".join(
        (
            "- uses: actions/checkout@v7",
            "- uses: owner/action@main",
            "- uses: owner/action@0123456789abcdef # v1",
        )
    )
    errors = _pin_errors(text, source="fixture.yml")

    assert len(errors) == 5
    assert all("fixture.yml" in error for error in errors)


def test_pin_policy_allows_local_docker_and_annotated_commit_uses() -> None:
    pinned = "0123456789abcdef0123456789abcdef01234567"
    text = "\n".join(
        (
            "- uses: ./local-action",
            "- uses: docker://python:3.12",
            f"- uses: owner/action@{pinned} # v1.2.3",
        )
    )

    assert _pin_errors(text, source="fixture.yml") == []


def test_checkout_policy_rejects_credential_persistence() -> None:
    pinned = "0123456789abcdef0123456789abcdef01234567"
    unsafe = f"- uses: actions/checkout@{pinned} # v7\n"
    safe = unsafe + "  with:\n    persist-credentials: false\n"

    assert len(_checkout_credential_errors(unsafe, source="fixture.yml")) == 1
    assert _checkout_credential_errors(safe, source="fixture.yml") == []
