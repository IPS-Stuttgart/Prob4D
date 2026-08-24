#!/usr/bin/env python3
"""Target-closed authorization and environment checks for retained CUT3R runs.

The helper has no provider, dataset, residual, truth, confirmation, or target I/O.
It verifies that a manually retried execution uses the exact historical merged-main
request bytes and reports only whether required repository-variable names are set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

REQUEST_SCHEMA: Final = "prob4d.cut3r-deform360-source-freeze-execution-request"
REQUEST_VERSION: Final = 2
PROFILE: Final = "cut3r-deform360-source-freeze"
AUTHORIZATION_MODE: Final = "merged-main-read-only-self-hosted-v1"
READINESS_SCHEMA: Final = "prob4d.cut3r-source-freeze-variable-readiness"
READINESS_VERSION: Final = 1
REQUEST_PATH: Final = (
    "protocols/execution_requests/cut3r_deform360_source_freeze_v2.json"
)
DRIVER_PATH: Final = "scripts/science/run_cut3r_source_freeze_execution.py"
EXPECTED_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "request_id",
        "issue_number",
        "profile",
        "authorization_mode",
        "supersedes_request_id",
        "source_protocol_path",
        "source_protocol_git_blob_sha",
        "source_group_count",
        "forbidden_target_group_count",
        "repository_write_token_on_self_hosted",
        "environment_approval_required",
        "source_rgb_frames_decoded",
        "source_prediction_payloads_opened",
        "source_residuals_or_truth_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "comparison_execution_authorized",
        "claim_boundary",
    }
)
FALSE_BOUNDARY_FIELDS: Final = (
    "repository_write_token_on_self_hosted",
    "environment_approval_required",
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "comparison_execution_authorized",
)
CLAIM_BOUNDARY: Final = (
    "Operational authorization and repository-variable presence only. This helper "
    "does not inspect retained paths, execute CUT3R, decode RGB, open source "
    "residuals or truth, access confirmation or target payloads, run "
    "BayesianPhysTwin or Causal4D, or establish a scientific result."
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _lower_hex(value: object, *, name: str, length: int) -> str:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(
            f"{name} must be an exact lowercase {length}-character hexadecimal value"
        )
    return value


def _variable_name(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Z][A-Z0-9_]*", value) is None:
        raise ValueError(
            "required variable names must match [A-Z][A-Z0-9_]* exactly"
        )
    return value


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an ordinary file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {name}: {error}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(dict[str, Any], value)


def _git_output(arguments: Sequence[str], *, repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git {' '.join(arguments)} failed") from error


def _git_success(arguments: Sequence[str], *, repository: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError(f"git {' '.join(arguments)} failed") from error
    if completed.returncode not in {0, 1}:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return completed.returncode == 0


def _relative_repository_path(path: Path, repository: Path, *, name: str) -> str:
    resolved_repository = repository.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_repository)
    except ValueError as error:
        raise ValueError(f"{name} must be inside the repository") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an ordinary repository file")
    return relative.as_posix()


def _validate_request_identity(request: Mapping[str, Any]) -> str:
    if set(request) != EXPECTED_REQUEST_FIELDS:
        missing = sorted(EXPECTED_REQUEST_FIELDS - set(request))
        extra = sorted(set(request) - EXPECTED_REQUEST_FIELDS)
        raise ValueError(
            "source-freeze execution request fields changed; "
            f"missing={missing}, extra={extra}"
        )
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unexpected source-freeze execution-request schema")
    if request.get("schema_version") != REQUEST_VERSION:
        raise ValueError("unsupported source-freeze execution-request version")
    if request.get("profile") != PROFILE:
        raise ValueError("source-freeze execution request names the wrong profile")
    if request.get("authorization_mode") != AUTHORIZATION_MODE:
        raise ValueError("source-freeze authorization mode changed")
    if any(request.get(name) is not False for name in FALSE_BOUNDARY_FIELDS):
        raise ValueError(
            "source-freeze execution request exceeds its target-closed boundary"
        )
    request_id = _lower_hex(request.get("request_id"), name="request_id", length=64)
    identity = dict(request)
    identity.pop("request_id", None)
    if _content_id(identity) != request_id:
        raise ValueError("source-freeze execution request ID mismatch")
    return request_id


def authorize_retry(
    *,
    repository: Path,
    request_path: Path,
    current_revision: str,
    execution_revision: str,
    expected_request_id: str,
) -> dict[str, str]:
    """Authorize one exact historical merged-main retry without changing bytes."""

    repository = repository.resolve()
    if not (repository / ".git").exists():
        raise ValueError("repository must be a Git checkout")
    request_relative = _relative_repository_path(
        request_path,
        repository,
        name="request",
    )
    if request_relative != REQUEST_PATH:
        raise ValueError("retry request path changed")

    current = _lower_hex(current_revision, name="current_revision", length=40)
    execution = _lower_hex(execution_revision, name="execution_revision", length=40)
    expected_id = _lower_hex(
        expected_request_id,
        name="expected_request_id",
        length=64,
    )
    head = _lower_hex(
        _git_output(["rev-parse", "HEAD"], repository=repository),
        name="repository HEAD",
        length=40,
    )
    if head != current:
        raise ValueError("current workflow checkout differs from current_revision")
    if _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"],
        repository=repository,
    ):
        raise ValueError("retry authorization checkout must be clean")
    if not _git_success(
        ["merge-base", "--is-ancestor", execution, current],
        repository=repository,
    ):
        raise ValueError("execution_revision is not an ancestor of current main")

    current_request_blob = _lower_hex(
        _git_output(
            ["rev-parse", f"{current}:{request_relative}"],
            repository=repository,
        ),
        name="current request blob",
        length=40,
    )
    execution_request_blob = _lower_hex(
        _git_output(
            ["rev-parse", f"{execution}:{request_relative}"],
            repository=repository,
        ),
        name="execution request blob",
        length=40,
    )
    if current_request_blob != execution_request_blob:
        raise ValueError("historical and current execution-request bytes differ")

    request = _load_json_object(request_path, name="source-freeze execution request")
    request_id = _validate_request_identity(request)
    if request_id != expected_id:
        raise ValueError("workflow-dispatch request_id differs from retained request")

    protocol_path = request.get("source_protocol_path")
    if type(protocol_path) is not str or protocol_path.startswith("/"):
        raise ValueError("source_protocol_path must be a repository-relative string")
    expected_protocol_blob = _lower_hex(
        request.get("source_protocol_git_blob_sha"),
        name="source_protocol_git_blob_sha",
        length=40,
    )
    historical_protocol_blob = _lower_hex(
        _git_output(
            ["rev-parse", f"{execution}:{protocol_path}"],
            repository=repository,
        ),
        name="historical protocol blob",
        length=40,
    )
    if historical_protocol_blob != expected_protocol_blob:
        raise ValueError("historical protocol bytes differ from the retained request")
    if not _git_success(
        ["cat-file", "-e", f"{execution}:{DRIVER_PATH}"],
        repository=repository,
    ):
        raise ValueError("historical execution revision lacks the reviewed driver")

    ancestry = _git_output(
        ["rev-list", "--parents", "-n", "1", execution],
        repository=repository,
    ).split()
    if not ancestry or ancestry[0] != execution:
        raise ValueError("failed to resolve the historical execution ancestry")
    base_revision = _lower_hex(
        ancestry[1] if len(ancestry) > 1 else execution,
        name="execution base revision",
        length=40,
    )
    return {
        "head_sha": execution,
        "base_sha": base_revision,
        "request_id": request_id,
        "profile": PROFILE,
        "authorization_mode": AUTHORIZATION_MODE,
        "trigger_mode": "workflow_dispatch_exact_retry",
        "current_main_sha": current,
    }


def variable_readiness(
    required_variables: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Return a bounded presence-only report; variable values never enter it."""

    names = tuple(sorted({_variable_name(value) for value in required_variables}))
    if not names:
        raise ValueError("at least one required variable name is required")
    missing = tuple(name for name in names if not environment.get(name, "").strip())
    return {
        "schema": READINESS_SCHEMA,
        "schema_version": READINESS_VERSION,
        "required_variables": list(names),
        "configured_variable_count": len(names) - len(missing),
        "missing_variables": list(missing),
        "ready": not missing,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_json(path: Path, value: object) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError("refusing to write readiness report through a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_github_output(path: Path, values: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            if isinstance(value, bool):
                encoded = "true" if value else "false"
            elif isinstance(value, (list, dict)):
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                encoded = str(value)
            if "\n" in encoded or "\r" in encoded:
                raise ValueError(f"GitHub output {name!r} contains a newline")
            output.write(f"{name}={encoded}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    retry = subparsers.add_parser(
        "authorize-retry",
        help="authorize an exact historical merged-main source-freeze retry",
    )
    retry.add_argument("--repository", type=Path, required=True)
    retry.add_argument("--request", type=Path, required=True)
    retry.add_argument("--current-revision", required=True)
    retry.add_argument("--execution-revision", required=True)
    retry.add_argument("--expected-request-id", required=True)
    retry.add_argument("--github-output", type=Path)

    variables = subparsers.add_parser(
        "check-variables",
        help="check required repository-variable names without publishing values",
    )
    variables.add_argument("--required-variable", action="append", default=[])
    variables.add_argument("--report", type=Path)
    variables.add_argument("--github-output", type=Path)
    variables.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "authorize-retry":
        result = authorize_retry(
            repository=args.repository,
            request_path=args.request,
            current_revision=args.current_revision,
            execution_revision=args.execution_revision,
            expected_request_id=args.expected_request_id,
        )
        if args.github_output is not None:
            _write_github_output(args.github_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    report = variable_readiness(
        args.required_variable,
        environment=os.environ,
    )
    if args.report is not None:
        _write_json(args.report, report)
    if args.github_output is not None:
        _write_github_output(
            args.github_output,
            {
                "ready": report["ready"],
                "configured_variable_count": report["configured_variable_count"],
                "missing_variables_json": report["missing_variables"],
            },
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready and report["ready"] is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
