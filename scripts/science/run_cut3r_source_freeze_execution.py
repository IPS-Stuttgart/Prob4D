#!/usr/bin/env python3
"""Drive one target-closed retained CUT3R source-freeze execution.

The driver validates a content-addressed merged-main request, invokes the existing
outcome-blind source-freeze builder, creates the immutable comparison lock only
after a support-positive decision, sanitizes retained logs, and emits one compact
execution summary. It never runs CUT3R or opens residual, truth, confirmation, or
target payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

REQUEST_SCHEMA: Final = "prob4d.cut3r-deform360-source-freeze-execution-request"
REQUEST_VERSION: Final = 2
SUMMARY_SCHEMA: Final = "prob4d.cut3r-source-freeze-execution-summary"
SUMMARY_VERSION: Final = 2
PROFILE: Final = "cut3r-deform360-source-freeze"
AUTHORIZATION_MODE: Final = "merged-main-read-only-self-hosted-v1"
SUPPORT_PASS: Final = "source-support-freeze-ready"
SUPPORT_NEGATIVE: Final = "insufficient-common-camera-support"
TECHNICAL_FAILURE: Final = "technical-failure"
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
FALSE_REQUEST_FIELDS: Final = (
    "repository_write_token_on_self_hosted",
    "environment_approval_required",
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "comparison_execution_authorized",
)
FALSE_FREEZE_BOUNDARY_FIELDS: Final = (
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "source_future_geometry_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "downstream_physical_innovations_opened",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {name} from {path}: {error}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(dict[str, Any], value)


def _lower_hex(value: object, *, name: str, length: int) -> str:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{name} must be an exact lowercase {length}-character digest")
    return value


def _source_freeze_id(freeze: Mapping[str, Any]) -> str:
    source_freeze_id = _lower_hex(
        freeze.get("source_freeze_id"),
        name="source-freeze source_freeze_id",
        length=64,
    )
    identity = dict(freeze)
    identity.pop("source_freeze_id")
    if _sha256_json(identity) != source_freeze_id:
        raise ValueError("source-freeze source_freeze_id mismatch")
    return source_freeze_id


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _git_output(arguments: Sequence[str], *, cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git {' '.join(arguments)} failed: {error}") from error


def validate_request(
    request_path: Path,
    *,
    repository: Path,
    expected_revision: str | None = None,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Validate one immutable v2 request and its bound protocol bytes."""

    request = _load_json_object(
        request_path,
        name="source-freeze execution request",
    )
    if set(request) != EXPECTED_REQUEST_FIELDS:
        missing = sorted(EXPECTED_REQUEST_FIELDS - set(request))
        extra = sorted(set(request) - EXPECTED_REQUEST_FIELDS)
        raise ValueError(f"execution request fields changed; missing={missing}, extra={extra}")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("unexpected source-freeze execution-request schema")
    if request["schema_version"] != REQUEST_VERSION:
        raise ValueError("unsupported source-freeze execution-request version")
    if request["issue_number"] != 49:
        raise ValueError("source-freeze execution request is not bound to issue 49")
    if request["profile"] != PROFILE:
        raise ValueError("source-freeze execution request names the wrong profile")
    if request["authorization_mode"] != AUTHORIZATION_MODE:
        raise ValueError("source-freeze authorization mode changed")
    _lower_hex(
        request["supersedes_request_id"],
        name="supersedes_request_id",
        length=64,
    )
    if request["source_group_count"] != 10:
        raise ValueError("source-freeze source group count changed")
    if request["forbidden_target_group_count"] != 12:
        raise ValueError("source-freeze forbidden target group count changed")
    if any(request[name] is not False for name in FALSE_REQUEST_FIELDS):
        raise ValueError("source-freeze execution request exceeds its target-closed boundary")

    request_id = _lower_hex(request["request_id"], name="request_id", length=64)
    identity = dict(request)
    identity.pop("request_id")
    if _sha256_json(identity) != request_id:
        raise ValueError("source-freeze execution request ID mismatch")

    protocol_relative = _literal_string(
        request["source_protocol_path"],
        name="source_protocol_path",
    )
    if protocol_relative != "protocols/cut3r_deform360_source_v1.json":
        raise ValueError("source-freeze protocol path changed")
    protocol_path = repository / protocol_relative
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise ValueError("source-freeze protocol must be an ordinary repository file")
    measured_blob = _git_output(["hash-object", protocol_relative], cwd=repository)
    expected_blob = _lower_hex(
        request["source_protocol_git_blob_sha"],
        name="source_protocol_git_blob_sha",
        length=40,
    )
    if measured_blob != expected_blob:
        raise ValueError("source-freeze protocol Git blob differs from the request")

    head = _lower_hex(
        _git_output(["rev-parse", "HEAD"], cwd=repository),
        name="repository revision",
        length=40,
    )
    if expected_revision is not None and head != _lower_hex(
        expected_revision,
        name="expected_revision",
        length=40,
    ):
        raise ValueError("checked-out repository revision differs from authorization")
    if require_clean and _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
    ):
        raise ValueError("source-freeze execution checkout must be clean")
    return request


def _publish_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to replace different retained bytes: {path}")
    path.write_bytes(encoded)


def _sanitize_log(text: str, replacements: Mapping[str, str]) -> str:
    sanitized = text
    for value in sorted(
        (item for item in replacements if item),
        key=len,
        reverse=True,
    ):
        sanitized = sanitized.replace(value, replacements[value])
    return sanitized


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    log: list[str],
) -> int:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    log.append(f"$ {' '.join(arguments)}\n")
    log.append(completed.stdout)
    log.append(completed.stderr)
    return int(completed.returncode)


def _prob4d_executable() -> Path:
    executable = Path(sys.executable).with_name("prob4d")
    if not executable.is_file():
        located = shutil.which("prob4d")
        if located is None:
            raise ValueError("installed Prob4D command is unavailable")
        executable = Path(located)
    return executable


def _validate_freeze(
    freeze: Mapping[str, Any],
    *,
    builder_status: int,
    output_directory: Path,
) -> str:
    if freeze.get("source_group_count") != 10:
        raise ValueError("source freeze does not contain exactly ten source groups")
    if freeze.get("forbidden_target_group_count") != 12:
        raise ValueError("source freeze does not retain all twelve forbidden targets")
    boundary = freeze.get("information_boundary")
    if type(boundary) is not dict:
        raise ValueError("source freeze has no exact information boundary")
    boundary_mapping = cast(dict[str, Any], boundary)
    if any(boundary_mapping.get(name) is not False for name in FALSE_FREEZE_BOUNDARY_FIELDS):
        raise ValueError("source-freeze information boundary was exceeded")

    specification = output_directory / "cut3r-comparison-spec.json"
    lock = output_directory / "cut3r-comparison-lock.json"
    if builder_status == 0:
        if freeze.get("decision") != SUPPORT_PASS:
            raise ValueError("successful source freeze has the wrong decision")
        if not specification.is_file() or not lock.is_file():
            raise ValueError("successful source freeze lacks canonical comparison outputs")
        return SUPPORT_PASS
    if builder_status == 3:
        if freeze.get("decision") != SUPPORT_NEGATIVE:
            raise ValueError("support-negative source freeze has the wrong decision")
        if specification.exists() or lock.exists():
            raise ValueError("support-negative source freeze published comparison outputs")
        return SUPPORT_NEGATIVE
    raise ValueError("unexpected source-freeze builder status")


def execute(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    output_directory = args.output_dir.resolve()
    request = validate_request(
        args.request.resolve(),
        repository=repository,
        expected_revision=args.expected_revision,
        require_clean=False,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    raw_log: list[str] = []

    builder = repository / "scripts/science/build_cut3r_deform360_source_freeze.py"
    builder_status = _run(
        (
            sys.executable,
            os.fspath(builder),
            "--repository",
            os.fspath(repository),
            "--protocol",
            os.fspath(repository / cast(str, request["source_protocol_path"])),
            "--selection",
            os.fspath(args.selection.resolve()),
            "--processed-root",
            os.fspath(args.processed_root.resolve()),
            "--cut3r-checkout",
            os.fspath(args.cut3r_checkout.resolve()),
            "--checkpoint",
            os.fspath(args.checkpoint.resolve()),
            "--prob4d-wheel",
            os.fspath(args.prob4d_wheel.resolve()),
            "--output-dir",
            os.fspath(output_directory),
        ),
        cwd=repository,
        log=raw_log,
    )
    (output_directory / "builder-exit-status.txt").write_text(
        f"{builder_status}\n",
        encoding="utf-8",
    )

    if builder_status == 0:
        prob4d = _prob4d_executable()
        commands = (
            (
                os.fspath(prob4d),
                "prediction",
                "cut3r-comparison",
                "build",
                os.fspath(output_directory / "cut3r-comparison-spec.json"),
                "--output",
                os.fspath(output_directory / "cut3r-comparison-lock.json"),
            ),
            (
                os.fspath(prob4d),
                "prediction",
                "cut3r-comparison",
                "verify",
                os.fspath(output_directory / "cut3r-comparison-lock.json"),
            ),
            (
                os.fspath(prob4d),
                "prediction",
                "cut3r-comparison",
                "summarize",
                os.fspath(output_directory / "cut3r-comparison-lock.json"),
                "--json",
            ),
        )
        for index, command in enumerate(commands):
            status = _run(command, cwd=repository, log=raw_log)
            if status != 0:
                builder_status = 100 + index
                break
        if builder_status == 0:
            summary_command = commands[-1]
            completed = subprocess.run(
                list(summary_command),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            (output_directory / "cut3r-comparison-summary.json").write_text(
                completed.stdout,
                encoding="utf-8",
            )

    replacements = {
        os.fspath(args.selection.resolve()): "<BPT_SELECTION_LOCK>",
        os.fspath(args.processed_root.resolve()): "<DEFORM360_PROCESSED_ROOT>",
        os.fspath(args.cut3r_checkout.resolve()): "<CUT3R_CHECKOUT>",
        os.fspath(args.checkpoint.resolve()): "<CUT3R_CHECKPOINT>",
        os.fspath(args.prob4d_wheel.resolve()): "<PROB4D_WHEEL>",
    }
    sanitized_log = _sanitize_log("".join(raw_log), replacements)
    (output_directory / "freeze.log").write_text(
        sanitized_log,
        encoding="utf-8",
    )

    freeze_path = output_directory / "cut3r-deform360-source-freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError(
            "source-freeze builder did not publish a retained decision; "
            f"builder_status={builder_status}"
        )
    freeze = _load_json_object(freeze_path, name="source-freeze artifact")
    if builder_status not in {0, 3}:
        raise RuntimeError(
            "source-freeze execution failed outside the registered decision "
            f"statuses; builder_status={builder_status}"
        )
    decision = _validate_freeze(
        freeze,
        builder_status=builder_status,
        output_directory=output_directory,
    )

    repository_revision = _lower_hex(
        _git_output(["rev-parse", "HEAD"], cwd=repository),
        name="repository revision",
        length=40,
    )
    request_id = cast(str, request["request_id"])
    artifact_name = (
        f"cut3r-source-freeze-v2-{request_id[:12]}-"
        f"{args.workflow_run_id}-{args.workflow_run_attempt}"
    )
    freeze_artifact_id = _source_freeze_id(freeze)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_VERSION,
        "request_id": request_id,
        "supersedes_request_id": request["supersedes_request_id"],
        "authorization_mode": request["authorization_mode"],
        "repository_revision": repository_revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "builder_exit_status": builder_status,
        "decision": decision,
        "freeze_artifact_id": freeze_artifact_id,
        "artifact_name": artifact_name,
        "source_group_count": freeze["source_group_count"],
        "forbidden_target_group_count": freeze["forbidden_target_group_count"],
        "source_rgb_frames_decoded": False,
        "source_prediction_payloads_opened": False,
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "comparison_execution_authorized": False,
    }
    _publish_json(output_directory / "execution-summary.json", summary)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as stream:
            stream.write(f"artifact_name={artifact_name}\n")
            stream.write(f"decision={decision}\n")
            stream.write(f"exit_status={builder_status}\n")
            stream.write(f"freeze_artifact_id={freeze_artifact_id}\n")
            stream.write(f"request_id={request_id}\n")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-request",
        help="validate one merged-main source-freeze request",
    )
    verify.add_argument("--repository", type=Path, required=True)
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--expected-revision")
    verify.add_argument("--github-output", type=Path)

    run = subparsers.add_parser(
        "execute",
        help="run the target-closed source freeze and retain its decision",
    )
    run.add_argument("--repository", type=Path, required=True)
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--expected-revision", required=True)
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--processed-root", type=Path, required=True)
    run.add_argument("--cut3r-checkout", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--prob4d-wheel", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--workflow-run-id", type=int, required=True)
    run.add_argument("--workflow-run-attempt", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-request":
        request = validate_request(
            args.request.resolve(),
            repository=args.repository.resolve(),
            expected_revision=args.expected_revision,
        )
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(f"request_id={request['request_id']}\n")
                stream.write(f"profile={request['profile']}\n")
                stream.write(f"authorization_mode={request['authorization_mode']}\n")
        print(
            json.dumps(
                {
                    "request_id": request["request_id"],
                    "profile": request["profile"],
                    "authorization_mode": request["authorization_mode"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "execute":
        return execute(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
