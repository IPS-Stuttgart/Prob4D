#!/usr/bin/env python3
"""Build a bounded metadata-only inventory of a staged Deform360 source bundle.

The audit never opens a dataset file. It traverses directory entries with
``os.scandir`` and records only ``lstat``-derived metadata. Symlinks are not
followed, target-like path components are skipped before stat or descent, and
all outputs are written outside the dataset root.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "prob4d.deform360-source-bundle-audit-protocol"
REQUEST_SCHEMA = "prob4d.deform360-source-bundle-audit-request"
RESULT_SCHEMA = "prob4d.deform360-source-bundle-audit-result"
PROFILE = "deform360-source-bundle-audit-v1"
ISSUE_NUMBER = 49
EXPECTED_SOURCE_ROOT = Path(
    "/home/florianpfaff/deform360-fresh-source-processed-v1-1a3f9b1"
)
EXPECTED_RUNNER_LABEL = "gpuserver4090"
EXPECTED_PROTOCOL_PATH = "protocols/deform360-source-bundle-audit-v1.json"
EXPECTED_FORBIDDEN_TOKENS = (
    "confirmation",
    "fresh-validation",
    "held-v8",
    "shadow",
    "target",
)
CLAIM_BOUNDARY = (
    "Metadata-only source-bundle inventory. No dataset file content, provider "
    "prediction, residual, target payload, target outcome, BayesianPhysTwin "
    "innovation, or Causal4D outcome is authorized."
)


class AuditContractError(ValueError):
    """Raised when a protocol or request violates the frozen contract."""


def _canonical_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _content_id(record: dict[str, Any], identity_field: str) -> str:
    identity = dict(record)
    identity.pop(identity_field, None)
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditContractError(f"{path} must contain a JSON object")
    return value


def _require_bool(record: dict[str, Any], key: str, expected: bool) -> None:
    value = record.get(key)
    if type(value) is not bool or value is not expected:
        raise AuditContractError(f"{key} must be {str(expected).lower()}")


def _require_int(
    record: dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditContractError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise AuditContractError(f"{key} must lie in [{minimum}, {maximum}]")
    return value


def _require_sha(value: object, *, name: str, length: int) -> str:
    text = str(value)
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise AuditContractError(f"{name} must be a lowercase {length}-hex digest")
    return text


def validate_protocol_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and return one frozen audit protocol."""

    if record.get("schema") != PROTOCOL_SCHEMA:
        raise AuditContractError("unexpected protocol schema")
    if record.get("schema_version") != 1:
        raise AuditContractError("protocol schema_version must be 1")
    if record.get("profile") != PROFILE:
        raise AuditContractError("unexpected protocol profile")
    if record.get("issue_number") != ISSUE_NUMBER:
        raise AuditContractError(f"issue_number must be {ISSUE_NUMBER}")
    if record.get("source_root") != str(EXPECTED_SOURCE_ROOT):
        raise AuditContractError("source_root differs from the reviewed source bundle")
    if record.get("runner_label") != EXPECTED_RUNNER_LABEL:
        raise AuditContractError("runner_label differs from the reviewed runner")
    if record.get("forbidden_path_tokens") != list(EXPECTED_FORBIDDEN_TOKENS):
        raise AuditContractError("forbidden_path_tokens differ from the reviewed set")

    _require_bool(record, "metadata_access_authorized", True)
    for key in (
        "file_content_reads_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_payloads_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
    ):
        _require_bool(record, key, False)

    limits = record.get("limits")
    if not isinstance(limits, dict):
        raise AuditContractError("limits must be a JSON object")
    _require_int(limits, "max_entries", minimum=1, maximum=5_000_000)
    _require_int(limits, "max_depth", minimum=1, maximum=128)
    _require_int(limits, "largest_file_limit", minimum=0, maximum=500)
    _require_int(limits, "sample_path_limit", minimum=0, maximum=1_000)

    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        raise AuditContractError("protocol claim_boundary differs from reviewed wording")
    actual_id = _require_sha(record.get("protocol_id"), name="protocol_id", length=64)
    if actual_id != _content_id(record, "protocol_id"):
        raise AuditContractError("protocol_id mismatch")
    return record


def validate_request_record(
    record: dict[str, Any],
    *,
    protocol: dict[str, Any],
    source_protocol_git_blob_sha: str,
) -> dict[str, Any]:
    """Validate a one-shot execution request against the merged protocol blob."""

    validate_protocol_record(protocol)
    if record.get("schema") != REQUEST_SCHEMA:
        raise AuditContractError("unexpected request schema")
    if record.get("schema_version") != 1:
        raise AuditContractError("request schema_version must be 1")
    if record.get("profile") != PROFILE:
        raise AuditContractError("unexpected request profile")
    if record.get("issue_number") != ISSUE_NUMBER:
        raise AuditContractError(f"issue_number must be {ISSUE_NUMBER}")
    if record.get("source_protocol_path") != EXPECTED_PROTOCOL_PATH:
        raise AuditContractError("request source_protocol_path mismatch")

    request_blob = _require_sha(
        record.get("source_protocol_git_blob_sha"),
        name="source_protocol_git_blob_sha",
        length=40,
    )
    merged_blob = _require_sha(
        source_protocol_git_blob_sha,
        name="expected source protocol Git blob SHA",
        length=40,
    )
    if request_blob != merged_blob:
        raise AuditContractError("request does not bind the merged protocol blob")
    if record.get("protocol_id") != protocol["protocol_id"]:
        raise AuditContractError("request protocol_id mismatch")

    _require_bool(record, "execution_authorized", True)
    _require_bool(record, "metadata_access_authorized", True)
    for key in (
        "file_content_reads_authorized",
        "prediction_execution_authorized",
        "provider_residuals_authorized",
        "target_payloads_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
    ):
        _require_bool(record, key, False)

    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        raise AuditContractError("request claim_boundary differs from reviewed wording")
    actual_id = _require_sha(record.get("request_id"), name="request_id", length=64)
    if actual_id != _content_id(record, "request_id"):
        raise AuditContractError("request_id mismatch")
    return record


def _forbidden_token(name: str, tokens: tuple[str, ...]) -> str | None:
    normalized = name.casefold().replace("_", "-").replace(" ", "-")
    return next((token for token in tokens if token in normalized), None)


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _manifest_line(relative_path: str, kind: str, mode: int, size: int) -> bytes:
    return f"{kind}\0{stat.S_IMODE(mode):04o}\0{size}\0{relative_path}\n".encode()


def _root_failure(decision: str, message: str, errno: int | None = None):
    result: dict[str, Any] = {"error": message}
    if errno is not None:
        result["errno"] = errno
    return decision, result


def scan_source_root(
    source_root: Path,
    *,
    forbidden_tokens: tuple[str, ...],
    max_entries: int,
    max_depth: int,
    largest_file_limit: int,
    sample_path_limit: int,
) -> tuple[str, dict[str, Any]]:
    """Scan dataset metadata without following links or opening files."""

    try:
        root_stat = source_root.lstat()
    except FileNotFoundError:
        return _root_failure("source-root-missing", "source root does not exist")
    except OSError as exc:
        return _root_failure(
            "source-root-unreadable",
            "source root metadata could not be read",
            exc.errno,
        )
    if stat.S_ISLNK(root_stat.st_mode):
        return _root_failure("source-root-symlink-rejected", "source root is a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        return _root_failure("source-root-not-directory", "source root is not a directory")

    counts: Counter[str] = Counter({"directory": 1})
    extensions: Counter[str] = Counter()
    depths: Counter[int] = Counter({0: 1})
    forbidden: Counter[str] = Counter()
    top_level: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    error_samples: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    largest: list[tuple[int, str]] = []
    manifest = hashlib.sha256()
    manifest.update(_manifest_line(".", "directory", root_stat.st_mode, root_stat.st_size))

    entry_count = 0
    total_file_bytes = 0
    maximum_depth_seen = 0
    limit_exceeded = False
    stack: list[tuple[Path, str, int]] = [(source_root, "", 0)]

    while stack and not limit_exceeded:
        directory, relative_directory, directory_depth = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            errors["directory-scan"] += 1
            if len(error_samples) < 50:
                error_samples.append(
                    {
                        "operation": "directory-scan",
                        "path": relative_directory or ".",
                        "errno": exc.errno,
                    }
                )
            continue

        child_directories: list[tuple[Path, str, int]] = []
        for entry in entries:
            forbidden_token = _forbidden_token(entry.name, forbidden_tokens)
            if forbidden_token is not None:
                forbidden[forbidden_token] += 1
                counts["forbidden-subtree-skipped"] += 1
                continue
            if entry_count >= max_entries:
                limit_exceeded = True
                break

            relative_path = (
                entry.name
                if not relative_directory
                else f"{relative_directory}/{entry.name}"
            )
            depth = directory_depth + 1
            if depth > max_depth:
                counts["depth-limit-skipped"] += 1
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                errors["lstat"] += 1
                if len(error_samples) < 50:
                    error_samples.append(
                        {"operation": "lstat", "path": relative_path, "errno": exc.errno}
                    )
                continue

            kind = _entry_type(metadata.st_mode)
            size = int(metadata.st_size)
            entry_count += 1
            counts[kind] += 1
            depths[depth] += 1
            maximum_depth_seen = max(maximum_depth_seen, depth)
            manifest.update(_manifest_line(relative_path, kind, metadata.st_mode, size))

            top_name = relative_path.split("/", 1)[0]
            top_level[top_name]["entries"] += 1
            top_level[top_name][kind] += 1
            if kind == "file":
                total_file_bytes += size
                top_level[top_name]["regular_file_bytes"] += size
                extensions[Path(entry.name).suffix.casefold() or "<none>"] += 1
                item = (size, relative_path)
                if largest_file_limit > 0:
                    if len(largest) < largest_file_limit:
                        heapq.heappush(largest, item)
                    elif item > largest[0]:
                        heapq.heapreplace(largest, item)

            if len(samples) < sample_path_limit:
                samples.append({"path": relative_path, "type": kind, "size_bytes": size})
            if kind == "directory":
                child_directories.append((Path(entry.path), relative_path, depth))

        stack.extend(reversed(child_directories))

    if limit_exceeded:
        decision = "entry-limit-exceeded"
    elif errors:
        decision = "source-bundle-partial"
    else:
        decision = "source-bundle-present"

    inventory = {
        "metadata_manifest_sha256": manifest.hexdigest(),
        "metadata_manifest_semantics": (
            "relative path, lstat type, POSIX mode, and size only"
        ),
        "entry_count_excluding_root": entry_count,
        "counts_including_root": dict(sorted(counts.items())),
        "total_regular_file_bytes": total_file_bytes,
        "maximum_depth_seen": maximum_depth_seen,
        "depth_counts_including_root": {
            str(depth): count for depth, count in sorted(depths.items())
        },
        "extension_counts": dict(sorted(extensions.items())),
        "top_level": {
            name: dict(sorted(summary.items()))
            for name, summary in sorted(top_level.items())
        },
        "largest_regular_files": [
            {"path": path, "size_bytes": size}
            for size, path in sorted(largest, key=lambda item: (-item[0], item[1]))
        ],
        "deterministic_path_sample": samples,
        "forbidden_path_components_skipped": sum(forbidden.values()),
        "forbidden_token_counts": dict(sorted(forbidden.items())),
        "metadata_error_counts": dict(sorted(errors.items())),
        "metadata_error_samples": error_samples,
        "entry_limit_exceeded": limit_exceeded,
    }
    return decision, inventory


def build_audit_result(
    *,
    protocol: dict[str, Any],
    request: dict[str, Any],
    source_protocol_git_blob_sha: str,
    prob4d_revision: str,
    runner_name: str,
    github_run_id: str,
) -> dict[str, Any]:
    """Validate contracts, scan the source root, and bind the result."""

    validate_request_record(
        request,
        protocol=protocol,
        source_protocol_git_blob_sha=source_protocol_git_blob_sha,
    )
    revision = _require_sha(prob4d_revision, name="prob4d_revision", length=40)
    limits = protocol["limits"]
    decision, inventory = scan_source_root(
        Path(protocol["source_root"]),
        forbidden_tokens=tuple(protocol["forbidden_path_tokens"]),
        max_entries=limits["max_entries"],
        max_depth=limits["max_depth"],
        largest_file_limit=limits["largest_file_limit"],
        sample_path_limit=limits["sample_path_limit"],
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "profile": PROFILE,
        "decision": decision,
        "issue_number": ISSUE_NUMBER,
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "source_protocol_git_blob_sha": source_protocol_git_blob_sha,
        "prob4d_revision": revision,
        "runner_name": runner_name,
        "github_run_id": str(github_run_id),
        "source_root": protocol["source_root"],
        "execution_boundary": {
            "metadata_accessed": True,
            "dataset_file_contents_opened": False,
            "symlinks_followed": False,
            "prediction_executed": False,
            "provider_residuals_opened": False,
            "target_payloads_opened": False,
            "target_outcomes_opened": False,
            "dataset_mutated": False,
            "outputs_written_outside_source_root": True,
        },
        "inventory": inventory,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["audit_id"] = _content_id(result, "audit_id")
    return result


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    inventory = result.get("inventory") or {}
    counts = inventory.get("counts_including_root") or {}
    lines = [
        "# Deform360 source-bundle metadata audit",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Audit ID: `{result['audit_id']}`",
        f"- Request ID: `{result['request_id']}`",
        f"- Source root: `{result['source_root']}`",
        "- Metadata manifest SHA-256: "
        f"`{inventory.get('metadata_manifest_sha256', 'unavailable')}`",
        f"- Regular files: `{counts.get('file', 0)}`",
        f"- Directories including root: `{counts.get('directory', 0)}`",
        f"- Symlinks: `{counts.get('symlink', 0)}`",
        f"- Regular-file bytes: `{inventory.get('total_regular_file_bytes', 0)}`",
        "- Forbidden path components skipped: "
        f"`{inventory.get('forbidden_path_components_skipped', 0)}`",
        "",
        "This audit read directory entries and lstat metadata only. It did not "
        "open dataset file contents, follow symlinks, execute predictions, inspect "
        "residuals or targets, or modify the source root.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _command_validate_protocol(args: argparse.Namespace) -> int:
    protocol = validate_protocol_record(_load_json(Path(args.protocol)))
    print(json.dumps({"protocol_id": protocol["protocol_id"]}, sort_keys=True))
    return 0


def _command_validate_request(args: argparse.Namespace) -> int:
    protocol = _load_json(Path(args.protocol))
    request = validate_request_record(
        _load_json(Path(args.request)),
        protocol=protocol,
        source_protocol_git_blob_sha=args.source_protocol_git_blob_sha,
    )
    print(json.dumps({"request_id": request["request_id"]}, sort_keys=True))
    return 0


def _command_inventory(args: argparse.Namespace) -> int:
    protocol = _load_json(Path(args.protocol))
    output = Path(args.output).resolve(strict=False)
    summary = Path(args.summary).resolve(strict=False)
    source_root = Path(protocol.get("source_root", "")).resolve(strict=False)
    if output == summary:
        raise AuditContractError("output and summary paths must differ")
    if output.is_relative_to(source_root) or summary.is_relative_to(source_root):
        raise AuditContractError("audit outputs must remain outside the source root")

    result = build_audit_result(
        protocol=protocol,
        request=_load_json(Path(args.request)),
        source_protocol_git_blob_sha=args.source_protocol_git_blob_sha,
        prob4d_revision=args.prob4d_revision,
        runner_name=args.runner_name,
        github_run_id=args.github_run_id,
    )
    _write_result(output, result)
    _write_summary(summary, result)
    print(json.dumps({"audit_id": result["audit_id"], "decision": result["decision"]}))
    return 0 if result["decision"] == "source-bundle-present" else 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_protocol = subparsers.add_parser("validate-protocol")
    validate_protocol.add_argument("--protocol", required=True)
    validate_protocol.set_defaults(function=_command_validate_protocol)

    validate_request = subparsers.add_parser("validate-request")
    validate_request.add_argument("--request", required=True)
    validate_request.add_argument("--protocol", required=True)
    validate_request.add_argument("--source-protocol-git-blob-sha", required=True)
    validate_request.set_defaults(function=_command_validate_request)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--request", required=True)
    inventory.add_argument("--protocol", required=True)
    inventory.add_argument("--source-protocol-git-blob-sha", required=True)
    inventory.add_argument("--prob4d-revision", required=True)
    inventory.add_argument("--runner-name", required=True)
    inventory.add_argument("--github-run-id", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--summary", required=True)
    inventory.set_defaults(function=_command_inventory)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except AuditContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
