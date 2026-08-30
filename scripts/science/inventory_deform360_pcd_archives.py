#!/usr/bin/env python3
"""Inventory official Deform360 persistent-point archives without reading arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_SCHEMA = "prob4d.deform360-pcd-query-inventory-protocol"
REQUEST_SCHEMA = "prob4d.deform360-pcd-query-inventory-request"
RESULT_SCHEMA = "prob4d.deform360-pcd-query-inventory-result"
PROFILE = "deform360-pcd-query-inventory-v1"
ISSUE_NUMBER = 49
EXPECTED_PROTOCOL_PATH = "protocols/deform360-pcd-query-inventory-v1.json"
EXPECTED_SOURCE_ROOT = (
    "/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository/processed"
)
EXPECTED_RUNNER_LABEL = "gpuserver4090"
REQUIRED_ARCHIVE_NAME = "pcd_clean.tar"
OBJECT_RE = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")
EPISODE_RE = re.compile(r"^episode_[0-9]+$")
FRAME_RE = re.compile(r"^[0-9]{6}\.npz$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLAIM_BOUNDARY = (
    "Tar-header-only inventory of official Deform360 pcd_clean archives. "
    "No archive member payload, NumPy array, provider prediction, physical-query score, "
    "target outcome, BayesianPhysTwin update, or Causal4D outcome is authorized."
)


class InventoryContractError(ValueError):
    """Raised when a protocol, request, or mounted-data boundary is invalid."""


def _canonical_bytes(record: dict[str, Any], identity_field: str | None = None) -> bytes:
    identity = dict(record)
    if identity_field is not None:
        identity.pop(identity_field, None)
    return json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_id(record: dict[str, Any], identity_field: str) -> str:
    return hashlib.sha256(_canonical_bytes(record, identity_field)).hexdigest()


def _require_exact_keys(record: dict[str, Any], expected: set[str], label: str) -> None:
    keys = set(record)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise InventoryContractError(
            f"{label} keys differ from contract; missing={missing}, unknown={unknown}"
        )


def _require_bool(record: dict[str, Any], key: str, expected: bool) -> None:
    value = record.get(key)
    if type(value) is not bool or value is not expected:
        raise InventoryContractError(f"{key} must be exactly {expected}")


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise InventoryContractError(f"{label} must be a positive integer")
    return value


def validate_protocol_record(record: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "schema_version",
        "profile",
        "issue_number",
        "source_root",
        "runner_label",
        "required_archive_name",
        "excluded_object_ids",
        "minimum_frame_count",
        "minimum_eligible_episodes_per_object",
        "minimum_eligible_object_count",
        "limits",
        "metadata_access_authorized",
        "archive_header_reads_authorized",
        "archive_member_payload_reads_authorized",
        "numeric_array_reads_authorized",
        "provider_predictions_authorized",
        "physical_query_scoring_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
        "claim_boundary",
        "protocol_id",
    }
    _require_exact_keys(record, expected_keys, "protocol")
    if record["schema"] != PROTOCOL_SCHEMA or record["schema_version"] != 1:
        raise InventoryContractError("unsupported protocol schema")
    if record["profile"] != PROFILE or record["issue_number"] != ISSUE_NUMBER:
        raise InventoryContractError("protocol profile or issue number mismatch")
    if record["source_root"] != EXPECTED_SOURCE_ROOT:
        raise InventoryContractError("source_root differs from the reviewed mount")
    if record["runner_label"] != EXPECTED_RUNNER_LABEL:
        raise InventoryContractError("runner_label differs from gpuserver4090")
    if record["required_archive_name"] != REQUIRED_ARCHIVE_NAME:
        raise InventoryContractError("required archive name differs from pcd_clean.tar")
    excluded = record["excluded_object_ids"]
    if not isinstance(excluded, list) or excluded != sorted(set(excluded)):
        raise InventoryContractError("excluded_object_ids must be a sorted unique list")
    if any(not isinstance(item, str) or OBJECT_RE.fullmatch(item) is None for item in excluded):
        raise InventoryContractError("excluded_object_ids contain malformed object identities")
    _require_positive_int(record["minimum_frame_count"], "minimum_frame_count")
    _require_positive_int(
        record["minimum_eligible_episodes_per_object"],
        "minimum_eligible_episodes_per_object",
    )
    _require_positive_int(record["minimum_eligible_object_count"], "minimum_eligible_object_count")
    limits = record["limits"]
    if not isinstance(limits, dict):
        raise InventoryContractError("limits must be an object")
    _require_exact_keys(
        limits,
        {"max_objects", "max_episodes_per_object", "max_tar_members"},
        "limits",
    )
    for key, value in limits.items():
        _require_positive_int(value, f"limits.{key}")
    _require_bool(record, "metadata_access_authorized", True)
    _require_bool(record, "archive_header_reads_authorized", True)
    for key in (
        "archive_member_payload_reads_authorized",
        "numeric_array_reads_authorized",
        "provider_predictions_authorized",
        "physical_query_scoring_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
    ):
        _require_bool(record, key, False)
    if record["claim_boundary"] != CLAIM_BOUNDARY:
        raise InventoryContractError("claim boundary differs from the reviewed wording")
    protocol_id = record["protocol_id"]
    if not isinstance(protocol_id, str) or SHA256_RE.fullmatch(protocol_id) is None:
        raise InventoryContractError("protocol_id must be lowercase SHA-256")
    if protocol_id != canonical_id(record, "protocol_id"):
        raise InventoryContractError("protocol_id does not match canonical content")
    return record


def validate_request_record(
    record: dict[str, Any],
    *,
    protocol: dict[str, Any],
    source_protocol_git_blob_sha: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "schema_version",
        "profile",
        "issue_number",
        "source_protocol_path",
        "source_protocol_git_blob_sha",
        "protocol_id",
        "execution_authorized",
        "metadata_access_authorized",
        "archive_header_reads_authorized",
        "archive_member_payload_reads_authorized",
        "numeric_array_reads_authorized",
        "provider_predictions_authorized",
        "physical_query_scoring_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
        "claim_boundary",
        "request_id",
    }
    _require_exact_keys(record, expected_keys, "request")
    validate_protocol_record(protocol)
    if record["schema"] != REQUEST_SCHEMA or record["schema_version"] != 1:
        raise InventoryContractError("unsupported request schema")
    if record["profile"] != PROFILE or record["issue_number"] != ISSUE_NUMBER:
        raise InventoryContractError("request profile or issue number mismatch")
    if record["source_protocol_path"] != EXPECTED_PROTOCOL_PATH:
        raise InventoryContractError("request source protocol path mismatch")
    if REVISION_RE.fullmatch(source_protocol_git_blob_sha) is None:
        raise InventoryContractError("merged protocol blob must be a lowercase Git blob SHA")
    if record["source_protocol_git_blob_sha"] != source_protocol_git_blob_sha:
        raise InventoryContractError("request does not bind the merged protocol blob")
    if record["protocol_id"] != protocol["protocol_id"]:
        raise InventoryContractError("request protocol_id mismatch")
    _require_bool(record, "execution_authorized", True)
    _require_bool(record, "metadata_access_authorized", True)
    _require_bool(record, "archive_header_reads_authorized", True)
    for key in (
        "archive_member_payload_reads_authorized",
        "numeric_array_reads_authorized",
        "provider_predictions_authorized",
        "physical_query_scoring_authorized",
        "target_outcomes_authorized",
        "dataset_mutation_authorized",
    ):
        _require_bool(record, key, False)
    if record["claim_boundary"] != CLAIM_BOUNDARY:
        raise InventoryContractError("request claim boundary mismatch")
    request_id = record["request_id"]
    if not isinstance(request_id, str) or SHA256_RE.fullmatch(request_id) is None:
        raise InventoryContractError("request_id must be lowercase SHA-256")
    if request_id != canonical_id(record, "request_id"):
        raise InventoryContractError("request_id does not match canonical content")
    return record


def _safe_tar_member(member: tarfile.TarInfo) -> tuple[bool, str]:
    name = member.name
    if not name or "\\" in name or name.startswith("/"):
        return False, "unsafe-member-path"
    parts = PurePosixPath(name).parts
    if any(part in {"", ".", ".."} for part in parts):
        return False, "unsafe-member-path"
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        return False, "unsupported-member-type"
    if not (member.isfile() or member.isdir()):
        return False, "unsupported-member-type"
    return True, "ok"


def inspect_archive_headers(
    archive: Path,
    *,
    max_tar_members: int,
    minimum_frame_count: int,
) -> dict[str, Any]:
    before = archive.lstat()
    if not stat.S_ISREG(before.st_mode):
        return {"eligible": False, "reason": "archive-not-regular-file"}
    frame_indices: list[int] = []
    frame_payload_bytes = 0
    member_count = 0
    unsafe_reason: str | None = None
    try:
        with tarfile.open(archive, mode="r:") as bundle:
            for member in bundle:
                member_count += 1
                if member_count > max_tar_members:
                    unsafe_reason = "tar-member-limit-exceeded"
                    break
                safe, reason = _safe_tar_member(member)
                if not safe:
                    unsafe_reason = reason
                    break
                path = PurePosixPath(member.name)
                if (
                    member.isfile()
                    and len(path.parts) >= 2
                    and path.parts[-2] == "pcd_clean"
                    and FRAME_RE.fullmatch(path.name) is not None
                ):
                    frame_indices.append(int(path.stem))
                    frame_payload_bytes += int(member.size)
    except (OSError, tarfile.TarError) as error:
        return {
            "eligible": False,
            "reason": "archive-header-read-failed",
            "error_type": type(error).__name__,
        }
    after = archive.lstat()
    stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    unique_indices = sorted(set(frame_indices))
    duplicates = len(frame_indices) - len(unique_indices)
    contiguous = bool(unique_indices) and unique_indices == list(
        range(unique_indices[0], unique_indices[-1] + 1)
    )
    starts_at_zero = bool(unique_indices) and unique_indices[0] == 0
    reason = unsafe_reason or "eligible"
    eligible = (
        unsafe_reason is None
        and stable
        and duplicates == 0
        and contiguous
        and starts_at_zero
        and len(unique_indices) >= minimum_frame_count
    )
    if not eligible and unsafe_reason is None:
        if not stable:
            reason = "archive-changed-during-header-scan"
        elif duplicates:
            reason = "duplicate-frame-members"
        elif not unique_indices:
            reason = "no-pcd-clean-frame-members"
        elif not starts_at_zero:
            reason = "frame-index-does-not-start-at-zero"
        elif not contiguous:
            reason = "frame-index-gap"
        else:
            reason = "insufficient-frame-count"
    return {
        "eligible": eligible,
        "reason": reason,
        "archive_size_bytes": int(before.st_size),
        "archive_mtime_ns": int(before.st_mtime_ns),
        "archive_stable_during_scan": stable,
        "tar_member_count": member_count,
        "frame_member_count": len(unique_indices),
        "frame_index_min": unique_indices[0] if unique_indices else None,
        "frame_index_max": unique_indices[-1] if unique_indices else None,
        "frame_indices_contiguous": contiguous,
        "duplicate_frame_member_count": duplicates,
        "frame_member_uncompressed_bytes": frame_payload_bytes,
    }


def inventory_source_root(root: Path, protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    expected = Path(EXPECTED_SOURCE_ROOT)
    if root != expected:
        raise InventoryContractError("runtime source root differs from reviewed source_root")
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return "source-root-missing", {"error": "reviewed source root does not exist"}
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise InventoryContractError("reviewed source root must be a physical directory")
    if root.resolve(strict=True) != expected:
        raise InventoryContractError("reviewed source root canonical path mismatch")

    limits = protocol["limits"]
    excluded = set(protocol["excluded_object_ids"])
    object_entries = sorted(root.iterdir(), key=lambda item: item.name)
    records: list[dict[str, Any]] = []
    ignored_top_level: Counter[str] = Counter()
    object_episode_counts: dict[str, int] = defaultdict(int)
    eligible_episode_counts: dict[str, int] = defaultdict(int)

    object_count = 0
    for object_path in object_entries:
        metadata = object_path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            ignored_top_level["symlink"] += 1
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            ignored_top_level["non-directory"] += 1
            continue
        if OBJECT_RE.fullmatch(object_path.name) is None:
            ignored_top_level["malformed-object-id"] += 1
            continue
        object_count += 1
        if object_count > limits["max_objects"]:
            raise InventoryContractError("object count exceeds reviewed limit")
        episode_paths = sorted(object_path.iterdir(), key=lambda item: item.name)
        episode_count = 0
        for episode_path in episode_paths:
            episode_metadata = episode_path.lstat()
            if stat.S_ISLNK(episode_metadata.st_mode) or not stat.S_ISDIR(episode_metadata.st_mode):
                continue
            if EPISODE_RE.fullmatch(episode_path.name) is None:
                continue
            episode_count += 1
            if episode_count > limits["max_episodes_per_object"]:
                raise InventoryContractError(
                    f"episode count exceeds reviewed limit for {object_path.name}"
                )
            archive = episode_path / REQUIRED_ARCHIVE_NAME
            object_episode_counts[object_path.name] += 1
            if not archive.exists():
                records.append(
                    {
                        "object_id": object_path.name,
                        "episode_id": episode_path.name,
                        "archive_relative_path": archive.relative_to(root).as_posix(),
                        "excluded_by_prior_evidence": object_path.name in excluded,
                        "eligible": False,
                        "reason": "archive-missing",
                    }
                )
                continue
            result = inspect_archive_headers(
                archive,
                max_tar_members=limits["max_tar_members"],
                minimum_frame_count=protocol["minimum_frame_count"],
            )
            record = {
                "object_id": object_path.name,
                "episode_id": episode_path.name,
                "archive_relative_path": archive.relative_to(root).as_posix(),
                "excluded_by_prior_evidence": object_path.name in excluded,
                **result,
            }
            if record["eligible"] and object_path.name in excluded:
                record["eligible"] = False
                record["reason"] = "excluded-by-prior-evidence-union"
            if record["eligible"]:
                eligible_episode_counts[object_path.name] += 1
            records.append(record)

    minimum_episodes = protocol["minimum_eligible_episodes_per_object"]
    eligible_objects = sorted(
        object_id
        for object_id, count in eligible_episode_counts.items()
        if count >= minimum_episodes
    )
    decision = (
        "inventory-ready"
        if len(eligible_objects) >= protocol["minimum_eligible_object_count"]
        else "inventory-insufficient"
    )
    reason_counts = Counter(str(record["reason"]) for record in records)
    metadata_manifest = {
        "records": records,
        "eligible_objects": eligible_objects,
        "object_episode_counts": dict(sorted(object_episode_counts.items())),
        "eligible_episode_counts": dict(sorted(eligible_episode_counts.items())),
    }
    return decision, {
        "root": str(root),
        "object_directory_count": object_count,
        "archive_record_count": len(records),
        "eligible_archive_count": sum(bool(record["eligible"]) for record in records),
        "eligible_object_count": len(eligible_objects),
        "eligible_objects": eligible_objects,
        "reason_counts": dict(sorted(reason_counts.items())),
        "ignored_top_level_counts": dict(sorted(ignored_top_level.items())),
        "object_episode_counts": dict(sorted(object_episode_counts.items())),
        "eligible_episode_counts": dict(sorted(eligible_episode_counts.items())),
        "records": records,
        "metadata_manifest_sha256": hashlib.sha256(_canonical_bytes(metadata_manifest)).hexdigest(),
    }


def build_inventory_result(
    *,
    protocol: dict[str, Any],
    request: dict[str, Any],
    source_protocol_git_blob_sha: str,
    prob4d_revision: str,
    runner_name: str,
    github_run_id: str,
) -> dict[str, Any]:
    if REVISION_RE.fullmatch(prob4d_revision) is None:
        raise InventoryContractError("prob4d_revision must be a lowercase commit SHA")
    if not runner_name:
        raise InventoryContractError("runner_name must be nonempty")
    decision, inventory = inventory_source_root(Path(protocol["source_root"]), protocol)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "profile": PROFILE,
        "issue_number": ISSUE_NUMBER,
        "decision": decision,
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "source_protocol_git_blob_sha": source_protocol_git_blob_sha,
        "prob4d_revision": prob4d_revision,
        "runner_name": runner_name,
        "github_run_id": github_run_id,
        "inventory": inventory,
        "execution_boundary": {
            "metadata_accessed": True,
            "archive_headers_read": True,
            "archive_member_payloads_read": False,
            "numeric_arrays_read": False,
            "provider_predictions_executed": False,
            "physical_query_scored": False,
            "target_outcomes_opened": False,
            "dataset_mutated": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["inventory_id"] = canonical_id(result, "inventory_id")
    return result


def render_summary(result: dict[str, Any]) -> str:
    inventory = result["inventory"]
    lines = [
        "# Deform360 persistent-point archive inventory",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Inventory ID: `{result['inventory_id']}`",
        f"- Object directories: `{inventory.get('object_directory_count', 0)}`",
        f"- Archive records: `{inventory.get('archive_record_count', 0)}`",
        f"- Eligible archives: `{inventory.get('eligible_archive_count', 0)}`",
        f"- Eligible objects: `{inventory.get('eligible_object_count', 0)}`",
        "",
        "Eligible object identities:",
        "",
    ]
    lines.extend(f"- `{object_id}`" for object_id in inventory.get("eligible_objects", []))
    lines.extend(
        [
            "",
            "Only tar headers and filesystem metadata were read. "
            "Archive payloads and arrays remained unopened.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InventoryContractError(f"{path} must contain one JSON object")
    return value


def _write_no_clobber(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise InventoryContractError(f"refusing to overwrite {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_protocol = subparsers.add_parser("validate-protocol")
    validate_protocol.add_argument("--protocol", type=Path, required=True)

    validate_request = subparsers.add_parser("validate-request")
    validate_request.add_argument("--request", type=Path, required=True)
    validate_request.add_argument("--protocol", type=Path, required=True)
    validate_request.add_argument("--source-protocol-git-blob-sha", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--request", type=Path, required=True)
    inventory.add_argument("--protocol", type=Path, required=True)
    inventory.add_argument("--source-protocol-git-blob-sha", required=True)
    inventory.add_argument("--prob4d-revision", required=True)
    inventory.add_argument("--runner-name", required=True)
    inventory.add_argument("--github-run-id", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        protocol = validate_protocol_record(_load_json(arguments.protocol))
        if arguments.command == "validate-protocol":
            print(json.dumps({"protocol_id": protocol["protocol_id"]}, sort_keys=True))
            return 0
        request = validate_request_record(
            _load_json(arguments.request),
            protocol=protocol,
            source_protocol_git_blob_sha=arguments.source_protocol_git_blob_sha,
        )
        if arguments.command == "validate-request":
            print(json.dumps({"request_id": request["request_id"]}, sort_keys=True))
            return 0
        result = build_inventory_result(
            protocol=protocol,
            request=request,
            source_protocol_git_blob_sha=arguments.source_protocol_git_blob_sha,
            prob4d_revision=arguments.prob4d_revision,
            runner_name=arguments.runner_name,
            github_run_id=arguments.github_run_id,
        )
        _write_no_clobber(
            arguments.output,
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
        _write_no_clobber(arguments.summary, render_summary(result))
        print(json.dumps({"decision": result["decision"], "inventory_id": result["inventory_id"]}))
        return 0
    except InventoryContractError as error:
        print(f"Deform360 PCD inventory contract failure: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
