#!/usr/bin/env python3
"""Bind and enrich the Deform360 metadata audit for the processed repository."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_IMPLEMENTATION_PATH = Path(__file__).with_name("audit_deform360_source_bundle_impl_v1.py")
_SPEC = importlib.util.spec_from_file_location(
    "prob4d_deform360_source_bundle_audit_impl_v1",
    _IMPLEMENTATION_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Deform360 audit implementation: {_IMPLEMENTATION_PATH}")
_IMPLEMENTATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPLEMENTATION)

_CANONICAL_SOURCE_ROOT = Path(
    "/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository"
)
_CENSUS_VERSION = 1
_EPISODE_NAME = re.compile(r"episode_[0-9]+\Z")
_IMPLEMENTATION.EXPECTED_SOURCE_ROOT = _CANONICAL_SOURCE_ROOT
_ORIGINAL_VALIDATE_PROTOCOL_RECORD = _IMPLEMENTATION.validate_protocol_record
_ORIGINAL_SCAN_SOURCE_ROOT = _IMPLEMENTATION.scan_source_root


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _matches_forbidden(name: str, forbidden_tokens: tuple[str, ...]) -> str | None:
    normalized = name.casefold().replace("_", "-").replace(" ", "-")
    return next((token for token in forbidden_tokens if token in normalized), None)


def _scan_episode_layout(
    episode_root: Path,
    *,
    forbidden_tokens: tuple[str, ...],
    remaining_entries: int,
    max_depth: int,
) -> tuple[dict[str, Any], int]:
    counts: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    basenames: Counter[str] = Counter()
    forbidden: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    layout: list[dict[str, str]] = []
    metadata_manifest = hashlib.sha256()
    total_regular_file_bytes = 0
    entry_count = 0
    depth_limit_skipped = 0
    entry_limit_exceeded = False
    stack: list[tuple[Path, str, int]] = [(episode_root, "", 0)]

    while stack and not entry_limit_exceeded:
        directory, relative_directory, depth = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            errors.append(
                {
                    "operation": "directory-scan",
                    "path": relative_directory or ".",
                    "errno": exc.errno,
                }
            )
            continue

        child_directories: list[tuple[Path, str, int]] = []
        for entry in entries:
            forbidden_token = _matches_forbidden(entry.name, forbidden_tokens)
            if forbidden_token is not None:
                forbidden[forbidden_token] += 1
                continue
            if entry_count >= remaining_entries:
                entry_limit_exceeded = True
                break
            relative_path = (
                entry.name
                if not relative_directory
                else f"{relative_directory}/{entry.name}"
            )
            child_depth = depth + 1
            if child_depth > max_depth:
                depth_limit_skipped += 1
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(
                    {"operation": "lstat", "path": relative_path, "errno": exc.errno}
                )
                continue
            entry_kind = _kind(metadata.st_mode)
            size = int(metadata.st_size)
            entry_count += 1
            counts[entry_kind] += 1
            layout.append({"path": relative_path, "type": entry_kind})
            metadata_manifest.update(
                f"{entry_kind}\0{stat.S_IMODE(metadata.st_mode):04o}\0"
                f"{size}\0{relative_path}\n".encode()
            )
            if entry_kind == "file":
                total_regular_file_bytes += size
                extensions[Path(entry.name).suffix.casefold() or "<none>"] += 1
                basenames[entry.name] += 1
            elif entry_kind == "directory":
                child_directories.append((Path(entry.path), relative_path, child_depth))

        stack.extend(reversed(child_directories))

    layout.sort(key=lambda value: (value["path"], value["type"]))
    decision = "episode-layout-present"
    if entry_limit_exceeded:
        decision = "episode-entry-limit-exceeded"
    elif errors:
        decision = "episode-layout-partial"
    return (
        {
            "decision": decision,
            "entry_count": entry_count,
            "counts": dict(sorted(counts.items())),
            "total_regular_file_bytes": total_regular_file_bytes,
            "extension_counts": dict(sorted(extensions.items())),
            "basename_counts": dict(sorted(basenames.items())),
            "layout_signature_sha256": _canonical_digest(layout),
            "metadata_signature_sha256": metadata_manifest.hexdigest(),
            "layout": layout,
            "forbidden_token_counts": dict(sorted(forbidden.items())),
            "depth_limit_skipped": depth_limit_skipped,
            "metadata_errors": errors[:50],
            "entry_limit_exceeded": entry_limit_exceeded,
        },
        entry_count,
    )


def build_processed_repository_census(
    source_root: Path,
    *,
    forbidden_tokens: tuple[str, ...],
    max_entries: int,
    max_depth: int,
) -> tuple[str, dict[str, Any]]:
    """Build an object/episode/layout census using directory metadata only."""

    processed_root = source_root / "processed"
    try:
        metadata = processed_root.lstat()
    except FileNotFoundError:
        return "processed-root-missing", {"error": "processed directory does not exist"}
    except OSError as exc:
        return "processed-root-unreadable", {
            "error": "processed directory metadata could not be read",
            "errno": exc.errno,
        }
    if stat.S_ISLNK(metadata.st_mode):
        return "processed-root-symlink-rejected", {"error": "processed root is a symlink"}
    if not stat.S_ISDIR(metadata.st_mode):
        return "processed-root-not-directory", {
            "error": "processed root is not a directory"
        }

    try:
        with os.scandir(processed_root) as iterator:
            root_entries = sorted(iterator, key=lambda item: item.name)
    except OSError as exc:
        return "processed-root-unreadable", {
            "error": "processed directory could not be enumerated",
            "errno": exc.errno,
        }

    objects: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    non_object_entries: list[dict[str, Any]] = []
    layout_groups: dict[str, dict[str, Any]] = {}
    global_extensions: Counter[str] = Counter()
    global_basenames: Counter[str] = Counter()
    total_episode_bytes = 0
    total_scanned_entries = 0
    census_errors: list[dict[str, Any]] = []
    forbidden_counts: Counter[str] = Counter()

    for object_entry in root_entries:
        forbidden_token = _matches_forbidden(object_entry.name, forbidden_tokens)
        if forbidden_token is not None:
            forbidden_counts[forbidden_token] += 1
            continue
        try:
            object_metadata = object_entry.stat(follow_symlinks=False)
        except OSError as exc:
            census_errors.append(
                {"operation": "lstat-object", "path": object_entry.name, "errno": exc.errno}
            )
            continue
        object_kind = _kind(object_metadata.st_mode)
        if object_kind != "directory":
            non_object_entries.append(
                {
                    "path": object_entry.name,
                    "type": object_kind,
                    "size_bytes": int(object_metadata.st_size),
                }
            )
            continue

        object_root = Path(object_entry.path)
        try:
            with os.scandir(object_root) as iterator:
                object_children = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            census_errors.append(
                {
                    "operation": "scan-object",
                    "path": object_entry.name,
                    "errno": exc.errno,
                }
            )
            continue

        object_episode_ids: list[str] = []
        object_layout_counts: Counter[str] = Counter()
        object_file_count = 0
        object_directory_count = 0
        object_regular_file_bytes = 0
        object_auxiliary_entries: list[dict[str, Any]] = []

        for child in object_children:
            forbidden_token = _matches_forbidden(child.name, forbidden_tokens)
            if forbidden_token is not None:
                forbidden_counts[forbidden_token] += 1
                continue
            try:
                child_metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                census_errors.append(
                    {
                        "operation": "lstat-object-child",
                        "path": f"{object_entry.name}/{child.name}",
                        "errno": exc.errno,
                    }
                )
                continue
            child_kind = _kind(child_metadata.st_mode)
            if child_kind != "directory" or _EPISODE_NAME.fullmatch(child.name) is None:
                object_auxiliary_entries.append(
                    {
                        "path": child.name,
                        "type": child_kind,
                        "size_bytes": int(child_metadata.st_size),
                    }
                )
                continue

            remaining_entries = max_entries - total_scanned_entries
            if remaining_entries <= 0:
                census_errors.append(
                    {
                        "operation": "entry-limit",
                        "path": f"{object_entry.name}/{child.name}",
                    }
                )
                break
            episode_layout, consumed = _scan_episode_layout(
                Path(child.path),
                forbidden_tokens=forbidden_tokens,
                remaining_entries=remaining_entries,
                max_depth=max_depth,
            )
            total_scanned_entries += consumed
            episode_record = {
                "object_id": object_entry.name,
                "episode_id": child.name,
                "relative_path": f"processed/{object_entry.name}/{child.name}",
                **{
                    key: value
                    for key, value in episode_layout.items()
                    if key not in {"layout", "basename_counts"}
                },
            }
            episodes.append(episode_record)
            object_episode_ids.append(child.name)
            signature = episode_layout["layout_signature_sha256"]
            object_layout_counts[signature] += 1
            counts = episode_layout["counts"]
            object_file_count += int(counts.get("file", 0))
            object_directory_count += int(counts.get("directory", 0))
            object_regular_file_bytes += int(
                episode_layout["total_regular_file_bytes"]
            )
            total_episode_bytes += int(episode_layout["total_regular_file_bytes"])
            global_extensions.update(episode_layout["extension_counts"])
            global_basenames.update(episode_layout["basename_counts"])

            layout_group = layout_groups.setdefault(
                signature,
                {
                    "layout_signature_sha256": signature,
                    "episode_count": 0,
                    "layout": episode_layout["layout"],
                    "examples": [],
                },
            )
            layout_group["episode_count"] += 1
            if len(layout_group["examples"]) < 10:
                layout_group["examples"].append(
                    f"{object_entry.name}/{child.name}"
                )
            if episode_layout["decision"] != "episode-layout-present":
                census_errors.append(
                    {
                        "operation": "episode-layout",
                        "path": f"{object_entry.name}/{child.name}",
                        "decision": episode_layout["decision"],
                    }
                )

        objects.append(
            {
                "object_id": object_entry.name,
                "episode_count": len(object_episode_ids),
                "episode_ids": sorted(object_episode_ids),
                "file_count": object_file_count,
                "directory_count": object_directory_count,
                "total_regular_file_bytes": object_regular_file_bytes,
                "layout_signature_counts": dict(sorted(object_layout_counts.items())),
                "auxiliary_entries": object_auxiliary_entries,
            }
        )

    objects.sort(key=lambda value: value["object_id"])
    episodes.sort(key=lambda value: (value["object_id"], value["episode_id"]))
    ordered_layout_groups = sorted(
        layout_groups.values(),
        key=lambda value: (-int(value["episode_count"]), value["layout_signature_sha256"]),
    )
    decision = "processed-census-present"
    if total_scanned_entries >= max_entries:
        decision = "processed-census-entry-limit-exceeded"
    elif census_errors:
        decision = "processed-census-partial"
    elif not objects or not episodes:
        decision = "processed-census-empty"

    census: dict[str, Any] = {
        "schema": "prob4d.deform360-processed-repository-census",
        "schema_version": _CENSUS_VERSION,
        "decision": decision,
        "source_root": str(source_root),
        "processed_root": str(processed_root),
        "object_count": len(objects),
        "episode_count": len(episodes),
        "scanned_episode_entry_count": total_scanned_entries,
        "total_episode_regular_file_bytes": total_episode_bytes,
        "objects": objects,
        "episodes": episodes,
        "layout_signatures": ordered_layout_groups,
        "global_extension_counts": dict(sorted(global_extensions.items())),
        "global_basename_counts": dict(sorted(global_basenames.items())),
        "non_object_entries": non_object_entries,
        "forbidden_token_counts": dict(sorted(forbidden_counts.items())),
        "errors": census_errors[:100],
        "execution_boundary": {
            "dataset_file_contents_opened": False,
            "symlinks_followed": False,
            "dataset_mutated": False,
        },
    }
    census["census_id"] = _canonical_digest(census)
    return decision, census


def validate_protocol_record(record: dict[str, Any]) -> dict[str, Any]:
    validated = _ORIGINAL_VALIDATE_PROTOCOL_RECORD(record)
    if record.get("structured_processed_census_version") != _CENSUS_VERSION:
        raise _IMPLEMENTATION.AuditContractError(
            "structured_processed_census_version must be 1"
        )
    return validated


def scan_source_root(
    source_root: Path,
    *,
    forbidden_tokens: tuple[str, ...],
    max_entries: int,
    max_depth: int,
    largest_file_limit: int,
    sample_path_limit: int,
) -> tuple[str, dict[str, Any]]:
    decision, inventory = _ORIGINAL_SCAN_SOURCE_ROOT(
        source_root,
        forbidden_tokens=forbidden_tokens,
        max_entries=max_entries,
        max_depth=max_depth,
        largest_file_limit=largest_file_limit,
        sample_path_limit=sample_path_limit,
    )
    if decision != "source-bundle-present":
        return decision, inventory
    census_decision, census = build_processed_repository_census(
        source_root,
        forbidden_tokens=forbidden_tokens,
        max_entries=max_entries,
        max_depth=max_depth,
    )
    inventory["processed_repository_census"] = census
    if census_decision != "processed-census-present":
        return "source-bundle-partial", inventory
    return decision, inventory


_IMPLEMENTATION.validate_protocol_record = validate_protocol_record
_IMPLEMENTATION.scan_source_root = scan_source_root

for _name in dir(_IMPLEMENTATION):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPLEMENTATION, _name)

# Re-export wrapper-owned helpers after implementation names are copied.
globals()["build_processed_repository_census"] = build_processed_repository_census
globals()["validate_protocol_record"] = validate_protocol_record
globals()["scan_source_root"] = scan_source_root

if __name__ == "__main__":
    raise SystemExit(_IMPLEMENTATION.main())
