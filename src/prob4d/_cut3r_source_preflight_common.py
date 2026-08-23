"""Common strict primitives for the retained CUT3R source preflight."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

REQUEST_SCHEMA: Final = "prob4d.cut3r-deform360-source-comparison-preflight-request"
REQUEST_VERSION: Final = 1
SOURCE_FREEZE_SCHEMA: Final = "prob4d.cut3r-deform360-source-freeze"
SOURCE_FREEZE_VERSION: Final = 1
SOURCE_FREEZE_READY: Final = "source-support-freeze-ready"
SOURCE_ROLES: Final = ("development", "calibration", "source_evaluation")
SIDECAR_NAMES: Final = (
    "aligned_timestamps.txt",
    "alignment.json",
    "metadata.json",
)
REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "issue_number",
        "preflight_request_id",
        "source_freeze_path",
        "comparison_spec_path",
        "comparison_lock_path",
        "source_group_count",
        "forbidden_target_group_count",
        "expected_case_count",
        "source_rgb_frames_decoded",
        "source_prediction_payloads_opened",
        "source_residuals_or_truth_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "comparison_execution_authorized",
        "claim_boundary",
    }
)
FALSE_FIELDS: Final = (
    "source_rgb_frames_decoded",
    "source_prediction_payloads_opened",
    "source_residuals_or_truth_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "comparison_execution_authorized",
)
SOURCE_GROUP_FIELDS: Final = frozenset(
    {"object_id", "episode_id", "stratum", "group_id", "role"}
)
TARGET_GROUP_FIELDS: Final = frozenset({"object_id", "episode_id", "stratum"})
SOURCE_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "group_id",
        "object_id",
        "episode_id",
        "camera",
        "relative_episode_path",
        "relative_camera_path",
        "input_video_sha256",
        "input_video_byte_count",
        "aligned_timestamp_count",
        "sidecar_sha256",
        "sidecar_byte_count",
        "source_case_id",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record_id(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name} {path}: {error}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    keys = set(value)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _revision(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-character Git revision")
    return text


def _safe_relative(value: object, *, name: str) -> str:
    text = _literal_string(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a confined relative path")
    return path.as_posix()


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _confined_path(root: Path, relative: str, *, name: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} escapes or is unavailable under {root}: {relative}") from error
    return resolved


def _confined_regular_file(root: Path, relative: str, *, name: str) -> Path:
    resolved = _confined_path(root, relative, name=name)
    try:
        metadata = resolved.stat()
    except OSError as error:
        raise ValueError(f"failed to stat {name}: {resolved}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file: {resolved}")
    return resolved


def _confined_directory(root: Path, relative: str, *, name: str) -> Path:
    resolved = _confined_path(root, relative, name=name)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"refusing to hash symbolic link: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"refusing to hash non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"file changed while it was being hashed: {path}")
    return digest.hexdigest()


def validate_request(path: Path, *, repository: Path) -> dict[str, Any]:
    request = _load_json(path, name="source-comparison preflight request")
    _exact_keys(request, REQUEST_FIELDS, name="source-comparison preflight request")
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != REQUEST_VERSION:
        raise ValueError("unsupported source-comparison preflight request")
    if request.get("issue_number") != 49:
        raise ValueError("preflight request is not bound to issue 49")
    if request.get("source_group_count") != 10:
        raise ValueError("preflight source group count changed")
    if request.get("forbidden_target_group_count") != 12:
        raise ValueError("preflight forbidden target count changed")
    if request.get("expected_case_count") != 40:
        raise ValueError("preflight expected case count changed")
    if any(request.get(name) is not False for name in FALSE_FIELDS):
        raise ValueError("preflight request exceeds its outcome-blind boundary")
    recorded_id = _sha256(request.get("preflight_request_id"), name="preflight_request_id")
    unsigned = dict(request)
    unsigned.pop("preflight_request_id")
    if recorded_id != _record_id(unsigned):
        raise ValueError("preflight_request_id does not match the canonical request content")
    for field in ("source_freeze_path", "comparison_spec_path", "comparison_lock_path"):
        relative = _safe_relative(request.get(field), name=field)
        if not relative.startswith("protocols/locks/") or not relative.endswith(".json"):
            raise ValueError(f"{field} must name a repository lock JSON")
        try:
            _confined_regular_file(repository, relative, name=field)
        except ValueError as error:
            raise ValueError(f"required merged lock is missing or invalid: {relative}") from error
        request[field] = relative
    _literal_string(request.get("claim_boundary"), name="claim_boundary")
    return cast(dict[str, Any], json.loads(_canonical_json_bytes(request)))
