"""Independent custody validation for retained CUT3R source-comparison artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn, cast

from ._atomic_file import atomic_write_bytes

CASE_SCHEMA: Final = "prob4d.cut3r-source-comparison-case"
CASE_SCHEMA_VERSION: Final = 1
SHARD_SCHEMA: Final = "prob4d.cut3r-source-comparison-shard"
SHARD_SCHEMA_VERSION: Final = 1
CUSTODY_SCHEMA: Final = "prob4d.cut3r-source-comparison-custody"
CUSTODY_SCHEMA_VERSION: Final = 1

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_ALLOWED_ROLES: Final = frozenset({"development", "calibration", "source_evaluation"})
_ALLOWED_STATUSES: Final = frozenset({"ordinary-success", "retained-technical-failure"})
_ALLOWED_SCOPES: Final = frozenset({"development-smoke", "frozen-source-shard"})
_FORBIDDEN_IMAGE_SUFFIXES: Final = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_FALSE_CASE_BOUNDARIES: Final = (
    "source_residuals_or_truth_opened",
    "candidate_reference_file_contents_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
    "bayesian_phystwin_executed",
    "causal4d_executed",
)
_PROGRESS_FIELDS: Final = (
    "source_rgb_frames_decoded",
    "cut3r_inference_executed",
    "source_predictions_written",
)
_CASE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "plan_id",
        "case_id",
        "group_id",
        "role",
        "status",
        "elapsed_seconds",
        "failure",
        "members",
        *_PROGRESS_FIELDS,
        *_FALSE_CASE_BOUNDARIES,
        "artifact_id",
    }
)
_MEMBER_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_FALSE_SHARD_BOUNDARIES: Final = (
    "source_residuals_or_truth_opened",
    "target_payloads_opened",
    "target_outcomes_opened",
)
_SHARD_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "plan_id",
        "scope",
        "shard_index",
        "shard_count",
        "case_count",
        "ordinary_success_count",
        "retained_technical_failure_count",
        "case_artifact_ids",
        *_FALSE_SHARD_BOUNDARIES,
        "artifact_id",
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


def content_id(value: object) -> str:
    """Return the canonical SHA-256 content identity used by the executor."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a genuine Boolean")
    return value


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_regular_bytes(path: Path, *, name: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    try:
        before = path.stat()
    except OSError as error:
        raise ValueError(f"failed to stat {name}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular file")
    try:
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ValueError(f"failed to read {name}: {error}") from error
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"{name} changed while being read")
    return payload


def _file_digest(path: Path, *, name: str) -> tuple[str, int]:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    try:
        before = path.stat()
    except OSError as error:
        raise ValueError(f"failed to stat {name}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise ValueError(f"failed to hash {name}: {error}") from error
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"{name} changed while being hashed")
    return digest.hexdigest(), before.st_size


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{name} contains non-finite JSON constant {value!r}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    raw = _read_regular_bytes(path, name=name)
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to parse {name}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _canonical_relative_path(value: object, *, name: str) -> PurePosixPath:
    relative = _require_string(value, name=name)
    if "\\" in relative:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"{name} must be a canonical relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not contain empty, dot, or parent components")
    if path.as_posix() != relative:
        raise ValueError(f"{name} is not in canonical POSIX form")
    return path


def _scan_regular_files(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("case artifact root must be a real directory")
    resolved_root = root.resolve(strict=True)
    result: dict[str, Path] = {}
    for directory, directory_names, file_names in os.walk(
        resolved_root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        for child_name in directory_names:
            child = current / child_name
            if child.is_symlink():
                raise ValueError(
                    f"case artifact contains symbolic-link directory "
                    f"{child.relative_to(resolved_root).as_posix()}"
                )
        for file_name in file_names:
            path = current / file_name
            if path.is_symlink():
                raise ValueError(
                    f"case artifact contains symbolic-link file "
                    f"{path.relative_to(resolved_root).as_posix()}"
                )
            relative = path.relative_to(resolved_root).as_posix()
            if relative == "case_manifest.json":
                continue
            if relative in result:
                raise ValueError(f"case artifact repeats member {relative!r}")
            result[relative] = path
    return result


def _validate_declared_members(
    case_root: Path,
    members_value: object,
    *,
    forbid_decoded_frames: bool,
) -> list[dict[str, object]]:
    if type(members_value) is not list or not members_value:
        raise ValueError("case manifest members must be a nonempty JSON array")
    declared: dict[str, tuple[str, int]] = {}
    normalized: list[dict[str, object]] = []
    for index, raw_member in enumerate(members_value):
        if type(raw_member) is not dict:
            raise ValueError(f"case member {index} must be a JSON object")
        member = cast(dict[str, Any], raw_member)
        if set(member) != _MEMBER_FIELDS:
            raise ValueError(f"case member {index} has unexpected fields")
        path = _canonical_relative_path(member["path"], name=f"members[{index}].path")
        relative = path.as_posix()
        if relative == "case_manifest.json":
            raise ValueError("case_manifest.json must not declare itself as a member")
        if relative in declared:
            raise ValueError(f"case manifest repeats member {relative!r}")
        if forbid_decoded_frames and (
            path.parts[0] == "decoded" or path.suffix.lower() in _FORBIDDEN_IMAGE_SUFFIXES
        ):
            raise ValueError(f"decoded source frame retained in case artifact: {relative}")
        digest = _require_sha256(
            member["sha256"],
            name=f"members[{index}].sha256",
        )
        byte_count = _require_integer(
            member["byte_count"],
            name=f"members[{index}].byte_count",
        )
        declared[relative] = (digest, byte_count)
        normalized.append(
            {"path": relative, "sha256": digest, "byte_count": byte_count}
        )

    actual = _scan_regular_files(case_root)
    if set(actual) != set(declared):
        missing = sorted(set(declared) - set(actual))
        undeclared = sorted(set(actual) - set(declared))
        raise ValueError(
            "case member roster mismatch: "
            f"missing={missing!r}, undeclared={undeclared!r}"
        )
    for relative, path in actual.items():
        measured_digest, measured_size = _file_digest(
            path,
            name=f"case member {relative}",
        )
        expected_digest, expected_size = declared[relative]
        if measured_digest != expected_digest:
            raise ValueError(f"case member digest mismatch: {relative}")
        if measured_size != expected_size:
            raise ValueError(f"case member byte-count mismatch: {relative}")
    return normalized


def validate_case_artifact(
    case_root: Path,
    *,
    expected_plan_id: str | None = None,
    require_success: bool = False,
    forbid_decoded_frames: bool = True,
) -> dict[str, Any]:
    """Validate one retained case and every byte named by its manifest."""

    root = case_root.resolve(strict=True)
    manifest = _load_json(root / "case_manifest.json", name="case manifest")
    if set(manifest) != _CASE_FIELDS:
        raise ValueError("case manifest has unexpected or missing fields")
    if manifest.get("schema") != CASE_SCHEMA:
        raise ValueError("unexpected case artifact schema")
    if manifest.get("schema_version") != CASE_SCHEMA_VERSION:
        raise ValueError("unsupported case artifact schema version")
    plan_id = _require_sha256(manifest["plan_id"], name="case plan_id")
    if expected_plan_id is not None and plan_id != _require_sha256(
        expected_plan_id,
        name="expected_plan_id",
    ):
        raise ValueError("case plan identity differs from the authorized plan")
    case_id = _require_string(manifest["case_id"], name="case_id")
    if case_id != root.name:
        raise ValueError("case manifest identity differs from its directory name")
    _require_string(manifest["group_id"], name="group_id")
    role = _require_string(manifest["role"], name="role")
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"unknown case role {role!r}")
    status = _require_string(manifest["status"], name="status")
    if status not in _ALLOWED_STATUSES:
        raise ValueError(f"unknown case status {status!r}")
    elapsed = manifest["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("elapsed_seconds must be a finite nonnegative real value")
    for field in _PROGRESS_FIELDS:
        _require_boolean(manifest[field], name=field)
    for field in _FALSE_CASE_BOUNDARIES:
        if _require_boolean(manifest[field], name=field):
            raise ValueError(f"case artifact exceeds information boundary: {field}")

    failure = manifest["failure"]
    if status == "ordinary-success":
        if failure is not None:
            raise ValueError("ordinary-success case must not retain a failure message")
        if not all(cast(bool, manifest[field]) for field in _PROGRESS_FIELDS):
            raise ValueError("ordinary-success case has an incomplete execution stage")
    else:
        if type(failure) is not str or not failure or failure != failure.strip():
            raise ValueError("technical failure must retain one bounded exact message")
        if len(failure) > 2000:
            raise ValueError("technical failure message exceeds the retained bound")
    if require_success and status != "ordinary-success":
        raise ValueError("case artifact is not an ordinary success")

    normalized_members = _validate_declared_members(
        root,
        manifest["members"],
        forbid_decoded_frames=forbid_decoded_frames,
    )
    if normalized_members != manifest["members"]:
        raise ValueError("case manifest members are not canonically ordered or typed")
    recorded_id = _require_sha256(manifest["artifact_id"], name="case artifact_id")
    unsigned = dict(manifest)
    unsigned.pop("artifact_id")
    if recorded_id != content_id(unsigned):
        raise ValueError("case artifact content identity is invalid")
    return manifest


def _case_directories(output_root: Path) -> list[Path]:
    cases_root = output_root / "cases"
    if cases_root.is_symlink() or not cases_root.is_dir():
        raise ValueError("output root has no real cases directory")
    result: list[Path] = []
    for child in sorted(cases_root.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise ValueError(f"cases directory contains symbolic link {child.name!r}")
        if not child.is_dir():
            raise ValueError(f"cases directory contains non-directory {child.name!r}")
        result.append(child)
    return result


def validate_shard_artifact(
    output_root: Path,
    shard_report_path: Path,
    *,
    expected_plan_id: str | None = None,
    require_success: bool = True,
    forbid_decoded_frames: bool = True,
) -> dict[str, Any]:
    """Validate a shard report and the exact retained cases that it references."""

    root = output_root.resolve(strict=True)
    report = _load_json(shard_report_path.resolve(strict=True), name="shard report")
    if set(report) != _SHARD_FIELDS:
        raise ValueError("shard report has unexpected or missing fields")
    if report.get("schema") != SHARD_SCHEMA:
        raise ValueError("unexpected shard report schema")
    if report.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported shard report schema version")
    plan_id = _require_sha256(report["plan_id"], name="shard plan_id")
    if expected_plan_id is not None and plan_id != _require_sha256(
        expected_plan_id,
        name="expected_plan_id",
    ):
        raise ValueError("shard plan identity differs from the authorized plan")
    scope = _require_string(report["scope"], name="scope")
    if scope not in _ALLOWED_SCOPES:
        raise ValueError(f"unknown shard scope {scope!r}")
    shard_count = _require_integer(report["shard_count"], name="shard_count", minimum=1)
    shard_index = _require_integer(report["shard_index"], name="shard_index")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    case_count = _require_integer(report["case_count"], name="case_count", minimum=1)
    success_count = _require_integer(
        report["ordinary_success_count"],
        name="ordinary_success_count",
    )
    failure_count = _require_integer(
        report["retained_technical_failure_count"],
        name="retained_technical_failure_count",
    )
    if success_count + failure_count != case_count:
        raise ValueError("shard status counts do not sum to case_count")
    if scope == "development-smoke" and case_count != 1:
        raise ValueError("development smoke must contain exactly one case")
    if scope == "frozen-source-shard" and case_count != 20:
        raise ValueError("each frozen two-way source shard must contain exactly 20 cases")
    for field in _FALSE_SHARD_BOUNDARIES:
        if _require_boolean(report[field], name=field):
            raise ValueError(f"shard report exceeds information boundary: {field}")

    raw_ids = report["case_artifact_ids"]
    if type(raw_ids) is not list or len(raw_ids) != case_count:
        raise ValueError("case_artifact_ids must match case_count")
    reported_ids = [
        _require_sha256(value, name=f"case_artifact_ids[{index}]")
        for index, value in enumerate(raw_ids)
    ]
    if len(set(reported_ids)) != len(reported_ids):
        raise ValueError("shard report repeats a case artifact identity")

    cases_by_id: dict[str, dict[str, Any]] = {}
    for directory in _case_directories(root):
        case = validate_case_artifact(
            directory,
            expected_plan_id=plan_id,
            require_success=False,
            forbid_decoded_frames=forbid_decoded_frames,
        )
        artifact_id = cast(str, case["artifact_id"])
        if artifact_id in cases_by_id:
            raise ValueError("two case directories share one artifact identity")
        cases_by_id[artifact_id] = case
    if any(artifact_id not in cases_by_id for artifact_id in reported_ids):
        raise ValueError("shard report references a missing case artifact")
    selected = [cases_by_id[artifact_id] for artifact_id in reported_ids]
    measured_success = sum(case["status"] == "ordinary-success" for case in selected)
    measured_failure = sum(
        case["status"] == "retained-technical-failure" for case in selected
    )
    if (measured_success, measured_failure) != (success_count, failure_count):
        raise ValueError("shard status counts differ from retained case manifests")
    if scope == "development-smoke" and selected[0]["role"] != "development":
        raise ValueError("development smoke references a non-development case")
    if require_success and measured_failure:
        raise ValueError("shard contains retained technical failures")

    recorded_id = _require_sha256(report["artifact_id"], name="shard artifact_id")
    unsigned_report = dict(report)
    unsigned_report.pop("artifact_id")
    if recorded_id != content_id(unsigned_report):
        raise ValueError("shard report content identity is invalid")

    receipt: dict[str, Any] = {
        "schema": CUSTODY_SCHEMA,
        "schema_version": CUSTODY_SCHEMA_VERSION,
        "decision": "source-comparison-custody-valid",
        "plan_id": plan_id,
        "shard_report_artifact_id": recorded_id,
        "scope": scope,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_count": case_count,
        "ordinary_success_count": measured_success,
        "retained_technical_failure_count": measured_failure,
        "case_ids": sorted(cast(str, case["case_id"]) for case in selected),
        "case_artifact_ids": sorted(reported_ids),
        "decoded_source_frames_retained": False,
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }
    receipt["receipt_id"] = content_id(receipt)
    return receipt


def write_custody_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Publish a validated receipt atomically without replacing different bytes."""

    encoded = (
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        atomic_write_bytes(path, encoded, overwrite=False)
    except FileExistsError:
        if _read_regular_bytes(path, name="existing custody receipt") != encoded:
            raise


__all__ = [
    "CASE_SCHEMA",
    "CASE_SCHEMA_VERSION",
    "CUSTODY_SCHEMA",
    "CUSTODY_SCHEMA_VERSION",
    "SHARD_SCHEMA",
    "SHARD_SCHEMA_VERSION",
    "content_id",
    "validate_case_artifact",
    "validate_shard_artifact",
    "write_custody_receipt",
]
