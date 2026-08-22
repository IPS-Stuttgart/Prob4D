"""Shared strict primitives for retained CUT3R source-freeze verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

SOURCE_FREEZE_SCHEMA: Final = "prob4d.cut3r-deform360-source-freeze"
SOURCE_FREEZE_VERSION: Final = 1
SUPPORT_PASS: Final = "source-support-freeze-ready"
SUPPORT_NEGATIVE: Final = "insufficient-common-camera-support"

_SOURCE_ROLES: Final = ("development", "calibration", "source_evaluation")
_REQUIRED_SIDECARS: Final = (
    "aligned_timestamps.txt",
    "alignment.json",
    "metadata.json",
)
_REQUIRED_STREAM_MEMBERS: Final = frozenset({"video", "timestamps", "alignment", "metadata"})
_INFORMATION_BOUNDARY: Final = {
    "camera_panel_change_after_freeze_allowed": False,
    "downstream_physical_innovations_opened": False,
    "replacement_after_freeze_allowed": False,
    "source_future_geometry_opened": False,
    "source_prediction_payloads_opened": False,
    "source_residuals_or_truth_opened": False,
    "source_rgb_frames_decoded": False,
    "source_rgb_video_bytes_hashed": True,
    "target_outcomes_opened": False,
    "target_payloads_opened": False,
}

_BASE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_name",
        "decision",
        "source_protocol",
        "deform360_selection",
        "provider",
        "prob4d",
        "source_group_count",
        "source_groups",
        "forbidden_target_group_count",
        "forbidden_target_groups",
        "support",
        "camera_calibration_inputs",
        "camera_panel",
        "source_cases",
        "information_boundary",
        "claim_boundary",
        "source_freeze_id",
    }
)
_PASS_FIELDS: Final = _BASE_FIELDS | {"comparison_spec_sha256"}
_SOURCE_GROUP_FIELDS: Final = frozenset(
    {"group_id", "object_id", "episode_id", "stratum", "role"}
)
_TARGET_GROUP_FIELDS: Final = frozenset({"object_id", "episode_id", "stratum"})
_FILE_IDENTITY_FIELDS: Final = frozenset({"sha256", "byte_count"})
_SELECTION_IDENTITY_FIELDS: Final = frozenset(
    {"sha256", "byte_count", "selection_artifact_sha256", "selection_sha256"}
)
_PROVIDER_FIELDS: Final = frozenset(
    {
        "repository",
        "revision",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_byte_count",
        "execution_mode",
        "revisit_count",
        "global_alignment",
        "second_pass_allowed",
    }
)
_PROB4D_FIELDS: Final = frozenset(
    {
        "revision",
        "distribution_filename",
        "distribution_sha256",
        "distribution_byte_count",
    }
)
_SUPPORT_FIELDS: Final = frozenset(
    {
        "required_frame_interval",
        "common_supported_camera_count",
        "common_supported_cameras",
        "minimum_common_supported_cameras",
        "stream_rows",
    }
)
_STREAM_ROW_FIELDS: Final = frozenset(
    {
        "camera",
        "missing_required_members",
        "aligned_timestamp_count",
        "required_frame_count",
        "supported",
        "group_id",
        "object_id",
        "episode_id",
    }
)
_CALIBRATION_FIELDS: Final = frozenset(
    {"group_id", "object_id", "episode_id", "intrinsics", "extrinsics"}
)
_CALIBRATION_IDENTITY_FIELDS: Final = frozenset({"relative_path", "sha256", "byte_count"})
_CAMERA_PANEL_FIELDS: Final = frozenset(
    {
        "selected_cameras",
        "panel_size",
        "selection_rule",
        "first_camera_rule",
        "camera_center_maximum_deviation_m",
        "camera_direction",
    }
)
_SOURCE_CASE_FIELDS: Final = frozenset(
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
_COMPARISON_SPEC_FIELDS: Final = frozenset(
    {
        "protocol_name",
        "provider_revision",
        "checkpoint_sha256",
        "prob4d_revision",
        "prob4d_distribution_sha256",
        "window_size",
        "overlap",
        "confidence_threshold",
        "storage_dtype",
        "random_seeds",
        "groups",
        "group_roles",
        "include_revisit_diagnostic",
    }
)
_COMPARISON_GROUP_FIELDS: Final = frozenset({"group_id", "cases"})
_COMPARISON_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "input_video_sha256",
        "input_video_byte_count",
        "frame_start",
        "frame_stop_exclusive",
        "evaluation_frame_start",
        "evaluation_frame_stop_exclusive",
    }
)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _regular_file_snapshot(path: Path, *, name: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {path}")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError(f"failed to stat {name} at {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")
    return metadata


def _read_regular_bytes(path: Path, *, name: str) -> bytes:
    before = _regular_file_snapshot(path, name=name)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"failed to read {name} at {path}: {error}") from error
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"{name} changed while it was being read: {path}")
    return payload


def _load_json_object(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_bytes(path, name=name)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"failed to parse strict {name} JSON at {path}: {error}") from error
    return dict(_mapping(value, name=name)), payload


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


def _content_id(value: Mapping[str, Any], *, id_field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(id_field, None)
    return _sha256_json(unsigned)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(Mapping[str, Any], value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _sequence(value: object, *, name: str, nonempty: bool = False) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    if nonempty and not value:
        raise ValueError(f"{name} must be nonempty")
    return cast(Sequence[Any], value)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a genuine Boolean")
    return value


def _finite_number(value: object, *, name: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a genuine finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        threshold = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {threshold}")
    return result


def _sha256(value: object, *, name: str) -> str:
    digest = _string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _revision(value: object, *, name: str) -> str:
    revision = _string(value, name=name)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"{name} must be an exact lowercase 40-character Git revision")
    return revision


def _basename(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if "/" in result or "\\" in result or result in {".", ".."}:
        raise ValueError(f"{name} must be a portable basename")
    return result


def _relative_path(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    path = PurePosixPath(result)
    if path.is_absolute() or path.as_posix() != result:
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts) or "\\" in result:
        raise ValueError(f"{name} must not escape or normalize its directory")
    return result


def _file_identity(value: object, *, name: str, minimum_bytes: int = 1) -> dict[str, Any]:
    mapping = _mapping(value, name=name)
    _exact_fields(mapping, _FILE_IDENTITY_FIELDS, name=name)
    return {
        "sha256": _sha256(mapping["sha256"], name=f"{name}.sha256"),
        "byte_count": _integer(
            mapping["byte_count"],
            name=f"{name}.byte_count",
            minimum=minimum_bytes,
        ),
    }
