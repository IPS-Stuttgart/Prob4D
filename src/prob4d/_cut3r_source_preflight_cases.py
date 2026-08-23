"""Canonical source-group and source-case parsing for the CUT3R preflight."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ._cut3r_source_preflight_common import (
    SIDECAR_NAMES,
    SOURCE_CASE_FIELDS,
    SOURCE_GROUP_FIELDS,
    SOURCE_ROLES,
    TARGET_GROUP_FIELDS,
    _exact_keys,
    _integer,
    _literal_string,
    _record_id,
    _safe_relative,
    _sha256,
)


def _validate_source_groups(
    value: object,
    *,
    expected_count: int,
) -> dict[str, dict[str, Any]]:
    if type(value) is not list or len(value) != expected_count:
        raise ValueError(f"source_groups must contain exactly {expected_count} records")
    result: dict[str, dict[str, Any]] = {}
    object_episodes: set[tuple[str, int]] = set()
    for index, raw in enumerate(cast(list[object], value)):
        if type(raw) is not dict:
            raise ValueError(f"source_groups[{index}] must be a JSON object")
        record = cast(dict[str, Any], raw)
        _exact_keys(record, SOURCE_GROUP_FIELDS, name=f"source_groups[{index}]")
        object_id = _literal_string(
            record.get("object_id"),
            name=f"source_groups[{index}].object_id",
        )
        episode_id = _integer(record.get("episode_id"), name=f"source_groups[{index}].episode_id")
        stratum = _literal_string(record.get("stratum"), name=f"source_groups[{index}].stratum")
        group_id = _literal_string(record.get("group_id"), name=f"source_groups[{index}].group_id")
        role = _literal_string(record.get("role"), name=f"source_groups[{index}].role")
        if role not in SOURCE_ROLES:
            raise ValueError(f"source_groups[{index}].role is not canonical")
        if group_id in result:
            raise ValueError(f"source_groups repeats group_id {group_id!r}")
        if (object_id, episode_id) in object_episodes:
            raise ValueError("source_groups repeats an object/episode")
        object_episodes.add((object_id, episode_id))
        result[group_id] = {
            "object_id": object_id,
            "episode_id": episode_id,
            "stratum": stratum,
            "group_id": group_id,
            "role": role,
        }
    return result


def _validate_target_groups(
    value: object,
    *,
    expected_count: int,
    source_object_episodes: set[tuple[str, int]],
) -> None:
    if type(value) is not list or len(value) != expected_count:
        raise ValueError(f"forbidden_target_groups must contain exactly {expected_count} records")
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(cast(list[object], value)):
        if type(raw) is not dict:
            raise ValueError(f"forbidden_target_groups[{index}] must be a JSON object")
        record = cast(dict[str, Any], raw)
        _exact_keys(record, TARGET_GROUP_FIELDS, name=f"forbidden_target_groups[{index}]")
        object_id = _literal_string(
            record.get("object_id"), name=f"forbidden_target_groups[{index}].object_id"
        )
        episode_id = _integer(
            record.get("episode_id"), name=f"forbidden_target_groups[{index}].episode_id"
        )
        _literal_string(record.get("stratum"), name=f"forbidden_target_groups[{index}].stratum")
        key = (object_id, episode_id)
        if key in seen:
            raise ValueError("forbidden_target_groups repeats an object/episode")
        if key in source_object_episodes:
            raise ValueError("source and forbidden-target object/episodes overlap")
        seen.add(key)


def _collect_source_case_descriptors(
    freeze: Mapping[str, Any],
    *,
    source_groups: Mapping[str, Mapping[str, Any]],
    expected_case_count: int,
) -> list[dict[str, object]]:
    raw_cases = freeze.get("source_cases")
    if type(raw_cases) is not list or len(raw_cases) != expected_case_count:
        raise ValueError(f"source_cases must contain exactly {expected_case_count} records")
    descriptors: list[dict[str, object]] = []
    case_ids: set[str] = set()
    source_case_ids: set[str] = set()
    for index, raw in enumerate(cast(list[object], raw_cases)):
        if type(raw) is not dict:
            raise ValueError(f"source_cases[{index}] must be a JSON object")
        case = cast(dict[str, Any], raw)
        _exact_keys(case, SOURCE_CASE_FIELDS, name=f"source_cases[{index}]")
        case_id = _literal_string(case.get("case_id"), name=f"source_cases[{index}].case_id")
        group_id = _literal_string(case.get("group_id"), name=f"source_cases[{index}].group_id")
        group = source_groups.get(group_id)
        if group is None:
            raise ValueError(f"source case {case_id!r} names unknown group {group_id!r}")
        object_id = _literal_string(case.get("object_id"), name=f"source_cases[{index}].object_id")
        episode_id = _integer(case.get("episode_id"), name=f"source_cases[{index}].episode_id")
        camera = _literal_string(case.get("camera"), name=f"source_cases[{index}].camera")
        if object_id != group["object_id"] or episode_id != group["episode_id"]:
            raise ValueError(f"source case {case_id!r} disagrees with its group identity")
        expected_episode = f"{object_id}/episode_{episode_id:04d}"
        relative_episode = _safe_relative(
            case.get("relative_episode_path"),
            name=f"source_cases[{index}].relative_episode_path",
        )
        if relative_episode != expected_episode:
            raise ValueError(f"source case {case_id!r} has a noncanonical episode path")
        relative_camera = _safe_relative(
            case.get("relative_camera_path"),
            name=f"source_cases[{index}].relative_camera_path",
        )
        if relative_camera != f"{relative_episode}/{camera}":
            raise ValueError(f"source case {case_id!r} has a noncanonical camera path")
        if case_id != f"{group_id}-{camera}":
            raise ValueError(f"source case {case_id!r} has a noncanonical case ID")
        digest = _sha256(
            case.get("input_video_sha256"),
            name=f"source_cases[{index}].input_video_sha256",
        )
        byte_count = _integer(
            case.get("input_video_byte_count"),
            name=f"source_cases[{index}].input_video_byte_count",
            minimum=1,
        )
        _integer(
            case.get("aligned_timestamp_count"),
            name=f"source_cases[{index}].aligned_timestamp_count",
            minimum=1,
        )
        raw_sidecar_sha = case.get("sidecar_sha256")
        raw_sidecar_bytes = case.get("sidecar_byte_count")
        if type(raw_sidecar_sha) is not dict or type(raw_sidecar_bytes) is not dict:
            raise ValueError(f"source_cases[{index}] sidecar identities must be JSON objects")
        sidecar_sha = cast(dict[str, Any], raw_sidecar_sha)
        sidecar_bytes = cast(dict[str, Any], raw_sidecar_bytes)
        _exact_keys(
            sidecar_sha,
            set(SIDECAR_NAMES),
            name=f"source_cases[{index}].sidecar_sha256",
        )
        _exact_keys(
            sidecar_bytes,
            set(SIDECAR_NAMES),
            name=f"source_cases[{index}].sidecar_byte_count",
        )
        sidecars = {
            name: {
                "relative_path": f"{relative_camera}/{name}",
                "sha256": _sha256(
                    sidecar_sha.get(name),
                    name=f"source_cases[{index}].sidecar_sha256.{name}",
                ),
                "byte_count": _integer(
                    sidecar_bytes.get(name),
                    name=f"source_cases[{index}].sidecar_byte_count.{name}",
                    minimum=1,
                ),
            }
            for name in SIDECAR_NAMES
        }
        source_case_id = _sha256(
            case.get("source_case_id"),
            name=f"source_cases[{index}].source_case_id",
        )
        unsigned = dict(case)
        unsigned.pop("source_case_id")
        if source_case_id != _record_id(unsigned):
            raise ValueError(f"source case {case_id!r} content identity is invalid")
        if case_id in case_ids or source_case_id in source_case_ids:
            raise ValueError("source_cases repeats a case identity")
        case_ids.add(case_id)
        source_case_ids.add(source_case_id)
        descriptors.append(
            {
                "group_id": group_id,
                "role": group["role"],
                "case_id": case_id,
                "view_id": camera,
                "object_id": object_id,
                "episode_id": episode_id,
                "relative_episode_path": relative_episode,
                "relative_video_path": f"{relative_camera}/undistorted.mp4",
                "video_sha256": digest,
                "video_byte_count": byte_count,
                "source_case_id": source_case_id,
                "sidecars": sidecars,
            }
        )
    return sorted(descriptors, key=lambda item: cast(str, item["case_id"]))
