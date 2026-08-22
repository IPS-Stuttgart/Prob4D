"""Structural record validation for retained CUT3R source freezes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._cut3r_source_freeze_verify_common import (
    _CALIBRATION_FIELDS,
    _CALIBRATION_IDENTITY_FIELDS,
    _CAMERA_PANEL_FIELDS,
    _REQUIRED_SIDECARS,
    _REQUIRED_STREAM_MEMBERS,
    _SOURCE_CASE_FIELDS,
    _SOURCE_GROUP_FIELDS,
    _SOURCE_ROLES,
    _STREAM_ROW_FIELDS,
    _SUPPORT_FIELDS,
    _TARGET_GROUP_FIELDS,
    _boolean,
    _exact_fields,
    _finite_number,
    _integer,
    _mapping,
    _relative_path,
    _sequence,
    _sha256,
    _sha256_json,
    _string,
)


def _source_groups(value: object) -> list[dict[str, Any]]:
    records = _sequence(value, name="source_groups", nonempty=True)
    result: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    object_episodes: set[tuple[str, int]] = set()
    for index, item in enumerate(records):
        mapping = _mapping(item, name=f"source_groups[{index}]")
        _exact_fields(mapping, _SOURCE_GROUP_FIELDS, name=f"source_groups[{index}]")
        group_id = _string(mapping["group_id"], name=f"source_groups[{index}].group_id")
        object_id = _string(mapping["object_id"], name=f"source_groups[{index}].object_id")
        episode_id = _integer(
            mapping["episode_id"],
            name=f"source_groups[{index}].episode_id",
        )
        stratum = _string(mapping["stratum"], name=f"source_groups[{index}].stratum")
        role = _string(mapping["role"], name=f"source_groups[{index}].role")
        if role not in _SOURCE_ROLES:
            raise ValueError(f"source_groups[{index}].role is unsupported: {role!r}")
        if group_id in group_ids:
            raise ValueError(f"source_groups repeats group_id {group_id!r}")
        key = (object_id, episode_id)
        if key in object_episodes:
            raise ValueError(f"source_groups repeats object/episode {key!r}")
        group_ids.add(group_id)
        object_episodes.add(key)
        result.append(
            {
                "group_id": group_id,
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "role": role,
            }
        )
    expected = sorted(result, key=lambda record: (record["object_id"], record["episode_id"]))
    if result != expected:
        raise ValueError("source_groups must use canonical object/episode ordering")
    return result


def _target_groups(value: object) -> list[dict[str, Any]]:
    records = _sequence(value, name="forbidden_target_groups", nonempty=True)
    result: list[dict[str, Any]] = []
    object_episodes: set[tuple[str, int]] = set()
    for index, item in enumerate(records):
        mapping = _mapping(item, name=f"forbidden_target_groups[{index}]")
        _exact_fields(mapping, _TARGET_GROUP_FIELDS, name=f"forbidden_target_groups[{index}]")
        object_id = _string(
            mapping["object_id"],
            name=f"forbidden_target_groups[{index}].object_id",
        )
        episode_id = _integer(
            mapping["episode_id"],
            name=f"forbidden_target_groups[{index}].episode_id",
        )
        stratum = _string(
            mapping["stratum"],
            name=f"forbidden_target_groups[{index}].stratum",
        )
        key = (object_id, episode_id)
        if key in object_episodes:
            raise ValueError(f"forbidden_target_groups repeats object/episode {key!r}")
        object_episodes.add(key)
        result.append(
            {"object_id": object_id, "episode_id": episode_id, "stratum": stratum}
        )
    expected = sorted(result, key=lambda record: (record["object_id"], record["episode_id"]))
    if result != expected:
        raise ValueError("forbidden_target_groups must use canonical object/episode ordering")
    return result


def _calibration_identity(
    value: object,
    *,
    name: str,
    expected_relative_path: str,
) -> dict[str, Any]:
    mapping = _mapping(value, name=name)
    _exact_fields(mapping, _CALIBRATION_IDENTITY_FIELDS, name=name)
    relative_path = _relative_path(mapping["relative_path"], name=f"{name}.relative_path")
    if relative_path != expected_relative_path:
        raise ValueError(
            f"{name}.relative_path differs from its group identity: "
            f"expected={expected_relative_path!r}, observed={relative_path!r}"
        )
    return {
        "relative_path": relative_path,
        "sha256": _sha256(mapping["sha256"], name=f"{name}.sha256"),
        "byte_count": _integer(mapping["byte_count"], name=f"{name}.byte_count", minimum=1),
    }


def _calibration_inputs(
    value: object,
    *,
    groups_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = _sequence(value, name="camera_calibration_inputs", nonempty=True)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(records):
        name = f"camera_calibration_inputs[{index}]"
        mapping = _mapping(item, name=name)
        _exact_fields(mapping, _CALIBRATION_FIELDS, name=name)
        group_id = _string(mapping["group_id"], name=f"{name}.group_id")
        if group_id not in groups_by_id:
            raise ValueError(f"{name}.group_id does not name a source group: {group_id!r}")
        if group_id in seen:
            raise ValueError(f"camera_calibration_inputs repeats group_id {group_id!r}")
        seen.add(group_id)
        group = groups_by_id[group_id]
        object_id = _string(mapping["object_id"], name=f"{name}.object_id")
        episode_id = _integer(mapping["episode_id"], name=f"{name}.episode_id")
        if (object_id, episode_id) != (group["object_id"], group["episode_id"]):
            raise ValueError(f"{name} object/episode differs from source group {group_id!r}")
        episode_path = f"{object_id}/episode_{episode_id:04d}"
        result.append(
            {
                "group_id": group_id,
                "object_id": object_id,
                "episode_id": episode_id,
                "intrinsics": _calibration_identity(
                    mapping["intrinsics"],
                    name=f"{name}.intrinsics",
                    expected_relative_path=f"{episode_path}/undistorted_intrinsics.npy",
                ),
                "extrinsics": _calibration_identity(
                    mapping["extrinsics"],
                    name=f"{name}.extrinsics",
                    expected_relative_path=f"{episode_path}/extrinsics.npy",
                ),
            }
        )
    if seen != set(groups_by_id):
        missing = sorted(set(groups_by_id) - seen)
        raise ValueError(f"camera_calibration_inputs misses source groups: {missing}")
    if result != sorted(result, key=lambda record: record["group_id"]):
        raise ValueError("camera_calibration_inputs must use canonical group_id ordering")
    return result


def _support(
    value: object,
    *,
    groups_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    mapping = _mapping(value, name="support")
    _exact_fields(mapping, _SUPPORT_FIELDS, name="support")
    interval = _sequence(
        mapping["required_frame_interval"],
        name="support.required_frame_interval",
    )
    if len(interval) != 2:
        raise ValueError("support.required_frame_interval must contain [start, stop_exclusive]")
    frame_start = _integer(interval[0], name="support.required_frame_interval[0]")
    frame_stop = _integer(
        interval[1],
        name="support.required_frame_interval[1]",
        minimum=1,
    )
    if frame_stop <= frame_start:
        raise ValueError("support.required_frame_interval must be nonempty")

    raw_common = _sequence(
        mapping["common_supported_cameras"],
        name="support.common_supported_cameras",
    )
    common = [
        _string(camera, name=f"support.common_supported_cameras[{index}]")
        for index, camera in enumerate(raw_common)
    ]
    if len(common) != len(set(common)) or common != sorted(common):
        raise ValueError("support.common_supported_cameras must be sorted and unique")
    declared_common_count = _integer(
        mapping["common_supported_camera_count"],
        name="support.common_supported_camera_count",
    )
    if declared_common_count != len(common):
        raise ValueError("support.common_supported_camera_count differs from its camera list")
    minimum_common = _integer(
        mapping["minimum_common_supported_cameras"],
        name="support.minimum_common_supported_cameras",
        minimum=1,
    )

    raw_rows = _sequence(mapping["stream_rows"], name="support.stream_rows", nonempty=True)
    rows: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    supported_by_group: dict[str, set[str]] = {group_id: set() for group_id in groups_by_id}
    for index, item in enumerate(raw_rows):
        name = f"support.stream_rows[{index}]"
        row = _mapping(item, name=name)
        _exact_fields(row, _STREAM_ROW_FIELDS, name=name)
        group_id = _string(row["group_id"], name=f"{name}.group_id")
        camera = _string(row["camera"], name=f"{name}.camera")
        if group_id not in groups_by_id:
            raise ValueError(f"{name}.group_id does not name a source group: {group_id!r}")
        key = (group_id, camera)
        if key in rows_by_key:
            raise ValueError(f"support.stream_rows repeats group/camera {key!r}")
        group = groups_by_id[group_id]
        object_id = _string(row["object_id"], name=f"{name}.object_id")
        episode_id = _integer(row["episode_id"], name=f"{name}.episode_id")
        if (object_id, episode_id) != (group["object_id"], group["episode_id"]):
            raise ValueError(f"{name} object/episode differs from source group {group_id!r}")
        missing_raw = _sequence(
            row["missing_required_members"],
            name=f"{name}.missing_required_members",
        )
        missing = [
            _string(member, name=f"{name}.missing_required_members[{position}]")
            for position, member in enumerate(missing_raw)
        ]
        if len(missing) != len(set(missing)) or missing != sorted(missing):
            raise ValueError(f"{name}.missing_required_members must be sorted and unique")
        if not set(missing).issubset(_REQUIRED_STREAM_MEMBERS):
            raise ValueError(f"{name}.missing_required_members names an unknown member")
        timestamp_count = _integer(
            row["aligned_timestamp_count"],
            name=f"{name}.aligned_timestamp_count",
        )
        required_count = _integer(
            row["required_frame_count"],
            name=f"{name}.required_frame_count",
            minimum=1,
        )
        if required_count != frame_stop:
            raise ValueError(f"{name}.required_frame_count differs from the frozen prefix stop")
        supported = _boolean(row["supported"], name=f"{name}.supported")
        expected_supported = not missing and timestamp_count >= required_count
        if supported is not expected_supported:
            raise ValueError(f"{name}.supported disagrees with missing members and timestamps")
        normalized = {
            "camera": camera,
            "missing_required_members": missing,
            "aligned_timestamp_count": timestamp_count,
            "required_frame_count": required_count,
            "supported": supported,
            "group_id": group_id,
            "object_id": object_id,
            "episode_id": episode_id,
        }
        rows.append(normalized)
        rows_by_key[key] = normalized
        if supported:
            supported_by_group[group_id].add(camera)
    if rows != sorted(rows, key=lambda row: (row["group_id"], row["camera"])):
        raise ValueError("support.stream_rows must use canonical group/camera ordering")
    measured_common = sorted(set.intersection(*supported_by_group.values()))
    if common != measured_common:
        raise ValueError(
            "support.common_supported_cameras differs from the intersection reconstructed "
            "from stream_rows"
        )
    return (
        {
            "required_frame_interval": [frame_start, frame_stop],
            "common_supported_camera_count": declared_common_count,
            "common_supported_cameras": common,
            "minimum_common_supported_cameras": minimum_common,
            "stream_rows": rows,
        },
        rows_by_key,
    )


def _camera_panel(
    value: object,
    *,
    common_cameras: Sequence[str],
) -> dict[str, Any]:
    mapping = _mapping(value, name="camera_panel")
    _exact_fields(mapping, _CAMERA_PANEL_FIELDS, name="camera_panel")
    raw_selected = _sequence(
        mapping["selected_cameras"],
        name="camera_panel.selected_cameras",
        nonempty=True,
    )
    selected = [
        _string(camera, name=f"camera_panel.selected_cameras[{index}]")
        for index, camera in enumerate(raw_selected)
    ]
    if len(selected) != len(set(selected)):
        raise ValueError("camera_panel.selected_cameras must be unique")
    panel_size = _integer(mapping["panel_size"], name="camera_panel.panel_size", minimum=1)
    if len(selected) != panel_size:
        raise ValueError("camera_panel.panel_size differs from selected_cameras")
    if not set(selected).issubset(common_cameras):
        raise ValueError("camera_panel.selected_cameras is not a subset of common support")

    deviations_raw = _mapping(
        mapping["camera_center_maximum_deviation_m"],
        name="camera_panel.camera_center_maximum_deviation_m",
    )
    directions_raw = _mapping(mapping["camera_direction"], name="camera_panel.camera_direction")
    if set(deviations_raw) != set(selected) or set(directions_raw) != set(selected):
        raise ValueError("camera_panel direction/deviation keys must equal selected_cameras")
    deviations = {
        camera: _finite_number(
            deviations_raw[camera],
            name=f"camera_panel.camera_center_maximum_deviation_m[{camera!r}]",
            minimum=0.0,
        )
        for camera in selected
    }
    directions: dict[str, list[float]] = {}
    for camera in selected:
        values = _sequence(
            directions_raw[camera],
            name=f"camera_panel.camera_direction[{camera!r}]",
        )
        if len(values) != 3:
            raise ValueError(f"camera_panel.camera_direction[{camera!r}] must be a 3-vector")
        direction = [
            _finite_number(
                coordinate,
                name=f"camera_panel.camera_direction[{camera!r}][{index}]",
            )
            for index, coordinate in enumerate(values)
        ]
        norm = math.sqrt(sum(coordinate * coordinate for coordinate in direction))
        if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"camera_panel.camera_direction[{camera!r}] must have unit norm")
        directions[camera] = direction
    return {
        "selected_cameras": selected,
        "panel_size": panel_size,
        "selection_rule": _string(
            mapping["selection_rule"],
            name="camera_panel.selection_rule",
        ),
        "first_camera_rule": _string(
            mapping["first_camera_rule"],
            name="camera_panel.first_camera_rule",
        ),
        "camera_center_maximum_deviation_m": deviations,
        "camera_direction": directions,
    }


def _source_case(
    value: object,
    *,
    index: int,
    groups_by_id: Mapping[str, Mapping[str, Any]],
    selected_cameras: set[str],
    frame_stop: int,
    support_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    name = f"source_cases[{index}]"
    mapping = _mapping(value, name=name)
    _exact_fields(mapping, _SOURCE_CASE_FIELDS, name=name)
    group_id = _string(mapping["group_id"], name=f"{name}.group_id")
    camera = _string(mapping["camera"], name=f"{name}.camera")
    if group_id not in groups_by_id:
        raise ValueError(f"{name}.group_id does not name a source group: {group_id!r}")
    if camera not in selected_cameras:
        raise ValueError(f"{name}.camera is outside the selected panel: {camera!r}")
    support = support_rows.get((group_id, camera))
    if support is None or support["supported"] is not True:
        raise ValueError(f"{name} does not have a corresponding supported stream row")
    group = groups_by_id[group_id]
    object_id = _string(mapping["object_id"], name=f"{name}.object_id")
    episode_id = _integer(mapping["episode_id"], name=f"{name}.episode_id")
    if (object_id, episode_id) != (group["object_id"], group["episode_id"]):
        raise ValueError(f"{name} object/episode differs from source group {group_id!r}")
    case_id = _string(mapping["case_id"], name=f"{name}.case_id")
    expected_case_id = f"{group_id}-{camera}"
    if case_id != expected_case_id:
        raise ValueError(f"{name}.case_id must be {expected_case_id!r}")
    episode_path = f"{object_id}/episode_{episode_id:04d}"
    relative_episode_path = _relative_path(
        mapping["relative_episode_path"],
        name=f"{name}.relative_episode_path",
    )
    relative_camera_path = _relative_path(
        mapping["relative_camera_path"],
        name=f"{name}.relative_camera_path",
    )
    if relative_episode_path != episode_path:
        raise ValueError(f"{name}.relative_episode_path differs from object/episode identity")
    if relative_camera_path != f"{episode_path}/{camera}":
        raise ValueError(f"{name}.relative_camera_path differs from object/episode/camera identity")
    aligned_count = _integer(
        mapping["aligned_timestamp_count"],
        name=f"{name}.aligned_timestamp_count",
        minimum=frame_stop,
    )

    sidecar_sha_raw = _mapping(mapping["sidecar_sha256"], name=f"{name}.sidecar_sha256")
    sidecar_count_raw = _mapping(
        mapping["sidecar_byte_count"],
        name=f"{name}.sidecar_byte_count",
    )
    if set(sidecar_sha_raw) != set(_REQUIRED_SIDECARS) or set(sidecar_count_raw) != set(
        _REQUIRED_SIDECARS
    ):
        raise ValueError(f"{name} must bind exactly the three required sidecars")
    sidecar_sha = {
        member: _sha256(
            sidecar_sha_raw[member],
            name=f"{name}.sidecar_sha256[{member!r}]",
        )
        for member in _REQUIRED_SIDECARS
    }
    sidecar_counts = {
        member: _integer(
            sidecar_count_raw[member],
            name=f"{name}.sidecar_byte_count[{member!r}]",
            minimum=1,
        )
        for member in _REQUIRED_SIDECARS
    }
    normalized: dict[str, Any] = {
        "case_id": case_id,
        "group_id": group_id,
        "object_id": object_id,
        "episode_id": episode_id,
        "camera": camera,
        "relative_episode_path": relative_episode_path,
        "relative_camera_path": relative_camera_path,
        "input_video_sha256": _sha256(
            mapping["input_video_sha256"],
            name=f"{name}.input_video_sha256",
        ),
        "input_video_byte_count": _integer(
            mapping["input_video_byte_count"],
            name=f"{name}.input_video_byte_count",
            minimum=1,
        ),
        "aligned_timestamp_count": aligned_count,
        "sidecar_sha256": sidecar_sha,
        "sidecar_byte_count": sidecar_counts,
    }
    declared_case_id = _sha256(mapping["source_case_id"], name=f"{name}.source_case_id")
    measured_case_id = _sha256_json(normalized)
    if declared_case_id != measured_case_id:
        raise ValueError(f"{name}.source_case_id does not match its exact content")
    normalized["source_case_id"] = declared_case_id
    return normalized
