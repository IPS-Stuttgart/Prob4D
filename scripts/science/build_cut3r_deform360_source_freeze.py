#!/usr/bin/env python3
"""Freeze the source-only Deform360 input bundle for the CUT3R comparison.

The command is deliberately outcome blind.  It validates the already frozen
Deform360 object/episode roster, inspects source-only camera calibration and
support metadata, chooses one deterministic geometry-balanced camera panel,
hashes the exact retained source video/checkpoint/wheel bytes, and emits the
existing ``prob4d prediction cut3r-comparison`` specification.

It never decodes source RGB frames, loads predictions or residuals, or touches a
confirmation object.  A support-negative artifact is still retained when fewer
than the preregistered number of common camera streams are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

SOURCE_FREEZE_SCHEMA: Final = "prob4d.cut3r-deform360-source-freeze"
SOURCE_FREEZE_VERSION: Final = 1
SUPPORT_PASS: Final = "source-support-freeze-ready"
SUPPORT_NEGATIVE: Final = "insufficient-common-camera-support"

FloatArray = NDArray[np.floating[Any]]


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
        raise ValueError(f"failed to read {name} from {path}: {error}") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(dict[str, Any], value)


def _strict_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a genuine Boolean")
    return value


def _sha256_string(value: object, *, name: str) -> str:
    result = _strict_string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _revision(value: object, *, name: str) -> str:
    result = _strict_string(value, name=name)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be an exact lowercase 40-character Git revision")
    return result


def _regular_snapshot(path: Path, *, name: str) -> tuple[os.stat_result, Path]:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError(f"failed to stat {name} at {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")
    return metadata, path.resolve(strict=True)


def _sha256_file(path: Path, *, name: str) -> tuple[str, int]:
    before, resolved = _regular_snapshot(path, name=name)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ValueError(f"failed to hash {name} at {path}: {error}") from error
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"{name} changed while it was being hashed: {path}")
    return digest.hexdigest(), before.st_size


def _git_revision(repository: Path, *, name: str) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", os.fspath(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"failed to resolve {name} Git revision: {error}") from error
    return _revision(revision, name=f"{name} revision")


def _publish_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite different retained bytes: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_tuple(value: object, *, name: str) -> tuple[str, int, str]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    mapping = cast(dict[str, Any], value)
    object_id = _strict_string(mapping.get("object_id"), name=f"{name}.object_id")
    episode_id = _strict_integer(mapping.get("episode_id"), name=f"{name}.episode_id")
    stratum = _strict_string(mapping.get("stratum"), name=f"{name}.stratum")
    return object_id, episode_id, stratum


def _selection_roster(selection: Mapping[str, Any], role: str) -> set[tuple[str, int, str]]:
    root = selection.get("selection")
    if type(root) is not dict:
        raise ValueError("selection lock has no canonical selection object")
    records = cast(dict[str, Any], root).get(role)
    if type(records) is not list:
        raise ValueError(f"selection.{role} must be a JSON array")
    result = {
        _record_tuple(record, name=f"selection.{role}[{index}]")
        for index, record in enumerate(records)
    }
    if len(result) != len(records):
        raise ValueError(f"selection.{role} repeats an object/episode record")
    return result


def _protocol_roster(
    protocol: Mapping[str, Any],
    field: str,
    *,
    include_role: bool,
) -> tuple[dict[str, Any], ...]:
    records = protocol.get(field)
    if type(records) is not list or not records:
        raise ValueError(f"protocol.{field} must be a nonempty JSON array")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw_record in enumerate(records):
        if type(raw_record) is not dict:
            raise ValueError(f"protocol.{field}[{index}] must be a JSON object")
        record = cast(dict[str, Any], raw_record)
        object_id, episode_id, stratum = _record_tuple(
            record,
            name=f"protocol.{field}[{index}]",
        )
        key = (object_id, episode_id)
        if key in seen:
            raise ValueError(f"protocol.{field} repeats {key!r}")
        seen.add(key)
        normalized: dict[str, Any] = {
            "object_id": object_id,
            "episode_id": episode_id,
            "stratum": stratum,
        }
        if include_role:
            normalized["group_id"] = _strict_string(
                record.get("group_id"),
                name=f"protocol.{field}[{index}].group_id",
            )
            role = _strict_string(
                record.get("role"),
                name=f"protocol.{field}[{index}].role",
            )
            if role not in {"development", "calibration", "source_evaluation"}:
                raise ValueError(f"unknown source role {role!r}")
            normalized["role"] = role
        result.append(normalized)
    return tuple(sorted(result, key=lambda item: (item["object_id"], item["episode_id"])))


def _validate_protocol_and_selection(
    protocol: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    protocol_path: Path,
    selection_path: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if protocol.get("schema") != "prob4d.cut3r-deform360-source-freeze-protocol":
        raise ValueError("unexpected CUT3R Deform360 source protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported CUT3R Deform360 source protocol version")
    dataset = protocol.get("source_dataset")
    if type(dataset) is not dict:
        raise ValueError("protocol.source_dataset must be a JSON object")
    dataset_mapping = cast(dict[str, Any], dataset)
    measured_selection_sha, _ = _sha256_file(selection_path, name="selection lock")
    expected_selection_sha = _sha256_string(
        dataset_mapping.get("selection_file_sha256"),
        name="source_dataset.selection_file_sha256",
    )
    if measured_selection_sha != expected_selection_sha:
        raise ValueError(
            "selection lock bytes differ from the frozen source protocol: "
            f"expected={expected_selection_sha}, measured={measured_selection_sha}"
        )
    for field in ("selection_artifact_sha256", "selection_sha256"):
        expected = _sha256_string(dataset_mapping.get(field), name=f"source_dataset.{field}")
        if selection.get(field) != expected:
            raise ValueError(f"selection lock {field} differs from the source protocol")

    source_groups = _protocol_roster(protocol, "source_groups", include_role=True)
    target_groups = _protocol_roster(protocol, "forbidden_target_groups", include_role=False)
    source_selection = _selection_roster(selection, "calibration")
    target_selection = _selection_roster(selection, "confirmation")
    source_triplets = {
        (record["object_id"], record["episode_id"], record["stratum"])
        for record in source_groups
    }
    target_triplets = {
        (record["object_id"], record["episode_id"], record["stratum"])
        for record in target_groups
    }
    if source_triplets != source_selection:
        raise ValueError("protocol source roster differs from selection.calibration")
    if target_triplets != target_selection:
        raise ValueError("protocol forbidden target roster differs from selection.confirmation")
    if {(item[0], item[1]) for item in source_triplets}.intersection(
        (item[0], item[1]) for item in target_triplets
    ):
        raise ValueError("source and forbidden target rosters overlap")

    roles = [record["role"] for record in source_groups]
    role_assignment = protocol.get("source_role_assignment")
    if type(role_assignment) is not dict:
        raise ValueError("protocol.source_role_assignment must be a JSON object")
    role_mapping = cast(dict[str, Any], role_assignment)
    expected_counts = {
        "development": _strict_integer(
            role_mapping.get("development_count"),
            name="source_role_assignment.development_count",
            minimum=1,
        ),
        "calibration": _strict_integer(
            role_mapping.get("calibration_count"),
            name="source_role_assignment.calibration_count",
            minimum=1,
        ),
        "source_evaluation": _strict_integer(
            role_mapping.get("source_evaluation_count"),
            name="source_role_assignment.source_evaluation_count",
            minimum=1,
        ),
    }
    measured_counts = {role: roles.count(role) for role in expected_counts}
    if measured_counts != expected_counts:
        raise ValueError(
            f"source role counts differ from the protocol: {measured_counts!r}"
        )
    hash_seed = _strict_string(
        role_mapping.get("hash_seed"),
        name="source_role_assignment.hash_seed",
    )
    allocation = role_mapping.get("allocation_by_rank")
    if type(allocation) is not list or not allocation:
        raise ValueError(
            "source_role_assignment.allocation_by_rank must be a nonempty JSON array"
        )
    normalized_allocation = [
        _strict_string(role, name=f"source_role_assignment.allocation_by_rank[{index}]")
        for index, role in enumerate(allocation)
    ]
    if any(role not in expected_counts for role in normalized_allocation):
        raise ValueError("source_role_assignment.allocation_by_rank names an unknown role")
    records_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for record in source_groups:
        records_by_stratum.setdefault(cast(str, record["stratum"]), []).append(record)
    for stratum, records in records_by_stratum.items():
        if len(records) != len(normalized_allocation):
            raise ValueError(
                f"stratum {stratum!r} has {len(records)} groups but the role allocation "
                f"has {len(normalized_allocation)} ranks"
            )
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(
                (
                    f"{hash_seed}\0{record['object_id']}\0{record['episode_id']}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        for rank, (record, expected_role) in enumerate(
            zip(ranked, normalized_allocation, strict=True),
            start=1,
        ):
            if record["role"] != expected_role:
                raise ValueError(
                    f"source role assignment differs at {stratum} rank {rank}: "
                    f"expected={expected_role!r}, measured={record['role']!r}"
                )
    _sha256_file(protocol_path, name="source protocol")
    return source_groups, target_groups


def _load_numpy_mapping(path: Path, *, name: str) -> dict[str, FloatArray]:
    _regular_snapshot(path, name=name)
    try:
        loaded = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to load trusted official {name}: {error}") from error
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            raw: object = {key: loaded[key] for key in loaded.files}
        elif isinstance(loaded, np.ndarray) and loaded.shape == ():
            raw = loaded.item()
        else:
            raw = loaded
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()
    if type(raw) is not dict:
        raise ValueError(f"{name} must contain a camera-name mapping")
    result: dict[str, FloatArray] = {}
    for raw_key, raw_value in cast(dict[object, object], raw).items():
        camera = _strict_string(raw_key, name=f"{name} camera name")
        array = np.asarray(raw_value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}[{camera!r}] must be finite")
        result[camera] = array
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _camera_center(extrinsic: FloatArray, *, camera: str) -> FloatArray:
    if extrinsic.shape != (4, 4):
        raise ValueError(f"extrinsics[{camera!r}] must have shape (4, 4)")
    if not np.allclose(extrinsic[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7, rtol=0.0):
        raise ValueError(f"extrinsics[{camera!r}] must be homogeneous camera-to-world")
    rotation = extrinsic[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3, rtol=2e-3):
        raise ValueError(f"extrinsics[{camera!r}] rotation is not orthonormal")
    if not 0.995 <= float(np.linalg.det(rotation)) <= 1.005:
        raise ValueError(f"extrinsics[{camera!r}] rotation is not proper")
    return extrinsic[:3, 3].copy()


def _timestamp_count(path: Path) -> int:
    _regular_snapshot(path, name="aligned timestamps")
    count = 0
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    count += 1
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"failed to read aligned timestamps at {path}: {error}") from error
    return count


def _episode_directory(processed_root: Path, record: Mapping[str, Any]) -> Path:
    object_id = cast(str, record["object_id"])
    episode_id = cast(int, record["episode_id"])
    candidate = processed_root / object_id / f"episode_{episode_id:04d}"
    if candidate.is_symlink():
        raise ValueError(f"source episode directory must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"source episode directory is unavailable: {candidate}: {error}"
        ) from error
    if not resolved.is_dir():
        raise ValueError(f"source episode path is not a directory: {candidate}")
    return resolved


def _stream_support(
    episode: Path,
    camera: str,
    *,
    required_frames: int,
) -> tuple[bool, dict[str, Any]]:
    camera_dir = episode / camera
    required = {
        "video": camera_dir / "undistorted.mp4",
        "timestamps": camera_dir / "aligned_timestamps.txt",
        "alignment": camera_dir / "alignment.json",
        "metadata": camera_dir / "metadata.json",
    }
    missing = sorted(
        name
        for name, path in required.items()
        if not path.is_file() or path.is_symlink()
    )
    timestamp_count = 0
    if not missing:
        timestamp_count = _timestamp_count(required["timestamps"])
    supported = not missing and timestamp_count >= required_frames
    return supported, {
        "camera": camera,
        "missing_required_members": missing,
        "aligned_timestamp_count": timestamp_count,
        "required_frame_count": required_frames,
        "supported": supported,
    }


def _mean_direction(
    centers: Mapping[str, Sequence[FloatArray]],
    *,
    maximum_deviation_m: float,
) -> tuple[dict[str, FloatArray], dict[str, float]]:
    representative: dict[str, FloatArray] = {}
    deviations: dict[str, float] = {}
    for camera, values in centers.items():
        stack = np.stack(values, axis=0)
        center = np.mean(stack, axis=0)
        deviation = float(np.max(np.linalg.norm(stack - center[None, :], axis=1)))
        if deviation > maximum_deviation_m:
            raise ValueError(
                f"camera {camera!r} changes by {deviation:.6g} m across source episodes; "
                f"maximum is {maximum_deviation_m:.6g} m"
            )
        representative[camera] = center
        deviations[camera] = deviation
    constellation_center = np.mean(np.stack(list(representative.values()), axis=0), axis=0)
    directions: dict[str, FloatArray] = {}
    for camera, center in representative.items():
        direction = center - constellation_center
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 1e-9:
            raise ValueError(f"camera {camera!r} has no stable constellation direction")
        directions[camera] = direction / norm
    return directions, deviations


def _select_camera_panel(
    directions: Mapping[str, FloatArray],
    *,
    panel_size: int,
    tie_tolerance: float,
) -> tuple[str, ...]:
    if len(directions) < panel_size:
        raise ValueError("not enough camera directions for the requested panel")
    positive_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    first_scores = {
        camera: float(direction @ positive_x)
        for camera, direction in directions.items()
    }
    maximum_first = max(first_scores.values())
    first = min(
        camera
        for camera, score in first_scores.items()
        if maximum_first - score <= tie_tolerance
    )
    selected = [first]
    while len(selected) < panel_size:
        scores: dict[str, float] = {}
        for camera, direction in directions.items():
            if camera in selected:
                continue
            minimum_angle = min(
                math.acos(
                    float(
                        np.clip(
                            direction @ directions[chosen],
                            -1.0,
                            1.0,
                        )
                    )
                )
                for chosen in selected
            )
            scores[camera] = minimum_angle
        maximum = max(scores.values())
        chosen = min(
            camera for camera, score in scores.items() if maximum - score <= tie_tolerance
        )
        selected.append(chosen)
    return tuple(selected)


def _case_record(
    record: Mapping[str, Any],
    *,
    camera: str,
    episode: Path,
    frame_start: int,
    frame_stop: int,
    evaluation_start: int,
    evaluation_stop: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_id = cast(str, record["object_id"])
    episode_id = cast(int, record["episode_id"])
    group_id = cast(str, record["group_id"])
    camera_dir = episode / camera
    paths = {
        "video": camera_dir / "undistorted.mp4",
        "timestamps": camera_dir / "aligned_timestamps.txt",
        "alignment": camera_dir / "alignment.json",
        "metadata": camera_dir / "metadata.json",
    }
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name, path in paths.items():
        digest, size = _sha256_file(path, name=f"{group_id}/{camera}/{name}")
        digests[name] = digest
        sizes[name] = size
    timestamp_count = _timestamp_count(paths["timestamps"])
    if timestamp_count < frame_stop:
        raise ValueError(f"{group_id}/{camera} no longer supports the frozen prefix")
    case_id = f"{group_id}-{camera}"
    comparison = {
        "case_id": case_id,
        "input_video_sha256": digests["video"],
        "input_video_byte_count": sizes["video"],
        "frame_start": frame_start,
        "frame_stop_exclusive": frame_stop,
        "evaluation_frame_start": evaluation_start,
        "evaluation_frame_stop_exclusive": evaluation_stop,
    }
    locator: dict[str, Any] = {
        "case_id": case_id,
        "group_id": group_id,
        "object_id": object_id,
        "episode_id": episode_id,
        "camera": camera,
        "relative_episode_path": f"{object_id}/episode_{episode_id:04d}",
        "relative_camera_path": f"{object_id}/episode_{episode_id:04d}/{camera}",
        "input_video_sha256": digests["video"],
        "input_video_byte_count": sizes["video"],
        "aligned_timestamp_count": timestamp_count,
        "sidecar_sha256": {
            "aligned_timestamps.txt": digests["timestamps"],
            "alignment.json": digests["alignment"],
            "metadata.json": digests["metadata"],
        },
        "sidecar_byte_count": {
            "aligned_timestamps.txt": sizes["timestamps"],
            "alignment.json": sizes["alignment"],
            "metadata.json": sizes["metadata"],
        },
    }
    locator["source_case_id"] = _sha256_json(locator)
    return comparison, locator


def build_source_freeze(
    *,
    repository: Path,
    protocol_path: Path,
    selection_path: Path,
    processed_root: Path,
    cut3r_checkout: Path,
    checkpoint_path: Path,
    prob4d_wheel: Path,
    output_directory: Path,
) -> dict[str, Any]:
    protocol = _load_json_object(protocol_path, name="source protocol")
    selection = _load_json_object(selection_path, name="Deform360 selection lock")
    source_groups, target_groups = _validate_protocol_and_selection(
        protocol,
        selection,
        protocol_path=protocol_path,
        selection_path=selection_path,
    )
    target_ids = {cast(str, record["object_id"]) for record in target_groups}
    processed_resolved = processed_root.resolve(strict=True)
    if any(target_id in processed_resolved.parts for target_id in target_ids):
        raise ValueError("processed source root resolves inside a forbidden target object path")

    provider = cast(dict[str, Any], protocol["provider"])
    expected_cut3r_revision = _revision(provider.get("revision"), name="provider.revision")
    measured_cut3r_revision = _git_revision(cut3r_checkout, name="CUT3R")
    if measured_cut3r_revision != expected_cut3r_revision:
        raise ValueError(
            "CUT3R checkout differs from the frozen provider revision: "
            f"expected={expected_cut3r_revision}, measured={measured_cut3r_revision}"
        )
    checkpoint_name = _strict_string(
        provider.get("checkpoint_filename"),
        name="provider.checkpoint_filename",
    )
    if checkpoint_path.name != checkpoint_name:
        raise ValueError(
            f"checkpoint filename must be {checkpoint_name!r}, got {checkpoint_path.name!r}"
        )
    checkpoint_sha, checkpoint_size = _sha256_file(checkpoint_path, name="CUT3R checkpoint")
    prob4d_revision = _git_revision(repository, name="Prob4D")
    prob4d_wheel_sha, prob4d_wheel_size = _sha256_file(prob4d_wheel, name="Prob4D wheel")
    protocol_sha, protocol_size = _sha256_file(protocol_path, name="source protocol")
    selection_sha, selection_size = _sha256_file(selection_path, name="selection lock")

    windowing = cast(dict[str, Any], protocol["windowing"])
    frame_start = _strict_integer(windowing.get("frame_start"), name="windowing.frame_start")
    frame_stop = _strict_integer(
        windowing.get("frame_stop_exclusive"),
        name="windowing.frame_stop_exclusive",
        minimum=1,
    )
    evaluation_start = _strict_integer(
        windowing.get("evaluation_frame_start"),
        name="windowing.evaluation_frame_start",
    )
    evaluation_stop = _strict_integer(
        windowing.get("evaluation_frame_stop_exclusive"),
        name="windowing.evaluation_frame_stop_exclusive",
        minimum=1,
    )
    if not frame_start <= evaluation_start < evaluation_stop <= frame_stop:
        raise ValueError("evaluation interval must lie inside the source prefix")

    camera_policy = cast(dict[str, Any], protocol["camera_panel"])
    panel_size = _strict_integer(
        camera_policy.get("panel_size"),
        name="camera_panel.panel_size",
        minimum=1,
    )
    minimum_common = _strict_integer(
        camera_policy.get("minimum_common_supported_cameras"),
        name="camera_panel.minimum_common_supported_cameras",
        minimum=panel_size,
    )
    tie_tolerance = float(camera_policy.get("tie_tolerance"))
    if not math.isfinite(tie_tolerance) or not 0.0 <= tie_tolerance < 1.0:
        raise ValueError("camera_panel.tie_tolerance must be finite in [0, 1)")
    maximum_center_deviation_m = float(
        camera_policy.get("maximum_cross_episode_camera_center_deviation_m", 0.02)
    )
    if not math.isfinite(maximum_center_deviation_m) or maximum_center_deviation_m <= 0.0:
        raise ValueError(
            "camera_panel.maximum_cross_episode_camera_center_deviation_m must be positive"
        )

    episode_by_group: dict[str, Path] = {}
    supported_by_group: dict[str, set[str]] = {}
    support_rows: list[dict[str, Any]] = []
    centers_by_camera: dict[str, list[FloatArray]] = {}
    for record in source_groups:
        object_id = cast(str, record["object_id"])
        if object_id in target_ids:
            raise ValueError(f"source record names forbidden target object {object_id!r}")
        episode = _episode_directory(processed_resolved, record)
        if any(target_id in episode.parts for target_id in target_ids):
            raise ValueError(f"source episode resolves through a forbidden target path: {episode}")
        group_id = cast(str, record["group_id"])
        episode_by_group[group_id] = episode
        intrinsics = _load_numpy_mapping(
            episode / "undistorted_intrinsics.npy",
            name=f"{group_id} intrinsics",
        )
        extrinsics = _load_numpy_mapping(
            episode / "extrinsics.npy",
            name=f"{group_id} extrinsics",
        )
        cameras = sorted(set(intrinsics).intersection(extrinsics))
        if not cameras:
            raise ValueError(f"{group_id} has no camera with both intrinsics and extrinsics")
        supported: set[str] = set()
        for camera in cameras:
            intrinsic = intrinsics[camera]
            if intrinsic.shape != (3, 3):
                raise ValueError(f"intrinsics[{camera!r}] must have shape (3, 3)")
            stream_supported, row = _stream_support(
                episode,
                camera,
                required_frames=frame_stop,
            )
            row.update(
                {
                    "group_id": group_id,
                    "object_id": object_id,
                    "episode_id": record["episode_id"],
                }
            )
            support_rows.append(row)
            if stream_supported:
                supported.add(camera)
                centers_by_camera.setdefault(camera, []).append(
                    _camera_center(extrinsics[camera], camera=camera)
                )
        supported_by_group[group_id] = supported

    common_supported = set.intersection(*supported_by_group.values())
    support_rows.sort(key=lambda item: (item["group_id"], item["camera"]))
    decision = SUPPORT_PASS if len(common_supported) >= minimum_common else SUPPORT_NEGATIVE
    freeze_base: dict[str, Any] = {
        "schema": SOURCE_FREEZE_SCHEMA,
        "schema_version": SOURCE_FREEZE_VERSION,
        "protocol_name": _strict_string(protocol.get("protocol_name"), name="protocol_name"),
        "decision": decision,
        "source_protocol": {
            "sha256": protocol_sha,
            "byte_count": protocol_size,
        },
        "deform360_selection": {
            "sha256": selection_sha,
            "byte_count": selection_size,
            "selection_artifact_sha256": selection["selection_artifact_sha256"],
            "selection_sha256": selection["selection_sha256"],
        },
        "provider": {
            "repository": provider["repository"],
            "revision": measured_cut3r_revision,
            "checkpoint_filename": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_byte_count": checkpoint_size,
            "execution_mode": provider["execution_mode"],
            "revisit_count": provider["revisit_count"],
            "global_alignment": provider["global_alignment"],
            "second_pass_allowed": provider["second_pass_allowed"],
        },
        "prob4d": {
            "revision": prob4d_revision,
            "distribution_filename": prob4d_wheel.name,
            "distribution_sha256": prob4d_wheel_sha,
            "distribution_byte_count": prob4d_wheel_size,
        },
        "source_group_count": len(source_groups),
        "source_groups": list(source_groups),
        "forbidden_target_group_count": len(target_groups),
        "forbidden_target_groups": list(target_groups),
        "support": {
            "required_frame_interval": [frame_start, frame_stop],
            "common_supported_camera_count": len(common_supported),
            "common_supported_cameras": sorted(common_supported),
            "minimum_common_supported_cameras": minimum_common,
            "stream_rows": support_rows,
        },
        "camera_panel": None,
        "source_cases": [],
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }

    comparison_spec: dict[str, Any] | None = None
    if decision == SUPPORT_PASS:
        common_centers = {
            camera: centers_by_camera[camera]
            for camera in sorted(common_supported)
            if len(centers_by_camera.get(camera, ())) == len(source_groups)
        }
        if len(common_centers) < minimum_common:
            raise ValueError("common camera support and calibrated-center support disagree")
        directions, deviations = _mean_direction(
            common_centers,
            maximum_deviation_m=maximum_center_deviation_m,
        )
        selected_cameras = _select_camera_panel(
            directions,
            panel_size=panel_size,
            tie_tolerance=tie_tolerance,
        )
        freeze_base["camera_panel"] = {
            "selected_cameras": list(selected_cameras),
            "panel_size": panel_size,
            "selection_rule": camera_policy["selection_rule"],
            "first_camera_rule": camera_policy["first_camera_rule"],
            "camera_center_maximum_deviation_m": {
                camera: deviations[camera] for camera in selected_cameras
            },
            "camera_direction": {
                camera: [float(value) for value in directions[camera]]
                for camera in selected_cameras
            },
        }
        groups: list[dict[str, Any]] = []
        source_cases: list[dict[str, Any]] = []
        roles: dict[str, list[str]] = {
            "development": [],
            "calibration": [],
            "source_evaluation": [],
        }
        for record in source_groups:
            group_id = cast(str, record["group_id"])
            roles[cast(str, record["role"])].append(group_id)
            cases: list[dict[str, Any]] = []
            for camera in selected_cameras:
                comparison_case, locator = _case_record(
                    record,
                    camera=camera,
                    episode=episode_by_group[group_id],
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                    evaluation_start=evaluation_start,
                    evaluation_stop=evaluation_stop,
                )
                cases.append(comparison_case)
                source_cases.append(locator)
            groups.append(
                {
                    "group_id": group_id,
                    "cases": sorted(cases, key=lambda item: item["case_id"]),
                }
            )
        source_cases.sort(key=lambda item: item["case_id"])
        freeze_base["source_cases"] = source_cases
        comparison_spec = {
            "protocol_name": protocol["protocol_name"],
            "provider_revision": measured_cut3r_revision,
            "checkpoint_sha256": checkpoint_sha,
            "prob4d_revision": prob4d_revision,
            "prob4d_distribution_sha256": prob4d_wheel_sha,
            "window_size": windowing["window_size"],
            "overlap": windowing["overlap"],
            "confidence_threshold": provider["confidence_threshold"],
            "storage_dtype": windowing["storage_dtype"],
            "random_seeds": windowing["random_seeds"],
            "groups": sorted(groups, key=lambda item: item["group_id"]),
            "group_roles": {role: sorted(group_ids) for role, group_ids in roles.items()},
            "include_revisit_diagnostic": windowing["include_revisit_diagnostic"],
        }
        freeze_base["comparison_spec_sha256"] = _sha256_json(comparison_spec)

    freeze_base["source_freeze_id"] = _sha256_json(freeze_base)
    output_directory.mkdir(parents=True, exist_ok=True)
    freeze_name = cast(dict[str, Any], protocol["freeze_outputs"])["source_freeze"]
    _publish_json(output_directory / freeze_name, freeze_base)
    if comparison_spec is not None:
        comparison_name = cast(dict[str, Any], protocol["freeze_outputs"])["comparison_spec"]
        _publish_json(output_directory / comparison_name, comparison_spec)
    return freeze_base


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prob4d-wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        freeze = build_source_freeze(
            repository=arguments.repository,
            protocol_path=arguments.protocol,
            selection_path=arguments.selection,
            processed_root=arguments.processed_root,
            cut3r_checkout=arguments.cut3r_checkout,
            checkpoint_path=arguments.checkpoint,
            prob4d_wheel=arguments.prob4d_wheel,
            output_directory=arguments.output_dir,
        )
    except (FileExistsError, ValueError) as error:
        print(f"CUT3R Deform360 source freeze failed: {error}")
        return 2
    print(json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False))
    return 0 if freeze["decision"] == SUPPORT_PASS else 3


if __name__ == "__main__":
    raise SystemExit(main())
