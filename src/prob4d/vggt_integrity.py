"""Integrity-bound metadata for official VGGT prediction exports.

The VGGT baseline is optional and GPU-heavy, but its provenance and cached
prediction products can be validated with NumPy only.  This module defines a
closed versioned run record, content identities for model/sample/run state, and
confined file verification shared by the producer and provider-neutral adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_mapping,
    require_revision,
    require_sha256,
)

VGGT_RUN_SCHEMA: Final = "prob4d.vggt-baseline-run"
VGGT_RUN_VERSION: Final = 2
VGGT_REPRESENTATIONS: Final = ("world_points", "depth_unprojected")
VGGT_OFFICIAL_REPOSITORY: Final = "facebookresearch/vggt"
VGGT_METHOD: Final = "VGGT-1B"

_MODEL_ID_DOMAIN: Final = "prob4d.vggt-model-set.v1"
_SAMPLE_ID_DOMAIN: Final = "prob4d.vggt-sample-run.v1"
_RUN_ID_DOMAIN: Final = "prob4d.vggt-run.v1"

_MEMBER_FIELDS: Final = frozenset(
    {
        "representation",
        "path",
        "sha256",
        "byte_count",
        "point_shape",
        "point_dtype",
        "camera_extrinsics_shape",
        "camera_extrinsics_dtype",
        "camera_intrinsics_shape",
        "camera_intrinsics_dtype",
        "valid_point_count",
        "invalid_point_count",
    }
)
_SAMPLE_FIELDS: Final = frozenset(
    {
        "sample_id",
        "sample_run_id",
        "input_video_sha256",
        "input_video_byte_count",
        "frame_count",
        "representations",
    }
)
_RUN_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "run_id",
        "method",
        "official_repository",
        "vggt_commit",
        "model_set_id",
        "loader_module_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_revision",
        "preprocess_mode",
        "partition_index",
        "partition_count",
        "samples",
        "dataset_root",
        "output_root",
        "elapsed_seconds",
    }
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic finite JSON bytes for an identity record."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def record_id(domain: str, value: Mapping[str, Any]) -> str:
    """Content-address one canonical record under an explicit domain."""

    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one ordinary readable file."""

    digest = hashlib.sha256()
    source = Path(path)
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read {source.name!r}") from error
    return digest.hexdigest()


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_sha256(value, name=name)


def _optional_revision(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_revision(value, name=name)


def checkpoint_identity(
    *,
    checkpoint: object,
    checkpoint_sha256: object,
    checkpoint_revision: object,
) -> dict[str, str]:
    """Validate and normalize one immutable local or remote checkpoint identity."""

    locator = require_exact_string(checkpoint, name="VGGT checkpoint")
    checksum = _optional_sha256(checkpoint_sha256, name="VGGT checkpoint SHA-256")
    revision = _optional_revision(
        checkpoint_revision,
        name="VGGT checkpoint revision",
    )
    if (checksum is None) == (revision is None):
        raise ValueError("exactly one of checkpoint_sha256 or checkpoint_revision is required")
    if checksum is not None:
        return {"kind": "local-file-sha256", "value": checksum}
    assert revision is not None
    return {
        "kind": "remote-revision",
        "locator": locator,
        "value": revision,
    }


def build_model_set_id(
    *,
    vggt_commit: object,
    checkpoint: object,
    checkpoint_sha256: object,
    checkpoint_revision: object,
    preprocess_mode: object,
) -> str:
    """Bind VGGT code, exact checkpoint, and image preprocessing semantics."""

    commit = require_revision(vggt_commit, name="VGGT commit")
    preprocess = require_exact_string(preprocess_mode, name="VGGT preprocess mode")
    if preprocess not in {"crop", "pad"}:
        raise ValueError("VGGT preprocess mode must be 'crop' or 'pad'")
    identity = {
        "vggt_commit": commit,
        "checkpoint": checkpoint_identity(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_revision=checkpoint_revision,
        ),
        "preprocess_mode": preprocess,
    }
    return record_id(_MODEL_ID_DOMAIN, identity)


def safe_relative_path(value: object, *, name: str) -> str:
    """Require a confined portable POSIX-relative path."""

    text = require_exact_string(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return path.as_posix()


def relative_member(path: Path, *, root: Path, name: str) -> str:
    """Return one safe path relative to a declared root."""

    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} must lie inside its declared root") from error
    return safe_relative_path(relative.as_posix(), name=name)


def resolve_member(root: Path, relative_path: object, *, name: str) -> Path:
    """Resolve a safe ordinary member without traversing symbolic links."""

    relative = safe_relative_path(relative_path, name=name)
    root_resolved = root.resolve()
    current = root_resolved
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    resolved = current.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes its declared root") from error
    if not resolved.is_file():
        raise ValueError(f"{name} is missing or is not an ordinary file")
    return resolved


def _shape_record(shape: Sequence[int]) -> list[int]:
    return [int(value) for value in shape]


def _require_shape(value: object, *, name: str, rank: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != rank:
        raise ValueError(f"{name} must be a JSON array of length {rank}")
    result = tuple(
        require_exact_integer(item, name=f"{name}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    return result


def describe_prediction_archive(
    path: str | Path,
    *,
    representation: str,
    relative_path: str,
) -> dict[str, Any]:
    """Validate and describe one official VGGT prediction archive."""

    if representation not in VGGT_REPRESENTATIONS:
        raise ValueError(f"unsupported VGGT representation {representation!r}")
    source = Path(path)
    safe_path = safe_relative_path(relative_path, name="VGGT prediction path")
    expected_fields = {"point_map", "camera_extrinsics", "camera_intrinsics"}
    try:
        with np.load(source, allow_pickle=False) as archive:
            fields = set(archive.files)
            if fields != expected_fields:
                raise ValueError(
                    "VGGT prediction archive fields changed; "
                    f"missing={sorted(expected_fields - fields)}, "
                    f"extra={sorted(fields - expected_fields)}"
                )
            point_map = np.asarray(archive["point_map"])
            extrinsics = np.asarray(archive["camera_extrinsics"])
            intrinsics = np.asarray(archive["camera_intrinsics"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("VGGT prediction"):
            raise
        raise ValueError(f"cannot load VGGT prediction archive {source.name!r}") from error

    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError("VGGT point_map must have shape (T, H, W, 3)")
    frame_count = int(point_map.shape[0])
    if extrinsics.shape != (frame_count, 3, 4):
        raise ValueError("VGGT camera_extrinsics must have shape (T, 3, 4)")
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError("VGGT camera_intrinsics must have shape (T, 3, 3)")
    if point_map.dtype.kind != "f":
        raise ValueError("VGGT point_map must use a floating dtype")
    if extrinsics.dtype.kind != "f" or intrinsics.dtype.kind != "f":
        raise ValueError("VGGT camera matrices must use floating dtypes")
    if not np.all(np.isfinite(extrinsics)) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("VGGT camera matrices must be finite")
    valid = np.all(np.isfinite(point_map), axis=-1)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("VGGT prediction contains no finite 3-D point")
    total = int(valid.size)
    return {
        "representation": representation,
        "path": safe_path,
        "sha256": file_sha256(source),
        "byte_count": int(source.stat().st_size),
        "point_shape": _shape_record(point_map.shape),
        "point_dtype": str(point_map.dtype),
        "camera_extrinsics_shape": _shape_record(extrinsics.shape),
        "camera_extrinsics_dtype": str(extrinsics.dtype),
        "camera_intrinsics_shape": _shape_record(intrinsics.shape),
        "camera_intrinsics_dtype": str(intrinsics.dtype),
        "valid_point_count": valid_count,
        "invalid_point_count": total - valid_count,
    }


def member_identity_record(member: Mapping[str, Any]) -> dict[str, Any]:
    """Return path-independent prediction-member identity fields."""

    return {key: member[key] for key in sorted(_MEMBER_FIELDS - {"path"})}


def _validated_member(value: object) -> dict[str, Any]:
    mapping = require_mapping(value, name="VGGT representation descriptor")
    require_exact_fields(mapping, _MEMBER_FIELDS, name="VGGT representation descriptor")
    representation = require_exact_string(
        mapping["representation"],
        name="VGGT representation",
    )
    if representation not in VGGT_REPRESENTATIONS:
        raise ValueError(f"unsupported VGGT representation {representation!r}")
    result: dict[str, Any] = {
        "representation": representation,
        "path": safe_relative_path(mapping["path"], name="VGGT prediction path"),
        "sha256": require_sha256(mapping["sha256"], name="VGGT prediction SHA-256"),
        "byte_count": require_exact_integer(
            mapping["byte_count"],
            name="VGGT prediction byte_count",
            minimum=1,
        ),
        "point_shape": list(_require_shape(mapping["point_shape"], name="point_shape", rank=4)),
        "point_dtype": require_exact_string(mapping["point_dtype"], name="point_dtype"),
        "camera_extrinsics_shape": list(
            _require_shape(
                mapping["camera_extrinsics_shape"],
                name="camera_extrinsics_shape",
                rank=3,
            )
        ),
        "camera_extrinsics_dtype": require_exact_string(
            mapping["camera_extrinsics_dtype"],
            name="camera_extrinsics_dtype",
        ),
        "camera_intrinsics_shape": list(
            _require_shape(
                mapping["camera_intrinsics_shape"],
                name="camera_intrinsics_shape",
                rank=3,
            )
        ),
        "camera_intrinsics_dtype": require_exact_string(
            mapping["camera_intrinsics_dtype"],
            name="camera_intrinsics_dtype",
        ),
        "valid_point_count": require_exact_integer(
            mapping["valid_point_count"],
            name="valid_point_count",
            minimum=1,
        ),
        "invalid_point_count": require_exact_integer(
            mapping["invalid_point_count"],
            name="invalid_point_count",
            minimum=0,
        ),
    }
    point_shape = tuple(result["point_shape"])
    frame_count = point_shape[0]
    if tuple(result["camera_extrinsics_shape"]) != (frame_count, 3, 4):
        raise ValueError("camera_extrinsics_shape disagrees with point_shape")
    if tuple(result["camera_intrinsics_shape"]) != (frame_count, 3, 3):
        raise ValueError("camera_intrinsics_shape disagrees with point_shape")
    point_count = int(np.prod(point_shape[:-1]))
    if result["valid_point_count"] + result["invalid_point_count"] != point_count:
        raise ValueError("VGGT valid/invalid point counts disagree with point_shape")
    return result


def sample_identity_record(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable identity record for one dataset sample."""

    representations = sample["representations"]
    assert isinstance(representations, list)
    return {
        "sample_id": sample["sample_id"],
        "input_video_sha256": sample["input_video_sha256"],
        "input_video_byte_count": sample["input_video_byte_count"],
        "frame_count": sample["frame_count"],
        "representations": [member_identity_record(member) for member in representations],
    }


def build_sample_record(
    *,
    sample_id: str,
    input_video_path: str | Path,
    representations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one path-confined, content-addressed sample record."""

    normalized_sample_id = safe_relative_path(sample_id, name="VGGT sample_id")
    video = Path(input_video_path)
    if not video.is_file() or video.is_symlink():
        raise ValueError("VGGT input video must be one ordinary file")
    members = [_validated_member(member) for member in representations]
    members.sort(key=lambda item: str(item["representation"]))
    names = [str(item["representation"]) for item in members]
    if not members or len(set(names)) != len(names):
        raise ValueError("VGGT sample representations must be nonempty and unique")
    frame_counts = {int(item["point_shape"][0]) for item in members}
    if len(frame_counts) != 1:
        raise ValueError("VGGT sample representations disagree on frame count")
    record: dict[str, Any] = {
        "sample_id": normalized_sample_id,
        "sample_run_id": "",
        "input_video_sha256": file_sha256(video),
        "input_video_byte_count": int(video.stat().st_size),
        "frame_count": frame_counts.pop(),
        "representations": members,
    }
    record["sample_run_id"] = record_id(
        _SAMPLE_ID_DOMAIN,
        sample_identity_record(record),
    )
    return record


def _validated_sample(value: object) -> dict[str, Any]:
    mapping = require_mapping(value, name="VGGT sample record")
    require_exact_fields(mapping, _SAMPLE_FIELDS, name="VGGT sample record")
    raw_members = mapping["representations"]
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("VGGT sample representations must be a nonempty JSON array")
    members = [_validated_member(member) for member in raw_members]
    names = [str(member["representation"]) for member in members]
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("VGGT sample representations must be sorted and unique")
    frame_count = require_exact_integer(
        mapping["frame_count"],
        name="VGGT frame_count",
        minimum=1,
    )
    if any(int(member["point_shape"][0]) != frame_count for member in members):
        raise ValueError("VGGT member frame count disagrees with sample record")
    result: dict[str, Any] = {
        "sample_id": safe_relative_path(mapping["sample_id"], name="VGGT sample_id"),
        "sample_run_id": require_sha256(
            mapping["sample_run_id"],
            name="VGGT sample_run_id",
        ),
        "input_video_sha256": require_sha256(
            mapping["input_video_sha256"],
            name="VGGT input-video SHA-256",
        ),
        "input_video_byte_count": require_exact_integer(
            mapping["input_video_byte_count"],
            name="VGGT input-video byte_count",
            minimum=1,
        ),
        "frame_count": frame_count,
        "representations": members,
    }
    expected = record_id(_SAMPLE_ID_DOMAIN, sample_identity_record(result))
    if result["sample_run_id"] != expected:
        raise ValueError("VGGT sample_run_id mismatch")
    return result


def run_identity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable, path- and timing-independent run identity record."""

    samples = record["samples"]
    assert isinstance(samples, list)
    return {
        "method": record["method"],
        "official_repository": record["official_repository"],
        "vggt_commit": record["vggt_commit"],
        "model_set_id": record["model_set_id"],
        "loader_module_sha256": record["loader_module_sha256"],
        "preprocess_mode": record["preprocess_mode"],
        "partition_index": record["partition_index"],
        "partition_count": record["partition_count"],
        "samples": [sample_identity_record(sample) for sample in samples],
    }


def build_run_record(
    *,
    vggt_commit: str,
    loader_module_sha256: str,
    checkpoint: str,
    checkpoint_sha256: str | None,
    checkpoint_revision: str | None,
    preprocess_mode: str,
    partition_index: int,
    partition_count: int,
    samples: Sequence[Mapping[str, Any]],
    dataset_root: str | Path,
    output_root: str | Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Create a closed version-2 VGGT run record."""

    commit = require_revision(vggt_commit, name="VGGT commit")
    loader_id = require_sha256(
        loader_module_sha256,
        name="VGGT loader-module SHA-256",
    )
    preprocess = require_exact_string(preprocess_mode, name="VGGT preprocess mode")
    if preprocess not in {"crop", "pad"}:
        raise ValueError("VGGT preprocess mode must be 'crop' or 'pad'")
    index = require_exact_integer(partition_index, name="partition_index", minimum=0)
    count = require_exact_integer(partition_count, name="partition_count", minimum=1)
    if index >= count:
        raise ValueError("partition_index must be smaller than partition_count")
    normalized_samples = [_validated_sample(sample) for sample in samples]
    normalized_samples.sort(key=lambda item: str(item["sample_id"]))
    sample_ids = [str(item["sample_id"]) for item in normalized_samples]
    if not normalized_samples or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("VGGT run samples must be nonempty and unique")
    if type(elapsed_seconds) not in {int, float}:
        raise ValueError("elapsed_seconds must be one finite JSON number")
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    checkpoint_name = require_exact_string(checkpoint, name="VGGT checkpoint")
    checksum = _optional_sha256(
        checkpoint_sha256,
        name="VGGT checkpoint SHA-256",
    )
    revision = _optional_revision(
        checkpoint_revision,
        name="VGGT checkpoint revision",
    )
    model_set_id = build_model_set_id(
        vggt_commit=commit,
        checkpoint=checkpoint_name,
        checkpoint_sha256=checksum,
        checkpoint_revision=revision,
        preprocess_mode=preprocess,
    )
    record: dict[str, Any] = {
        "schema": VGGT_RUN_SCHEMA,
        "schema_version": VGGT_RUN_VERSION,
        "run_id": "",
        "method": VGGT_METHOD,
        "official_repository": VGGT_OFFICIAL_REPOSITORY,
        "vggt_commit": commit,
        "model_set_id": model_set_id,
        "loader_module_sha256": loader_id,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checksum,
        "checkpoint_revision": revision,
        "preprocess_mode": preprocess,
        "partition_index": index,
        "partition_count": count,
        "samples": normalized_samples,
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "elapsed_seconds": elapsed,
    }
    record["run_id"] = record_id(_RUN_ID_DOMAIN, run_identity_record(record))
    return record


def validate_run_record(value: object) -> dict[str, Any]:
    """Strictly validate and normalize one VGGT run record."""

    mapping = require_mapping(value, name="VGGT run metadata")
    require_exact_fields(mapping, _RUN_FIELDS, name="VGGT run metadata")
    schema = require_exact_string(mapping["schema"], name="VGGT run schema")
    version = require_exact_integer(
        mapping["schema_version"],
        name="VGGT run schema_version",
        minimum=1,
    )
    if schema != VGGT_RUN_SCHEMA:
        raise ValueError("unsupported VGGT run metadata schema")
    if version != VGGT_RUN_VERSION:
        raise ValueError("unsupported VGGT run metadata version")
    method = require_exact_string(mapping["method"], name="VGGT method")
    repository = require_exact_string(
        mapping["official_repository"],
        name="VGGT official repository",
    )
    if method != VGGT_METHOD or repository != VGGT_OFFICIAL_REPOSITORY:
        raise ValueError("VGGT method or official repository identity changed")
    raw_samples = mapping["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("VGGT run samples must be a nonempty JSON array")
    samples = [_validated_sample(sample) for sample in raw_samples]
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("VGGT run samples must be sorted and unique")
    index = require_exact_integer(
        mapping["partition_index"],
        name="partition_index",
        minimum=0,
    )
    count = require_exact_integer(
        mapping["partition_count"],
        name="partition_count",
        minimum=1,
    )
    if index >= count:
        raise ValueError("partition_index must be smaller than partition_count")
    if type(mapping["elapsed_seconds"]) not in {int, float}:
        raise ValueError("elapsed_seconds must be one finite JSON number")
    elapsed = float(mapping["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    commit = require_revision(mapping["vggt_commit"], name="VGGT commit")
    checkpoint_name = require_exact_string(mapping["checkpoint"], name="VGGT checkpoint")
    checksum = _optional_sha256(
        mapping["checkpoint_sha256"],
        name="VGGT checkpoint SHA-256",
    )
    revision = _optional_revision(
        mapping["checkpoint_revision"],
        name="VGGT checkpoint revision",
    )
    preprocess = require_exact_string(
        mapping["preprocess_mode"],
        name="VGGT preprocess mode",
    )
    expected_model_set = build_model_set_id(
        vggt_commit=commit,
        checkpoint=checkpoint_name,
        checkpoint_sha256=checksum,
        checkpoint_revision=revision,
        preprocess_mode=preprocess,
    )
    result: dict[str, Any] = {
        "schema": VGGT_RUN_SCHEMA,
        "schema_version": VGGT_RUN_VERSION,
        "run_id": require_sha256(mapping["run_id"], name="VGGT run_id"),
        "method": method,
        "official_repository": repository,
        "vggt_commit": commit,
        "model_set_id": require_sha256(
            mapping["model_set_id"],
            name="VGGT model_set_id",
        ),
        "loader_module_sha256": require_sha256(
            mapping["loader_module_sha256"],
            name="VGGT loader-module SHA-256",
        ),
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checksum,
        "checkpoint_revision": revision,
        "preprocess_mode": preprocess,
        "partition_index": index,
        "partition_count": count,
        "samples": samples,
        "dataset_root": require_exact_string(
            mapping["dataset_root"],
            name="VGGT dataset_root",
        ),
        "output_root": require_exact_string(
            mapping["output_root"],
            name="VGGT output_root",
        ),
        "elapsed_seconds": elapsed,
    }
    if result["model_set_id"] != expected_model_set:
        raise ValueError("VGGT model_set_id mismatch")
    expected_run = record_id(_RUN_ID_DOMAIN, run_identity_record(result))
    if result["run_id"] != expected_run:
        raise ValueError("VGGT run_id mismatch")
    return result


def load_vggt_run_metadata(path: str | Path) -> dict[str, Any]:
    """Load strict, duplicate-free version-2 VGGT run metadata."""

    record = load_json_object(path, name="VGGT run metadata")
    if record.get("schema") != VGGT_RUN_SCHEMA:
        raise ValueError(
            "legacy or unpinned VGGT metadata cannot enter the provider-neutral "
            "boundary; rerun with an exact checkpoint revision"
        )
    return validate_run_record(record)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def save_vggt_run_metadata(
    path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    """Persist one run record atomically and refuse identity-changing overwrite."""

    destination = Path(path)
    normalized = validate_run_record(record)
    content = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        existing = load_vggt_run_metadata(destination)
        if existing["run_id"] != normalized["run_id"]:
            raise ValueError("refusing to replace different VGGT run metadata")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def find_sample_record(
    run_record: Mapping[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    """Return one exact sample record from validated run metadata."""

    normalized_id = safe_relative_path(sample_id, name="VGGT sample_id")
    samples = run_record["samples"]
    assert isinstance(samples, list)
    matches = [sample for sample in samples if sample["sample_id"] == normalized_id]
    if len(matches) != 1:
        raise ValueError(f"VGGT run metadata does not contain sample {normalized_id!r}")
    return dict(matches[0])


def verify_sample_files(
    *,
    sample: Mapping[str, Any],
    dataset_root: str | Path,
    output_root: str | Path,
    representations: Sequence[str],
) -> tuple[Path, dict[str, Path]]:
    """Verify exact input-video and cached-prediction bytes for one sample."""

    selected = tuple(representations)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected VGGT representations must be nonempty and unique")
    if any(name not in VGGT_REPRESENTATIONS for name in selected):
        raise ValueError("selected VGGT representation is unsupported")
    video = resolve_member(
        Path(dataset_root),
        sample["sample_id"],
        name="VGGT input-video path",
    )
    if int(video.stat().st_size) != sample["input_video_byte_count"]:
        raise ValueError("VGGT input-video byte count mismatch")
    if file_sha256(video) != sample["input_video_sha256"]:
        raise ValueError("VGGT input-video SHA-256 mismatch")
    members = sample["representations"]
    assert isinstance(members, list)
    by_name = {str(member["representation"]): member for member in members}
    resolved: dict[str, Path] = {}
    for representation in selected:
        if representation not in by_name:
            raise ValueError(f"VGGT sample lacks representation {representation!r}")
        member = by_name[representation]
        path = resolve_member(
            Path(output_root),
            member["path"],
            name=f"VGGT {representation} prediction path",
        )
        if int(path.stat().st_size) != member["byte_count"]:
            raise ValueError(f"VGGT {representation} byte count mismatch")
        if file_sha256(path) != member["sha256"]:
            raise ValueError(f"VGGT {representation} SHA-256 mismatch")
        described = describe_prediction_archive(
            path,
            representation=representation,
            relative_path=str(member["path"]),
        )
        if member_identity_record(described) != member_identity_record(member):
            raise ValueError(f"VGGT {representation} archive semantics changed")
        resolved[representation] = path
    return video, resolved


__all__ = [
    "VGGT_METHOD",
    "VGGT_OFFICIAL_REPOSITORY",
    "VGGT_REPRESENTATIONS",
    "VGGT_RUN_SCHEMA",
    "VGGT_RUN_VERSION",
    "build_model_set_id",
    "build_run_record",
    "build_sample_record",
    "canonical_json_bytes",
    "checkpoint_identity",
    "describe_prediction_archive",
    "file_sha256",
    "find_sample_record",
    "load_vggt_run_metadata",
    "member_identity_record",
    "record_id",
    "relative_member",
    "resolve_member",
    "run_identity_record",
    "safe_relative_path",
    "sample_identity_record",
    "save_vggt_run_metadata",
    "validate_run_record",
    "verify_sample_files",
]
