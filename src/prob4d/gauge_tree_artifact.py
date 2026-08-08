"""Portable sparse artifacts for causal square-root gauge-tree priors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
from numpy.lib.npyio import NpzFile

from ._gauge_tree_common import canonical_json_sha256
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1

GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA: Final = "prob4d.gauge-tree-square-root-prior-artifact"
GAUGE_TREE_PRIOR_ARTIFACT_VERSION: Final = 1
GAUGE_TREE_PRIOR_ARTIFACT_STORAGE: Final = "compressed-npz-v1"

_PARENT_KEY: Final = "parent_indices"
_TRANSITION_KEY: Final = "transition_matrices"
_INNOVATION_KEY: Final = "innovation_scale_tril"
_ARRAY_KEYS: Final = (_PARENT_KEY, _TRANSITION_KEY, _INNOVATION_KEY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid sparse gauge-tree artifact manifest") from error
    if not isinstance(value, dict):
        raise ValueError("sparse gauge-tree artifact manifest must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} has invalid keys: missing={missing}, unexpected={unexpected}"
        )


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    text = value
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _require_exact_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a genuine integer")
    return value


def _artifact_id(prior_record: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "schema": GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
            "version": GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
            "storage": GAUGE_TREE_PRIOR_ARTIFACT_STORAGE,
            "array_keys": list(_ARRAY_KEYS),
            "prior": dict(prior_record),
        }
    )


def _portable_payload_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("sparse gauge-tree payload path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("sparse gauge-tree payload path must stay below the manifest directory")
    return path


def _resolve_payload(manifest: Path, value: Any) -> Path:
    relative = _portable_payload_relative_path(value)
    root = manifest.parent.resolve()
    payload = manifest.parent.joinpath(*relative.parts)
    resolved = payload.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "sparse gauge-tree payload resolves outside the manifest directory"
        ) from error
    cursor = manifest.parent
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("sparse gauge-tree artifact paths must not contain symlinks")
    return payload


def _payload_relative_path(manifest: Path, payload: Path) -> str:
    root = manifest.parent.resolve()
    resolved = payload.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "sparse gauge-tree payload must be stored below the manifest directory"
        ) from error
    if resolved == manifest.resolve():
        raise ValueError("sparse gauge-tree manifest and payload paths must differ")
    return PurePosixPath(relative.as_posix()).as_posix()


def _link_complete_temporary(temporary: Path, path: Path) -> None:
    os.link(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_npz(path: Path, prior: GaugeTreeSquareRootPriorV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sparse gauge-tree payload: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                parent_indices=prior.parent_indices,
                transition_matrices=prior.transition_matrices,
                innovation_scale_tril=prior.innovation_scale_tril,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _link_complete_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sparse gauge-tree manifest: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _link_complete_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_payload(
    payload: Path,
    *,
    gauge_ids: tuple[str, ...],
    source_joint_covariance_sha256: str | None,
    representation_semantics: str,
) -> GaugeTreeSquareRootPriorV1:
    if not payload.is_file():
        raise ValueError("sparse gauge-tree payload does not exist")
    if payload.is_symlink():
        raise ValueError("sparse gauge-tree payload must not be a symbolic link")
    try:
        archive = np.load(payload, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("invalid sparse gauge-tree payload") from error
    if not isinstance(archive, NpzFile):
        raise ValueError("sparse gauge-tree payload must be an NPZ archive")
    with archive as arrays:
        if tuple(arrays.files) != _ARRAY_KEYS:
            raise ValueError("sparse gauge-tree payload contains unexpected arrays")
        try:
            parents = arrays[_PARENT_KEY]
            transitions = arrays[_TRANSITION_KEY]
            innovation_scale = arrays[_INNOVATION_KEY]
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError("invalid sparse gauge-tree payload") from error
        return GaugeTreeSquareRootPriorV1(
            gauge_ids=gauge_ids,
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=innovation_scale,
            source_joint_covariance_sha256=source_joint_covariance_sha256,
            representation_semantics=representation_semantics,
        )


def _prior_from_record(
    payload: Path,
    prior_record: Mapping[str, Any],
) -> GaugeTreeSquareRootPriorV1:
    required_prior_keys = {
        "schema",
        "version",
        "representation_semantics",
        "gauge_dimension",
        "gauge_ids",
        "parent_indices",
        "transition_matrices",
        "innovation_scale_tril",
        "source_joint_covariance_sha256",
        "prior_id",
        "gauge_count",
        "factor_storage_nbytes",
        "dense_covariance_nbytes",
    }
    _require_exact_keys(prior_record, required_prior_keys, name="sparse gauge-tree prior record")
    raw_gauge_ids = prior_record["gauge_ids"]
    if not isinstance(raw_gauge_ids, list) or any(
        not isinstance(value, str) for value in raw_gauge_ids
    ):
        raise ValueError("sparse gauge-tree gauge_ids must be a list of strings")
    source_digest = prior_record["source_joint_covariance_sha256"]
    if source_digest is not None:
        source_digest = _require_sha256(
            source_digest,
            name="source_joint_covariance_sha256",
        )
    prior = _load_payload(
        payload,
        gauge_ids=tuple(raw_gauge_ids),
        source_joint_covariance_sha256=source_digest,
        representation_semantics=str(prior_record["representation_semantics"]),
    )
    if canonical_json_sha256(prior.to_dict()) != canonical_json_sha256(dict(prior_record)):
        raise ValueError("sparse gauge-tree prior record does not match the payload")
    return prior


def load_gauge_tree_prior_artifact(
    manifest_path: str | Path,
    *,
    expected_prior_id: str | None = None,
) -> GaugeTreeSquareRootPriorV1:
    """Load and fully validate one portable sparse gauge-tree prior artifact."""

    manifest = Path(manifest_path)
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("sparse gauge-tree manifest must be a regular file")
    record = _load_json(manifest)
    _require_exact_keys(
        record,
        {"schema", "version", "prior", "payload", "artifact_id"},
        name="sparse gauge-tree artifact manifest",
    )
    if record["schema"] != GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA:
        raise ValueError("unsupported sparse gauge-tree artifact schema")
    version = _require_exact_int(record["version"], name="sparse gauge-tree artifact version")
    if version != GAUGE_TREE_PRIOR_ARTIFACT_VERSION:
        raise ValueError("unsupported sparse gauge-tree artifact version")
    prior_record = record["prior"]
    payload_record = record["payload"]
    if not isinstance(prior_record, dict) or not isinstance(payload_record, dict):
        raise ValueError("sparse gauge-tree prior and payload records must be objects")
    _require_exact_keys(
        payload_record,
        {"path", "sha256", "allow_pickle", "storage", "array_keys"},
        name="sparse gauge-tree payload record",
    )
    if payload_record["allow_pickle"] is not False:
        raise ValueError("sparse gauge-tree payload must declare allow_pickle=false")
    if payload_record["storage"] != GAUGE_TREE_PRIOR_ARTIFACT_STORAGE:
        raise ValueError("unsupported sparse gauge-tree payload storage")
    if payload_record["array_keys"] != list(_ARRAY_KEYS):
        raise ValueError("sparse gauge-tree payload array keys changed")
    payload = _resolve_payload(manifest, payload_record["path"])
    if not payload.is_file() or payload.is_symlink():
        raise ValueError("sparse gauge-tree payload must be a regular file")
    expected_payload_sha256 = _require_sha256(
        payload_record["sha256"],
        name="sparse gauge-tree payload sha256",
    )
    if _sha256(payload) != expected_payload_sha256:
        raise ValueError("sparse gauge-tree payload checksum mismatch")
    prior = _prior_from_record(payload, prior_record)
    artifact_id = _require_sha256(record["artifact_id"], name="sparse gauge-tree artifact_id")
    if artifact_id != _artifact_id(prior_record):
        raise ValueError("sparse gauge-tree artifact_id does not match its content")
    if expected_prior_id is not None:
        expected = _require_sha256(expected_prior_id, name="expected_prior_id")
        if prior.prior_id != expected:
            raise ValueError("sparse gauge-tree prior_id differs from the expected identity")
    return prior


def save_gauge_tree_prior_artifact(
    manifest_path: str | Path,
    prior: GaugeTreeSquareRootPriorV1,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Publish a portable sparse prior without replacing any different artifact."""

    if not isinstance(prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("prior must be a GaugeTreeSquareRootPriorV1")
    manifest = Path(manifest_path)
    payload = Path(payload_path) if payload_path is not None else manifest.with_suffix(".npz")
    relative_payload = _payload_relative_path(manifest, payload)

    if manifest.exists():
        existing = load_gauge_tree_prior_artifact(manifest)
        if existing.prior_id != prior.prior_id:
            raise FileExistsError(
                "refusing to overwrite a different sparse gauge-tree artifact"
            )
        record = _load_json(manifest)
        existing_payload = _resolve_payload(manifest, record["payload"]["path"])
        if existing_payload.resolve() != payload.resolve():
            raise FileExistsError("existing sparse gauge-tree artifact uses another payload path")
        return manifest, existing_payload

    if payload.exists():
        existing = _load_payload(
            payload,
            gauge_ids=prior.gauge_ids,
            source_joint_covariance_sha256=prior.source_joint_covariance_sha256,
            representation_semantics=prior.representation_semantics,
        )
        if existing.prior_id != prior.prior_id:
            raise FileExistsError(
                "refusing to reuse a different sparse gauge-tree payload"
            )
    else:
        _atomic_write_npz(payload, prior)

    prior_record = prior.to_dict()
    record = {
        "schema": GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
        "version": GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
        "prior": prior_record,
        "payload": {
            "path": relative_payload,
            "sha256": _sha256(payload),
            "allow_pickle": False,
            "storage": GAUGE_TREE_PRIOR_ARTIFACT_STORAGE,
            "array_keys": list(_ARRAY_KEYS),
        },
        "artifact_id": _artifact_id(prior_record),
    }
    _atomic_write_text(
        manifest,
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    loaded = load_gauge_tree_prior_artifact(manifest, expected_prior_id=prior.prior_id)
    if loaded.prior_id != prior.prior_id:
        raise RuntimeError("published sparse gauge-tree artifact failed identity replay")
    return manifest, payload


__all__ = [
    "GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA",
    "GAUGE_TREE_PRIOR_ARTIFACT_STORAGE",
    "GAUGE_TREE_PRIOR_ARTIFACT_VERSION",
    "load_gauge_tree_prior_artifact",
    "save_gauge_tree_prior_artifact",
]
