"""Fail-closed persistence for portable sparse gauge-tree prior artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._gauge_tree_artifact_common import (
    MEMBER_NAMES,
    GaugeTreePriorArrayMemberV1,
    GaugeTreePriorArtifactV1,
    canonical_json_bytes,
    require_mapping,
)
from ._gauge_tree_common import canonical_array_descriptor
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not permitted")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(f"artifact path crosses symbolic link {candidate}")


def _read_stable_bytes(path: Path, *, name: str) -> bytes:
    _reject_symlink_components(path)
    try:
        before = os.stat(path, follow_symlinks=False)
        payload = path.read_bytes()
        after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{name} is unreadable") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise ValueError(f"{name} changed while it was being read")
    return payload


def _write_create_if_absent(path: Path, payload: bytes, *, name: str) -> None:
    _reject_symlink_components(path.parent)
    if path.is_symlink():
        raise ValueError(f"{name} destination must not be a symbolic link")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if _read_stable_bytes(path, name=name) != payload:
            raise FileExistsError(f"refusing to replace different {name}: {path}")
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _npy_payload(
    value: object,
    *,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, bytes]:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return array, buffer.getvalue()


def _member_from_array(
    name: str,
    array: np.ndarray,
    payload: bytes,
) -> GaugeTreePriorArrayMemberV1:
    file_digest = hashlib.sha256(payload).hexdigest()
    descriptor = canonical_array_descriptor(array)
    return GaugeTreePriorArrayMemberV1(
        path=f"gauge-tree-prior-{name}-{file_digest}.npy",
        byte_count=len(payload),
        file_sha256=file_digest,
        dtype=str(descriptor["dtype"]),
        shape=tuple(
            int(value)
            for value in cast(list[int], descriptor["shape"])
        ),
        content_sha256=str(descriptor["sha256"]),
    )


def _load_member(
    manifest_path: Path,
    member: GaugeTreePriorArrayMemberV1,
    *,
    name: str,
) -> np.ndarray:
    path = manifest_path.parent / member.path
    payload = _read_stable_bytes(path, name=f"{name} payload")
    if len(payload) != member.byte_count:
        raise ValueError(f"{name} payload byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != member.file_sha256:
        raise ValueError(f"{name} payload SHA-256 mismatch")
    buffer = io.BytesIO(payload)
    try:
        array = np.load(buffer, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{name} payload is not a valid non-pickled NPY array"
        ) from error
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ValueError(f"{name} payload must be a non-object NumPy array")
    if buffer.read(1) != b"":
        raise ValueError(f"{name} payload contains trailing bytes")
    descriptor = canonical_array_descriptor(array)
    if str(descriptor["dtype"]) != member.dtype:
        raise ValueError(f"{name} payload dtype mismatch")
    descriptor_shape = tuple(
        int(value) for value in cast(list[int], descriptor["shape"])
    )
    if descriptor_shape != member.shape:
        raise ValueError(f"{name} payload shape mismatch")
    if str(descriptor["sha256"]) != member.content_sha256:
        raise ValueError(f"{name} payload content identity mismatch")
    return array


@dataclass(frozen=True, slots=True)
class LoadedGaugeTreePriorArtifactV1:
    """A fully validated manifest and reconstructed sparse prior."""

    manifest: GaugeTreePriorArtifactV1
    prior: GaugeTreeSquareRootPriorV1


def load_gauge_tree_prior_artifact(
    manifest_path: str | Path,
) -> LoadedGaugeTreePriorArtifactV1:
    """Load an exact manifest snapshot and all content-addressed NPY members."""

    path = Path(manifest_path)
    payload = _read_stable_bytes(path, name="gauge-tree prior manifest")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("gauge-tree prior manifest is invalid JSON") from error
    manifest = GaugeTreePriorArtifactV1.from_record(
        require_mapping(raw, name="gauge-tree prior manifest")
    )
    parents = _load_member(path, manifest.parent_indices, name="parent_indices")
    transitions = _load_member(
        path,
        manifest.transition_matrices,
        name="transition_matrices",
    )
    scales = _load_member(
        path,
        manifest.innovation_scale_tril,
        name="innovation_scale_tril",
    )
    prior = GaugeTreeSquareRootPriorV1(
        gauge_ids=manifest.gauge_ids,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        source_joint_covariance_sha256=(
            manifest.source_joint_covariance_sha256
        ),
        representation_semantics=manifest.representation_semantics,
    )
    if prior.prior_id != manifest.prior_id:
        raise ValueError("loaded gauge-tree prior identity differs from manifest")
    return LoadedGaugeTreePriorArtifactV1(manifest=manifest, prior=prior)


def write_gauge_tree_prior_artifact(
    prior: GaugeTreeSquareRootPriorV1,
    manifest_path: str | Path,
) -> LoadedGaugeTreePriorArtifactV1:
    """Publish a portable sparse prior without replacing different content."""

    if not isinstance(prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("prior must be a GaugeTreeSquareRootPriorV1")
    path = Path(manifest_path)
    _reject_symlink_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)
    if path.is_symlink():
        raise ValueError("gauge-tree prior manifest must not be a symbolic link")

    parent_array, parent_payload = _npy_payload(
        prior.parent_indices,
        dtype=np.dtype("<i8"),
    )
    transition_array, transition_payload = _npy_payload(
        prior.transition_matrices,
        dtype=np.dtype("<f8"),
    )
    scale_array, scale_payload = _npy_payload(
        prior.innovation_scale_tril,
        dtype=np.dtype("<f8"),
    )
    members_and_payloads = {
        "parent_indices": (
            _member_from_array(
                "parent-indices",
                parent_array,
                parent_payload,
            ),
            parent_payload,
        ),
        "transition_matrices": (
            _member_from_array(
                "transition-matrices",
                transition_array,
                transition_payload,
            ),
            transition_payload,
        ),
        "innovation_scale_tril": (
            _member_from_array(
                "innovation-scale-tril",
                scale_array,
                scale_payload,
            ),
            scale_payload,
        ),
    }
    manifest = GaugeTreePriorArtifactV1(
        prior_id=prior.prior_id,
        gauge_ids=prior.gauge_ids,
        representation_semantics=prior.representation_semantics,
        source_joint_covariance_sha256=prior.source_joint_covariance_sha256,
        parent_indices=members_and_payloads["parent_indices"][0],
        transition_matrices=members_and_payloads["transition_matrices"][0],
        innovation_scale_tril=members_and_payloads["innovation_scale_tril"][0],
    )
    for name in MEMBER_NAMES:
        member, member_payload = members_and_payloads[name]
        _write_create_if_absent(
            path.parent / member.path,
            member_payload,
            name=f"{name} payload",
        )
    _write_create_if_absent(
        path,
        canonical_json_bytes(manifest.to_record()),
        name="gauge-tree prior manifest",
    )
    loaded = load_gauge_tree_prior_artifact(path)
    if loaded.manifest.artifact_id != manifest.artifact_id:
        raise RuntimeError("published gauge-tree prior artifact changed identity")
    return loaded
