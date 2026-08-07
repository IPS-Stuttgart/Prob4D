"""Portable sparse artifacts for causal gauge-tree square-root priors.

The artifact is an additive sidecar for :class:`GaugeTreeSquareRootPriorV1`.
It stores the causal parent list in a strict JSON manifest and the two dense
``K x 7 x 7`` factor arrays as deterministic NPY members. Existing provider-v2
and observation-factor schemas remain unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    GAUGE_TREE_PRIOR_SCHEMA,
    GAUGE_TREE_PRIOR_SEMANTICS,
    GAUGE_TREE_PRIOR_VERSION,
    canonical_array_descriptor,
    canonical_json_sha256,
)
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_mapping,
    require_sha256,
    require_string_sequence,
)
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1

GAUGE_TREE_ARTIFACT_SCHEMA: Final = "prob4d.gauge-tree-square-root-artifact"
GAUGE_TREE_ARTIFACT_VERSION: Final = 1
GAUGE_TREE_ARTIFACT_MANIFEST: Final = "manifest.json"

_TRANSITION_MEMBER: Final = "transition_matrices.npy"
_INNOVATION_MEMBER: Final = "innovation_scale_tril.npy"
_EXPECTED_FILES: Final = frozenset(
    {GAUGE_TREE_ARTIFACT_MANIFEST, _TRANSITION_MEMBER, _INNOVATION_MEMBER}
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "artifact_id",
        "prior_id",
        "prior_schema",
        "prior_version",
        "representation_semantics",
        "gauge_dimension",
        "gauge_count",
        "gauge_ids",
        "parent_indices",
        "source_joint_covariance_sha256",
        "factor_storage_nbytes",
        "dense_covariance_nbytes",
        "members",
    }
)
_MEMBER_FIELDS: Final = frozenset(
    {"path", "sha256", "bytes", "dtype", "shape", "semantic_sha256"}
)
_MEMBER_NAMES: Final = frozenset({"transition_matrices", "innovation_scale_tril"})


@dataclass(frozen=True, slots=True)
class GaugeTreePriorArtifactSummary:
    """Validated identity and storage summary for one sparse gauge-tree artifact."""

    artifact_id: str
    prior_id: str
    gauge_count: int
    factor_storage_nbytes: int
    dense_covariance_nbytes: int
    serialized_nbytes: int
    source_joint_covariance_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "prior_id": self.prior_id,
            "gauge_count": self.gauge_count,
            "factor_storage_nbytes": self.factor_storage_nbytes,
            "dense_covariance_nbytes": self.dense_covariance_nbytes,
            "serialized_nbytes": self.serialized_nbytes,
            "source_joint_covariance_sha256": self.source_joint_covariance_sha256,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _member_descriptor(path: Path, *, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    with io.BytesIO(payload) as stream:
        array = np.load(stream, allow_pickle=False)
        if stream.tell() != len(payload):
            raise ValueError(f"{path.name} contains trailing bytes")
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{path.name} must contain one NumPy array")
    semantic = canonical_array_descriptor(array)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "semantic_sha256": semantic["sha256"],
    }


def _artifact_payload(
    prior: GaugeTreeSquareRootPriorV1,
    members: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": GAUGE_TREE_ARTIFACT_SCHEMA,
        "version": GAUGE_TREE_ARTIFACT_VERSION,
        "prior_id": prior.prior_id,
        "prior_schema": GAUGE_TREE_PRIOR_SCHEMA,
        "prior_version": GAUGE_TREE_PRIOR_VERSION,
        "representation_semantics": prior.representation_semantics,
        "gauge_dimension": GAUGE_DIMENSION,
        "gauge_count": prior.gauge_count,
        "gauge_ids": list(prior.gauge_ids),
        "parent_indices": [int(value) for value in prior.parent_indices],
        "source_joint_covariance_sha256": prior.source_joint_covariance_sha256,
        "factor_storage_nbytes": prior.factor_storage_nbytes,
        "dense_covariance_nbytes": prior.dense_covariance_nbytes,
        "members": {name: dict(value) for name, value in members.items()},
    }


def _with_artifact_id(payload: Mapping[str, object]) -> dict[str, object]:
    manifest = dict(payload)
    manifest["artifact_id"] = canonical_json_sha256(payload)
    return manifest


def _require_parent_indices(value: Any, *, gauge_count: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("parent_indices must be a sequence of integers")
    parents = tuple(
        require_exact_integer(item, name=f"parent_indices[{index}]")
        for index, item in enumerate(value)
    )
    if len(parents) != gauge_count:
        raise ValueError("parent_indices length must equal gauge_count")
    return parents


def _require_shape(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of integers")
    return tuple(
        require_exact_integer(item, name=f"{name}[{index}]", minimum=0)
        for index, item in enumerate(value)
    )


def _validate_member_descriptor(
    value: Any,
    *,
    name: str,
    expected_path: str,
    expected_shape: tuple[int, ...],
) -> Mapping[str, Any]:
    descriptor = require_mapping(value, name=name)
    require_exact_fields(descriptor, _MEMBER_FIELDS, name=name)
    path = require_exact_string(descriptor["path"], name=f"{name}.path")
    if path != expected_path:
        raise ValueError(f"{name}.path must be exactly {expected_path!r}")
    require_sha256(descriptor["sha256"], name=f"{name}.sha256")
    require_exact_integer(descriptor["bytes"], name=f"{name}.bytes", minimum=1)
    dtype = require_exact_string(descriptor["dtype"], name=f"{name}.dtype")
    if dtype != np.dtype(np.float64).str:
        raise ValueError(f"{name}.dtype must be float64")
    shape = _require_shape(descriptor["shape"], name=f"{name}.shape")
    if shape != expected_shape:
        raise ValueError(f"{name}.shape must be {expected_shape}")
    require_sha256(descriptor["semantic_sha256"], name=f"{name}.semantic_sha256")
    return descriptor


def _resolve_member(root: Path, name: str) -> Path:
    candidate = root / name
    if candidate.is_symlink():
        raise ValueError(f"gauge-tree artifact member {name!r} must not be a symlink")
    if not candidate.is_file():
        raise ValueError(f"gauge-tree artifact member {name!r} is missing or not a file")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"gauge-tree artifact member {name!r} escapes its root") from error
    return resolved


def _load_array_member(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    name: str,
    expected_path: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    validated = _validate_member_descriptor(
        descriptor,
        name=name,
        expected_path=expected_path,
        expected_shape=expected_shape,
    )
    path = _resolve_member(root, expected_path)
    payload = path.read_bytes()
    expected_bytes = require_exact_integer(
        validated["bytes"], name=f"{name}.bytes", minimum=1
    )
    if len(payload) != expected_bytes:
        raise ValueError(f"{name} byte count does not match its manifest")
    if _sha256_bytes(payload) != require_sha256(
        validated["sha256"], name=f"{name}.sha256"
    ):
        raise ValueError(f"{name} SHA-256 does not match its manifest")
    with io.BytesIO(payload) as stream:
        try:
            array = np.load(stream, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"{name} must contain one valid NPY array") from error
        if stream.tell() != len(payload):
            raise ValueError(f"{name} contains trailing bytes")
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must contain one NumPy array")
    if array.dtype.str != np.dtype(np.float64).str:
        raise ValueError(f"{name} dtype does not match float64")
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape does not match its manifest")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must use C-contiguous storage")
    semantic = canonical_array_descriptor(array)
    if semantic["sha256"] != require_sha256(
        validated["semantic_sha256"], name=f"{name}.semantic_sha256"
    ):
        raise ValueError(f"{name} semantic digest does not match its manifest")
    return array


def _load_artifact(
    path: str | Path,
) -> tuple[GaugeTreeSquareRootPriorV1, GaugeTreePriorArtifactSummary]:
    root = Path(path)
    if root.is_symlink():
        raise ValueError("gauge-tree artifact root must not be a symlink")
    if not root.is_dir():
        raise ValueError("gauge-tree artifact root must be a directory")
    entries = tuple(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("gauge-tree artifact members must not be symlinks")
    actual_files = frozenset(entry.name for entry in entries)
    if actual_files != _EXPECTED_FILES:
        missing = sorted(_EXPECTED_FILES - actual_files)
        extra = sorted(actual_files - _EXPECTED_FILES)
        raise ValueError(
            f"gauge-tree artifact files changed; missing={missing}, extra={extra}"
        )

    manifest_path = _resolve_member(root, GAUGE_TREE_ARTIFACT_MANIFEST)
    manifest = load_json_object(manifest_path, name="gauge-tree artifact manifest")
    require_exact_fields(manifest, _MANIFEST_FIELDS, name="gauge-tree artifact manifest")
    if require_exact_string(manifest["schema"], name="schema") != GAUGE_TREE_ARTIFACT_SCHEMA:
        raise ValueError("gauge-tree artifact schema changed")
    if require_exact_integer(manifest["version"], name="version") != GAUGE_TREE_ARTIFACT_VERSION:
        raise ValueError("gauge-tree artifact version changed")
    artifact_id = require_sha256(manifest["artifact_id"], name="artifact_id")
    prior_id = require_sha256(manifest["prior_id"], name="prior_id")
    if require_exact_string(manifest["prior_schema"], name="prior_schema") != (
        GAUGE_TREE_PRIOR_SCHEMA
    ):
        raise ValueError("gauge-tree prior schema changed")
    if require_exact_integer(manifest["prior_version"], name="prior_version") != (
        GAUGE_TREE_PRIOR_VERSION
    ):
        raise ValueError("gauge-tree prior version changed")
    semantics = require_exact_string(
        manifest["representation_semantics"], name="representation_semantics"
    )
    if semantics != GAUGE_TREE_PRIOR_SEMANTICS:
        raise ValueError("gauge-tree representation semantics changed")
    if require_exact_integer(manifest["gauge_dimension"], name="gauge_dimension") != (
        GAUGE_DIMENSION
    ):
        raise ValueError("gauge dimension changed")
    gauge_count = require_exact_integer(
        manifest["gauge_count"], name="gauge_count", minimum=1
    )
    gauge_ids = require_string_sequence(manifest["gauge_ids"], name="gauge_ids")
    if len(gauge_ids) != gauge_count:
        raise ValueError("gauge_ids length must equal gauge_count")
    parents = _require_parent_indices(manifest["parent_indices"], gauge_count=gauge_count)
    raw_source_digest = manifest["source_joint_covariance_sha256"]
    source_digest = (
        None
        if raw_source_digest is None
        else require_sha256(raw_source_digest, name="source_joint_covariance_sha256")
    )
    factor_storage_nbytes = require_exact_integer(
        manifest["factor_storage_nbytes"], name="factor_storage_nbytes", minimum=1
    )
    dense_covariance_nbytes = require_exact_integer(
        manifest["dense_covariance_nbytes"], name="dense_covariance_nbytes", minimum=1
    )
    members = require_mapping(manifest["members"], name="members")
    require_exact_fields(members, _MEMBER_NAMES, name="members")
    expected_shape = (gauge_count, GAUGE_DIMENSION, GAUGE_DIMENSION)
    transitions = _load_array_member(
        root,
        require_mapping(members["transition_matrices"], name="members.transition_matrices"),
        name="members.transition_matrices",
        expected_path=_TRANSITION_MEMBER,
        expected_shape=expected_shape,
    )
    scales = _load_array_member(
        root,
        require_mapping(members["innovation_scale_tril"], name="members.innovation_scale_tril"),
        name="members.innovation_scale_tril",
        expected_path=_INNOVATION_MEMBER,
        expected_shape=expected_shape,
    )
    prior = GaugeTreeSquareRootPriorV1(
        gauge_ids=gauge_ids,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        source_joint_covariance_sha256=source_digest,
        representation_semantics=semantics,
    )
    if prior.prior_id != prior_id:
        raise ValueError("gauge-tree prior identity does not match its manifest")
    if prior.factor_storage_nbytes != factor_storage_nbytes:
        raise ValueError("factor_storage_nbytes does not match the reconstructed prior")
    if prior.dense_covariance_nbytes != dense_covariance_nbytes:
        raise ValueError("dense_covariance_nbytes does not match the reconstructed prior")
    identity_payload = dict(manifest)
    identity_payload.pop("artifact_id")
    if canonical_json_sha256(identity_payload) != artifact_id:
        raise ValueError("gauge-tree artifact identity does not match its manifest")
    serialized_nbytes = sum(entry.stat().st_size for entry in entries)
    summary = GaugeTreePriorArtifactSummary(
        artifact_id=artifact_id,
        prior_id=prior_id,
        gauge_count=gauge_count,
        factor_storage_nbytes=factor_storage_nbytes,
        dense_covariance_nbytes=dense_covariance_nbytes,
        serialized_nbytes=serialized_nbytes,
        source_joint_covariance_sha256=source_digest,
    )
    return prior, summary


def write_gauge_tree_prior_artifact(
    prior: GaugeTreeSquareRootPriorV1,
    path: str | Path,
) -> GaugeTreePriorArtifactSummary:
    """Publish one deterministic sparse gauge-tree artifact without overwriting."""

    if not isinstance(prior, GaugeTreeSquareRootPriorV1):
        raise TypeError("prior must be a GaugeTreeSquareRootPriorV1")
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        transition_path = staging / _TRANSITION_MEMBER
        innovation_path = staging / _INNOVATION_MEMBER
        _write_npy(transition_path, prior.transition_matrices)
        _write_npy(innovation_path, prior.innovation_scale_tril)
        members = {
            "transition_matrices": _member_descriptor(transition_path, root=staging),
            "innovation_scale_tril": _member_descriptor(innovation_path, root=staging),
        }
        manifest = _with_artifact_id(_artifact_payload(prior, members))
        _write_manifest(staging / GAUGE_TREE_ARTIFACT_MANIFEST, manifest)
        _fsync_directory(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to replace existing artifact {destination}")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _, summary = _load_artifact(destination)
    return summary


def load_gauge_tree_prior_artifact(path: str | Path) -> GaugeTreeSquareRootPriorV1:
    """Load and fully verify one portable sparse gauge-tree artifact."""

    prior, _ = _load_artifact(path)
    return prior


def verify_gauge_tree_prior_artifact(
    path: str | Path,
) -> GaugeTreePriorArtifactSummary:
    """Verify one artifact and return its immutable identity summary."""

    _, summary = _load_artifact(path)
    return summary


__all__ = [
    "GAUGE_TREE_ARTIFACT_MANIFEST",
    "GAUGE_TREE_ARTIFACT_SCHEMA",
    "GAUGE_TREE_ARTIFACT_VERSION",
    "GaugeTreePriorArtifactSummary",
    "load_gauge_tree_prior_artifact",
    "verify_gauge_tree_prior_artifact",
    "write_gauge_tree_prior_artifact",
]
