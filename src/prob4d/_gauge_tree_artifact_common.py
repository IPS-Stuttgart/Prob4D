"""Closed schemas and validation for portable sparse gauge-tree prior artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final, cast

import numpy as np

from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    GAUGE_TREE_PRIOR_SEMANTICS,
    canonical_json_sha256,
    validate_sha256,
)

GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA: Final = "prob4d.gauge-tree-prior-artifact"
GAUGE_TREE_PRIOR_ARTIFACT_VERSION: Final = 1
GAUGE_TREE_PRIOR_STORAGE_SEMANTICS: Final = (
    "content-addressed-non-pickled-npy-members-v1"
)
MAX_NPY_HEADER_BYTES: Final = 65_536
MEMBER_NAMES: Final = (
    "parent_indices",
    "transition_matrices",
    "innovation_scale_tril",
)
MEMBER_FILE_LABELS: Final = {
    "parent_indices": "parent-indices",
    "transition_matrices": "transition-matrices",
    "innovation_scale_tril": "innovation-scale-tril",
}
_MEMBER_FIELDS: Final = frozenset(
    {
        "path",
        "byte_count",
        "file_sha256",
        "dtype",
        "shape",
        "content_sha256",
    }
)
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "storage_semantics",
        "prior_id",
        "gauge_count",
        "gauge_ids",
        "representation_semantics",
        "source_joint_covariance_sha256",
        "parent_indices",
        "transition_matrices",
        "innovation_scale_tril",
    }
)


def require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty string without surrounding whitespace"
        )
    return value


def require_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def validate_digest(value: object, *, name: str) -> str:
    digest = validate_sha256(require_nonempty_string(value, name=name), name=name)
    if digest is None:
        raise RuntimeError("required digest validation returned None")
    return digest


def validate_optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return validate_digest(value, name=name)


def validate_gauge_ids_strict(values: Sequence[object]) -> tuple[str, ...]:
    gauge_ids = tuple(
        require_nonempty_string(value, name="gauge_id") for value in values
    )
    if not gauge_ids:
        raise ValueError("gauge_ids must not be empty")
    if len(set(gauge_ids)) != len(gauge_ids):
        raise ValueError("gauge_ids must be unique")
    return gauge_ids


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _require_shape(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a shape sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item,
            (int, np.integer),
        ):
            raise ValueError(f"{name} entries must be nonnegative integers")
        integer = int(item)
        if integer < 0:
            raise ValueError(f"{name} entries must be nonnegative integers")
        result.append(integer)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _validate_member_path(value: object) -> str:
    path = require_nonempty_string(value, name="array member path")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != path
        or "/" in path
        or "\\" in path
    ):
        raise ValueError("array member path must be one confined relative filename")
    if pure.suffix != ".npy" or pure.name in {".", ".."}:
        raise ValueError("array member path must name one NPY file")
    return path


def _validate_dtype(value: object, *, name: str) -> str:
    text = require_nonempty_string(value, name=name)
    try:
        dtype = np.dtype(text)
    except TypeError as error:
        raise ValueError(f"{name} is not a valid NumPy dtype") from error
    if dtype.hasobject:
        raise ValueError(f"{name} must not contain Python objects")
    return dtype.str


@dataclass(frozen=True, slots=True)
class GaugeTreePriorArrayMemberV1:
    """One exact non-pickled NPY member bound by file and array identity."""

    path: str
    byte_count: int
    file_sha256: str
    dtype: str
    shape: tuple[int, ...]
    content_sha256: str

    def __post_init__(self) -> None:
        path = _validate_member_path(self.path)
        byte_count = require_positive_integer(
            self.byte_count,
            name="array member byte_count",
        )
        file_digest = validate_digest(
            self.file_sha256,
            name="array member file_sha256",
        )
        dtype = _validate_dtype(self.dtype, name="array member dtype")
        shape = _require_shape(self.shape, name="array member shape")
        content_digest = validate_digest(
            self.content_sha256,
            name="array member content_sha256",
        )
        data_byte_count = math.prod(shape) * np.dtype(dtype).itemsize
        if byte_count <= data_byte_count:
            raise ValueError("array member byte_count omits its NPY header")
        if byte_count > data_byte_count + MAX_NPY_HEADER_BYTES:
            raise ValueError("array member byte_count exceeds the bounded NPY header")

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "file_sha256", file_digest)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "content_sha256", content_digest)

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> GaugeTreePriorArrayMemberV1:
        if set(value) != _MEMBER_FIELDS:
            missing = sorted(_MEMBER_FIELDS - value.keys())
            extra = sorted(value.keys() - _MEMBER_FIELDS)
            raise ValueError(
                "gauge-tree array member fields changed; "
                f"missing={missing}, extra={extra}"
            )
        return cls(
            path=value.get("path"),
            byte_count=value.get("byte_count"),
            file_sha256=value.get("file_sha256"),
            dtype=value.get("dtype"),
            shape=value.get("shape"),
            content_sha256=value.get("content_sha256"),
        )


@dataclass(frozen=True, slots=True)
class GaugeTreePriorArtifactV1:
    """Closed manifest for one portable sparse gauge-tree prior."""

    prior_id: str
    gauge_ids: tuple[str, ...]
    representation_semantics: str
    source_joint_covariance_sha256: str | None
    parent_indices: GaugeTreePriorArrayMemberV1
    transition_matrices: GaugeTreePriorArrayMemberV1
    innovation_scale_tril: GaugeTreePriorArrayMemberV1
    artifact_id: str | None = None
    storage_semantics: str = GAUGE_TREE_PRIOR_STORAGE_SEMANTICS

    def __post_init__(self) -> None:
        prior_id = validate_digest(self.prior_id, name="prior_id")
        gauge_ids = validate_gauge_ids_strict(self.gauge_ids)
        if self.representation_semantics != GAUGE_TREE_PRIOR_SEMANTICS:
            raise ValueError("gauge-tree prior representation semantics changed")
        source_digest = validate_optional_digest(
            self.source_joint_covariance_sha256,
            name="source_joint_covariance_sha256",
        )
        if self.storage_semantics != GAUGE_TREE_PRIOR_STORAGE_SEMANTICS:
            raise ValueError("gauge-tree prior storage semantics changed")

        expected_shapes = {
            "parent_indices": (len(gauge_ids),),
            "transition_matrices": (
                len(gauge_ids),
                GAUGE_DIMENSION,
                GAUGE_DIMENSION,
            ),
            "innovation_scale_tril": (
                len(gauge_ids),
                GAUGE_DIMENSION,
                GAUGE_DIMENSION,
            ),
        }
        expected_dtypes = {
            "parent_indices": np.dtype("<i8").str,
            "transition_matrices": np.dtype("<f8").str,
            "innovation_scale_tril": np.dtype("<f8").str,
        }
        members = {
            "parent_indices": self.parent_indices,
            "transition_matrices": self.transition_matrices,
            "innovation_scale_tril": self.innovation_scale_tril,
        }
        if len({member.path for member in members.values()}) != len(members):
            raise ValueError("gauge-tree prior array member paths must be distinct")
        for name, member in members.items():
            if member.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} member shape must be {expected_shapes[name]}"
                )
            if member.dtype != expected_dtypes[name]:
                raise ValueError(f"{name} member dtype must be {expected_dtypes[name]}")
            expected_path = (
                f"gauge-tree-prior-{MEMBER_FILE_LABELS[name]}-"
                f"{member.file_sha256}.npy"
            )
            if member.path != expected_path:
                raise ValueError(f"{name} member path is not content-addressed")

        object.__setattr__(self, "prior_id", prior_id)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "source_joint_covariance_sha256", source_digest)
        expected_id = canonical_json_sha256(self.identity_record())
        if self.artifact_id is not None:
            supplied = validate_digest(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError("gauge-tree prior artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def gauge_count(self) -> int:
        return len(self.gauge_ids)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
            "schema_version": GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
            "storage_semantics": self.storage_semantics,
            "prior_id": self.prior_id,
            "gauge_count": self.gauge_count,
            "gauge_ids": list(self.gauge_ids),
            "representation_semantics": self.representation_semantics,
            "source_joint_covariance_sha256": self.source_joint_covariance_sha256,
            "parent_indices": self.parent_indices.to_record(),
            "transition_matrices": self.transition_matrices.to_record(),
            "innovation_scale_tril": self.innovation_scale_tril.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> GaugeTreePriorArtifactV1:
        if set(value) != _ARTIFACT_FIELDS:
            missing = sorted(_ARTIFACT_FIELDS - value.keys())
            extra = sorted(value.keys() - _ARTIFACT_FIELDS)
            raise ValueError(
                "gauge-tree prior artifact fields changed; "
                f"missing={missing}, extra={extra}"
            )
        if value.get("schema") != GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA:
            raise ValueError("unexpected gauge-tree prior artifact schema")
        if value.get("schema_version") != GAUGE_TREE_PRIOR_ARTIFACT_VERSION:
            raise ValueError("unsupported gauge-tree prior artifact version")
        gauge_count = require_positive_integer(
            value.get("gauge_count"),
            name="gauge_count",
        )
        raw_ids = value.get("gauge_ids")
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ValueError("gauge_ids must be a sequence")
        gauge_ids = validate_gauge_ids_strict(raw_ids)
        if len(gauge_ids) != gauge_count:
            raise ValueError("gauge_count differs from gauge_ids")
        return cls(
            prior_id=validate_digest(value.get("prior_id"), name="prior_id"),
            gauge_ids=gauge_ids,
            representation_semantics=require_nonempty_string(
                value.get("representation_semantics"),
                name="representation_semantics",
            ),
            source_joint_covariance_sha256=validate_optional_digest(
                value.get("source_joint_covariance_sha256"),
                name="source_joint_covariance_sha256",
            ),
            parent_indices=GaugeTreePriorArrayMemberV1.from_record(
                require_mapping(value.get("parent_indices"), name="parent_indices")
            ),
            transition_matrices=GaugeTreePriorArrayMemberV1.from_record(
                require_mapping(
                    value.get("transition_matrices"),
                    name="transition_matrices",
                )
            ),
            innovation_scale_tril=GaugeTreePriorArrayMemberV1.from_record(
                require_mapping(
                    value.get("innovation_scale_tril"),
                    name="innovation_scale_tril",
                )
            ),
            artifact_id=validate_digest(
                value.get("artifact_id"),
                name="artifact_id",
            ),
            storage_semantics=require_nonempty_string(
                value.get("storage_semantics"),
                name="storage_semantics",
            ),
        )
