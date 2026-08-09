"""Portable tree-sparse observation artifacts without dense gauge covariance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np

from ._gauge_tree_artifact_common import (
    GaugeTreePriorArrayMemberV1,
    canonical_json_bytes,
    require_mapping,
    validate_gauge_ids_strict,
)
from ._gauge_tree_artifact_io import (
    _load_member,
    _npy_payload,
    _read_stable_bytes,
    _reject_duplicate_keys,
    _reject_json_constant,
    _write_create_if_absent,
    load_gauge_tree_prior_artifact,
    write_gauge_tree_prior_artifact,
)
from ._gauge_tree_common import canonical_array_descriptor, canonical_json_sha256
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_revision,
    require_sha256,
    require_string_sequence,
)
from .gauge_tree_prior_artifact import gauge_tree_prior_artifact_id
from .tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    build_tree_sparse_observation_factors,
)

TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA: Final = (
    "prob4d.tree-sparse-observation-artifact"
)
TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION: Final = 1
TREE_SPARSE_OBSERVATION_STORAGE_SEMANTICS: Final = (
    "content-addressed-non-pickled-npy-members-v1"
)
TREE_SPARSE_OBSERVATION_CLAIM_BOUNDARY: Final = (
    "This artifact preserves selected explicit-gauge observation rows and one "
    "portable sparse gauge-tree prior without serializing a dense joint gauge "
    "covariance. It does not establish provider competence, covariance calibration, "
    "physical-query identifiability, BayesianPhysTwin benefit, Causal4D benefit, "
    "deployment safety, or state of the art."
)
TREE_SPARSE_OBSERVATION_MAX_MANIFEST_BYTES: Final = 4_194_304

_ARRAY_NAMES: Final = (
    "world_mean_m",
    "conditional_world_covariance_m2",
    "local_gauge_jacobian",
    "gauge_indices",
    "association_probability",
    "prior_reliability",
    "prior_nominal_probability",
    "composite_weight",
    "point_ids",
    "frame_indices",
    "view_indices",
    "factor_indices",
    "correlation_group_indices",
)
_FLOAT_ARRAY_NAMES: Final = frozenset(
    {
        "world_mean_m",
        "conditional_world_covariance_m2",
        "local_gauge_jacobian",
        "association_probability",
        "prior_reliability",
        "prior_nominal_probability",
        "composite_weight",
    }
)
_INT_ARRAY_NAMES: Final = frozenset(set(_ARRAY_NAMES) - _FLOAT_ARRAY_NAMES)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "storage_semantics",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "causal_frame_stop",
        "observation_count",
        "gauge_tree_prior_artifact_id",
        "gauge_tree_prior_id",
        "gauge_ids",
        "view_id_table",
        "factor_id_table",
        "correlation_group_id_table",
        "array_members",
        "metadata",
        "claim_boundary",
    }
)

TreeSparseObservationArrayMemberV1 = GaugeTreePriorArrayMemberV1


def _string_table(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    values = cast(tuple[object, ...], value)
    normalized = tuple(
        require_exact_string(item, name=f"{name} item") for item in values
    )
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"{name} must be sorted and unique")
    return normalized


def _expected_shapes(observation_count: int) -> dict[str, tuple[int, ...]]:
    return {
        "world_mean_m": (observation_count, 3),
        "conditional_world_covariance_m2": (observation_count, 3, 3),
        "local_gauge_jacobian": (observation_count, 3, 7),
        "gauge_indices": (observation_count,),
        "association_probability": (observation_count,),
        "prior_reliability": (observation_count,),
        "prior_nominal_probability": (observation_count,),
        "composite_weight": (observation_count,),
        "point_ids": (observation_count,),
        "frame_indices": (observation_count,),
        "view_indices": (observation_count,),
        "factor_indices": (observation_count,),
        "correlation_group_indices": (observation_count,),
    }


def _member_path(name: str, file_sha256: str) -> str:
    label = name.replace("_", "-")
    return f"tree-sparse-observation-{label}-{file_sha256}.npy"


def _member_from_array(
    name: str,
    array: np.ndarray,
    payload: bytes,
) -> TreeSparseObservationArrayMemberV1:
    file_digest = hashlib.sha256(payload).hexdigest()
    descriptor = canonical_array_descriptor(array)
    return TreeSparseObservationArrayMemberV1(
        path=_member_path(name, file_digest),
        byte_count=len(payload),
        file_sha256=file_digest,
        dtype=str(descriptor["dtype"]),
        shape=tuple(int(item) for item in cast(list[int], descriptor["shape"])),
        content_sha256=str(descriptor["sha256"]),
    )


def _canonical_table(values: tuple[str, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    table = tuple(sorted(set(values)))
    positions = {value: index for index, value in enumerate(table)}
    indices = np.fromiter(
        (positions[value] for value in values),
        dtype=np.int64,
        count=len(values),
    )
    return table, indices


def _decode_table(
    table: tuple[str, ...],
    indices: np.ndarray,
    *,
    name: str,
) -> tuple[str, ...]:
    if indices.dtype != np.dtype("<i8") and indices.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} indices must use int64")
    if indices.ndim != 1:
        raise ValueError(f"{name} indices must be a vector")
    if np.any(indices < 0) or np.any(indices >= len(table)):
        raise ValueError(f"{name} indices reference an unknown table entry")
    return tuple(table[int(index)] for index in indices)


@dataclass(frozen=True, slots=True)
class TreeSparseObservationArtifactV1:
    """Closed manifest binding sparse observation rows to one sparse prior."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    observation_count: int
    gauge_tree_prior_artifact_id: str
    gauge_tree_prior_id: str
    gauge_ids: tuple[str, ...]
    view_id_table: tuple[str, ...]
    factor_id_table: tuple[str, ...]
    correlation_group_id_table: tuple[str, ...]
    array_members: Mapping[str, TreeSparseObservationArrayMemberV1]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    storage_semantics: str = TREE_SPARSE_OBSERVATION_STORAGE_SEMANTICS
    claim_boundary: str = TREE_SPARSE_OBSERVATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        sequence_id = require_exact_string(self.sequence_id, name="sequence_id")
        case_id = require_exact_string(self.case_id, name="case_id")
        stream_id = require_exact_string(self.stream_id, name="stream_id")
        source_repository = require_exact_string(
            self.source_repository,
            name="source_repository",
        )
        source_revision = require_revision(
            self.source_revision,
            name="source_revision",
        )
        causal_frame_stop = require_exact_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        observation_count = require_exact_integer(
            self.observation_count,
            name="observation_count",
            minimum=1,
        )
        prior_artifact_id = require_sha256(
            self.gauge_tree_prior_artifact_id,
            name="gauge_tree_prior_artifact_id",
        )
        prior_id = require_sha256(
            self.gauge_tree_prior_id,
            name="gauge_tree_prior_id",
        )
        gauge_ids = validate_gauge_ids_strict(self.gauge_ids)
        view_table = _string_table(self.view_id_table, name="view_id_table")
        factor_table = _string_table(self.factor_id_table, name="factor_id_table")
        group_table = _string_table(
            self.correlation_group_id_table,
            name="correlation_group_id_table",
        )
        if self.storage_semantics != TREE_SPARSE_OBSERVATION_STORAGE_SEMANTICS:
            raise ValueError("tree-sparse observation storage semantics changed")
        if self.claim_boundary != TREE_SPARSE_OBSERVATION_CLAIM_BOUNDARY:
            raise ValueError("tree-sparse observation claim boundary changed")
        if not isinstance(self.array_members, Mapping):
            raise TypeError("array_members must be a mapping")
        members = dict(self.array_members)
        if set(members) != set(_ARRAY_NAMES):
            missing = sorted(set(_ARRAY_NAMES) - members.keys())
            extra = sorted(members.keys() - set(_ARRAY_NAMES))
            raise ValueError(
                f"tree-sparse observation array inventory changed; "
                f"missing={missing}, extra={extra}"
            )
        expected_shapes = _expected_shapes(observation_count)
        paths: set[str] = set()
        for name in _ARRAY_NAMES:
            member = members[name]
            if not isinstance(member, TreeSparseObservationArrayMemberV1):
                raise TypeError(f"array member {name} has the wrong type")
            expected_dtype = (
                np.dtype("<f8").str if name in _FLOAT_ARRAY_NAMES else np.dtype("<i8").str
            )
            if member.dtype != expected_dtype:
                raise ValueError(f"array member {name} must use {expected_dtype}")
            if member.shape != expected_shapes[name]:
                raise ValueError(
                    f"array member {name} shape must be {expected_shapes[name]}"
                )
            if member.path != _member_path(name, member.file_sha256):
                raise ValueError(f"array member {name} path is not content-addressed")
            if member.path in paths:
                raise ValueError("tree-sparse observation member paths must be distinct")
            paths.add(member.path)
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="tree-sparse observation metadata",
        )

        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "source_repository", source_repository)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "gauge_tree_prior_artifact_id", prior_artifact_id)
        object.__setattr__(self, "gauge_tree_prior_id", prior_id)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "view_id_table", view_table)
        object.__setattr__(self, "factor_id_table", factor_table)
        object.__setattr__(self, "correlation_group_id_table", group_table)
        object.__setattr__(self, "array_members", MappingProxyType(members))
        object.__setattr__(self, "metadata", metadata)

        expected_id = canonical_json_sha256(self.identity_record())
        if self.artifact_id is not None:
            supplied = require_sha256(self.artifact_id, name="artifact_id")
            if supplied != expected_id:
                raise ValueError("tree-sparse observation artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def gauge_tree_prior_manifest_filename(self) -> str:
        return f"gauge-tree-prior-{self.gauge_tree_prior_artifact_id}.json"

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA,
            "schema_version": TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
            "storage_semantics": self.storage_semantics,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "causal_frame_stop": self.causal_frame_stop,
            "observation_count": self.observation_count,
            "gauge_tree_prior_artifact_id": self.gauge_tree_prior_artifact_id,
            "gauge_tree_prior_id": self.gauge_tree_prior_id,
            "gauge_ids": list(self.gauge_ids),
            "view_id_table": list(self.view_id_table),
            "factor_id_table": list(self.factor_id_table),
            "correlation_group_id_table": list(self.correlation_group_id_table),
            "array_members": {
                name: self.array_members[name].to_record() for name in _ARRAY_NAMES
            },
            "metadata": plain_json(self.metadata),
            "claim_boundary": self.claim_boundary,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> TreeSparseObservationArtifactV1:
        require_exact_fields(value, _MANIFEST_FIELDS, name="tree-sparse observation manifest")
        if value.get("schema") != TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA:
            raise ValueError("unexpected tree-sparse observation artifact schema")
        version = require_exact_integer(
            value.get("schema_version"),
            name="schema_version",
            minimum=1,
        )
        if version != TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION:
            raise ValueError("unsupported tree-sparse observation artifact version")
        raw_ids = value.get("gauge_ids")
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ValueError("gauge_ids must be a sequence")
        member_records = require_mapping(
            value.get("array_members"),
            name="array_members",
        )
        if set(member_records) != set(_ARRAY_NAMES):
            raise ValueError("tree-sparse observation array inventory changed")
        members = {
            name: TreeSparseObservationArrayMemberV1.from_record(
                require_mapping(member_records[name], name=f"array_members.{name}")
            )
            for name in _ARRAY_NAMES
        }
        return cls(
            sequence_id=require_exact_string(value.get("sequence_id"), name="sequence_id"),
            case_id=require_exact_string(value.get("case_id"), name="case_id"),
            stream_id=require_exact_string(value.get("stream_id"), name="stream_id"),
            source_repository=require_exact_string(
                value.get("source_repository"),
                name="source_repository",
            ),
            source_revision=require_revision(
                value.get("source_revision"),
                name="source_revision",
            ),
            causal_frame_stop=require_exact_integer(
                value.get("causal_frame_stop"),
                name="causal_frame_stop",
                minimum=1,
            ),
            observation_count=require_exact_integer(
                value.get("observation_count"),
                name="observation_count",
                minimum=1,
            ),
            gauge_tree_prior_artifact_id=require_sha256(
                value.get("gauge_tree_prior_artifact_id"),
                name="gauge_tree_prior_artifact_id",
            ),
            gauge_tree_prior_id=require_sha256(
                value.get("gauge_tree_prior_id"),
                name="gauge_tree_prior_id",
            ),
            gauge_ids=validate_gauge_ids_strict(cast(Sequence[object], raw_ids)),
            view_id_table=require_string_sequence(
                value.get("view_id_table"),
                name="view_id_table",
            ),
            factor_id_table=require_string_sequence(
                value.get("factor_id_table"),
                name="factor_id_table",
            ),
            correlation_group_id_table=require_string_sequence(
                value.get("correlation_group_id_table"),
                name="correlation_group_id_table",
            ),
            array_members=members,
            metadata=require_finite_json_mapping(
                value.get("metadata"),
                name="metadata",
            ),
            artifact_id=require_sha256(value.get("artifact_id"), name="artifact_id"),
            storage_semantics=require_exact_string(
                value.get("storage_semantics"),
                name="storage_semantics",
            ),
            claim_boundary=require_exact_string(
                value.get("claim_boundary"),
                name="claim_boundary",
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedTreeSparseObservationArtifactV1:
    """A verified manifest and reconstructed tree-sparse execution object."""

    manifest: TreeSparseObservationArtifactV1
    factors: TreeSparseStackedObservationFactors


def _array_payloads(
    factors: TreeSparseStackedObservationFactors,
) -> tuple[
    dict[str, tuple[TreeSparseObservationArrayMemberV1, bytes]],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    view_table, view_indices = _canonical_table(factors.view_ids)
    factor_table, factor_indices = _canonical_table(factors.factor_ids)
    group_table, group_indices = _canonical_table(factors.correlation_group_ids)
    raw_arrays: dict[str, tuple[object, np.dtype[Any]]] = {
        "world_mean_m": (factors.world_mean_m, np.dtype("<f8")),
        "conditional_world_covariance_m2": (
            factors.conditional_world_covariance_m2,
            np.dtype("<f8"),
        ),
        "local_gauge_jacobian": (factors.local_gauge_jacobian, np.dtype("<f8")),
        "gauge_indices": (factors.gauge_indices, np.dtype("<i8")),
        "association_probability": (factors.association_probability, np.dtype("<f8")),
        "prior_reliability": (factors.prior_reliability, np.dtype("<f8")),
        "prior_nominal_probability": (
            factors.prior_nominal_probability,
            np.dtype("<f8"),
        ),
        "composite_weight": (factors.composite_weight, np.dtype("<f8")),
        "point_ids": (factors.point_ids, np.dtype("<i8")),
        "frame_indices": (factors.frame_indices, np.dtype("<i8")),
        "view_indices": (view_indices, np.dtype("<i8")),
        "factor_indices": (factor_indices, np.dtype("<i8")),
        "correlation_group_indices": (group_indices, np.dtype("<i8")),
    }
    result: dict[str, tuple[TreeSparseObservationArrayMemberV1, bytes]] = {}
    for name in _ARRAY_NAMES:
        array, payload = _npy_payload(raw_arrays[name][0], dtype=raw_arrays[name][1])
        result[name] = (_member_from_array(name, array, payload), payload)
    return result, view_table, factor_table, group_table


def write_tree_sparse_observation_artifact(
    factors: TreeSparseStackedObservationFactors,
    manifest_path: str | Path,
    *,
    sequence_id: str,
    case_id: str,
    stream_id: str,
    source_repository: str,
    source_revision: str,
    metadata: Mapping[str, Any] | None = None,
) -> LoadedTreeSparseObservationArtifactV1:
    """Publish sparse rows and their sparse prior without dense covariance I/O."""

    if not isinstance(factors, TreeSparseStackedObservationFactors):
        raise TypeError("factors must be a TreeSparseStackedObservationFactors")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior_artifact_id = gauge_tree_prior_artifact_id(factors.gauge_tree_prior)
    prior_manifest_path = path.parent / f"gauge-tree-prior-{prior_artifact_id}.json"
    loaded_prior = write_gauge_tree_prior_artifact(
        factors.gauge_tree_prior,
        prior_manifest_path,
    )
    if loaded_prior.manifest.artifact_id != prior_artifact_id:
        raise RuntimeError("published gauge-tree prior changed artifact identity")

    members_and_payloads, view_table, factor_table, group_table = _array_payloads(factors)
    manifest = TreeSparseObservationArtifactV1(
        sequence_id=sequence_id,
        case_id=case_id,
        stream_id=stream_id,
        source_repository=source_repository,
        source_revision=source_revision,
        causal_frame_stop=factors.causal_frame_stop,
        observation_count=factors.observation_count,
        gauge_tree_prior_artifact_id=prior_artifact_id,
        gauge_tree_prior_id=factors.gauge_tree_prior.prior_id,
        gauge_ids=factors.gauge_ids,
        view_id_table=view_table,
        factor_id_table=factor_table,
        correlation_group_id_table=group_table,
        array_members={name: members_and_payloads[name][0] for name in _ARRAY_NAMES},
        metadata={} if metadata is None else metadata,
    )
    reserved_names = {
        manifest.gauge_tree_prior_manifest_filename,
        *(member.path for member in manifest.array_members.values()),
    }
    if path.name in reserved_names:
        raise ValueError("tree-sparse observation manifest collides with a payload path")
    for name in _ARRAY_NAMES:
        member, payload = members_and_payloads[name]
        _write_create_if_absent(
            path.parent / member.path,
            payload,
            name=f"tree-sparse observation {name} payload",
        )
    _write_create_if_absent(
        path,
        canonical_json_bytes(manifest.to_record()),
        name="tree-sparse observation manifest",
    )
    loaded = load_tree_sparse_observation_artifact(path)
    if loaded.manifest.artifact_id != manifest.artifact_id:
        raise RuntimeError("published tree-sparse observation changed identity")
    return loaded


def load_tree_sparse_observation_artifact(
    manifest_path: str | Path,
) -> LoadedTreeSparseObservationArtifactV1:
    """Load sparse observation rows and prior without materializing a dense prior."""

    path = Path(manifest_path)
    payload = _read_stable_bytes(
        path,
        name="tree-sparse observation manifest",
        maximum_bytes=TREE_SPARSE_OBSERVATION_MAX_MANIFEST_BYTES,
    )
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("tree-sparse observation manifest is invalid JSON") from error
    manifest = TreeSparseObservationArtifactV1.from_record(
        require_mapping(raw, name="tree-sparse observation manifest")
    )
    prior_path = path.parent / manifest.gauge_tree_prior_manifest_filename
    loaded_prior = load_gauge_tree_prior_artifact(prior_path)
    if loaded_prior.manifest.artifact_id != manifest.gauge_tree_prior_artifact_id:
        raise ValueError("gauge-tree prior artifact identity differs from observation manifest")
    if loaded_prior.prior.prior_id != manifest.gauge_tree_prior_id:
        raise ValueError("gauge-tree prior identity differs from observation manifest")
    if loaded_prior.prior.gauge_ids != manifest.gauge_ids:
        raise ValueError("gauge-tree prior order differs from observation manifest")

    arrays = {
        name: _load_member(path, manifest.array_members[name], name=name)
        for name in _ARRAY_NAMES
    }
    view_ids = _decode_table(
        manifest.view_id_table,
        arrays["view_indices"],
        name="view_id",
    )
    factor_ids = _decode_table(
        manifest.factor_id_table,
        arrays["factor_indices"],
        name="factor_id",
    )
    group_ids = _decode_table(
        manifest.correlation_group_id_table,
        arrays["correlation_group_indices"],
        name="correlation_group_id",
    )
    factors = build_tree_sparse_observation_factors(
        loaded_prior.prior,
        world_mean_m=arrays["world_mean_m"],
        conditional_world_covariance_m2=arrays["conditional_world_covariance_m2"],
        local_gauge_jacobian=arrays["local_gauge_jacobian"],
        gauge_indices=arrays["gauge_indices"],
        association_probability=arrays["association_probability"],
        prior_reliability=arrays["prior_reliability"],
        prior_nominal_probability=arrays["prior_nominal_probability"],
        composite_weight=arrays["composite_weight"],
        point_ids=arrays["point_ids"],
        frame_indices=arrays["frame_indices"],
        view_ids=view_ids,
        factor_ids=factor_ids,
        correlation_group_ids=group_ids,
        causal_frame_stop=manifest.causal_frame_stop,
    )
    if factors.observation_count != manifest.observation_count:
        raise ValueError("loaded observation count differs from manifest")
    return LoadedTreeSparseObservationArtifactV1(
        manifest=manifest,
        factors=factors,
    )


__all__ = [
    "LoadedTreeSparseObservationArtifactV1",
    "TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA",
    "TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION",
    "TREE_SPARSE_OBSERVATION_CLAIM_BOUNDARY",
    "TREE_SPARSE_OBSERVATION_STORAGE_SEMANTICS",
    "TreeSparseObservationArrayMemberV1",
    "TreeSparseObservationArtifactV1",
    "load_tree_sparse_observation_artifact",
    "write_tree_sparse_observation_artifact",
]
