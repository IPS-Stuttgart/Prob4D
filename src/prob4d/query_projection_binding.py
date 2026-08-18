"""Bind query-covariance projections to exact Jacobian and row identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from .query_covariance_relevance import (
    QueryCovarianceProjectionV1,
    project_joint_covariance_to_query,
)

FloatArray = NDArray[np.float64]

QUERY_JACOBIAN_BINDING_SCHEMA: Final = "bayesian_phystwin.query_jacobian_binding"
QUERY_JACOBIAN_BINDING_VERSION: Final = 1
OBSERVATION_ROW_BINDING_SCHEMA: Final = "phys4d.observation-row-binding"
OBSERVATION_ROW_BINDING_VERSION: Final = 1
QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY: Final = (
    "Target-blind query-lineage infrastructure only. This binding identifies the "
    "exact query-Jacobian bytes and ordered observation rows used by a covariance "
    "projection. It does not establish provider competence, calibrated uncertainty, "
    "physical-query benefit, Causal4D intervention benefit, deployment safety, or "
    "state of the art."
)
BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA: Final = (
    "prob4d.bound-query-covariance-projection"
)
BOUND_QUERY_COVARIANCE_PROJECTION_VERSION: Final = 1
BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY: Final = (
    "This receipt binds a neutral Prob4D query-covariance projection to the exact "
    "caller-owned Jacobian bytes, ordered observation rows, source observation, and "
    "covariance inputs used for the calculation. It does not define the physical "
    "query, select a covariance treatment, authorize an update, or establish "
    "BayesianPhysTwin or Causal4D benefit."
)

_BINDING_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "schema",
        "schema_version",
        "query_name",
        "component_order",
        "physical_unit",
        "coordinate_frame",
        "source_observation_artifact_id",
        "provider_manifest_id",
        "causal_frame_stop",
        "query_jacobian",
        "observation_rows",
        "target_outcomes_used",
        "future_frames_used",
        "claim_boundary",
        "metadata",
    }
)
_ARRAY_DESCRIPTOR_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})
_ROW_BINDING_FIELDS: Final = frozenset(
    {"schema", "schema_version", "count", "sha256"}
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with literal string keys")
    return cast(Mapping[str, Any], value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _text(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _component_order(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("component_order must be a JSON array")
    result = tuple(_text(item, name="component_order entry") for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError("component_order must be nonempty and unique")
    return result


def _canonical_jacobian(value: object) -> FloatArray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 2:
        if raw.shape[1:] != (3,):
            raise ValueError("scalar query_jacobian must have shape (N, 3)")
        raw = raw[None, ...]
    elif raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("query_jacobian must have shape (Q, N, 3) or (N, 3)")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("query_jacobian must contain at least one query and one row")
    if not np.all(np.isfinite(raw)):
        raise ValueError("query_jacobian must be finite")
    result = np.ascontiguousarray(raw, dtype=np.dtype("<f8"))
    result.setflags(write=False)
    return cast(FloatArray, result)


def _canonical_array(value: object, *, name: str) -> FloatArray:
    result = np.ascontiguousarray(
        np.asarray(value, dtype=np.float64),
        dtype=np.dtype("<f8"),
    )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return cast(FloatArray, result)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _array_descriptor(value: np.ndarray) -> Mapping[str, Any]:
    return frozen_finite_json_mapping(
        {
            "dtype": "<f8",
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        },
        name="array descriptor",
    )


def _validated_numeric_descriptor(
    value: object,
    *,
    name: str,
    expected_observation_count: int,
    covariance: bool,
) -> Mapping[str, Any]:
    source = _mapping(value, name=name)
    _exact_fields(source, _ARRAY_DESCRIPTOR_FIELDS, name=name)
    if source["dtype"] != "<f8":
        raise ValueError(f"{name} dtype must be <f8")
    shape_raw = source["shape"]
    if isinstance(shape_raw, (str, bytes)) or not isinstance(shape_raw, Sequence):
        raise ValueError(f"{name} shape must be a JSON array")
    shape = tuple(
        _integer(
            item,
            name=f"{name} shape[{index}]",
            minimum=(0 if not covariance and index == 2 else 1),
        )
        for index, item in enumerate(shape_raw)
    )
    expected_prefix = (expected_observation_count, 3)
    if len(shape) != 3 or shape[:2] != expected_prefix:
        raise ValueError(f"{name} must have shape [N, 3, 3] or [N, 3, R]")
    if covariance and shape[2] != 3:
        raise ValueError(f"{name} must have shape [N, 3, 3]")
    digest = _sha256(source["sha256"], name=f"{name} sha256")
    return frozen_finite_json_mapping(
        {"dtype": "<f8", "shape": list(shape), "sha256": digest},
        name=name,
    )


def _row_ids(value: object, *, expected_count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("row_ids must be a sequence of canonical strings")
    result = tuple(_text(item, name="row_ids entry") for item in value)
    if len(result) != expected_count:
        raise ValueError("row_ids length must equal the Jacobian observation count")
    if len(result) != len(set(result)):
        raise ValueError("row_ids must be unique")
    return result


def _row_ids_sha256(value: Sequence[str]) -> str:
    return _content_id({"row_ids": list(value)})


def _validated_array_descriptor(value: object) -> tuple[tuple[int, int, int], str]:
    source = _mapping(value, name="query_jacobian descriptor")
    _exact_fields(source, _ARRAY_DESCRIPTOR_FIELDS, name="query_jacobian descriptor")
    if source["dtype"] != "<f8":
        raise ValueError("query_jacobian dtype must be <f8")
    shape_raw = source["shape"]
    if isinstance(shape_raw, (str, bytes)) or not isinstance(shape_raw, Sequence):
        raise ValueError("query_jacobian shape must be a JSON array")
    shape = tuple(
        _integer(item, name=f"query_jacobian shape[{index}]", minimum=1)
        for index, item in enumerate(shape_raw)
    )
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("query_jacobian shape must be [Q, N, 3]")
    return cast(tuple[int, int, int], shape), _sha256(
        source["sha256"],
        name="query_jacobian sha256",
    )


def _validated_row_binding(value: object) -> tuple[int, str]:
    source = _mapping(value, name="observation_rows")
    _exact_fields(source, _ROW_BINDING_FIELDS, name="observation_rows")
    if source["schema"] != OBSERVATION_ROW_BINDING_SCHEMA:
        raise ValueError("observation-row binding schema changed")
    if (
        _integer(
            source["schema_version"],
            name="observation-row binding schema_version",
            minimum=1,
        )
        != OBSERVATION_ROW_BINDING_VERSION
    ):
        raise ValueError("observation-row binding version changed")
    return (
        _integer(source["count"], name="observation row count", minimum=1),
        _sha256(source["sha256"], name="observation rows sha256"),
    )


@dataclass(frozen=True, slots=True)
class ValidatedQueryJacobianBindingV1:
    """Independently validated BayesianPhysTwin query-Jacobian binding."""

    artifact_id: str
    query_name: str
    component_order: tuple[str, ...]
    physical_unit: str
    coordinate_frame: str
    source_observation_artifact_id: str
    provider_manifest_id: str
    causal_frame_stop: int
    query_jacobian_shape: tuple[int, int, int]
    query_jacobian_sha256: str
    observation_count: int
    row_ids_sha256: str
    record: Mapping[str, Any] = field(repr=False)

    @property
    def query_dimension(self) -> int:
        return self.query_jacobian_shape[0]

    def validate_payload(self, query_jacobian: object, row_ids: object) -> FloatArray:
        jacobian = _canonical_jacobian(query_jacobian)
        if tuple(jacobian.shape) != self.query_jacobian_shape:
            raise ValueError("query_jacobian shape differs from its binding")
        if _array_sha256(jacobian) != self.query_jacobian_sha256:
            raise ValueError("query_jacobian bytes differ from their binding")
        rows = _row_ids(row_ids, expected_count=self.observation_count)
        if _row_ids_sha256(rows) != self.row_ids_sha256:
            raise ValueError("row_ids differ from their binding")
        return jacobian


def validate_query_jacobian_binding(value: object) -> ValidatedQueryJacobianBindingV1:
    source = _mapping(value, name="query Jacobian binding")
    _exact_fields(source, _BINDING_FIELDS, name="query Jacobian binding")
    if source["schema"] != QUERY_JACOBIAN_BINDING_SCHEMA:
        raise ValueError("query Jacobian binding schema changed")
    if (
        _integer(
            source["schema_version"],
            name="query Jacobian binding schema_version",
            minimum=1,
        )
        != QUERY_JACOBIAN_BINDING_VERSION
    ):
        raise ValueError("query Jacobian binding version changed")
    if source["target_outcomes_used"] is not False:
        raise ValueError("query Jacobian binding must be target blind")
    if source["future_frames_used"] is not False:
        raise ValueError("query Jacobian binding must be causal-prefix only")
    if source["claim_boundary"] != QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY:
        raise ValueError("query Jacobian binding claim boundary changed")
    shape, jacobian_sha = _validated_array_descriptor(source["query_jacobian"])
    row_count, row_sha = _validated_row_binding(source["observation_rows"])
    components = _component_order(source["component_order"])
    if len(components) != shape[0]:
        raise ValueError("component_order length differs from query dimension")
    if row_count != shape[1]:
        raise ValueError("observation-row count differs from Jacobian shape")
    unsigned = dict(source)
    supplied_id = _sha256(unsigned.pop("artifact_id"), name="artifact_id")
    if supplied_id != _content_id(unsigned):
        raise ValueError("artifact_id does not match query Jacobian binding")
    record = frozen_finite_json_mapping(source, name="query Jacobian binding")
    return ValidatedQueryJacobianBindingV1(
        artifact_id=supplied_id,
        query_name=_text(source["query_name"], name="query_name"),
        component_order=components,
        physical_unit=_text(source["physical_unit"], name="physical_unit"),
        coordinate_frame=_text(source["coordinate_frame"], name="coordinate_frame"),
        source_observation_artifact_id=_sha256(
            source["source_observation_artifact_id"],
            name="source_observation_artifact_id",
        ),
        provider_manifest_id=_sha256(
            source["provider_manifest_id"],
            name="provider_manifest_id",
        ),
        causal_frame_stop=_integer(
            source["causal_frame_stop"],
            name="causal_frame_stop",
            minimum=1,
        ),
        query_jacobian_shape=shape,
        query_jacobian_sha256=jacobian_sha,
        observation_count=row_count,
        row_ids_sha256=row_sha,
        record=record,
    )


@dataclass(frozen=True, slots=True)
class BoundQueryCovarianceProjectionV1:
    """Content-addressed receipt for a lineage-bound covariance projection."""

    binding: ValidatedQueryJacobianBindingV1
    local_covariance_descriptor: Mapping[str, Any]
    low_rank_factor_descriptor: Mapping[str, Any]
    projection: QueryCovarianceProjectionV1
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ValidatedQueryJacobianBindingV1):
            raise TypeError("binding must be a ValidatedQueryJacobianBindingV1")
        if not isinstance(self.projection, QueryCovarianceProjectionV1):
            raise TypeError("projection must be a QueryCovarianceProjectionV1")
        if self.projection.observation_count != self.binding.observation_count:
            raise ValueError("projection observation count differs from query binding")
        if self.projection.query_dimension != self.binding.query_dimension:
            raise ValueError("projection query dimension differs from query binding")
        local = _validated_numeric_descriptor(
            self.local_covariance_descriptor,
            name="local_covariance_m2",
            expected_observation_count=self.binding.observation_count,
            covariance=True,
        )
        factor = _validated_numeric_descriptor(
            self.low_rank_factor_descriptor,
            name="low_rank_factor_m",
            expected_observation_count=self.binding.observation_count,
            covariance=False,
        )
        object.__setattr__(self, "local_covariance_descriptor", local)
        object.__setattr__(self, "low_rank_factor_descriptor", factor)
        expected_id = _content_id(self.descriptor())
        if self.artifact_id is not None and _sha256(
            self.artifact_id,
            name="artifact_id",
        ) != expected_id:
            raise ValueError("artifact_id does not match bound query projection")
        object.__setattr__(self, "artifact_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA,
            "schema_version": BOUND_QUERY_COVARIANCE_PROJECTION_VERSION,
            "query_jacobian_binding_id": self.binding.artifact_id,
            "source_observation_artifact_id": (
                self.binding.source_observation_artifact_id
            ),
            "provider_manifest_id": self.binding.provider_manifest_id,
            "query_jacobian_sha256": self.binding.query_jacobian_sha256,
            "row_ids_sha256": self.binding.row_ids_sha256,
            "local_covariance_m2": plain_json(self.local_covariance_descriptor),
            "low_rank_factor_m": plain_json(self.low_rank_factor_descriptor),
            "projection_summary": self.projection.summary(),
            "claim_boundary": BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}


def project_bound_joint_covariance_to_query(
    query_jacobian_binding: object,
    query_jacobian: object,
    row_ids: object,
    local_covariance_m2: object,
    low_rank_factor_m: object,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> BoundQueryCovarianceProjectionV1:
    """Project covariance only after exact query and row lineage validation."""

    binding = validate_query_jacobian_binding(query_jacobian_binding)
    jacobian = binding.validate_payload(query_jacobian, row_ids)
    local = _canonical_array(local_covariance_m2, name="local_covariance_m2")
    factor = _canonical_array(low_rank_factor_m, name="low_rank_factor_m")
    if local.shape != (binding.observation_count, 3, 3):
        raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
    if factor.ndim != 3 or factor.shape[:2] != (binding.observation_count, 3):
        raise ValueError("low_rank_factor_m must have shape (N, 3, R)")
    projection = project_joint_covariance_to_query(
        jacobian,
        local,
        factor,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return BoundQueryCovarianceProjectionV1(
        binding=binding,
        local_covariance_descriptor=_array_descriptor(local),
        low_rank_factor_descriptor=_array_descriptor(factor),
        projection=projection,
    )


def write_bound_query_covariance_projection(
    projection: BoundQueryCovarianceProjectionV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(projection, BoundQueryCovarianceProjectionV1):
        raise TypeError("projection must be a BoundQueryCovarianceProjectionV1")
    encoded = json.dumps(
        projection.to_record(),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, encoded, overwrite=overwrite)


__all__ = [
    "BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY",
    "BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA",
    "BOUND_QUERY_COVARIANCE_PROJECTION_VERSION",
    "QUERY_JACOBIAN_BINDING_CLAIM_BOUNDARY",
    "QUERY_JACOBIAN_BINDING_SCHEMA",
    "QUERY_JACOBIAN_BINDING_VERSION",
    "BoundQueryCovarianceProjectionV1",
    "ValidatedQueryJacobianBindingV1",
    "project_bound_joint_covariance_to_query",
    "validate_query_jacobian_binding",
    "write_bound_query_covariance_projection",
]
