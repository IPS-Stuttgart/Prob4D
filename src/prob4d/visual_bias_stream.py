"""Append-only recursive use of one shared visual-bias nuisance prior.

Each update binds one ``VisualBiasNuisanceV1`` sidecar to one observation-factor
stream update.  All rows share the same latent bias state and prior across time;
the prior is represented once rather than duplicated as an independent block for
every recursive update.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
    require_string_sequence,
)
from .visual_bias import VisualBiasNuisanceV1

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

VISUAL_BIAS_MODEL_SCHEMA: Final = "prob4d.visual-bias-model.v1"
VISUAL_BIAS_STREAM_SCHEMA: Final = "prob4d.visual-bias-nuisance-stream"
VISUAL_BIAS_STREAM_VERSION: Final = 1
VISUAL_BIAS_STREAM_UPDATE_SCHEMA: Final = "prob4d.visual-bias-stream-update.v1"
VISUAL_BIAS_STREAM_CLAIM_BOUNDARY: Final = (
    "This artifact binds several causal observation updates to one persistent "
    "source-calibrated visual-bias latent and prior. It does not establish provider "
    "competence, target calibration, physical-state identifiability, guarded-query "
    "benefit, Causal4D intervention benefit, deployment safety, or state of the art."
)

_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "stream_key",
        "bias_model_id",
        "bias_ids",
        "basis_names",
        "orthogonalization_semantics",
        "gauge_projection_tolerance",
        "updates",
        "payload",
        "arrays",
        "model_metadata",
        "metadata",
        "claim_boundary",
    }
)
_UPDATE_FIELDS: Final = frozenset(
    {
        "schema",
        "bias_model_id",
        "observation_stream_update_id",
        "visual_bias_artifact_id",
        "observation_artifact_id",
        "observation_identity_sha256",
        "frame_start",
        "frame_stop_exclusive",
        "row_start",
        "row_stop_exclusive",
        "maximum_gauge_projection",
        "previous_update_id",
        "update_id",
    }
)
_PAYLOAD_FIELDS: Final = frozenset({"path", "sha256", "byte_count", "allow_pickle"})
_ARRAY_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})
_ARRAY_NAMES: Final = (
    "row_update_indices",
    "row_bias_indices",
    "bias_jacobian",
    "joint_bias_covariance",
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_real(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _safe_relative_path(value: object, *, name: str) -> str:
    path = require_exact_string(value, name=name)
    if "\\" in path:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    if pure.as_posix() != path:
        raise ValueError(f"{name} must be a canonical POSIX relative path")
    return path


def _resolved_member(root: Path, relative_path: str, *, name: str) -> Path:
    safe = _safe_relative_path(relative_path, name=name)
    root_resolved = root.resolve()
    current = root_resolved
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    candidate = current.resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the manifest directory") from error
    return candidate


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


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **dict(arrays))  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_joint_covariance(
    value: np.ndarray,
    *,
    latent_dimension: int,
) -> FloatArray:
    covariance = np.asarray(value)
    expected = (latent_dimension, latent_dimension)
    if covariance.dtype != np.dtype(np.float64) or covariance.shape != expected:
        raise ValueError(f"joint_bias_covariance must be float64 with shape {expected}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("joint_bias_covariance must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
        raise ValueError("joint_bias_covariance must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("joint_bias_covariance must be positive semidefinite")
    return cast(FloatArray, covariance)


def _bias_model_record(
    *,
    bias_ids: tuple[str, ...],
    basis_names: tuple[str, ...],
    covariance: np.ndarray,
    orthogonalization_semantics: str,
    gauge_projection_tolerance: float,
    model_metadata: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema": VISUAL_BIAS_MODEL_SCHEMA,
        "bias_ids": list(bias_ids),
        "basis_names": list(basis_names),
        "joint_bias_covariance": _array_descriptor(covariance),
        "orthogonalization_semantics": orthogonalization_semantics,
        "gauge_projection_tolerance": gauge_projection_tolerance,
        "model_metadata": plain_json(model_metadata),
    }


@dataclass(frozen=True)
class VisualBiasStreamUpdateV1:
    """One recursive observation update bound into an append-only hash chain."""

    bias_model_id: str
    observation_stream_update_id: str
    visual_bias_artifact_id: str
    observation_artifact_id: str
    observation_identity_sha256: str
    frame_start: int
    frame_stop_exclusive: int
    row_start: int
    row_stop_exclusive: int
    maximum_gauge_projection: float
    previous_update_id: str | None
    update_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "bias_model_id",
            "observation_stream_update_id",
            "visual_bias_artifact_id",
            "observation_artifact_id",
            "observation_identity_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        frame_start = require_exact_integer(
            self.frame_start,
            name="frame_start",
            minimum=0,
        )
        frame_stop = require_exact_integer(
            self.frame_stop_exclusive,
            name="frame_stop_exclusive",
            minimum=1,
        )
        row_start = require_exact_integer(self.row_start, name="row_start", minimum=0)
        row_stop = require_exact_integer(
            self.row_stop_exclusive,
            name="row_stop_exclusive",
            minimum=1,
        )
        if frame_stop <= frame_start:
            raise ValueError("visual-bias update frame interval must be nonempty")
        if row_stop <= row_start:
            raise ValueError("visual-bias update row interval must be nonempty")
        projection = _finite_nonnegative_real(
            self.maximum_gauge_projection,
            name="maximum_gauge_projection",
        )
        previous = self.previous_update_id
        if previous is not None:
            previous = require_sha256(previous, name="previous_update_id")
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop_exclusive", frame_stop)
        object.__setattr__(self, "row_start", row_start)
        object.__setattr__(self, "row_stop_exclusive", row_stop)
        object.__setattr__(self, "maximum_gauge_projection", projection)
        object.__setattr__(self, "previous_update_id", previous)
        expected = _sha256_json(self.identity_record())
        supplied = self.update_id
        if (
            supplied is not None
            and require_sha256(
                supplied,
                name="update_id",
            )
            != expected
        ):
            raise ValueError("visual-bias stream update ID mismatch")
        object.__setattr__(self, "update_id", expected)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": VISUAL_BIAS_STREAM_UPDATE_SCHEMA,
            "bias_model_id": self.bias_model_id,
            "observation_stream_update_id": self.observation_stream_update_id,
            "visual_bias_artifact_id": self.visual_bias_artifact_id,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "frame_start": self.frame_start,
            "frame_stop_exclusive": self.frame_stop_exclusive,
            "row_start": self.row_start,
            "row_stop_exclusive": self.row_stop_exclusive,
            "maximum_gauge_projection": self.maximum_gauge_projection,
            "previous_update_id": self.previous_update_id,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "update_id": self.update_id}

    @classmethod
    def from_record(cls, value: object) -> VisualBiasStreamUpdateV1:
        mapping = require_mapping(value, name="visual-bias stream update")
        require_exact_fields(mapping, _UPDATE_FIELDS, name="visual-bias stream update")
        if mapping["schema"] != VISUAL_BIAS_STREAM_UPDATE_SCHEMA:
            raise ValueError("unsupported visual-bias stream update schema")
        previous_value = mapping["previous_update_id"]
        previous_update_id = (
            None
            if previous_value is None
            else require_sha256(previous_value, name="previous_update_id")
        )
        return cls(
            bias_model_id=require_sha256(mapping["bias_model_id"], name="bias_model_id"),
            observation_stream_update_id=require_sha256(
                mapping["observation_stream_update_id"],
                name="observation_stream_update_id",
            ),
            visual_bias_artifact_id=require_sha256(
                mapping["visual_bias_artifact_id"],
                name="visual_bias_artifact_id",
            ),
            observation_artifact_id=require_sha256(
                mapping["observation_artifact_id"],
                name="observation_artifact_id",
            ),
            observation_identity_sha256=require_sha256(
                mapping["observation_identity_sha256"],
                name="observation_identity_sha256",
            ),
            frame_start=require_exact_integer(
                mapping["frame_start"],
                name="frame_start",
                minimum=0,
            ),
            frame_stop_exclusive=require_exact_integer(
                mapping["frame_stop_exclusive"],
                name="frame_stop_exclusive",
                minimum=1,
            ),
            row_start=require_exact_integer(
                mapping["row_start"],
                name="row_start",
                minimum=0,
            ),
            row_stop_exclusive=require_exact_integer(
                mapping["row_stop_exclusive"],
                name="row_stop_exclusive",
                minimum=1,
            ),
            maximum_gauge_projection=_finite_nonnegative_real(
                mapping["maximum_gauge_projection"],
                name="maximum_gauge_projection",
            ),
            previous_update_id=previous_update_id,
            update_id=require_sha256(mapping["update_id"], name="update_id"),
        )


@dataclass(frozen=True)
class VisualBiasNuisanceStreamV1:
    """Several causal updates sharing exactly one visual-bias latent prior."""

    stream_key: str
    bias_ids: tuple[str, ...]
    basis_names: tuple[str, ...]
    orthogonalization_semantics: str
    gauge_projection_tolerance: float
    updates: tuple[VisualBiasStreamUpdateV1, ...]
    row_update_indices: IntArray
    row_bias_indices: IntArray
    bias_jacobian: FloatArray
    joint_bias_covariance: FloatArray
    model_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bias_model_id: str | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        stream_key = require_exact_string(self.stream_key, name="stream_key")
        if type(self.bias_ids) is not tuple or type(self.basis_names) is not tuple:
            raise TypeError("bias_ids and basis_names must be canonical tuples")
        validated_bias_ids: tuple[str, ...] = require_string_sequence(
            self.bias_ids,
            name="bias_ids",
        )
        validated_basis_names: tuple[str, ...] = require_string_sequence(
            self.basis_names,
            name="basis_names",
        )
        if len(set(validated_bias_ids)) != len(validated_bias_ids):
            raise ValueError("bias_ids must be unique")
        if len(set(validated_basis_names)) != len(validated_basis_names):
            raise ValueError("basis_names must be unique")
        semantics = require_exact_string(
            self.orthogonalization_semantics,
            name="orthogonalization_semantics",
        )
        tolerance = _positive_real(
            self.gauge_projection_tolerance,
            name="gauge_projection_tolerance",
        )
        if (
            type(self.updates) is not tuple
            or not self.updates
            or not all(isinstance(update, VisualBiasStreamUpdateV1) for update in self.updates)
        ):
            raise ValueError("updates must be a nonempty tuple of VisualBiasStreamUpdateV1")

        update_ids = tuple(str(update.update_id) for update in self.updates)
        if len(update_ids) != len(set(update_ids)):
            raise ValueError("visual-bias stream update IDs must be unique")
        for attribute in (
            "observation_stream_update_id",
            "visual_bias_artifact_id",
            "observation_artifact_id",
            "observation_identity_sha256",
        ):
            values = tuple(getattr(update, attribute) for update in self.updates)
            if len(values) != len(set(values)):
                raise ValueError(f"visual-bias stream {attribute} values must be unique")
        for index, update in enumerate(self.updates):
            expected_previous = None if index == 0 else self.updates[index - 1].update_id
            if update.previous_update_id != expected_previous:
                raise ValueError("visual-bias stream previous-update chain is invalid")
            if index == 0:
                if update.row_start != 0:
                    raise ValueError("the first visual-bias update must start at row zero")
            else:
                previous = self.updates[index - 1]
                if update.row_start != previous.row_stop_exclusive:
                    raise ValueError("visual-bias update row intervals must be contiguous")
                if update.frame_start < previous.frame_stop_exclusive:
                    raise ValueError("visual-bias update frame intervals must not overlap")
        if self.updates[-1].row_stop_exclusive < 1:
            raise ValueError("visual-bias stream must contain rows")

        row_update = np.asarray(self.row_update_indices)
        row_bias = np.asarray(self.row_bias_indices)
        row_count = self.updates[-1].row_stop_exclusive
        if row_update.dtype != np.dtype(np.int64) or row_update.shape != (row_count,):
            raise ValueError("row_update_indices must be int64 with one entry per row")
        if row_bias.dtype != np.dtype(np.int64) or row_bias.shape != (row_count,):
            raise ValueError("row_bias_indices must be int64 with one entry per row")
        if np.any(row_update < 0) or np.any(row_update >= len(self.updates)):
            raise ValueError("row_update_indices refer to an unknown update")
        if np.any(row_bias < 0) or np.any(row_bias >= len(validated_bias_ids)):
            raise ValueError("row_bias_indices refer to an unknown bias ID")
        for index, update in enumerate(self.updates):
            expected = cast(
                IntArray,
                np.full(
                    update.row_stop_exclusive - update.row_start,
                    index,
                    dtype=np.int64,
                ),
            )
            if not np.array_equal(
                row_update[update.row_start : update.row_stop_exclusive],
                expected,
            ):
                raise ValueError("row_update_indices contradict update row intervals")

        jacobian = np.asarray(self.bias_jacobian)
        expected_jacobian = (row_count, 3, len(validated_basis_names))
        if jacobian.dtype != np.dtype(np.float64) or jacobian.shape != expected_jacobian:
            raise ValueError(f"bias_jacobian must be float64 with shape {expected_jacobian}")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("bias_jacobian must be finite")
        latent_dimension = len(validated_bias_ids) * len(validated_basis_names)
        covariance = _validate_joint_covariance(
            self.joint_bias_covariance,
            latent_dimension=latent_dimension,
        )
        validated_model_metadata: Mapping[str, Any] = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.model_metadata,
                name="visual-bias model metadata",
            ),
            name="visual-bias model metadata",
        )
        validated_metadata: Mapping[str, Any] = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="visual-bias stream metadata",
            ),
            name="visual-bias stream metadata",
        )
        model_record = _bias_model_record(
            bias_ids=validated_bias_ids,
            basis_names=validated_basis_names,
            covariance=covariance,
            orthogonalization_semantics=semantics,
            gauge_projection_tolerance=tolerance,
            model_metadata=validated_model_metadata,
        )
        expected_model_id = _sha256_json(model_record)
        if (
            self.bias_model_id is not None
            and require_sha256(
                self.bias_model_id,
                name="bias_model_id",
            )
            != expected_model_id
        ):
            raise ValueError("visual-bias model ID mismatch")
        for update in self.updates:
            if update.bias_model_id != expected_model_id:
                raise ValueError("visual-bias update belongs to another bias model")
            if (
                semantics == "conditional-whitened-global-gauge-projection-v1"
                and update.maximum_gauge_projection > tolerance
            ):
                raise ValueError("visual-bias update exceeds the shared gauge tolerance")

        object.__setattr__(self, "stream_key", stream_key)
        object.__setattr__(self, "bias_ids", validated_bias_ids)
        object.__setattr__(self, "basis_names", validated_basis_names)
        object.__setattr__(self, "orthogonalization_semantics", semantics)
        object.__setattr__(self, "gauge_projection_tolerance", tolerance)
        object.__setattr__(
            self,
            "row_update_indices",
            _readonly(row_update, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "row_bias_indices",
            _readonly(row_bias, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "bias_jacobian",
            _readonly(jacobian, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "joint_bias_covariance",
            _readonly(covariance, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "model_metadata", validated_model_metadata)
        object.__setattr__(self, "metadata", validated_metadata)
        object.__setattr__(self, "bias_model_id", expected_model_id)
        expected_artifact_id = _sha256_json(self.identity_record())
        if (
            self.artifact_id is not None
            and require_sha256(
                self.artifact_id,
                name="artifact_id",
            )
            != expected_artifact_id
        ):
            raise ValueError("visual-bias stream artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_artifact_id)

    @property
    def observation_count(self) -> int:
        return int(self.row_bias_indices.size)

    @property
    def basis_dimension(self) -> int:
        return len(self.basis_names)

    @property
    def latent_dimension(self) -> int:
        return len(self.bias_ids) * self.basis_dimension

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "row_update_indices": np.asarray(self.row_update_indices),
            "row_bias_indices": np.asarray(self.row_bias_indices),
            "bias_jacobian": np.asarray(self.bias_jacobian),
            "joint_bias_covariance": np.asarray(self.joint_bias_covariance),
        }

    def array_descriptors(self) -> dict[str, dict[str, object]]:
        return {name: _array_descriptor(value) for name, value in self.arrays().items()}

    def model_record(self) -> dict[str, object]:
        return _bias_model_record(
            bias_ids=self.bias_ids,
            basis_names=self.basis_names,
            covariance=self.joint_bias_covariance,
            orthogonalization_semantics=self.orthogonalization_semantics,
            gauge_projection_tolerance=self.gauge_projection_tolerance,
            model_metadata=self.model_metadata,
        )

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": VISUAL_BIAS_STREAM_SCHEMA,
            "schema_version": VISUAL_BIAS_STREAM_VERSION,
            "stream_key": self.stream_key,
            "bias_model_id": self.bias_model_id,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
            "updates": [update.to_record() for update in self.updates],
            "arrays": self.array_descriptors(),
            "model_metadata": plain_json(self.model_metadata),
            "metadata": plain_json(self.metadata),
            "claim_boundary": VISUAL_BIAS_STREAM_CLAIM_BOUNDARY,
        }

    def global_design(self) -> FloatArray:
        result: FloatArray = np.zeros(
            (3 * self.observation_count, self.latent_dimension),
            dtype=np.float64,
        )
        width = self.basis_dimension
        for row, bias_index in enumerate(self.row_bias_indices):
            start = int(bias_index) * width
            result[3 * row : 3 * row + 3, start : start + width] = self.bias_jacobian[row]
        result.setflags(write=False)
        return result

    def low_rank_factor(self, *, relative_tolerance: float = 1e-12) -> FloatArray:
        tolerance = _positive_real(relative_tolerance, name="relative_tolerance")
        eigenvalues, eigenvectors = np.linalg.eigh(self.joint_bias_covariance)
        maximum = max(float(np.max(eigenvalues)), 0.0)
        threshold = tolerance * max(maximum, 1.0)
        keep = eigenvalues > threshold
        if not np.any(keep):
            empty: FloatArray = np.zeros(
                (self.observation_count, 3, 0),
                dtype=np.float64,
            )
            empty.setflags(write=False)
            return empty
        root = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
        result: FloatArray = np.empty(
            (self.observation_count, 3, int(np.count_nonzero(keep))),
            dtype=np.float64,
        )
        width = self.basis_dimension
        for row, bias_index in enumerate(self.row_bias_indices):
            start = int(bias_index) * width
            result[row] = self.bias_jacobian[row] @ root[start : start + width]
        result.setflags(write=False)
        return result

    def summary(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "stream_key": self.stream_key,
            "bias_model_id": self.bias_model_id,
            "update_count": len(self.updates),
            "observation_count": self.observation_count,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "latent_dimension": self.latent_dimension,
            "first_frame": self.updates[0].frame_start,
            "last_frame_stop_exclusive": self.updates[-1].frame_stop_exclusive,
            "claim_boundary": VISUAL_BIAS_STREAM_CLAIM_BOUNDARY,
        }


def _shared_contract(nuisance: VisualBiasNuisanceV1) -> tuple[object, ...]:
    return (
        nuisance.bias_ids,
        nuisance.basis_names,
        nuisance.orthogonalization_semantics,
        nuisance.gauge_projection_tolerance,
    )


def _require_compatible_nuisance(
    nuisance: VisualBiasNuisanceV1,
    *,
    reference: VisualBiasNuisanceV1,
) -> None:
    if not isinstance(nuisance, VisualBiasNuisanceV1):
        raise TypeError("nuisances must contain VisualBiasNuisanceV1 values")
    if _shared_contract(nuisance) != _shared_contract(reference):
        raise ValueError("visual-bias sidecars do not share one bias model contract")
    if not np.array_equal(
        nuisance.joint_bias_covariance,
        reference.joint_bias_covariance,
    ):
        raise ValueError("visual-bias sidecars do not share the exact joint prior")


def build_visual_bias_nuisance_stream(
    *,
    stream_key: str,
    nuisances: Sequence[VisualBiasNuisanceV1],
    observation_stream_update_ids: Sequence[str],
    frame_intervals: Sequence[tuple[int, int]],
    model_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VisualBiasNuisanceStreamV1:
    """Bind several sidecars to one persistent visual-bias latent state."""

    nuisance_values = tuple(nuisances)
    update_ids = tuple(observation_stream_update_ids)
    intervals = tuple(frame_intervals)
    if not nuisance_values:
        raise ValueError("nuisances must not be empty")
    if len(nuisance_values) != len(update_ids) or len(nuisance_values) != len(intervals):
        raise ValueError("nuisances, update IDs, and frame intervals must have equal length")
    reference = nuisance_values[0]
    if not isinstance(reference, VisualBiasNuisanceV1):
        raise TypeError("nuisances must contain VisualBiasNuisanceV1 values")
    for nuisance in nuisance_values:
        _require_compatible_nuisance(nuisance, reference=reference)

    normalized_model_metadata = frozen_finite_json_mapping(
        {} if model_metadata is None else model_metadata,
        name="visual-bias model metadata",
    )
    model_record = _bias_model_record(
        bias_ids=reference.bias_ids,
        basis_names=reference.basis_names,
        covariance=reference.joint_bias_covariance,
        orthogonalization_semantics=reference.orthogonalization_semantics,
        gauge_projection_tolerance=reference.gauge_projection_tolerance,
        model_metadata=normalized_model_metadata,
    )
    model_id = _sha256_json(model_record)

    updates: list[VisualBiasStreamUpdateV1] = []
    row_updates: list[IntArray] = []
    row_bias: list[IntArray] = []
    jacobians: list[FloatArray] = []
    row_start = 0
    previous_update_id: str | None = None
    previous_frame_stop: int | None = None
    for index, (nuisance, observation_update_id, interval) in enumerate(
        zip(nuisance_values, update_ids, intervals, strict=True)
    ):
        if type(interval) is not tuple or len(interval) != 2:
            raise ValueError("frame intervals must be canonical (start, stop) tuples")
        frame_start = require_exact_integer(interval[0], name="frame_start", minimum=0)
        frame_stop = require_exact_integer(
            interval[1],
            name="frame_stop_exclusive",
            minimum=1,
        )
        if frame_stop <= frame_start:
            raise ValueError("frame intervals must be nonempty")
        if previous_frame_stop is not None and frame_start < previous_frame_stop:
            raise ValueError("frame intervals must not overlap")
        sidecar_id = nuisance.artifact_id
        if sidecar_id is None:
            raise ValueError("visual-bias sidecar lacks an artifact ID")
        row_stop = row_start + nuisance.observation_count
        update = VisualBiasStreamUpdateV1(
            bias_model_id=model_id,
            observation_stream_update_id=observation_update_id,
            visual_bias_artifact_id=sidecar_id,
            observation_artifact_id=nuisance.observation_artifact_id,
            observation_identity_sha256=nuisance.observation_identity_sha256,
            frame_start=frame_start,
            frame_stop_exclusive=frame_stop,
            row_start=row_start,
            row_stop_exclusive=row_stop,
            maximum_gauge_projection=nuisance.maximum_gauge_projection,
            previous_update_id=previous_update_id,
        )
        updates.append(update)
        row_updates.append(
            cast(
                IntArray,
                np.full(nuisance.observation_count, index, dtype=np.int64),
            )
        )
        row_bias.append(cast(IntArray, np.asarray(nuisance.row_bias_indices, dtype=np.int64)))
        jacobians.append(cast(FloatArray, np.asarray(nuisance.bias_jacobian, dtype=np.float64)))
        row_start = row_stop
        if update.update_id is None:
            raise AssertionError("validated visual-bias update lacks an ID")
        previous_update_id = update.update_id
        previous_frame_stop = frame_stop

    return VisualBiasNuisanceStreamV1(
        stream_key=stream_key,
        bias_ids=reference.bias_ids,
        basis_names=reference.basis_names,
        orthogonalization_semantics=reference.orthogonalization_semantics,
        gauge_projection_tolerance=reference.gauge_projection_tolerance,
        updates=tuple(updates),
        row_update_indices=np.concatenate(row_updates),
        row_bias_indices=np.concatenate(row_bias),
        bias_jacobian=np.concatenate(jacobians, axis=0),
        joint_bias_covariance=reference.joint_bias_covariance,
        model_metadata=normalized_model_metadata,
        metadata={} if metadata is None else metadata,
        bias_model_id=model_id,
    )


def append_visual_bias_nuisance(
    stream: VisualBiasNuisanceStreamV1,
    nuisance: VisualBiasNuisanceV1,
    *,
    observation_stream_update_id: str,
    frame_interval: tuple[int, int],
) -> VisualBiasNuisanceStreamV1:
    """Return one strict append while preserving every existing update ID."""

    if not isinstance(stream, VisualBiasNuisanceStreamV1):
        raise TypeError("stream must be a VisualBiasNuisanceStreamV1")
    reference = VisualBiasNuisanceV1(
        observation_artifact_id=stream.updates[0].observation_artifact_id,
        observation_identity_sha256=stream.updates[0].observation_identity_sha256,
        bias_ids=stream.bias_ids,
        basis_names=stream.basis_names,
        row_bias_indices=stream.row_bias_indices[
            stream.updates[0].row_start : stream.updates[0].row_stop_exclusive
        ],
        bias_jacobian=stream.bias_jacobian[
            stream.updates[0].row_start : stream.updates[0].row_stop_exclusive
        ],
        joint_bias_covariance=stream.joint_bias_covariance,
        orthogonalization_semantics=stream.orthogonalization_semantics,
        maximum_gauge_projection=stream.updates[0].maximum_gauge_projection,
        gauge_projection_tolerance=stream.gauge_projection_tolerance,
    )
    _require_compatible_nuisance(nuisance, reference=reference)
    if type(frame_interval) is not tuple or len(frame_interval) != 2:
        raise ValueError("frame_interval must be a canonical (start, stop) tuple")
    frame_start = require_exact_integer(frame_interval[0], name="frame_start", minimum=0)
    frame_stop = require_exact_integer(
        frame_interval[1],
        name="frame_stop_exclusive",
        minimum=1,
    )
    if frame_start < stream.updates[-1].frame_stop_exclusive:
        raise ValueError("appended frame interval overlaps retained updates")
    if frame_stop <= frame_start:
        raise ValueError("appended frame interval must be nonempty")
    observation_update_id = require_sha256(
        observation_stream_update_id,
        name="observation_stream_update_id",
    )
    if observation_update_id in {update.observation_stream_update_id for update in stream.updates}:
        raise ValueError("observation_stream_update_id is already present")
    sidecar_id = nuisance.artifact_id
    if sidecar_id is None:
        raise ValueError("visual-bias sidecar lacks an artifact ID")
    row_start = stream.observation_count
    row_stop = row_start + nuisance.observation_count
    model_id = stream.bias_model_id
    if model_id is None:
        raise ValueError("visual-bias stream lacks a model ID")
    update = VisualBiasStreamUpdateV1(
        bias_model_id=model_id,
        observation_stream_update_id=observation_update_id,
        visual_bias_artifact_id=sidecar_id,
        observation_artifact_id=nuisance.observation_artifact_id,
        observation_identity_sha256=nuisance.observation_identity_sha256,
        frame_start=frame_start,
        frame_stop_exclusive=frame_stop,
        row_start=row_start,
        row_stop_exclusive=row_stop,
        maximum_gauge_projection=nuisance.maximum_gauge_projection,
        previous_update_id=stream.updates[-1].update_id,
    )
    return VisualBiasNuisanceStreamV1(
        stream_key=stream.stream_key,
        bias_ids=stream.bias_ids,
        basis_names=stream.basis_names,
        orthogonalization_semantics=stream.orthogonalization_semantics,
        gauge_projection_tolerance=stream.gauge_projection_tolerance,
        updates=(*stream.updates, update),
        row_update_indices=np.concatenate(
            (
                stream.row_update_indices,
                np.full(nuisance.observation_count, len(stream.updates), dtype=np.int64),
            )
        ),
        row_bias_indices=np.concatenate((stream.row_bias_indices, nuisance.row_bias_indices)),
        bias_jacobian=np.concatenate((stream.bias_jacobian, nuisance.bias_jacobian), axis=0),
        joint_bias_covariance=stream.joint_bias_covariance,
        model_metadata=stream.model_metadata,
        metadata=stream.metadata,
        bias_model_id=stream.bias_model_id,
    )


def _validate_array_descriptor(value: object, array: np.ndarray, *, name: str) -> None:
    mapping = require_mapping(value, name=f"{name} descriptor")
    require_exact_fields(mapping, _ARRAY_FIELDS, name=f"{name} descriptor")
    if dict(mapping) != _array_descriptor(array):
        raise ValueError(f"{name} descriptor does not match payload bytes")


def write_visual_bias_nuisance_stream(
    stream: VisualBiasNuisanceStreamV1,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist one content-addressed stream with a fail-closed writer lock."""

    if not isinstance(stream, VisualBiasNuisanceStreamV1):
        raise TypeError("stream must be a VisualBiasNuisanceStreamV1")
    manifest = Path(manifest_path)
    payload = Path(payload_path) if payload_path is not None else manifest.with_suffix(".npz")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_payload = payload.resolve().relative_to(manifest.parent.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "visual-bias stream payload must lie inside the manifest directory"
        ) from error
    relative_payload = _safe_relative_path(relative_payload, name="payload path")
    lock = manifest.with_name(f".{manifest.name}.lock")
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError("visual-bias stream writer is already active") from error
    os.close(lock_descriptor)
    try:
        if manifest.exists() or payload.exists():
            if not manifest.exists() or not payload.exists():
                raise ValueError("visual-bias stream destination is a partial artifact")
            existing = load_visual_bias_nuisance_stream(manifest)
            if existing.artifact_id != stream.artifact_id:
                raise ValueError("refusing to replace a different visual-bias stream")
            return manifest, payload
        _atomic_write_npz(payload, stream.arrays())
        payload_bytes = payload.read_bytes()
        record = {
            **stream.identity_record(),
            "artifact_id": stream.artifact_id,
            "payload": {
                "path": relative_payload,
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "byte_count": len(payload_bytes),
                "allow_pickle": False,
            },
        }
        _atomic_write_json(manifest, record)
        return manifest, payload
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(manifest.parent)


def load_visual_bias_nuisance_stream(path: str | Path) -> VisualBiasNuisanceStreamV1:
    """Load one strict stream from an exact non-pickled payload snapshot."""

    manifest_path = Path(path).resolve()
    record = load_json_object(manifest_path, name="visual-bias stream manifest")
    require_exact_fields(record, _MANIFEST_FIELDS, name="visual-bias stream manifest")
    if record["schema"] != VISUAL_BIAS_STREAM_SCHEMA:
        raise ValueError("unsupported visual-bias stream schema")
    if record["schema_version"] != VISUAL_BIAS_STREAM_VERSION:
        raise ValueError("unsupported visual-bias stream version")
    if record["claim_boundary"] != VISUAL_BIAS_STREAM_CLAIM_BOUNDARY:
        raise ValueError("visual-bias stream claim boundary changed")
    payload_record = require_mapping(record["payload"], name="payload")
    require_exact_fields(payload_record, _PAYLOAD_FIELDS, name="payload")
    if payload_record["allow_pickle"] is not False:
        raise ValueError("visual-bias stream payload must require allow_pickle=False")
    relative_payload = _safe_relative_path(payload_record["path"], name="payload path")
    payload_path = _resolved_member(
        manifest_path.parent,
        relative_payload,
        name="payload path",
    )
    try:
        payload_bytes = payload_path.read_bytes()
    except OSError as error:
        raise ValueError("visual-bias stream payload is unreadable") from error
    expected_bytes = require_exact_integer(
        payload_record["byte_count"],
        name="payload byte_count",
        minimum=1,
    )
    if len(payload_bytes) != expected_bytes:
        raise ValueError("visual-bias stream payload byte count mismatch")
    if hashlib.sha256(payload_bytes).hexdigest() != require_sha256(
        payload_record["sha256"],
        name="payload sha256",
    ):
        raise ValueError("visual-bias stream payload SHA-256 mismatch")
    try:
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
            if set(payload.files) != set(_ARRAY_NAMES):
                raise ValueError("visual-bias stream NPZ member set changed")
            row_update = np.asarray(payload["row_update_indices"])
            row_bias = np.asarray(payload["row_bias_indices"])
            jacobian = np.asarray(payload["bias_jacobian"])
            covariance = np.asarray(payload["joint_bias_covariance"])
    except (OSError, ValueError) as error:
        raise ValueError("visual-bias stream payload is invalid") from error
    arrays_record = require_mapping(record["arrays"], name="arrays")
    require_exact_fields(arrays_record, frozenset(_ARRAY_NAMES), name="arrays")
    for name, array in {
        "row_update_indices": row_update,
        "row_bias_indices": row_bias,
        "bias_jacobian": jacobian,
        "joint_bias_covariance": covariance,
    }.items():
        _validate_array_descriptor(arrays_record[name], array, name=name)
    update_values = record["updates"]
    if not isinstance(update_values, list):
        raise ValueError("updates must be a JSON array")
    stream_key = require_exact_string(record["stream_key"], name="stream_key")
    orthogonalization_semantics = require_exact_string(
        record["orthogonalization_semantics"],
        name="orthogonalization_semantics",
    )
    gauge_projection_tolerance = _positive_real(
        record["gauge_projection_tolerance"],
        name="gauge_projection_tolerance",
    )
    bias_model_id = require_sha256(record["bias_model_id"], name="bias_model_id")
    artifact_id = require_sha256(record["artifact_id"], name="artifact_id")
    stream = VisualBiasNuisanceStreamV1(
        stream_key=stream_key,
        bias_ids=require_string_sequence(record["bias_ids"], name="bias_ids"),
        basis_names=require_string_sequence(record["basis_names"], name="basis_names"),
        orthogonalization_semantics=orthogonalization_semantics,
        gauge_projection_tolerance=gauge_projection_tolerance,
        updates=tuple(VisualBiasStreamUpdateV1.from_record(value) for value in update_values),
        row_update_indices=row_update,
        row_bias_indices=row_bias,
        bias_jacobian=jacobian,
        joint_bias_covariance=covariance,
        model_metadata=require_finite_json_mapping(
            record["model_metadata"],
            name="visual-bias model metadata",
        ),
        metadata=require_finite_json_mapping(
            record["metadata"],
            name="visual-bias stream metadata",
        ),
        bias_model_id=bias_model_id,
        artifact_id=artifact_id,
    )
    if stream.array_descriptors() != dict(arrays_record):
        raise ValueError("visual-bias stream array descriptors changed after validation")
    return stream


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d observation visual-bias-stream",
        description="Validate recursive visual-bias nuisance streams.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a stream and payload")
    validate.add_argument("manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "validate":
        stream = load_visual_bias_nuisance_stream(arguments.manifest)
        print(json.dumps(stream.summary(), indent=2, sort_keys=True))
        return 0
    parser.error("unsupported visual-bias stream command")
    return 2


__all__ = [
    "VISUAL_BIAS_MODEL_SCHEMA",
    "VISUAL_BIAS_STREAM_CLAIM_BOUNDARY",
    "VISUAL_BIAS_STREAM_SCHEMA",
    "VISUAL_BIAS_STREAM_UPDATE_SCHEMA",
    "VISUAL_BIAS_STREAM_VERSION",
    "VisualBiasNuisanceStreamV1",
    "VisualBiasStreamUpdateV1",
    "append_visual_bias_nuisance",
    "build_visual_bias_nuisance_stream",
    "load_visual_bias_nuisance_stream",
    "main",
    "write_visual_bias_nuisance_stream",
]


if __name__ == "__main__":
    raise SystemExit(main())
