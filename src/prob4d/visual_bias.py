"""Explicit low-rank visual-bias nuisance factors for Prob4D observations.

The sidecar binds an existing observation artifact and exact row ordering while
keeping coherent visual bias separate from conditional point covariance and from
the existing Sim(3) gauge nuisance.  It is additive infrastructure: downstream
estimators must bind and consume the sidecar explicitly rather than silently
changing a frozen observation-factor schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
    require_string_sequence,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

VISUAL_BIAS_NUISANCE_SCHEMA: Final = "prob4d.visual-bias-nuisance"
VISUAL_BIAS_NUISANCE_VERSION: Final = 1
ORTHOGONALIZATION_SEMANTICS: Final = (
    "not-orthogonalized",
    "conditional-whitened-global-gauge-projection-v1",
)

_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "observation_artifact_id",
        "observation_identity_sha256",
        "bias_ids",
        "basis_names",
        "orthogonalization_semantics",
        "maximum_gauge_projection",
        "gauge_projection_tolerance",
        "payload",
        "arrays",
        "metadata",
    }
)
_PAYLOAD_FIELDS: Final = frozenset({"path", "sha256", "byte_count", "allow_pickle"})
_ARRAY_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})
_ARRAY_NAMES: Final = (
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read visual-bias payload {path.name!r}") from error
    return digest.hexdigest()


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


def _require_finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _safe_relative_path(value: object, *, name: str) -> str:
    path = require_exact_string(value, name=name)
    if "\\" in path:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return pure.as_posix()


def _resolved_member(root: Path, relative_path: str, *, name: str) -> Path:
    safe = _safe_relative_path(relative_path, name=name)
    root_resolved = root.resolve()
    current = root_resolved
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    candidate = current.resolve()
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
            np.savez_compressed(stream, **arrays)
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


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_spd_blocks(covariance: np.ndarray) -> None:
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError("conditional_covariance must have shape (N, 3, 3)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("conditional_covariance must be finite")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError("conditional_covariance must be symmetric")
    for index, block in enumerate(covariance):
        try:
            np.linalg.cholesky(block)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"conditional_covariance block {index} must be positive definite"
            ) from error


@dataclass(frozen=True)
class OrthogonalizedVisualBiasBasis:
    """Result of projecting a shared bias basis out of the gauge-design span."""

    bias_jacobian: FloatArray
    gauge_rank: int
    maximum_projection_before: float
    maximum_projection_after: float

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.bias_jacobian, dtype=np.float64)
        if jacobian.ndim != 3 or jacobian.shape[1] != 3 or jacobian.shape[2] < 1:
            raise ValueError("bias_jacobian must have shape (N, 3, R)")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("bias_jacobian must be finite")
        if isinstance(self.gauge_rank, bool) or not isinstance(self.gauge_rank, int):
            raise ValueError("gauge_rank must be an integer")
        if self.gauge_rank < 0:
            raise ValueError("gauge_rank must be non-negative")
        before = _require_finite_real(
            self.maximum_projection_before,
            name="maximum_projection_before",
            minimum=0.0,
        )
        after = _require_finite_real(
            self.maximum_projection_after,
            name="maximum_projection_after",
            minimum=0.0,
        )
        object.__setattr__(
            self,
            "bias_jacobian",
            _readonly(jacobian, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "maximum_projection_before", before)
        object.__setattr__(self, "maximum_projection_after", after)


def _maximum_column_projection(
    basis: np.ndarray,
    orthonormal_gauge: np.ndarray,
) -> float:
    if basis.shape[1] == 0 or orthonormal_gauge.shape[1] == 0:
        return 0.0
    values: list[float] = []
    for column in range(basis.shape[1]):
        vector = basis[:, column]
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float64).eps:
            values.append(0.0)
            continue
        projection = orthonormal_gauge.T @ vector
        values.append(float(np.linalg.norm(projection) / norm))
    return max(values, default=0.0)


def orthogonalize_visual_bias_basis(
    bias_jacobian: np.ndarray,
    gauge_design: np.ndarray,
    conditional_covariance: np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
) -> OrthogonalizedVisualBiasBasis:
    """Project a global bias basis out of the conditional-whitened gauge span.

    ``gauge_design`` may contain the complete block-sparse global gauge design,
    not merely one row-local 3x7 Jacobian.  This makes the operation meaningful
    even when every individual 3-D row is locally full rank.
    """

    bias = np.asarray(bias_jacobian, dtype=np.float64)
    gauge = np.asarray(gauge_design, dtype=np.float64)
    covariance = np.asarray(conditional_covariance, dtype=np.float64)
    if bias.ndim != 3 or bias.shape[1] != 3 or bias.shape[2] < 1:
        raise ValueError("bias_jacobian must have shape (N, 3, R)")
    if gauge.ndim != 3 or gauge.shape[:2] != bias.shape[:2]:
        raise ValueError("gauge_design must have shape (N, 3, K)")
    if gauge.shape[2] < 1:
        raise ValueError("gauge_design must contain at least one gauge column")
    if not np.all(np.isfinite(bias)) or not np.all(np.isfinite(gauge)):
        raise ValueError("bias and gauge designs must be finite")
    if covariance.shape != (bias.shape[0], 3, 3):
        raise ValueError("conditional_covariance shape differs from bias rows")
    _validate_spd_blocks(covariance)
    tolerance = _require_finite_real(
        relative_tolerance,
        name="relative_tolerance",
        strictly_positive=True,
    )

    whitened_bias: list[np.ndarray] = []
    whitened_gauge: list[np.ndarray] = []
    cholesky_blocks: list[np.ndarray] = []
    for index in range(bias.shape[0]):
        cholesky = np.linalg.cholesky(covariance[index])
        cholesky_blocks.append(cholesky)
        whitened_bias.append(np.linalg.solve(cholesky, bias[index]))
        whitened_gauge.append(np.linalg.solve(cholesky, gauge[index]))
    stacked_bias = np.concatenate(whitened_bias, axis=0)
    stacked_gauge = np.concatenate(whitened_gauge, axis=0)

    left, singular_values, _ = np.linalg.svd(stacked_gauge, full_matrices=False)
    if singular_values.size == 0:
        rank = 0
    else:
        threshold = tolerance * float(singular_values[0])
        rank = int(np.sum(singular_values > threshold))
    gauge_basis = left[:, :rank]
    before = _maximum_column_projection(stacked_bias, gauge_basis)
    residual = stacked_bias - gauge_basis @ (gauge_basis.T @ stacked_bias)
    after = _maximum_column_projection(residual, gauge_basis)

    rows: list[np.ndarray] = []
    for index, cholesky in enumerate(cholesky_blocks):
        start = 3 * index
        rows.append(cholesky @ residual[start : start + 3])
    return OrthogonalizedVisualBiasBasis(
        bias_jacobian=np.stack(rows),
        gauge_rank=rank,
        maximum_projection_before=before,
        maximum_projection_after=after,
    )


@dataclass(frozen=True)
class VisualBiasNuisanceV1:
    """Portable explicit shared-bias Jacobians and their complete joint prior."""

    observation_artifact_id: str
    observation_identity_sha256: str
    bias_ids: tuple[str, ...]
    basis_names: tuple[str, ...]
    row_bias_indices: IntArray
    bias_jacobian: FloatArray
    joint_bias_covariance: FloatArray
    orthogonalization_semantics: str
    maximum_gauge_projection: float
    gauge_projection_tolerance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        observation_artifact_id = require_sha256(
            self.observation_artifact_id,
            name="observation_artifact_id",
        )
        observation_identity_sha256 = require_sha256(
            self.observation_identity_sha256,
            name="observation_identity_sha256",
        )
        if type(self.bias_ids) is not tuple:
            raise TypeError("bias_ids must be a canonical tuple")
        if type(self.basis_names) is not tuple:
            raise TypeError("basis_names must be a canonical tuple")
        bias_ids = require_string_sequence(self.bias_ids, name="bias_ids")
        basis_names = require_string_sequence(self.basis_names, name="basis_names")
        if len(set(bias_ids)) != len(bias_ids):
            raise ValueError("bias_ids must be unique")
        if len(set(basis_names)) != len(basis_names):
            raise ValueError("basis_names must be unique")

        row_indices = np.asarray(self.row_bias_indices)
        if row_indices.dtype != np.dtype(np.int64) or row_indices.ndim != 1:
            raise ValueError("row_bias_indices must be a one-dimensional int64 array")
        if row_indices.size < 1:
            raise ValueError("visual-bias sidecar requires at least one observation row")
        if np.any(row_indices < 0) or np.any(row_indices >= len(bias_ids)):
            raise ValueError("row_bias_indices refer to an unknown bias ID")

        jacobian = np.asarray(self.bias_jacobian)
        expected_jacobian_shape = (row_indices.size, 3, len(basis_names))
        if jacobian.dtype != np.dtype(np.float64) or jacobian.shape != expected_jacobian_shape:
            raise ValueError(
                "bias_jacobian must be float64 with shape "
                f"{expected_jacobian_shape}"
            )
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("bias_jacobian must be finite")

        covariance = np.asarray(self.joint_bias_covariance)
        latent_dimension = len(bias_ids) * len(basis_names)
        expected_covariance_shape = (latent_dimension, latent_dimension)
        if covariance.dtype != np.dtype(np.float64) or covariance.shape != (
            expected_covariance_shape
        ):
            raise ValueError(
                "joint_bias_covariance must be float64 with shape "
                f"{expected_covariance_shape}"
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("joint_bias_covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("joint_bias_covariance must be symmetric")
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) < -1e-10 * scale:
            raise ValueError("joint_bias_covariance must be positive semidefinite")

        semantics = require_exact_string(
            self.orthogonalization_semantics,
            name="orthogonalization_semantics",
        )
        if semantics not in ORTHOGONALIZATION_SEMANTICS:
            raise ValueError(
                "orthogonalization_semantics must be one of "
                + ", ".join(ORTHOGONALIZATION_SEMANTICS)
            )
        maximum_projection = _require_finite_real(
            self.maximum_gauge_projection,
            name="maximum_gauge_projection",
            minimum=0.0,
        )
        projection_tolerance = _require_finite_real(
            self.gauge_projection_tolerance,
            name="gauge_projection_tolerance",
            strictly_positive=True,
        )
        if (
            semantics == "conditional-whitened-global-gauge-projection-v1"
            and maximum_projection > projection_tolerance
        ):
            raise ValueError("orthogonalized bias basis exceeds its gauge projection tolerance")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="visual-bias metadata",
            ),
            name="visual-bias metadata",
        )

        object.__setattr__(self, "observation_artifact_id", observation_artifact_id)
        object.__setattr__(
            self,
            "observation_identity_sha256",
            observation_identity_sha256,
        )
        object.__setattr__(self, "bias_ids", bias_ids)
        object.__setattr__(self, "basis_names", basis_names)
        object.__setattr__(
            self,
            "row_bias_indices",
            _readonly(row_indices, dtype=np.dtype(np.int64)),
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
        object.__setattr__(self, "orthogonalization_semantics", semantics)
        object.__setattr__(self, "maximum_gauge_projection", maximum_projection)
        object.__setattr__(self, "gauge_projection_tolerance", projection_tolerance)
        object.__setattr__(self, "metadata", metadata)

        expected_id = _sha256_json(self.identity_record())
        supplied = self.artifact_id
        if supplied is not None and require_sha256(
            supplied,
            name="artifact_id",
        ) != expected_id:
            raise ValueError("visual-bias nuisance artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

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
            "row_bias_indices": np.asarray(self.row_bias_indices),
            "bias_jacobian": np.asarray(self.bias_jacobian),
            "joint_bias_covariance": np.asarray(self.joint_bias_covariance),
        }

    def array_descriptors(self) -> dict[str, dict[str, object]]:
        return {name: _array_descriptor(value) for name, value in self.arrays().items()}

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": VISUAL_BIAS_NUISANCE_SCHEMA,
            "schema_version": VISUAL_BIAS_NUISANCE_VERSION,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "maximum_gauge_projection": self.maximum_gauge_projection,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
            "arrays": self.array_descriptors(),
            "metadata": plain_json(self.metadata),
        }

    def global_design(self) -> FloatArray:
        """Return the explicit block-sparse design with shape (3N, S*R)."""

        result = np.zeros(
            (3 * self.observation_count, self.latent_dimension),
            dtype=np.float64,
        )
        width = self.basis_dimension
        for row, bias_index in enumerate(self.row_bias_indices):
            start = int(bias_index) * width
            result[3 * row : 3 * row + 3, start : start + width] = (
                self.bias_jacobian[row]
            )
        result.setflags(write=False)
        return result

    def low_rank_factor(self, *, relative_tolerance: float = 1e-12) -> FloatArray:
        """Return rowwise factors U such that B Sigma B' = U U'."""

        tolerance = _require_finite_real(
            relative_tolerance,
            name="relative_tolerance",
            strictly_positive=True,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(self.joint_bias_covariance)
        maximum = max(float(np.max(eigenvalues)), 0.0)
        threshold = tolerance * max(maximum, 1.0)
        keep = eigenvalues > threshold
        if not np.any(keep):
            result = np.zeros((self.observation_count, 3, 0), dtype=np.float64)
            result.setflags(write=False)
            return result
        root = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
        result = np.empty(
            (self.observation_count, 3, int(np.count_nonzero(keep))),
            dtype=np.float64,
        )
        width = self.basis_dimension
        for row, bias_index in enumerate(self.row_bias_indices):
            start = int(bias_index) * width
            result[row] = self.bias_jacobian[row] @ root[start : start + width]
        result.setflags(write=False)
        return result

    def marginal_covariance(self) -> FloatArray:
        factor = self.low_rank_factor()
        covariance = np.einsum("nir,njr->nij", factor, factor)
        covariance.setflags(write=False)
        return covariance

    def summary(self) -> dict[str, object]:
        eigenvalues = np.linalg.eigvalsh(self.joint_bias_covariance)
        return {
            "artifact_id": self.artifact_id,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_identity_sha256": self.observation_identity_sha256,
            "observation_count": self.observation_count,
            "bias_ids": list(self.bias_ids),
            "basis_names": list(self.basis_names),
            "latent_dimension": self.latent_dimension,
            "effective_prior_rank": int(np.count_nonzero(eigenvalues > 1e-12)),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "maximum_gauge_projection": self.maximum_gauge_projection,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
        }


def _validate_array_descriptor(
    value: object,
    array: np.ndarray,
    *,
    name: str,
) -> None:
    mapping = require_mapping(value, name=f"{name} descriptor")
    require_exact_fields(mapping, _ARRAY_FIELDS, name=f"{name} descriptor")
    expected = _array_descriptor(array)
    if dict(mapping) != expected:
        raise ValueError(f"{name} descriptor does not match payload bytes")


def write_visual_bias_nuisance(
    nuisance: VisualBiasNuisanceV1,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write one strict non-pickled sidecar and atomic manifest."""

    manifest = Path(manifest_path)
    payload = Path(payload_path) if payload_path is not None else manifest.with_suffix(".npz")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_payload = payload.resolve().relative_to(manifest.parent.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("visual-bias payload must lie inside the manifest directory") from error
    relative_payload = _safe_relative_path(relative_payload, name="payload path")

    if manifest.exists() or payload.exists():
        if not manifest.exists() or not payload.exists():
            raise ValueError("visual-bias destination contains a partial retained artifact")
        existing = load_visual_bias_nuisance(manifest)
        if existing.artifact_id != nuisance.artifact_id:
            raise ValueError("refusing to replace a different visual-bias nuisance")
        return manifest, payload

    arrays = nuisance.arrays()
    _atomic_write_npz(payload, arrays)
    payload_sha = _sha256_file(payload)
    payload_bytes = payload.stat().st_size
    record = {
        **nuisance.identity_record(),
        "artifact_id": nuisance.artifact_id,
        "payload": {
            "path": relative_payload,
            "sha256": payload_sha,
            "byte_count": payload_bytes,
            "allow_pickle": False,
        },
    }
    _atomic_write_json(manifest, record)
    return manifest, payload


def load_visual_bias_nuisance(path: str | Path) -> VisualBiasNuisanceV1:
    manifest_path = Path(path).resolve()
    record = load_json_object(manifest_path, name="visual-bias nuisance manifest")
    require_exact_fields(record, _MANIFEST_FIELDS, name="visual-bias nuisance manifest")
    if record["schema"] != VISUAL_BIAS_NUISANCE_SCHEMA:
        raise ValueError("unsupported visual-bias nuisance schema")
    if record["schema_version"] != VISUAL_BIAS_NUISANCE_VERSION:
        raise ValueError("unsupported visual-bias nuisance version")
    payload_record = require_mapping(record["payload"], name="visual-bias payload")
    require_exact_fields(payload_record, _PAYLOAD_FIELDS, name="visual-bias payload")
    if type(payload_record["allow_pickle"]) is not bool or payload_record["allow_pickle"]:
        raise ValueError("visual-bias payload must declare allow_pickle=false")
    payload_relative = _safe_relative_path(payload_record["path"], name="payload path")
    payload_sha = require_sha256(payload_record["sha256"], name="payload sha256")
    byte_count = payload_record["byte_count"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise ValueError("payload byte_count must be a positive integer")
    payload_path = _resolved_member(
        manifest_path.parent,
        payload_relative,
        name="visual-bias payload path",
    )
    if not payload_path.is_file():
        raise ValueError("visual-bias payload is missing")
    if payload_path.stat().st_size != byte_count:
        raise ValueError("visual-bias payload byte count mismatch")
    if _sha256_file(payload_path) != payload_sha:
        raise ValueError("visual-bias payload SHA-256 mismatch")

    arrays_record = require_mapping(record["arrays"], name="visual-bias arrays")
    if set(arrays_record) != set(_ARRAY_NAMES):
        raise ValueError("visual-bias array member set changed")
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != set(_ARRAY_NAMES):
            raise ValueError("visual-bias NPZ member set changed")
        row_indices = np.asarray(payload["row_bias_indices"])
        jacobian = np.asarray(payload["bias_jacobian"])
        covariance = np.asarray(payload["joint_bias_covariance"])
    for name, array in {
        "row_bias_indices": row_indices,
        "bias_jacobian": jacobian,
        "joint_bias_covariance": covariance,
    }.items():
        _validate_array_descriptor(arrays_record[name], array, name=name)

    bias_ids = require_string_sequence(record["bias_ids"], name="bias_ids")
    basis_names = require_string_sequence(record["basis_names"], name="basis_names")
    metadata = require_finite_json_mapping(record["metadata"], name="visual-bias metadata")
    nuisance = VisualBiasNuisanceV1(
        observation_artifact_id=record["observation_artifact_id"],
        observation_identity_sha256=record["observation_identity_sha256"],
        bias_ids=bias_ids,
        basis_names=basis_names,
        row_bias_indices=row_indices,
        bias_jacobian=jacobian,
        joint_bias_covariance=covariance,
        orthogonalization_semantics=record["orthogonalization_semantics"],
        maximum_gauge_projection=record["maximum_gauge_projection"],
        gauge_projection_tolerance=record["gauge_projection_tolerance"],
        metadata=metadata,
        artifact_id=record["artifact_id"],
    )
    if nuisance.array_descriptors() != dict(arrays_record):
        raise ValueError("visual-bias array descriptors changed after validation")
    return nuisance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d observation visual-bias",
        description="Validate explicit low-rank visual-bias nuisance sidecars.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a sidecar and payload")
    validate.add_argument("manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "validate":
        nuisance = load_visual_bias_nuisance(arguments.manifest)
        print(json.dumps(nuisance.summary(), indent=2, sort_keys=True))
        return 0
    parser.error("unsupported visual-bias command")
    return 2


__all__ = [
    "ORTHOGONALIZATION_SEMANTICS",
    "OrthogonalizedVisualBiasBasis",
    "VISUAL_BIAS_NUISANCE_SCHEMA",
    "VISUAL_BIAS_NUISANCE_VERSION",
    "VisualBiasNuisanceV1",
    "load_visual_bias_nuisance",
    "main",
    "orthogonalize_visual_bias_basis",
    "write_visual_bias_nuisance",
]


if __name__ == "__main__":
    raise SystemExit(main())
