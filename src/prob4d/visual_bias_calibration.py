"""Source-only, equal-group calibration of coherent visual-bias priors.

The calibration unit is a complete source object or acquisition session. Dense
rows inside one group receive equal total weight so one long sequence cannot
silently dominate the fitted prior. Candidate bias modes are selected by
leave-one-group-out predictive Gaussian NLL; rank zero is an explicit valid
outcome and prevents promotion when coherent-bias evidence is insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array, immutable_integer_array
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
    require_string_sequence,
)
from .visual_bias import (
    VisualBiasNuisanceV1,
    orthogonalize_visual_bias_basis,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

VISUAL_BIAS_CALIBRATION_SCHEMA: Final = "prob4d.visual-bias-calibration"
VISUAL_BIAS_CALIBRATION_VERSION: Final = 1
VISUAL_BIAS_GROUP_WEIGHTING: Final = "equal-total-group-information-v1"
VISUAL_BIAS_RANK_SELECTION: Final = (
    "leave-one-group-out-equal-group-normalized-residual-gaussian-nll-prefix-rank-v1"
)
VISUAL_BIAS_COVARIANCE_ESTIMATOR: Final = (
    "equal-group-noise-corrected-second-moment-diagonal-shrinkage-psd-v1"
)
VISUAL_BIAS_CLAIM_BOUNDARY: Final = (
    "This artifact calibrates one zero-mean coherent visual-bias prior from "
    "source/calibration groups only. It does not establish provider competence, "
    "target calibration, physical-state identifiability, guarded-query benefit, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)

_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "provider_manifest_id",
        "calibration_source_id",
        "basis_names",
        "selected_rank",
        "group_ids",
        "group_definition",
        "residual_definition",
        "uses_truth",
        "uses_target_outcomes",
        "uses_downstream_physical_innovation",
        "orthogonalization_semantics",
        "group_weighting",
        "rank_selection_semantics",
        "covariance_estimator_semantics",
        "ridge",
        "covariance_shrinkage",
        "minimum_nll_improvement",
        "selection_tolerance",
        "gauge_projection_tolerance",
        "payload",
        "arrays",
        "metadata",
        "claim_boundary",
    }
)
_PAYLOAD_FIELDS: Final = frozenset({"path", "sha256", "byte_count", "allow_pickle"})
_ARRAY_FIELDS: Final = frozenset({"dtype", "shape", "sha256"})
_ARRAY_NAMES: Final = (
    "selected_covariance",
    "rank_mean_nll",
    "rank_group_nll",
    "rank_group_coefficients",
    "rank_group_coefficient_covariances",
    "group_row_counts",
    "group_maximum_gauge_projection",
)
_ORTHOGONALIZATION_SEMANTICS: Final = (
    "not-orthogonalized",
    "conditional-whitened-global-gauge-projection-v1",
)


class _StrictJsonSnapshotError(ValueError):
    """Marker for strict JSON failures with complete artifact context."""


def _snapshot_regular_bytes(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open {name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError(f"{name} changed while it was read")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise ValueError(f"{name} byte count changed while it was read")
        return content
    finally:
        os.close(descriptor)


def _load_json_snapshot(path: Path, *, name: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJsonSnapshotError(f"{name} contains duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise _StrictJsonSnapshotError(f"{name} contains non-finite JSON number {token!r}")

    content = _snapshot_regular_bytes(path, name=name)
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except _StrictJsonSnapshotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain one UTF-8 JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


@contextmanager
def _exclusive_writer_lock(path: Path) -> Iterator[None]:
    lock = path.with_name(f".{path.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise ValueError("visual-bias calibration writer lock already exists") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(lock.parent)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(lock.parent)


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
        raise ValueError(f"cannot read visual-bias calibration payload {path.name!r}") from error
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
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
    maximum: float | None = None,
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
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return bool(value)


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


def _validate_spd_blocks(covariance: np.ndarray, *, name: str) -> None:
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (N, 3, 3)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    for index, block in enumerate(covariance):
        try:
            np.linalg.cholesky(block)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} block {index} must be positive definite") from error


@dataclass(frozen=True)
class VisualBiasCalibrationGroup:
    """One independent source object/session used to calibrate coherent bias."""

    group_id: str
    residual: FloatArray
    bias_jacobian: FloatArray
    conditional_covariance: FloatArray
    gauge_design: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        group_id = require_exact_string(self.group_id, name="group_id")
        residual = np.asarray(self.residual)
        jacobian = np.asarray(self.bias_jacobian)
        covariance = np.asarray(self.conditional_covariance)
        if residual.dtype != np.dtype(np.float64) or residual.ndim != 2 or residual.shape[1] != 3:
            raise ValueError("residual must be float64 with shape (N, 3)")
        if residual.shape[0] < 1 or not np.all(np.isfinite(residual)):
            raise ValueError("residual must contain finite rows")
        if (
            jacobian.dtype != np.dtype(np.float64)
            or jacobian.ndim != 3
            or jacobian.shape[:2] != residual.shape
            or jacobian.shape[2] < 1
        ):
            raise ValueError("bias_jacobian must be float64 with shape (N, 3, R)")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("bias_jacobian must be finite")
        if covariance.dtype != np.dtype(np.float64) or covariance.shape != (
            residual.shape[0],
            3,
            3,
        ):
            raise ValueError("conditional_covariance must be float64 with shape (N, 3, 3)")
        _validate_spd_blocks(covariance, name="conditional_covariance")

        gauge: np.ndarray | None = None
        if self.gauge_design is not None:
            gauge = np.asarray(self.gauge_design)
            if (
                gauge.dtype != np.dtype(np.float64)
                or gauge.ndim != 3
                or gauge.shape[:2] != residual.shape
                or gauge.shape[2] < 1
            ):
                raise ValueError("gauge_design must be float64 with shape (N, 3, K)")
            if not np.all(np.isfinite(gauge)):
                raise ValueError("gauge_design must be finite")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="visual-bias calibration-group metadata",
            ),
            name="visual-bias calibration-group metadata",
        )
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "residual", cast(FloatArray, immutable_array(residual)))
        object.__setattr__(
            self,
            "bias_jacobian",
            cast(FloatArray, immutable_array(jacobian)),
        )
        object.__setattr__(
            self,
            "conditional_covariance",
            cast(FloatArray, immutable_array(covariance)),
        )
        object.__setattr__(
            self,
            "gauge_design",
            None if gauge is None else cast(FloatArray, immutable_array(gauge)),
        )
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(self.residual.shape[0])

    @property
    def basis_count(self) -> int:
        return int(self.bias_jacobian.shape[2])


@dataclass(frozen=True)
class _PreparedGroup:
    group_id: str
    row_count: int
    whitened_residual: FloatArray
    whitened_design: FloatArray
    maximum_gauge_projection: float


@dataclass(frozen=True)
class VisualBiasCalibrationV1:
    """Replayable source calibration and rank-selection result."""

    provider_manifest_id: str
    calibration_source_id: str
    basis_names: tuple[str, ...]
    selected_rank: int
    group_ids: tuple[str, ...]
    group_definition: str
    residual_definition: str
    uses_truth: bool
    orthogonalization_semantics: str
    ridge: float
    covariance_shrinkage: float
    minimum_nll_improvement: float
    selection_tolerance: float
    gauge_projection_tolerance: float
    selected_covariance: FloatArray
    rank_mean_nll: FloatArray
    rank_group_nll: FloatArray
    rank_group_coefficients: FloatArray
    rank_group_coefficient_covariances: FloatArray
    group_row_counts: IntArray
    group_maximum_gauge_projection: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        provider_manifest_id = require_sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        calibration_source_id = require_sha256(
            self.calibration_source_id,
            name="calibration_source_id",
        )
        if type(self.basis_names) is not tuple:
            raise TypeError("basis_names must be a canonical tuple")
        basis_names = require_string_sequence(self.basis_names, name="basis_names")
        if len(set(basis_names)) != len(basis_names):
            raise ValueError("basis_names must be unique")
        if type(self.group_ids) is not tuple:
            raise TypeError("group_ids must be a canonical tuple")
        group_ids = require_string_sequence(self.group_ids, name="group_ids")
        if len(group_ids) < 3:
            raise ValueError("visual-bias calibration requires at least three groups")
        if group_ids != tuple(sorted(group_ids)) or len(set(group_ids)) != len(group_ids):
            raise ValueError("group_ids must be unique and canonically sorted")
        selected_rank = require_exact_integer(
            self.selected_rank,
            name="selected_rank",
            minimum=0,
        )
        basis_count = len(basis_names)
        if selected_rank > basis_count:
            raise ValueError("selected_rank exceeds the candidate basis count")
        group_definition = require_exact_string(
            self.group_definition,
            name="group_definition",
        )
        residual_definition = require_exact_string(
            self.residual_definition,
            name="residual_definition",
        )
        uses_truth = _require_boolean(self.uses_truth, name="uses_truth")
        semantics = require_exact_string(
            self.orthogonalization_semantics,
            name="orthogonalization_semantics",
        )
        if semantics not in _ORTHOGONALIZATION_SEMANTICS:
            raise ValueError("unsupported orthogonalization semantics")
        ridge = _require_finite_real(
            self.ridge,
            name="ridge",
            strictly_positive=True,
        )
        shrinkage = _require_finite_real(
            self.covariance_shrinkage,
            name="covariance_shrinkage",
            minimum=0.0,
            maximum=1.0,
        )
        minimum_improvement = _require_finite_real(
            self.minimum_nll_improvement,
            name="minimum_nll_improvement",
            minimum=0.0,
        )
        selection_tolerance = _require_finite_real(
            self.selection_tolerance,
            name="selection_tolerance",
            strictly_positive=True,
        )
        projection_tolerance = _require_finite_real(
            self.gauge_projection_tolerance,
            name="gauge_projection_tolerance",
            strictly_positive=True,
        )

        group_count = len(group_ids)
        covariance = np.asarray(self.selected_covariance)
        rank_mean_nll = np.asarray(self.rank_mean_nll)
        rank_group_nll = np.asarray(self.rank_group_nll)
        coefficients = np.asarray(self.rank_group_coefficients)
        coefficient_covariances = np.asarray(self.rank_group_coefficient_covariances)
        row_counts = np.asarray(self.group_row_counts)
        projections = np.asarray(self.group_maximum_gauge_projection)
        if covariance.dtype != np.dtype(np.float64) or covariance.shape != (
            selected_rank,
            selected_rank,
        ):
            raise ValueError("selected_covariance shape or dtype changed")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("selected_covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("selected_covariance must be symmetric")
        if selected_rank and float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError("selected_covariance must be positive semidefinite")
        if rank_mean_nll.dtype != np.dtype(np.float64) or rank_mean_nll.shape != (basis_count + 1,):
            raise ValueError("rank_mean_nll shape or dtype changed")
        if rank_group_nll.dtype != np.dtype(np.float64) or rank_group_nll.shape != (
            basis_count + 1,
            group_count,
        ):
            raise ValueError("rank_group_nll shape or dtype changed")
        if coefficients.dtype != np.dtype(np.float64) or coefficients.shape != (
            basis_count + 1,
            group_count,
            basis_count,
        ):
            raise ValueError("rank_group_coefficients shape or dtype changed")
        if coefficient_covariances.dtype != np.dtype(
            np.float64
        ) or coefficient_covariances.shape != (
            basis_count + 1,
            group_count,
            basis_count,
            basis_count,
        ):
            raise ValueError("rank_group_coefficient_covariances shape or dtype changed")
        if row_counts.dtype != np.dtype(np.int64) or row_counts.shape != (group_count,):
            raise ValueError("group_row_counts shape or dtype changed")
        if projections.dtype != np.dtype(np.float64) or projections.shape != (group_count,):
            raise ValueError("group_maximum_gauge_projection shape or dtype changed")
        arrays = (
            rank_mean_nll,
            rank_group_nll,
            coefficients,
            coefficient_covariances,
            projections,
        )
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("visual-bias calibration arrays must be finite")
        if np.any(row_counts < 1):
            raise ValueError("group_row_counts must be positive")
        if np.any(projections < 0.0):
            raise ValueError("group gauge projections must be non-negative")
        if semantics != "not-orthogonalized" and np.any(projections > projection_tolerance):
            raise ValueError("orthogonalized calibration basis exceeds its tolerance")
        expected_mean = np.mean(rank_group_nll, axis=1)
        if not np.allclose(rank_mean_nll, expected_mean, atol=1e-12, rtol=1e-12):
            raise ValueError("rank_mean_nll does not equal the equal-group mean")
        baseline = float(rank_mean_nll[0])
        selected_score = float(rank_mean_nll[selected_rank])
        if selected_rank == 0:
            best_nonzero = float(np.min(rank_mean_nll[1:]))
            if baseline - best_nonzero >= minimum_improvement:
                raise ValueError("rank-zero decision contradicts the registered improvement gate")
        else:
            if baseline - selected_score < minimum_improvement:
                raise ValueError("selected rank does not pass the registered NLL improvement")
            global_best = float(np.min(rank_mean_nll))
            if selected_score > global_best + selection_tolerance:
                raise ValueError("selected rank is not globally score-optimal")
            eligible = np.flatnonzero(rank_mean_nll <= global_best + selection_tolerance)
            if int(eligible[0]) != selected_rank:
                raise ValueError("selected rank violates the smaller-rank tie break")

        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="visual-bias calibration metadata",
            ),
            name="visual-bias calibration metadata",
        )
        object.__setattr__(self, "provider_manifest_id", provider_manifest_id)
        object.__setattr__(self, "calibration_source_id", calibration_source_id)
        object.__setattr__(self, "basis_names", basis_names)
        object.__setattr__(self, "selected_rank", selected_rank)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "group_definition", group_definition)
        object.__setattr__(self, "residual_definition", residual_definition)
        object.__setattr__(self, "uses_truth", uses_truth)
        object.__setattr__(self, "orthogonalization_semantics", semantics)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "covariance_shrinkage", shrinkage)
        object.__setattr__(self, "minimum_nll_improvement", minimum_improvement)
        object.__setattr__(self, "selection_tolerance", selection_tolerance)
        object.__setattr__(self, "gauge_projection_tolerance", projection_tolerance)
        object.__setattr__(
            self,
            "selected_covariance",
            cast(FloatArray, immutable_array(covariance)),
        )
        object.__setattr__(
            self,
            "rank_mean_nll",
            cast(FloatArray, immutable_array(rank_mean_nll)),
        )
        object.__setattr__(
            self,
            "rank_group_nll",
            cast(FloatArray, immutable_array(rank_group_nll)),
        )
        object.__setattr__(
            self,
            "rank_group_coefficients",
            cast(FloatArray, immutable_array(coefficients)),
        )
        object.__setattr__(
            self,
            "rank_group_coefficient_covariances",
            cast(FloatArray, immutable_array(coefficient_covariances)),
        )
        object.__setattr__(
            self,
            "group_row_counts",
            cast(IntArray, immutable_integer_array(row_counts, name="group_row_counts")),
        )
        object.__setattr__(
            self,
            "group_maximum_gauge_projection",
            cast(FloatArray, immutable_array(projections)),
        )
        object.__setattr__(self, "metadata", metadata)
        expected_id = _sha256_json(self.identity_record())
        supplied = self.artifact_id
        if (
            supplied is not None
            and require_sha256(
                supplied,
                name="artifact_id",
            )
            != expected_id
        ):
            raise ValueError("visual-bias calibration artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def basis_count(self) -> int:
        return len(self.basis_names)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def promoted(self) -> bool:
        return self.selected_rank > 0

    @property
    def selected_basis_names(self) -> tuple[str, ...]:
        return self.basis_names[: self.selected_rank]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "selected_covariance": np.asarray(self.selected_covariance),
            "rank_mean_nll": np.asarray(self.rank_mean_nll),
            "rank_group_nll": np.asarray(self.rank_group_nll),
            "rank_group_coefficients": np.asarray(self.rank_group_coefficients),
            "rank_group_coefficient_covariances": np.asarray(
                self.rank_group_coefficient_covariances
            ),
            "group_row_counts": np.asarray(self.group_row_counts),
            "group_maximum_gauge_projection": np.asarray(self.group_maximum_gauge_projection),
        }

    def array_descriptors(self) -> dict[str, dict[str, object]]:
        return {name: _array_descriptor(value) for name, value in self.arrays().items()}

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": VISUAL_BIAS_CALIBRATION_SCHEMA,
            "schema_version": VISUAL_BIAS_CALIBRATION_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "calibration_source_id": self.calibration_source_id,
            "basis_names": list(self.basis_names),
            "selected_rank": self.selected_rank,
            "group_ids": list(self.group_ids),
            "group_definition": self.group_definition,
            "residual_definition": self.residual_definition,
            "uses_truth": self.uses_truth,
            "uses_target_outcomes": False,
            "uses_downstream_physical_innovation": False,
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "group_weighting": VISUAL_BIAS_GROUP_WEIGHTING,
            "rank_selection_semantics": VISUAL_BIAS_RANK_SELECTION,
            "covariance_estimator_semantics": VISUAL_BIAS_COVARIANCE_ESTIMATOR,
            "ridge": self.ridge,
            "covariance_shrinkage": self.covariance_shrinkage,
            "minimum_nll_improvement": self.minimum_nll_improvement,
            "selection_tolerance": self.selection_tolerance,
            "gauge_projection_tolerance": self.gauge_projection_tolerance,
            "arrays": self.array_descriptors(),
            "metadata": plain_json(self.metadata),
            "claim_boundary": VISUAL_BIAS_CLAIM_BOUNDARY,
        }

    def summary(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "provider_manifest_id": self.provider_manifest_id,
            "calibration_source_id": self.calibration_source_id,
            "group_count": self.group_count,
            "basis_names": list(self.basis_names),
            "selected_rank": self.selected_rank,
            "selected_basis_names": list(self.selected_basis_names),
            "promoted": self.promoted,
            "rank_mean_nll": [float(value) for value in self.rank_mean_nll],
            "baseline_to_selected_nll_improvement": float(
                self.rank_mean_nll[0] - self.rank_mean_nll[self.selected_rank]
            ),
            "orthogonalization_semantics": self.orthogonalization_semantics,
            "claim_boundary": VISUAL_BIAS_CLAIM_BOUNDARY,
        }


def _prepare_group(
    group: VisualBiasCalibrationGroup,
    *,
    use_gauge_projection: bool,
    gauge_projection_tolerance: float,
) -> _PreparedGroup:
    jacobian = np.asarray(group.bias_jacobian, dtype=np.float64)
    maximum_projection = 0.0
    if use_gauge_projection:
        assert group.gauge_design is not None
        result = orthogonalize_visual_bias_basis(
            jacobian,
            group.gauge_design,
            group.conditional_covariance,
        )
        jacobian = np.asarray(result.bias_jacobian, dtype=np.float64)
        maximum_projection = float(result.maximum_projection_after)
        if maximum_projection > gauge_projection_tolerance:
            raise ValueError(
                f"group {group.group_id!r} retains gauge projection "
                f"{maximum_projection} above {gauge_projection_tolerance}"
            )
    whitened_residual: list[np.ndarray] = []
    whitened_design: list[np.ndarray] = []
    group_scale = 1.0 / np.sqrt(float(group.row_count))
    for row in range(group.row_count):
        cholesky = np.linalg.cholesky(group.conditional_covariance[row])
        whitened_residual.append(group_scale * np.linalg.solve(cholesky, group.residual[row]))
        whitened_design.append(group_scale * np.linalg.solve(cholesky, jacobian[row]))
    return _PreparedGroup(
        group_id=group.group_id,
        row_count=group.row_count,
        whitened_residual=cast(
            FloatArray,
            immutable_array(np.concatenate(whitened_residual)),
        ),
        whitened_design=cast(
            FloatArray,
            immutable_array(np.concatenate(whitened_design, axis=0)),
        ),
        maximum_gauge_projection=maximum_projection,
    )


def _coefficient_posterior(
    group: _PreparedGroup,
    rank: int,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rank == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    design = group.whitened_design[:, :rank]
    information = design.T @ design + ridge * np.eye(rank, dtype=np.float64)
    try:
        cholesky = np.linalg.cholesky(information)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"group {group.group_id!r} bias information is not positive definite"
        ) from error
    inverse = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, np.eye(rank)))
    coefficient = inverse @ design.T @ group.whitened_residual
    return coefficient, inverse


def _psd_covariance(
    coefficients: Sequence[np.ndarray],
    coefficient_covariances: Sequence[np.ndarray],
    *,
    shrinkage: float,
) -> np.ndarray:
    if not coefficients:
        raise ValueError("bias covariance estimation requires calibration groups")
    rank = coefficients[0].shape[0]
    if rank == 0:
        return np.zeros((0, 0), dtype=np.float64)
    raw = np.mean(
        [
            np.outer(coefficient, coefficient) - covariance
            for coefficient, covariance in zip(
                coefficients,
                coefficient_covariances,
                strict=True,
            )
        ],
        axis=0,
    )
    raw = 0.5 * (raw + raw.T)
    target = np.diag(np.maximum(np.diag(raw), 0.0))
    shrunk = (1.0 - shrinkage) * raw + shrinkage * target
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (shrunk + shrunk.T))
    clipped = np.maximum(eigenvalues, 0.0)
    covariance = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def _predictive_nll(group: _PreparedGroup, covariance: np.ndarray, rank: int) -> float:
    residual = group.whitened_residual
    if rank == 0 or not np.any(covariance):
        return 0.5 * (np.log(2.0 * np.pi) + float(residual @ residual))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > 1e-15 * max(float(np.max(eigenvalues)), 1.0)
    if not np.any(keep):
        return 0.5 * (np.log(2.0 * np.pi) + float(residual @ residual))
    root = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    low_rank = group.whitened_design[:, :rank] @ root
    gram = np.eye(low_rank.shape[1], dtype=np.float64) + low_rank.T @ low_rank
    sign, log_determinant = np.linalg.slogdet(gram)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("visual-bias predictive covariance is not positive definite")
    projection = low_rank.T @ residual
    correction = float(projection @ np.linalg.solve(gram, projection))
    quadratic = max(float(residual @ residual) - correction, 0.0)
    return 0.5 * (np.log(2.0 * np.pi) + float(log_determinant) + quadratic)


def fit_visual_bias_calibration(
    groups: Sequence[VisualBiasCalibrationGroup],
    *,
    basis_names: Sequence[str],
    provider_manifest_id: str,
    calibration_source_id: str,
    group_definition: str,
    residual_definition: str,
    uses_truth: bool,
    ridge: float = 1e-8,
    covariance_shrinkage: float = 0.25,
    minimum_nll_improvement: float = 1e-4,
    selection_tolerance: float = 1e-10,
    gauge_projection_tolerance: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> VisualBiasCalibrationV1:
    """Fit and source-select a zero-mean coherent visual-bias covariance.

    Candidate ranks are prefixes of ``basis_names``. Every leave-one-group-out
    fold estimates the covariance on complete training groups and scores the
    held-out residual under ``D + B Sigma B'``. Rank zero is retained unless a
    nonzero rank improves equal-group per-coordinate NLL by the frozen margin.
    """

    names = tuple(basis_names)
    if not names or any(type(value) is not str for value in names):
        raise ValueError("basis_names must contain genuine strings")
    names = require_string_sequence(names, name="basis_names")
    if len(set(names)) != len(names):
        raise ValueError("basis_names must be unique")
    values = tuple(groups)
    valid_groups = all(isinstance(group, VisualBiasCalibrationGroup) for group in values)
    if len(values) < 3 or not valid_groups:
        raise ValueError("groups must contain at least three VisualBiasCalibrationGroup values")
    ordered = tuple(sorted(values, key=lambda group: group.group_id))
    group_ids = tuple(group.group_id for group in ordered)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("calibration group IDs must be unique")
    if any(group.basis_count != len(names) for group in ordered):
        raise ValueError("every group bias basis must match basis_names")
    gauge_presence = {group.gauge_design is not None for group in ordered}
    if len(gauge_presence) != 1:
        raise ValueError("all calibration groups must consistently provide gauge_design")
    use_gauge_projection = gauge_presence == {True}
    ridge_value = _require_finite_real(ridge, name="ridge", strictly_positive=True)
    shrinkage = _require_finite_real(
        covariance_shrinkage,
        name="covariance_shrinkage",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_improvement = _require_finite_real(
        minimum_nll_improvement,
        name="minimum_nll_improvement",
        minimum=0.0,
    )
    tolerance = _require_finite_real(
        selection_tolerance,
        name="selection_tolerance",
        strictly_positive=True,
    )
    projection_tolerance = _require_finite_real(
        gauge_projection_tolerance,
        name="gauge_projection_tolerance",
        strictly_positive=True,
    )
    prepared = tuple(
        _prepare_group(
            group,
            use_gauge_projection=use_gauge_projection,
            gauge_projection_tolerance=projection_tolerance,
        )
        for group in ordered
    )

    rank_count = len(names) + 1
    group_count = len(prepared)
    coefficients: FloatArray = np.zeros((rank_count, group_count, len(names)), dtype=np.float64)
    coefficient_covariances: FloatArray = np.zeros(
        (rank_count, group_count, len(names), len(names)),
        dtype=np.float64,
    )
    per_rank_coefficients: list[list[np.ndarray]] = []
    per_rank_covariances: list[list[np.ndarray]] = []
    for rank in range(rank_count):
        rank_coefficients: list[np.ndarray] = []
        rank_covariances: list[np.ndarray] = []
        for group_index, group in enumerate(prepared):
            coefficient, covariance = _coefficient_posterior(
                group,
                rank,
                ridge=ridge_value,
            )
            rank_coefficients.append(coefficient)
            rank_covariances.append(covariance)
            coefficients[rank, group_index, :rank] = coefficient
            coefficient_covariances[rank, group_index, :rank, :rank] = covariance
        per_rank_coefficients.append(rank_coefficients)
        per_rank_covariances.append(rank_covariances)

    rank_group_nll: FloatArray = np.empty((rank_count, group_count), dtype=np.float64)
    for rank in range(rank_count):
        for heldout_index, heldout in enumerate(prepared):
            training_coefficients = [
                value
                for index, value in enumerate(per_rank_coefficients[rank])
                if index != heldout_index
            ]
            training_covariances = [
                value
                for index, value in enumerate(per_rank_covariances[rank])
                if index != heldout_index
            ]
            covariance = _psd_covariance(
                training_coefficients,
                training_covariances,
                shrinkage=shrinkage,
            )
            rank_group_nll[rank, heldout_index] = _predictive_nll(
                heldout,
                covariance,
                rank,
            )
    rank_mean_nll = np.mean(rank_group_nll, axis=1)
    best_score = float(np.min(rank_mean_nll))
    tied = np.flatnonzero(rank_mean_nll <= best_score + tolerance)
    best_rank = int(tied[0])
    baseline = float(rank_mean_nll[0])
    selected_rank = (
        best_rank
        if best_rank > 0 and baseline - float(rank_mean_nll[best_rank]) >= minimum_improvement
        else 0
    )
    selected_covariance = _psd_covariance(
        per_rank_coefficients[selected_rank],
        per_rank_covariances[selected_rank],
        shrinkage=shrinkage,
    )
    return VisualBiasCalibrationV1(
        provider_manifest_id=provider_manifest_id,
        calibration_source_id=calibration_source_id,
        basis_names=names,
        selected_rank=selected_rank,
        group_ids=group_ids,
        group_definition=group_definition,
        residual_definition=residual_definition,
        uses_truth=uses_truth,
        orthogonalization_semantics=(
            "conditional-whitened-global-gauge-projection-v1"
            if use_gauge_projection
            else "not-orthogonalized"
        ),
        ridge=ridge_value,
        covariance_shrinkage=shrinkage,
        minimum_nll_improvement=minimum_improvement,
        selection_tolerance=tolerance,
        gauge_projection_tolerance=projection_tolerance,
        selected_covariance=selected_covariance,
        rank_mean_nll=rank_mean_nll,
        rank_group_nll=rank_group_nll,
        rank_group_coefficients=coefficients,
        rank_group_coefficient_covariances=coefficient_covariances,
        group_row_counts=np.asarray([group.row_count for group in prepared], dtype=np.int64),
        group_maximum_gauge_projection=np.asarray(
            [group.maximum_gauge_projection for group in prepared],
            dtype=np.float64,
        ),
        metadata={} if metadata is None else metadata,
    )


def build_visual_bias_nuisance_from_calibration(
    calibration: VisualBiasCalibrationV1,
    *,
    observation_artifact_id: str,
    observation_identity_sha256: str,
    bias_id: str,
    bias_jacobian: FloatArray,
    conditional_covariance: FloatArray | None = None,
    gauge_design: FloatArray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VisualBiasNuisanceV1:
    """Instantiate one observation sidecar from a promoted source calibration."""

    if not calibration.promoted:
        raise ValueError("rank-zero visual-bias calibration cannot produce a nuisance sidecar")
    bias_id = require_exact_string(bias_id, name="bias_id")
    candidate = np.asarray(bias_jacobian)
    if (
        candidate.dtype != np.dtype(np.float64)
        or candidate.ndim != 3
        or candidate.shape[1] != 3
        or candidate.shape[2] != calibration.basis_count
    ):
        raise ValueError("bias_jacobian must be float64 with shape (N, 3, basis_count)")
    selected = candidate[:, :, : calibration.selected_rank]
    maximum_projection = 0.0
    if calibration.orthogonalization_semantics != "not-orthogonalized":
        if gauge_design is None or conditional_covariance is None:
            raise ValueError(
                "orthogonalized calibration requires gauge_design and conditional_covariance"
            )
        result = orthogonalize_visual_bias_basis(
            selected,
            gauge_design,
            conditional_covariance,
        )
        selected = np.asarray(result.bias_jacobian, dtype=np.float64)
        maximum_projection = float(result.maximum_projection_after)
        if maximum_projection > calibration.gauge_projection_tolerance:
            raise ValueError("observation bias basis exceeds calibration gauge tolerance")
    elif gauge_design is not None or conditional_covariance is not None:
        raise ValueError("unprojected calibration must not silently change basis semantics")
    sidecar_metadata: dict[str, Any] = {
        "visual_bias_calibration_artifact_id": calibration.artifact_id,
        "provider_manifest_id": calibration.provider_manifest_id,
        "calibration_source_id": calibration.calibration_source_id,
        "calibration_group_ids": list(calibration.group_ids),
        "selected_rank": calibration.selected_rank,
        "selected_basis_names": list(calibration.selected_basis_names),
        "uses_truth_for_source_calibration": calibration.uses_truth,
        "uses_target_outcomes": False,
        "uses_downstream_physical_innovation": False,
    }
    if metadata is not None:
        overlap = sidecar_metadata.keys() & metadata.keys()
        if overlap:
            raise ValueError(
                f"sidecar metadata may not replace calibration fields: {sorted(overlap)}"
            )
        sidecar_metadata.update(plain_json(metadata))
    return VisualBiasNuisanceV1(
        observation_artifact_id=observation_artifact_id,
        observation_identity_sha256=observation_identity_sha256,
        bias_ids=(bias_id,),
        basis_names=calibration.selected_basis_names,
        row_bias_indices=np.zeros(selected.shape[0], dtype=np.int64),
        bias_jacobian=np.asarray(selected, dtype=np.float64),
        joint_bias_covariance=np.asarray(
            calibration.selected_covariance,
            dtype=np.float64,
        ),
        orthogonalization_semantics=calibration.orthogonalization_semantics,
        maximum_gauge_projection=maximum_projection,
        gauge_projection_tolerance=calibration.gauge_projection_tolerance,
        metadata=sidecar_metadata,
    )


def _validate_array_descriptor(value: object, array: np.ndarray, *, name: str) -> None:
    mapping = require_mapping(value, name=f"{name} descriptor")
    require_exact_fields(mapping, _ARRAY_FIELDS, name=f"{name} descriptor")
    if dict(mapping) != _array_descriptor(array):
        raise ValueError(f"{name} descriptor does not match payload bytes")


def write_visual_bias_calibration(
    calibration: VisualBiasCalibrationV1,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist one content-addressed calibration without overwriting drift."""

    manifest = Path(manifest_path)
    payload = Path(payload_path) if payload_path is not None else manifest.with_suffix(".npz")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_payload = payload.resolve().relative_to(manifest.parent.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "visual-bias calibration payload must lie inside manifest directory"
        ) from error
    relative_payload = _safe_relative_path(relative_payload, name="payload path")
    with _exclusive_writer_lock(manifest):
        if manifest.exists() or payload.exists():
            if not manifest.exists() or not payload.exists():
                raise ValueError("visual-bias calibration destination contains a partial artifact")
            existing_record = _load_json_snapshot(
                manifest,
                name="visual-bias calibration manifest",
            )
            existing_payload_record = require_mapping(
                existing_record.get("payload"),
                name="payload descriptor",
            )
            existing_payload = _resolved_member(
                manifest.parent,
                _safe_relative_path(
                    existing_payload_record.get("path"),
                    name="payload path",
                ),
                name="payload path",
            )
            if existing_payload != payload.resolve():
                raise ValueError("existing calibration references a different payload path")
            existing = load_visual_bias_calibration(manifest)
            if existing.artifact_id != calibration.artifact_id:
                raise ValueError("refusing to replace a different visual-bias calibration")
            return manifest, payload
        arrays = calibration.arrays()
        _atomic_write_npz(payload, arrays)
        record = {
            **calibration.identity_record(),
            "artifact_id": calibration.artifact_id,
            "payload": {
                "path": relative_payload,
                "sha256": _sha256_file(payload),
                "byte_count": payload.stat().st_size,
                "allow_pickle": False,
            },
        }
        _atomic_write_json(manifest, record)
    return manifest, payload


def load_visual_bias_calibration(path: str | Path) -> VisualBiasCalibrationV1:
    """Load and independently reconstruct one calibration artifact."""

    manifest_path = Path(path).expanduser().absolute()
    record = _load_json_snapshot(
        manifest_path,
        name="visual-bias calibration manifest",
    )
    require_exact_fields(record, _MANIFEST_FIELDS, name="visual-bias calibration manifest")
    if record["schema"] != VISUAL_BIAS_CALIBRATION_SCHEMA:
        raise ValueError("unsupported visual-bias calibration schema")
    if record["schema_version"] != VISUAL_BIAS_CALIBRATION_VERSION:
        raise ValueError("unsupported visual-bias calibration version")
    if record["uses_target_outcomes"] is not False:
        raise ValueError("visual-bias calibration must not use target outcomes")
    if record["uses_downstream_physical_innovation"] is not False:
        raise ValueError("visual-bias calibration must not use physical innovations")
    if record["group_weighting"] != VISUAL_BIAS_GROUP_WEIGHTING:
        raise ValueError("unsupported visual-bias group weighting")
    if record["rank_selection_semantics"] != VISUAL_BIAS_RANK_SELECTION:
        raise ValueError("unsupported visual-bias rank selection")
    if record["covariance_estimator_semantics"] != VISUAL_BIAS_COVARIANCE_ESTIMATOR:
        raise ValueError("unsupported visual-bias covariance estimator")
    if record["claim_boundary"] != VISUAL_BIAS_CLAIM_BOUNDARY:
        raise ValueError("visual-bias calibration claim boundary changed")
    payload_record = require_mapping(record["payload"], name="payload descriptor")
    require_exact_fields(payload_record, _PAYLOAD_FIELDS, name="payload descriptor")
    if payload_record["allow_pickle"] is not False:
        raise ValueError("visual-bias calibration payload must disable pickle")
    payload = _resolved_member(
        manifest_path.parent,
        _safe_relative_path(payload_record["path"], name="payload path"),
        name="payload path",
    )
    if not payload.is_file():
        raise ValueError("visual-bias calibration payload is missing")
    byte_count = require_exact_integer(
        payload_record["byte_count"],
        name="payload byte_count",
        minimum=1,
    )
    payload_bytes = _snapshot_regular_bytes(
        payload,
        name="visual-bias calibration payload",
    )
    if len(payload_bytes) != byte_count:
        raise ValueError("visual-bias calibration payload byte count changed")
    payload_digest = hashlib.sha256(payload_bytes).hexdigest()
    if payload_digest != require_sha256(
        payload_record["sha256"],
        name="payload sha256",
    ):
        raise ValueError("visual-bias calibration payload SHA-256 changed")
    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAY_NAMES):
            raise ValueError("visual-bias calibration payload members changed")
        arrays = {name: np.asarray(archive[name]) for name in _ARRAY_NAMES}
    descriptors = require_mapping(record["arrays"], name="array descriptors")
    if set(descriptors) != set(_ARRAY_NAMES):
        raise ValueError("visual-bias calibration array descriptors changed")
    for name, array in arrays.items():
        _validate_array_descriptor(descriptors[name], array, name=name)
    metadata = require_finite_json_mapping(
        record["metadata"],
        name="visual-bias calibration metadata",
    )
    calibration = VisualBiasCalibrationV1(
        provider_manifest_id=record["provider_manifest_id"],
        calibration_source_id=record["calibration_source_id"],
        basis_names=require_string_sequence(record["basis_names"], name="basis_names"),
        selected_rank=record["selected_rank"],
        group_ids=require_string_sequence(record["group_ids"], name="group_ids"),
        group_definition=record["group_definition"],
        residual_definition=record["residual_definition"],
        uses_truth=_require_boolean(record["uses_truth"], name="uses_truth"),
        orthogonalization_semantics=record["orthogonalization_semantics"],
        ridge=require_json_number(record["ridge"], name="ridge"),
        covariance_shrinkage=require_json_number(
            record["covariance_shrinkage"],
            name="covariance_shrinkage",
        ),
        minimum_nll_improvement=require_json_number(
            record["minimum_nll_improvement"],
            name="minimum_nll_improvement",
        ),
        selection_tolerance=require_json_number(
            record["selection_tolerance"],
            name="selection_tolerance",
        ),
        gauge_projection_tolerance=require_json_number(
            record["gauge_projection_tolerance"],
            name="gauge_projection_tolerance",
        ),
        selected_covariance=cast(FloatArray, arrays["selected_covariance"]),
        rank_mean_nll=cast(FloatArray, arrays["rank_mean_nll"]),
        rank_group_nll=cast(FloatArray, arrays["rank_group_nll"]),
        rank_group_coefficients=cast(
            FloatArray,
            arrays["rank_group_coefficients"],
        ),
        rank_group_coefficient_covariances=cast(
            FloatArray,
            arrays["rank_group_coefficient_covariances"],
        ),
        group_row_counts=cast(IntArray, arrays["group_row_counts"]),
        group_maximum_gauge_projection=cast(
            FloatArray,
            arrays["group_maximum_gauge_projection"],
        ),
        metadata=metadata,
        artifact_id=record["artifact_id"],
    )
    return calibration


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d-visual-bias-calibration",
        description="validate source-only visual-bias calibration artifacts",
    )
    parser.add_argument("manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    calibration = load_visual_bias_calibration(arguments.manifest)
    print(json.dumps(calibration.summary(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "VISUAL_BIAS_CALIBRATION_SCHEMA",
    "VISUAL_BIAS_CALIBRATION_VERSION",
    "VisualBiasCalibrationGroup",
    "VisualBiasCalibrationV1",
    "build_visual_bias_nuisance_from_calibration",
    "fit_visual_bias_calibration",
    "load_visual_bias_calibration",
    "main",
    "write_visual_bias_calibration",
]


if __name__ == "__main__":
    raise SystemExit(main())
