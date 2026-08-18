"""Independent validation for ``phys4d.observation_belief`` version 1.

This module deliberately does not import :mod:`prob4d`.  It reimplements the
closed NPZ schema, numerical invariants, and content-address calculation so that
producer and verifier bugs are less likely to share one implementation path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

VERIFICATION_SCHEMA = "prob4d.independent-observation-verification"
VERIFICATION_VERSION = 1
VERIFIER_IMPLEMENTATION = "prob4d-independent-observation-verifier-v1"
OBSERVATION_SCHEMA = "phys4d.observation_belief"
OBSERVATION_VERSION = 1
VERIFICATION_CLAIM_BOUNDARY = (
    "This report independently validates serialization, schema, numerical invariants, "
    "and content identity for one ObservationBeliefV1 artifact. It is interoperability "
    "and corruption-detection evidence only; it does not establish provider accuracy, "
    "uncertainty calibration, physical-query benefit, deployment safety, or state of "
    "the art."
)

_DESCRIPTOR_FIELDS = frozenset(
    {
        "artifact_id",
        "case_id",
        "causal_frame_stop",
        "factor_names",
        "metadata",
        "schema_name",
        "schema_version",
        "source_artifact_sha256",
        "source_repository",
        "source_revision",
        "stream_id",
        "view_names",
        "window_names",
    }
)
_ARRAY_DTYPES = {
    "association_probability": np.dtype("float64"),
    "correlation_group_ids": np.dtype("int64"),
    "declared_frame_ids": np.dtype("int64"),
    "entity_ids": np.dtype("int64"),
    "factor_group_ids": np.dtype("int64"),
    "frame_ids": np.dtype("int64"),
    "group_composite_weight": np.dtype("float64"),
    "group_ids": np.dtype("int64"),
    "group_prior_nominal_probability": np.dtype("float64"),
    "local_covariance_m2": np.dtype("float64"),
    "low_rank_factor_m": np.dtype("float64"),
    "mean_xyz_m": np.dtype("float64"),
    "prior_reliability": np.dtype("float64"),
    "view_indices": np.dtype("int64"),
    "window_indices": np.dtype("int64"),
}
_REQUIRED_ARCHIVE_MEMBERS = frozenset(
    {"descriptor_json.npy", *(f"{name}.npy" for name in _ARRAY_DTYPES)}
)
_CHECKS = (
    "closed-npz-member-set",
    "bounded-archive-resources",
    "strict-finite-json-descriptor",
    "closed-descriptor-fields",
    "exact-array-dtypes",
    "array-shapes-and-finiteness",
    "exclusive-causal-boundary",
    "positive-definite-local-covariance",
    "unique-observation-identities",
    "group-assignment-consistency",
    "content-address-match",
)


class _StrictJsonError(ValueError):
    """Internal marker for descriptor parse failures with complete context."""


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    """Resource bounds applied before NumPy opens an NPZ archive."""

    max_members: int = 32
    max_archive_bytes: int = 8 * 1024**3
    max_uncompressed_bytes: int = 32 * 1024**3
    max_compression_ratio: float = 1000.0

    def __post_init__(self) -> None:
        if type(self.max_members) is not int or self.max_members < len(
            _REQUIRED_ARCHIVE_MEMBERS
        ):
            raise ValueError("max_members is smaller than the closed contract")
        for name, value in (
            ("max_archive_bytes", self.max_archive_bytes),
            ("max_uncompressed_bytes", self.max_uncompressed_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_compression_ratio, bool)
            or not isinstance(self.max_compression_ratio, (int, float))
            or not math.isfinite(float(self.max_compression_ratio))
            or float(self.max_compression_ratio) < 1.0
        ):
            raise ValueError("max_compression_ratio must be finite and at least one")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_members": self.max_members,
            "max_archive_bytes": self.max_archive_bytes,
            "max_uncompressed_bytes": self.max_uncompressed_bytes,
            "max_compression_ratio": float(self.max_compression_ratio),
        }


DEFAULT_LIMITS = VerificationLimits()


@dataclass(frozen=True, slots=True)
class ArrayVerification:
    """Content identity and shape of one validated array member."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Content-addressed result of one independent verification."""

    artifact_id: str
    artifact_file_sha256: str
    descriptor_sha256: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    causal_frame_stop: int
    frame_count: int
    observation_count: int
    group_count: int
    factor_rank: int
    arrays: tuple[ArrayVerification, ...]
    limits: VerificationLimits

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_name": VERIFICATION_SCHEMA,
            "schema_version": VERIFICATION_VERSION,
            "status": "valid",
            "verifier_implementation": VERIFIER_IMPLEMENTATION,
            "artifact_id": self.artifact_id,
            "artifact_file_sha256": self.artifact_file_sha256,
            "descriptor_sha256": self.descriptor_sha256,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "causal_frame_stop": self.causal_frame_stop,
            "frame_count": self.frame_count,
            "observation_count": self.observation_count,
            "group_count": self.group_count,
            "factor_rank": self.factor_rank,
            "arrays": [item.to_dict() for item in self.arrays],
            "limits": self.limits.to_dict(),
            "checks": list(_CHECKS),
            "claim_boundary": VERIFICATION_CLAIM_BOUNDARY,
        }

    @property
    def report_id(self) -> str:
        return hashlib.sha256(_canonical_json(self._unsigned_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {"report_id": self.report_id, **self._unsigned_dict()}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _artifact_id(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    payload = dict(descriptor)
    payload.pop("artifact_id", None)
    digest = hashlib.sha256()
    digest.update(_canonical_json(payload))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return cast(str, value)


def _require_string_list(
    value: Any,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not list or (not allow_empty and not value):
        raise ValueError(f"{name} must be a JSON array of nonempty strings")
    if any(type(item) is not str or not item for item in value):
        raise ValueError(f"{name} must contain only nonempty exact strings")
    return tuple(cast(list[str], value))


def _validate_finite_json(value: Any, *, path: str = "$") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"metadata contains a non-finite number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_finite_json(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"metadata contains a non-string key at {path}")
            _validate_finite_json(item, path=f"{path}.{key}")
        return
    raise ValueError(f"metadata contains unsupported JSON value at {path}")


def _loads_descriptor(content: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJsonError(
                    f"observation descriptor contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise _StrictJsonError(
            f"observation descriptor contains non-finite JSON token {token!r}"
        )

    try:
        value = json.loads(
            content,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except _StrictJsonError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError("observation descriptor is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("observation descriptor must contain one JSON object")
    _validate_finite_json(value)
    return cast(dict[str, Any], value)


def _descriptor_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("descriptor_json must be one scalar UTF-8 string")
    item = array.item()
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("descriptor_json is not UTF-8") from error
    if type(item) is not str:
        raise ValueError("descriptor_json must be one scalar UTF-8 string")
    return item


def _preflight_archive(path: Path, limits: VerificationLimits) -> None:
    try:
        archive_size = path.stat().st_size
    except OSError as error:
        raise ValueError("observation artifact cannot be read") from error
    if archive_size <= 0 or archive_size > limits.max_archive_bytes:
        raise ValueError("observation artifact exceeds the compressed-size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("observation artifact is not a valid ZIP container") from error
    if len(members) > limits.max_members:
        raise ValueError("observation artifact has too many ZIP members")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError("observation artifact contains duplicate ZIP member names")
    if set(names) != _REQUIRED_ARCHIVE_MEMBERS:
        missing = sorted(_REQUIRED_ARCHIVE_MEMBERS - set(names))
        extra = sorted(set(names) - _REQUIRED_ARCHIVE_MEMBERS)
        raise ValueError(
            f"observation artifact member set changed; missing={missing}, extra={extra}"
        )
    uncompressed_size = 0
    for member in members:
        path_parts = Path(member.filename).parts
        if (
            member.is_dir()
            or len(path_parts) != 1
            or path_parts[0] in {"", ".", ".."}
            or member.flag_bits & 0x1
        ):
            raise ValueError("observation artifact contains an unsafe ZIP member")
        uncompressed_size += member.file_size
        if member.file_size > 0:
            if member.compress_size <= 0:
                raise ValueError("observation artifact contains an invalid compressed member")
            ratio = member.file_size / member.compress_size
            if ratio > float(limits.max_compression_ratio):
                raise ValueError("observation artifact exceeds the compression-ratio limit")
    if uncompressed_size > limits.max_uncompressed_bytes:
        raise ValueError("observation artifact exceeds the uncompressed-size limit")


def _load_payload(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"descriptor_json", *_ARRAY_DTYPES}:
                raise ValueError("observation artifact NPZ members changed")
            descriptor = _loads_descriptor(_descriptor_text(archive["descriptor_json"]))
            arrays = {name: np.asarray(archive[name]) for name in _ARRAY_DTYPES}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("observation ", "descriptor_json", "metadata ")
        ):
            raise
        raise ValueError("observation artifact is not a valid non-pickled NPZ") from error
    return descriptor, arrays


def _validate_descriptor(descriptor: Mapping[str, Any]) -> dict[str, object]:
    fields = set(descriptor)
    if fields != _DESCRIPTOR_FIELDS:
        missing = sorted(_DESCRIPTOR_FIELDS - fields)
        extra = sorted(fields - _DESCRIPTOR_FIELDS)
        raise ValueError(
            f"observation descriptor fields changed; missing={missing}, extra={extra}"
        )
    if descriptor["schema_name"] != OBSERVATION_SCHEMA:
        raise ValueError("unsupported observation schema")
    if type(descriptor["schema_version"]) is not int or (
        descriptor["schema_version"] != OBSERVATION_VERSION
    ):
        raise ValueError("unsupported observation schema version")
    if type(descriptor["causal_frame_stop"]) is not int or (
        descriptor["causal_frame_stop"] < 1
    ):
        raise ValueError("causal_frame_stop must be a positive integer")
    if type(descriptor["metadata"]) is not dict:
        raise ValueError("metadata must be a finite JSON object")
    _validate_finite_json(descriptor["metadata"], path="$.metadata")
    return {
        "artifact_id": _require_sha256(
            descriptor["artifact_id"], name="artifact_id"
        ),
        "case_id": _require_nonempty_string(descriptor["case_id"], name="case_id"),
        "stream_id": _require_nonempty_string(
            descriptor["stream_id"], name="stream_id"
        ),
        "source_repository": _require_nonempty_string(
            descriptor["source_repository"], name="source_repository"
        ),
        "source_revision": _require_nonempty_string(
            descriptor["source_revision"], name="source_revision"
        ),
        "source_artifact_sha256": _require_sha256(
            descriptor["source_artifact_sha256"],
            name="source_artifact_sha256",
        ),
        "causal_frame_stop": cast(int, descriptor["causal_frame_stop"]),
        "view_names": _require_string_list(
            descriptor["view_names"], name="view_names", allow_empty=False
        ),
        "window_names": _require_string_list(
            descriptor["window_names"], name="window_names", allow_empty=False
        ),
        "factor_names": _require_string_list(
            descriptor["factor_names"], name="factor_names", allow_empty=True
        ),
    }


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    descriptor: Mapping[str, object],
) -> tuple[int, int, int, int]:
    for name, expected_dtype in _ARRAY_DTYPES.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(
                f"array {name!r} has dtype {arrays[name].dtype}, expected {expected_dtype}"
            )

    declared_frames = arrays["declared_frame_ids"]
    mean = arrays["mean_xyz_m"]
    frame_ids = arrays["frame_ids"]
    entity_ids = arrays["entity_ids"]
    view_indices = arrays["view_indices"]
    window_indices = arrays["window_indices"]
    correlation_groups = arrays["correlation_group_ids"]
    factor_groups = arrays["factor_group_ids"]
    prior_reliability = arrays["prior_reliability"]
    association_probability = arrays["association_probability"]
    local_covariance = arrays["local_covariance_m2"]
    factors = arrays["low_rank_factor_m"]
    group_ids = arrays["group_ids"]
    group_prior = arrays["group_prior_nominal_probability"]
    group_weight = arrays["group_composite_weight"]

    causal_frame_stop = cast(int, descriptor["causal_frame_stop"])
    view_names = cast(tuple[str, ...], descriptor["view_names"])
    window_names = cast(tuple[str, ...], descriptor["window_names"])
    factor_names = cast(tuple[str, ...], descriptor["factor_names"])

    if (
        declared_frames.ndim != 1
        or len(declared_frames) == 0
        or np.any(declared_frames < 0)
        or np.any(np.diff(declared_frames) <= 0)
        or np.any(declared_frames >= causal_frame_stop)
    ):
        raise ValueError("declared_frame_ids violates its ordered causal contract")
    if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) == 0:
        raise ValueError("mean_xyz_m must have nonempty shape (N, 3)")
    observation_count = len(mean)
    for name in (
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "prior_reliability",
        "association_probability",
    ):
        if arrays[name].shape != (observation_count,):
            raise ValueError(f"array {name!r} must have shape ({observation_count},)")
    if local_covariance.shape != (observation_count, 3, 3):
        raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
    factor_rank = len(factor_names)
    if factors.shape != (observation_count, 3, factor_rank):
        raise ValueError("low_rank_factor_m shape disagrees with factor_names")

    for name in (
        "mean_xyz_m",
        "local_covariance_m2",
        "low_rank_factor_m",
        "prior_reliability",
        "association_probability",
        "group_prior_nominal_probability",
        "group_composite_weight",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"array {name!r} contains non-finite values")
    if np.any(entity_ids < 0) or np.any(correlation_groups < 0) or np.any(
        factor_groups < 0
    ):
        raise ValueError("entity and group identifiers must be nonnegative")
    if np.any(frame_ids < 0) or np.any(frame_ids >= causal_frame_stop):
        raise ValueError("observation frames cross the exclusive causal boundary")
    if not np.all(np.isin(frame_ids, declared_frames)):
        raise ValueError("observation frames are not declared")
    if np.any(view_indices < 0) or np.any(view_indices >= len(view_names)):
        raise ValueError("view_indices reference unavailable views")
    if np.any(window_indices < 0) or np.any(window_indices >= len(window_names)):
        raise ValueError("window_indices reference unavailable windows")
    for name, values in (
        ("prior_reliability", prior_reliability),
        ("association_probability", association_probability),
        ("group_prior_nominal_probability", group_prior),
    ):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"array {name!r} must lie in [0, 1]")
    if np.any((group_weight <= 0.0) | (group_weight > 1.0)):
        raise ValueError("group_composite_weight must lie in (0, 1]")

    symmetric = 0.5 * (local_covariance + np.swapaxes(local_covariance, 1, 2))
    if not np.allclose(local_covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError("local_covariance_m2 must be symmetric")
    if np.any(np.min(np.linalg.eigvalsh(symmetric), axis=1) <= 0.0):
        raise ValueError("local_covariance_m2 must be positive definite")

    expected_groups = np.unique(correlation_groups)
    if group_ids.ndim != 1 or not np.array_equal(group_ids, expected_groups):
        raise ValueError("group_ids must equal sorted unique correlation_group_ids")
    group_count = len(group_ids)
    if group_prior.shape != (group_count,) or group_weight.shape != (group_count,):
        raise ValueError("group probability and weight arrays must identify every group")

    identities = np.column_stack(
        (frame_ids, entity_ids, view_indices, window_indices)
    )
    if len(identities) != len(np.unique(identities, axis=0)):
        raise ValueError("observation identity (frame, entity, view, window) is not unique")
    return len(declared_frames), observation_count, group_count, factor_rank


def verify_observation_belief(
    path: str | Path,
    *,
    limits: VerificationLimits = DEFAULT_LIMITS,
) -> VerificationReport:
    """Independently validate one closed, non-pickled observation artifact."""

    if not isinstance(limits, VerificationLimits):
        raise TypeError("limits must be a VerificationLimits instance")
    source = Path(path)
    _preflight_archive(source, limits)
    raw_descriptor, arrays = _load_payload(source)
    descriptor = _validate_descriptor(raw_descriptor)
    frame_count, observation_count, group_count, factor_rank = _validate_arrays(
        arrays, descriptor
    )
    computed_artifact_id = _artifact_id(raw_descriptor, arrays)
    if computed_artifact_id != descriptor["artifact_id"]:
        raise ValueError("observation artifact digest does not match its payload")

    array_reports = tuple(
        ArrayVerification(
            name=name,
            dtype=values.dtype.str,
            shape=tuple(int(value) for value in values.shape),
            sha256=_array_sha256(values),
        )
        for name, values in sorted(arrays.items())
    )
    return VerificationReport(
        artifact_id=computed_artifact_id,
        artifact_file_sha256=_file_sha256(source),
        descriptor_sha256=hashlib.sha256(_canonical_json(raw_descriptor)).hexdigest(),
        case_id=cast(str, descriptor["case_id"]),
        stream_id=cast(str, descriptor["stream_id"]),
        source_repository=cast(str, descriptor["source_repository"]),
        source_revision=cast(str, descriptor["source_revision"]),
        causal_frame_stop=cast(int, descriptor["causal_frame_stop"]),
        frame_count=frame_count,
        observation_count=observation_count,
        group_count=group_count,
        factor_rank=factor_rank,
        arrays=array_reports,
        limits=limits,
    )


def write_verification_report(
    path: str | Path,
    report: VerificationReport,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one complete verification report atomically."""

    if not isinstance(report, VerificationReport):
        raise TypeError("report must be a VerificationReport")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be Boolean")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            report.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError:
                raise FileExistsError(target) from None
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_LIMITS",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_VERSION",
    "VERIFICATION_CLAIM_BOUNDARY",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_VERSION",
    "VERIFIER_IMPLEMENTATION",
    "ArrayVerification",
    "VerificationLimits",
    "VerificationReport",
    "verify_observation_belief",
    "write_verification_report",
]
