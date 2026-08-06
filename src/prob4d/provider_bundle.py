"""Provider-neutral, content-addressed prediction-window bundles.

The bundle binds exact prediction-window bytes to provider, model-set, source,
and causal source-lineage identities without importing a provider implementation.
A valid bundle is an interoperability and provenance statement; it is not evidence
of calibrated uncertainty or downstream physical-twin benefit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

import numpy as np

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._scientific_scalars import require_genuine_integer
from .data import (
    PREDICTION_WINDOW_NPZ_SCHEMA,
    PREDICTION_WINDOW_NPZ_VERSION,
    DenseStorageDType,
    PredictionWindow,
)

PROVIDER_INGEST_SPEC_SCHEMA: Final = "prob4d.provider-window-ingest-spec"
PROVIDER_INGEST_SPEC_VERSION: Final = 1
PROVIDER_WINDOW_BUNDLE_SCHEMA: Final = "prob4d.provider-window-bundle"
PROVIDER_WINDOW_BUNDLE_VERSION: Final = 1
PROVIDER_WINDOW_VERIFICATION_SCHEMA: Final = "prob4d.provider-window-bundle-verification"
PROVIDER_WINDOW_VERIFICATION_VERSION: Final = 1
PROVIDER_WINDOW_CLAIM_BOUNDARY: Final = (
    "This artifact binds exact provider-window bytes, source-frame lineage, and "
    "provider/model/source identities. It does not establish prospective target "
    "calibration, independence between providers, BayesianPhysTwin benefit, "
    "Causal4D intervention benefit, deployment safety, or state of the art."
)

CoordinateSemantics = Literal["independent-window-sim3"]
FrameIndexSemantics = Literal["absolute-source-frame"]
SourceLineageSemantics = Literal["complete-source-interval"]
ArchiveSchema = Literal["prob4d.prediction-window-npz.v2", "legacy-unversioned"]

COORDINATE_SEMANTICS: Final[CoordinateSemantics] = "independent-window-sim3"
FRAME_INDEX_SEMANTICS: Final[FrameIndexSemantics] = "absolute-source-frame"
SOURCE_LINEAGE_SEMANTICS: Final[SourceLineageSemantics] = "complete-source-interval"
VERSIONED_ARCHIVE_SCHEMA: Final[ArchiveSchema] = (
    f"{PREDICTION_WINDOW_NPZ_SCHEMA}.v{PREDICTION_WINDOW_NPZ_VERSION}"
)
LEGACY_ARCHIVE_SCHEMA: Final[ArchiveSchema] = "legacy-unversioned"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _strict_string(value: object, *, name: str, maximum_length: int = 512) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or len(value) > maximum_length:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be Boolean")
    return value


def _strict_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    return require_genuine_integer(
        value,
        name=name,
        minimum=minimum,
        maximum=maximum,
    )


def _strict_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        if type(key) is not str:
            raise TypeError(f"{name} keys must be strings")
    return cast(Mapping[str, Any], value)


def _strict_list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    return cast(list[Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _strict_sha256(value: object, *, name: str) -> str:
    digest = _strict_string(value, name=name, maximum_length=64)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _strict_content_identity(value: object, *, name: str) -> str:
    identity = _strict_string(value, name=name)
    if identity.startswith("sha256:"):
        _strict_sha256(identity.removeprefix("sha256:"), name=name)
        return identity
    if identity.startswith("git:"):
        revision = identity.removeprefix("git:")
        if _GIT_OBJECT.fullmatch(revision) is None:
            raise ValueError(f"{name} git identity must contain a full object ID")
        return identity
    if identity.startswith("oci:sha256:"):
        _strict_sha256(identity.removeprefix("oci:sha256:"), name=name)
        return identity
    raise ValueError(f"{name} must use git:, sha256:, or oci:sha256: identity semantics")


def _safe_relative_path(value: object, *, name: str) -> str:
    text = _strict_string(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must be a POSIX relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    canonical = path.as_posix()
    if canonical != text:
        raise ValueError(f"{name} must be a canonical POSIX relative path")
    return canonical


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_strict_json(path: str | os.PathLike[str], *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is unreadable or invalid strict JSON") from error
    return _strict_mapping(payload, name=name)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read provider payload {path}") from error
    return digest.hexdigest()


def _frame_indices_sha256(frame_indices: np.ndarray) -> str:
    canonical = np.asarray(frame_indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _resolve_payload_member(
    payload_root: str | os.PathLike[str],
    relative_path: object,
    *,
    name: str,
) -> Path:
    relative = _safe_relative_path(relative_path, name=name)
    try:
        root = Path(payload_root).resolve(strict=True)
    except OSError as error:
        raise ValueError("payload_root does not exist") from error
    if not root.is_dir():
        raise ValueError("payload_root must be a directory")

    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        try:
            mode = candidate.lstat().st_mode
        except OSError as error:
            raise ValueError(f"{name} is missing: {relative!r}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} contains a symlink: {relative!r}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} escapes payload_root") from error
    if not resolved.is_file():
        raise ValueError(f"{name} is not a regular file: {relative!r}")
    return resolved


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        information = path.stat()
    except OSError as error:
        raise ValueError(f"cannot stat provider payload {path}") from error
    return (
        int(information.st_dev),
        int(information.st_ino),
        int(information.st_size),
        int(information.st_mtime_ns),
    )


def _archive_schema(
    path: Path,
) -> tuple[ArchiveSchema, DenseStorageDType, str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            files = set(archive.files)
            versioned = {
                "schema_name",
                "schema_version",
                "dense_storage_dtype",
            }.issubset(files)
            point_map = archive["point_map"]
            if point_map.dtype == np.dtype(np.float32):
                storage_dtype: DenseStorageDType = "float32"
            elif point_map.dtype == np.dtype(np.float64):
                storage_dtype = "float64"
            else:
                raise ValueError("prediction point_map must use float32 or float64")
            stored_id_array = archive["window_id"]
            if stored_id_array.shape != () or stored_id_array.dtype.kind not in {"U", "S"}:
                raise ValueError("prediction window_id must be one scalar string")
            stored_id = _strict_string(
                str(stored_id_array.item()),
                name="prediction window_id",
            )
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"provider payload {path} is not a readable prediction archive") from error
    return (
        VERSIONED_ARCHIVE_SCHEMA if versioned else LEGACY_ARCHIVE_SCHEMA,
        storage_dtype,
        stored_id,
    )


def _load_prediction_window(
    path: Path,
    *,
    expected_window_id: str,
    allow_legacy_window_archives: bool,
) -> tuple[PredictionWindow, ArchiveSchema]:
    archive_schema, storage_dtype, stored_window_id = _archive_schema(path)
    if stored_window_id != expected_window_id:
        raise ValueError("prediction archive window_id differs from the ingest specification")
    if archive_schema == LEGACY_ARCHIVE_SCHEMA and not allow_legacy_window_archives:
        raise ValueError(
            "legacy unversioned prediction archives require allow_legacy_window_archives=true"
        )
    try:
        window = PredictionWindow.from_npz(
            path,
            window_id=expected_window_id,
            dense_storage_dtype=storage_dtype,
        )
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"provider prediction window {expected_window_id!r} is invalid") from error
    return window, archive_schema


@dataclass(frozen=True, slots=True)
class ProviderWindowRecordV1:
    """One exact prediction-window payload and its complete causal source interval."""

    window_id: str
    relative_path: str
    payload_sha256: str
    payload_byte_count: int
    archive_schema: ArchiveSchema
    source_frame_start: int
    source_frame_stop_exclusive: int
    output_frame_start: int
    output_frame_stop_exclusive: int
    frame_count: int
    height: int
    width: int
    dense_storage_dtype: DenseStorageDType
    has_scene_flow: bool
    has_ray_directions: bool
    frame_indices_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_id", _strict_string(self.window_id, name="window_id"))
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative_path(self.relative_path, name="relative_path"),
        )
        object.__setattr__(
            self,
            "payload_sha256",
            _strict_sha256(self.payload_sha256, name="payload_sha256"),
        )
        object.__setattr__(
            self,
            "payload_byte_count",
            _strict_integer(
                self.payload_byte_count,
                name="payload_byte_count",
                minimum=1,
            ),
        )
        if self.archive_schema not in {VERSIONED_ARCHIVE_SCHEMA, LEGACY_ARCHIVE_SCHEMA}:
            raise ValueError("unsupported archive_schema")
        source_start = _strict_integer(
            self.source_frame_start,
            name="source_frame_start",
            minimum=0,
        )
        source_stop = _strict_integer(
            self.source_frame_stop_exclusive,
            name="source_frame_stop_exclusive",
            minimum=1,
        )
        output_start = _strict_integer(
            self.output_frame_start,
            name="output_frame_start",
            minimum=0,
        )
        output_stop = _strict_integer(
            self.output_frame_stop_exclusive,
            name="output_frame_stop_exclusive",
            minimum=1,
        )
        if source_stop <= source_start:
            raise ValueError("source frame interval must be nonempty")
        if output_stop <= output_start:
            raise ValueError("output frame interval must be nonempty")
        if output_start < source_start or output_stop > source_stop:
            raise ValueError("output frames must lie inside the complete source interval")
        object.__setattr__(self, "source_frame_start", source_start)
        object.__setattr__(self, "source_frame_stop_exclusive", source_stop)
        object.__setattr__(self, "output_frame_start", output_start)
        object.__setattr__(self, "output_frame_stop_exclusive", output_stop)
        for field_name in ("frame_count", "height", "width"):
            object.__setattr__(
                self,
                field_name,
                _strict_integer(getattr(self, field_name), name=field_name, minimum=1),
            )
        if self.dense_storage_dtype not in {"float32", "float64"}:
            raise ValueError("dense_storage_dtype must be float32 or float64")
        object.__setattr__(
            self,
            "has_scene_flow",
            _strict_bool(self.has_scene_flow, name="has_scene_flow"),
        )
        object.__setattr__(
            self,
            "has_ray_directions",
            _strict_bool(self.has_ray_directions, name="has_ray_directions"),
        )
        object.__setattr__(
            self,
            "frame_indices_sha256",
            _strict_sha256(self.frame_indices_sha256, name="frame_indices_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "relative_path": self.relative_path,
            "payload_sha256": self.payload_sha256,
            "payload_byte_count": self.payload_byte_count,
            "archive_schema": self.archive_schema,
            "source_frame_start": self.source_frame_start,
            "source_frame_stop_exclusive": self.source_frame_stop_exclusive,
            "output_frame_start": self.output_frame_start,
            "output_frame_stop_exclusive": self.output_frame_stop_exclusive,
            "frame_count": self.frame_count,
            "height": self.height,
            "width": self.width,
            "dense_storage_dtype": self.dense_storage_dtype,
            "has_scene_flow": self.has_scene_flow,
            "has_ray_directions": self.has_ray_directions,
            "frame_indices_sha256": self.frame_indices_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderWindowRecordV1:
        mapping = _strict_mapping(value, name="provider window record")
        _exact_keys(
            mapping,
            {
                "window_id",
                "relative_path",
                "payload_sha256",
                "payload_byte_count",
                "archive_schema",
                "source_frame_start",
                "source_frame_stop_exclusive",
                "output_frame_start",
                "output_frame_stop_exclusive",
                "frame_count",
                "height",
                "width",
                "dense_storage_dtype",
                "has_scene_flow",
                "has_ray_directions",
                "frame_indices_sha256",
            },
            name="provider window record",
        )
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class ProviderWindowBundleV1:
    """Provider-neutral manifest over exact independently gauged windows."""

    provider_name: str
    provider_version: str
    implementation_identity: str
    model_set_identity: str
    source_identity: str
    windows: tuple[ProviderWindowRecordV1, ...]
    coordinate_semantics: CoordinateSemantics = COORDINATE_SEMANTICS
    frame_index_semantics: FrameIndexSemantics = FRAME_INDEX_SEMANTICS
    source_lineage_semantics: SourceLineageSemantics = SOURCE_LINEAGE_SEMANTICS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_name",
            _strict_string(self.provider_name, name="provider_name"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _strict_string(self.provider_version, name="provider_version"),
        )
        for field_name in (
            "implementation_identity",
            "model_set_identity",
            "source_identity",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_content_identity(getattr(self, field_name), name=field_name),
            )
        if self.coordinate_semantics != COORDINATE_SEMANTICS:
            raise ValueError("unsupported coordinate_semantics")
        if self.frame_index_semantics != FRAME_INDEX_SEMANTICS:
            raise ValueError("unsupported frame_index_semantics")
        if self.source_lineage_semantics != SOURCE_LINEAGE_SEMANTICS:
            raise ValueError("unsupported source_lineage_semantics")
        if (
            type(self.windows) is not tuple
            or not self.windows
            or not all(isinstance(item, ProviderWindowRecordV1) for item in self.windows)
        ):
            raise ValueError("windows must be a nonempty tuple of ProviderWindowRecordV1")
        ordering = tuple(
            (item.source_frame_start, item.source_frame_stop_exclusive, item.window_id)
            for item in self.windows
        )
        if ordering != tuple(sorted(ordering)):
            raise ValueError("windows must use canonical source-interval ordering")
        for attribute in ("window_id", "relative_path", "payload_sha256"):
            values = tuple(getattr(item, attribute) for item in self.windows)
            if len(values) != len(set(values)):
                raise ValueError(f"provider window {attribute} values must be unique")
        common_contract = {
            (
                item.height,
                item.width,
                item.dense_storage_dtype,
                item.has_scene_flow,
                item.has_ray_directions,
            )
            for item in self.windows
        }
        if len(common_contract) != 1:
            raise ValueError(
                "all provider windows must share resolution, storage dtype, and capabilities"
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        first = self.windows[0]
        values = ["point-map"]
        if first.has_scene_flow:
            values.append("scene-flow")
        if first.has_ray_directions:
            values.append("ray-directions")
        return tuple(values)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_WINDOW_BUNDLE_SCHEMA,
            "schema_version": PROVIDER_WINDOW_BUNDLE_VERSION,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "implementation_identity": self.implementation_identity,
            "model_set_identity": self.model_set_identity,
            "source_identity": self.source_identity,
            "coordinate_semantics": self.coordinate_semantics,
            "frame_index_semantics": self.frame_index_semantics,
            "source_lineage_semantics": self.source_lineage_semantics,
            "capabilities": list(self.capabilities),
            "windows": [item.to_dict() for item in self.windows],
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_WINDOW_CLAIM_BOUNDARY,
        }

    @property
    def bundle_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["bundle_id"] = self.bundle_id
        return result


@dataclass(frozen=True, slots=True)
class _IngestWindowSpec:
    window_id: str
    relative_path: str
    source_frame_start: int
    source_frame_stop_exclusive: int

    @classmethod
    def from_dict(cls, value: object) -> _IngestWindowSpec:
        mapping = _strict_mapping(value, name="ingest window")
        _exact_keys(
            mapping,
            {
                "window_id",
                "path",
                "source_frame_start",
                "source_frame_stop_exclusive",
            },
            name="ingest window",
        )
        return cls(
            window_id=_strict_string(mapping["window_id"], name="window_id"),
            relative_path=_safe_relative_path(mapping["path"], name="path"),
            source_frame_start=_strict_integer(
                mapping["source_frame_start"],
                name="source_frame_start",
                minimum=0,
            ),
            source_frame_stop_exclusive=_strict_integer(
                mapping["source_frame_stop_exclusive"],
                name="source_frame_stop_exclusive",
                minimum=1,
            ),
        )


def _record_from_payload(
    specification: _IngestWindowSpec,
    *,
    payload_root: Path,
    allow_legacy_window_archives: bool,
) -> ProviderWindowRecordV1:
    if specification.source_frame_stop_exclusive <= specification.source_frame_start:
        raise ValueError("ingest source frame interval must be nonempty")
    payload = _resolve_payload_member(
        payload_root,
        specification.relative_path,
        name=f"provider window {specification.window_id!r}",
    )
    before = _file_signature(payload)
    initial_sha = _sha256_file(payload)
    window, archive_schema = _load_prediction_window(
        payload,
        expected_window_id=specification.window_id,
        allow_legacy_window_archives=allow_legacy_window_archives,
    )
    final_sha = _sha256_file(payload)
    after = _file_signature(payload)
    if before != after or initial_sha != final_sha:
        raise ValueError("provider payload changed while it was being admitted")
    return ProviderWindowRecordV1(
        window_id=window.window_id,
        relative_path=specification.relative_path,
        payload_sha256=final_sha,
        payload_byte_count=after[2],
        archive_schema=archive_schema,
        source_frame_start=specification.source_frame_start,
        source_frame_stop_exclusive=specification.source_frame_stop_exclusive,
        output_frame_start=int(window.frame_indices[0]),
        output_frame_stop_exclusive=int(window.frame_indices[-1]) + 1,
        frame_count=int(window.frame_indices.size),
        height=int(window.shape[1]),
        width=int(window.shape[2]),
        dense_storage_dtype=window.dense_storage_dtype,
        has_scene_flow=window.scene_flow is not None,
        has_ray_directions=window.ray_directions is not None,
        frame_indices_sha256=_frame_indices_sha256(window.frame_indices),
    )


def _build_from_spec_mapping(
    specification: Mapping[str, Any],
    *,
    payload_root: Path,
) -> ProviderWindowBundleV1:
    _exact_keys(
        specification,
        {
            "schema_name",
            "schema_version",
            "provider_name",
            "provider_version",
            "implementation_identity",
            "model_set_identity",
            "source_identity",
            "coordinate_semantics",
            "frame_index_semantics",
            "source_lineage_semantics",
            "allow_legacy_window_archives",
            "windows",
            "metadata",
        },
        name="provider ingest specification",
    )
    if specification["schema_name"] != PROVIDER_INGEST_SPEC_SCHEMA:
        raise ValueError("unsupported provider ingest specification schema")
    version = _strict_integer(
        specification["schema_version"],
        name="schema_version",
        minimum=1,
    )
    if version != PROVIDER_INGEST_SPEC_VERSION:
        raise ValueError("unsupported provider ingest specification version")
    allow_legacy = _strict_bool(
        specification["allow_legacy_window_archives"],
        name="allow_legacy_window_archives",
    )
    window_values = _strict_list(specification["windows"], name="windows")
    if not window_values:
        raise ValueError("provider ingest specification must contain windows")
    window_specs = tuple(_IngestWindowSpec.from_dict(item) for item in window_values)
    records = tuple(
        sorted(
            (
                _record_from_payload(
                    item,
                    payload_root=payload_root,
                    allow_legacy_window_archives=allow_legacy,
                )
                for item in window_specs
            ),
            key=lambda item: (
                item.source_frame_start,
                item.source_frame_stop_exclusive,
                item.window_id,
            ),
        )
    )
    return ProviderWindowBundleV1(
        provider_name=specification["provider_name"],
        provider_version=specification["provider_version"],
        implementation_identity=specification["implementation_identity"],
        model_set_identity=specification["model_set_identity"],
        source_identity=specification["source_identity"],
        coordinate_semantics=specification["coordinate_semantics"],
        frame_index_semantics=specification["frame_index_semantics"],
        source_lineage_semantics=specification["source_lineage_semantics"],
        windows=records,
        metadata=_strict_mapping(specification["metadata"], name="metadata"),
    )


def build_provider_window_bundle(
    specification_path: str | os.PathLike[str],
    *,
    payload_root: str | os.PathLike[str] | None = None,
) -> ProviderWindowBundleV1:
    """Build one canonical bundle from a strict provider ingest specification."""

    specification_file = Path(specification_path)
    specification = _load_strict_json(
        specification_file,
        name="provider ingest specification",
    )
    root = specification_file.parent if payload_root is None else Path(payload_root)
    return _build_from_spec_mapping(specification, payload_root=root)


def _verify_motioncrafter_manifest(path: Path) -> Mapping[str, object]:
    from .motioncrafter_integrity import verify_motioncrafter_prediction_manifest

    return verify_motioncrafter_prediction_manifest(path, verify_hashes=True)


def build_motioncrafter_provider_window_bundle(
    manifest_path: str | os.PathLike[str],
    *,
    provider_version: str | None = None,
) -> ProviderWindowBundleV1:
    """Adapt one integrity-bound MotionCrafter prediction manifest."""

    path = Path(manifest_path)
    verification = _verify_motioncrafter_manifest(path)
    if verification.get("integrity_bound") is not True:
        raise ValueError("MotionCrafter provider adaptation requires artifact_integrity")
    manifest = _load_strict_json(path, name="MotionCrafter prediction manifest")
    commit = _strict_string(
        manifest.get("motioncrafter_commit"),
        name="motioncrafter_commit",
    )
    if _GIT_OBJECT.fullmatch(commit) is None:
        raise ValueError("motioncrafter_commit must be a full lowercase Git object ID")
    config = _strict_mapping(manifest.get("config"), name="MotionCrafter config")
    model_set_sha = _strict_sha256(
        config.get("model_source_set_sha256"),
        name="model_source_set_sha256",
    )
    model_type = _strict_string(config.get("model_type"), name="model_type")
    integrity = _strict_mapping(
        manifest.get("artifact_integrity"),
        name="artifact_integrity",
    )
    run_spec_sha = _strict_sha256(
        integrity.get("run_spec_sha256"),
        name="run_spec_sha256",
    )
    run_spec = _strict_mapping(integrity.get("run_spec"), name="run_spec")
    input_video = _strict_mapping(run_spec.get("input_video"), name="input_video")
    source_sha = _strict_sha256(input_video.get("sha256"), name="input video sha256")
    window_values = _strict_list(manifest.get("overlap_windows"), name="overlap_windows")
    windows: list[dict[str, object]] = []
    for value in window_values:
        item = _strict_mapping(value, name="MotionCrafter overlap window")
        windows.append(
            {
                "window_id": item.get("window_id"),
                "path": item.get("path"),
                "source_frame_start": item.get("start_frame"),
                "source_frame_stop_exclusive": item.get("stop_frame"),
            }
        )
    seed_schedule = manifest.get("stochastic_seed_schedule")
    seed_policy: object = "implicit-legacy-common"
    if isinstance(seed_schedule, Mapping):
        seed_policy = seed_schedule.get("policy", seed_policy)
    specification: Mapping[str, Any] = {
        "schema_name": PROVIDER_INGEST_SPEC_SCHEMA,
        "schema_version": PROVIDER_INGEST_SPEC_VERSION,
        "provider_name": "MotionCrafter",
        "provider_version": (
            f"{model_type}@{commit[:12]}" if provider_version is None else provider_version
        ),
        "implementation_identity": f"git:{commit}",
        "model_set_identity": f"sha256:{model_set_sha}",
        "source_identity": f"sha256:{source_sha}",
        "coordinate_semantics": COORDINATE_SEMANTICS,
        "frame_index_semantics": FRAME_INDEX_SEMANTICS,
        "source_lineage_semantics": SOURCE_LINEAGE_SEMANTICS,
        "allow_legacy_window_archives": True,
        "windows": windows,
        "metadata": {
            "adapter": "prob4d.motioncrafter-integrity.v1",
            "motioncrafter_manifest_sha256": _sha256_file(path),
            "motioncrafter_run_spec_sha256": run_spec_sha,
            "motioncrafter_seed_policy": seed_policy,
        },
    }
    return _build_from_spec_mapping(specification, payload_root=path.parent)


def provider_window_bundle_from_dict(value: object) -> ProviderWindowBundleV1:
    """Parse and independently verify one bundle manifest identity."""

    mapping = _strict_mapping(value, name="provider window bundle")
    _exact_keys(
        mapping,
        {
            "schema_name",
            "schema_version",
            "provider_name",
            "provider_version",
            "implementation_identity",
            "model_set_identity",
            "source_identity",
            "coordinate_semantics",
            "frame_index_semantics",
            "source_lineage_semantics",
            "capabilities",
            "windows",
            "metadata",
            "claim_boundary",
            "bundle_id",
        },
        name="provider window bundle",
    )
    if mapping["schema_name"] != PROVIDER_WINDOW_BUNDLE_SCHEMA:
        raise ValueError("unsupported provider window bundle schema")
    version = _strict_integer(mapping["schema_version"], name="schema_version", minimum=1)
    if version != PROVIDER_WINDOW_BUNDLE_VERSION:
        raise ValueError("unsupported provider window bundle version")
    if mapping["claim_boundary"] != PROVIDER_WINDOW_CLAIM_BOUNDARY:
        raise ValueError("provider window bundle claim_boundary mismatch")
    window_values = _strict_list(mapping["windows"], name="windows")
    bundle = ProviderWindowBundleV1(
        provider_name=mapping["provider_name"],
        provider_version=mapping["provider_version"],
        implementation_identity=mapping["implementation_identity"],
        model_set_identity=mapping["model_set_identity"],
        source_identity=mapping["source_identity"],
        coordinate_semantics=mapping["coordinate_semantics"],
        frame_index_semantics=mapping["frame_index_semantics"],
        source_lineage_semantics=mapping["source_lineage_semantics"],
        windows=tuple(ProviderWindowRecordV1.from_dict(item) for item in window_values),
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )
    capabilities = _strict_list(mapping["capabilities"], name="capabilities")
    if capabilities != list(bundle.capabilities):
        raise ValueError("provider window bundle capabilities mismatch")
    bundle_id = _strict_sha256(mapping["bundle_id"], name="bundle_id")
    if bundle_id != bundle.bundle_id:
        raise ValueError("provider window bundle identity mismatch")
    return bundle


def load_provider_window_bundle(
    path: str | os.PathLike[str],
) -> ProviderWindowBundleV1:
    """Load one strict bundle manifest without opening prediction payloads."""

    return provider_window_bundle_from_dict(_load_strict_json(path, name="provider window bundle"))


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


def write_provider_window_bundle(
    bundle: ProviderWindowBundleV1,
    path: str | os.PathLike[str],
) -> None:
    """Persist one canonical bundle atomically without overwriting another identity."""

    if not isinstance(bundle, ProviderWindowBundleV1):
        raise TypeError("bundle must be a ProviderWindowBundleV1")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    lock_path = destination.with_name(f".{destination.name}.lock")
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(f"provider bundle writer is already active: {lock_path}") from error
    temporary_name: str | None = None
    try:
        os.close(lock_descriptor)
        if destination.exists():
            try:
                existing = destination.read_bytes()
            except OSError as error:
                raise ValueError("existing provider bundle cannot be read") from error
            if existing == payload:
                return
            raise FileExistsError(
                "refuse to overwrite a different content-addressed provider bundle"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(destination.parent)


def verify_provider_window_bundle(
    bundle: ProviderWindowBundleV1,
    *,
    payload_root: str | os.PathLike[str],
) -> dict[str, object]:
    """Reopen every exact payload and compare it with the sealed bundle record."""

    if not isinstance(bundle, ProviderWindowBundleV1):
        raise TypeError("bundle must be a ProviderWindowBundleV1")
    root = Path(payload_root)
    total_bytes = 0
    for record in bundle.windows:
        actual = _record_from_payload(
            _IngestWindowSpec(
                window_id=record.window_id,
                relative_path=record.relative_path,
                source_frame_start=record.source_frame_start,
                source_frame_stop_exclusive=(record.source_frame_stop_exclusive),
            ),
            payload_root=root,
            allow_legacy_window_archives=(record.archive_schema == LEGACY_ARCHIVE_SCHEMA),
        )
        if actual.payload_byte_count != record.payload_byte_count:
            raise ValueError(f"payload byte count mismatch for {record.window_id!r}")
        if actual.payload_sha256 != record.payload_sha256:
            raise ValueError(f"payload SHA-256 mismatch for {record.window_id!r}")
        if actual != record:
            raise ValueError(f"payload contract mismatch for {record.window_id!r}")
        total_bytes += record.payload_byte_count
    return {
        "schema_name": PROVIDER_WINDOW_VERIFICATION_SCHEMA,
        "schema_version": PROVIDER_WINDOW_VERIFICATION_VERSION,
        "bundle_id": bundle.bundle_id,
        "provider_name": bundle.provider_name,
        "provider_version": bundle.provider_version,
        "verified_window_count": len(bundle.windows),
        "verified_payload_byte_count": total_bytes,
        "capabilities": list(bundle.capabilities),
        "claim_boundary": PROVIDER_WINDOW_CLAIM_BOUNDARY,
    }


def _metadata_summary(bundle: ProviderWindowBundleV1) -> dict[str, object]:
    return {
        "schema_name": PROVIDER_WINDOW_VERIFICATION_SCHEMA,
        "schema_version": PROVIDER_WINDOW_VERIFICATION_VERSION,
        "bundle_id": bundle.bundle_id,
        "provider_name": bundle.provider_name,
        "provider_version": bundle.provider_version,
        "declared_window_count": len(bundle.windows),
        "capabilities": list(bundle.capabilities),
        "payloads_verified": False,
        "claim_boundary": PROVIDER_WINDOW_CLAIM_BOUNDARY,
    }


def main_ingest(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="build a provider-neutral bundle from a strict ingest specification"
    )
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--payload-root", type=Path)
    arguments = parser.parse_args(argv)
    root = (
        arguments.specification.parent if arguments.payload_root is None else arguments.payload_root
    )
    bundle = build_provider_window_bundle(
        arguments.specification,
        payload_root=root,
    )
    write_provider_window_bundle(bundle, arguments.output)
    report = verify_provider_window_bundle(bundle, payload_root=root)
    report["bundle_path"] = str(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main_ingest_motioncrafter(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="adapt an integrity-bound MotionCrafter manifest to the provider bundle"
    )
    parser.add_argument("predictions_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provider-version")
    arguments = parser.parse_args(argv)
    bundle = build_motioncrafter_provider_window_bundle(
        arguments.predictions_manifest,
        provider_version=arguments.provider_version,
    )
    write_provider_window_bundle(bundle, arguments.output)
    report = verify_provider_window_bundle(
        bundle,
        payload_root=arguments.predictions_manifest.parent,
    )
    report["bundle_path"] = str(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main_validate(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="validate a provider-neutral bundle and optionally its exact payloads"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    arguments = parser.parse_args(argv)
    bundle = load_provider_window_bundle(arguments.bundle)
    if arguments.metadata_only:
        report = _metadata_summary(bundle)
    else:
        root = arguments.bundle.parent if arguments.payload_root is None else arguments.payload_root
        report = verify_provider_window_bundle(bundle, payload_root=root)
        report["payloads_verified"] = True
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "COORDINATE_SEMANTICS",
    "FRAME_INDEX_SEMANTICS",
    "LEGACY_ARCHIVE_SCHEMA",
    "PROVIDER_INGEST_SPEC_SCHEMA",
    "PROVIDER_INGEST_SPEC_VERSION",
    "PROVIDER_WINDOW_BUNDLE_SCHEMA",
    "PROVIDER_WINDOW_BUNDLE_VERSION",
    "PROVIDER_WINDOW_CLAIM_BOUNDARY",
    "PROVIDER_WINDOW_VERIFICATION_SCHEMA",
    "PROVIDER_WINDOW_VERIFICATION_VERSION",
    "SOURCE_LINEAGE_SEMANTICS",
    "VERSIONED_ARCHIVE_SCHEMA",
    "ProviderWindowBundleV1",
    "ProviderWindowRecordV1",
    "build_motioncrafter_provider_window_bundle",
    "build_provider_window_bundle",
    "load_provider_window_bundle",
    "provider_window_bundle_from_dict",
    "verify_provider_window_bundle",
    "write_provider_window_bundle",
]
