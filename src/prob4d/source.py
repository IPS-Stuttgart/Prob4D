"""Provider-neutral source contracts for windowed 4-D predictions.

The contract in this module sits *before* Prob4D's estimator/provider boundary.
It normalizes provider-specific prediction manifests without changing their bytes
or making them part of the claim-bearing observation schemas. Existing
MotionCrafter manifests remain the source of truth and are adapted additively.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._strict_json import load_json_object
from .lineage import motioncrafter_temporal_lineage_manifest
from .motioncrafter import (
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    validate_motioncrafter_seed_schedule,
)
from .motioncrafter_integrity import MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA

WINDOWED_4D_SOURCE_MANIFEST_SCHEMA: Final = "prob4d.windowed-4d-source-manifest"
WINDOWED_4D_SOURCE_MANIFEST_VERSION: Final = 1
MOTIONCRAFTER_SOURCE_ADAPTER_SCHEMA: Final = (
    "prob4d.motioncrafter-prediction-manifest-adapter.v1"
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


def _strict_sha256(value: Any, *, name: str) -> str:
    digest = _strict_string(value, name=name)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} has a noncanonical SHA-256 format")
    return digest


def _strict_optional_sha256(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _strict_sha256(value, name=name)


def _strict_optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _strict_string(value, name=name)


def _safe_relative_path(value: Any, *, name: str) -> str:
    path = _strict_string(value, name=name)
    if "\\" in path:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return pure.as_posix()


def _strict_nonempty_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    mapping = _strict_mapping(value, name=name)
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    return frozen_finite_json_mapping(mapping, name=name)


def _is_exact_revision(value: str) -> bool:
    return len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


def _load_json_bytes_object(raw: bytes, *, name: str) -> dict[str, Any]:
    """Decode one finite JSON object from the exact bytes being authenticated."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON object key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON number {token!r}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except UnicodeError as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _validate_strict_motioncrafter_seed_fields(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Reject coercion-dependent seed fields before legacy validation helpers."""

    root_seed = _strict_integer(config.get("seed"), name="config.seed", minimum=0)
    if root_seed >= 2**32:
        raise ValueError("config.seed must be smaller than 2**32")
    _strict_string(
        config.get("seed_policy", MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON),
        name="config.seed_policy",
    )
    schedule = manifest.get("stochastic_seed_schedule")
    if schedule is None:
        return
    mapping = _strict_mapping(schedule, name="stochastic_seed_schedule")
    _strict_string(mapping.get("schema"), name="stochastic_seed_schedule.schema")
    _strict_string(mapping.get("policy"), name="stochastic_seed_schedule.policy")
    schedule_seed = _strict_integer(
        mapping.get("root_seed"),
        name="stochastic_seed_schedule.root_seed",
        minimum=0,
    )
    if schedule_seed >= 2**32:
        raise ValueError("stochastic_seed_schedule.root_seed must be smaller than 2**32")
    calls = _strict_list(mapping.get("calls"), name="stochastic_seed_schedule.calls")
    for index, value in enumerate(calls):
        call = _strict_mapping(value, name=f"stochastic seed call {index}")
        _strict_string(call.get("call_id"), name=f"stochastic seed call {index} call_id")
        _strict_string(call.get("product"), name=f"stochastic seed call {index} product")
        effective_seed = _strict_integer(
            call.get("effective_seed"),
            name=f"stochastic seed call {index} effective_seed",
            minimum=0,
        )
        if effective_seed >= 2**32:
            raise ValueError(
                f"stochastic seed call {index} effective_seed must be smaller than 2**32"
            )
        if "window_id" in call:
            _strict_string(
                call["window_id"],
                name=f"stochastic seed call {index} window_id",
            )
        for field_name in (
            "source_frame_start",
            "source_frame_stop_exclusive",
        ):
            if field_name in call:
                _strict_integer(
                    call[field_name],
                    name=f"stochastic seed call {index} {field_name}",
                    minimum=0,
                )


@dataclass(frozen=True, slots=True)
class Windowed4DGeometryV1:
    """Nominal source-window geometry, independent of a concrete provider."""

    nominal_window_size: int
    nominal_overlap: int
    frame_stride: int = 1

    def __post_init__(self) -> None:
        window_size = _strict_integer(
            self.nominal_window_size,
            name="nominal_window_size",
            minimum=1,
        )
        overlap = _strict_integer(
            self.nominal_overlap,
            name="nominal_overlap",
            minimum=0,
        )
        stride = _strict_integer(self.frame_stride, name="frame_stride", minimum=1)
        if overlap >= window_size:
            raise ValueError("nominal_overlap must be smaller than nominal_window_size")
        object.__setattr__(self, "nominal_window_size", window_size)
        object.__setattr__(self, "nominal_overlap", overlap)
        object.__setattr__(self, "frame_stride", stride)

    def to_dict(self) -> dict[str, int]:
        return {
            "nominal_window_size": self.nominal_window_size,
            "nominal_overlap": self.nominal_overlap,
            "frame_stride": self.frame_stride,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Windowed4DGeometryV1:
        mapping = _strict_mapping(value, name="window geometry")
        _exact_keys(
            mapping,
            {"nominal_window_size", "nominal_overlap", "frame_stride"},
            name="window geometry",
        )
        return cls(
            nominal_window_size=mapping["nominal_window_size"],
            nominal_overlap=mapping["nominal_overlap"],
            frame_stride=mapping["frame_stride"],
        )


@dataclass(frozen=True, slots=True)
class Windowed4DDataSemanticsV1:
    """Standardized meanings of the dense arrays carried by source windows."""

    point_representation: str
    flow_representation: str
    ray_representation: str
    uncertainty_representation: str

    def __post_init__(self) -> None:
        for name in (
            "point_representation",
            "flow_representation",
            "ray_representation",
            "uncertainty_representation",
        ):
            object.__setattr__(
                self,
                name,
                _strict_string(getattr(self, name), name=name),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "point_representation": self.point_representation,
            "flow_representation": self.flow_representation,
            "ray_representation": self.ray_representation,
            "uncertainty_representation": self.uncertainty_representation,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Windowed4DDataSemanticsV1:
        mapping = _strict_mapping(value, name="data semantics")
        _exact_keys(
            mapping,
            {
                "point_representation",
                "flow_representation",
                "ray_representation",
                "uncertainty_representation",
            },
            name="data semantics",
        )
        return cls(
            point_representation=mapping["point_representation"],
            flow_representation=mapping["flow_representation"],
            ray_representation=mapping["ray_representation"],
            uncertainty_representation=mapping["uncertainty_representation"],
        )


@dataclass(frozen=True, slots=True)
class Windowed4DSourceWindowV1:
    """One independently decoded source window and its optional byte identity."""

    window_id: str
    payload_path: str
    source_frame_start: int
    source_frame_stop_exclusive: int
    payload_sha256: str | None = None
    payload_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        window_id = _strict_string(self.window_id, name="window_id")
        payload_path = _safe_relative_path(self.payload_path, name="payload_path")
        start = _strict_integer(
            self.source_frame_start,
            name="source_frame_start",
            minimum=0,
        )
        stop = _strict_integer(
            self.source_frame_stop_exclusive,
            name="source_frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ValueError("source_frame_stop_exclusive must be greater than start")
        payload_sha256 = _strict_optional_sha256(
            self.payload_sha256,
            name="payload_sha256",
        )
        if self.payload_bytes is None:
            payload_bytes = None
        else:
            payload_bytes = _strict_integer(
                self.payload_bytes,
                name="payload_bytes",
                minimum=0,
            )
        if (payload_sha256 is None) != (payload_bytes is None):
            raise ValueError("payload_sha256 and payload_bytes must be supplied together")
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "payload_path", payload_path)
        object.__setattr__(self, "source_frame_start", start)
        object.__setattr__(self, "source_frame_stop_exclusive", stop)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "payload_bytes", payload_bytes)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="window metadata"),
        )

    @property
    def payload_identity_bound(self) -> bool:
        return self.payload_sha256 is not None and self.payload_bytes is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "payload_path": self.payload_path,
            "source_frame_start": self.source_frame_start,
            "source_frame_stop_exclusive": self.source_frame_stop_exclusive,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> Windowed4DSourceWindowV1:
        mapping = _strict_mapping(value, name="source window")
        _exact_keys(
            mapping,
            {
                "window_id",
                "payload_path",
                "source_frame_start",
                "source_frame_stop_exclusive",
                "payload_sha256",
                "payload_bytes",
                "metadata",
            },
            name="source window",
        )
        return cls(
            window_id=mapping["window_id"],
            payload_path=mapping["payload_path"],
            source_frame_start=mapping["source_frame_start"],
            source_frame_stop_exclusive=mapping["source_frame_stop_exclusive"],
            payload_sha256=mapping["payload_sha256"],
            payload_bytes=mapping["payload_bytes"],
            metadata=_strict_mapping(mapping["metadata"], name="window metadata"),
        )


@dataclass(frozen=True, slots=True)
class Windowed4DSourceManifestV1:
    """Content-addressed neutral view of one provider-specific source manifest."""

    source_provider_id: str
    source_provider_revision: str
    model_set_id: str | None
    source_manifest_sha256: str
    source_manifest_file_sha256: str | None
    coordinate_frame: str
    length_unit: str
    geometry: Windowed4DGeometryV1
    data_semantics: Windowed4DDataSemanticsV1
    stochastic_policy: Mapping[str, Any]
    temporal_lineage: Mapping[str, Any]
    windows: tuple[Windowed4DSourceWindowV1, ...]
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = WINDOWED_4D_SOURCE_MANIFEST_SCHEMA
    schema_version: int = WINDOWED_4D_SOURCE_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != WINDOWED_4D_SOURCE_MANIFEST_SCHEMA:
            raise ValueError("unexpected windowed 4-D source-manifest schema")
        if (
            _strict_integer(self.schema_version, name="schema_version", minimum=1)
            != WINDOWED_4D_SOURCE_MANIFEST_VERSION
        ):
            raise ValueError("unsupported windowed 4-D source-manifest version")
        provider_id = _strict_string(self.source_provider_id, name="source_provider_id")
        revision = _strict_string(
            self.source_provider_revision,
            name="source_provider_revision",
        )
        model_set_id = _strict_optional_string(self.model_set_id, name="model_set_id")
        source_manifest_sha256 = _strict_sha256(
            self.source_manifest_sha256,
            name="source_manifest_sha256",
        )
        source_manifest_file_sha256 = _strict_optional_sha256(
            self.source_manifest_file_sha256,
            name="source_manifest_file_sha256",
        )
        coordinate_frame = _strict_string(self.coordinate_frame, name="coordinate_frame")
        length_unit = _strict_string(self.length_unit, name="length_unit")
        if not isinstance(self.geometry, Windowed4DGeometryV1):
            raise ValueError("geometry must be a Windowed4DGeometryV1 value")
        if not isinstance(self.data_semantics, Windowed4DDataSemanticsV1):
            raise ValueError("data_semantics must be a Windowed4DDataSemanticsV1 value")
        windows = tuple(self.windows)
        if not windows:
            raise ValueError("windowed 4-D source manifest must contain at least one window")
        if any(not isinstance(window, Windowed4DSourceWindowV1) for window in windows):
            raise ValueError("windows must contain Windowed4DSourceWindowV1 values")
        window_ids = [window.window_id for window in windows]
        payload_paths = [window.payload_path for window in windows]
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("source window IDs must be unique")
        if len(set(payload_paths)) != len(payload_paths):
            raise ValueError("source window payload paths must be unique")
        object.__setattr__(self, "source_provider_id", provider_id)
        object.__setattr__(self, "source_provider_revision", revision)
        object.__setattr__(self, "model_set_id", model_set_id)
        object.__setattr__(self, "source_manifest_sha256", source_manifest_sha256)
        object.__setattr__(
            self,
            "source_manifest_file_sha256",
            source_manifest_file_sha256,
        )
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "length_unit", length_unit)
        object.__setattr__(
            self,
            "stochastic_policy",
            _strict_nonempty_mapping(self.stochastic_policy, name="stochastic_policy"),
        )
        object.__setattr__(
            self,
            "temporal_lineage",
            _strict_nonempty_mapping(self.temporal_lineage, name="temporal_lineage"),
        )
        object.__setattr__(self, "windows", windows)
        object.__setattr__(
            self,
            "provider_metadata",
            frozen_finite_json_mapping(
                self.provider_metadata,
                name="provider_metadata",
            ),
        )

    @property
    def claim_ready_source_identity(self) -> bool:
        """Whether source and payload identities are complete enough for a frozen run.

        This does not establish provider accuracy or authorize a Bayesian update.
        It only reports whether the neutral source descriptor carries an exact
        provider revision, model-set identity, original file digest, and payload
        identities for every retained window.
        """

        return (
            _is_exact_revision(self.source_provider_revision)
            and self.model_set_id is not None
            and self.source_manifest_file_sha256 is not None
            and all(window.payload_identity_bound for window in self.windows)
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "source_provider_id": self.source_provider_id,
            "source_provider_revision": self.source_provider_revision,
            "model_set_id": self.model_set_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_file_sha256": self.source_manifest_file_sha256,
            "coordinate_frame": self.coordinate_frame,
            "length_unit": self.length_unit,
            "geometry": self.geometry.to_dict(),
            "data_semantics": self.data_semantics.to_dict(),
            "stochastic_policy": plain_json(self.stochastic_policy),
            "temporal_lineage": plain_json(self.temporal_lineage),
            "windows": [window.to_dict() for window in self.windows],
            "provider_metadata": plain_json(self.provider_metadata),
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_dict(cls, value: Any) -> Windowed4DSourceManifestV1:
        mapping = _strict_mapping(value, name="windowed 4-D source manifest")
        _exact_keys(
            mapping,
            {
                "artifact_id",
                "schema_name",
                "schema_version",
                "source_provider_id",
                "source_provider_revision",
                "model_set_id",
                "source_manifest_sha256",
                "source_manifest_file_sha256",
                "coordinate_frame",
                "length_unit",
                "geometry",
                "data_semantics",
                "stochastic_policy",
                "temporal_lineage",
                "windows",
                "provider_metadata",
            },
            name="windowed 4-D source manifest",
        )
        artifact_id = _strict_sha256(mapping["artifact_id"], name="artifact_id")
        windows = tuple(
            Windowed4DSourceWindowV1.from_dict(item)
            for item in _strict_list(mapping["windows"], name="windows")
        )
        artifact = cls(
            schema_name=mapping["schema_name"],
            schema_version=mapping["schema_version"],
            source_provider_id=mapping["source_provider_id"],
            source_provider_revision=mapping["source_provider_revision"],
            model_set_id=mapping["model_set_id"],
            source_manifest_sha256=mapping["source_manifest_sha256"],
            source_manifest_file_sha256=mapping["source_manifest_file_sha256"],
            coordinate_frame=mapping["coordinate_frame"],
            length_unit=mapping["length_unit"],
            geometry=Windowed4DGeometryV1.from_dict(mapping["geometry"]),
            data_semantics=Windowed4DDataSemanticsV1.from_dict(
                mapping["data_semantics"]
            ),
            stochastic_policy=_strict_mapping(
                mapping["stochastic_policy"],
                name="stochastic_policy",
            ),
            temporal_lineage=_strict_mapping(
                mapping["temporal_lineage"],
                name="temporal_lineage",
            ),
            windows=windows,
            provider_metadata=_strict_mapping(
                mapping["provider_metadata"],
                name="provider_metadata",
            ),
        )
        if artifact_id != artifact.artifact_id:
            raise ValueError("windowed 4-D source artifact_id does not match content")
        return artifact


def _integrity_members(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    integrity = manifest.get("artifact_integrity")
    if integrity is None:
        return {}
    mapping = _strict_mapping(integrity, name="artifact_integrity")
    if mapping.get("schema") != MOTIONCRAFTER_ARTIFACT_INTEGRITY_SCHEMA:
        raise ValueError("unsupported MotionCrafter artifact-integrity schema")
    members = _strict_list(mapping.get("members"), name="artifact_integrity members")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(members):
        descriptor = _strict_mapping(value, name=f"artifact member {index}")
        path = _safe_relative_path(
            descriptor.get("path"),
            name=f"artifact member {index} path",
        )
        if path in result:
            raise ValueError(f"duplicate artifact-integrity member path {path!r}")
        _strict_sha256(
            descriptor.get("sha256"),
            name=f"artifact member {path!r} sha256",
        )
        _strict_integer(
            descriptor.get("bytes"),
            name=f"artifact member {path!r} bytes",
            minimum=0,
        )
        _strict_string(
            descriptor.get("kind"),
            name=f"artifact member {path!r} kind",
        )
        result[path] = descriptor
    return result


def _motioncrafter_model_set_id(config: Mapping[str, Any]) -> str | None:
    schema = config.get("model_source_schema")
    digest = config.get("model_source_set_sha256")
    if schema is None and digest is None:
        return None
    if schema is None or digest is None:
        raise ValueError("MotionCrafter model-source schema and digest must be paired")
    source_schema = _strict_string(schema, name="model_source_schema")
    source_digest = _strict_sha256(digest, name="model_source_set_sha256")
    return f"{source_schema}:{source_digest}"


def adapt_motioncrafter_prediction_manifest(
    manifest: Mapping[str, Any],
    *,
    source_manifest_file_sha256: str | None = None,
) -> Windowed4DSourceManifestV1:
    """Normalize a parsed MotionCrafter ``predictions.json`` without payload I/O."""

    try:
        normalized = json.loads(
            json.dumps(plain_json(manifest), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("MotionCrafter prediction manifest must be finite JSON") from error
    if not isinstance(normalized, dict):
        raise ValueError("MotionCrafter prediction manifest must be a JSON object")
    if (
        _strict_integer(normalized.get("format_version"), name="format_version", minimum=1)
        != 1
    ):
        raise ValueError("unsupported MotionCrafter prediction-manifest format_version")
    source_revision = _strict_string(
        normalized.get("motioncrafter_commit"),
        name="motioncrafter_commit",
    )
    config = _strict_mapping(normalized.get("config"), name="config")
    window_size = _strict_integer(
        config.get("window_size"),
        name="config.window_size",
        minimum=1,
    )
    overlap = _strict_integer(
        config.get("overlap"),
        name="config.overlap",
        minimum=0,
    )
    frame_stride = _strict_integer(
        config.get("frame_stride", 1),
        name="config.frame_stride",
        minimum=1,
    )
    _validate_strict_motioncrafter_seed_fields(normalized, config)
    seed_report = validate_motioncrafter_seed_schedule(normalized)
    raw_temporal_lineage = normalized.get("temporal_lineage")
    if raw_temporal_lineage is None:
        temporal_lineage = frozen_finite_json_mapping(
            motioncrafter_temporal_lineage_manifest(
                window_size=window_size,
                overlap=overlap,
            ),
            name="temporal_lineage",
        )
        temporal_lineage_source = "reconstructed-from-legacy-config"
    else:
        temporal_lineage = _strict_nonempty_mapping(
            raw_temporal_lineage,
            name="temporal_lineage",
        )
        temporal_lineage_source = "manifest"
    raw_windows = _strict_list(normalized.get("overlap_windows"), name="overlap_windows")
    integrity_members = _integrity_members(normalized)
    integrity_is_bound = "artifact_integrity" in normalized
    windows: list[Windowed4DSourceWindowV1] = []
    for index, value in enumerate(raw_windows):
        record = _strict_mapping(value, name=f"overlap window {index}")
        window_id = _strict_string(
            record.get("window_id"),
            name=f"overlap window {index} window_id",
        )
        payload_path = _safe_relative_path(
            record.get("path"),
            name=f"overlap window {window_id!r} path",
        )
        descriptor = integrity_members.get(payload_path)
        if integrity_is_bound and descriptor is None:
            raise ValueError(
                f"integrity-bound manifest lacks descriptor for window {window_id!r}"
            )
        payload_sha256: str | None = None
        payload_bytes: int | None = None
        metadata: dict[str, Any] = {}
        if descriptor is not None:
            if descriptor.get("kind") != "independently_decoded_overlap_window":
                raise ValueError(
                    f"artifact kind changed for overlap window {window_id!r}"
                )
            payload_sha256 = _strict_sha256(
                descriptor.get("sha256"),
                name=f"overlap window {window_id!r} sha256",
            )
            payload_bytes = _strict_integer(
                descriptor.get("bytes"),
                name=f"overlap window {window_id!r} bytes",
                minimum=0,
            )
            metadata["source_artifact_kind"] = descriptor["kind"]
        windows.append(
            Windowed4DSourceWindowV1(
                window_id=window_id,
                payload_path=payload_path,
                source_frame_start=_strict_integer(
                    record.get("start_frame"),
                    name=f"overlap window {window_id!r} start_frame",
                    minimum=0,
                ),
                source_frame_stop_exclusive=_strict_integer(
                    record.get("stop_frame"),
                    name=f"overlap window {window_id!r} stop_frame",
                    minimum=1,
                ),
                payload_sha256=payload_sha256,
                payload_bytes=payload_bytes,
                metadata=metadata,
            )
        )
    if not windows:
        raise ValueError("MotionCrafter prediction manifest has no overlap windows")

    schedule = normalized.get("stochastic_seed_schedule")
    schedule_sha256 = (
        _sha256_json(_strict_mapping(schedule, name="stochastic_seed_schedule"))
        if schedule is not None
        else None
    )
    stochastic_policy = {
        "schema": "prob4d.windowed-4d-stochastic-policy.v1",
        "provider_schema": seed_report["schema"],
        "policy": seed_report["policy"],
        "root_seed": seed_report["root_seed"],
        "call_count": seed_report["call_count"],
        "schedule_source": seed_report["source"],
        "schedule_sha256": schedule_sha256,
    }
    disjoint_path = _safe_relative_path(
        normalized.get("disjoint_baseline"),
        name="disjoint_baseline",
    )
    latent_path = _safe_relative_path(
        normalized.get("latent_linear_baseline"),
        name="latent_linear_baseline",
    )
    integrity = normalized.get("artifact_integrity")
    provider_metadata: dict[str, Any] = {
        "adapter_schema": MOTIONCRAFTER_SOURCE_ADAPTER_SCHEMA,
        "source_format_version": 1,
        "disjoint_baseline_path": disjoint_path,
        "latent_linear_baseline_path": latent_path,
        "artifact_integrity_bound": integrity is not None,
        "temporal_lineage_source": temporal_lineage_source,
    }
    if integrity is not None:
        integrity_mapping = _strict_mapping(integrity, name="artifact_integrity")
        provider_metadata["artifact_integrity_schema"] = integrity_mapping["schema"]
        run_spec_sha256 = integrity_mapping.get("run_spec_sha256")
        if run_spec_sha256 is not None:
            provider_metadata["run_spec_sha256"] = _strict_sha256(
                run_spec_sha256,
                name="artifact_integrity.run_spec_sha256",
            )
    return Windowed4DSourceManifestV1(
        source_provider_id="motioncrafter",
        source_provider_revision=source_revision,
        model_set_id=_motioncrafter_model_set_id(config),
        source_manifest_sha256=_sha256_json(normalized),
        source_manifest_file_sha256=_strict_optional_sha256(
            source_manifest_file_sha256,
            name="source_manifest_file_sha256",
        ),
        coordinate_frame="window-local-sim3-gauge",
        length_unit="provider-native-unscaled",
        geometry=Windowed4DGeometryV1(
            nominal_window_size=window_size,
            nominal_overlap=overlap,
            frame_stride=frame_stride,
        ),
        data_semantics=Windowed4DDataSemanticsV1(
            point_representation="dense-3d-point-map-in-window-gauge",
            flow_representation="optional-forward-3d-scene-flow-in-window-gauge",
            ray_representation="optional-calibrated-viewing-direction",
            uncertainty_representation="not-embedded; calibrated-by-prob4d",
        ),
        stochastic_policy=stochastic_policy,
        temporal_lineage=temporal_lineage,
        windows=tuple(windows),
        provider_metadata=provider_metadata,
    )


def load_motioncrafter_source_manifest(
    path: str | Path,
) -> Windowed4DSourceManifestV1:
    """Strictly load and adapt one MotionCrafter source manifest.

    Only the JSON manifest is opened. Prediction payloads remain unopened until a
    downstream Prob4D source selector admits them.
    """

    source = Path(path)
    raw = source.read_bytes()
    manifest = _load_json_bytes_object(
        raw,
        name="MotionCrafter prediction manifest",
    )
    return adapt_motioncrafter_prediction_manifest(
        manifest,
        source_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _serialized_source_manifest(artifact: Windowed4DSourceManifestV1) -> bytes:
    return (
        json.dumps(artifact.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def save_windowed_4d_source_manifest(
    artifact: Windowed4DSourceManifestV1,
    path: str | Path,
) -> None:
    """Append-only persistence for a normalized source-manifest artifact."""

    target = Path(path)
    payload = _serialized_source_manifest(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == payload:
            return
        raise FileExistsError(
            f"refusing to replace a different windowed 4-D source artifact: {target}"
        )
    descriptor = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(descriptor.name)
    try:
        with descriptor:
            descriptor.write(payload)
            descriptor.flush()
            os.fsync(descriptor.fileno())
        if temporary.read_bytes() != payload:
            raise OSError("temporary source-manifest artifact failed validation")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() == payload:
                return
            raise FileExistsError(
                f"refusing to replace a different windowed 4-D source artifact: {target}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def load_windowed_4d_source_manifest(
    path: str | Path,
) -> Windowed4DSourceManifestV1:
    payload = load_json_object(path, name="windowed 4-D source manifest")
    return Windowed4DSourceManifestV1.from_dict(payload)


__all__ = [
    "MOTIONCRAFTER_SOURCE_ADAPTER_SCHEMA",
    "WINDOWED_4D_SOURCE_MANIFEST_SCHEMA",
    "WINDOWED_4D_SOURCE_MANIFEST_VERSION",
    "Windowed4DDataSemanticsV1",
    "Windowed4DGeometryV1",
    "Windowed4DSourceManifestV1",
    "Windowed4DSourceWindowV1",
    "adapt_motioncrafter_prediction_manifest",
    "load_motioncrafter_source_manifest",
    "load_windowed_4d_source_manifest",
    "save_windowed_4d_source_manifest",
]
