"""Provider-neutral, content-addressed manifests for canonical 4-D predictions.

The contract deliberately separates a prediction provider's identity and causal
source lineage from provider-specific launch metadata.  Payloads are canonical
``PredictionWindow`` NPZ archives; paths are retrieval metadata, while payload
bytes, output-frame identities, source dependencies, stochastic members, and
shared-dependence groups determine portable identities.
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
from typing import Any, Final

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
    require_string_sequence,
)
from .data import (
    DENSE_STORAGE_DTYPES,
    PREDICTION_WINDOW_NPZ_SCHEMA,
    PREDICTION_WINDOW_NPZ_VERSION,
    PredictionWindow,
)

PREDICTION_PROVIDER_MANIFEST_SCHEMA: Final = (
    "prob4d.prediction-provider-manifest"
)
PREDICTION_PROVIDER_MANIFEST_VERSION: Final = 1
SOURCE_DEPENDENCY_SEMANTICS: Final = (
    "per-output-exclusive-source-frame-interval-v1"
)

COORDINATE_SEMANTICS: Final = (
    "window-local-sim3",
    "sequence-local-sim3",
    "camera-local-metric",
    "metric-world",
)
PRODUCT_ROLES: Final = (
    "independent-window",
    "disjoint-baseline",
    "overlap-baseline",
    "external-sequence",
)
POINT_SEMANTICS: Final = ("dense-point-map",)
FLOW_SEMANTICS: Final = ("absent", "forward-point-displacement")
RAY_SEMANTICS: Final = ("absent", "camera-ray-unit-vector")

_FRAME_LINEAGE_FIELDS: Final = frozenset(
    {
        "output_frame_id",
        "source_frame_start",
        "source_frame_stop_exclusive",
        "contributor_ids",
    }
)
_PAYLOAD_FIELDS: Final = frozenset(
    {
        "payload_id",
        "product_role",
        "window_id",
        "path",
        "sha256",
        "byte_count",
        "view_id",
        "stochastic_member_id",
        "dependence_group_ids",
        "dense_storage_dtype",
        "payload_schema",
        "payload_schema_version",
        "has_scene_flow",
        "has_ray_directions",
        "frame_lineage",
    }
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "sequence_id",
        "provider_family",
        "provider_repository",
        "provider_revision",
        "provider_run_id",
        "model_set_id",
        "loader_id",
        "coordinate_semantics",
        "point_semantics",
        "flow_semantics",
        "ray_semantics",
        "source_dependency_semantics",
        "payloads",
        "metadata",
    }
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
        raise ValueError(f"cannot read prediction payload {path.name!r}") from error
    return digest.hexdigest()


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return bool(value)


def _require_choice(value: object, choices: Sequence[str], *, name: str) -> str:
    normalized = require_exact_string(value, name=name)
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return normalized


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


def _relative_member(path: Path, *, root: Path, name: str) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} must lie inside the manifest directory") from error
    return _safe_relative_path(relative.as_posix(), name=name)


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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class PredictionFrameLineageV1:
    """Exact causal source interval for one output frame."""

    output_frame_id: int
    source_frame_start: int
    source_frame_stop_exclusive: int
    contributor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        output = require_exact_integer(
            self.output_frame_id,
            name="output_frame_id",
            minimum=0,
        )
        start = require_exact_integer(
            self.source_frame_start,
            name="source_frame_start",
            minimum=0,
        )
        stop = require_exact_integer(
            self.source_frame_stop_exclusive,
            name="source_frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ValueError("source frame stop must be greater than source frame start")
        if type(self.contributor_ids) is not tuple:
            raise TypeError("contributor_ids must be a canonical tuple")
        contributors = require_string_sequence(
            self.contributor_ids,
            name="contributor_ids",
        )
        if len(set(contributors)) != len(contributors):
            raise ValueError("contributor_ids must be unique")
        object.__setattr__(self, "output_frame_id", output)
        object.__setattr__(self, "source_frame_start", start)
        object.__setattr__(self, "source_frame_stop_exclusive", stop)
        object.__setattr__(self, "contributor_ids", contributors)

    def to_record(self) -> dict[str, object]:
        return {
            "output_frame_id": self.output_frame_id,
            "source_frame_start": self.source_frame_start,
            "source_frame_stop_exclusive": self.source_frame_stop_exclusive,
            "contributor_ids": list(self.contributor_ids),
        }

    @classmethod
    def from_record(cls, value: object) -> PredictionFrameLineageV1:
        mapping = require_mapping(value, name="prediction frame lineage")
        require_exact_fields(mapping, _FRAME_LINEAGE_FIELDS, name="frame lineage")
        contributors = require_string_sequence(
            mapping["contributor_ids"],
            name="contributor_ids",
        )
        return cls(
            output_frame_id=mapping["output_frame_id"],
            source_frame_start=mapping["source_frame_start"],
            source_frame_stop_exclusive=mapping["source_frame_stop_exclusive"],
            contributor_ids=contributors,
        )


@dataclass(frozen=True)
class PredictionPayloadDescriptorV1:
    """One canonical prediction payload and its causal/dependence semantics."""

    product_role: str
    window_id: str
    path: str
    sha256: str
    byte_count: int
    view_id: str
    stochastic_member_id: str
    dependence_group_ids: tuple[str, ...]
    dense_storage_dtype: str
    has_scene_flow: bool
    has_ray_directions: bool
    frame_lineage: tuple[PredictionFrameLineageV1, ...]
    payload_schema: str = PREDICTION_WINDOW_NPZ_SCHEMA
    payload_schema_version: int = PREDICTION_WINDOW_NPZ_VERSION
    payload_id: str | None = None

    def __post_init__(self) -> None:
        role = _require_choice(self.product_role, PRODUCT_ROLES, name="product_role")
        window_id = require_exact_string(self.window_id, name="window_id")
        path = _safe_relative_path(self.path, name="payload path")
        sha256 = require_sha256(self.sha256, name="payload sha256")
        byte_count = require_exact_integer(
            self.byte_count,
            name="payload byte_count",
            minimum=1,
        )
        view_id = require_exact_string(self.view_id, name="view_id")
        stochastic_member_id = require_exact_string(
            self.stochastic_member_id,
            name="stochastic_member_id",
        )
        if type(self.dependence_group_ids) is not tuple:
            raise TypeError("dependence_group_ids must be a canonical tuple")
        dependence_groups = require_string_sequence(
            self.dependence_group_ids,
            name="dependence_group_ids",
        )
        if len(set(dependence_groups)) != len(dependence_groups):
            raise ValueError("dependence_group_ids must be unique")
        dense_dtype = _require_choice(
            self.dense_storage_dtype,
            DENSE_STORAGE_DTYPES,
            name="dense_storage_dtype",
        )
        payload_schema = require_exact_string(
            self.payload_schema,
            name="payload_schema",
        )
        if payload_schema != PREDICTION_WINDOW_NPZ_SCHEMA:
            raise ValueError("unsupported canonical prediction payload schema")
        payload_schema_version = require_exact_integer(
            self.payload_schema_version,
            name="payload_schema_version",
            minimum=1,
        )
        if payload_schema_version != PREDICTION_WINDOW_NPZ_VERSION:
            raise ValueError("unsupported canonical prediction payload schema version")
        has_scene_flow = _require_boolean(
            self.has_scene_flow,
            name="has_scene_flow",
        )
        has_ray_directions = _require_boolean(
            self.has_ray_directions,
            name="has_ray_directions",
        )
        if type(self.frame_lineage) is not tuple or not self.frame_lineage:
            raise TypeError("frame_lineage must be a nonempty canonical tuple")
        lineage = tuple(self.frame_lineage)
        if any(not isinstance(item, PredictionFrameLineageV1) for item in lineage):
            raise TypeError("frame_lineage must contain PredictionFrameLineageV1 values")
        frame_ids = tuple(item.output_frame_id for item in lineage)
        if tuple(sorted(frame_ids)) != frame_ids or len(set(frame_ids)) != len(frame_ids):
            raise ValueError("output frame IDs must be strictly increasing and unique")

        object.__setattr__(self, "product_role", role)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "view_id", view_id)
        object.__setattr__(self, "stochastic_member_id", stochastic_member_id)
        object.__setattr__(self, "dependence_group_ids", dependence_groups)
        object.__setattr__(self, "dense_storage_dtype", dense_dtype)
        object.__setattr__(self, "payload_schema", payload_schema)
        object.__setattr__(self, "payload_schema_version", payload_schema_version)
        object.__setattr__(self, "has_scene_flow", has_scene_flow)
        object.__setattr__(self, "has_ray_directions", has_ray_directions)
        object.__setattr__(self, "frame_lineage", lineage)

        expected_id = _sha256_json(self.identity_record())
        supplied = self.payload_id
        if supplied is not None and require_sha256(
            supplied,
            name="payload_id",
        ) != expected_id:
            raise ValueError("prediction payload ID mismatch")
        object.__setattr__(self, "payload_id", expected_id)

    @property
    def output_frame_ids(self) -> tuple[int, ...]:
        return tuple(item.output_frame_id for item in self.frame_lineage)

    @property
    def source_frame_start(self) -> int:
        return min(item.source_frame_start for item in self.frame_lineage)

    @property
    def source_frame_stop_exclusive(self) -> int:
        return max(item.source_frame_stop_exclusive for item in self.frame_lineage)

    def is_causally_admitted(self, causal_frame_stop: int) -> bool:
        cutoff = require_exact_integer(
            causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        return all(
            item.source_frame_stop_exclusive <= cutoff for item in self.frame_lineage
        )

    def identity_record(self) -> dict[str, object]:
        return {
            "product_role": self.product_role,
            "window_id": self.window_id,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "view_id": self.view_id,
            "stochastic_member_id": self.stochastic_member_id,
            "dependence_group_ids": list(self.dependence_group_ids),
            "dense_storage_dtype": self.dense_storage_dtype,
            "payload_schema": self.payload_schema,
            "payload_schema_version": self.payload_schema_version,
            "has_scene_flow": self.has_scene_flow,
            "has_ray_directions": self.has_ray_directions,
            "frame_lineage": [item.to_record() for item in self.frame_lineage],
        }

    def to_record(self) -> dict[str, object]:
        return {
            "payload_id": self.payload_id,
            "path": self.path,
            **self.identity_record(),
        }

    @classmethod
    def from_record(cls, value: object) -> PredictionPayloadDescriptorV1:
        mapping = require_mapping(value, name="prediction payload descriptor")
        require_exact_fields(mapping, _PAYLOAD_FIELDS, name="payload descriptor")
        dependence_groups = require_string_sequence(
            mapping["dependence_group_ids"],
            name="dependence_group_ids",
        )
        raw_lineage = mapping["frame_lineage"]
        if not isinstance(raw_lineage, list) or not raw_lineage:
            raise ValueError("frame_lineage must be a nonempty JSON array")
        return cls(
            product_role=mapping["product_role"],
            window_id=mapping["window_id"],
            path=mapping["path"],
            sha256=mapping["sha256"],
            byte_count=mapping["byte_count"],
            view_id=mapping["view_id"],
            stochastic_member_id=mapping["stochastic_member_id"],
            dependence_group_ids=dependence_groups,
            dense_storage_dtype=mapping["dense_storage_dtype"],
            payload_schema=mapping["payload_schema"],
            payload_schema_version=mapping["payload_schema_version"],
            has_scene_flow=mapping["has_scene_flow"],
            has_ray_directions=mapping["has_ray_directions"],
            frame_lineage=tuple(
                PredictionFrameLineageV1.from_record(item) for item in raw_lineage
            ),
            payload_id=mapping["payload_id"],
        )


@dataclass(frozen=True)
class PredictionProviderManifestV1:
    """Portable provider identity plus canonical prediction payloads."""

    sequence_id: str
    provider_family: str
    provider_repository: str
    provider_revision: str
    provider_run_id: str
    model_set_id: str
    loader_id: str
    coordinate_semantics: str
    point_semantics: str
    flow_semantics: str
    ray_semantics: str
    payloads: tuple[PredictionPayloadDescriptorV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_dependency_semantics: str = SOURCE_DEPENDENCY_SEMANTICS
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        string_fields = {
            "sequence_id": self.sequence_id,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
        }
        normalized_strings = {
            name: require_exact_string(value, name=name)
            for name, value in string_fields.items()
        }
        revision = require_revision(
            self.provider_revision,
            name="provider_revision",
        )
        provider_run_id = require_sha256(
            self.provider_run_id,
            name="provider_run_id",
        )
        model_set_id = require_sha256(self.model_set_id, name="model_set_id")
        loader_id = require_sha256(self.loader_id, name="loader_id")
        coordinate_semantics = _require_choice(
            self.coordinate_semantics,
            COORDINATE_SEMANTICS,
            name="coordinate_semantics",
        )
        point_semantics = _require_choice(
            self.point_semantics,
            POINT_SEMANTICS,
            name="point_semantics",
        )
        flow_semantics = _require_choice(
            self.flow_semantics,
            FLOW_SEMANTICS,
            name="flow_semantics",
        )
        ray_semantics = _require_choice(
            self.ray_semantics,
            RAY_SEMANTICS,
            name="ray_semantics",
        )
        source_semantics = require_exact_string(
            self.source_dependency_semantics,
            name="source_dependency_semantics",
        )
        if source_semantics != SOURCE_DEPENDENCY_SEMANTICS:
            raise ValueError("unsupported source-dependency semantics")
        if type(self.payloads) is not tuple or not self.payloads:
            raise TypeError("payloads must be a nonempty canonical tuple")
        payloads = tuple(self.payloads)
        if any(not isinstance(item, PredictionPayloadDescriptorV1) for item in payloads):
            raise TypeError("payloads must contain PredictionPayloadDescriptorV1 values")
        payload_paths = [item.path for item in payloads]
        if len(set(payload_paths)) != len(payload_paths):
            raise ValueError("payload paths must be unique")
        payload_ids: list[str] = []
        for item in payloads:
            if item.payload_id is None:
                raise ValueError("payload IDs must be materialized")
            payload_ids.append(item.payload_id)
        if len(set(payload_ids)) != len(payload_ids):
            raise ValueError("payload IDs must be unique")
        window_ids = [item.window_id for item in payloads]
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("window IDs must be unique")
        if flow_semantics == "absent" and any(item.has_scene_flow for item in payloads):
            raise ValueError("manifest flow semantics contradict payload contents")
        if flow_semantics != "absent" and not all(
            item.has_scene_flow for item in payloads
        ):
            raise ValueError("all payloads must provide the declared flow semantics")
        if ray_semantics == "absent" and any(
            item.has_ray_directions for item in payloads
        ):
            raise ValueError("manifest ray semantics contradict payload contents")
        if ray_semantics != "absent" and not all(
            item.has_ray_directions for item in payloads
        ):
            raise ValueError("all payloads must provide the declared ray semantics")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.metadata,
                name="prediction-provider metadata",
            ),
            name="prediction-provider metadata",
        )

        for name, value in normalized_strings.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "provider_revision", revision)
        object.__setattr__(self, "provider_run_id", provider_run_id)
        object.__setattr__(self, "model_set_id", model_set_id)
        object.__setattr__(self, "loader_id", loader_id)
        object.__setattr__(self, "coordinate_semantics", coordinate_semantics)
        object.__setattr__(self, "point_semantics", point_semantics)
        object.__setattr__(self, "flow_semantics", flow_semantics)
        object.__setattr__(self, "ray_semantics", ray_semantics)
        object.__setattr__(self, "source_dependency_semantics", source_semantics)
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "metadata", metadata)

        expected_id = _sha256_json(self.identity_record())
        supplied = self.artifact_id
        if supplied is not None and require_sha256(
            supplied,
            name="artifact_id",
        ) != expected_id:
            raise ValueError("prediction-provider manifest artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": PREDICTION_PROVIDER_MANIFEST_SCHEMA,
            "schema_version": PREDICTION_PROVIDER_MANIFEST_VERSION,
            "sequence_id": self.sequence_id,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "provider_run_id": self.provider_run_id,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "coordinate_semantics": self.coordinate_semantics,
            "point_semantics": self.point_semantics,
            "flow_semantics": self.flow_semantics,
            "ray_semantics": self.ray_semantics,
            "source_dependency_semantics": self.source_dependency_semantics,
            "payloads": [item.identity_record() for item in self.payloads],
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        record = self.identity_record()
        record["artifact_id"] = self.artifact_id
        record["payloads"] = [item.to_record() for item in self.payloads]
        return record

    @classmethod
    def from_record(cls, value: object) -> PredictionProviderManifestV1:
        mapping = require_mapping(value, name="prediction-provider manifest")
        require_exact_fields(mapping, _MANIFEST_FIELDS, name="provider manifest")
        if mapping["schema"] != PREDICTION_PROVIDER_MANIFEST_SCHEMA:
            raise ValueError("unsupported prediction-provider manifest schema")
        if mapping["schema_version"] != PREDICTION_PROVIDER_MANIFEST_VERSION:
            raise ValueError("unsupported prediction-provider manifest version")
        raw_payloads = mapping["payloads"]
        if not isinstance(raw_payloads, list) or not raw_payloads:
            raise ValueError("prediction-provider manifest requires payloads")
        metadata = require_finite_json_mapping(
            mapping["metadata"],
            name="prediction-provider metadata",
        )
        return cls(
            sequence_id=mapping["sequence_id"],
            provider_family=mapping["provider_family"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            provider_run_id=mapping["provider_run_id"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            coordinate_semantics=mapping["coordinate_semantics"],
            point_semantics=mapping["point_semantics"],
            flow_semantics=mapping["flow_semantics"],
            ray_semantics=mapping["ray_semantics"],
            source_dependency_semantics=mapping["source_dependency_semantics"],
            payloads=tuple(
                PredictionPayloadDescriptorV1.from_record(item)
                for item in raw_payloads
            ),
            metadata=metadata,
            artifact_id=mapping["artifact_id"],
        )

    def admitted_payloads(
        self,
        causal_frame_stop: int,
    ) -> tuple[PredictionPayloadDescriptorV1, ...]:
        cutoff = require_exact_integer(
            causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        return tuple(
            item for item in self.payloads if item.is_causally_admitted(cutoff)
        )

    def summary(self, *, causal_frame_stop: int | None = None) -> dict[str, object]:
        summary: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "sequence_id": self.sequence_id,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "provider_run_id": self.provider_run_id,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "coordinate_semantics": self.coordinate_semantics,
            "payload_count": len(self.payloads),
            "output_frame_count": sum(
                len(item.frame_lineage) for item in self.payloads
            ),
            "source_frame_start": min(
                item.source_frame_start for item in self.payloads
            ),
            "source_frame_stop_exclusive": max(
                item.source_frame_stop_exclusive for item in self.payloads
            ),
            "dependence_group_count": len(
                {
                    group
                    for payload in self.payloads
                    for group in payload.dependence_group_ids
                }
            ),
        }
        if causal_frame_stop is not None:
            admitted = self.admitted_payloads(causal_frame_stop)
            summary.update(
                causal_frame_stop=causal_frame_stop,
                admitted_payload_count=len(admitted),
                admitted_payload_ids=[item.payload_id for item in admitted],
            )
        return summary


def save_prediction_provider_manifest(
    path: str | Path,
    manifest: PredictionProviderManifestV1,
) -> Path:
    """Persist one manifest atomically, allowing only an idempotent rewrite."""

    destination = Path(path)
    record = manifest.to_record()
    content = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        existing = load_prediction_provider_manifest(destination)
        if existing.to_record() != record:
            raise ValueError("refusing to replace a different prediction-provider manifest")
        return destination
    _atomic_write_text(destination, content)
    return destination


def load_prediction_provider_manifest(
    path: str | Path,
) -> PredictionProviderManifestV1:
    record = load_json_object(path, name="prediction-provider manifest")
    return PredictionProviderManifestV1.from_record(record)


def verify_prediction_provider_manifest(
    path: str | Path,
    *,
    verify_payloads: bool = True,
    causal_frame_stop: int | None = None,
) -> tuple[PredictionProviderManifestV1, dict[str, object]]:
    """Validate manifest identity and, by default, every canonical payload byte."""

    manifest_path = Path(path).resolve()
    manifest = load_prediction_provider_manifest(manifest_path)
    verified_payloads = 0
    verified_bytes = 0
    if verify_payloads:
        for index, descriptor in enumerate(manifest.payloads):
            member = _resolved_member(
                manifest_path.parent,
                descriptor.path,
                name=f"prediction payload {index} path",
            )
            if not member.is_file():
                raise ValueError(f"prediction payload {descriptor.path!r} is missing")
            stat = member.stat()
            if stat.st_size != descriptor.byte_count:
                raise ValueError(
                    f"prediction payload byte count mismatch for {descriptor.path!r}"
                )
            if _sha256_file(member) != descriptor.sha256:
                raise ValueError(
                    f"prediction payload SHA-256 mismatch for {descriptor.path!r}"
                )
            window = PredictionWindow.from_npz(
                member,
                dense_storage_dtype=descriptor.dense_storage_dtype,
            )
            if window.window_id != descriptor.window_id:
                raise ValueError("prediction payload window identity changed")
            if tuple(int(value) for value in window.frame_indices) != (
                descriptor.output_frame_ids
            ):
                raise ValueError("prediction payload output-frame identities changed")
            if (window.scene_flow is not None) != descriptor.has_scene_flow:
                raise ValueError("prediction payload scene-flow declaration changed")
            if (window.ray_directions is not None) != descriptor.has_ray_directions:
                raise ValueError("prediction payload ray declaration changed")
            if window.dense_storage_dtype != descriptor.dense_storage_dtype:
                raise ValueError("prediction payload dense storage dtype changed")
            verified_payloads += 1
            verified_bytes += stat.st_size
    report = {
        **manifest.summary(causal_frame_stop=causal_frame_stop),
        "payloads_verified": verify_payloads,
        "verified_payload_count": verified_payloads,
        "verified_payload_bytes": verified_bytes,
    }
    return manifest, report


def _motioncrafter_seed_members(
    record: Mapping[str, Any],
) -> tuple[str, int, dict[str, int]]:
    config = require_mapping(record.get("config"), name="MotionCrafter config")
    seed = require_exact_integer(config.get("seed"), name="MotionCrafter seed", minimum=0)
    policy = require_exact_string(
        config.get("seed_policy", "legacy-common"),
        name="MotionCrafter seed policy",
    )
    schedule = record.get("stochastic_seed_schedule")
    if schedule is None:
        return policy, seed, {}
    schedule_mapping = require_mapping(
        schedule,
        name="MotionCrafter stochastic seed schedule",
    )
    calls = schedule_mapping.get("calls")
    if not isinstance(calls, list):
        raise ValueError("MotionCrafter seed schedule calls must be a list")
    result: dict[str, int] = {}
    for raw_call in calls:
        call = require_mapping(raw_call, name="MotionCrafter seed-schedule call")
        if call.get("product") != "independently_decoded_overlap_window":
            continue
        window_id = require_exact_string(
            call.get("window_id"),
            name="MotionCrafter seed-schedule window_id",
        )
        effective_seed = require_exact_integer(
            call.get("effective_seed"),
            name="MotionCrafter effective seed",
            minimum=0,
        )
        if window_id in result:
            raise ValueError("MotionCrafter seed schedule repeats a window ID")
        result[window_id] = effective_seed
    return policy, seed, result


def import_motioncrafter_prediction_manifest(
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    sequence_id: str,
    view_id: str = "camera-0",
) -> PredictionProviderManifestV1:
    """Convert one verified MotionCrafter bundle into the neutral contract."""

    from .motioncrafter_integrity import verify_motioncrafter_prediction_manifest

    source_path = Path(source_manifest_path).resolve()
    verification = verify_motioncrafter_prediction_manifest(
        source_path,
        verify_hashes=True,
    )
    if verification.get("integrity_bound") is not True:
        raise ValueError("provider-neutral import requires an integrity-bound bundle")
    record = load_json_object(source_path, name="MotionCrafter prediction manifest")
    if record.get("format_version") != 1:
        raise ValueError("unsupported MotionCrafter prediction manifest")
    config = require_mapping(record.get("config"), name="MotionCrafter config")
    model_set_id = require_sha256(
        config.get("model_source_set_sha256"),
        name="MotionCrafter model-set ID",
    )
    loader_id = require_sha256(
        config.get("model_loader_module_sha256"),
        name="MotionCrafter model-loader ID",
    )
    provider_revision = require_revision(
        record.get("motioncrafter_commit"),
        name="MotionCrafter revision",
    )
    integrity = require_mapping(
        record.get("artifact_integrity"),
        name="MotionCrafter artifact integrity",
    )
    provider_run_id = require_sha256(
        integrity.get("run_spec_sha256"),
        name="MotionCrafter run-spec ID",
    )
    members = integrity.get("members")
    if not isinstance(members, list):
        raise ValueError("MotionCrafter artifact members must be a list")
    member_records: dict[str, Mapping[str, Any]] = {}
    for raw_member in members:
        member = require_mapping(raw_member, name="MotionCrafter artifact member")
        relative = _safe_relative_path(member.get("path"), name="artifact member path")
        if relative in member_records:
            raise ValueError("MotionCrafter artifact members repeat a path")
        member_records[relative] = member

    seed_policy, root_seed, seed_members = _motioncrafter_seed_members(record)
    windows = record.get("overlap_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("MotionCrafter manifest contains no overlap windows")
    payloads: list[PredictionPayloadDescriptorV1] = []
    source_root = source_path.parent
    for index, raw_window in enumerate(windows):
        window_record = require_mapping(
            raw_window,
            name=f"MotionCrafter overlap window {index}",
        )
        window_id = require_exact_string(
            window_record.get("window_id"),
            name="MotionCrafter window_id",
        )
        relative = _safe_relative_path(
            window_record.get("path"),
            name=f"MotionCrafter window {window_id!r} path",
        )
        member = member_records.get(relative)
        if member is None:
            raise ValueError("MotionCrafter window lacks an integrity descriptor")
        sha256 = require_sha256(member.get("sha256"), name="window payload sha256")
        byte_count = require_exact_integer(
            member.get("bytes"),
            name="window payload byte count",
            minimum=1,
        )
        start = require_exact_integer(
            window_record.get("start_frame"),
            name="MotionCrafter source_frame_start",
            minimum=0,
        )
        stop = require_exact_integer(
            window_record.get("stop_frame"),
            name="MotionCrafter source_frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ValueError("MotionCrafter source frame interval is invalid")
        payload_path = _resolved_member(
            source_root,
            relative,
            name=f"MotionCrafter window {window_id!r} path",
        )
        prediction = PredictionWindow.from_npz(payload_path)
        if prediction.window_id != window_id:
            raise ValueError("MotionCrafter manifest and payload window IDs differ")
        effective_seed = seed_members.get(window_id, root_seed)
        stochastic_id_record = {
            "schema": "prob4d.motioncrafter-stochastic-member.v1",
            "policy": seed_policy,
            "root_seed": root_seed,
            "effective_seed": effective_seed,
            "window_id": window_id,
        }
        stochastic_member_id = (
            "prob4d.motioncrafter-stochastic-member.v1:"
            + _sha256_json(stochastic_id_record)
        )
        lineage = tuple(
            PredictionFrameLineageV1(
                output_frame_id=int(frame_id),
                source_frame_start=start,
                source_frame_stop_exclusive=stop,
                contributor_ids=(window_id,),
            )
            for frame_id in prediction.frame_indices
        )
        input_video = require_mapping(
            require_mapping(
                integrity.get("run_spec"),
                name="MotionCrafter run spec",
            ).get("input_video"),
            name="MotionCrafter input-video descriptor",
        )
        input_video_sha = require_sha256(
            input_video.get("sha256"),
            name="MotionCrafter input-video SHA-256",
        )
        payloads.append(
            PredictionPayloadDescriptorV1(
                product_role="independent-window",
                window_id=window_id,
                path=_relative_member(
                    payload_path,
                    root=Path(output_path).parent,
                    name=f"MotionCrafter window {window_id!r} payload path",
                ),
                sha256=sha256,
                byte_count=byte_count,
                view_id=view_id,
                stochastic_member_id=stochastic_member_id,
                dependence_group_ids=(
                    f"model-set:{model_set_id}",
                    f"input-video:{input_video_sha}",
                    f"stochastic-member:{effective_seed}",
                ),
                dense_storage_dtype=prediction.dense_storage_dtype,
                has_scene_flow=prediction.scene_flow is not None,
                has_ray_directions=prediction.ray_directions is not None,
                frame_lineage=lineage,
            )
        )
    has_flow = {item.has_scene_flow for item in payloads}
    has_rays = {item.has_ray_directions for item in payloads}
    if len(has_flow) != 1 or len(has_rays) != 1:
        raise ValueError("MotionCrafter windows have inconsistent optional fields")
    source_manifest_sha = _sha256_file(source_path)
    manifest = PredictionProviderManifestV1(
        sequence_id=sequence_id,
        provider_family="MotionCrafter",
        provider_repository="TencentARC/MotionCrafter",
        provider_revision=provider_revision,
        provider_run_id=provider_run_id,
        model_set_id=model_set_id,
        loader_id=loader_id,
        coordinate_semantics="window-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics=(
            "forward-point-displacement" if True in has_flow else "absent"
        ),
        ray_semantics=("camera-ray-unit-vector" if True in has_rays else "absent"),
        payloads=tuple(payloads),
        metadata={
            "source_adapter": "prob4d-motioncrafter-provider-neutral-v1",
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_format_version": 1,
            "stochastic_seed_policy": seed_policy,
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    )
    save_prediction_provider_manifest(output_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction",
        description="Import and validate provider-neutral prediction manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-motioncrafter",
        help="convert an integrity-bound MotionCrafter bundle",
    )
    import_parser.add_argument("source_manifest")
    import_parser.add_argument("output")
    import_parser.add_argument("--sequence-id", required=True)
    import_parser.add_argument("--view-id", default="camera-0")

    validate_parser = subparsers.add_parser(
        "validate",
        help="strictly validate a neutral manifest and its payloads",
    )
    validate_parser.add_argument("manifest")
    validate_parser.add_argument("--metadata-only", action="store_true")
    validate_parser.add_argument("--causal-frame-stop", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "import-motioncrafter":
        manifest = import_motioncrafter_prediction_manifest(
            arguments.source_manifest,
            arguments.output,
            sequence_id=arguments.sequence_id,
            view_id=arguments.view_id,
        )
        print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "validate":
        _, report = verify_prediction_provider_manifest(
            arguments.manifest,
            verify_payloads=not arguments.metadata_only,
            causal_frame_stop=arguments.causal_frame_stop,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    parser.error("unsupported prediction command")
    return 2


__all__ = [
    "COORDINATE_SEMANTICS",
    "FLOW_SEMANTICS",
    "POINT_SEMANTICS",
    "PREDICTION_PROVIDER_MANIFEST_SCHEMA",
    "PREDICTION_PROVIDER_MANIFEST_VERSION",
    "PRODUCT_ROLES",
    "PredictionFrameLineageV1",
    "PredictionPayloadDescriptorV1",
    "PredictionProviderManifestV1",
    "RAY_SEMANTICS",
    "SOURCE_DEPENDENCY_SEMANTICS",
    "import_motioncrafter_prediction_manifest",
    "load_prediction_provider_manifest",
    "main",
    "save_prediction_provider_manifest",
    "verify_prediction_provider_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
