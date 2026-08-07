"""Bind target provider manifests to one frozen held-out promotion protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from ._heldout_promotion_common import (
    _SHA256,
    _atomic_write_json,
    _exact_keys,
    _load_json,
    _repository,
    _revision,
    _strict_bool,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    load_promotion_lock,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import _sha256_json, _strict_integer
from .deform360_cohort_binding import (
    Deform360OfficialHubCohortBindingV1,
    load_deform360_cohort_binding,
)
from .prediction_provider_manifest import (
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    load_prediction_provider_manifest,
)

TARGET_PROVIDER_ADMISSION_CONFIG_SCHEMA = "prob4d.heldout-target-provider-admission-config"
TARGET_PROVIDER_ADMISSION_CONFIG_VERSION = 1
TARGET_PROVIDER_ADMISSION_SCHEMA = "prob4d.heldout-target-provider-admission"
TARGET_PROVIDER_ADMISSION_VERSION = 1
TARGET_PROVIDER_ADMISSION_CLAIM_BOUNDARY = (
    "This artifact admits exact provider-manifest metadata for every frozen target "
    "object before target outcomes are evaluated. It binds source/model/loader "
    "semantics, causal cutoffs, manifest bytes, and admitted payload identities. "
    "It does not open prediction payloads or target outcomes and does not establish "
    "provider competence, calibration, BayesianPhysTwin benefit, Causal4D benefit, "
    "deployment safety, or state of the art."
)

Stratum = Literal["sheet", "volumetric"]

_REQUEST_FIELDS = {
    "group_id",
    "expected_sequence_id",
    "manifest_path",
    "causal_frame_stop",
}
_CONFIG_FIELDS = {
    "schema_name",
    "schema_version",
    "prediction_run_spec_id",
    "target_outcomes_used",
    "entries",
    "metadata",
}
_PAYLOAD_FIELDS = {
    "payload_id",
    "window_id",
    "output_frame_ids",
    "source_frame_start",
    "source_frame_stop_exclusive",
    "dependence_group_ids",
}
_ENTRY_FIELDS = {
    "group_id",
    "episode_id",
    "stratum",
    "sequence_id",
    "manifest_sha256",
    "manifest_artifact_id",
    "provider_run_id",
    "causal_frame_stop",
    "admitted_payloads",
}
_ADMISSION_FIELDS = {
    "schema_name",
    "schema_version",
    "promotion_lock_id",
    "cohort_binding_id",
    "source_repository",
    "source_revision",
    "prediction_run_spec_id",
    "provider_family",
    "provider_repository",
    "provider_revision",
    "model_set_id",
    "loader_id",
    "coordinate_semantics",
    "point_semantics",
    "flow_semantics",
    "ray_semantics",
    "source_dependency_semantics",
    "target_outcomes_used",
    "entries",
    "metadata",
    "claim_boundary",
    "target_provider_admission_id",
}


def _stratum(value: object, *, name: str) -> Stratum:
    result = _strict_string(value, name=name)
    if result not in {"sheet", "volumetric"}:
        raise ValueError(f"{name} must be 'sheet' or 'volumetric'")
    return cast(Stratum, result)


def _safe_relative_path(value: object, *, name: str) -> str:
    result = _strict_string(value, name=name)
    if "\\" in result:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    path = PurePosixPath(result)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return path.as_posix()


def _resolve_member(root: Path, relative_path: str, *, name: str) -> Path:
    root_resolved = root.resolve()
    current = root_resolved
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic link")
    candidate = current.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the request directory") from error
    if not candidate.is_file():
        raise ValueError(f"{name} is not a regular file")
    return candidate


def _manifest_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read provider manifest {path}") from error


def _canonical_strings(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be a list of strings")
    result = tuple(
        _strict_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(cast(Sequence[object], value))
    )
    if not result or result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty, sorted, and unique")
    return result


def _canonical_integers(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be a list of integers")
    result = tuple(
        _strict_integer(item, name=f"{name}[{index}]", minimum=0)
        for index, item in enumerate(cast(Sequence[object], value))
    )
    if not result or result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty, sorted, and unique")
    return result


@dataclass(frozen=True, slots=True)
class TargetProviderManifestRequestV1:
    """Retrieval metadata for one target provider manifest."""

    group_id: str
    expected_sequence_id: str
    manifest_path: str
    causal_frame_stop: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _strict_string(self.group_id, name="group_id"))
        object.__setattr__(
            self,
            "expected_sequence_id",
            _strict_string(self.expected_sequence_id, name="expected_sequence_id"),
        )
        object.__setattr__(
            self,
            "manifest_path",
            _safe_relative_path(self.manifest_path, name="manifest_path"),
        )
        object.__setattr__(
            self,
            "causal_frame_stop",
            _strict_integer(self.causal_frame_stop, name="causal_frame_stop", minimum=1),
        )

    @classmethod
    def from_dict(cls, value: object) -> TargetProviderManifestRequestV1:
        mapping = _strict_mapping(value, name="target provider request")
        _exact_keys(mapping, _REQUEST_FIELDS, name="target provider request")
        return cls(
            group_id=mapping["group_id"],
            expected_sequence_id=mapping["expected_sequence_id"],
            manifest_path=mapping["manifest_path"],
            causal_frame_stop=mapping["causal_frame_stop"],
        )


@dataclass(frozen=True, slots=True)
class AdmittedTargetPayloadV1:
    """One causally admitted payload identity without opening its dense bytes."""

    payload_id: str
    window_id: str
    output_frame_ids: tuple[int, ...]
    source_frame_start: int
    source_frame_stop_exclusive: int
    dependence_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload_id",
            _strict_digest(self.payload_id, name="payload_id", pattern=_SHA256),
        )
        object.__setattr__(
            self,
            "window_id",
            _strict_string(self.window_id, name="window_id"),
        )
        frames = _canonical_integers(self.output_frame_ids, name="output_frame_ids")
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
            raise ValueError("source_frame_stop_exclusive must exceed source_frame_start")
        groups = _canonical_strings(
            self.dependence_group_ids,
            name="dependence_group_ids",
        )
        object.__setattr__(self, "output_frame_ids", frames)
        object.__setattr__(self, "source_frame_start", start)
        object.__setattr__(self, "source_frame_stop_exclusive", stop)
        object.__setattr__(self, "dependence_group_ids", groups)

    @classmethod
    def from_descriptor(cls, value: PredictionPayloadDescriptorV1) -> AdmittedTargetPayloadV1:
        if value.payload_id is None:
            raise ValueError("provider payload ID is not materialized")
        return cls(
            payload_id=value.payload_id,
            window_id=value.window_id,
            output_frame_ids=tuple(sorted(value.output_frame_ids)),
            source_frame_start=value.source_frame_start,
            source_frame_stop_exclusive=value.source_frame_stop_exclusive,
            dependence_group_ids=tuple(sorted(value.dependence_group_ids)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_id": self.payload_id,
            "window_id": self.window_id,
            "output_frame_ids": list(self.output_frame_ids),
            "source_frame_start": self.source_frame_start,
            "source_frame_stop_exclusive": self.source_frame_stop_exclusive,
            "dependence_group_ids": list(self.dependence_group_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> AdmittedTargetPayloadV1:
        mapping = _strict_mapping(value, name="admitted target payload")
        _exact_keys(mapping, _PAYLOAD_FIELDS, name="admitted target payload")
        return cls(
            payload_id=mapping["payload_id"],
            window_id=mapping["window_id"],
            output_frame_ids=_canonical_integers(
                mapping["output_frame_ids"],
                name="output_frame_ids",
            ),
            source_frame_start=mapping["source_frame_start"],
            source_frame_stop_exclusive=mapping["source_frame_stop_exclusive"],
            dependence_group_ids=_canonical_strings(
                mapping["dependence_group_ids"],
                name="dependence_group_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetProviderManifestAdmissionV1:
    """One target object's exact provider manifest and causal admission."""

    group_id: str
    episode_id: int
    stratum: Stratum
    sequence_id: str
    manifest_sha256: str
    manifest_artifact_id: str
    provider_run_id: str
    causal_frame_stop: int
    admitted_payloads: tuple[AdmittedTargetPayloadV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _strict_string(self.group_id, name="group_id"))
        object.__setattr__(
            self,
            "episode_id",
            _strict_integer(self.episode_id, name="episode_id", minimum=0),
        )
        object.__setattr__(self, "stratum", _stratum(self.stratum, name="stratum"))
        object.__setattr__(
            self,
            "sequence_id",
            _strict_string(self.sequence_id, name="sequence_id"),
        )
        for field_name in ("manifest_sha256", "manifest_artifact_id", "provider_run_id"):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        cutoff = _strict_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if type(self.admitted_payloads) is not tuple or not self.admitted_payloads:
            raise ValueError("admitted_payloads must be a nonempty tuple")
        payloads = tuple(sorted(self.admitted_payloads, key=lambda item: item.payload_id))
        if not all(isinstance(item, AdmittedTargetPayloadV1) for item in payloads):
            raise ValueError("admitted_payloads must contain AdmittedTargetPayloadV1 values")
        payload_ids = tuple(item.payload_id for item in payloads)
        if len(payload_ids) != len(set(payload_ids)):
            raise ValueError("admitted payload IDs must be unique")
        if any(item.source_frame_stop_exclusive > cutoff for item in payloads):
            raise ValueError("admitted payload source lineage crosses the causal cutoff")
        object.__setattr__(self, "causal_frame_stop", cutoff)
        object.__setattr__(self, "admitted_payloads", payloads)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "sequence_id": self.sequence_id,
            "manifest_sha256": self.manifest_sha256,
            "manifest_artifact_id": self.manifest_artifact_id,
            "provider_run_id": self.provider_run_id,
            "causal_frame_stop": self.causal_frame_stop,
            "admitted_payloads": [item.to_dict() for item in self.admitted_payloads],
        }

    @classmethod
    def from_dict(cls, value: object) -> TargetProviderManifestAdmissionV1:
        mapping = _strict_mapping(value, name="target provider manifest admission")
        _exact_keys(mapping, _ENTRY_FIELDS, name="target provider manifest admission")
        raw_payloads = _strict_list(mapping["admitted_payloads"], name="admitted_payloads")
        return cls(
            group_id=mapping["group_id"],
            episode_id=mapping["episode_id"],
            stratum=mapping["stratum"],
            sequence_id=mapping["sequence_id"],
            manifest_sha256=mapping["manifest_sha256"],
            manifest_artifact_id=mapping["manifest_artifact_id"],
            provider_run_id=mapping["provider_run_id"],
            causal_frame_stop=mapping["causal_frame_stop"],
            admitted_payloads=tuple(
                AdmittedTargetPayloadV1.from_dict(item) for item in raw_payloads
            ),
        )


@dataclass(frozen=True, slots=True)
class HeldoutTargetProviderAdmissionV1:
    """Complete target-manifest admission under one promotion lock."""

    promotion_lock_id: str
    cohort_binding_id: str
    source_repository: str
    source_revision: str
    prediction_run_spec_id: str
    provider_family: str
    provider_repository: str
    provider_revision: str
    model_set_id: str
    loader_id: str
    coordinate_semantics: str
    point_semantics: str
    flow_semantics: str
    ray_semantics: str
    source_dependency_semantics: str
    target_outcomes_used: bool
    entries: tuple[TargetProviderManifestAdmissionV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "promotion_lock_id",
            "cohort_binding_id",
            "prediction_run_spec_id",
            "model_set_id",
            "loader_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        object.__setattr__(
            self,
            "source_repository",
            _repository(self.source_repository, name="source_repository"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _revision(self.source_revision, name="source_revision"),
        )
        object.__setattr__(
            self,
            "provider_family",
            _strict_string(self.provider_family, name="provider_family"),
        )
        object.__setattr__(
            self,
            "provider_repository",
            _repository(self.provider_repository, name="provider_repository"),
        )
        object.__setattr__(
            self,
            "provider_revision",
            _revision(self.provider_revision, name="provider_revision"),
        )
        for field_name in (
            "coordinate_semantics",
            "point_semantics",
            "flow_semantics",
            "ray_semantics",
            "source_dependency_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_string(getattr(self, field_name), name=field_name),
            )
        target_used = _strict_bool(self.target_outcomes_used, name="target_outcomes_used")
        if target_used:
            raise ValueError("target provider admission must not use target outcomes")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("entries must be a nonempty tuple")
        entries = tuple(sorted(self.entries, key=lambda item: item.group_id))
        if not all(isinstance(item, TargetProviderManifestAdmissionV1) for item in entries):
            raise ValueError("entries must contain TargetProviderManifestAdmissionV1 values")
        group_ids = tuple(item.group_id for item in entries)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("target provider admission group IDs must be unique")
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="target admission metadata"),
        )

    @property
    def target_group_ids(self) -> tuple[str, ...]:
        return tuple(item.group_id for item in self.entries)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": TARGET_PROVIDER_ADMISSION_SCHEMA,
            "schema_version": TARGET_PROVIDER_ADMISSION_VERSION,
            "promotion_lock_id": self.promotion_lock_id,
            "cohort_binding_id": self.cohort_binding_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "prediction_run_spec_id": self.prediction_run_spec_id,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "coordinate_semantics": self.coordinate_semantics,
            "point_semantics": self.point_semantics,
            "flow_semantics": self.flow_semantics,
            "ray_semantics": self.ray_semantics,
            "source_dependency_semantics": self.source_dependency_semantics,
            "target_outcomes_used": self.target_outcomes_used,
            "entries": [item.to_dict() for item in self.entries],
            "metadata": plain_json(self.metadata),
            "claim_boundary": TARGET_PROVIDER_ADMISSION_CLAIM_BOUNDARY,
        }

    @property
    def target_provider_admission_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "target_provider_admission_id": self.target_provider_admission_id,
        }


def target_provider_admission_from_dict(
    value: object,
) -> HeldoutTargetProviderAdmissionV1:
    mapping = _strict_mapping(value, name="target provider admission")
    _exact_keys(mapping, _ADMISSION_FIELDS, name="target provider admission")
    if mapping["schema_name"] != TARGET_PROVIDER_ADMISSION_SCHEMA:
        raise ValueError("unsupported target provider admission schema")
    version = _strict_integer(mapping["schema_version"], name="schema_version", minimum=1)
    if version != TARGET_PROVIDER_ADMISSION_VERSION:
        raise ValueError("unsupported target provider admission version")
    if mapping["claim_boundary"] != TARGET_PROVIDER_ADMISSION_CLAIM_BOUNDARY:
        raise ValueError("target provider admission claim boundary changed")
    raw_entries = _strict_list(mapping["entries"], name="entries")
    result = HeldoutTargetProviderAdmissionV1(
        promotion_lock_id=mapping["promotion_lock_id"],
        cohort_binding_id=mapping["cohort_binding_id"],
        source_repository=mapping["source_repository"],
        source_revision=mapping["source_revision"],
        prediction_run_spec_id=mapping["prediction_run_spec_id"],
        provider_family=mapping["provider_family"],
        provider_repository=mapping["provider_repository"],
        provider_revision=mapping["provider_revision"],
        model_set_id=mapping["model_set_id"],
        loader_id=mapping["loader_id"],
        coordinate_semantics=mapping["coordinate_semantics"],
        point_semantics=mapping["point_semantics"],
        flow_semantics=mapping["flow_semantics"],
        ray_semantics=mapping["ray_semantics"],
        source_dependency_semantics=mapping["source_dependency_semantics"],
        target_outcomes_used=mapping["target_outcomes_used"],
        entries=tuple(TargetProviderManifestAdmissionV1.from_dict(item) for item in raw_entries),
        metadata=_strict_mapping(mapping["metadata"], name="target admission metadata"),
    )
    supplied = _strict_digest(
        mapping["target_provider_admission_id"],
        name="target_provider_admission_id",
        pattern=_SHA256,
    )
    if supplied != result.target_provider_admission_id:
        raise ValueError("target provider admission ID mismatch")
    return result


def load_target_provider_admission(path: str | Path) -> HeldoutTargetProviderAdmissionV1:
    mapping, _ = _load_json(Path(path), name="target provider admission")
    return target_provider_admission_from_dict(mapping)


def write_target_provider_admission(
    value: HeldoutTargetProviderAdmissionV1,
    path: str | Path,
) -> Path:
    destination = Path(path)
    if destination.exists():
        existing = load_target_provider_admission(destination)
        if existing.to_dict() != value.to_dict():
            raise FileExistsError(destination)
        return destination
    _atomic_write_json(destination, value.to_dict())
    return destination


def _validate_lock_and_binding(
    lock: HeldoutProviderPromotionLockV1,
    binding: Deform360OfficialHubCohortBindingV1,
) -> None:
    if lock.calibration_group_ids != binding.calibration_group_ids:
        raise ValueError("cohort binding calibration groups differ from promotion lock")
    if lock.target_group_ids != binding.target_group_ids:
        raise ValueError("cohort binding target groups differ from promotion lock")
    if lock.bayesian_phystwin_repository != binding.source_repository:
        raise ValueError("cohort binding source repository differs from promotion lock")
    if lock.bayesian_phystwin_revision != binding.source_revision:
        raise ValueError("cohort binding source revision differs from promotion lock")
    if lock.frozen_artifact_ids.get("cohort_binding") != binding.cohort_binding_id:
        raise ValueError("promotion lock does not bind the supplied cohort artifact")


def _provider_contract(manifest: PredictionProviderManifestV1) -> tuple[str, ...]:
    return (
        manifest.provider_family,
        manifest.provider_repository,
        manifest.provider_revision,
        manifest.model_set_id,
        manifest.loader_id,
        manifest.coordinate_semantics,
        manifest.point_semantics,
        manifest.flow_semantics,
        manifest.ray_semantics,
        manifest.source_dependency_semantics,
    )


def _entry_from_manifest(
    request: TargetProviderManifestRequestV1,
    *,
    unit_episode_id: int,
    unit_stratum: Stratum,
    manifest_path: Path,
    manifest: PredictionProviderManifestV1,
) -> TargetProviderManifestAdmissionV1:
    if manifest.sequence_id != request.expected_sequence_id:
        raise ValueError(f"provider sequence changed for target group {request.group_id!r}")
    admitted = manifest.admitted_payloads(request.causal_frame_stop)
    if not admitted:
        raise ValueError(
            f"no provider payload is causally admitted for target group {request.group_id!r}"
        )
    if manifest.artifact_id is None:
        raise ValueError("provider manifest artifact ID is not materialized")
    return TargetProviderManifestAdmissionV1(
        group_id=request.group_id,
        episode_id=unit_episode_id,
        stratum=unit_stratum,
        sequence_id=manifest.sequence_id,
        manifest_sha256=hashlib.sha256(_manifest_bytes(manifest_path)).hexdigest(),
        manifest_artifact_id=manifest.artifact_id,
        provider_run_id=manifest.provider_run_id,
        causal_frame_stop=request.causal_frame_stop,
        admitted_payloads=tuple(AdmittedTargetPayloadV1.from_descriptor(item) for item in admitted),
    )


def build_target_provider_admission(
    lock: HeldoutProviderPromotionLockV1,
    binding: Deform360OfficialHubCohortBindingV1,
    config: object,
    *,
    request_root: str | Path,
) -> HeldoutTargetProviderAdmissionV1:
    """Build an outcome-blind admission from exact target manifest metadata."""

    _validate_lock_and_binding(lock, binding)
    mapping = _strict_mapping(config, name="target provider admission configuration")
    _exact_keys(mapping, _CONFIG_FIELDS, name="target provider admission configuration")
    if mapping["schema_name"] != TARGET_PROVIDER_ADMISSION_CONFIG_SCHEMA:
        raise ValueError("unsupported target provider admission configuration schema")
    version = _strict_integer(mapping["schema_version"], name="schema_version", minimum=1)
    if version != TARGET_PROVIDER_ADMISSION_CONFIG_VERSION:
        raise ValueError("unsupported target provider admission configuration version")
    run_spec_id = _strict_digest(
        mapping["prediction_run_spec_id"],
        name="prediction_run_spec_id",
        pattern=_SHA256,
    )
    if run_spec_id != lock.prediction_run_spec_id:
        raise ValueError("target request prediction run-spec differs from promotion lock")
    target_used = _strict_bool(mapping["target_outcomes_used"], name="target_outcomes_used")
    if target_used:
        raise ValueError("target provider admission cannot use target outcomes")
    raw_requests = _strict_list(mapping["entries"], name="entries")
    requests = tuple(
        sorted(
            (TargetProviderManifestRequestV1.from_dict(item) for item in raw_requests),
            key=lambda item: item.group_id,
        )
    )
    request_group_ids = tuple(item.group_id for item in requests)
    if request_group_ids != lock.target_group_ids:
        raise ValueError("target provider requests must cover the exact frozen target groups")
    if len(request_group_ids) != len(set(request_group_ids)):
        raise ValueError("target provider request group IDs must be unique")

    units = {unit.object_id: unit for unit in binding.target_units}
    root = Path(request_root)
    entries: list[TargetProviderManifestAdmissionV1] = []
    expected_contract: tuple[str, ...] | None = None
    first_manifest: PredictionProviderManifestV1 | None = None
    for request in requests:
        unit = units[request.group_id]
        path = _resolve_member(root, request.manifest_path, name="provider manifest path")
        manifest = load_prediction_provider_manifest(path)
        if manifest.provider_revision != lock.motioncrafter_revision:
            raise ValueError("target provider revision differs from promotion lock")
        if manifest.model_set_id != lock.model_set_id:
            raise ValueError("target provider model set differs from promotion lock")
        contract = _provider_contract(manifest)
        if expected_contract is None:
            expected_contract = contract
            first_manifest = manifest
        elif contract != expected_contract:
            raise ValueError("target provider contract drifts across frozen target groups")
        entries.append(
            _entry_from_manifest(
                request,
                unit_episode_id=unit.episode_id,
                unit_stratum=unit.stratum,
                manifest_path=path,
                manifest=manifest,
            )
        )
    if first_manifest is None:
        raise ValueError("target provider admission requires at least one manifest")
    metadata = _strict_mapping(mapping["metadata"], name="target admission metadata")
    return HeldoutTargetProviderAdmissionV1(
        promotion_lock_id=lock.promotion_lock_id,
        cohort_binding_id=binding.cohort_binding_id,
        source_repository=lock.source_repository,
        source_revision=lock.source_revision,
        prediction_run_spec_id=run_spec_id,
        provider_family=first_manifest.provider_family,
        provider_repository=first_manifest.provider_repository,
        provider_revision=first_manifest.provider_revision,
        model_set_id=first_manifest.model_set_id,
        loader_id=first_manifest.loader_id,
        coordinate_semantics=first_manifest.coordinate_semantics,
        point_semantics=first_manifest.point_semantics,
        flow_semantics=first_manifest.flow_semantics,
        ray_semantics=first_manifest.ray_semantics,
        source_dependency_semantics=first_manifest.source_dependency_semantics,
        target_outcomes_used=target_used,
        entries=tuple(entries),
        metadata=metadata,
    )


def validate_target_provider_admission_against_lock(
    admission: HeldoutTargetProviderAdmissionV1,
    lock: HeldoutProviderPromotionLockV1,
) -> None:
    """Require exact promotion-lock and target-group agreement."""

    if admission.promotion_lock_id != lock.promotion_lock_id:
        raise ValueError("target provider admission uses another promotion lock")
    if admission.source_repository != lock.source_repository:
        raise ValueError("target provider admission source repository changed")
    if admission.source_revision != lock.source_revision:
        raise ValueError("target provider admission source revision changed")
    if admission.prediction_run_spec_id != lock.prediction_run_spec_id:
        raise ValueError("target provider admission prediction run-spec changed")
    if admission.provider_revision != lock.motioncrafter_revision:
        raise ValueError("target provider admission provider revision changed")
    if admission.model_set_id != lock.model_set_id:
        raise ValueError("target provider admission model set changed")
    if admission.target_group_ids != lock.target_group_ids:
        raise ValueError("target provider admission target groups changed")
    if lock.frozen_artifact_ids.get("cohort_binding") != admission.cohort_binding_id:
        raise ValueError("target provider admission cohort binding changed")
    if admission.target_outcomes_used:
        raise ValueError("target provider admission used target outcomes")


def verify_target_provider_admission(
    observed: HeldoutTargetProviderAdmissionV1,
    lock: HeldoutProviderPromotionLockV1,
    binding: Deform360OfficialHubCohortBindingV1,
    config: object,
    *,
    request_root: str | Path,
) -> HeldoutTargetProviderAdmissionV1:
    replayed = build_target_provider_admission(
        lock,
        binding,
        config,
        request_root=request_root,
    )
    if observed.to_dict() != replayed.to_dict():
        raise ValueError("target provider admission does not match deterministic replay")
    return replayed


def admit_cli(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider target-admit",
        description="Bind exact target provider manifests before outcome evaluation.",
    )
    parser.add_argument("lock", type=Path)
    parser.add_argument("cohort_binding", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    lock = load_promotion_lock(parsed.lock)
    binding = load_deform360_cohort_binding(parsed.cohort_binding)
    config, _ = _load_json(parsed.config, name="target provider admission configuration")
    admission = build_target_provider_admission(
        lock,
        binding,
        config,
        request_root=parsed.config.parent,
    )
    write_target_provider_admission(admission, parsed.output)
    print(admission.target_provider_admission_id)
    return 0


def verify_cli(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider target-verify",
        description="Replay one target provider admission without opening payloads.",
    )
    parser.add_argument("admission", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("cohort_binding", type=Path)
    parser.add_argument("config", type=Path)
    parsed = parser.parse_args(arguments)
    observed = load_target_provider_admission(parsed.admission)
    lock = load_promotion_lock(parsed.lock)
    binding = load_deform360_cohort_binding(parsed.cohort_binding)
    config, _ = _load_json(parsed.config, name="target provider admission configuration")
    replayed = verify_target_provider_admission(
        observed,
        lock,
        binding,
        config,
        request_root=parsed.config.parent,
    )
    print(
        json.dumps(
            {
                "target_provider_admission_id": replayed.target_provider_admission_id,
                "promotion_lock_id": replayed.promotion_lock_id,
                "cohort_binding_id": replayed.cohort_binding_id,
                "target_group_count": len(replayed.entries),
                "target_outcomes_used": replayed.target_outcomes_used,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "TARGET_PROVIDER_ADMISSION_CLAIM_BOUNDARY",
    "TARGET_PROVIDER_ADMISSION_CONFIG_SCHEMA",
    "TARGET_PROVIDER_ADMISSION_CONFIG_VERSION",
    "TARGET_PROVIDER_ADMISSION_SCHEMA",
    "TARGET_PROVIDER_ADMISSION_VERSION",
    "AdmittedTargetPayloadV1",
    "HeldoutTargetProviderAdmissionV1",
    "TargetProviderManifestAdmissionV1",
    "TargetProviderManifestRequestV1",
    "admit_cli",
    "build_target_provider_admission",
    "load_target_provider_admission",
    "target_provider_admission_from_dict",
    "validate_target_provider_admission_against_lock",
    "verify_cli",
    "verify_target_provider_admission",
    "write_target_provider_admission",
]
