"""Bind Prob4D promotion to BayesianPhysTwin's official-Hub Stage-0 cohort.

The BayesianPhysTwin repository owns the target-blind Deform360 object and
episode selection. Prob4D must consume that exact committed artifact rather than
independently rediscovering or relabelling a cohort. This module validates all
nested selection identities, preserves the names/metadata-only information
boundary, and emits a compact content-addressed binding for held-out promotion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from ._heldout_promotion_common import (
    _atomic_write_json,
    _load_json,
    _repository,
    _revision,
)
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_string,
)

DEFORM360_SELECTION_SCHEMA = "bayesian-phystwin/deform360-official-hub-selection-v1"
DEFORM360_SELECTION_VERSION = 1
DEFORM360_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
DEFORM360_DATASET_REPOSITORY = "brownu/deform360"
DEFORM360_PROCESSING_REPOSITORY = "lhy0807/deform360"
BAYESIAN_PHYSTWIN_REPOSITORY = "IPS-Stuttgart/BayesianPhysTwin"
DEFORM360_SELECTION_PATH = (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)

DEFORM360_COHORT_BINDING_SCHEMA = "prob4d.deform360-official-hub-cohort-binding"
DEFORM360_COHORT_BINDING_VERSION = 1
DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY = (
    "This target-free artifact binds Prob4D promotion to BayesianPhysTwin's exact "
    "official-Hub Stage-0 Deform360 calibration and confirmation selection. It "
    "authenticates cohort custody and the names/metadata-only information boundary; "
    "it is not provider-competence, physical-benefit, uncertainty-calibration, safety, "
    "Causal4D, or state-of-the-art evidence."
)

Stratum = Literal["sheet", "volumetric"]

_SELECTION_FIELDS = {
    "available_raw_object_count",
    "cache_preflight",
    "content_selection_sha256",
    "dataset",
    "excluded_object_count",
    "implementation_revision",
    "information_boundary",
    "next_gate",
    "official_processing",
    "prior_protocols",
    "protocol_id",
    "protocol_sha256",
    "replacement_allowed_after_payload_access",
    "schema",
    "schema_version",
    "selection",
    "selection_artifact_sha256",
    "selection_sha256",
}
_DATASET_FIELDS = {"repo_id", "requested_revision", "resolved_revision", "raw_prefix"}
_PROCESSING_FIELDS = {
    "future_processing_revision_change_requires_new_protocol",
    "repository",
    "required_stages",
    "revision",
}
_INFORMATION_BOUNDARY_FIELDS = {
    "object_directory_names_opened",
    "object_metadata_json_opened",
    "opened_metadata_paths",
    "camera_media_opened",
    "tactile_arrays_opened",
    "robot_arrays_opened",
    "geometry_annotations_opened",
    "target_outcomes_opened",
}
_INFORMATION_BOUNDARY = {
    "object_directory_names_opened": True,
    "object_metadata_json_opened": True,
    "camera_media_opened": False,
    "tactile_arrays_opened": False,
    "robot_arrays_opened": False,
    "geometry_annotations_opened": False,
    "target_outcomes_opened": False,
}
_SELECTION_SPLIT_FIELDS = {"calibration", "confirmation"}
_UNIT_FIELDS = {"object_id", "stratum", "episode_id", "metadata_path", "metadata_sha256"}
_BINDING_FIELDS = {
    "schema_name",
    "schema_version",
    "source_repository",
    "source_revision",
    "source_path",
    "selection_schema",
    "selection_schema_version",
    "selection_artifact_sha256",
    "content_selection_sha256",
    "selection_sha256",
    "selection_implementation_revision",
    "protocol_id",
    "protocol_sha256",
    "dataset_repository",
    "dataset_requested_revision",
    "dataset_resolved_revision",
    "processing_repository",
    "processing_revision",
    "statistical_unit",
    "calibration_units",
    "target_units",
    "calibration_group_ids",
    "target_group_ids",
    "information_boundary",
    "replacement_allowed_after_payload_access",
    "claim_boundary",
    "cohort_binding_id",
}
_BINDING_INFORMATION_BOUNDARY = {
    "object_directory_names_opened": True,
    "object_metadata_json_opened": True,
    "camera_media_opened": False,
    "tactile_arrays_opened": False,
    "robot_arrays_opened": False,
    "geometry_annotations_opened": False,
    "target_outcomes_opened": False,
}


def _strict_string_list(value: Any, *, name: str) -> list[str]:
    items = _strict_list(value, name=name)
    return [
        _strict_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(items)
    ]


def _canonical_source_path(value: Any) -> str:
    result = _strict_string(value, name="source_path")
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source_path must be a repository-relative canonical path")
    if result != DEFORM360_SELECTION_PATH:
        raise ValueError(
            f"source_path must identify the authoritative selection at "
            f"{DEFORM360_SELECTION_PATH!r}"
        )
    return result


def _stratum(value: Any, *, name: str) -> Stratum:
    result = _strict_string(value, name=name)
    if result not in {"sheet", "volumetric"}:
        raise ValueError(f"{name} must be 'sheet' or 'volumetric'")
    return cast(Stratum, result)


@dataclass(frozen=True, slots=True)
class Deform360CohortUnitV1:
    """One exact object/episode unit from the official-Hub Stage-0 selection."""

    object_id: str
    stratum: Stratum
    episode_id: int
    metadata_path: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        object_id = _strict_string(self.object_id, name="object_id")
        if "/" in object_id or object_id in {".", ".."}:
            raise ValueError("object_id must be one canonical path component")
        stratum = _stratum(self.stratum, name="stratum")
        episode_id = _strict_integer(self.episode_id, name="episode_id", minimum=0)
        metadata_path = _strict_string(self.metadata_path, name="metadata_path")
        expected_path = f"raw/{object_id}/metadata.json"
        if metadata_path != expected_path:
            raise ValueError(
                f"metadata_path must equal {expected_path!r} for object {object_id!r}"
            )
        metadata_sha256 = _strict_digest(
            self.metadata_sha256,
            name="metadata_sha256",
            pattern=_SHA256,
        )
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "metadata_path", metadata_path)
        object.__setattr__(self, "metadata_sha256", metadata_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "stratum": self.stratum,
            "episode_id": self.episode_id,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        name: str = "cohort unit",
    ) -> Deform360CohortUnitV1:
        mapping = _strict_mapping(value, name=name)
        _exact_keys(mapping, _UNIT_FIELDS, name=name)
        return cls(
            object_id=mapping["object_id"],
            stratum=mapping["stratum"],
            episode_id=mapping["episode_id"],
            metadata_path=mapping["metadata_path"],
            metadata_sha256=mapping["metadata_sha256"],
        )


def _canonical_units(
    value: Any,
    *,
    name: str,
    expected_count: int,
    expected_per_stratum: int,
) -> tuple[Deform360CohortUnitV1, ...]:
    raw_units = _strict_list(value, name=name)
    units = tuple(
        sorted(
            (
                Deform360CohortUnitV1.from_dict(item, name=f"{name}[{index}]")
                for index, item in enumerate(raw_units)
            ),
            key=lambda item: item.object_id,
        )
    )
    if len(units) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} physical objects")
    object_ids = tuple(unit.object_id for unit in units)
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"{name} object_id values must be unique")
    for stratum in ("sheet", "volumetric"):
        count = sum(unit.stratum == stratum for unit in units)
        if count != expected_per_stratum:
            raise ValueError(
                f"{name} must contain exactly {expected_per_stratum} {stratum} objects"
            )
    return units


def _validate_selection_hashes(selection: Mapping[str, Any]) -> None:
    supplied_selection = _strict_digest(
        selection["selection_sha256"],
        name="selection_sha256",
        pattern=_SHA256,
    )
    nested_selection = _strict_mapping(selection["selection"], name="selection")
    if supplied_selection != _sha256_json(nested_selection):
        raise ValueError("selection_sha256 does not match the selected cohort")

    supplied_content = _strict_digest(
        selection["content_selection_sha256"],
        name="content_selection_sha256",
        pattern=_SHA256,
    )
    content = dict(selection)
    content.pop("content_selection_sha256")
    content.pop("implementation_revision")
    content.pop("selection_artifact_sha256")
    if supplied_content != _sha256_json(content):
        raise ValueError("content_selection_sha256 does not match selection content")

    supplied_artifact = _strict_digest(
        selection["selection_artifact_sha256"],
        name="selection_artifact_sha256",
        pattern=_SHA256,
    )
    artifact = dict(selection)
    artifact.pop("selection_artifact_sha256")
    if supplied_artifact != _sha256_json(artifact):
        raise ValueError("selection_artifact_sha256 does not match the committed artifact")


def validate_deform360_official_hub_selection(value: Any) -> dict[str, Any]:
    """Validate the authoritative BPT Stage-0 selection and all nested identities."""

    selection = _strict_mapping(value, name="Deform360 official-Hub selection")
    _exact_keys(selection, _SELECTION_FIELDS, name="Deform360 official-Hub selection")
    if selection["schema"] != DEFORM360_SELECTION_SCHEMA:
        raise ValueError("unsupported Deform360 official-Hub selection schema")
    version = _strict_integer(selection["schema_version"], name="schema_version", minimum=1)
    if version != DEFORM360_SELECTION_VERSION:
        raise ValueError("unsupported Deform360 official-Hub selection version")
    if selection["protocol_id"] != DEFORM360_PROTOCOL_ID:
        raise ValueError("unexpected Deform360 protocol_id")
    protocol_sha256 = _strict_digest(
        selection["protocol_sha256"],
        name="protocol_sha256",
        pattern=_SHA256,
    )
    implementation_revision = _revision(
        selection["implementation_revision"],
        name="implementation_revision",
    )
    _strict_integer(
        selection["available_raw_object_count"],
        name="available_raw_object_count",
        minimum=22,
    )
    _strict_integer(
        selection["excluded_object_count"],
        name="excluded_object_count",
        minimum=0,
    )
    _strict_mapping(selection["cache_preflight"], name="cache_preflight")
    _strict_mapping(selection["prior_protocols"], name="prior_protocols")
    _strict_string(selection["next_gate"], name="next_gate")
    if _strict_bool(
        selection["replacement_allowed_after_payload_access"],
        name="replacement_allowed_after_payload_access",
    ):
        raise ValueError("replacement after selected payload access must remain forbidden")

    dataset = _strict_mapping(selection["dataset"], name="dataset")
    _exact_keys(dataset, _DATASET_FIELDS, name="dataset")
    if dataset["repo_id"] != DEFORM360_DATASET_REPOSITORY:
        raise ValueError("unexpected Deform360 dataset repository")
    dataset_requested_revision = _strict_string(
        dataset["requested_revision"],
        name="dataset.requested_revision",
    )
    dataset_resolved_revision = _revision(
        dataset["resolved_revision"],
        name="dataset.resolved_revision",
    )
    if dataset["raw_prefix"] != "raw":
        raise ValueError("Deform360 raw_prefix must remain 'raw'")

    processing = _strict_mapping(selection["official_processing"], name="official_processing")
    _exact_keys(processing, _PROCESSING_FIELDS, name="official_processing")
    if processing["repository"] != DEFORM360_PROCESSING_REPOSITORY:
        raise ValueError("unexpected official Deform360 processing repository")
    processing_revision = _revision(
        processing["revision"],
        name="official_processing.revision",
    )
    if not _strict_bool(
        processing["future_processing_revision_change_requires_new_protocol"],
        name="official_processing.future_processing_revision_change_requires_new_protocol",
    ):
        raise ValueError("processing revision changes must require a new protocol")
    stages = _strict_string_list(
        processing["required_stages"],
        name="official_processing.required_stages",
    )
    if not stages or len(stages) != len(set(stages)):
        raise ValueError("official processing stages must be nonempty and unique")

    split = _strict_mapping(selection["selection"], name="selection")
    _exact_keys(split, _SELECTION_SPLIT_FIELDS, name="selection")
    calibration_units = _canonical_units(
        split["calibration"],
        name="selection.calibration",
        expected_count=10,
        expected_per_stratum=5,
    )
    target_units = _canonical_units(
        split["confirmation"],
        name="selection.confirmation",
        expected_count=12,
        expected_per_stratum=6,
    )
    calibration_ids = tuple(unit.object_id for unit in calibration_units)
    target_ids = tuple(unit.object_id for unit in target_units)
    if set(calibration_ids) & set(target_ids):
        raise ValueError("calibration and confirmation physical objects must be disjoint")

    boundary = _strict_mapping(selection["information_boundary"], name="information_boundary")
    _exact_keys(boundary, _INFORMATION_BOUNDARY_FIELDS, name="information_boundary")
    for field_name, expected in _INFORMATION_BOUNDARY.items():
        observed = _strict_bool(boundary[field_name], name=f"information_boundary.{field_name}")
        if observed is not expected:
            raise ValueError(f"information_boundary.{field_name} changed")
    opened_paths = _strict_string_list(
        boundary["opened_metadata_paths"],
        name="information_boundary.opened_metadata_paths",
    )
    expected_paths = sorted(
        unit.metadata_path for unit in (*calibration_units, *target_units)
    )
    if opened_paths != expected_paths:
        raise ValueError(
            "opened_metadata_paths must equal the exact selected object metadata paths"
        )

    _validate_selection_hashes(selection)
    return {
        "selection_artifact_sha256": selection["selection_artifact_sha256"],
        "content_selection_sha256": selection["content_selection_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "implementation_revision": implementation_revision,
        "protocol_id": DEFORM360_PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "dataset_requested_revision": dataset_requested_revision,
        "dataset_resolved_revision": dataset_resolved_revision,
        "processing_revision": processing_revision,
        "calibration_units": calibration_units,
        "target_units": target_units,
    }


@dataclass(frozen=True, slots=True)
class Deform360OfficialHubCohortBindingV1:
    """Portable binding from Prob4D promotion to one exact BPT Stage-0 selection."""

    source_repository: str
    source_revision: str
    source_path: str
    selection_artifact_sha256: str
    content_selection_sha256: str
    selection_sha256: str
    selection_implementation_revision: str
    protocol_id: str
    protocol_sha256: str
    dataset_repository: str
    dataset_requested_revision: str
    dataset_resolved_revision: str
    processing_repository: str
    processing_revision: str
    calibration_units: tuple[Deform360CohortUnitV1, ...]
    target_units: tuple[Deform360CohortUnitV1, ...]

    def __post_init__(self) -> None:
        source_repository = _repository(self.source_repository, name="source_repository")
        if source_repository != BAYESIAN_PHYSTWIN_REPOSITORY:
            raise ValueError(
                f"source_repository must equal {BAYESIAN_PHYSTWIN_REPOSITORY!r}"
            )
        object.__setattr__(self, "source_repository", source_repository)
        object.__setattr__(
            self,
            "source_revision",
            _revision(self.source_revision, name="source_revision"),
        )
        object.__setattr__(self, "source_path", _canonical_source_path(self.source_path))
        for field_name in (
            "selection_artifact_sha256",
            "content_selection_sha256",
            "selection_sha256",
            "protocol_sha256",
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
            "selection_implementation_revision",
            _revision(
                self.selection_implementation_revision,
                name="selection_implementation_revision",
            ),
        )
        if self.protocol_id != DEFORM360_PROTOCOL_ID:
            raise ValueError("unexpected protocol_id")
        if self.dataset_repository != DEFORM360_DATASET_REPOSITORY:
            raise ValueError("unexpected dataset_repository")
        object.__setattr__(
            self,
            "dataset_requested_revision",
            _strict_string(
                self.dataset_requested_revision,
                name="dataset_requested_revision",
            ),
        )
        object.__setattr__(
            self,
            "dataset_resolved_revision",
            _revision(
                self.dataset_resolved_revision,
                name="dataset_resolved_revision",
            ),
        )
        if self.processing_repository != DEFORM360_PROCESSING_REPOSITORY:
            raise ValueError("unexpected processing_repository")
        object.__setattr__(
            self,
            "processing_revision",
            _revision(self.processing_revision, name="processing_revision"),
        )
        calibration = tuple(self.calibration_units)
        target = tuple(self.target_units)
        if calibration != tuple(sorted(calibration, key=lambda item: item.object_id)):
            raise ValueError("calibration_units must be sorted by object_id")
        if target != tuple(sorted(target, key=lambda item: item.object_id)):
            raise ValueError("target_units must be sorted by object_id")
        if len(calibration) != 10 or len(target) != 12:
            raise ValueError("binding requires the exact 10/12 calibration/target object split")
        if not all(
            isinstance(unit, Deform360CohortUnitV1)
            for unit in (*calibration, *target)
        ):
            raise ValueError("cohort units must be Deform360CohortUnitV1 values")
        calibration_ids = tuple(unit.object_id for unit in calibration)
        target_ids = tuple(unit.object_id for unit in target)
        if len(set(calibration_ids)) != 10 or len(set(target_ids)) != 12:
            raise ValueError("cohort object IDs must be unique within each split")
        if set(calibration_ids) & set(target_ids):
            raise ValueError("calibration and target objects must be disjoint")
        for name, units, expected_count in (
            ("calibration", calibration, 5),
            ("target", target, 6),
        ):
            for stratum in ("sheet", "volumetric"):
                if sum(unit.stratum == stratum for unit in units) != expected_count:
                    raise ValueError(
                        f"{name} requires exactly {expected_count} {stratum} objects"
                    )
        object.__setattr__(self, "calibration_units", calibration)
        object.__setattr__(self, "target_units", target)

    @property
    def calibration_group_ids(self) -> tuple[str, ...]:
        return tuple(unit.object_id for unit in self.calibration_units)

    @property
    def target_group_ids(self) -> tuple[str, ...]:
        return tuple(unit.object_id for unit in self.target_units)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": DEFORM360_COHORT_BINDING_SCHEMA,
            "schema_version": DEFORM360_COHORT_BINDING_VERSION,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_path": self.source_path,
            "selection_schema": DEFORM360_SELECTION_SCHEMA,
            "selection_schema_version": DEFORM360_SELECTION_VERSION,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "content_selection_sha256": self.content_selection_sha256,
            "selection_sha256": self.selection_sha256,
            "selection_implementation_revision": self.selection_implementation_revision,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "dataset_repository": self.dataset_repository,
            "dataset_requested_revision": self.dataset_requested_revision,
            "dataset_resolved_revision": self.dataset_resolved_revision,
            "processing_repository": self.processing_repository,
            "processing_revision": self.processing_revision,
            "statistical_unit": "physical-object",
            "calibration_units": [unit.to_dict() for unit in self.calibration_units],
            "target_units": [unit.to_dict() for unit in self.target_units],
            "calibration_group_ids": list(self.calibration_group_ids),
            "target_group_ids": list(self.target_group_ids),
            "information_boundary": dict(_BINDING_INFORMATION_BOUNDARY),
            "replacement_allowed_after_payload_access": False,
            "claim_boundary": DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY,
        }

    @property
    def cohort_binding_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "cohort_binding_id": self.cohort_binding_id}


def build_deform360_official_hub_cohort_binding(
    selection_value: Any,
    *,
    source_repository: str,
    source_revision: str,
    source_path: str,
) -> Deform360OfficialHubCohortBindingV1:
    """Bind one exact authoritative selection to its repository source identity."""

    validated = validate_deform360_official_hub_selection(selection_value)
    return Deform360OfficialHubCohortBindingV1(
        source_repository=source_repository,
        source_revision=source_revision,
        source_path=source_path,
        selection_artifact_sha256=validated["selection_artifact_sha256"],
        content_selection_sha256=validated["content_selection_sha256"],
        selection_sha256=validated["selection_sha256"],
        selection_implementation_revision=validated["implementation_revision"],
        protocol_id=validated["protocol_id"],
        protocol_sha256=validated["protocol_sha256"],
        dataset_repository=DEFORM360_DATASET_REPOSITORY,
        dataset_requested_revision=validated["dataset_requested_revision"],
        dataset_resolved_revision=validated["dataset_resolved_revision"],
        processing_repository=DEFORM360_PROCESSING_REPOSITORY,
        processing_revision=validated["processing_revision"],
        calibration_units=validated["calibration_units"],
        target_units=validated["target_units"],
    )


def deform360_cohort_binding_from_dict(value: Any) -> Deform360OfficialHubCohortBindingV1:
    """Parse and independently replay one portable cohort binding."""

    mapping = _strict_mapping(value, name="Deform360 cohort binding")
    _exact_keys(mapping, _BINDING_FIELDS, name="Deform360 cohort binding")
    if mapping["schema_name"] != DEFORM360_COHORT_BINDING_SCHEMA:
        raise ValueError("unsupported Deform360 cohort binding schema")
    version = _strict_integer(mapping["schema_version"], name="schema_version", minimum=1)
    if version != DEFORM360_COHORT_BINDING_VERSION:
        raise ValueError("unsupported Deform360 cohort binding version")
    if mapping["selection_schema"] != DEFORM360_SELECTION_SCHEMA:
        raise ValueError("unexpected selection_schema")
    selection_version = _strict_integer(
        mapping["selection_schema_version"],
        name="selection_schema_version",
        minimum=1,
    )
    if selection_version != DEFORM360_SELECTION_VERSION:
        raise ValueError("unexpected selection_schema_version")
    if mapping["statistical_unit"] != "physical-object":
        raise ValueError("statistical_unit must remain 'physical-object'")
    boundary = _strict_mapping(mapping["information_boundary"], name="information_boundary")
    _exact_keys(boundary, set(_BINDING_INFORMATION_BOUNDARY), name="information_boundary")
    normalized_boundary = {
        key: _strict_bool(boundary[key], name=f"information_boundary.{key}")
        for key in _BINDING_INFORMATION_BOUNDARY
    }
    if normalized_boundary != _BINDING_INFORMATION_BOUNDARY:
        raise ValueError("Deform360 cohort binding information_boundary mismatch")
    if _strict_bool(
        mapping["replacement_allowed_after_payload_access"],
        name="replacement_allowed_after_payload_access",
    ):
        raise ValueError("replacement after payload access must remain forbidden")
    if mapping["claim_boundary"] != DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY:
        raise ValueError("Deform360 cohort binding claim_boundary mismatch")

    binding = Deform360OfficialHubCohortBindingV1(
        source_repository=mapping["source_repository"],
        source_revision=mapping["source_revision"],
        source_path=mapping["source_path"],
        selection_artifact_sha256=mapping["selection_artifact_sha256"],
        content_selection_sha256=mapping["content_selection_sha256"],
        selection_sha256=mapping["selection_sha256"],
        selection_implementation_revision=mapping["selection_implementation_revision"],
        protocol_id=mapping["protocol_id"],
        protocol_sha256=mapping["protocol_sha256"],
        dataset_repository=mapping["dataset_repository"],
        dataset_requested_revision=mapping["dataset_requested_revision"],
        dataset_resolved_revision=mapping["dataset_resolved_revision"],
        processing_repository=mapping["processing_repository"],
        processing_revision=mapping["processing_revision"],
        calibration_units=tuple(
            Deform360CohortUnitV1.from_dict(item, name=f"calibration_units[{index}]")
            for index, item in enumerate(
                _strict_list(mapping["calibration_units"], name="calibration_units")
            )
        ),
        target_units=tuple(
            Deform360CohortUnitV1.from_dict(item, name=f"target_units[{index}]")
            for index, item in enumerate(
                _strict_list(mapping["target_units"], name="target_units")
            )
        ),
    )
    for field_name, expected in (
        ("calibration_group_ids", list(binding.calibration_group_ids)),
        ("target_group_ids", list(binding.target_group_ids)),
    ):
        values = _strict_list(mapping[field_name], name=field_name)
        if values != expected:
            raise ValueError(f"{field_name} does not match bound cohort units")
    supplied_id = _strict_digest(
        mapping["cohort_binding_id"],
        name="cohort_binding_id",
        pattern=_SHA256,
    )
    if supplied_id != binding.cohort_binding_id:
        raise ValueError("Deform360 cohort_binding_id mismatch")
    return binding


def validate_deform360_cohort_binding_against_selection(
    binding: Deform360OfficialHubCohortBindingV1,
    selection_value: Any,
) -> None:
    """Rebind a portable cohort artifact to the exact source selection bytes."""

    if not isinstance(binding, Deform360OfficialHubCohortBindingV1):
        raise ValueError("binding must be Deform360OfficialHubCohortBindingV1")
    rebuilt = build_deform360_official_hub_cohort_binding(
        selection_value,
        source_repository=binding.source_repository,
        source_revision=binding.source_revision,
        source_path=binding.source_path,
    )
    if rebuilt.to_dict() != binding.to_dict():
        raise ValueError("Deform360 cohort binding disagrees with the exact source selection")


def validate_promotion_config_against_deform360_binding(
    config_value: Any,
    binding: Deform360OfficialHubCohortBindingV1,
) -> None:
    """Require a promotion configuration to use exactly the sealed 10/12 split."""

    if not isinstance(binding, Deform360OfficialHubCohortBindingV1):
        raise ValueError("binding must be Deform360OfficialHubCohortBindingV1")
    config = _strict_mapping(config_value, name="promotion lock configuration")
    if config.get("bayesian_phystwin_repository") != binding.source_repository:
        raise ValueError(
            "promotion configuration bayesian_phystwin_repository disagrees with cohort binding"
        )
    if config.get("bayesian_phystwin_revision") != binding.source_revision:
        raise ValueError(
            "promotion configuration bayesian_phystwin_revision disagrees with cohort binding"
        )
    for field_name, expected in (
        ("calibration_group_ids", list(binding.calibration_group_ids)),
        ("target_group_ids", list(binding.target_group_ids)),
    ):
        observed = _strict_list(config.get(field_name), name=field_name)
        if observed != expected:
            raise ValueError(f"promotion configuration {field_name} disagrees with cohort binding")
    development = _strict_string_list(
        config.get("development_group_ids"),
        name="development_group_ids",
    )
    if set(development) & (
        set(binding.calibration_group_ids) | set(binding.target_group_ids)
    ):
        raise ValueError("promotion development groups overlap the bound BPT cohort")
    minimum_target = _strict_integer(
        config.get("minimum_target_group_count"),
        name="minimum_target_group_count",
        minimum=1,
    )
    if minimum_target != len(binding.target_group_ids):
        raise ValueError(
            "minimum_target_group_count must equal the complete bound confirmation cohort"
        )
    frozen = _strict_mapping(config.get("frozen_artifact_ids"), name="frozen_artifact_ids")
    if frozen.get("cohort_binding") != binding.cohort_binding_id:
        raise ValueError(
            "frozen_artifact_ids.cohort_binding must equal the Deform360 cohort_binding_id"
        )


def write_deform360_cohort_binding(
    binding: Deform360OfficialHubCohortBindingV1,
    path: str | Path,
) -> None:
    """Publish one cohort binding atomically without rewriting existing evidence."""

    if not isinstance(binding, Deform360OfficialHubCohortBindingV1):
        raise ValueError("binding must be Deform360OfficialHubCohortBindingV1")
    _atomic_write_json(Path(path), binding.to_dict())


def load_deform360_cohort_binding(
    path: str | Path,
) -> Deform360OfficialHubCohortBindingV1:
    """Load and fully replay one portable cohort binding."""

    value, _ = _load_json(Path(path), name="Deform360 cohort binding")
    return deform360_cohort_binding_from_dict(value)


def bind_cli(arguments: Sequence[str]) -> int:
    """Implement the grouped ``cohort-bind`` command."""

    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider cohort-bind",
        description=(
            "Bind promotion to BayesianPhysTwin's committed official-Hub "
            "Deform360 Stage-0 selection."
        ),
    )
    parser.add_argument("selection", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-repository",
        default=BAYESIAN_PHYSTWIN_REPOSITORY,
    )
    parser.add_argument("--source-path", default=DEFORM360_SELECTION_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    selection, _ = _load_json(parsed.selection, name="Deform360 official-Hub selection")
    binding = build_deform360_official_hub_cohort_binding(
        selection,
        source_repository=parsed.source_repository,
        source_revision=parsed.source_revision,
        source_path=parsed.source_path,
    )
    write_deform360_cohort_binding(binding, parsed.output)
    print(binding.cohort_binding_id)
    return 0


def verify_cli(arguments: Sequence[str]) -> int:
    """Implement the grouped ``cohort-verify`` command."""

    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider cohort-verify",
        description="Replay a Deform360 cohort binding and optionally rebind its selection.",
    )
    parser.add_argument("binding", type=Path)
    parser.add_argument("--selection", type=Path)
    parsed = parser.parse_args(arguments)
    binding = load_deform360_cohort_binding(parsed.binding)
    selection_verified = False
    if parsed.selection is not None:
        selection, _ = _load_json(
            parsed.selection,
            name="Deform360 official-Hub selection",
        )
        validate_deform360_cohort_binding_against_selection(binding, selection)
        selection_verified = True
    print(
        json.dumps(
            {
                "cohort_binding_id": binding.cohort_binding_id,
                "selection_artifact_sha256": binding.selection_artifact_sha256,
                "selection_verified": selection_verified,
                "calibration_object_count": len(binding.calibration_group_ids),
                "target_object_count": len(binding.target_group_ids),
                "dataset_resolved_revision": binding.dataset_resolved_revision,
                "processing_revision": binding.processing_revision,
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "BAYESIAN_PHYSTWIN_REPOSITORY",
    "DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY",
    "DEFORM360_COHORT_BINDING_SCHEMA",
    "DEFORM360_COHORT_BINDING_VERSION",
    "DEFORM360_DATASET_REPOSITORY",
    "DEFORM360_PROCESSING_REPOSITORY",
    "DEFORM360_PROTOCOL_ID",
    "DEFORM360_SELECTION_PATH",
    "DEFORM360_SELECTION_SCHEMA",
    "DEFORM360_SELECTION_VERSION",
    "Deform360CohortUnitV1",
    "Deform360OfficialHubCohortBindingV1",
    "bind_cli",
    "build_deform360_official_hub_cohort_binding",
    "deform360_cohort_binding_from_dict",
    "load_deform360_cohort_binding",
    "validate_deform360_cohort_binding_against_selection",
    "validate_deform360_official_hub_selection",
    "validate_promotion_config_against_deform360_binding",
    "verify_cli",
    "write_deform360_cohort_binding",
]
