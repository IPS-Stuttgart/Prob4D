"""Portable binding to an authoritative Deform360 Stage-0 selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._deform360_cohort_schema import (
    _BINDING_FIELDS,
    _BINDING_INFORMATION_BOUNDARY,
    BAYESIAN_PHYSTWIN_REPOSITORY,
    DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY,
    DEFORM360_COHORT_BINDING_SCHEMA,
    DEFORM360_COHORT_BINDING_VERSION,
    DEFORM360_DATASET_REPOSITORY,
    DEFORM360_PROCESSING_REPOSITORY,
    DEFORM360_PROTOCOL_ID,
    DEFORM360_SELECTION_SCHEMA,
    DEFORM360_SELECTION_VERSION,
    Deform360CohortUnitV1,
    _canonical_source_path,
    validate_deform360_official_hub_selection,
)
from ._heldout_promotion_common import _repository, _revision
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
            raise ValueError(f"source_repository must equal {BAYESIAN_PHYSTWIN_REPOSITORY!r}")
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
        if not all(isinstance(unit, Deform360CohortUnitV1) for unit in (*calibration, *target)):
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
                    raise ValueError(f"{name} requires exactly {expected_count} {stratum} objects")
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


def deform360_cohort_binding_from_dict(
    value: Any,
) -> Deform360OfficialHubCohortBindingV1:
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
            for index, item in enumerate(_strict_list(mapping["target_units"], name="target_units"))
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
