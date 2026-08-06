"""Stage-0 Deform360 selection schema and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from ._heldout_promotion_common import _revision
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
DEFORM360_SELECTION_PATH = "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"

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
_UNIT_FIELDS = {
    "object_id",
    "stratum",
    "episode_id",
    "metadata_path",
    "metadata_sha256",
}
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
    return [_strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(items)]


def _canonical_source_path(value: Any) -> str:
    result = _strict_string(value, name="source_path")
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source_path must be a repository-relative canonical path")
    if result != DEFORM360_SELECTION_PATH:
        raise ValueError(
            f"source_path must identify the authoritative selection at {DEFORM360_SELECTION_PATH!r}"
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
            raise ValueError(f"metadata_path must equal {expected_path!r} for object {object_id!r}")
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
    def from_dict(cls, value: Any, *, name: str = "cohort unit") -> Deform360CohortUnitV1:
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
    expected_paths = sorted(unit.metadata_path for unit in (*calibration_units, *target_units))
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
