"""Freeze/lock validation for the retained CUT3R source preflight."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

from ._cut3r_source_preflight_cases import (
    _collect_source_case_descriptors,
    _validate_source_groups,
    _validate_target_groups,
)
from ._cut3r_source_preflight_common import (
    SOURCE_FREEZE_READY,
    SOURCE_FREEZE_SCHEMA,
    SOURCE_FREEZE_VERSION,
    SOURCE_ROLES,
    _exact_keys,
    _integer,
    _literal_string,
    _record_id,
    _revision,
    _sha256,
)

_PROVIDER_FIELDS: Final = frozenset(
    {
        "repository",
        "revision",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_byte_count",
        "execution_mode",
        "revisit_count",
        "global_alignment",
        "second_pass_allowed",
    }
)
_PROB4D_FIELDS: Final = frozenset(
    {
        "revision",
        "distribution_filename",
        "distribution_sha256",
        "distribution_byte_count",
    }
)
_INFORMATION_BOUNDARY: Final = {
    "camera_panel_change_after_freeze_allowed": False,
    "downstream_physical_innovations_opened": False,
    "replacement_after_freeze_allowed": False,
    "source_future_geometry_opened": False,
    "source_prediction_payloads_opened": False,
    "source_residuals_or_truth_opened": False,
    "source_rgb_frames_decoded": False,
    "source_rgb_video_bytes_hashed": True,
    "target_outcomes_opened": False,
    "target_payloads_opened": False,
}


def _load_comparison_lock(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from prob4d.cut3r_comparison import (
            build_cut3r_comparison_lock,
            load_cut3r_comparison_lock,
        )
    except ImportError as error:
        raise ValueError("Prob4D must be installed to validate the comparison lock") from error
    expected = build_cut3r_comparison_lock(spec)
    retained = load_cut3r_comparison_lock(path)
    if retained != expected:
        raise ValueError("retained comparison lock differs from the canonical comparison spec")
    return retained


def _validate_source_freeze(
    freeze: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    spec: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        freeze.get("schema") != SOURCE_FREEZE_SCHEMA
        or freeze.get("schema_version") != SOURCE_FREEZE_VERSION
    ):
        raise ValueError("unsupported CUT3R Deform360 source freeze")
    if freeze.get("decision") != SOURCE_FREEZE_READY:
        raise ValueError("source freeze did not pass the preregistered support gate")
    freeze_id = _sha256(freeze.get("source_freeze_id"), name="source_freeze_id")
    unsigned_freeze = dict(freeze)
    unsigned_freeze.pop("source_freeze_id")
    if freeze_id != _record_id(unsigned_freeze):
        raise ValueError("source_freeze_id does not match the canonical freeze content")
    if freeze.get("information_boundary") != _INFORMATION_BOUNDARY:
        raise ValueError("source freeze information boundary changed or was exceeded")
    _literal_string(freeze.get("claim_boundary"), name="source freeze claim_boundary")

    expected_source_count = cast(int, request["source_group_count"])
    if freeze.get("source_group_count") != expected_source_count:
        raise ValueError("source freeze group count differs from the request")
    source_groups = _validate_source_groups(
        freeze.get("source_groups"), expected_count=expected_source_count
    )
    expected_target_count = cast(int, request["forbidden_target_group_count"])
    if freeze.get("forbidden_target_group_count") != expected_target_count:
        raise ValueError("source freeze forbidden-target count differs from the request")
    _validate_target_groups(
        freeze.get("forbidden_target_groups"),
        expected_count=expected_target_count,
        source_object_episodes={
            (cast(str, group["object_id"]), cast(int, group["episode_id"]))
            for group in source_groups.values()
        },
    )
    spec_sha = _record_id(spec)
    if _sha256(freeze.get("comparison_spec_sha256"), name="comparison_spec_sha256") != spec_sha:
        raise ValueError("comparison spec bytes differ from the source freeze")

    provider = freeze.get("provider")
    if type(provider) is not dict:
        raise ValueError("source freeze provider must be a JSON object")
    provider_map = cast(dict[str, Any], provider)
    _exact_keys(provider_map, _PROVIDER_FIELDS, name="source freeze provider")
    provider_repository = _literal_string(
        provider_map.get("repository"), name="source freeze provider.repository"
    )
    provider_revision = _revision(
        provider_map.get("revision"), name="source freeze provider.revision"
    )
    checkpoint_filename = _literal_string(
        provider_map.get("checkpoint_filename"),
        name="source freeze provider.checkpoint_filename",
    )
    checkpoint_sha = _sha256(
        provider_map.get("checkpoint_sha256"),
        name="source freeze provider.checkpoint_sha256",
    )
    checkpoint_bytes = _integer(
        provider_map.get("checkpoint_byte_count"),
        name="source freeze provider.checkpoint_byte_count",
        minimum=1,
    )
    if provider_map.get("execution_mode") != "recurrent-online":
        raise ValueError("source freeze no longer binds recurrent-online CUT3R execution")
    if provider_map.get("revisit_count") != 1:
        raise ValueError("source freeze CUT3R revisit count changed")
    if provider_map.get("global_alignment") is not False:
        raise ValueError("source freeze unexpectedly enables CUT3R global alignment")
    if provider_map.get("second_pass_allowed") is not False:
        raise ValueError("source freeze unexpectedly permits a second CUT3R pass")
    lock_provider = cast(Mapping[str, Any], lock["provider"])
    if provider_repository != lock_provider["repository"]:
        raise ValueError("source freeze and comparison lock name different providers")
    if (
        provider_revision != spec.get("provider_revision")
        or provider_revision != lock_provider["revision"]
    ):
        raise ValueError("source freeze, spec, and lock bind different provider revisions")
    if (
        checkpoint_sha != spec.get("checkpoint_sha256")
        or checkpoint_sha != lock_provider["checkpoint_sha256"]
    ):
        raise ValueError("source freeze, spec, and lock bind different checkpoints")

    prob4d = freeze.get("prob4d")
    if type(prob4d) is not dict:
        raise ValueError("source freeze prob4d must be a JSON object")
    prob4d_map = cast(dict[str, Any], prob4d)
    _exact_keys(prob4d_map, _PROB4D_FIELDS, name="source freeze prob4d")
    prob4d_revision = _revision(prob4d_map.get("revision"), name="source freeze prob4d.revision")
    distribution_filename = _literal_string(
        prob4d_map.get("distribution_filename"),
        name="source freeze prob4d.distribution_filename",
    )
    distribution_sha = _sha256(
        prob4d_map.get("distribution_sha256"),
        name="source freeze prob4d.distribution_sha256",
    )
    distribution_bytes = _integer(
        prob4d_map.get("distribution_byte_count"),
        name="source freeze prob4d.distribution_byte_count",
        minimum=1,
    )
    lock_prob4d = cast(Mapping[str, Any], lock["prob4d"])
    if prob4d_revision != spec.get("prob4d_revision") or prob4d_revision != lock_prob4d["revision"]:
        raise ValueError("source freeze, spec, and lock bind different Prob4D revisions")
    if (
        distribution_sha != spec.get("prob4d_distribution_sha256")
        or distribution_sha != lock_prob4d["distribution_sha256"]
    ):
        raise ValueError("source freeze, spec, and lock bind different Prob4D distributions")

    descriptors = _collect_source_case_descriptors(
        freeze,
        source_groups=source_groups,
        expected_case_count=cast(int, request["expected_case_count"]),
    )
    lock_cases: dict[str, tuple[str, str, int]] = {}
    for raw_group in cast(list[Mapping[str, Any]], lock["groups"]):
        group_id = cast(str, raw_group["group_id"])
        for raw_case in cast(list[Mapping[str, Any]], raw_group["cases"]):
            case_id = cast(str, raw_case["case_id"])
            lock_cases[case_id] = (
                group_id,
                cast(str, raw_case["input_video_sha256"]),
                cast(int, raw_case["input_video_byte_count"]),
            )
    if set(lock_cases) != {cast(str, item["case_id"]) for item in descriptors}:
        raise ValueError("source freeze and comparison lock contain different source cases")
    for descriptor in descriptors:
        case_id = cast(str, descriptor["case_id"])
        expected = lock_cases[case_id]
        measured = (
            cast(str, descriptor["group_id"]),
            cast(str, descriptor["video_sha256"]),
            cast(int, descriptor["video_byte_count"]),
        )
        if measured != expected:
            raise ValueError(f"source case {case_id!r} differs from the comparison lock")
    expected_roles = {
        role: sorted(group_id for group_id, group in source_groups.items() if group["role"] == role)
        for role in SOURCE_ROLES
    }
    if lock["group_roles"] != expected_roles:
        raise ValueError("source freeze roles differ from the comparison lock")
    panel = freeze.get("camera_panel")
    if type(panel) is not dict:
        raise ValueError("source freeze has no selected camera panel")
    selected = cast(dict[str, Any], panel).get("selected_cameras")
    if type(selected) is not list or any(type(item) is not str for item in selected):
        raise ValueError("source freeze selected camera panel is invalid")
    if len(selected) * len(source_groups) != len(descriptors):
        raise ValueError("selected camera panel does not explain the frozen case count")
    if {cast(str, item["view_id"]) for item in descriptors} != set(selected):
        raise ValueError("source cases differ from the selected camera panel")
    return {
        "source_freeze_id": freeze_id,
        "comparison_spec_sha256": spec_sha,
        "comparison_lock_id": lock["lock_id"],
        "provider_repository": provider_repository,
        "provider_revision": provider_revision,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_byte_count": checkpoint_bytes,
        "prob4d_revision": prob4d_revision,
        "prob4d_distribution_filename": distribution_filename,
        "prob4d_distribution_sha256": distribution_sha,
        "prob4d_distribution_byte_count": distribution_bytes,
        "descriptors": descriptors,
    }
