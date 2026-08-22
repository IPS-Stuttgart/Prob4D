"""External protocol and selection bindings for CUT3R source freezes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._cut3r_source_freeze_verify_common import (
    SUPPORT_PASS,
    _COMPARISON_CASE_FIELDS,
    _COMPARISON_GROUP_FIELDS,
    _COMPARISON_SPEC_FIELDS,
    _SOURCE_ROLES,
    _boolean,
    _exact_fields,
    _finite_number,
    _integer,
    _mapping,
    _revision,
    _sequence,
    _sha256,
    _string,
)


def _comparison_spec(
    value: object,
    *,
    freeze: Mapping[str, Any],
    source_groups: Sequence[Mapping[str, Any]],
    source_cases: Sequence[Mapping[str, Any]],
    frame_interval: Sequence[int],
) -> dict[str, Any]:
    mapping = _mapping(value, name="CUT3R comparison specification")
    _exact_fields(mapping, _COMPARISON_SPEC_FIELDS, name="CUT3R comparison specification")
    if _string(mapping["protocol_name"], name="comparison.protocol_name") != freeze[
        "protocol_name"
    ]:
        raise ValueError("comparison protocol_name differs from the source freeze")
    provider = cast(Mapping[str, Any], freeze["provider"])
    prob4d = cast(Mapping[str, Any], freeze["prob4d"])
    if _revision(mapping["provider_revision"], name="comparison.provider_revision") != provider[
        "revision"
    ]:
        raise ValueError("comparison provider_revision differs from the source freeze")
    if _sha256(mapping["checkpoint_sha256"], name="comparison.checkpoint_sha256") != provider[
        "checkpoint_sha256"
    ]:
        raise ValueError("comparison checkpoint_sha256 differs from the source freeze")
    if _revision(mapping["prob4d_revision"], name="comparison.prob4d_revision") != prob4d[
        "revision"
    ]:
        raise ValueError("comparison prob4d_revision differs from the source freeze")
    if _sha256(
        mapping["prob4d_distribution_sha256"],
        name="comparison.prob4d_distribution_sha256",
    ) != prob4d["distribution_sha256"]:
        raise ValueError("comparison Prob4D distribution differs from the source freeze")

    window_size = _integer(mapping["window_size"], name="comparison.window_size", minimum=2)
    overlap = _integer(mapping["overlap"], name="comparison.overlap", minimum=1)
    if overlap >= window_size:
        raise ValueError("comparison.overlap must be smaller than window_size")
    confidence = _finite_number(
        mapping["confidence_threshold"],
        name="comparison.confidence_threshold",
        minimum=0.0,
    )
    storage_dtype = _string(mapping["storage_dtype"], name="comparison.storage_dtype")
    if storage_dtype not in {"float32", "float64"}:
        raise ValueError("comparison.storage_dtype must be float32 or float64")
    raw_seeds = _sequence(mapping["random_seeds"], name="comparison.random_seeds", nonempty=True)
    seeds = [
        _integer(seed, name=f"comparison.random_seeds[{index}]")
        for index, seed in enumerate(raw_seeds)
    ]
    if len(seeds) != len(set(seeds)) or seeds != sorted(seeds):
        raise ValueError("comparison.random_seeds must be sorted and unique")
    include_revisit = _boolean(
        mapping["include_revisit_diagnostic"],
        name="comparison.include_revisit_diagnostic",
    )

    case_by_id = {cast(str, case["case_id"]): case for case in source_cases}
    groups_raw = _sequence(mapping["groups"], name="comparison.groups", nonempty=True)
    groups: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_cases: set[str] = set()
    frame_start, frame_stop = frame_interval
    for group_index, item in enumerate(groups_raw):
        group_name = f"comparison.groups[{group_index}]"
        group = _mapping(item, name=group_name)
        _exact_fields(group, _COMPARISON_GROUP_FIELDS, name=group_name)
        group_id = _string(group["group_id"], name=f"{group_name}.group_id")
        if group_id in seen_groups:
            raise ValueError(f"comparison.groups repeats group_id {group_id!r}")
        seen_groups.add(group_id)
        cases_raw = _sequence(group["cases"], name=f"{group_name}.cases", nonempty=True)
        cases: list[dict[str, Any]] = []
        for case_index, raw_case in enumerate(cases_raw):
            case_name = f"{group_name}.cases[{case_index}]"
            case = _mapping(raw_case, name=case_name)
            _exact_fields(case, _COMPARISON_CASE_FIELDS, name=case_name)
            case_id = _string(case["case_id"], name=f"{case_name}.case_id")
            if case_id in seen_cases:
                raise ValueError(f"comparison.groups repeats case_id {case_id!r}")
            seen_cases.add(case_id)
            source_case = case_by_id.get(case_id)
            if source_case is None or source_case["group_id"] != group_id:
                raise ValueError(f"{case_name}.case_id is not bound to source group {group_id!r}")
            observed_start = _integer(case["frame_start"], name=f"{case_name}.frame_start")
            observed_stop = _integer(
                case["frame_stop_exclusive"],
                name=f"{case_name}.frame_stop_exclusive",
                minimum=1,
            )
            evaluation_start = _integer(
                case["evaluation_frame_start"],
                name=f"{case_name}.evaluation_frame_start",
            )
            evaluation_stop = _integer(
                case["evaluation_frame_stop_exclusive"],
                name=f"{case_name}.evaluation_frame_stop_exclusive",
                minimum=1,
            )
            if [observed_start, observed_stop] != [frame_start, frame_stop]:
                raise ValueError(f"{case_name} source interval differs from the source freeze")
            if not observed_start <= evaluation_start < evaluation_stop <= observed_stop:
                raise ValueError(f"{case_name} evaluation interval is outside its source prefix")
            video_sha = _sha256(
                case["input_video_sha256"],
                name=f"{case_name}.input_video_sha256",
            )
            video_size = _integer(
                case["input_video_byte_count"],
                name=f"{case_name}.input_video_byte_count",
                minimum=1,
            )
            if (video_sha, video_size) != (
                source_case["input_video_sha256"],
                source_case["input_video_byte_count"],
            ):
                raise ValueError(f"{case_name} input video differs from the source freeze")
            cases.append(
                {
                    "case_id": case_id,
                    "input_video_sha256": video_sha,
                    "input_video_byte_count": video_size,
                    "frame_start": observed_start,
                    "frame_stop_exclusive": observed_stop,
                    "evaluation_frame_start": evaluation_start,
                    "evaluation_frame_stop_exclusive": evaluation_stop,
                }
            )
        if cases != sorted(cases, key=lambda record: record["case_id"]):
            raise ValueError(f"{group_name}.cases must use canonical case_id ordering")
        groups.append({"group_id": group_id, "cases": cases})
    if groups != sorted(groups, key=lambda record: record["group_id"]):
        raise ValueError("comparison.groups must use canonical group_id ordering")
    expected_group_ids = {cast(str, group["group_id"]) for group in source_groups}
    if seen_groups != expected_group_ids or seen_cases != set(case_by_id):
        raise ValueError(
            "comparison.groups must cover every frozen source group and case exactly once"
        )

    roles_raw = _mapping(mapping["group_roles"], name="comparison.group_roles")
    _exact_fields(roles_raw, set(_SOURCE_ROLES), name="comparison.group_roles")
    roles: dict[str, list[str]] = {}
    assigned: list[str] = []
    for role in _SOURCE_ROLES:
        raw_ids = _sequence(
            roles_raw[role],
            name=f"comparison.group_roles.{role}",
            nonempty=True,
        )
        identifiers = [
            _string(group_id, name=f"comparison.group_roles.{role}[{index}]")
            for index, group_id in enumerate(raw_ids)
        ]
        if identifiers != sorted(set(identifiers)):
            raise ValueError(f"comparison.group_roles.{role} must be sorted and unique")
        roles[role] = identifiers
        assigned.extend(identifiers)
    expected_roles = {
        role: sorted(
            cast(str, group["group_id"])
            for group in source_groups
            if group["role"] == role
        )
        for role in _SOURCE_ROLES
    }
    if roles != expected_roles or len(assigned) != len(set(assigned)):
        raise ValueError("comparison.group_roles differs from the frozen source roles")

    return {
        "protocol_name": freeze["protocol_name"],
        "provider_revision": provider["revision"],
        "checkpoint_sha256": provider["checkpoint_sha256"],
        "prob4d_revision": prob4d["revision"],
        "prob4d_distribution_sha256": prob4d["distribution_sha256"],
        "window_size": window_size,
        "overlap": overlap,
        "confidence_threshold": confidence,
        "storage_dtype": storage_dtype,
        "random_seeds": seeds,
        "groups": groups,
        "group_roles": roles,
        "include_revisit_diagnostic": include_revisit,
    }


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    protocol_bytes: bytes,
    freeze: Mapping[str, Any],
    source_groups: Sequence[Mapping[str, Any]],
    target_groups: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any] | None,
) -> None:
    if protocol.get("schema") != "prob4d.cut3r-deform360-source-freeze-protocol":
        raise ValueError("source protocol has an unsupported schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("source protocol has an unsupported schema version")
    identity = cast(Mapping[str, Any], freeze["source_protocol"])
    measured_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if (measured_sha, len(protocol_bytes)) != (identity["sha256"], identity["byte_count"]):
        raise ValueError("source protocol bytes differ from the source-freeze binding")
    if protocol.get("protocol_name") != freeze["protocol_name"]:
        raise ValueError("source protocol name differs from the source freeze")
    if protocol.get("claim_boundary") != freeze["claim_boundary"]:
        raise ValueError("source protocol claim boundary differs from the source freeze")
    if protocol.get("information_boundary") != freeze["information_boundary"]:
        raise ValueError("source protocol information boundary differs from the source freeze")
    if protocol.get("source_groups") != list(source_groups):
        raise ValueError("source protocol source_groups differ from the source freeze")
    if protocol.get("forbidden_target_groups") != list(target_groups):
        raise ValueError("source protocol forbidden_target_groups differ from the source freeze")

    provider = _mapping(protocol.get("provider"), name="source protocol provider")
    freeze_provider = cast(Mapping[str, Any], freeze["provider"])
    for field in (
        "repository",
        "revision",
        "checkpoint_filename",
        "execution_mode",
        "revisit_count",
        "global_alignment",
        "second_pass_allowed",
    ):
        if provider.get(field) != freeze_provider[field]:
            raise ValueError(f"source protocol provider.{field} differs from the source freeze")
    source_dataset = _mapping(
        protocol.get("source_dataset"),
        name="source protocol source_dataset",
    )
    selection = cast(Mapping[str, Any], freeze["deform360_selection"])
    for field in ("selection_artifact_sha256", "selection_sha256"):
        if source_dataset.get(field) != selection[field]:
            raise ValueError(f"source protocol source_dataset.{field} differs from the freeze")

    support = cast(Mapping[str, Any], freeze["support"])
    camera_policy = _mapping(protocol.get("camera_panel"), name="source protocol camera_panel")
    if camera_policy.get("minimum_common_supported_cameras") != support[
        "minimum_common_supported_cameras"
    ]:
        raise ValueError("source protocol minimum common-camera count differs from the freeze")
    if freeze["decision"] == SUPPORT_PASS:
        panel = cast(Mapping[str, Any], freeze["camera_panel"])
        for field in ("panel_size", "selection_rule", "first_camera_rule"):
            if camera_policy.get(field) != panel[field]:
                raise ValueError(f"source protocol camera_panel.{field} differs from the freeze")

    windowing = _mapping(protocol.get("windowing"), name="source protocol windowing")
    if [windowing.get("frame_start"), windowing.get("frame_stop_exclusive")] != support[
        "required_frame_interval"
    ]:
        raise ValueError("source protocol source interval differs from the source freeze")
    if comparison is not None:
        expected = _expected_comparison_from_protocol(
            protocol,
            freeze=freeze,
            source_groups=source_groups,
            source_cases=cast(Sequence[Mapping[str, Any]], freeze["source_cases"]),
        )
        if comparison != expected:
            raise ValueError(
                "comparison specification differs from the independently reconstructed "
                "protocol-bound specification"
            )


def _expected_comparison_from_protocol(
    protocol: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    source_groups: Sequence[Mapping[str, Any]],
    source_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    windowing = _mapping(protocol.get("windowing"), name="source protocol windowing")
    provider = _mapping(protocol.get("provider"), name="source protocol provider")
    cases_by_group: dict[str, list[dict[str, Any]]] = {
        cast(str, group["group_id"]): [] for group in source_groups
    }
    for source_case in source_cases:
        cases_by_group[cast(str, source_case["group_id"])].append(
            {
                "case_id": source_case["case_id"],
                "input_video_sha256": source_case["input_video_sha256"],
                "input_video_byte_count": source_case["input_video_byte_count"],
                "frame_start": windowing.get("frame_start"),
                "frame_stop_exclusive": windowing.get("frame_stop_exclusive"),
                "evaluation_frame_start": windowing.get("evaluation_frame_start"),
                "evaluation_frame_stop_exclusive": windowing.get(
                    "evaluation_frame_stop_exclusive"
                ),
            }
        )
    groups = [
        {
            "group_id": group_id,
            "cases": sorted(cases, key=lambda case: cast(str, case["case_id"])),
        }
        for group_id, cases in sorted(cases_by_group.items())
    ]
    roles = {
        role: sorted(
            cast(str, group["group_id"])
            for group in source_groups
            if group["role"] == role
        )
        for role in _SOURCE_ROLES
    }
    freeze_provider = cast(Mapping[str, Any], freeze["provider"])
    prob4d = cast(Mapping[str, Any], freeze["prob4d"])
    return {
        "protocol_name": freeze["protocol_name"],
        "provider_revision": freeze_provider["revision"],
        "checkpoint_sha256": freeze_provider["checkpoint_sha256"],
        "prob4d_revision": prob4d["revision"],
        "prob4d_distribution_sha256": prob4d["distribution_sha256"],
        "window_size": windowing.get("window_size"),
        "overlap": windowing.get("overlap"),
        "confidence_threshold": provider.get("confidence_threshold"),
        "storage_dtype": windowing.get("storage_dtype"),
        "random_seeds": windowing.get("random_seeds"),
        "groups": groups,
        "group_roles": roles,
        "include_revisit_diagnostic": windowing.get("include_revisit_diagnostic"),
    }


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    selection_bytes: bytes,
    freeze: Mapping[str, Any],
    source_groups: Sequence[Mapping[str, Any]],
    target_groups: Sequence[Mapping[str, Any]],
) -> None:
    identity = cast(Mapping[str, Any], freeze["deform360_selection"])
    measured_sha = hashlib.sha256(selection_bytes).hexdigest()
    if (measured_sha, len(selection_bytes)) != (identity["sha256"], identity["byte_count"]):
        raise ValueError("Deform360 selection-lock bytes differ from the source-freeze binding")
    for field in ("selection_artifact_sha256", "selection_sha256"):
        if selection.get(field) != identity[field]:
            raise ValueError(f"Deform360 selection lock {field} differs from the freeze")
    root = _mapping(selection.get("selection"), name="Deform360 selection.selection")
    source = _selection_records(root.get("calibration"), name="selection.calibration")
    target = _selection_records(root.get("confirmation"), name="selection.confirmation")
    expected_source = [
        {
            "object_id": group["object_id"],
            "episode_id": group["episode_id"],
            "stratum": group["stratum"],
        }
        for group in source_groups
    ]
    if source != expected_source:
        raise ValueError("Deform360 calibration selection differs from source_groups")
    if target != list(target_groups):
        raise ValueError("Deform360 confirmation selection differs from forbidden_target_groups")


def _selection_records(value: object, *, name: str) -> list[dict[str, Any]]:
    records = _sequence(value, name=name, nonempty=True)
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(records):
        mapping = _mapping(item, name=f"{name}[{index}]")
        object_id = _string(mapping.get("object_id"), name=f"{name}[{index}].object_id")
        episode_id = _integer(mapping.get("episode_id"), name=f"{name}[{index}].episode_id")
        stratum = _string(mapping.get("stratum"), name=f"{name}[{index}].stratum")
        key = (object_id, episode_id)
        if key in seen:
            raise ValueError(f"{name} repeats object/episode {key!r}")
        seen.add(key)
        result.append(
            {"object_id": object_id, "episode_id": episode_id, "stratum": stratum}
        )
    return sorted(result, key=lambda record: (record["object_id"], record["episode_id"]))
