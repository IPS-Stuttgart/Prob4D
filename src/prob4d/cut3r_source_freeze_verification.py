"""Portable verification of retained CUT3R Deform360 source-freeze artifacts.

The verifier checks the complete internal content address and cross-record
invariants without source data. Optional protocol, Deform360 selection-lock, and
comparison-spec files bind the retained artifact to the frozen external inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ._cut3r_source_freeze_verify_bindings import (
    _comparison_spec,
    _validate_protocol,
    _validate_selection,
)
from ._cut3r_source_freeze_verify_common import (
    SOURCE_FREEZE_SCHEMA,
    SOURCE_FREEZE_VERSION,
    SUPPORT_NEGATIVE,
    SUPPORT_PASS,
    _BASE_FIELDS,
    _INFORMATION_BOUNDARY,
    _PASS_FIELDS,
    _PROB4D_FIELDS,
    _PROVIDER_FIELDS,
    _SELECTION_IDENTITY_FIELDS,
    _basename,
    _boolean,
    _content_id,
    _exact_fields,
    _file_identity,
    _integer,
    _load_json_object,
    _mapping,
    _revision,
    _sequence,
    _sha256,
    _sha256_json,
    _string,
)
from ._cut3r_source_freeze_verify_structure import (
    _calibration_inputs,
    _camera_panel,
    _source_case,
    _source_groups,
    _support,
    _target_groups,
)


def validate_source_freeze(
    value: object,
    *,
    comparison_spec: object | None = None,
    protocol: object | None = None,
    protocol_bytes: bytes | None = None,
    selection: object | None = None,
    selection_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate one source-freeze artifact and optional exact external bindings."""

    freeze = _mapping(value, name="CUT3R Deform360 source freeze")
    decision = _string(freeze.get("decision"), name="decision")
    if decision not in {SUPPORT_PASS, SUPPORT_NEGATIVE}:
        raise ValueError(f"unsupported source-freeze decision: {decision!r}")
    _exact_fields(
        freeze,
        _PASS_FIELDS if decision == SUPPORT_PASS else _BASE_FIELDS,
        name="CUT3R Deform360 source freeze",
    )
    if freeze["schema"] != SOURCE_FREEZE_SCHEMA:
        raise ValueError("unsupported CUT3R Deform360 source-freeze schema")
    schema_version = _integer(
        freeze["schema_version"],
        name="schema_version",
        minimum=1,
    )
    if schema_version != SOURCE_FREEZE_VERSION:
        raise ValueError("unsupported CUT3R Deform360 source-freeze version")
    protocol_name = _string(freeze["protocol_name"], name="protocol_name")

    source_protocol = _file_identity(freeze["source_protocol"], name="source_protocol")
    selection_identity_raw = _mapping(
        freeze["deform360_selection"],
        name="deform360_selection",
    )
    _exact_fields(
        selection_identity_raw,
        _SELECTION_IDENTITY_FIELDS,
        name="deform360_selection",
    )
    selection_identity = {
        "sha256": _sha256(selection_identity_raw["sha256"], name="deform360_selection.sha256"),
        "byte_count": _integer(
            selection_identity_raw["byte_count"],
            name="deform360_selection.byte_count",
            minimum=1,
        ),
        "selection_artifact_sha256": _sha256(
            selection_identity_raw["selection_artifact_sha256"],
            name="deform360_selection.selection_artifact_sha256",
        ),
        "selection_sha256": _sha256(
            selection_identity_raw["selection_sha256"],
            name="deform360_selection.selection_sha256",
        ),
    }

    provider_raw = _mapping(freeze["provider"], name="provider")
    _exact_fields(provider_raw, _PROVIDER_FIELDS, name="provider")
    provider = {
        "repository": _string(provider_raw["repository"], name="provider.repository"),
        "revision": _revision(provider_raw["revision"], name="provider.revision"),
        "checkpoint_filename": _basename(
            provider_raw["checkpoint_filename"],
            name="provider.checkpoint_filename",
        ),
        "checkpoint_sha256": _sha256(
            provider_raw["checkpoint_sha256"],
            name="provider.checkpoint_sha256",
        ),
        "checkpoint_byte_count": _integer(
            provider_raw["checkpoint_byte_count"],
            name="provider.checkpoint_byte_count",
            minimum=1,
        ),
        "execution_mode": _string(
            provider_raw["execution_mode"],
            name="provider.execution_mode",
        ),
        "revisit_count": _integer(
            provider_raw["revisit_count"],
            name="provider.revisit_count",
        ),
        "global_alignment": _boolean(
            provider_raw["global_alignment"],
            name="provider.global_alignment",
        ),
        "second_pass_allowed": _boolean(
            provider_raw["second_pass_allowed"],
            name="provider.second_pass_allowed",
        ),
    }
    if provider["repository"] != "CUT3R/CUT3R":
        raise ValueError("provider.repository must remain CUT3R/CUT3R for schema version 1")
    if (
        provider["execution_mode"] != "recurrent-online"
        or provider["revisit_count"] != 1
        or provider["global_alignment"] is not False
        or provider["second_pass_allowed"] is not False
    ):
        raise ValueError("provider causal execution declarations changed for schema version 1")

    prob4d_raw = _mapping(freeze["prob4d"], name="prob4d")
    _exact_fields(prob4d_raw, _PROB4D_FIELDS, name="prob4d")
    prob4d = {
        "revision": _revision(prob4d_raw["revision"], name="prob4d.revision"),
        "distribution_filename": _basename(
            prob4d_raw["distribution_filename"],
            name="prob4d.distribution_filename",
        ),
        "distribution_sha256": _sha256(
            prob4d_raw["distribution_sha256"],
            name="prob4d.distribution_sha256",
        ),
        "distribution_byte_count": _integer(
            prob4d_raw["distribution_byte_count"],
            name="prob4d.distribution_byte_count",
            minimum=1,
        ),
    }

    source_groups = _source_groups(freeze["source_groups"])
    target_groups = _target_groups(freeze["forbidden_target_groups"])
    if _integer(freeze["source_group_count"], name="source_group_count", minimum=1) != len(
        source_groups
    ):
        raise ValueError("source_group_count differs from source_groups")
    if _integer(
        freeze["forbidden_target_group_count"],
        name="forbidden_target_group_count",
        minimum=1,
    ) != len(target_groups):
        raise ValueError("forbidden_target_group_count differs from forbidden_target_groups")
    source_keys = {(group["object_id"], group["episode_id"]) for group in source_groups}
    target_keys = {(group["object_id"], group["episode_id"]) for group in target_groups}
    if source_keys.intersection(target_keys):
        raise ValueError("source_groups and forbidden_target_groups overlap")
    groups_by_id = {cast(str, group["group_id"]): group for group in source_groups}

    support, support_rows = _support(freeze["support"], groups_by_id=groups_by_id)
    expected_decision = (
        SUPPORT_PASS
        if support["common_supported_camera_count"] >= support["minimum_common_supported_cameras"]
        else SUPPORT_NEGATIVE
    )
    if decision != expected_decision:
        raise ValueError("decision disagrees with reconstructed common-camera support")
    calibration_inputs = _calibration_inputs(
        freeze["camera_calibration_inputs"],
        groups_by_id=groups_by_id,
    )

    source_cases: list[dict[str, Any]] = []
    camera_panel: dict[str, Any] | None = None
    declared_comparison_sha: str | None = None
    if decision == SUPPORT_NEGATIVE:
        if freeze["camera_panel"] is not None:
            raise ValueError("a support-negative source freeze must not contain a camera panel")
        if freeze["source_cases"] != []:
            raise ValueError("a support-negative source freeze must not contain source cases")
        if comparison_spec is not None:
            raise ValueError("a support-negative source freeze cannot bind a comparison spec")
    else:
        camera_panel = _camera_panel(
            freeze["camera_panel"],
            common_cameras=cast(Sequence[str], support["common_supported_cameras"]),
        )
        raw_cases = _sequence(freeze["source_cases"], name="source_cases", nonempty=True)
        selected = set(cast(Sequence[str], camera_panel["selected_cameras"]))
        seen_cases: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for index, raw_case in enumerate(raw_cases):
            source_case = _source_case(
                raw_case,
                index=index,
                groups_by_id=groups_by_id,
                selected_cameras=selected,
                frame_stop=cast(Sequence[int], support["required_frame_interval"])[1],
                support_rows=support_rows,
            )
            if source_case["case_id"] in seen_cases:
                raise ValueError(f"source_cases repeats case_id {source_case['case_id']!r}")
            pair = (cast(str, source_case["group_id"]), cast(str, source_case["camera"]))
            if pair in seen_pairs:
                raise ValueError(f"source_cases repeats group/camera {pair!r}")
            seen_cases.add(cast(str, source_case["case_id"]))
            seen_pairs.add(pair)
            source_cases.append(source_case)
        if source_cases != sorted(source_cases, key=lambda record: record["case_id"]):
            raise ValueError("source_cases must use canonical case_id ordering")
        expected_pairs = {
            (group_id, camera) for group_id in groups_by_id for camera in selected
        }
        if seen_pairs != expected_pairs:
            missing = sorted(expected_pairs - seen_pairs)
            extra = sorted(seen_pairs - expected_pairs)
            raise ValueError(
                "source_cases must cover the full source-group/camera panel; "
                f"missing={missing}, extra={extra}"
            )
        declared_comparison_sha = _sha256(
            freeze["comparison_spec_sha256"],
            name="comparison_spec_sha256",
        )

    information = _mapping(freeze["information_boundary"], name="information_boundary")
    if dict(information) != _INFORMATION_BOUNDARY:
        raise ValueError("information_boundary changed from the schema-v1 no-access contract")
    claim_boundary = _string(freeze["claim_boundary"], name="claim_boundary")

    declared_id = _sha256(freeze["source_freeze_id"], name="source_freeze_id")
    measured_id = _content_id(freeze, id_field="source_freeze_id")
    if declared_id != measured_id:
        raise ValueError("source_freeze_id does not match the exact artifact content")

    normalized: dict[str, Any] = {
        "schema": SOURCE_FREEZE_SCHEMA,
        "schema_version": SOURCE_FREEZE_VERSION,
        "protocol_name": protocol_name,
        "decision": decision,
        "source_protocol": source_protocol,
        "deform360_selection": selection_identity,
        "provider": provider,
        "prob4d": prob4d,
        "source_group_count": len(source_groups),
        "source_groups": source_groups,
        "forbidden_target_group_count": len(target_groups),
        "forbidden_target_groups": target_groups,
        "support": support,
        "camera_calibration_inputs": calibration_inputs,
        "camera_panel": camera_panel,
        "source_cases": source_cases,
        "information_boundary": dict(_INFORMATION_BOUNDARY),
        "claim_boundary": claim_boundary,
    }
    if declared_comparison_sha is not None:
        normalized["comparison_spec_sha256"] = declared_comparison_sha
    normalized["source_freeze_id"] = declared_id
    if normalized != dict(freeze):
        raise ValueError("source freeze is internally valid but not in canonical form")

    normalized_comparison: dict[str, Any] | None = None
    if comparison_spec is not None:
        normalized_comparison = _comparison_spec(
            comparison_spec,
            freeze=normalized,
            source_groups=source_groups,
            source_cases=source_cases,
            frame_interval=cast(Sequence[int], support["required_frame_interval"]),
        )
        measured_comparison_sha = _sha256_json(normalized_comparison)
        if measured_comparison_sha != declared_comparison_sha:
            raise ValueError("comparison specification bytes/content differ from the freeze digest")
        if normalized_comparison != comparison_spec:
            raise ValueError("comparison specification is valid but not in canonical form")

    if protocol is not None:
        if protocol_bytes is None:
            raise ValueError("protocol bytes are required with a protocol object")
        _validate_protocol(
            _mapping(protocol, name="source protocol"),
            protocol_bytes=protocol_bytes,
            freeze=normalized,
            source_groups=source_groups,
            target_groups=target_groups,
            comparison=normalized_comparison,
        )
    if selection is not None:
        if selection_bytes is None:
            raise ValueError("selection bytes are required with a selection object")
        _validate_selection(
            _mapping(selection, name="Deform360 selection lock"),
            selection_bytes=selection_bytes,
            freeze=normalized,
            source_groups=source_groups,
            target_groups=target_groups,
        )
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_freeze", type=Path)
    parser.add_argument("--comparison-spec", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--require-complete-bindings",
        action="store_true",
        help=(
            "require protocol and selection bytes, plus a comparison spec for "
            "support-positive artifacts"
        ),
    )
    parser.add_argument(
        "--require-support-pass",
        action="store_true",
        help="return exit status 3 for a valid support-negative artifact",
    )
    parser.add_argument("--json", action="store_true", help="print a deterministic JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        freeze, _ = _load_json_object(arguments.source_freeze, name="source-freeze artifact")
        comparison = None
        if arguments.comparison_spec is not None:
            comparison, _ = _load_json_object(
                arguments.comparison_spec,
                name="CUT3R comparison specification",
            )
        protocol = None
        protocol_bytes = None
        if arguments.protocol is not None:
            protocol, protocol_bytes = _load_json_object(
                arguments.protocol,
                name="source protocol",
            )
        selection = None
        selection_bytes = None
        if arguments.selection is not None:
            selection, selection_bytes = _load_json_object(
                arguments.selection,
                name="Deform360 selection lock",
            )
        validated = validate_source_freeze(
            freeze,
            comparison_spec=comparison,
            protocol=protocol,
            protocol_bytes=protocol_bytes,
            selection=selection,
            selection_bytes=selection_bytes,
        )
        complete_bindings = bool(
            protocol is not None
            and selection is not None
            and (
                validated["decision"] == SUPPORT_NEGATIVE
                or comparison is not None
            )
        )
        if arguments.require_complete_bindings and not complete_bindings:
            raise ValueError(
                "complete verification requires --protocol and --selection, plus "
                "--comparison-spec for a support-positive artifact"
            )
    except ValueError as error:
        print(f"CUT3R Deform360 source-freeze verification failed: {error}", file=sys.stderr)
        return 2

    summary = {
        "source_freeze_id": validated["source_freeze_id"],
        "decision": validated["decision"],
        "source_group_count": validated["source_group_count"],
        "forbidden_target_group_count": validated["forbidden_target_group_count"],
        "common_supported_camera_count": validated["support"][
            "common_supported_camera_count"
        ],
        "selected_camera_count": (
            0 if validated["camera_panel"] is None else validated["camera_panel"]["panel_size"]
        ),
        "source_case_count": len(validated["source_cases"]),
        "bindings_verified": {
            "protocol": protocol is not None,
            "selection": selection is not None,
            "comparison_spec": comparison is not None,
        },
        "complete_bindings_verified": complete_bindings,
    }
    if arguments.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(summary["source_freeze_id"])
    if arguments.require_support_pass and validated["decision"] == SUPPORT_NEGATIVE:
        return 3
    return 0


__all__ = [
    "SOURCE_FREEZE_SCHEMA",
    "SOURCE_FREEZE_VERSION",
    "SUPPORT_NEGATIVE",
    "SUPPORT_PASS",
    "main",
    "validate_source_freeze",
]


if __name__ == "__main__":
    raise SystemExit(main())
