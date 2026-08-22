"""Independent canonical-row support manifests for CUT3R source metrics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._cut3r_source_competence_audit_common import (
    _ENTRY_FIELDS,
    _MANIFEST_FIELDS,
    _MANIFEST_INPUT_FIELDS,
    MANIFEST_SCHEMA,
    PROPER_SCORE_SEMANTICS,
    VERSION,
)
from ._cut3r_source_competence_audit_lock import (
    validate_cut3r_source_competence_audit_lock,
)
from ._cut3r_source_competence_v2_lock import (
    validate_cut3r_source_competence_v2_lock,
)
from .cut3r_comparison import validate_cut3r_comparison_lock
from .cut3r_source_competence import (
    _canonical_json,
    _exact_keys,
    _record_id,
    _sha256,
    _strict_boolean,
    _strict_integer,
    _strict_mapping,
    _strict_string,
    validate_cut3r_source_competence_lock,
)

EntryKey = tuple[str, str, int, int]


def _identity(value: Any, *, name: str) -> str | int:
    if type(value) is int and value >= 0:
        return cast(int, value)
    if type(value) is str:
        return _strict_string(value, name=name)
    raise ValueError(f"{name} must be a non-negative integer or exact string")


def _axis(value: Any, *, name: str) -> str | int:
    return _identity(value, name=name)


def _row_digest(rows: Sequence[Sequence[Any]]) -> str:
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _normalize_row(
    value: Any,
    *,
    name: str,
    expected_length: int,
    key: EntryKey,
    kind: str,
) -> list[Any]:
    if type(value) is not list or len(value) != expected_length:
        raise ValueError(f"{name} must be a canonical {expected_length}-element array")
    group_id = _strict_string(value[0], name=f"{name}[0]")
    case_id = _strict_string(value[1], name=f"{name}[1]")
    frame_index = _strict_integer(value[2], name=f"{name}[2]")
    if (group_id, case_id, frame_index) != key[:3]:
        raise ValueError(f"{name} does not match its enclosing group/case/frame")
    if kind == "point":
        return [
            group_id,
            case_id,
            frame_index,
            _identity(value[3], name=f"{name}[3]"),
            _strict_string(value[4], name=f"{name}[4]"),
        ]
    if kind == "endpoint":
        return [
            group_id,
            case_id,
            frame_index,
            _strict_string(value[3], name=f"{name}[3]"),
            _identity(value[4], name=f"{name}[4]"),
            _strict_string(value[5], name=f"{name}[5]"),
        ]
    if kind == "proper_score":
        return [
            group_id,
            case_id,
            frame_index,
            _identity(value[3], name=f"{name}[3]"),
            _axis(value[4], name=f"{name}[4]"),
            _strict_string(value[5], name=f"{name}[5]"),
        ]
    if kind == "seam":
        left = _strict_string(value[3], name=f"{name}[3]")
        right = _strict_string(value[4], name=f"{name}[4]")
        if left == right:
            raise ValueError(f"{name} must identify two distinct windows")
        return [
            group_id,
            case_id,
            frame_index,
            left,
            right,
            _identity(value[5], name=f"{name}[5]"),
            _strict_string(value[6], name=f"{name}[6]"),
        ]
    raise AssertionError(f"unknown support-row kind: {kind}")


def _normalize_rows(
    value: Any,
    *,
    name: str,
    expected_length: int,
    key: EntryKey,
    kind: str,
    allow_empty: bool,
) -> list[list[Any]]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    rows = [
        _normalize_row(
            item,
            name=f"{name}[{index}]",
            expected_length=expected_length,
            key=key,
            kind=kind,
        )
        for index, item in enumerate(value)
    ]
    encoded = [_canonical_json(row) for row in rows]
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{name} contains duplicate canonical rows")
    return rows


def _point_identity(row: Sequence[Any]) -> tuple[Any, ...]:
    return (row[0], row[1], row[2], row[3], row[4])


def _validate_row_relations(entry: Mapping[str, Any]) -> None:
    point_rows = cast(list[list[Any]], entry["point_rows"])
    endpoint_rows = cast(list[list[Any]], entry["endpoint_rows"])
    score_rows = cast(list[list[Any]], entry["proper_score_rows"])
    seam_rows = cast(list[list[Any]], entry["seam_rows"])
    points = {_point_identity(row) for row in point_rows}
    endpoint_points = {(row[0], row[1], row[2], row[4], row[5]) for row in endpoint_rows}
    seam_points = {(row[0], row[1], row[2], row[5], row[6]) for row in seam_rows}
    if not endpoint_points.issubset(points):
        raise ValueError("endpoint support references material points outside point support")
    if not seam_points.issubset(points):
        raise ValueError("seam support references material points outside point support")
    axes_by_point: dict[tuple[Any, ...], list[str | int]] = defaultdict(list)
    for row in score_rows:
        point = (row[0], row[1], row[2], row[3], row[5])
        if point not in points:
            raise ValueError("proper-score support references a point outside point support")
        axes_by_point[point].append(cast(str | int, row[4]))
    if set(axes_by_point) != points:
        raise ValueError("every point support row must contribute to the proper score")
    axis_signature: tuple[str, ...] | None = None
    for point, axes in axes_by_point.items():
        canonical_axes = tuple(sorted(_canonical_json(axis).decode("utf-8") for axis in axes))
        if len(canonical_axes) != 3 or len(set(canonical_axes)) != 3:
            raise ValueError(f"proper-score point {point!r} must have three distinct axes")
        if axis_signature is None:
            axis_signature = canonical_axes
        elif canonical_axes != axis_signature:
            raise ValueError("proper-score coordinate axes must be consistent across points")


def _normalize_entry(value: Any, *, index: int) -> dict[str, Any]:
    raw = _strict_mapping(value, name=f"entries[{index}]")
    _exact_keys(raw, _ENTRY_FIELDS, name=f"entries[{index}]")
    key: EntryKey = (
        _strict_string(raw["group_id"], name="group_id"),
        _strict_string(raw["case_id"], name="case_id"),
        _strict_integer(raw["frame_index"], name="frame_index"),
        _strict_integer(raw["random_seed"], name="random_seed"),
    )
    entry = {
        "group_id": key[0],
        "case_id": key[1],
        "frame_index": key[2],
        "random_seed": key[3],
        "point_rows": _normalize_rows(
            raw["point_rows"],
            name="point_rows",
            expected_length=5,
            key=key,
            kind="point",
            allow_empty=False,
        ),
        "endpoint_rows": _normalize_rows(
            raw["endpoint_rows"],
            name="endpoint_rows",
            expected_length=6,
            key=key,
            kind="endpoint",
            allow_empty=False,
        ),
        "proper_score_rows": _normalize_rows(
            raw["proper_score_rows"],
            name="proper_score_rows",
            expected_length=6,
            key=key,
            kind="proper_score",
            allow_empty=False,
        ),
        "seam_rows": _normalize_rows(
            raw["seam_rows"],
            name="seam_rows",
            expected_length=7,
            key=key,
            kind="seam",
            allow_empty=True,
        ),
    }
    _validate_row_relations(entry)
    return entry


def _expected_entry_keys(source_lock: Mapping[str, Any]) -> set[EntryKey]:
    seeds = cast(list[int], source_lock["random_seeds"])
    result: set[EntryKey] = set()
    for group in cast(list[dict[str, Any]], source_lock["source_evaluation_groups"]):
        group_id = cast(str, group["group_id"])
        for case in cast(list[dict[str, Any]], group["cases"]):
            case_id = cast(str, case["case_id"])
            start = cast(int, case["evaluation_frame_start"])
            stop = cast(int, case["evaluation_frame_stop_exclusive"])
            result.update(
                (group_id, case_id, frame, seed) for frame in range(start, stop) for seed in seeds
            )
    return result


def metric_support_from_manifest_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_entry(entry, index=0)
    point_rows = cast(list[list[Any]], normalized["point_rows"])
    endpoint_rows = cast(list[list[Any]], normalized["endpoint_rows"])
    score_rows = cast(list[list[Any]], normalized["proper_score_rows"])
    seam_rows = cast(list[list[Any]], normalized["seam_rows"])
    return {
        "point_support_sha256": _row_digest(point_rows),
        "point_support_count": len(point_rows),
        "endpoint_support_sha256": _row_digest(endpoint_rows),
        "endpoint_support_count": len(endpoint_rows),
        "proper_score_support_sha256": _row_digest(score_rows),
        "proper_score_dimension": len(score_rows),
        "proper_score_semantics": PROPER_SCORE_SEMANTICS,
        "seam_support_sha256": _row_digest(seam_rows) if seam_rows else None,
        "seam_support_count": len(seam_rows),
    }


def build_cut3r_metric_support_manifest(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    specification: Any,
) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_lock = validate_cut3r_source_competence_lock(
        comparison,
        source_competence_lock,
    )
    v2_lock = validate_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        common_support_lock,
    )
    audit = validate_cut3r_source_competence_audit_lock(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
    )
    spec = _strict_mapping(specification, name="metric support manifest input")
    _exact_keys(spec, _MANIFEST_INPUT_FIELDS, name="metric support manifest input")
    raw_entries = spec["entries"]
    if type(raw_entries) is not list:
        raise ValueError("entries must be a JSON array")
    entries = [_normalize_entry(value, index=index) for index, value in enumerate(raw_entries)]
    keys = [
        (
            cast(str, entry["group_id"]),
            cast(str, entry["case_id"]),
            cast(int, entry["frame_index"]),
            cast(int, entry["random_seed"]),
        )
        for entry in entries
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("metric support manifest contains duplicate entry keys")
    entries = [entry for _, entry in sorted(zip(keys, entries, strict=True))]
    if cast(bool, audit["require_complete_manifest_roster"]):
        expected = _expected_entry_keys(source_lock)
        actual = set(keys)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "metric support manifest does not match the frozen complete roster; "
                f"missing={missing}, extra={extra}"
            )
    source_truth = _strict_boolean(spec["source_truth_used"], name="source_truth_used")
    target_payloads = _strict_boolean(
        spec["target_payloads_opened"],
        name="target_payloads_opened",
    )
    target_outcomes = _strict_boolean(
        spec["target_outcomes_opened"],
        name="target_outcomes_opened",
    )
    if not source_truth:
        raise ValueError("metric support manifest requires source truth")
    if target_payloads or target_outcomes:
        raise ValueError("metric support manifest may not open target data")
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "support_audit_lock_id": audit["support_audit_lock_id"],
        "common_support_definition_sha256": v2_lock["common_support_definition_sha256"],
        "proper_score_reference_artifact_id": audit["proper_score_reference_artifact_id"],
        "proper_score_reference_sha256": audit["proper_score_reference_sha256"],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "entries": entries,
    }
    payload["metric_support_manifest_id"] = _record_id(payload)
    return validate_cut3r_metric_support_manifest(
        comparison,
        source_lock,
        v2_lock,
        audit,
        payload,
    )


def validate_cut3r_metric_support_manifest(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    value: Any,
) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_lock = validate_cut3r_source_competence_lock(
        comparison,
        source_competence_lock,
    )
    v2_lock = validate_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        common_support_lock,
    )
    audit = validate_cut3r_source_competence_audit_lock(
        comparison,
        source_lock,
        v2_lock,
        audit_lock,
    )
    payload = _strict_mapping(value, name="metric support manifest")
    _exact_keys(payload, _MANIFEST_FIELDS, name="metric support manifest")
    if payload["schema"] != MANIFEST_SCHEMA or payload["schema_version"] != VERSION:
        raise ValueError("unsupported metric support manifest")
    expected_ids = {
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "support_audit_lock_id": audit["support_audit_lock_id"],
        "common_support_definition_sha256": v2_lock["common_support_definition_sha256"],
        "proper_score_reference_artifact_id": audit["proper_score_reference_artifact_id"],
        "proper_score_reference_sha256": audit["proper_score_reference_sha256"],
    }
    for name, expected in expected_ids.items():
        if _sha256(payload[name], name=name) != expected:
            raise ValueError(f"metric support manifest uses a different {name}")
    spec = {
        "source_truth_used": payload["source_truth_used"],
        "target_payloads_opened": payload["target_payloads_opened"],
        "target_outcomes_opened": payload["target_outcomes_opened"],
        "entries": payload["entries"],
    }
    expected = build_cut3r_metric_support_manifest_unchecked(source_lock, audit, spec)
    if payload != expected:
        raise ValueError("metric support manifest changed from its bound rows")
    return cast(dict[str, Any], json.loads(_canonical_json(expected)))


def build_cut3r_metric_support_manifest_unchecked(
    source_lock: Mapping[str, Any],
    audit_lock: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    raw_entries = specification["entries"]
    if type(raw_entries) is not list:
        raise ValueError("entries must be a JSON array")
    entries = [_normalize_entry(value, index=index) for index, value in enumerate(raw_entries)]
    keys = [
        (
            cast(str, entry["group_id"]),
            cast(str, entry["case_id"]),
            cast(int, entry["frame_index"]),
            cast(int, entry["random_seed"]),
        )
        for entry in entries
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("metric support manifest contains duplicate entry keys")
    entries = [entry for _, entry in sorted(zip(keys, entries, strict=True))]
    if cast(bool, audit_lock["require_complete_manifest_roster"]):
        expected_keys = _expected_entry_keys(source_lock)
        if set(keys) != expected_keys:
            missing = sorted(expected_keys - set(keys))
            extra = sorted(set(keys) - expected_keys)
            raise ValueError(
                "metric support manifest does not match the frozen complete roster; "
                f"missing={missing}, extra={extra}"
            )
    if not _strict_boolean(
        specification["source_truth_used"],
        name="source_truth_used",
    ):
        raise ValueError("metric support manifest requires source truth")
    if _strict_boolean(
        specification["target_payloads_opened"],
        name="target_payloads_opened",
    ) or _strict_boolean(
        specification["target_outcomes_opened"],
        name="target_outcomes_opened",
    ):
        raise ValueError("metric support manifest may not open target data")
    result: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": audit_lock["comparison_lock_id"],
        "source_competence_lock_id": audit_lock["source_competence_lock_id"],
        "common_support_lock_id": audit_lock["common_support_lock_id"],
        "support_audit_lock_id": audit_lock["support_audit_lock_id"],
        "common_support_definition_sha256": audit_lock["common_support_definition_sha256"],
        "proper_score_reference_artifact_id": audit_lock["proper_score_reference_artifact_id"],
        "proper_score_reference_sha256": audit_lock["proper_score_reference_sha256"],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "entries": entries,
    }
    result["metric_support_manifest_id"] = _record_id(result)
    return result


__all__ = [
    "build_cut3r_metric_support_manifest",
    "metric_support_from_manifest_entry",
    "validate_cut3r_metric_support_manifest",
]
