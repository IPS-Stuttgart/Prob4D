"""Build source-only CUT3R competence evidence from complete paired frame records.

The comparison lock freezes arms, cases, evaluation frames, and random seeds. This
module closes the remaining execution gap by converting complete frame-level
records into one equal-seed, equal-case, equal-group
:class:`SourceProviderCompetenceReportV1`. Missing frames, seeds, arms, or groups
fail closed; a complete group may instead carry one predeclared technical-failure
code. Target payloads and outcomes remain forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._atomic_file import atomic_write_bytes
from .cut3r_comparison import (
    CUT3R_COMPARISON_GROUP_UNIT,
    load_cut3r_comparison_lock,
    validate_cut3r_comparison_lock,
)
from .source_provider_competence import (
    SourceProviderCompetencePolicyV1,
    SourceProviderCompetenceReportV1,
    SourceProviderGroupResultV1,
    load_source_provider_competence,
    write_source_provider_competence,
)

CUT3R_SOURCE_COMPETENCE_LOCK_SCHEMA: Final = "prob4d.cut3r-source-competence-lock"
CUT3R_SOURCE_COMPETENCE_LOCK_VERSION: Final = 1
CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA: Final = (
    "prob4d.cut3r-source-competence-records"
)
CUT3R_SOURCE_COMPETENCE_RECORDS_VERSION: Final = 1
CUT3R_SOURCE_COMPETENCE_CLAIM_BOUNDARY: Final = (
    "This source-only artifact aggregates the exact frozen CUT3R comparison into "
    "complete object/session competence evidence. It cannot change an arm, omit "
    "nested records, authorize target access, establish BayesianPhysTwin or "
    "Causal4D benefit, establish deployment safety, or establish state of the art."
)
CUT3R_SOURCE_COMPETENCE_WEIGHTING: Final = {
    "within_case": "equal-frame-mean-within-each-frozen-seed-v1",
    "within_group": "equal-seed-means-then-equal-case-means-v1",
    "across_groups": "equal-complete-group-mean-v1",
    "technical_failures": "retain-complete-group-without-scored-metrics-v1",
}

_LOCK_SPEC_FIELDS: Final = frozenset(
    {
        "contrast_id",
        "candidate_provider_manifest_id",
        "baseline_provider_manifest_id",
        "cohort_binding_id",
        "group_definition",
        "record_definition_sha256",
        "policy",
    }
)
_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "comparison_protocol_name",
        "group_unit",
        "source_evaluation_groups",
        "random_seeds",
        "contrast",
        "candidate_provider_manifest_id",
        "baseline_provider_manifest_id",
        "cohort_binding_id",
        "group_definition",
        "record_definition_sha256",
        "policy",
        "weighting",
        "source_truth_required",
        "source_access",
        "target_access",
        "claim_boundary",
        "source_competence_lock_id",
    }
)
_SOURCE_GROUP_FIELDS: Final = frozenset({"group_id", "cases"})
_SOURCE_CASE_FIELDS: Final = frozenset(
    {"case_id", "evaluation_frame_start", "evaluation_frame_stop_exclusive"}
)
_CONTRAST_FIELDS: Final = frozenset(
    {
        "contrast_id",
        "treatment_arm",
        "control_arm",
        "claim_eligible",
        "enabled",
    }
)
_RECORDS_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "record_definition_sha256",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "group_failures",
        "records",
    }
)
_FAILURE_FIELDS: Final = frozenset(
    {"group_id", "technical_failure_code", "metadata"}
)
_RECORD_FIELDS: Final = frozenset(
    {
        "group_id",
        "case_id",
        "frame_index",
        "random_seed",
        "arm_id",
        "point_error_m",
        "endpoint_error_m",
        "proper_score",
        "seam_error_m",
        "association_correct_count",
        "association_predicted_count",
        "identity_retained_count",
        "identity_reference_count",
        "support_retained_count",
        "support_reference_count",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with exact string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    keys = set(value)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise ValueError(f"{name} has noncanonical keys; missing={missing}, extra={extra}")


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty exact string without surrounding whitespace"
        )
    return cast(str, value)


def _strict_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a genuine integer >= {minimum}")
    return cast(int, value)


def _strict_boolean(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a genuine Boolean")
    return cast(bool, value)


def _finite_number(value: Any, *, name: str, nonnegative: bool) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a genuine finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_nonnegative(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name=name, nonnegative=True)


def _sha256(value: Any, *, name: str) -> str:
    digest = _strict_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _strict_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _finite_json(value: Any, *, name: str) -> Any:
    if value is None or type(value) in {bool, str}:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain non-finite numbers")
        return value
    if type(value) is list:
        return [
            _finite_json(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError(f"{name} must use exact string keys")
        return {
            key: _finite_json(value[key], name=f"{name}.{key}")
            for key in sorted(value)
        }
    raise ValueError(f"{name} contains an unsupported JSON value")


def _canonical_random_seeds(value: Any, *, name: str) -> list[int]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a nonempty JSON array")
    seeds = [
        _strict_integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    if seeds != sorted(set(seeds)):
        raise ValueError(f"{name} must be sorted and unique")
    return seeds


def _comparison_context(comparison_lock: Any) -> dict[str, Any]:
    comparison = validate_cut3r_comparison_lock(comparison_lock)
    source_group_ids = list(comparison["group_roles"]["source_evaluation"])
    source_group_set = set(source_group_ids)
    source_groups: list[dict[str, Any]] = []
    cases_by_group: dict[str, dict[str, tuple[int, int]]] = {}
    for raw_group in comparison["groups"]:
        group = cast(dict[str, Any], raw_group)
        group_id = cast(str, group["group_id"])
        if group_id not in source_group_set:
            continue
        cases: list[dict[str, Any]] = []
        case_lookup: dict[str, tuple[int, int]] = {}
        for raw_case in group["cases"]:
            case = cast(dict[str, Any], raw_case)
            case_id = cast(str, case["case_id"])
            start = cast(int, case["evaluation_frame_start"])
            stop = cast(int, case["evaluation_frame_stop_exclusive"])
            cases.append(
                {
                    "case_id": case_id,
                    "evaluation_frame_start": start,
                    "evaluation_frame_stop_exclusive": stop,
                }
            )
            case_lookup[case_id] = (start, stop)
        source_groups.append({"group_id": group_id, "cases": cases})
        cases_by_group[group_id] = case_lookup
    source_groups_by_id = {group["group_id"]: group for group in source_groups}
    normalized_source_groups = [source_groups_by_id[group_id] for group_id in source_group_ids]
    contrasts = {
        cast(str, contrast["contrast_id"]): cast(dict[str, Any], contrast)
        for contrast in comparison["registered_contrasts"]
    }
    arms = {
        cast(str, arm["arm_id"]): cast(dict[str, Any], arm)
        for arm in comparison["arms"]
    }
    return {
        "comparison": comparison,
        "source_group_ids": source_group_ids,
        "source_groups": normalized_source_groups,
        "cases_by_group": cases_by_group,
        "contrasts": contrasts,
        "arms": arms,
    }


def _canonical_source_groups(value: Any, *, name: str) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a nonempty JSON array")
    groups: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_cases: set[str] = set()
    for group_index, raw_group in enumerate(value):
        group = _strict_mapping(raw_group, name=f"{name}[{group_index}]")
        _exact_keys(group, _SOURCE_GROUP_FIELDS, name=f"{name}[{group_index}]")
        group_id = _strict_string(group["group_id"], name="group_id")
        if group_id in seen_groups:
            raise ValueError(f"duplicate source group: {group_id!r}")
        seen_groups.add(group_id)
        raw_cases = group["cases"]
        if type(raw_cases) is not list or not raw_cases:
            raise ValueError(f"source group {group_id!r} must contain cases")
        cases: list[dict[str, Any]] = []
        for case_index, raw_case in enumerate(raw_cases):
            case = _strict_mapping(
                raw_case,
                name=f"{name}[{group_index}].cases[{case_index}]",
            )
            _exact_keys(
                case,
                _SOURCE_CASE_FIELDS,
                name=f"{name}[{group_index}].cases[{case_index}]",
            )
            case_id = _strict_string(case["case_id"], name="case_id")
            if case_id in seen_cases:
                raise ValueError(f"duplicate source case: {case_id!r}")
            seen_cases.add(case_id)
            start = _strict_integer(
                case["evaluation_frame_start"],
                name="evaluation_frame_start",
            )
            stop = _strict_integer(
                case["evaluation_frame_stop_exclusive"],
                name="evaluation_frame_stop_exclusive",
                minimum=1,
            )
            if stop <= start:
                raise ValueError("evaluation frame interval must be nonempty")
            cases.append(
                {
                    "case_id": case_id,
                    "evaluation_frame_start": start,
                    "evaluation_frame_stop_exclusive": stop,
                }
            )
        groups.append({"group_id": group_id, "cases": cases})
    return groups


def _canonical_contrast(value: Any) -> dict[str, Any]:
    contrast = _strict_mapping(value, name="CUT3R source competence contrast")
    _exact_keys(contrast, _CONTRAST_FIELDS, name="CUT3R source competence contrast")
    return {
        "contrast_id": _strict_string(contrast["contrast_id"], name="contrast_id"),
        "treatment_arm": _strict_string(
            contrast["treatment_arm"],
            name="treatment_arm",
        ),
        "control_arm": _strict_string(contrast["control_arm"], name="control_arm"),
        "claim_eligible": _strict_boolean(
            contrast["claim_eligible"],
            name="claim_eligible",
        ),
        "enabled": _strict_boolean(contrast["enabled"], name="enabled"),
    }


def build_cut3r_source_competence_lock(
    comparison_lock: Any,
    specification: Any,
) -> dict[str, Any]:
    """Freeze one claim-eligible source contrast and its competence policy."""

    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    spec = _strict_mapping(specification, name="CUT3R source competence specification")
    _exact_keys(spec, _LOCK_SPEC_FIELDS, name="CUT3R source competence specification")
    contrast_id = _strict_string(spec["contrast_id"], name="contrast_id")
    contrasts = cast(dict[str, dict[str, Any]], context["contrasts"])
    if contrast_id not in contrasts:
        raise ValueError(f"unknown CUT3R registered contrast: {contrast_id!r}")
    contrast = _canonical_contrast(contrasts[contrast_id])
    if not contrast["enabled"] or not contrast["claim_eligible"]:
        raise ValueError("source competence requires an enabled claim-eligible contrast")
    arms = cast(dict[str, dict[str, Any]], context["arms"])
    for field in ("treatment_arm", "control_arm"):
        arm = arms[cast(str, contrast[field])]
        if not arm["enabled"] or not arm["causal"] or not arm["claim_eligible"]:
            raise ValueError("source competence arms must be enabled, causal, and claim eligible")
    policy = SourceProviderCompetencePolicyV1.from_dict(spec["policy"])
    source_group_ids = cast(list[str], context["source_group_ids"])
    if policy.minimum_evaluable_groups > len(source_group_ids):
        raise ValueError(
            "minimum_evaluable_groups exceeds the frozen source-evaluation roster"
        )
    candidate_provider_manifest_id = _sha256(
        spec["candidate_provider_manifest_id"],
        name="candidate_provider_manifest_id",
    )
    baseline_provider_manifest_id = _sha256(
        spec["baseline_provider_manifest_id"],
        name="baseline_provider_manifest_id",
    )
    if candidate_provider_manifest_id == baseline_provider_manifest_id:
        raise ValueError("candidate and baseline provider manifests must be distinct")
    payload: dict[str, Any] = {
        "schema": CUT3R_SOURCE_COMPETENCE_LOCK_SCHEMA,
        "schema_version": CUT3R_SOURCE_COMPETENCE_LOCK_VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "comparison_protocol_name": comparison["protocol_name"],
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "source_evaluation_groups": context["source_groups"],
        "random_seeds": comparison["random_seeds"],
        "contrast": contrast,
        "candidate_provider_manifest_id": candidate_provider_manifest_id,
        "baseline_provider_manifest_id": baseline_provider_manifest_id,
        "cohort_binding_id": _sha256(
            spec["cohort_binding_id"],
            name="cohort_binding_id",
        ),
        "group_definition": _strict_string(
            spec["group_definition"],
            name="group_definition",
        ),
        "record_definition_sha256": _sha256(
            spec["record_definition_sha256"],
            name="record_definition_sha256",
        ),
        "policy": policy.to_dict(),
        "weighting": dict(CUT3R_SOURCE_COMPETENCE_WEIGHTING),
        "source_truth_required": True,
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": CUT3R_SOURCE_COMPETENCE_CLAIM_BOUNDARY,
    }
    payload["source_competence_lock_id"] = _record_id(payload)
    return validate_cut3r_source_competence_lock(comparison, payload)


def validate_cut3r_source_competence_lock(
    comparison_lock: Any,
    value: Any,
) -> dict[str, Any]:
    """Validate a source-competence lock and exact comparison binding."""

    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    payload = _strict_mapping(value, name="CUT3R source competence lock")
    _exact_keys(payload, _LOCK_FIELDS, name="CUT3R source competence lock")
    if _strict_string(payload["schema"], name="schema") != (
        CUT3R_SOURCE_COMPETENCE_LOCK_SCHEMA
    ):
        raise ValueError("unsupported CUT3R source competence lock schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_SOURCE_COMPETENCE_LOCK_VERSION
    ):
        raise ValueError("unsupported CUT3R source competence lock version")
    comparison_lock_id = _sha256(payload["comparison_lock_id"], name="comparison_lock_id")
    if comparison_lock_id != comparison["lock_id"]:
        raise ValueError("source competence lock is bound to a different comparison")
    protocol_name = _strict_string(
        payload["comparison_protocol_name"],
        name="comparison_protocol_name",
    )
    if protocol_name != comparison["protocol_name"]:
        raise ValueError("source competence protocol name changed from the comparison")
    if _strict_string(payload["group_unit"], name="group_unit") != (
        CUT3R_COMPARISON_GROUP_UNIT
    ):
        raise ValueError("source competence must use complete object/session groups")
    source_groups = _canonical_source_groups(
        payload["source_evaluation_groups"],
        name="source_evaluation_groups",
    )
    if source_groups != context["source_groups"]:
        raise ValueError("source competence roster changed from the comparison lock")
    random_seeds = _canonical_random_seeds(payload["random_seeds"], name="random_seeds")
    if random_seeds != comparison["random_seeds"]:
        raise ValueError("source competence seeds changed from the comparison lock")
    contrast = _canonical_contrast(payload["contrast"])
    contrasts = cast(dict[str, dict[str, Any]], context["contrasts"])
    contrast_id = cast(str, contrast["contrast_id"])
    if contrast_id not in contrasts or contrast != _canonical_contrast(contrasts[contrast_id]):
        raise ValueError("source competence contrast changed from the comparison lock")
    if not contrast["enabled"] or not contrast["claim_eligible"]:
        raise ValueError("source competence contrast must remain claim eligible")
    policy = SourceProviderCompetencePolicyV1.from_dict(payload["policy"])
    source_group_ids = cast(list[str], context["source_group_ids"])
    if policy.minimum_evaluable_groups > len(source_group_ids):
        raise ValueError(
            "minimum_evaluable_groups exceeds the frozen source-evaluation roster"
        )
    candidate_provider_manifest_id = _sha256(
        payload["candidate_provider_manifest_id"],
        name="candidate_provider_manifest_id",
    )
    baseline_provider_manifest_id = _sha256(
        payload["baseline_provider_manifest_id"],
        name="baseline_provider_manifest_id",
    )
    if candidate_provider_manifest_id == baseline_provider_manifest_id:
        raise ValueError("candidate and baseline provider manifests must be distinct")
    arms = cast(dict[str, dict[str, Any]], context["arms"])
    for field in ("treatment_arm", "control_arm"):
        arm = arms[cast(str, contrast[field])]
        if not arm["enabled"] or not arm["causal"] or not arm["claim_eligible"]:
            raise ValueError(
                "source competence arms must be enabled, causal, and claim eligible"
            )
    if payload["weighting"] != CUT3R_SOURCE_COMPETENCE_WEIGHTING:
        raise ValueError("source competence weighting changed")
    if not _strict_boolean(payload["source_truth_required"], name="source_truth_required"):
        raise ValueError("source competence scoring requires source truth")
    if _strict_string(payload["source_access"], name="source_access") != "source-only":
        raise ValueError("source competence lock must remain source-only")
    if _strict_string(payload["target_access"], name="target_access") != "forbidden":
        raise ValueError("source competence lock cannot authorize target access")
    claim_boundary = _strict_string(payload["claim_boundary"], name="claim_boundary")
    if claim_boundary != CUT3R_SOURCE_COMPETENCE_CLAIM_BOUNDARY:
        raise ValueError("source competence claim boundary changed")
    lock_id = _sha256(
        payload["source_competence_lock_id"],
        name="source_competence_lock_id",
    )
    normalized: dict[str, Any] = {
        "schema": CUT3R_SOURCE_COMPETENCE_LOCK_SCHEMA,
        "schema_version": CUT3R_SOURCE_COMPETENCE_LOCK_VERSION,
        "comparison_lock_id": comparison_lock_id,
        "comparison_protocol_name": protocol_name,
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "source_evaluation_groups": source_groups,
        "random_seeds": random_seeds,
        "contrast": contrast,
        "candidate_provider_manifest_id": candidate_provider_manifest_id,
        "baseline_provider_manifest_id": baseline_provider_manifest_id,
        "cohort_binding_id": _sha256(
            payload["cohort_binding_id"],
            name="cohort_binding_id",
        ),
        "group_definition": _strict_string(
            payload["group_definition"],
            name="group_definition",
        ),
        "record_definition_sha256": _sha256(
            payload["record_definition_sha256"],
            name="record_definition_sha256",
        ),
        "policy": policy.to_dict(),
        "weighting": dict(CUT3R_SOURCE_COMPETENCE_WEIGHTING),
        "source_truth_required": True,
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": claim_boundary,
        "source_competence_lock_id": lock_id,
    }
    unsigned = dict(normalized)
    unsigned.pop("source_competence_lock_id")
    if lock_id != _record_id(unsigned):
        raise ValueError("source_competence_lock_id does not match canonical content")
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def load_cut3r_source_competence_lock(
    comparison_lock: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_source_competence_lock(comparison_lock, _strict_json(path))


def _publish_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    load_existing: Any,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("artifact destination must not be a symbolic link")
    encoded = _canonical_json(payload) + b"\n"
    try:
        atomic_write_bytes(destination, encoded, overwrite=False)
    except FileExistsError:
        existing = load_existing(destination)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different artifact: {destination}"
            ) from None
        return existing
    return dict(payload)


def write_cut3r_source_competence_lock(
    comparison_lock: Any,
    path: str | Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_source_competence_lock(comparison_lock, lock)
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_source_competence_lock(
            comparison_lock,
            existing,
        ),
    )


def _normalize_failure(value: Any, *, index: int) -> dict[str, Any]:
    failure = _strict_mapping(value, name=f"group_failures[{index}]")
    _exact_keys(failure, _FAILURE_FIELDS, name=f"group_failures[{index}]")
    metadata = _strict_mapping(failure["metadata"], name="failure metadata")
    return {
        "group_id": _strict_string(failure["group_id"], name="group_id"),
        "technical_failure_code": _strict_string(
            failure["technical_failure_code"],
            name="technical_failure_code",
        ),
        "metadata": _finite_json(dict(metadata), name="failure metadata"),
    }


def _normalize_record(value: Any, *, index: int) -> dict[str, Any]:
    record = _strict_mapping(value, name=f"records[{index}]")
    _exact_keys(record, _RECORD_FIELDS, name=f"records[{index}]")
    normalized = {
        "group_id": _strict_string(record["group_id"], name="group_id"),
        "case_id": _strict_string(record["case_id"], name="case_id"),
        "frame_index": _strict_integer(record["frame_index"], name="frame_index"),
        "random_seed": _strict_integer(record["random_seed"], name="random_seed"),
        "arm_id": _strict_string(record["arm_id"], name="arm_id"),
        "point_error_m": _finite_number(
            record["point_error_m"],
            name="point_error_m",
            nonnegative=True,
        ),
        "endpoint_error_m": _finite_number(
            record["endpoint_error_m"],
            name="endpoint_error_m",
            nonnegative=True,
        ),
        "proper_score": _finite_number(
            record["proper_score"],
            name="proper_score",
            nonnegative=False,
        ),
        "seam_error_m": _optional_nonnegative(
            record["seam_error_m"],
            name="seam_error_m",
        ),
        "association_correct_count": _strict_integer(
            record["association_correct_count"],
            name="association_correct_count",
        ),
        "association_predicted_count": _strict_integer(
            record["association_predicted_count"],
            name="association_predicted_count",
        ),
        "identity_retained_count": _strict_integer(
            record["identity_retained_count"],
            name="identity_retained_count",
        ),
        "identity_reference_count": _strict_integer(
            record["identity_reference_count"],
            name="identity_reference_count",
        ),
        "support_retained_count": _strict_integer(
            record["support_retained_count"],
            name="support_retained_count",
        ),
        "support_reference_count": _strict_integer(
            record["support_reference_count"],
            name="support_reference_count",
        ),
    }
    if normalized["association_correct_count"] > normalized["association_predicted_count"]:
        raise ValueError("association_correct_count exceeds association_predicted_count")
    if normalized["identity_retained_count"] > normalized["identity_reference_count"]:
        raise ValueError("identity_retained_count exceeds identity_reference_count")
    if normalized["support_retained_count"] > normalized["support_reference_count"]:
        raise ValueError("support_retained_count exceeds support_reference_count")
    return normalized


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        cast(str, record["group_id"]),
        cast(str, record["case_id"]),
        cast(int, record["frame_index"]),
        cast(int, record["random_seed"]),
        cast(str, record["arm_id"]),
    )


def _normalize_records(
    comparison_lock: Any,
    source_competence_lock: Any,
    value: Any,
) -> dict[str, Any]:
    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    lock = validate_cut3r_source_competence_lock(comparison, source_competence_lock)
    payload = _strict_mapping(value, name="CUT3R source competence records")
    _exact_keys(payload, _RECORDS_FIELDS, name="CUT3R source competence records")
    if _strict_string(payload["schema"], name="schema") != (
        CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA
    ):
        raise ValueError("unsupported CUT3R source competence records schema")
    if (
        _strict_integer(payload["schema_version"], name="schema_version", minimum=1)
        != CUT3R_SOURCE_COMPETENCE_RECORDS_VERSION
    ):
        raise ValueError("unsupported CUT3R source competence records version")
    if _sha256(payload["comparison_lock_id"], name="comparison_lock_id") != (
        comparison["lock_id"]
    ):
        raise ValueError("source competence records use a different comparison lock")
    if _sha256(
        payload["source_competence_lock_id"],
        name="source_competence_lock_id",
    ) != lock["source_competence_lock_id"]:
        raise ValueError("source competence records use a different competence lock")
    if _sha256(
        payload["record_definition_sha256"],
        name="record_definition_sha256",
    ) != lock["record_definition_sha256"]:
        raise ValueError("source competence records use a different record definition")
    if not _strict_boolean(payload["source_truth_used"], name="source_truth_used"):
        raise ValueError("source competence records require declared source truth")
    if _strict_boolean(payload["target_payloads_opened"], name="target_payloads_opened"):
        raise ValueError("source competence records may not open target payloads")
    if _strict_boolean(payload["target_outcomes_opened"], name="target_outcomes_opened"):
        raise ValueError("source competence records may not open target outcomes")

    raw_failures = payload["group_failures"]
    if type(raw_failures) is not list:
        raise ValueError("group_failures must be a JSON array")
    failures = [_normalize_failure(item, index=index) for index, item in enumerate(raw_failures)]
    failures.sort(key=lambda item: cast(str, item["group_id"]))
    failure_ids = [cast(str, item["group_id"]) for item in failures]
    if len(failure_ids) != len(set(failure_ids)):
        raise ValueError("group_failures contain duplicate groups")
    source_group_ids = cast(list[str], context["source_group_ids"])
    unknown_failures = sorted(set(failure_ids) - set(source_group_ids))
    if unknown_failures:
        raise ValueError(f"group_failures contain unknown source groups: {unknown_failures}")

    raw_records = payload["records"]
    if type(raw_records) is not list:
        raise ValueError("records must be a JSON array")
    records = [_normalize_record(item, index=index) for index, item in enumerate(raw_records)]
    records.sort(key=_record_key)
    keys = [_record_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("source competence records contain duplicate frame-arm keys")

    cases_by_group = cast(dict[str, dict[str, tuple[int, int]]], context["cases_by_group"])
    seeds = cast(list[int], lock["random_seeds"])
    contrast = cast(dict[str, Any], lock["contrast"])
    candidate_arm = cast(str, contrast["treatment_arm"])
    baseline_arm = cast(str, contrast["control_arm"])
    allowed_arms = {candidate_arm, baseline_arm}
    failed_group_set = set(failure_ids)
    for record in records:
        group_id = cast(str, record["group_id"])
        case_id = cast(str, record["case_id"])
        frame_index = cast(int, record["frame_index"])
        random_seed = cast(int, record["random_seed"])
        arm_id = cast(str, record["arm_id"])
        if group_id not in cases_by_group:
            raise ValueError(f"record uses unknown source group: {group_id!r}")
        if group_id in failed_group_set:
            raise ValueError("technical-failure groups must not contain scored records")
        if case_id not in cases_by_group[group_id]:
            raise ValueError(f"record uses case {case_id!r} outside group {group_id!r}")
        start, stop = cases_by_group[group_id][case_id]
        if not start <= frame_index < stop:
            raise ValueError("record frame lies outside the frozen evaluation interval")
        if random_seed not in seeds:
            raise ValueError(f"record uses unfrozen random seed: {random_seed}")
        if arm_id not in allowed_arms:
            raise ValueError(f"record uses an arm outside the frozen contrast: {arm_id!r}")

    expected_keys: set[tuple[str, str, int, int, str]] = set()
    for group_id in source_group_ids:
        if group_id in failed_group_set:
            continue
        for case_id, (start, stop) in cases_by_group[group_id].items():
            if stop - start < 2:
                raise ValueError("source competence drift requires at least two evaluation frames")
            for frame_index in range(start, stop):
                for random_seed in seeds:
                    expected_keys.add(
                        (group_id, case_id, frame_index, random_seed, candidate_arm)
                    )
                    expected_keys.add(
                        (group_id, case_id, frame_index, random_seed, baseline_arm)
                    )
    actual_keys = set(keys)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            "source competence records do not match the frozen complete paired roster; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    lookup = {_record_key(record): record for record in records}
    for group_id, case_id, frame_index, random_seed, _ in sorted(expected_keys):
        pair_key = (group_id, case_id, frame_index, random_seed)
        candidate = lookup[(*pair_key, candidate_arm)]
        baseline = lookup[(*pair_key, baseline_arm)]
        for field in ("identity_reference_count", "support_reference_count"):
            if candidate[field] != baseline[field]:
                raise ValueError(
                    f"paired arms use different arm-neutral {field} at {pair_key!r}"
                )

    by_case_seed_arm: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case_seed_arm[
            (
                cast(str, record["group_id"]),
                cast(str, record["case_id"]),
                cast(int, record["random_seed"]),
                cast(str, record["arm_id"]),
            )
        ].append(record)
    for key, nested in by_case_seed_arm.items():
        if sum(cast(int, item["association_predicted_count"]) for item in nested) <= 0:
            raise ValueError(f"association denominator is zero for {key!r}")
        if sum(cast(int, item["identity_reference_count"]) for item in nested) <= 0:
            raise ValueError(f"identity denominator is zero for {key!r}")
        if sum(cast(int, item["support_reference_count"]) for item in nested) <= 0:
            raise ValueError(f"support denominator is zero for {key!r}")
        if not any(item["seam_error_m"] is not None for item in nested):
            raise ValueError(f"seam observations are missing for {key!r}")

    covered_groups = failed_group_set | {cast(str, record["group_id"]) for record in records}
    if covered_groups != set(source_group_ids):
        missing_groups = sorted(set(source_group_ids) - covered_groups)
        raise ValueError(f"source groups were silently omitted: {missing_groups}")
    normalized: dict[str, Any] = {
        "schema": CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
        "schema_version": CUT3R_SOURCE_COMPETENCE_RECORDS_VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": lock["source_competence_lock_id"],
        "record_definition_sha256": lock["record_definition_sha256"],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": failures,
        "records": records,
    }
    return cast(dict[str, Any], json.loads(_canonical_json(normalized)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _absolute_slope(frames: Sequence[int], values: Sequence[float]) -> float:
    if len(frames) != len(values) or len(frames) < 2:
        raise ValueError("drift slope requires at least two paired values")
    mean_frame = _mean([float(frame) for frame in frames])
    mean_value = _mean(values)
    denominator = math.fsum((float(frame) - mean_frame) ** 2 for frame in frames)
    if denominator <= 0.0:
        raise ValueError("drift slope frame support is singular")
    numerator = math.fsum(
        (float(frame) - mean_frame) * (value - mean_value)
        for frame, value in zip(frames, values, strict=True)
    )
    return abs(numerator / denominator)


def _case_seed_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ordered = sorted(records, key=lambda item: cast(int, item["frame_index"]))
    point_errors = [cast(float, item["point_error_m"]) for item in ordered]
    endpoint_errors = [cast(float, item["endpoint_error_m"]) for item in ordered]
    seam_errors = [
        cast(float, item["seam_error_m"])
        for item in ordered
        if item["seam_error_m"] is not None
    ]
    association_denominator = sum(
        cast(int, item["association_predicted_count"]) for item in ordered
    )
    identity_denominator = sum(
        cast(int, item["identity_reference_count"]) for item in ordered
    )
    support_denominator = sum(
        cast(int, item["support_reference_count"]) for item in ordered
    )
    return {
        "point_mse_m2": _mean([value * value for value in point_errors]),
        "endpoint_mse_m2": _mean([value * value for value in endpoint_errors]),
        "proper_score": _mean([cast(float, item["proper_score"]) for item in ordered]),
        "seam_mse_m2": _mean([value * value for value in seam_errors]),
        "absolute_drift_slope_m_per_frame": _absolute_slope(
            [cast(int, item["frame_index"]) for item in ordered],
            point_errors,
        ),
        "association_precision": (
            sum(cast(int, item["association_correct_count"]) for item in ordered)
            / association_denominator
        ),
        "identity_retention": (
            sum(cast(int, item["identity_retained_count"]) for item in ordered)
            / identity_denominator
        ),
        "support_retention": (
            sum(cast(int, item["support_retained_count"]) for item in ordered)
            / support_denominator
        ),
    }


def _hierarchical_group_metrics(
    group_id: str,
    arm_id: str,
    *,
    cases: Mapping[str, tuple[int, int]],
    seeds: Sequence[int],
    record_lookup: Mapping[tuple[str, str, int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    case_metrics: list[dict[str, float]] = []
    for case_id in cases:
        seed_metrics = [
            _case_seed_metrics(record_lookup[(group_id, case_id, seed, arm_id)])
            for seed in seeds
        ]
        case_metrics.append(
            {
                metric: _mean([item[metric] for item in seed_metrics])
                for metric in seed_metrics[0]
            }
        )
    group = {
        metric: _mean([item[metric] for item in case_metrics])
        for metric in case_metrics[0]
    }
    return {
        "proper_score": group["proper_score"],
        "point_rmse_m": math.sqrt(group["point_mse_m2"]),
        "endpoint_rmse_m": math.sqrt(group["endpoint_mse_m2"]),
        "seam_rmse_m": math.sqrt(group["seam_mse_m2"]),
        "absolute_drift_slope_m_per_frame": group[
            "absolute_drift_slope_m_per_frame"
        ],
        "association_precision": group["association_precision"],
        "identity_retention": group["identity_retention"],
        "support_retention": group["support_retention"],
    }


def build_cut3r_source_competence_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    records: Any,
) -> SourceProviderCompetenceReportV1:
    """Aggregate complete paired records into the existing competence contract."""

    context = _comparison_context(comparison_lock)
    comparison = cast(dict[str, Any], context["comparison"])
    lock = validate_cut3r_source_competence_lock(comparison, source_competence_lock)
    normalized_records = _normalize_records(comparison, lock, records)
    records_id = _record_id(normalized_records)
    failures = {
        cast(str, item["group_id"]): cast(dict[str, Any], item)
        for item in normalized_records["group_failures"]
    }
    contrast = cast(dict[str, Any], lock["contrast"])
    candidate_arm = cast(str, contrast["treatment_arm"])
    baseline_arm = cast(str, contrast["control_arm"])
    seeds = cast(list[int], lock["random_seeds"])
    cases_by_group = cast(dict[str, dict[str, tuple[int, int]]], context["cases_by_group"])
    nested: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in normalized_records["records"]:
        item = cast(dict[str, Any], record)
        nested[
            (
                cast(str, item["group_id"]),
                cast(str, item["case_id"]),
                cast(int, item["random_seed"]),
                cast(str, item["arm_id"]),
            )
        ].append(item)

    group_results: list[SourceProviderGroupResultV1] = []
    for group_id in cast(list[str], context["source_group_ids"]):
        binding_metadata = {
            "case_ids": list(cases_by_group[group_id]),
            "random_seeds": seeds,
            "comparison_lock_id": comparison["lock_id"],
            "source_competence_lock_id": lock["source_competence_lock_id"],
            "records_id": records_id,
            "contrast_id": contrast["contrast_id"],
            "candidate_arm": candidate_arm,
            "baseline_arm": baseline_arm,
        }
        if group_id in failures:
            failure = failures[group_id]
            group_results.append(
                SourceProviderGroupResultV1(
                    group_id=group_id,
                    candidate_proper_score=None,
                    baseline_proper_score=None,
                    candidate_point_rmse_m=None,
                    baseline_point_rmse_m=None,
                    candidate_endpoint_rmse_m=None,
                    baseline_endpoint_rmse_m=None,
                    absolute_drift_slope_m_per_frame=None,
                    seam_rmse_m=None,
                    association_precision=None,
                    identity_retention=None,
                    support_retention=None,
                    technical_failure_code=cast(str, failure["technical_failure_code"]),
                    metadata={
                        "binding": binding_metadata,
                        "failure": failure["metadata"],
                    },
                )
            )
            continue
        candidate = _hierarchical_group_metrics(
            group_id,
            candidate_arm,
            cases=cases_by_group[group_id],
            seeds=seeds,
            record_lookup=nested,
        )
        baseline = _hierarchical_group_metrics(
            group_id,
            baseline_arm,
            cases=cases_by_group[group_id],
            seeds=seeds,
            record_lookup=nested,
        )
        group_results.append(
            SourceProviderGroupResultV1(
                group_id=group_id,
                candidate_proper_score=candidate["proper_score"],
                baseline_proper_score=baseline["proper_score"],
                candidate_point_rmse_m=candidate["point_rmse_m"],
                baseline_point_rmse_m=baseline["point_rmse_m"],
                candidate_endpoint_rmse_m=candidate["endpoint_rmse_m"],
                baseline_endpoint_rmse_m=baseline["endpoint_rmse_m"],
                absolute_drift_slope_m_per_frame=candidate[
                    "absolute_drift_slope_m_per_frame"
                ],
                seam_rmse_m=candidate["seam_rmse_m"],
                association_precision=candidate["association_precision"],
                identity_retention=candidate["identity_retention"],
                support_retention=candidate["support_retention"],
                metadata={"binding": binding_metadata},
            )
        )

    return SourceProviderCompetenceReportV1(
        provider_manifest_id=cast(str, lock["candidate_provider_manifest_id"]),
        cohort_binding_id=cast(str, lock["cohort_binding_id"]),
        group_definition=cast(str, lock["group_definition"]),
        policy=SourceProviderCompetencePolicyV1.from_dict(lock["policy"]),
        groups=tuple(group_results),
        source_truth_used=True,
        target_payloads_opened=False,
        target_outcomes_opened=False,
        metadata={
            "comparison_lock_id": comparison["lock_id"],
            "source_competence_lock_id": lock["source_competence_lock_id"],
            "records_id": records_id,
            "record_definition_sha256": lock["record_definition_sha256"],
            "contrast_id": contrast["contrast_id"],
            "candidate_arm": candidate_arm,
            "baseline_arm": baseline_arm,
            "baseline_provider_manifest_id": lock["baseline_provider_manifest_id"],
            "weighting": dict(CUT3R_SOURCE_COMPETENCE_WEIGHTING),
            "frame_arm_record_count": len(normalized_records["records"]),
            "technical_failure_group_count": len(normalized_records["group_failures"]),
        },
    )


def validate_cut3r_source_competence_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    records: Any,
    report: SourceProviderCompetenceReportV1,
) -> SourceProviderCompetenceReportV1:
    if not isinstance(report, SourceProviderCompetenceReportV1):
        raise TypeError("report must be SourceProviderCompetenceReportV1")
    expected = build_cut3r_source_competence_report(
        comparison_lock,
        source_competence_lock,
        records,
    )
    if report.to_dict() != expected.to_dict():
        raise ValueError("source competence report does not match the bound records")
    return expected


def load_cut3r_source_competence_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    records: Any,
    path: str | Path,
) -> SourceProviderCompetenceReportV1:
    return validate_cut3r_source_competence_report(
        comparison_lock,
        source_competence_lock,
        records,
        load_source_provider_competence(path),
    )


def write_cut3r_source_competence_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    records: Any,
    path: str | Path,
    report: SourceProviderCompetenceReportV1,
) -> SourceProviderCompetenceReportV1:
    payload = validate_cut3r_source_competence_report(
        comparison_lock,
        source_competence_lock,
        records,
        report,
    )
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("artifact destination must not be a symbolic link")
    try:
        write_source_provider_competence(destination, payload)
    except FileExistsError:
        existing = load_cut3r_source_competence_report(
            comparison_lock,
            source_competence_lock,
            records,
            destination,
        )
        if existing.to_dict() != payload.to_dict():
            raise FileExistsError(
                f"refusing to replace a different artifact: {destination}"
            ) from None
        return existing
    return payload


def cut3r_source_competence_summary(
    report: SourceProviderCompetenceReportV1,
) -> dict[str, Any]:
    if not isinstance(report, SourceProviderCompetenceReportV1):
        raise TypeError("report must be SourceProviderCompetenceReportV1")
    metadata = cast(Mapping[str, Any], report.metadata)
    return {
        "source_provider_competence_id": report.source_provider_competence_id,
        "source_competence_lock_id": metadata["source_competence_lock_id"],
        "records_id": metadata["records_id"],
        "contrast_id": metadata["contrast_id"],
        "candidate_arm": metadata["candidate_arm"],
        "baseline_arm": metadata["baseline_arm"],
        "group_count": report.group_count,
        "evaluable_group_count": report.evaluable_group_count,
        "technical_failure_count": report.technical_failure_count,
        "mean_quality_status": report.mean_quality_status,
        "identity_reliability_status": report.identity_reliability_status,
        "source_competence_pass": report.source_competence_pass,
        "mean_quality_reasons": list(report.mean_quality_reasons),
        "identity_reliability_reasons": list(report.identity_reliability_reasons),
        "target_access": "forbidden",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction cut3r-source-competence",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze a source competence lock")
    freeze.add_argument("comparison_lock")
    freeze.add_argument("specification")
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=_freeze_command)

    verify_lock = subparsers.add_parser("verify-lock", help="verify a competence lock")
    verify_lock.add_argument("comparison_lock")
    verify_lock.add_argument("source_competence_lock")
    verify_lock.set_defaults(handler=_verify_lock_command)

    report = subparsers.add_parser("report", help="build the bound competence report")
    report.add_argument("comparison_lock")
    report.add_argument("source_competence_lock")
    report.add_argument("records")
    report.add_argument("--output", required=True)
    report.add_argument("--require-pass", action="store_true")
    report.set_defaults(handler=_report_command)

    verify_report = subparsers.add_parser("verify-report", help="rebuild and verify a report")
    verify_report.add_argument("comparison_lock")
    verify_report.add_argument("source_competence_lock")
    verify_report.add_argument("records")
    verify_report.add_argument("report")
    verify_report.add_argument("--require-pass", action="store_true")
    verify_report.set_defaults(handler=_verify_report_command)

    summarize = subparsers.add_parser("summarize", help="summarize a verified report")
    summarize.add_argument("comparison_lock")
    summarize.add_argument("source_competence_lock")
    summarize.add_argument("records")
    summarize.add_argument("report")
    summarize.add_argument("--json", action="store_true")
    summarize.set_defaults(handler=_summarize_command)
    return parser


def _freeze_command(arguments: argparse.Namespace) -> int:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    lock = build_cut3r_source_competence_lock(
        comparison,
        _strict_json(arguments.specification),
    )
    write_cut3r_source_competence_lock(comparison, arguments.output, lock)
    print(lock["source_competence_lock_id"])
    return 0


def _verify_lock_command(arguments: argparse.Namespace) -> int:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    lock = load_cut3r_source_competence_lock(
        comparison,
        arguments.source_competence_lock,
    )
    print(lock["source_competence_lock_id"])
    return 0


def _report_command(arguments: argparse.Namespace) -> int:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    lock = load_cut3r_source_competence_lock(
        comparison,
        arguments.source_competence_lock,
    )
    records = _strict_json(arguments.records)
    report = build_cut3r_source_competence_report(comparison, lock, records)
    write_cut3r_source_competence_report(
        comparison,
        lock,
        records,
        arguments.output,
        report,
    )
    print(report.source_provider_competence_id)
    return 0 if report.source_competence_pass or not arguments.require_pass else 3


def _verify_report_command(arguments: argparse.Namespace) -> int:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    lock = load_cut3r_source_competence_lock(
        comparison,
        arguments.source_competence_lock,
    )
    records = _strict_json(arguments.records)
    report = load_cut3r_source_competence_report(
        comparison,
        lock,
        records,
        arguments.report,
    )
    print(report.source_provider_competence_id)
    return 0 if report.source_competence_pass or not arguments.require_pass else 3


def _summarize_command(arguments: argparse.Namespace) -> int:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    lock = load_cut3r_source_competence_lock(
        comparison,
        arguments.source_competence_lock,
    )
    records = _strict_json(arguments.records)
    report = load_cut3r_source_competence_report(
        comparison,
        lock,
        records,
        arguments.report,
    )
    summary = cut3r_source_competence_summary(report)
    if arguments.json:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(f"report_id: {summary['source_provider_competence_id']}")
        print(f"contrast: {summary['contrast_id']}")
        print(f"mean quality: {summary['mean_quality_status']}")
        print(f"identity/reliability: {summary['identity_reliability_status']}")
        print(f"source competence pass: {summary['source_competence_pass']}")
        print(f"target access: {summary['target_access']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


__all__ = [
    "CUT3R_SOURCE_COMPETENCE_CLAIM_BOUNDARY",
    "CUT3R_SOURCE_COMPETENCE_LOCK_SCHEMA",
    "CUT3R_SOURCE_COMPETENCE_LOCK_VERSION",
    "CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA",
    "CUT3R_SOURCE_COMPETENCE_RECORDS_VERSION",
    "CUT3R_SOURCE_COMPETENCE_WEIGHTING",
    "build_cut3r_source_competence_lock",
    "build_cut3r_source_competence_report",
    "cut3r_source_competence_summary",
    "load_cut3r_source_competence_lock",
    "load_cut3r_source_competence_report",
    "main",
    "validate_cut3r_source_competence_lock",
    "validate_cut3r_source_competence_report",
    "write_cut3r_source_competence_lock",
    "write_cut3r_source_competence_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
