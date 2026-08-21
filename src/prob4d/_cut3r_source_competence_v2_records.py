"""Exact paired metric-support validation for CUT3R source competence v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ._cut3r_source_competence_v2_common import (
    _RECORD_FIELDS,
    _RECORDS_FIELDS,
    RECORDS_SCHEMA,
    VERSION,
    _support,
)
from ._cut3r_source_competence_v2_lock import (
    validate_cut3r_source_competence_v2_lock,
)
from .cut3r_comparison import validate_cut3r_comparison_lock
from .cut3r_source_competence import (
    CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
    _canonical_json,
    _exact_keys,
    _normalize_records,
    _sha256,
    _strict_boolean,
    _strict_integer,
    _strict_mapping,
    _strict_string,
    validate_cut3r_source_competence_lock,
)


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        cast(str, record["group_id"]),
        cast(str, record["case_id"]),
        cast(int, record["frame_index"]),
        cast(int, record["random_seed"]),
        cast(str, record["arm_id"]),
    )


def _normalize_v2_records(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    payload = _strict_mapping(value, name="source competence v2 records")
    _exact_keys(payload, _RECORDS_FIELDS, name="source competence v2 records")
    if payload["schema"] != RECORDS_SCHEMA or payload["schema_version"] != VERSION:
        raise ValueError("unsupported source competence v2 records")
    expected_ids = {
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "common_support_definition_sha256": v2_lock[
            "common_support_definition_sha256"
        ],
    }
    for name, expected in expected_ids.items():
        if _sha256(payload[name], name=name) != expected:
            raise ValueError(f"v2 records use a different {name}")
    if not _strict_boolean(payload["source_truth_used"], name="source_truth_used"):
        raise ValueError("v2 records require source truth")
    if _strict_boolean(payload["target_payloads_opened"], name="target_payloads_opened"):
        raise ValueError("v2 records may not open target payloads")
    if _strict_boolean(payload["target_outcomes_opened"], name="target_outcomes_opened"):
        raise ValueError("v2 records may not open target outcomes")
    raw_records = payload["records"]
    if type(raw_records) is not list:
        raise ValueError("records must be a JSON array")
    supports: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    stripped_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        record = _strict_mapping(raw_record, name=f"records[{index}]")
        _exact_keys(record, _RECORD_FIELDS, name=f"records[{index}]")
        stripped = {name: record[name] for name in _RECORD_FIELDS if name != "metric_support"}
        key = (
            _strict_string(record["group_id"], name="group_id"),
            _strict_string(record["case_id"], name="case_id"),
            _strict_integer(record["frame_index"], name="frame_index"),
            _strict_integer(record["random_seed"], name="random_seed"),
            _strict_string(record["arm_id"], name="arm_id"),
        )
        if key in supports:
            raise ValueError("v2 records contain duplicate frame-arm keys")
        supports[key] = _support(
            record["metric_support"],
            seam_present=record["seam_error_m"] is not None,
        )
        stripped_records.append(stripped)
    v1_input = {
        "schema": CUT3R_SOURCE_COMPETENCE_RECORDS_SCHEMA,
        "schema_version": 1,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "record_definition_sha256": source_lock["record_definition_sha256"],
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": payload["group_failures"],
        "records": stripped_records,
    }
    normalized_v1 = _normalize_records(comparison, source_lock, v1_input)
    normalized_records: list[dict[str, Any]] = []
    for v1_record in cast(list[dict[str, Any]], normalized_v1["records"]):
        key = _record_key(v1_record)
        normalized_records.append({**v1_record, "metric_support": supports[key]})
    treatment = cast(str, v2_lock["contrast"]["treatment_arm"])
    control = cast(str, v2_lock["contrast"]["control_arm"])
    lookup = {_record_key(record): record for record in normalized_records}
    pair_keys = sorted({key[:-1] for key in lookup})
    for pair_key in pair_keys:
        candidate = lookup[(*pair_key, treatment)]
        baseline = lookup[(*pair_key, control)]
        if candidate["metric_support"] != baseline["metric_support"]:
            raise ValueError(
                "paired arms use different exact metric support "
                f"at {pair_key!r}"
            )
    normalized = {
        "schema": RECORDS_SCHEMA,
        "schema_version": VERSION,
        **expected_ids,
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "group_failures": normalized_v1["group_failures"],
        "records": normalized_records,
    }
    return (
        cast(dict[str, Any], json.loads(_canonical_json(normalized))),
        normalized_v1,
    )


def _safe_ratio(candidate: float, baseline: float) -> tuple[float | None, str | None]:
    if baseline > 0.0:
        return candidate / baseline, None
    if candidate == 0.0:
        return 1.0, None
    return None, "baseline-zero-with-positive-candidate"


def _group_pair(
    group_id: str,
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    policy: Mapping[str, float],
) -> dict[str, Any]:
    seam_ratio, seam_reason = _safe_ratio(
        candidate["seam_rmse_m"], baseline["seam_rmse_m"]
    )
    drift_ratio, drift_reason = _safe_ratio(
        candidate["absolute_drift_slope_m_per_frame"],
        baseline["absolute_drift_slope_m_per_frame"],
    )
    association_delta = candidate["association_precision"] - baseline[
        "association_precision"
    ]
    identity_delta = candidate["identity_retention"] - baseline["identity_retention"]
    support_delta = candidate["support_retention"] - baseline["support_retention"]
    quality_reasons = []
    identity_reasons = []
    if seam_reason:
        quality_reasons.append(f"seam-{seam_reason}")
    elif seam_ratio is not None and seam_ratio > policy[
        "maximum_worst_group_seam_rmse_ratio"
    ]:
        quality_reasons.append("seam-rmse-ratio-exceeded")
    if drift_reason:
        quality_reasons.append(f"drift-{drift_reason}")
    elif drift_ratio is not None and drift_ratio > policy[
        "maximum_worst_group_absolute_drift_slope_ratio"
    ]:
        quality_reasons.append("drift-slope-ratio-exceeded")
    for name, delta, threshold in (
        (
            "association-precision",
            association_delta,
            policy["minimum_worst_group_association_precision_delta"],
        ),
        (
            "identity-retention",
            identity_delta,
            policy["minimum_worst_group_identity_retention_delta"],
        ),
        (
            "support-retention",
            support_delta,
            policy["minimum_worst_group_support_retention_delta"],
        ),
    ):
        if delta < threshold:
            identity_reasons.append(f"{name}-delta-below-minimum")
    return {
        "group_id": group_id,
        "technical_failure_code": None,
        "candidate": dict(candidate),
        "baseline": dict(baseline),
        "paired": {
            "seam_rmse_ratio": seam_ratio,
            "absolute_drift_slope_ratio": drift_ratio,
            "association_precision_delta": association_delta,
            "identity_retention_delta": identity_delta,
            "support_retention_delta": support_delta,
        },
        "paired_quality_pass": not quality_reasons,
        "paired_identity_pass": not identity_reasons,
        "paired_quality_reasons": sorted(quality_reasons),
        "paired_identity_reasons": sorted(identity_reasons),
    }
