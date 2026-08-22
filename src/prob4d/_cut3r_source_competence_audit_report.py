"""Replay-complete audited CUT3R source-competence receipts and gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ._cut3r_source_competence_audit_common import (
    _REPORT_FIELDS,
    CLAIM_BOUNDARY,
    PROPER_SCORE_REFERENCE_FIT_SCOPE,
    PROPER_SCORE_SEMANTICS,
    REPORT_SCHEMA,
    VERSION,
    Status,
)
from ._cut3r_source_competence_audit_lock import (
    validate_cut3r_source_competence_audit_lock,
)
from ._cut3r_source_competence_audit_manifest import (
    metric_support_from_manifest_entry,
    validate_cut3r_metric_support_manifest,
)
from ._cut3r_source_competence_v2_lock import (
    validate_cut3r_source_competence_v2_lock,
)
from ._cut3r_source_competence_v2_records import _normalize_v2_records
from ._cut3r_source_competence_v2_report import (
    validate_cut3r_source_competence_v2_report,
)
from .cut3r_comparison import validate_cut3r_comparison_lock
from .cut3r_source_competence import (
    _canonical_json,
    _exact_keys,
    _record_id,
    _sha256,
    _strict_boolean,
    _strict_mapping,
    _strict_string,
    validate_cut3r_source_competence_lock,
)
from .fresh_provider_readiness import ReadinessGateV1

PairKey = tuple[str, str, int, int]
RecordKey = tuple[str, str, int, int, str]
_STATUS_VALUES = {"pass", "fail", "technical-failure", "not-evaluated"}


def _pair_key(value: Mapping[str, Any]) -> PairKey:
    return (
        cast(str, value["group_id"]),
        cast(str, value["case_id"]),
        cast(int, value["frame_index"]),
        cast(int, value["random_seed"]),
    )


def _record_key(value: Mapping[str, Any]) -> RecordKey:
    return (*_pair_key(value), cast(str, value["arm_id"]))


def _status(value: Any, *, name: str) -> Status:
    result = _strict_string(value, name=name)
    if result not in _STATUS_VALUES:
        raise ValueError(f"{name} is not a supported readiness status")
    return cast(Status, result)


def _reasons(value: Any, *, name: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    reasons = [_strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if reasons != sorted(set(reasons)):
        raise ValueError(f"{name} must be sorted and duplicate-free")
    return reasons


def build_cut3r_source_competence_support_audit_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    records: Any,
    metric_support_manifest: Any,
    source_competence_report_v2: Any,
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
    manifest = validate_cut3r_metric_support_manifest(
        comparison,
        source_lock,
        v2_lock,
        audit,
        metric_support_manifest,
    )
    normalized_records, _ = _normalize_v2_records(
        comparison,
        source_lock,
        v2_lock,
        records,
    )
    v2_report = validate_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
        source_competence_report_v2,
    )
    treatment = cast(str, v2_lock["contrast"]["treatment_arm"])
    control = cast(str, v2_lock["contrast"]["control_arm"])
    records_by_key = {
        _record_key(record): record
        for record in cast(list[dict[str, Any]], normalized_records["records"])
    }
    entries_by_key = {
        _pair_key(entry): entry
        for entry in cast(list[dict[str, Any]], manifest["entries"])
    }
    record_pairs = {key[:-1] for key in records_by_key}
    manifest_pairs = set(entries_by_key)
    if manifest_pairs != record_pairs:
        missing = sorted(record_pairs - manifest_pairs)
        extra = sorted(manifest_pairs - record_pairs)
        raise ValueError(
            "metric support manifest and scored records use different pair rosters; "
            f"missing={missing}, extra={extra}"
        )
    for pair_key in sorted(record_pairs):
        computed = metric_support_from_manifest_entry(entries_by_key[pair_key])
        for arm_id in (treatment, control):
            record = records_by_key[(*pair_key, arm_id)]
            if record["metric_support"] != computed:
                raise ValueError(
                    "independently reconstructed metric support disagrees with "
                    f"the {arm_id!r} scored record at {pair_key!r}"
                )
    mean_status = _status(v2_report["mean_quality_status"], name="mean_quality_status")
    identity_status = _status(
        v2_report["identity_reliability_status"],
        name="identity_reliability_status",
    )
    mean_reasons = _reasons(v2_report["mean_quality_reasons"], name="mean_quality_reasons")
    identity_reasons = _reasons(
        v2_report["identity_reliability_reasons"],
        name="identity_reliability_reasons",
    )
    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "support_audit_lock_id": audit["support_audit_lock_id"],
        "records_id": _record_id(normalized_records),
        "metric_support_manifest_id": manifest["metric_support_manifest_id"],
        "source_competence_report_v2_id": v2_report[
            "source_competence_report_v2_id"
        ],
        "proper_score_semantics": audit["proper_score_semantics"],
        "proper_score_reference_artifact_id": audit[
            "proper_score_reference_artifact_id"
        ],
        "proper_score_reference_sha256": audit["proper_score_reference_sha256"],
        "proper_score_reference_fit_scope": audit[
            "proper_score_reference_fit_scope"
        ],
        "verified_pair_count": len(record_pairs),
        "support_manifest_status": "pass",
        "proper_score_reference_binding_status": "pass",
        "mean_quality_status": mean_status,
        "identity_reliability_status": identity_status,
        "mean_quality_reasons": mean_reasons,
        "identity_reliability_reasons": identity_reasons,
        "audited_source_competence_pass": bool(v2_report["source_competence_pass"]),
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["source_competence_support_audit_report_id"] = _record_id(payload)
    return validate_cut3r_source_competence_support_audit_report_content(payload)


def validate_cut3r_source_competence_support_audit_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    records: Any,
    metric_support_manifest: Any,
    source_competence_report_v2: Any,
    report: Any,
) -> dict[str, Any]:
    supplied = validate_cut3r_source_competence_support_audit_report_content(report)
    expected = build_cut3r_source_competence_support_audit_report(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        audit_lock,
        records,
        metric_support_manifest,
        source_competence_report_v2,
    )
    if supplied != expected:
        raise ValueError("support audit report does not match its bound evidence")
    return expected


def validate_cut3r_source_competence_support_audit_report_content(
    value: Any,
) -> dict[str, Any]:
    payload = _strict_mapping(value, name="source competence support audit report")
    _exact_keys(payload, _REPORT_FIELDS, name="source competence support audit report")
    if payload["schema"] != REPORT_SCHEMA or payload["schema_version"] != VERSION:
        raise ValueError("unsupported source competence support audit report")
    for name in (
        "comparison_lock_id",
        "source_competence_lock_id",
        "common_support_lock_id",
        "support_audit_lock_id",
        "records_id",
        "metric_support_manifest_id",
        "source_competence_report_v2_id",
        "proper_score_reference_artifact_id",
        "proper_score_reference_sha256",
    ):
        _sha256(payload[name], name=name)
    if type(payload["verified_pair_count"]) is not int or payload["verified_pair_count"] < 1:
        raise ValueError("verified_pair_count must be a genuine positive integer")
    if payload["proper_score_semantics"] != PROPER_SCORE_SEMANTICS:
        raise ValueError("support audit report proper-score semantics changed")
    if (
        payload["proper_score_reference_fit_scope"]
        != PROPER_SCORE_REFERENCE_FIT_SCOPE
    ):
        raise ValueError("support audit report reference fit scope changed")
    if payload["support_manifest_status"] != "pass":
        raise ValueError("support audit reports require a passing reconstructed manifest")
    if payload["proper_score_reference_binding_status"] != "pass":
        raise ValueError("support audit reports require a passing reference binding")
    mean_status = _status(payload["mean_quality_status"], name="mean_quality_status")
    identity_status = _status(
        payload["identity_reliability_status"],
        name="identity_reliability_status",
    )
    mean_reasons = _reasons(payload["mean_quality_reasons"], name="mean_quality_reasons")
    identity_reasons = _reasons(
        payload["identity_reliability_reasons"],
        name="identity_reliability_reasons",
    )
    if mean_status == "pass" and mean_reasons:
        raise ValueError("passing mean status may not carry failure reasons")
    if mean_status != "pass" and not mean_reasons:
        raise ValueError("nonpassing mean status requires failure reasons")
    if mean_status != "pass" and identity_status != "not-evaluated":
        raise ValueError("identity must remain not-evaluated after a mean failure")
    if identity_status == "pass" and identity_reasons:
        raise ValueError("passing identity status may not carry failure reasons")
    if identity_status == "not-evaluated" and identity_reasons:
        raise ValueError("not-evaluated identity status may not carry reasons")
    if identity_status in {"fail", "technical-failure"} and not identity_reasons:
        raise ValueError("nonpassing identity status requires failure reasons")
    expected_pass = mean_status == "pass" and identity_status == "pass"
    actual_pass = _strict_boolean(
        payload["audited_source_competence_pass"],
        name="audited_source_competence_pass",
    )
    if actual_pass != expected_pass:
        raise ValueError("audited source-competence pass flag disagrees with statuses")
    if not _strict_boolean(payload["source_truth_used"], name="source_truth_used"):
        raise ValueError("support audit report requires source truth")
    if _strict_boolean(payload["target_payloads_opened"], name="target_payloads_opened"):
        raise ValueError("support audit report may not open target payloads")
    if _strict_boolean(payload["target_outcomes_opened"], name="target_outcomes_opened"):
        raise ValueError("support audit report may not open target outcomes")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("support audit report claim boundary changed")
    report_id = _sha256(
        payload["source_competence_support_audit_report_id"],
        name="source_competence_support_audit_report_id",
    )
    unsigned = dict(payload)
    unsigned.pop("source_competence_support_audit_report_id")
    if _record_id(unsigned) != report_id:
        raise ValueError("support audit report content identity changed")
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def source_competence_gates_audited(
    report: Mapping[str, Any],
) -> tuple[ReadinessGateV1, ReadinessGateV1]:
    artifact = validate_cut3r_source_competence_support_audit_report_content(report)
    report_id = cast(str, artifact["source_competence_support_audit_report_id"])
    metadata = {
        "common_support_lock_id": artifact["common_support_lock_id"],
        "support_audit_lock_id": artifact["support_audit_lock_id"],
        "records_id": artifact["records_id"],
        "metric_support_manifest_id": artifact["metric_support_manifest_id"],
        "source_competence_report_v2_id": artifact[
            "source_competence_report_v2_id"
        ],
        "proper_score_semantics": artifact["proper_score_semantics"],
        "proper_score_reference_artifact_id": artifact[
            "proper_score_reference_artifact_id"
        ],
        "proper_score_reference_sha256": artifact[
            "proper_score_reference_sha256"
        ],
        "proper_score_reference_fit_scope": artifact[
            "proper_score_reference_fit_scope"
        ],
    }
    mean_status = cast(Status, artifact["mean_quality_status"])
    mean_gate = ReadinessGateV1(
        gate_name="source-mean",
        status=cast(Any, mean_status),
        evidence_id=report_id,
        reason_codes=(
            ()
            if mean_status == "pass"
            else tuple(cast(list[str], artifact["mean_quality_reasons"]))
        ),
        metadata=metadata,
    )
    if mean_status != "pass":
        return mean_gate, ReadinessGateV1(
            gate_name="identity-reliability",
            status="not-evaluated",
            evidence_id=None,
        )
    identity_status = cast(Status, artifact["identity_reliability_status"])
    return mean_gate, ReadinessGateV1(
        gate_name="identity-reliability",
        status=cast(Any, identity_status),
        evidence_id=report_id,
        reason_codes=(
            ()
            if identity_status == "pass"
            else tuple(cast(list[str], artifact["identity_reliability_reasons"]))
        ),
        metadata=metadata,
    )


__all__ = [
    "build_cut3r_source_competence_support_audit_report",
    "source_competence_gates_audited",
    "validate_cut3r_source_competence_support_audit_report",
    "validate_cut3r_source_competence_support_audit_report_content",
]
