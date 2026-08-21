"""Paired source-competence reporting and readiness adaptation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._cut3r_source_competence_v2_common import (
    CLAIM_BOUNDARY,
    REPORT_SCHEMA,
    VERSION,
    WEIGHTING,
    Status,
    _REPORT_FIELDS,
)
from ._cut3r_source_competence_v2_lock import (
    validate_cut3r_source_competence_v2_lock,
)
from ._cut3r_source_competence_v2_records import (
    _group_pair,
    _normalize_v2_records,
)
from .cut3r_comparison import validate_cut3r_comparison_lock
from .cut3r_source_competence import (
    _canonical_json,
    _comparison_context,
    _exact_keys,
    _hierarchical_group_metrics,
    _mean,
    _record_id,
    _sha256,
    _strict_boolean,
    _strict_mapping,
    build_cut3r_source_competence_report,
    validate_cut3r_source_competence_lock,
)
from .fresh_provider_readiness import ReadinessGateV1
from .source_provider_competence import SourceProviderCompetenceReportV1


def _aggregate(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluable = [group for group in groups if group["technical_failure_code"] is None]
    if not evaluable:
        return {"evaluable_group_count": 0}
    paired = [cast(Mapping[str, Any], group["paired"]) for group in evaluable]
    seam = [item["seam_rmse_ratio"] for item in paired]
    drift = [item["absolute_drift_slope_ratio"] for item in paired]
    complete_seam = all(value is not None for value in seam)
    complete_drift = all(value is not None for value in drift)
    values = {
        "association_precision": [
            cast(float, item["association_precision_delta"]) for item in paired
        ],
        "identity_retention": [
            cast(float, item["identity_retention_delta"]) for item in paired
        ],
        "support_retention": [
            cast(float, item["support_retention_delta"]) for item in paired
        ],
    }
    return {
        "evaluable_group_count": len(evaluable),
        "mean_seam_rmse_ratio": (
            _mean([cast(float, value) for value in seam]) if complete_seam else None
        ),
        "worst_group_seam_rmse_ratio": (
            max(cast(list[float], seam)) if complete_seam else None
        ),
        "mean_absolute_drift_slope_ratio": (
            _mean([cast(float, value) for value in drift]) if complete_drift else None
        ),
        "worst_group_absolute_drift_slope_ratio": (
            max(cast(list[float], drift)) if complete_drift else None
        ),
        "mean_association_precision_delta": _mean(values["association_precision"]),
        "worst_group_association_precision_delta": min(
            values["association_precision"]
        ),
        "mean_identity_retention_delta": _mean(values["identity_retention"]),
        "worst_group_identity_retention_delta": min(values["identity_retention"]),
        "mean_support_retention_delta": _mean(values["support_retention"]),
        "worst_group_support_retention_delta": min(values["support_retention"]),
        "paired_quality_group_pass_fraction": _mean(
            [1.0 if group["paired_quality_pass"] else 0.0 for group in evaluable]
        ),
        "paired_identity_group_pass_fraction": _mean(
            [1.0 if group["paired_identity_pass"] else 0.0 for group in evaluable]
        ),
    }


def _statuses(
    v1_report: SourceProviderCompetenceReportV1,
    aggregate: Mapping[str, Any],
    policy: Mapping[str, float],
) -> tuple[Status, list[str], Status, list[str]]:
    mean_reasons = list(v1_report.mean_quality_reasons)
    if v1_report.mean_quality_status == "technical-failure":
        return "technical-failure", mean_reasons, "not-evaluated", []
    if v1_report.mean_quality_status != "pass":
        return "fail", mean_reasons, "not-evaluated", []
    for name, threshold in (
        ("mean_seam_rmse_ratio", policy["maximum_mean_seam_rmse_ratio"]),
        (
            "worst_group_seam_rmse_ratio",
            policy["maximum_worst_group_seam_rmse_ratio"],
        ),
        (
            "mean_absolute_drift_slope_ratio",
            policy["maximum_mean_absolute_drift_slope_ratio"],
        ),
        (
            "worst_group_absolute_drift_slope_ratio",
            policy["maximum_worst_group_absolute_drift_slope_ratio"],
        ),
    ):
        if aggregate.get(name) is None:
            mean_reasons.append(f"{name.replace('_', '-')}-undefined")
        elif aggregate[name] > threshold:
            mean_reasons.append(f"{name.replace('_', '-')}-exceeded")
    if aggregate["paired_quality_group_pass_fraction"] < policy[
        "minimum_paired_quality_group_pass_fraction"
    ]:
        mean_reasons.append("paired-quality-group-pass-fraction-below-minimum")
    if mean_reasons:
        return "fail", sorted(set(mean_reasons)), "not-evaluated", []
    identity_reasons = list(v1_report.identity_reliability_reasons)
    if v1_report.identity_reliability_status != "pass":
        status = cast(Status, v1_report.identity_reliability_status)
        return "pass", [], status, sorted(set(identity_reasons))
    for name, threshold in (
        (
            "mean_association_precision_delta",
            policy["minimum_mean_association_precision_delta"],
        ),
        (
            "worst_group_association_precision_delta",
            policy["minimum_worst_group_association_precision_delta"],
        ),
        (
            "mean_identity_retention_delta",
            policy["minimum_mean_identity_retention_delta"],
        ),
        (
            "worst_group_identity_retention_delta",
            policy["minimum_worst_group_identity_retention_delta"],
        ),
        (
            "mean_support_retention_delta",
            policy["minimum_mean_support_retention_delta"],
        ),
        (
            "worst_group_support_retention_delta",
            policy["minimum_worst_group_support_retention_delta"],
        ),
    ):
        if aggregate[name] < threshold:
            identity_reasons.append(f"{name.replace('_', '-')}-below-minimum")
    if aggregate["paired_identity_group_pass_fraction"] < policy[
        "minimum_paired_identity_group_pass_fraction"
    ]:
        identity_reasons.append("paired-identity-group-pass-fraction-below-minimum")
    return "pass", [], "pass" if not identity_reasons else "fail", sorted(
        set(identity_reasons)
    )


def _build_report(
    comparison: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    v2_lock: Mapping[str, Any],
    records: Any,
) -> dict[str, Any]:
    normalized, v1_records = _normalize_v2_records(
        comparison,
        source_lock,
        v2_lock,
        records,
    )
    v1_report = build_cut3r_source_competence_report(
        comparison,
        source_lock,
        v1_records,
    )
    context = _comparison_context(comparison)
    cases = cast(dict[str, dict[str, tuple[int, int]]], context["cases_by_group"])
    seeds = cast(list[int], v2_lock["random_seeds"])
    treatment = cast(str, v2_lock["contrast"]["treatment_arm"])
    control = cast(str, v2_lock["contrast"]["control_arm"])
    nested: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in cast(list[dict[str, Any]], normalized["records"]):
        nested[
            (
                cast(str, record["group_id"]),
                cast(str, record["case_id"]),
                cast(int, record["random_seed"]),
                cast(str, record["arm_id"]),
            )
        ].append(record)
    failures = {
        cast(str, item["group_id"]): cast(str, item["technical_failure_code"])
        for item in cast(list[dict[str, Any]], normalized["group_failures"])
    }
    policy = cast(dict[str, float], v2_lock["paired_policy"])
    groups = []
    for group_id in cast(list[str], v2_lock["source_evaluation_group_ids"]):
        if group_id in failures:
            groups.append(
                {
                    "group_id": group_id,
                    "technical_failure_code": failures[group_id],
                    "candidate": None,
                    "baseline": None,
                    "paired": None,
                    "paired_quality_pass": False,
                    "paired_identity_pass": False,
                    "paired_quality_reasons": ["technical-failure"],
                    "paired_identity_reasons": ["technical-failure"],
                }
            )
            continue
        candidate = _hierarchical_group_metrics(
            group_id,
            treatment,
            cases=cases[group_id],
            seeds=seeds,
            record_lookup=nested,
        )
        baseline = _hierarchical_group_metrics(
            group_id,
            control,
            cases=cases[group_id],
            seeds=seeds,
            record_lookup=nested,
        )
        groups.append(_group_pair(group_id, candidate, baseline, policy))
    aggregate = _aggregate(groups)
    mean_status, mean_reasons, identity_status, identity_reasons = _statuses(
        v1_report,
        aggregate,
        policy,
    )
    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "source_competence_lock_id": source_lock["source_competence_lock_id"],
        "common_support_lock_id": v2_lock["common_support_lock_id"],
        "records_id": _record_id(normalized),
        "v1_source_provider_competence_id": v1_report.source_provider_competence_id,
        "v1_report": v1_report.to_dict(),
        "paired_policy": policy,
        "groups": groups,
        "aggregate": aggregate,
        "mean_quality_status": mean_status,
        "identity_reliability_status": identity_status,
        "mean_quality_reasons": mean_reasons,
        "identity_reliability_reasons": identity_reasons,
        "source_competence_pass": mean_status == "pass" and identity_status == "pass",
        "source_truth_used": True,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "weighting": dict(WEIGHTING),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["source_competence_report_v2_id"] = _record_id(payload)
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def build_cut3r_source_competence_v2_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    records: Any,
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
    return _build_report(comparison, source_lock, v2_lock, records)


def validate_cut3r_source_competence_v2_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    records: Any,
    report: Any,
) -> dict[str, Any]:
    supplied = _strict_mapping(report, name="source competence v2 report")
    expected = build_cut3r_source_competence_v2_report(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        records,
    )
    if supplied != expected:
        raise ValueError("source competence v2 report does not match the bound records")
    return expected


def source_competence_gates_v2(
    report: Mapping[str, Any],
) -> tuple[ReadinessGateV1, ReadinessGateV1]:
    artifact = _strict_mapping(report, name="source competence v2 report")
    _exact_keys(artifact, _REPORT_FIELDS, name="source competence v2 report")
    if artifact["schema"] != REPORT_SCHEMA or artifact["schema_version"] != VERSION:
        raise ValueError("unsupported source competence v2 report")
    if artifact["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("source competence v2 claim boundary changed")
    if not _strict_boolean(artifact["source_truth_used"], name="source_truth_used"):
        raise ValueError("source competence v2 report requires source truth")
    if _strict_boolean(artifact["target_payloads_opened"], name="target_payloads_opened"):
        raise ValueError("source competence v2 report may not open target payloads")
    if _strict_boolean(artifact["target_outcomes_opened"], name="target_outcomes_opened"):
        raise ValueError("source competence v2 report may not open target outcomes")
    report_id = _sha256(
        artifact["source_competence_report_v2_id"],
        name="source_competence_report_v2_id",
    )
    unsigned = dict(artifact)
    unsigned.pop("source_competence_report_v2_id")
    if _record_id(unsigned) != report_id:
        raise ValueError("source competence v2 report content identity changed")
    report = artifact
    mean_status = cast(Status, report["mean_quality_status"])
    mean_gate = ReadinessGateV1(
        gate_name="source-mean",
        status=cast(Any, mean_status),
        evidence_id=report_id,
        reason_codes=(
            () if mean_status == "pass" else tuple(report["mean_quality_reasons"])
        ),
        metadata={
            "common_support_lock_id": report["common_support_lock_id"],
            "v1_source_provider_competence_id": report[
                "v1_source_provider_competence_id"
            ],
        },
    )
    if mean_status != "pass":
        return mean_gate, ReadinessGateV1(
            gate_name="identity-reliability",
            status="not-evaluated",
            evidence_id=None,
        )
    identity_status = cast(Status, report["identity_reliability_status"])
    return mean_gate, ReadinessGateV1(
        gate_name="identity-reliability",
        status=cast(Any, identity_status),
        evidence_id=report_id,
        reason_codes=(
            ()
            if identity_status == "pass"
            else tuple(report["identity_reliability_reasons"])
        ),
        metadata={
            "common_support_lock_id": report["common_support_lock_id"],
            "v1_source_provider_competence_id": report[
                "v1_source_provider_competence_id"
            ],
        },
    )
