"""Content-addressed CUT3R recurrent-state recovery-v2 report."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ._cut3r_recovery_v2_exact import (
    _exact_group_bootstrap,
    _leave_one_group_out,
    _triplet_v2,
)
from ._cut3r_recovery_v2_spec import (
    CUT3R_RECURRENT_STATE_RECOVERY_V2_CLAIM_BOUNDARY,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SCHEMA,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_VERSION,
    validate_cut3r_recurrent_state_recovery_v2_specification,
)
from .cut3r_recurrent_state_recovery import (
    CUT3R_RECURRENT_STATE_RECOVERY_ARMS,
    CUT3R_RECURRENT_STATE_RECOVERY_METRICS,
    _mean,
    build_cut3r_recurrent_state_recovery_report,
)
from .cut3r_source_competence import (
    _canonical_json,
    _publish_json,
    _record_id,
    _strict_json,
    _strict_mapping,
)


def _v1_bridge_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bootstrap_seed": 0,
        "bootstrap_replicates": 1,
        "confidence_level": specification["confidence_level"],
        "minimum_recurrence_gap_by_metric": specification[
            "minimum_recurrence_gap_by_metric"
        ],
        "minimum_valid_bootstrap_fraction": 0.0,
    }


def _build_v2_from_validated_v1_report(
    validated_v1_report: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    group_rows = cast(Sequence[Mapping[str, Any]], validated_v1_report["groups"])
    v2_group_rows: list[dict[str, Any]] = []
    for row in group_rows:
        if row["technical_failure_code"] is not None:
            v2_group_rows.append(
                {
                    "group_id": row["group_id"],
                    "technical_failure_code": row["technical_failure_code"],
                    "metrics": None,
                }
            )
            continue
        source_metrics = cast(Mapping[str, Mapping[str, Any]], row["metrics"])
        metrics: dict[str, Any] = {}
        for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS:
            threshold = cast(
                float,
                cast(
                    Mapping[str, Any],
                    specification["minimum_recurrence_gap_by_metric"],
                )[metric],
            )
            metric_values = source_metrics[metric]
            metrics[metric] = _triplet_v2(
                cast(float, metric_values["native_continuous"]),
                cast(float, metric_values["restarted_newest"]),
                cast(float, metric_values["restarted_prob4d_fused"]),
                denominator_threshold=threshold,
            )
        v2_group_rows.append(
            {
                "group_id": row["group_id"],
                "technical_failure_code": None,
                "metrics": metrics,
            }
        )
    evaluable_rows = [
        row for row in v2_group_rows if row["technical_failure_code"] is None
    ]
    if len(evaluable_rows) < 2:
        raise ValueError("v2 analysis requires at least two evaluable source groups")
    maximum_exact_group_count = cast(int, specification["maximum_exact_group_count"])
    if len(evaluable_rows) > maximum_exact_group_count:
        raise ValueError(
            "evaluable group count exceeds maximum_exact_group_count"
        )
    group_ids = [cast(str, row["group_id"]) for row in evaluable_rows]
    aggregate: dict[str, Any] = {}
    for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS:
        triplets = [
            (
                cast(float, cast(Mapping[str, Any], row["metrics"])[metric]["native_continuous"]),
                cast(float, cast(Mapping[str, Any], row["metrics"])[metric]["restarted_newest"]),
                cast(
                    float,
                    cast(Mapping[str, Any], row["metrics"])[metric][
                        "restarted_prob4d_fused"
                    ],
                ),
            )
            for row in evaluable_rows
        ]
        threshold = cast(
            float,
            cast(Mapping[str, Any], specification["minimum_recurrence_gap_by_metric"])[
                metric
            ],
        )
        point = _triplet_v2(
            _mean([item[0] for item in triplets]),
            _mean([item[1] for item in triplets]),
            _mean([item[2] for item in triplets]),
            denominator_threshold=threshold,
        )
        point["exact_group_bootstrap"] = _exact_group_bootstrap(
            triplets,
            denominator_threshold=threshold,
            confidence_level=cast(float, specification["confidence_level"]),
            minimum_valid_denominator_probability=cast(
                float,
                specification["minimum_valid_denominator_probability"],
            ),
            maximum_exact_group_count=maximum_exact_group_count,
            point_status=cast(str, point["status"]),
        )
        point["leave_one_group_out"] = _leave_one_group_out(
            group_ids,
            triplets,
            denominator_threshold=threshold,
            point=point,
        )
        aggregate[metric] = point

    payload: dict[str, Any] = {
        "schema": CUT3R_RECURRENT_STATE_RECOVERY_V2_SCHEMA,
        "schema_version": CUT3R_RECURRENT_STATE_RECOVERY_V2_VERSION,
        "comparison_lock_id": validated_v1_report["comparison_lock_id"],
        "comparison_protocol_name": validated_v1_report[
            "comparison_protocol_name"
        ],
        "group_unit": validated_v1_report["group_unit"],
        "source_evaluation_group_ids": validated_v1_report[
            "source_evaluation_group_ids"
        ],
        "group_count": validated_v1_report["group_count"],
        "evaluable_group_count": validated_v1_report["evaluable_group_count"],
        "technical_failure_count": validated_v1_report["technical_failure_count"],
        "arms": dict(CUT3R_RECURRENT_STATE_RECOVERY_ARMS),
        "evidence": {
            **cast(Mapping[str, Any], validated_v1_report["evidence"]),
            "v1_validation_bridge_report_id": validated_v1_report[
                "recurrent_state_recovery_report_id"
            ],
        },
        "metrics": list(CUT3R_RECURRENT_STATE_RECOVERY_METRICS),
        "metric_direction": "lower-is-better",
        "primary_endpoint": CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT,
        "secondary_endpoint": CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT,
        "recovery_definition": validated_v1_report["recovery_definition"],
        "analysis_specification": dict(specification),
        "groups": v2_group_rows,
        "aggregate": aggregate,
        "weighting": "equal-complete-source-group-mean-v1",
        "exact_small_sample_inference": True,
        "descriptive_only": True,
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": CUT3R_RECURRENT_STATE_RECOVERY_V2_CLAIM_BOUNDARY,
    }
    payload["recurrent_state_recovery_v2_report_id"] = _record_id(payload)
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def build_cut3r_recurrent_state_recovery_v2_report(
    comparison_lock: Any,
    fusion_source_competence_lock: Any,
    fusion_common_support_lock: Any,
    fusion_records: Any,
    fusion_report: Any,
    recurrence_source_competence_lock: Any,
    recurrence_common_support_lock: Any,
    recurrence_records: Any,
    recurrence_report: Any,
    analysis_specification: Any,
) -> dict[str, Any]:
    """Build a denominator-safe exact small-sample recovery report."""

    specification = validate_cut3r_recurrent_state_recovery_v2_specification(
        analysis_specification
    )
    validated_v1 = build_cut3r_recurrent_state_recovery_report(
        comparison_lock,
        fusion_source_competence_lock,
        fusion_common_support_lock,
        fusion_records,
        fusion_report,
        recurrence_source_competence_lock,
        recurrence_common_support_lock,
        recurrence_records,
        recurrence_report,
        _v1_bridge_specification(specification),
    )
    return _build_v2_from_validated_v1_report(validated_v1, specification)


def validate_cut3r_recurrent_state_recovery_v2_report(
    comparison_lock: Any,
    fusion_source_competence_lock: Any,
    fusion_common_support_lock: Any,
    fusion_records: Any,
    fusion_report: Any,
    recurrence_source_competence_lock: Any,
    recurrence_common_support_lock: Any,
    recurrence_records: Any,
    recurrence_report: Any,
    analysis_specification: Any,
    report: Any,
) -> dict[str, Any]:
    supplied = _strict_mapping(report, name="CUT3R recurrent-state recovery v2 report")
    expected = build_cut3r_recurrent_state_recovery_v2_report(
        comparison_lock,
        fusion_source_competence_lock,
        fusion_common_support_lock,
        fusion_records,
        fusion_report,
        recurrence_source_competence_lock,
        recurrence_common_support_lock,
        recurrence_records,
        recurrence_report,
        analysis_specification,
    )
    if dict(supplied) != expected:
        raise ValueError("v2 recovery report does not match bound evidence")
    return expected


def write_cut3r_recurrent_state_recovery_v2_report(
    comparison_lock: Any,
    fusion_source_competence_lock: Any,
    fusion_common_support_lock: Any,
    fusion_records: Any,
    fusion_report: Any,
    recurrence_source_competence_lock: Any,
    recurrence_common_support_lock: Any,
    recurrence_records: Any,
    recurrence_report: Any,
    analysis_specification: Any,
    path: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_recurrent_state_recovery_v2_report(
        comparison_lock,
        fusion_source_competence_lock,
        fusion_common_support_lock,
        fusion_records,
        fusion_report,
        recurrence_source_competence_lock,
        recurrence_common_support_lock,
        recurrence_records,
        recurrence_report,
        analysis_specification,
        report,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_recurrent_state_recovery_v2_report(
            comparison_lock,
            fusion_source_competence_lock,
            fusion_common_support_lock,
            fusion_records,
            fusion_report,
            recurrence_source_competence_lock,
            recurrence_common_support_lock,
            recurrence_records,
            recurrence_report,
            analysis_specification,
            existing,
        ),
    )


def load_cut3r_recurrent_state_recovery_v2_report(
    comparison_lock: Any,
    fusion_source_competence_lock: Any,
    fusion_common_support_lock: Any,
    fusion_records: Any,
    fusion_report: Any,
    recurrence_source_competence_lock: Any,
    recurrence_common_support_lock: Any,
    recurrence_records: Any,
    recurrence_report: Any,
    analysis_specification: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_recurrent_state_recovery_v2_report(
        comparison_lock,
        fusion_source_competence_lock,
        fusion_common_support_lock,
        fusion_records,
        fusion_report,
        recurrence_source_competence_lock,
        recurrence_common_support_lock,
        recurrence_records,
        recurrence_report,
        analysis_specification,
        _strict_json(path),
    )


def cut3r_recurrent_state_recovery_v2_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _strict_mapping(report, name="CUT3R recurrent-state recovery v2 report")
    metrics = cast(Mapping[str, Mapping[str, Any]], artifact["aggregate"])
    return {
        "recurrent_state_recovery_v2_report_id": artifact[
            "recurrent_state_recovery_v2_report_id"
        ],
        "comparison_lock_id": artifact["comparison_lock_id"],
        "evaluable_group_count": artifact["evaluable_group_count"],
        "technical_failure_count": artifact["technical_failure_count"],
        "primary_endpoint": artifact["primary_endpoint"],
        "secondary_endpoint": artifact["secondary_endpoint"],
        "metrics": {
            metric: {
                "prob4d_gain": metrics[metric]["prob4d_gain"],
                "prob4d_gain_interval": metrics[metric]["exact_group_bootstrap"][
                    "prob4d_gain_interval"
                ],
                "recurrence_gap": metrics[metric]["recurrence_gap"],
                "recovery_status": metrics[metric]["status"],
                "recovery_fraction": metrics[metric]["recovery_fraction"],
                "recovery_fraction_interval": metrics[metric][
                    "exact_group_bootstrap"
                ]["recovery_fraction_interval"],
                "valid_denominator_probability": metrics[metric][
                    "exact_group_bootstrap"
                ]["valid_denominator_probability"],
                "leave_one_group_out": metrics[metric]["leave_one_group_out"][
                    "summary"
                ],
            }
            for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS
        },
        "descriptive_only": artifact["descriptive_only"],
        "target_access": artifact["target_access"],
    }


__all__ = [
    "build_cut3r_recurrent_state_recovery_v2_report",
    "cut3r_recurrent_state_recovery_v2_summary",
    "load_cut3r_recurrent_state_recovery_v2_report",
    "validate_cut3r_recurrent_state_recovery_v2_report",
    "write_cut3r_recurrent_state_recovery_v2_report",
]
