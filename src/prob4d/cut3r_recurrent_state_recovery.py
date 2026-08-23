"""Quantify how much CUT3R recurrent-state loss is recovered by Prob4D fusion.

The report combines two independently validated, source-only common-support
competence reports: Prob4D-fused restarted windows versus the restarted-newest
baseline, and native continuous recurrence versus the same restarted-newest
baseline. It requires the canonical restarted-newest rows to be byte-identical
across both contrasts before computing any descriptive recovery statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._cut3r_source_competence_v2_records import _normalize_v2_records
from .cut3r_comparison import (
    CUT3R_COMPARISON_GROUP_UNIT,
    load_cut3r_comparison_lock,
    validate_cut3r_comparison_lock,
)
from .cut3r_source_competence import (
    _canonical_json,
    _exact_keys,
    _publish_json,
    _record_id,
    _strict_integer,
    _strict_json,
    _strict_mapping,
    load_cut3r_source_competence_lock,
    validate_cut3r_source_competence_lock,
)
from .cut3r_source_competence_v2 import (
    load_cut3r_source_competence_v2_lock,
    load_cut3r_source_competence_v2_report,
    validate_cut3r_source_competence_v2_report,
)

CUT3R_RECURRENT_STATE_RECOVERY_SCHEMA: Final = (
    "prob4d.cut3r-recurrent-state-recovery"
)
CUT3R_RECURRENT_STATE_RECOVERY_VERSION: Final = 1
CUT3R_RECURRENT_STATE_RECOVERY_CLAIM_BOUNDARY: Final = (
    "This source-only report is a descriptive mechanism analysis of three frozen "
    "CUT3R arms. It does not select a candidate, alter a readiness gate, authorize "
    "target access, establish BayesianPhysTwin or Causal4D benefit, establish "
    "deployment safety, or establish state of the art. Recovery is undefined when "
    "native continuous recurrence does not outperform the restarted-newest baseline."
)
CUT3R_RECURRENT_STATE_RECOVERY_METRICS: Final[tuple[str, ...]] = (
    "point_rmse_m",
    "endpoint_rmse_m",
    "proper_score",
    "seam_rmse_m",
    "absolute_drift_slope_m_per_frame",
)
CUT3R_RECURRENT_STATE_RECOVERY_ARMS: Final = {
    "native_continuous": "native-continuous",
    "restarted_newest": "restarted-newest",
    "restarted_prob4d_fused": "restarted-prob4d-fused",
}
CUT3R_RECURRENT_STATE_RECOVERY_BOOTSTRAP: Final = (
    "sha256-counter-equal-group-bootstrap-v1"
)

_SPEC_FIELDS: Final = frozenset(
    {
        "bootstrap_seed",
        "bootstrap_replicates",
        "confidence_level",
        "minimum_recurrence_gap_by_metric",
        "minimum_valid_bootstrap_fraction",
    }
)


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a genuine finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            relation = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _analysis_specification(value: Any) -> dict[str, Any]:
    specification = _strict_mapping(
        value,
        name="CUT3R recurrent-state recovery specification",
    )
    _exact_keys(
        specification,
        _SPEC_FIELDS,
        name="CUT3R recurrent-state recovery specification",
    )
    confidence_level = _finite_float(
        specification["confidence_level"],
        name="confidence_level",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if confidence_level <= 0.0:
        raise ValueError("confidence_level must be > 0")
    minimum_valid_fraction = _finite_float(
        specification["minimum_valid_bootstrap_fraction"],
        name="minimum_valid_bootstrap_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    gaps = _strict_mapping(
        specification["minimum_recurrence_gap_by_metric"],
        name="minimum_recurrence_gap_by_metric",
    )
    _exact_keys(
        gaps,
        set(CUT3R_RECURRENT_STATE_RECOVERY_METRICS),
        name="minimum_recurrence_gap_by_metric",
    )
    return {
        "bootstrap_seed": _strict_integer(
            specification["bootstrap_seed"],
            name="bootstrap_seed",
        ),
        "bootstrap_replicates": _strict_integer(
            specification["bootstrap_replicates"],
            name="bootstrap_replicates",
            minimum=1,
        ),
        "confidence_level": confidence_level,
        "minimum_recurrence_gap_by_metric": {
            metric: _finite_float(
                gaps[metric],
                name=f"minimum_recurrence_gap_by_metric.{metric}",
                minimum=0.0,
            )
            for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS
        },
        "minimum_valid_bootstrap_fraction": minimum_valid_fraction,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_index(
    *,
    seed: int,
    replicate: int,
    position: int,
    group_count: int,
) -> int:
    material = (
        f"{CUT3R_RECURRENT_STATE_RECOVERY_BOOTSTRAP}:"
        f"{seed}:{replicate}:{position}:{group_count}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big") % group_count


def _triplet(
    native: float,
    restarted: float,
    fused: float,
    *,
    denominator_tolerance: float,
) -> dict[str, Any]:
    prob4d_gain = restarted - fused
    recurrence_gap = restarted - native
    if recurrence_gap > denominator_tolerance:
        return {
            "native_continuous": native,
            "restarted_newest": restarted,
            "restarted_prob4d_fused": fused,
            "prob4d_gain": prob4d_gain,
            "recurrence_gap": recurrence_gap,
            "recovery_fraction": prob4d_gain / recurrence_gap,
            "status": "defined",
        }
    return {
        "native_continuous": native,
        "restarted_newest": restarted,
        "restarted_prob4d_fused": fused,
        "prob4d_gain": prob4d_gain,
        "recurrence_gap": recurrence_gap,
        "recovery_fraction": None,
        "status": "undefined-native-not-better",
    }


def _bootstrap_interval(
    group_triplets: Sequence[tuple[float, float, float]],
    *,
    specification: Mapping[str, Any],
    point_status: str,
) -> dict[str, Any]:
    replicates = cast(int, specification["bootstrap_replicates"])
    seed = cast(int, specification["bootstrap_seed"])
    tolerance = cast(float, specification["minimum_recurrence_gap"])
    confidence_level = cast(float, specification["confidence_level"])
    minimum_valid_fraction = cast(
        float,
        specification["minimum_valid_bootstrap_fraction"],
    )
    group_count = len(group_triplets)
    ratios: list[float] = []
    for replicate in range(replicates):
        selected = [
            group_triplets[
                _bootstrap_index(
                    seed=seed,
                    replicate=replicate,
                    position=position,
                    group_count=group_count,
                )
            ]
            for position in range(group_count)
        ]
        native = _mean([item[0] for item in selected])
        restarted = _mean([item[1] for item in selected])
        fused = _mean([item[2] for item in selected])
        recurrence_gap = restarted - native
        if recurrence_gap > tolerance:
            ratios.append((restarted - fused) / recurrence_gap)
    valid_count = len(ratios)
    valid_fraction = valid_count / replicates
    invalid_count = replicates - valid_count
    if point_status != "defined":
        interval_status = "not-applicable"
        lower = None
        upper = None
    elif valid_count == 0 or valid_fraction < minimum_valid_fraction:
        interval_status = "insufficient-valid-bootstrap-denominators"
        lower = None
        upper = None
    else:
        tail = (1.0 - confidence_level) / 2.0
        interval_status = "defined"
        lower = _percentile(ratios, tail)
        upper = _percentile(ratios, 1.0 - tail)
    return {
        "method": CUT3R_RECURRENT_STATE_RECOVERY_BOOTSTRAP,
        "replicate_count": replicates,
        "valid_replicate_count": valid_count,
        "invalid_replicate_count": invalid_count,
        "valid_replicate_fraction": valid_fraction,
        "minimum_valid_replicate_fraction": minimum_valid_fraction,
        "confidence_level": confidence_level,
        "interval_status": interval_status,
        "lower": lower,
        "upper": upper,
    }


def _contrast(
    source_lock: Mapping[str, Any],
    *,
    contrast_id: str,
    treatment: str,
    control: str,
    name: str,
) -> None:
    contrast = cast(Mapping[str, Any], source_lock["contrast"])
    expected = {
        "contrast_id": contrast_id,
        "treatment_arm": treatment,
        "control_arm": control,
        "claim_eligible": True,
        "enabled": True,
    }
    if contrast != expected:
        raise ValueError(f"{name} must use the canonical {contrast_id!r} contrast")


def _baseline_rows(records: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        cast(dict[str, Any], row)
        for row in cast(list[dict[str, Any]], records["records"])
        if row["arm_id"] == CUT3R_RECURRENT_STATE_RECOVERY_ARMS["restarted_newest"]
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["group_id"],
            row["case_id"],
            row["frame_index"],
            row["random_seed"],
        ),
    )


def _group_lookup(report: Mapping[str, Any], *, name: str) -> dict[str, Mapping[str, Any]]:
    raw_groups = report["groups"]
    if type(raw_groups) is not list:
        raise ValueError(f"{name}.groups must be a JSON array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(raw_groups):
        group = _strict_mapping(raw_group, name=f"{name}.groups[{index}]")
        group_id = group.get("group_id")
        if type(group_id) is not str or not group_id:
            raise ValueError(f"{name}.groups[{index}].group_id is invalid")
        if group_id in result:
            raise ValueError(f"{name} repeats group_id {group_id!r}")
        result[group_id] = group
    return result


def build_cut3r_recurrent_state_recovery_report(
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
    """Build a content-addressed three-arm recurrent-state recovery report."""

    comparison = validate_cut3r_comparison_lock(comparison_lock)
    specification = _analysis_specification(analysis_specification)
    fusion_source = validate_cut3r_source_competence_lock(
        comparison,
        fusion_source_competence_lock,
    )
    recurrence_source = validate_cut3r_source_competence_lock(
        comparison,
        recurrence_source_competence_lock,
    )
    _contrast(
        fusion_source,
        contrast_id="prob4d-fusion-value",
        treatment=CUT3R_RECURRENT_STATE_RECOVERY_ARMS["restarted_prob4d_fused"],
        control=CUT3R_RECURRENT_STATE_RECOVERY_ARMS["restarted_newest"],
        name="fusion source competence lock",
    )
    _contrast(
        recurrence_source,
        contrast_id="provider-recurrence-value",
        treatment=CUT3R_RECURRENT_STATE_RECOVERY_ARMS["native_continuous"],
        control=CUT3R_RECURRENT_STATE_RECOVERY_ARMS["restarted_newest"],
        name="recurrence source competence lock",
    )
    for field in ("cohort_binding_id", "group_definition", "baseline_provider_manifest_id"):
        if fusion_source[field] != recurrence_source[field]:
            raise ValueError(f"the two contrasts use different {field}")
    fusion = validate_cut3r_source_competence_v2_report(
        comparison,
        fusion_source,
        fusion_common_support_lock,
        fusion_records,
        fusion_report,
    )
    recurrence = validate_cut3r_source_competence_v2_report(
        comparison,
        recurrence_source,
        recurrence_common_support_lock,
        recurrence_records,
        recurrence_report,
    )
    normalized_fusion_records, _ = _normalize_v2_records(
        comparison,
        fusion_source,
        fusion_common_support_lock,
        fusion_records,
    )
    normalized_recurrence_records, _ = _normalize_v2_records(
        comparison,
        recurrence_source,
        recurrence_common_support_lock,
        recurrence_records,
    )
    if normalized_fusion_records["group_failures"] != normalized_recurrence_records[
        "group_failures"
    ]:
        raise ValueError("the two contrasts use different technical-failure evidence")
    if _baseline_rows(normalized_fusion_records) != _baseline_rows(
        normalized_recurrence_records
    ):
        raise ValueError(
            "the two contrasts do not use byte-identical restarted-newest rows"
        )

    source_group_ids = list(comparison["group_roles"]["source_evaluation"])
    fusion_groups = _group_lookup(fusion, name="fusion_report")
    recurrence_groups = _group_lookup(recurrence, name="recurrence_report")
    if list(fusion_groups) != source_group_ids or list(recurrence_groups) != source_group_ids:
        raise ValueError("source report group order changed from the comparison lock")

    group_rows: list[dict[str, Any]] = []
    evaluable_triplets: dict[str, list[tuple[float, float, float]]] = {
        metric: [] for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS
    }
    for group_id in source_group_ids:
        fusion_group = fusion_groups[group_id]
        recurrence_group = recurrence_groups[group_id]
        fusion_failure = fusion_group["technical_failure_code"]
        recurrence_failure = recurrence_group["technical_failure_code"]
        if fusion_failure != recurrence_failure:
            raise ValueError(f"group {group_id!r} has contrast-specific technical failure")
        if fusion_failure is not None:
            group_rows.append(
                {
                    "group_id": group_id,
                    "technical_failure_code": fusion_failure,
                    "metrics": None,
                }
            )
            continue
        fusion_baseline = cast(Mapping[str, Any], fusion_group["baseline"])
        recurrence_baseline = cast(Mapping[str, Any], recurrence_group["baseline"])
        if fusion_baseline != recurrence_baseline:
            raise ValueError(
                f"group {group_id!r} has different restarted-newest aggregate metrics"
            )
        fused = cast(Mapping[str, Any], fusion_group["candidate"])
        native = cast(Mapping[str, Any], recurrence_group["candidate"])
        metrics: dict[str, Any] = {}
        for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS:
            native_value = _finite_float(native[metric], name=f"{group_id}.{metric}.native")
            restarted_value = _finite_float(
                fusion_baseline[metric],
                name=f"{group_id}.{metric}.restarted",
            )
            fused_value = _finite_float(fused[metric], name=f"{group_id}.{metric}.fused")
            metrics[metric] = _triplet(
                native_value,
                restarted_value,
                fused_value,
                denominator_tolerance=specification[
                    "minimum_recurrence_gap_by_metric"
                ][metric],
            )
            evaluable_triplets[metric].append(
                (native_value, restarted_value, fused_value)
            )
        group_rows.append(
            {
                "group_id": group_id,
                "technical_failure_code": None,
                "metrics": metrics,
            }
        )

    evaluable_group_count = sum(
        row["technical_failure_code"] is None for row in group_rows
    )
    if evaluable_group_count == 0:
        raise ValueError("recurrent-state recovery requires at least one evaluable group")
    aggregate: dict[str, Any] = {}
    for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS:
        triplets = evaluable_triplets[metric]
        point = _triplet(
            _mean([item[0] for item in triplets]),
            _mean([item[1] for item in triplets]),
            _mean([item[2] for item in triplets]),
            denominator_tolerance=specification[
                "minimum_recurrence_gap_by_metric"
            ][metric],
        )
        bootstrap_specification = {
            **specification,
            "minimum_recurrence_gap": specification[
                "minimum_recurrence_gap_by_metric"
            ][metric],
        }
        point["bootstrap_interval"] = _bootstrap_interval(
            triplets,
            specification=bootstrap_specification,
            point_status=cast(str, point["status"]),
        )
        aggregate[metric] = point

    payload: dict[str, Any] = {
        "schema": CUT3R_RECURRENT_STATE_RECOVERY_SCHEMA,
        "schema_version": CUT3R_RECURRENT_STATE_RECOVERY_VERSION,
        "comparison_lock_id": comparison["lock_id"],
        "comparison_protocol_name": comparison["protocol_name"],
        "group_unit": CUT3R_COMPARISON_GROUP_UNIT,
        "source_evaluation_group_ids": source_group_ids,
        "group_count": len(source_group_ids),
        "evaluable_group_count": evaluable_group_count,
        "technical_failure_count": len(source_group_ids) - evaluable_group_count,
        "arms": dict(CUT3R_RECURRENT_STATE_RECOVERY_ARMS),
        "evidence": {
            "fusion_source_competence_report_v2_id": fusion[
                "source_competence_report_v2_id"
            ],
            "fusion_common_support_lock_id": fusion["common_support_lock_id"],
            "fusion_records_id": fusion["records_id"],
            "recurrence_source_competence_report_v2_id": recurrence[
                "source_competence_report_v2_id"
            ],
            "recurrence_common_support_lock_id": recurrence[
                "common_support_lock_id"
            ],
            "recurrence_records_id": recurrence["records_id"],
            "restarted_newest_provider_manifest_id": fusion_source[
                "baseline_provider_manifest_id"
            ],
            "byte_identical_restarted_newest_rows": True,
            "common_three_arm_metric_support_for_evaluable_groups": True,
        },
        "metrics": list(CUT3R_RECURRENT_STATE_RECOVERY_METRICS),
        "metric_direction": "lower-is-better",
        "recovery_definition": (
            "(restarted-newest - restarted-prob4d-fused) / "
            "(restarted-newest - native-continuous)"
        ),
        "analysis_specification": specification,
        "groups": group_rows,
        "aggregate": aggregate,
        "weighting": "equal-complete-source-group-mean-v1",
        "descriptive_only": True,
        "source_access": "source-only",
        "target_access": "forbidden",
        "claim_boundary": CUT3R_RECURRENT_STATE_RECOVERY_CLAIM_BOUNDARY,
    }
    payload["recurrent_state_recovery_report_id"] = _record_id(payload)
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def validate_cut3r_recurrent_state_recovery_report(
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
    supplied = _strict_mapping(report, name="CUT3R recurrent-state recovery report")
    expected = build_cut3r_recurrent_state_recovery_report(
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
    if supplied != expected:
        raise ValueError("recurrent-state recovery report does not match bound evidence")
    return expected


def write_cut3r_recurrent_state_recovery_report(
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
    payload = validate_cut3r_recurrent_state_recovery_report(
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
        load_existing=lambda existing: load_cut3r_recurrent_state_recovery_report(
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


def load_cut3r_recurrent_state_recovery_report(
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
    return validate_cut3r_recurrent_state_recovery_report(
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


def cut3r_recurrent_state_recovery_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _strict_mapping(report, name="CUT3R recurrent-state recovery report")
    metrics = cast(Mapping[str, Mapping[str, Any]], artifact["aggregate"])
    return {
        "recurrent_state_recovery_report_id": artifact[
            "recurrent_state_recovery_report_id"
        ],
        "comparison_lock_id": artifact["comparison_lock_id"],
        "group_count": artifact["group_count"],
        "evaluable_group_count": artifact["evaluable_group_count"],
        "technical_failure_count": artifact["technical_failure_count"],
        "recovery": {
            metric: {
                "status": metrics[metric]["status"],
                "recovery_fraction": metrics[metric]["recovery_fraction"],
                "bootstrap_interval": metrics[metric]["bootstrap_interval"],
            }
            for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS
        },
        "descriptive_only": artifact["descriptive_only"],
        "target_access": artifact["target_access"],
    }


def _add_bound_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("comparison_lock")
    parser.add_argument("fusion_source_competence_lock")
    parser.add_argument("fusion_common_support_lock")
    parser.add_argument("fusion_records")
    parser.add_argument("fusion_report")
    parser.add_argument("recurrence_source_competence_lock")
    parser.add_argument("recurrence_common_support_lock")
    parser.add_argument("recurrence_records")
    parser.add_argument("recurrence_report")
    parser.add_argument("analysis_specification")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction cut3r-recovery",
        description=__doc__,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the bound recovery report")
    _add_bound_evidence_arguments(build)
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="rebuild and verify a recovery report")
    _add_bound_evidence_arguments(verify)
    verify.add_argument("report")
    summarize = commands.add_parser("summarize", help="summarize a verified report")
    _add_bound_evidence_arguments(summarize)
    summarize.add_argument("report")
    summarize.add_argument("--json", action="store_true")
    return parser


def _load_bound_inputs(arguments: argparse.Namespace) -> tuple[Any, ...]:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    fusion_source = load_cut3r_source_competence_lock(
        comparison,
        arguments.fusion_source_competence_lock,
    )
    fusion_support = load_cut3r_source_competence_v2_lock(
        comparison,
        fusion_source,
        arguments.fusion_common_support_lock,
    )
    fusion_records = _strict_json(arguments.fusion_records)
    fusion_report = load_cut3r_source_competence_v2_report(
        comparison,
        fusion_source,
        fusion_support,
        fusion_records,
        arguments.fusion_report,
    )
    recurrence_source = load_cut3r_source_competence_lock(
        comparison,
        arguments.recurrence_source_competence_lock,
    )
    recurrence_support = load_cut3r_source_competence_v2_lock(
        comparison,
        recurrence_source,
        arguments.recurrence_common_support_lock,
    )
    recurrence_records = _strict_json(arguments.recurrence_records)
    recurrence_report = load_cut3r_source_competence_v2_report(
        comparison,
        recurrence_source,
        recurrence_support,
        recurrence_records,
        arguments.recurrence_report,
    )
    specification = _strict_json(arguments.analysis_specification)
    return (
        comparison,
        fusion_source,
        fusion_support,
        fusion_records,
        fusion_report,
        recurrence_source,
        recurrence_support,
        recurrence_records,
        recurrence_report,
        specification,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    inputs = _load_bound_inputs(arguments)
    if arguments.command == "build":
        report = build_cut3r_recurrent_state_recovery_report(*inputs)
        write_cut3r_recurrent_state_recovery_report(
            *inputs,
            arguments.output,
            report,
        )
        print(report["recurrent_state_recovery_report_id"])
        return 0
    report = load_cut3r_recurrent_state_recovery_report(
        *inputs,
        arguments.report,
    )
    if arguments.command == "verify":
        print(report["recurrent_state_recovery_report_id"])
        return 0
    summary = cut3r_recurrent_state_recovery_summary(report)
    if arguments.json:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(f"report_id: {summary['recurrent_state_recovery_report_id']}")
        print(f"evaluable groups: {summary['evaluable_group_count']}")
        for metric, result in summary["recovery"].items():
            print(
                f"{metric}: {result['status']} "
                f"recovery={result['recovery_fraction']}"
            )
        print(f"target access: {summary['target_access']}")
    return 0


__all__ = [
    "CUT3R_RECURRENT_STATE_RECOVERY_ARMS",
    "CUT3R_RECURRENT_STATE_RECOVERY_BOOTSTRAP",
    "CUT3R_RECURRENT_STATE_RECOVERY_CLAIM_BOUNDARY",
    "CUT3R_RECURRENT_STATE_RECOVERY_METRICS",
    "CUT3R_RECURRENT_STATE_RECOVERY_SCHEMA",
    "CUT3R_RECURRENT_STATE_RECOVERY_VERSION",
    "build_cut3r_recurrent_state_recovery_report",
    "cut3r_recurrent_state_recovery_summary",
    "load_cut3r_recurrent_state_recovery_report",
    "main",
    "validate_cut3r_recurrent_state_recovery_report",
    "write_cut3r_recurrent_state_recovery_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
