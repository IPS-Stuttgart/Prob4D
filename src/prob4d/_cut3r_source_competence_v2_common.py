"""Shared strict definitions for CUT3R source competence v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, cast

from .cut3r_source_competence import (
    _exact_keys,
    _finite_number,
    _sha256,
    _strict_integer,
    _strict_mapping,
    _strict_string,
)

LOCK_SCHEMA: Final = "prob4d.cut3r-source-competence-common-support-lock"
RECORDS_SCHEMA: Final = "prob4d.cut3r-source-competence-records"
REPORT_SCHEMA: Final = "prob4d.cut3r-source-competence-report"
VERSION: Final = 2
PROPER_SCORE_SEMANTICS: Final = "arm-neutral-fixed-scale-gaussian-score-v1"
CLAIM_BOUNDARY: Final = (
    "This source-only artifact proves exact paired metric support and evaluates "
    "paired CUT3R source competence on complete frozen physical object/session "
    "groups. It cannot change the arms, omit nested records, use arm-specific "
    "covariance in the source-mean score, authorize target access, establish "
    "BayesianPhysTwin or Causal4D benefit, or establish state of the art."
)
WEIGHTING: Final = {
    "within_case": "equal-frame-mean-within-each-frozen-seed-v1",
    "within_group": "equal-seed-means-then-equal-case-means-v1",
    "across_groups": "equal-complete-group-mean-v1",
    "technical_failures": "retain-complete-group-without-scored-metrics-v1",
    "paired_support": "exact-byte-identical-metric-support-v2",
}
Status = Literal["pass", "fail", "technical-failure", "not-evaluated"]

_LOCK_SPEC_FIELDS = frozenset(
    {
        "source_competence_policy",
        "common_support_definition_sha256",
        "proper_score_semantics",
        "paired_policy",
        "require_complete_source_roster",
    }
)
_LOCK_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "record_definition_sha256",
        "common_support_definition_sha256",
        "source_evaluation_group_ids",
        "random_seeds",
        "contrast",
        "source_competence_policy",
        "proper_score_semantics",
        "paired_policy",
        "require_complete_source_roster",
        "weighting",
        "source_truth_required",
        "target_access",
        "claim_boundary",
        "common_support_lock_id",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "maximum_mean_seam_rmse_ratio",
        "maximum_worst_group_seam_rmse_ratio",
        "maximum_mean_absolute_drift_slope_ratio",
        "maximum_worst_group_absolute_drift_slope_ratio",
        "minimum_mean_association_precision_delta",
        "minimum_worst_group_association_precision_delta",
        "minimum_mean_identity_retention_delta",
        "minimum_worst_group_identity_retention_delta",
        "minimum_mean_support_retention_delta",
        "minimum_worst_group_support_retention_delta",
        "minimum_paired_quality_group_pass_fraction",
        "minimum_paired_identity_group_pass_fraction",
    }
)
_RECORDS_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "common_support_lock_id",
        "record_definition_sha256",
        "common_support_definition_sha256",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "group_failures",
        "records",
    }
)
_RECORD_FIELDS = frozenset(
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
        "metric_support",
    }
)

_REPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "comparison_lock_id",
        "source_competence_lock_id",
        "common_support_lock_id",
        "records_id",
        "v1_source_provider_competence_id",
        "v1_report",
        "paired_policy",
        "groups",
        "aggregate",
        "mean_quality_status",
        "identity_reliability_status",
        "mean_quality_reasons",
        "identity_reliability_reasons",
        "source_competence_pass",
        "source_truth_used",
        "target_payloads_opened",
        "target_outcomes_opened",
        "weighting",
        "claim_boundary",
        "source_competence_report_v2_id",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "point_support_sha256",
        "point_support_count",
        "endpoint_support_sha256",
        "endpoint_support_count",
        "proper_score_support_sha256",
        "proper_score_dimension",
        "proper_score_semantics",
        "seam_support_sha256",
        "seam_support_count",
    }
)


def _positive(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name, nonnegative=True)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name, nonnegative=True)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _delta(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name, nonnegative=False)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return result


def _paired_policy(value: Any) -> dict[str, float]:
    mapping = _strict_mapping(value, name="paired policy")
    _exact_keys(mapping, _POLICY_FIELDS, name="paired policy")
    result: dict[str, float] = {}
    for name in (
        "maximum_mean_seam_rmse_ratio",
        "maximum_worst_group_seam_rmse_ratio",
        "maximum_mean_absolute_drift_slope_ratio",
        "maximum_worst_group_absolute_drift_slope_ratio",
    ):
        result[name] = _positive(mapping[name], name=name)
    for name in (
        "minimum_mean_association_precision_delta",
        "minimum_worst_group_association_precision_delta",
        "minimum_mean_identity_retention_delta",
        "minimum_worst_group_identity_retention_delta",
        "minimum_mean_support_retention_delta",
        "minimum_worst_group_support_retention_delta",
    ):
        result[name] = _delta(mapping[name], name=name)
    for name in (
        "minimum_paired_quality_group_pass_fraction",
        "minimum_paired_identity_group_pass_fraction",
    ):
        result[name] = _probability(mapping[name], name=name)
    return result


def _support(value: Any, *, seam_present: bool) -> dict[str, Any]:
    mapping = _strict_mapping(value, name="metric_support")
    _exact_keys(mapping, _SUPPORT_FIELDS, name="metric_support")
    semantics = _strict_string(
        mapping["proper_score_semantics"],
        name="proper_score_semantics",
    )
    if semantics != PROPER_SCORE_SEMANTICS:
        raise ValueError(
            "source-mean proper score must use arm-neutral fixed-scale semantics"
        )
    seam_count = _strict_integer(
        mapping["seam_support_count"],
        name="seam_support_count",
    )
    raw_seam_digest = mapping["seam_support_sha256"]
    seam_digest = None if raw_seam_digest is None else _sha256(
        raw_seam_digest,
        name="seam_support_sha256",
    )
    if seam_present:
        if seam_count <= 0 or seam_digest is None:
            raise ValueError("seam error and seam support identity disagree")
    elif seam_count != 0 or seam_digest is not None:
        raise ValueError("absent seam error requires zero count and null digest")
    return {
        "point_support_sha256": _sha256(
            mapping["point_support_sha256"],
            name="point_support_sha256",
        ),
        "point_support_count": _strict_integer(
            mapping["point_support_count"],
            name="point_support_count",
            minimum=1,
        ),
        "endpoint_support_sha256": _sha256(
            mapping["endpoint_support_sha256"],
            name="endpoint_support_sha256",
        ),
        "endpoint_support_count": _strict_integer(
            mapping["endpoint_support_count"],
            name="endpoint_support_count",
            minimum=1,
        ),
        "proper_score_support_sha256": _sha256(
            mapping["proper_score_support_sha256"],
            name="proper_score_support_sha256",
        ),
        "proper_score_dimension": _strict_integer(
            mapping["proper_score_dimension"],
            name="proper_score_dimension",
            minimum=1,
        ),
        "proper_score_semantics": semantics,
        "seam_support_sha256": seam_digest,
        "seam_support_count": seam_count,
    }


def _source_group_ids(source_lock: Mapping[str, Any]) -> list[str]:
    return [
        cast(str, group["group_id"])
        for group in cast(list[dict[str, Any]], source_lock["source_evaluation_groups"])
    ]
