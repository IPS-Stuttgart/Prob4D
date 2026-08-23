"""Prospective denominator-safe CUT3R recovery-v2 specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

from .cut3r_recurrent_state_recovery import (
    CUT3R_RECURRENT_STATE_RECOVERY_METRICS,
    _finite_float,
)
from .cut3r_source_competence import (
    _canonical_json,
    _exact_keys,
    _record_id,
    _strict_integer,
    _strict_json,
    _strict_mapping,
)

CUT3R_RECURRENT_STATE_RECOVERY_V2_SCHEMA: Final = (
    "prob4d.cut3r-recurrent-state-recovery-v2"
)
CUT3R_RECURRENT_STATE_RECOVERY_V2_VERSION: Final = 2
CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_SCHEMA: Final = (
    "prob4d.cut3r-recurrent-state-recovery-specification"
)
CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_VERSION: Final = 2
CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD: Final = (
    "exact-multinomial-equal-group-bootstrap-nearest-rank-v1"
)
CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT: Final = "prob4d_gain"
CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT: Final = (
    "recovery_fraction_when_recurrence_gap_separated"
)
CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_CLAIM_BOUNDARY: Final = (
    "This prospective specification was frozen before CUT3R source outcomes were "
    "opened. It authorizes only a descriptive source-only analysis of already "
    "registered three-arm evidence. It does not alter a provider, source gate, "
    "readiness decision, target protocol, BayesianPhysTwin guard, or Causal4D "
    "handoff, and target access remains forbidden."
)
CUT3R_RECURRENT_STATE_RECOVERY_V2_CLAIM_BOUNDARY: Final = (
    "This source-only report is a descriptive three-arm mechanism analysis. "
    "Absolute Prob4D gain is primary. The recovery fraction is secondary and is "
    "reported only when the recurrent-state gap exceeds the prospectively frozen "
    "metric-specific practical-separation floor. Exact group-bootstrap and "
    "leave-one-group-out diagnostics do not select a candidate, alter readiness "
    "gates, authorize target access, establish BayesianPhysTwin or Causal4D "
    "benefit, establish deployment safety, or establish state of the art."
)

_SPEC_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "primary_endpoint",
        "secondary_endpoint",
        "interval_method",
        "confidence_level",
        "minimum_valid_denominator_probability",
        "maximum_exact_group_count",
        "minimum_recurrence_gap_by_metric",
        "minimum_recurrence_gap_rationale_by_metric",
        "leave_one_group_out",
        "source_outcomes_opened_before_specification",
        "target_access",
        "claim_boundary",
    }
)
_SPEC_FIELDS: Final = _SPEC_PAYLOAD_FIELDS | {"analysis_specification_id"}


def _nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty exact string without surrounding whitespace"
        )
    return cast(str, value)


def _normalized_specification_payload(value: Any) -> dict[str, Any]:
    specification = _strict_mapping(
        value,
        name="CUT3R recurrent-state recovery v2 specification",
    )
    _exact_keys(
        specification,
        _SPEC_PAYLOAD_FIELDS,
        name="CUT3R recurrent-state recovery v2 specification",
    )
    if specification["schema"] != CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_SCHEMA:
        raise ValueError("unexpected CUT3R recurrent-state recovery v2 spec schema")
    schema_version = _strict_integer(
        specification["schema_version"],
        name="schema_version",
        minimum=1,
    )
    if schema_version != CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_VERSION:
        raise ValueError("unexpected CUT3R recurrent-state recovery v2 spec version")
    if (
        specification["primary_endpoint"]
        != CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT
    ):
        raise ValueError("v2 primary endpoint must be absolute Prob4D gain")
    if (
        specification["secondary_endpoint"]
        != CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT
    ):
        raise ValueError("v2 secondary endpoint must be denominator-safe recovery")
    if (
        specification["interval_method"]
        != CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD
    ):
        raise ValueError("unexpected v2 exact interval method")
    confidence_level = _finite_float(
        specification["confidence_level"],
        name="confidence_level",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if confidence_level <= 0.0:
        raise ValueError("confidence_level must be > 0")
    minimum_valid_probability = _finite_float(
        specification["minimum_valid_denominator_probability"],
        name="minimum_valid_denominator_probability",
        minimum=0.0,
        maximum=1.0,
    )
    if minimum_valid_probability <= 0.0:
        raise ValueError("minimum_valid_denominator_probability must be > 0")
    maximum_exact_group_count = _strict_integer(
        specification["maximum_exact_group_count"],
        name="maximum_exact_group_count",
        minimum=2,
    )
    if maximum_exact_group_count > 10:
        raise ValueError("maximum_exact_group_count must be <= 10")
    if specification["leave_one_group_out"] is not True:
        raise ValueError("leave_one_group_out must be true")
    if specification["source_outcomes_opened_before_specification"] is not False:
        raise ValueError(
            "source_outcomes_opened_before_specification must be false"
        )
    if specification["target_access"] != "forbidden":
        raise ValueError("target_access must remain forbidden")
    if (
        specification["claim_boundary"]
        != CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_CLAIM_BOUNDARY
    ):
        raise ValueError("unexpected v2 specification claim boundary")

    gaps = _strict_mapping(
        specification["minimum_recurrence_gap_by_metric"],
        name="minimum_recurrence_gap_by_metric",
    )
    rationales = _strict_mapping(
        specification["minimum_recurrence_gap_rationale_by_metric"],
        name="minimum_recurrence_gap_rationale_by_metric",
    )
    metric_fields = set(CUT3R_RECURRENT_STATE_RECOVERY_METRICS)
    _exact_keys(gaps, metric_fields, name="minimum_recurrence_gap_by_metric")
    _exact_keys(
        rationales,
        metric_fields,
        name="minimum_recurrence_gap_rationale_by_metric",
    )
    normalized_gaps: dict[str, float] = {}
    normalized_rationales: dict[str, str] = {}
    for metric in CUT3R_RECURRENT_STATE_RECOVERY_METRICS:
        gap = _finite_float(
            gaps[metric],
            name=f"minimum_recurrence_gap_by_metric.{metric}",
            minimum=0.0,
        )
        if gap <= 0.0:
            raise ValueError(
                f"minimum_recurrence_gap_by_metric.{metric} must be strictly positive"
            )
        normalized_gaps[metric] = gap
        normalized_rationales[metric] = _nonempty_string(
            rationales[metric],
            name=f"minimum_recurrence_gap_rationale_by_metric.{metric}",
        )

    return {
        "schema": CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_SCHEMA,
        "schema_version": CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_VERSION,
        "primary_endpoint": CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT,
        "secondary_endpoint": CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT,
        "interval_method": CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
        "confidence_level": confidence_level,
        "minimum_valid_denominator_probability": minimum_valid_probability,
        "maximum_exact_group_count": maximum_exact_group_count,
        "minimum_recurrence_gap_by_metric": normalized_gaps,
        "minimum_recurrence_gap_rationale_by_metric": normalized_rationales,
        "leave_one_group_out": True,
        "source_outcomes_opened_before_specification": False,
        "target_access": "forbidden",
        "claim_boundary": CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_CLAIM_BOUNDARY,
    }


def build_cut3r_recurrent_state_recovery_v2_specification(value: Any) -> dict[str, Any]:
    """Build a content-addressed prospective v2 analysis specification."""

    payload = _normalized_specification_payload(value)
    payload["analysis_specification_id"] = _record_id(payload)
    return cast(dict[str, Any], json.loads(_canonical_json(payload)))


def validate_cut3r_recurrent_state_recovery_v2_specification(
    value: Any,
) -> dict[str, Any]:
    """Validate a content-addressed prospective v2 analysis specification."""

    supplied = _strict_mapping(
        value,
        name="CUT3R recurrent-state recovery v2 specification",
    )
    _exact_keys(
        supplied,
        _SPEC_FIELDS,
        name="CUT3R recurrent-state recovery v2 specification",
    )
    payload = {key: supplied[key] for key in _SPEC_PAYLOAD_FIELDS}
    expected = build_cut3r_recurrent_state_recovery_v2_specification(payload)
    if dict(supplied) != expected:
        raise ValueError("CUT3R recurrent-state recovery v2 specification is invalid")
    return expected


def load_cut3r_recurrent_state_recovery_v2_specification(
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_recurrent_state_recovery_v2_specification(_strict_json(path))


__all__ = [
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_CLAIM_BOUNDARY",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SCHEMA",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_CLAIM_BOUNDARY",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_SCHEMA",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_VERSION",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_VERSION",
    "build_cut3r_recurrent_state_recovery_v2_specification",
    "load_cut3r_recurrent_state_recovery_v2_specification",
    "validate_cut3r_recurrent_state_recovery_v2_specification",
]
