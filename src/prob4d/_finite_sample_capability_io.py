"""Strict loading for finite-sample capability reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._finite_sample_capability_common import (
    FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY,
    FINITE_SAMPLE_CAPABILITY_SCHEMA,
    FINITE_SAMPLE_CAPABILITY_VERSION,
    canonical_json,
    coverage,
    exact_fields,
    load_json,
    strict_digest,
    strict_integer,
    strict_string,
    string_tuple,
)
from ._finite_sample_capability_model import FiniteSampleCapabilityV1
from ._finite_sample_capability_records import CalibrationStratumV1

_REPORT_FIELDS = {
    "schema_name",
    "schema_version",
    "promotion_lock_id",
    "cohort_binding_id",
    "calibration_group_ids",
    "target_group_ids",
    "requested_coverages",
    "lock_minimum_mean_accepted_coverage",
    "bootstrap_resamples",
    "calibration_strata",
    "populations",
    "target_design",
    "primary_levels_finite",
    "stratum_levels_finite",
    "all_levels_finite",
    "claim_boundary",
    "capability_id",
}
_STRATUM_FIELDS = {"stratum", "group_ids"}


def finite_sample_capability_from_dict(value: object) -> FiniteSampleCapabilityV1:
    """Strictly load and independently recompute every derived field."""

    if not isinstance(value, Mapping):
        raise ValueError("finite-sample capability report must be an object")
    exact_fields(value, _REPORT_FIELDS, name="finite-sample capability report")
    if value["schema_name"] != FINITE_SAMPLE_CAPABILITY_SCHEMA:
        raise ValueError("unsupported finite-sample capability schema")
    if (
        strict_integer(value["schema_version"], name="schema_version", minimum=1)
        != FINITE_SAMPLE_CAPABILITY_VERSION
    ):
        raise ValueError("unsupported finite-sample capability version")
    if value["claim_boundary"] != FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY:
        raise ValueError("finite-sample capability claim boundary changed")

    raw_strata = value["calibration_strata"]
    if not isinstance(raw_strata, list):
        raise ValueError("calibration_strata must be a list")
    strata: list[CalibrationStratumV1] = []
    for index, raw in enumerate(raw_strata):
        if not isinstance(raw, Mapping):
            raise ValueError(f"calibration_strata[{index}] must be an object")
        exact_fields(raw, _STRATUM_FIELDS, name=f"calibration_strata[{index}]")
        strata.append(
            CalibrationStratumV1(
                strict_string(
                    raw["stratum"],
                    name=f"calibration_strata[{index}].stratum",
                ),
                string_tuple(
                    raw["group_ids"],
                    name=f"calibration_strata[{index}].group_ids",
                ),
            )
        )

    raw_coverages = value["requested_coverages"]
    if not isinstance(raw_coverages, list):
        raise ValueError("requested_coverages must be a list")
    minimum_coverage = value["lock_minimum_mean_accepted_coverage"]
    report = FiniteSampleCapabilityV1(
        promotion_lock_id=value["promotion_lock_id"],
        cohort_binding_id=(
            None
            if value["cohort_binding_id"] is None
            else strict_digest(value["cohort_binding_id"], name="cohort_binding_id")
        ),
        calibration_group_ids=string_tuple(
            value["calibration_group_ids"],
            name="calibration_group_ids",
        ),
        target_group_ids=string_tuple(value["target_group_ids"], name="target_group_ids"),
        requested_coverages=tuple(
            coverage(item, name=f"requested_coverages[{index}]")
            for index, item in enumerate(raw_coverages)
        ),
        bootstrap_resamples=strict_integer(
            value["bootstrap_resamples"],
            name="bootstrap_resamples",
            minimum=100,
        ),
        calibration_strata=tuple(strata),
        lock_minimum_mean_accepted_coverage=(
            None
            if minimum_coverage is None
            else coverage(minimum_coverage, name="lock_minimum_mean_accepted_coverage")
        ),
    )
    strict_digest(value["capability_id"], name="capability_id")
    if canonical_json(value) != canonical_json(report.to_dict()):
        raise ValueError("finite-sample capability fields do not match recomputation")
    return report


def load_finite_sample_capability(path: Path) -> FiniteSampleCapabilityV1:
    """Load and validate one retained report."""

    return finite_sample_capability_from_dict(load_json(path))


__all__ = [
    "finite_sample_capability_from_dict",
    "load_finite_sample_capability",
]
