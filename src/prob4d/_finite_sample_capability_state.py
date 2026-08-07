"""Validation of immutable finite-sample capability constructor state."""

from __future__ import annotations

from ._finite_sample_capability_common import (
    coverage,
    coverages,
    strict_digest,
    strict_integer,
    string_tuple,
)
from ._finite_sample_capability_records import CalibrationStratumV1


def validate_capability_state(
    *,
    promotion_lock_id: object,
    cohort_binding_id: object,
    calibration_group_ids: object,
    target_group_ids: object,
    requested_coverages: tuple[float, ...],
    bootstrap_resamples: object,
    calibration_strata: object,
    minimum_accepted_coverage: object,
) -> tuple[
    str,
    str | None,
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
    int,
    tuple[CalibrationStratumV1, ...],
    float | None,
]:
    """Validate and canonicalize all content-bearing constructor inputs."""

    lock_id = strict_digest(promotion_lock_id, name="promotion_lock_id")
    binding_id = (
        None
        if cohort_binding_id is None
        else strict_digest(cohort_binding_id, name="cohort_binding_id")
    )
    calibration = string_tuple(calibration_group_ids, name="calibration_group_ids")
    target = string_tuple(target_group_ids, name="target_group_ids")
    if set(calibration) & set(target):
        raise ValueError("calibration and target groups must be disjoint")
    levels = coverages(requested_coverages)
    resamples = strict_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
        minimum=100,
    )
    accepted_coverage = (
        None
        if minimum_accepted_coverage is None
        else coverage(
            minimum_accepted_coverage,
            name="lock_minimum_mean_accepted_coverage",
        )
    )
    if type(calibration_strata) is not tuple or not all(
        isinstance(item, CalibrationStratumV1) for item in calibration_strata
    ):
        raise ValueError("calibration_strata must contain CalibrationStratumV1 values")
    strata = tuple(sorted(calibration_strata, key=lambda item: item.stratum))
    names = tuple(item.stratum for item in strata)
    if len(names) != len(set(names)):
        raise ValueError("calibration stratum names must be unique")
    if strata:
        flattened = tuple(group_id for item in strata for group_id in item.group_ids)
        if len(flattened) != len(set(flattened)):
            raise ValueError("calibration strata must be disjoint")
        if set(flattened) != set(calibration):
            raise ValueError("calibration strata must partition all calibration groups")
    return (
        lock_id,
        binding_id,
        calibration,
        target,
        levels,
        resamples,
        strata,
        accepted_coverage,
    )


__all__ = ["validate_capability_state"]
