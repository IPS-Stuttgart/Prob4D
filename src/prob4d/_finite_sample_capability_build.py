"""Target-free builders for finite-sample capability reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._finite_sample_capability_common import DEFAULT_COVERAGE_LEVELS
from ._finite_sample_capability_model import FiniteSampleCapabilityV1
from ._finite_sample_capability_records import CalibrationStratumV1
from ._heldout_promotion_lock import HeldoutProviderPromotionLockV1
from .deform360_cohort_binding import Deform360OfficialHubCohortBindingV1


def build_finite_sample_capability(
    lock: HeldoutProviderPromotionLockV1,
    *,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    calibration_strata: Mapping[str, Sequence[str]] | None = None,
    cohort_binding_id: str | None = None,
) -> FiniteSampleCapabilityV1:
    """Build a target-free report from one sealed promotion lock."""

    strata = ()
    if calibration_strata is not None:
        strata = tuple(
            CalibrationStratumV1(name, tuple(group_ids))
            for name, group_ids in calibration_strata.items()
        )
    return FiniteSampleCapabilityV1(
        promotion_lock_id=lock.promotion_lock_id,
        cohort_binding_id=cohort_binding_id,
        calibration_group_ids=lock.calibration_group_ids,
        target_group_ids=lock.target_group_ids,
        requested_coverages=tuple(coverage_levels),
        bootstrap_resamples=lock.bootstrap_resamples,
        calibration_strata=strata,
        lock_minimum_mean_accepted_coverage=lock.minimum_mean_accepted_coverage,
    )


def build_finite_sample_capability_from_cohort_binding(
    lock: HeldoutProviderPromotionLockV1,
    cohort_binding: Deform360OfficialHubCohortBindingV1,
    *,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> FiniteSampleCapabilityV1:
    """Derive Deform360 strata after exact lock-to-cohort validation."""

    if cohort_binding.calibration_group_ids != lock.calibration_group_ids:
        raise ValueError("cohort binding calibration groups differ from promotion lock")
    if cohort_binding.target_group_ids != lock.target_group_ids:
        raise ValueError("cohort binding target groups differ from promotion lock")
    strata = {
        stratum: tuple(
            sorted(
                unit.object_id
                for unit in cohort_binding.calibration_units
                if unit.stratum == stratum
            )
        )
        for stratum in ("sheet", "volumetric")
    }
    return build_finite_sample_capability(
        lock,
        coverage_levels=coverage_levels,
        calibration_strata=strata,
        cohort_binding_id=cohort_binding.cohort_binding_id,
    )


__all__ = [
    "build_finite_sample_capability",
    "build_finite_sample_capability_from_cohort_binding",
]
