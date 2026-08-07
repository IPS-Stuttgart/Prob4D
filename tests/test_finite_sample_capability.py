from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from prob4d.finite_sample_capability import (
    FINITE_SAMPLE_CAPABILITY_SCHEMA,
    build_finite_sample_capability,
    build_finite_sample_capability_from_cohort_binding,
    split_conformal_level,
)


def _lock() -> SimpleNamespace:
    return SimpleNamespace(
        promotion_lock_id="a" * 64,
        calibration_group_ids=tuple(f"calibration-{index:02d}" for index in range(10)),
        target_group_ids=tuple(f"target-{index:02d}" for index in range(12)),
        bootstrap_resamples=5000,
        minimum_mean_accepted_coverage=0.90,
    )


def _strata() -> dict[str, tuple[str, ...]]:
    lock = _lock()
    return {
        "sheet": lock.calibration_group_ids[:5],
        "volumetric": lock.calibration_group_ids[5:],
    }


def test_split_conformal_rank_availability_is_exact() -> None:
    ninety = split_conformal_level(10, 0.90)
    ninety_five = split_conformal_level(10, 0.95)
    stratum_ninety = split_conformal_level(5, 0.90)

    assert ninety == {
        "nominal_coverage": 0.90,
        "alpha": 0.10,
        "order_statistic_rank": 10,
        "finite_threshold": True,
        "guaranteed_coverage_lower_bound": 10 / 11,
        "minimum_group_count_for_finite_threshold": 9,
    }
    assert ninety_five["order_statistic_rank"] == 11
    assert ninety_five["finite_threshold"] is False
    assert ninety_five["guaranteed_coverage_lower_bound"] is None
    assert ninety_five["minimum_group_count_for_finite_threshold"] == 19
    assert stratum_ninety["order_statistic_rank"] == 6
    assert stratum_ninety["finite_threshold"] is False


def test_capability_separates_primary_and_stratum_limits() -> None:
    report = build_finite_sample_capability(
        _lock(),
        coverage_levels=(0.90, 0.95),
        calibration_strata=_strata(),
    )

    assert report.to_dict()["schema_name"] == FINITE_SAMPLE_CAPABILITY_SCHEMA
    assert report.primary_levels_finite is False
    assert report.stratum_levels_finite is False
    assert report.all_levels_finite is False
    populations = report.to_dict()["populations"]
    assert isinstance(populations, list)
    assert [population["group_count"] for population in populations] == [10, 5, 5]
    assert populations[0]["maximum_finite_coverage"] == 10 / 11
    assert populations[1]["maximum_finite_coverage"] == 5 / 6
    target = report.to_dict()["target_design"]
    assert target["target_group_count"] == 12
    assert target["bootstrap_empirical_mass_resolution"] == 1 / 5000
    assert target["all_favorable_one_sided_sign_probability"] == 1 / 4096


def test_lock_coverage_threshold_is_recorded_without_changing_requested_levels() -> None:
    report = build_finite_sample_capability(
        _lock(),
        coverage_levels=(0.80, 0.95),
    )

    assert report.requested_coverages == (0.80, 0.95)
    assert report.lock_minimum_mean_accepted_coverage == 0.90
    assert report.stratum_levels_finite is None


def test_strata_must_form_one_disjoint_partition() -> None:
    lock = _lock()
    with pytest.raises(ValueError, match="partition all calibration groups"):
        build_finite_sample_capability(
            lock,
            calibration_strata={"sheet": lock.calibration_group_ids[:5]},
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        build_finite_sample_capability(
            lock,
            calibration_strata={
                "sheet": lock.calibration_group_ids[:5],
                "volumetric": lock.calibration_group_ids[4:],
            },
        )


def test_deform360_binding_derives_strata_and_checks_exact_groups() -> None:
    lock = _lock()
    units = tuple(
        SimpleNamespace(
            object_id=group_id,
            stratum="sheet" if index < 5 else "volumetric",
        )
        for index, group_id in enumerate(lock.calibration_group_ids)
    )
    binding = SimpleNamespace(
        calibration_group_ids=lock.calibration_group_ids,
        target_group_ids=lock.target_group_ids,
        calibration_units=units,
        cohort_binding_id="b" * 64,
    )

    report = build_finite_sample_capability_from_cohort_binding(
        lock,
        binding,
        coverage_levels=(0.90,),
    )

    assert report.cohort_binding_id == "b" * 64
    assert tuple(item.stratum for item in report.calibration_strata) == (
        "sheet",
        "volumetric",
    )
    assert report.primary_levels_finite is True
    assert report.stratum_levels_finite is False

    changed = copy.copy(binding)
    changed.target_group_ids = lock.target_group_ids[:-1]
    with pytest.raises(ValueError, match="target groups differ"):
        build_finite_sample_capability_from_cohort_binding(lock, changed)
