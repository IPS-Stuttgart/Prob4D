"""Derived population and target-design records for finite-sample audits."""

from __future__ import annotations

from dataclasses import dataclass

from ._finite_sample_capability_common import (
    split_conformal_level,
    strict_integer,
    strict_string,
    string_tuple,
)


@dataclass(frozen=True, slots=True)
class CalibrationStratumV1:
    """One target-free calibration stratum of complete group IDs."""

    stratum: str
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stratum", strict_string(self.stratum, name="stratum"))
        object.__setattr__(
            self,
            "group_ids",
            string_tuple(self.group_ids, name=f"{self.stratum} group_ids"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"stratum": self.stratum, "group_ids": list(self.group_ids)}


def population_record(
    population_id: str,
    group_ids: tuple[str, ...],
    requested_coverages: tuple[float, ...],
    *,
    stratum: str | None,
) -> dict[str, object]:
    count = len(group_ids)
    return {
        "population_id": population_id,
        "stratum": stratum,
        "group_ids": list(group_ids),
        "group_count": count,
        "maximum_finite_coverage": count / (count + 1),
        "levels": [split_conformal_level(count, requested) for requested in requested_coverages],
    }


def target_design_record(group_count: int, bootstrap_resamples: int) -> dict[str, object]:
    count = strict_integer(group_count, name="target_group_count", minimum=1)
    resamples = strict_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
        minimum=100,
    )
    one_sided = 0.5**count
    return {
        "target_group_count": count,
        "bootstrap_resamples": resamples,
        "bootstrap_empirical_mass_resolution": 1.0 / resamples,
        "leave_one_group_out_replications": count,
        "all_favorable_one_sided_sign_probability": one_sided,
        "all_favorable_two_sided_sign_probability": min(1.0, 2.0 * one_sided),
    }


__all__ = [
    "CalibrationStratumV1",
    "population_record",
    "target_design_record",
]
