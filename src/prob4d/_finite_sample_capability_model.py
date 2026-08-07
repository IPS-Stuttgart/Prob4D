"""Immutable content-addressed finite-sample capability report."""

from __future__ import annotations

from dataclasses import dataclass

from ._finite_sample_capability_derive import capability_id, descriptor, populations
from ._finite_sample_capability_records import CalibrationStratumV1
from ._finite_sample_capability_state import validate_capability_state


@dataclass(frozen=True, slots=True)
class FiniteSampleCapabilityV1:
    """Capability report derived only from sealed independent-group metadata."""

    promotion_lock_id: str
    calibration_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    requested_coverages: tuple[float, ...]
    bootstrap_resamples: int
    calibration_strata: tuple[CalibrationStratumV1, ...] = ()
    cohort_binding_id: str | None = None
    lock_minimum_mean_accepted_coverage: float | None = None

    def __post_init__(self) -> None:
        values = validate_capability_state(
            promotion_lock_id=self.promotion_lock_id,
            cohort_binding_id=self.cohort_binding_id,
            calibration_group_ids=self.calibration_group_ids,
            target_group_ids=self.target_group_ids,
            requested_coverages=self.requested_coverages,
            bootstrap_resamples=self.bootstrap_resamples,
            calibration_strata=self.calibration_strata,
            minimum_accepted_coverage=self.lock_minimum_mean_accepted_coverage,
        )
        fields = (
            "promotion_lock_id",
            "cohort_binding_id",
            "calibration_group_ids",
            "target_group_ids",
            "requested_coverages",
            "bootstrap_resamples",
            "calibration_strata",
            "lock_minimum_mean_accepted_coverage",
        )
        for field, value in zip(fields, values, strict=True):
            object.__setattr__(self, field, value)

    @property
    def populations(self) -> tuple[dict[str, object], ...]:
        return populations(self)

    @property
    def primary_levels_finite(self) -> bool:
        return bool(descriptor(self)["primary_levels_finite"])

    @property
    def stratum_levels_finite(self) -> bool | None:
        value = descriptor(self)["stratum_levels_finite"]
        assert value is None or type(value) is bool
        return value

    @property
    def all_levels_finite(self) -> bool:
        return bool(descriptor(self)["all_levels_finite"])

    def descriptor(self) -> dict[str, object]:
        return descriptor(self)

    @property
    def capability_id(self) -> str:
        return capability_id(self)

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "capability_id": self.capability_id}


__all__ = ["FiniteSampleCapabilityV1"]
