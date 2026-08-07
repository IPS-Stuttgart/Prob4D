"""Deterministic fields derived from finite-sample capability state."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Protocol

from ._finite_sample_capability_common import (
    FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY,
    FINITE_SAMPLE_CAPABILITY_SCHEMA,
    FINITE_SAMPLE_CAPABILITY_VERSION,
    canonical_json,
)
from ._finite_sample_capability_records import (
    CalibrationStratumV1,
    population_record,
    target_design_record,
)


class CapabilityState(Protocol):
    @property
    def promotion_lock_id(self) -> str: ...

    @property
    def cohort_binding_id(self) -> str | None: ...

    @property
    def calibration_group_ids(self) -> tuple[str, ...]: ...

    @property
    def target_group_ids(self) -> tuple[str, ...]: ...

    @property
    def requested_coverages(self) -> tuple[float, ...]: ...

    @property
    def bootstrap_resamples(self) -> int: ...

    @property
    def calibration_strata(self) -> tuple[CalibrationStratumV1, ...]: ...

    @property
    def lock_minimum_mean_accepted_coverage(self) -> float | None: ...


def populations(state: CapabilityState) -> tuple[dict[str, object], ...]:
    result = [
        population_record(
            "calibration-all",
            state.calibration_group_ids,
            state.requested_coverages,
            stratum=None,
        )
    ]
    result.extend(
        population_record(
            f"calibration-{item.stratum}",
            item.group_ids,
            state.requested_coverages,
            stratum=item.stratum,
        )
        for item in state.calibration_strata
    )
    return tuple(result)


def population_is_finite(population: Mapping[str, object]) -> bool:
    levels = population["levels"]
    assert isinstance(levels, list)
    return all(isinstance(level, Mapping) and level["finite_threshold"] is True for level in levels)


def descriptor(state: CapabilityState) -> dict[str, object]:
    current_populations = populations(state)
    stratum_populations = current_populations[1:]
    stratum_finite = (
        None
        if not stratum_populations
        else all(population_is_finite(item) for item in stratum_populations)
    )
    return {
        "schema_name": FINITE_SAMPLE_CAPABILITY_SCHEMA,
        "schema_version": FINITE_SAMPLE_CAPABILITY_VERSION,
        "promotion_lock_id": state.promotion_lock_id,
        "cohort_binding_id": state.cohort_binding_id,
        "calibration_group_ids": list(state.calibration_group_ids),
        "target_group_ids": list(state.target_group_ids),
        "requested_coverages": list(state.requested_coverages),
        "lock_minimum_mean_accepted_coverage": (state.lock_minimum_mean_accepted_coverage),
        "bootstrap_resamples": state.bootstrap_resamples,
        "calibration_strata": [item.to_dict() for item in state.calibration_strata],
        "populations": list(current_populations),
        "target_design": target_design_record(
            len(state.target_group_ids),
            state.bootstrap_resamples,
        ),
        "primary_levels_finite": population_is_finite(current_populations[0]),
        "stratum_levels_finite": stratum_finite,
        "all_levels_finite": all(population_is_finite(item) for item in current_populations),
        "claim_boundary": FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY,
    }


def capability_id(state: CapabilityState) -> str:
    return hashlib.sha256(canonical_json(descriptor(state))).hexdigest()


__all__ = [
    "capability_id",
    "descriptor",
    "population_is_finite",
    "populations",
]
