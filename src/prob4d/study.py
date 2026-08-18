"""High-level target-free orchestration for one held-out provider study.

This module composes existing finite-sample capability checks with the additive
source-derived sensitivity audit. It deliberately stops before provider target
I/O: claim-bearing qualification, evaluation, and verification remain owned by
the existing frozen commands and artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._finite_sample_capability_common import DEFAULT_COVERAGE_LEVELS, coverage
from ._finite_sample_capability_model import FiniteSampleCapabilityV1
from ._finite_sample_capability_output import write_finite_sample_capability
from ._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    load_promotion_lock,
)
from .deform360_cohort_binding import (
    Deform360OfficialHubCohortBindingV1,
    load_deform360_cohort_binding,
)
from .finite_sample_capability import (
    build_finite_sample_capability,
    build_finite_sample_capability_from_cohort_binding,
)
from .study_sensitivity import (
    ALTERNATIVES,
    DEFAULT_POWER_LEVELS,
    PairedDifferenceScenarioV1,
    StudySensitivityV1,
    build_study_sensitivity,
    write_study_sensitivity,
)


@dataclass(frozen=True, slots=True)
class StudyPreflightV1:
    """In-memory index over the two target-free preflight artifacts."""

    capability: FiniteSampleCapabilityV1
    sensitivity: StudySensitivityV1

    def __post_init__(self) -> None:
        if self.capability.promotion_lock_id != self.sensitivity.promotion_lock_id:
            raise ValueError("preflight artifacts bind different promotion locks")
        if self.capability.target_group_ids != self.sensitivity.target_group_ids:
            raise ValueError("preflight artifacts bind different target groups")

    @property
    def promotion_lock_id(self) -> str:
        return self.capability.promotion_lock_id

    @property
    def target_group_count(self) -> int:
        return len(self.capability.target_group_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "promotion_lock_id": self.promotion_lock_id,
            "finite_sample_capability_id": self.capability.capability_id,
            "study_sensitivity_id": self.sensitivity.sensitivity_id,
            "target_group_count": self.target_group_count,
            "primary_levels_finite": self.capability.primary_levels_finite,
            "stratum_levels_finite": self.capability.stratum_levels_finite,
            "query_margin_detectable": self.sensitivity.query_margin_detectable,
            "target_outcomes_opened": False,
        }


@dataclass(frozen=True, slots=True)
class HeldoutProviderStudy:
    """Small read-only façade over one sealed held-out promotion lock."""

    lock: HeldoutProviderPromotionLockV1

    @classmethod
    def from_lock(cls, path: str | Path) -> HeldoutProviderStudy:
        return cls(load_promotion_lock(Path(path)))

    @property
    def promotion_lock_id(self) -> str:
        return self.lock.promotion_lock_id

    @property
    def group_counts(self) -> Mapping[str, int]:
        return {
            "development": len(self.lock.development_group_ids),
            "calibration": len(self.lock.calibration_group_ids),
            "target": len(self.lock.target_group_ids),
        }

    def finite_sample_capability(
        self,
        *,
        coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
        cohort_binding: Deform360OfficialHubCohortBindingV1 | None = None,
    ) -> FiniteSampleCapabilityV1:
        """Build the existing group-count and split-conformal capability report."""

        if cohort_binding is None:
            return build_finite_sample_capability(
                self.lock,
                coverage_levels=coverage_levels,
            )
        return build_finite_sample_capability_from_cohort_binding(
            self.lock,
            cohort_binding,
            coverage_levels=coverage_levels,
        )

    def sensitivity(
        self,
        *,
        source_summary_id: str,
        source_metric: str,
        paired_difference_scenarios: Sequence[PairedDifferenceScenarioV1],
        power_levels: Sequence[float] = DEFAULT_POWER_LEVELS,
        confidence_level: float = 0.95,
        alternative: str = "two-sided",
        accepted_group_counts: Sequence[int] | None = None,
    ) -> StudySensitivityV1:
        """Build the source-bound target-resolution report."""

        return build_study_sensitivity(
            self.lock,
            source_summary_id=source_summary_id,
            source_metric=source_metric,
            paired_difference_scenarios=paired_difference_scenarios,
            power_levels=power_levels,
            confidence_level=confidence_level,
            alternative=alternative,
            accepted_group_counts=accepted_group_counts,
        )

    def preflight(
        self,
        *,
        source_summary_id: str,
        source_metric: str,
        paired_difference_scenarios: Sequence[PairedDifferenceScenarioV1],
        coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
        power_levels: Sequence[float] = DEFAULT_POWER_LEVELS,
        confidence_level: float = 0.95,
        alternative: str = "two-sided",
        accepted_group_counts: Sequence[int] | None = None,
        cohort_binding: Deform360OfficialHubCohortBindingV1 | None = None,
    ) -> StudyPreflightV1:
        """Compose both target-free design reports without opening target outcomes."""

        return StudyPreflightV1(
            capability=self.finite_sample_capability(
                coverage_levels=coverage_levels,
                cohort_binding=cohort_binding,
            ),
            sensitivity=self.sensitivity(
                source_summary_id=source_summary_id,
                source_metric=source_metric,
                paired_difference_scenarios=paired_difference_scenarios,
                power_levels=power_levels,
                confidence_level=confidence_level,
                alternative=alternative,
                accepted_group_counts=accepted_group_counts,
            ),
        )


def write_study_preflight(preflight: StudyPreflightV1, output_dir: str | Path) -> None:
    """Publish both component reports after one all-destination no-clobber check."""

    directory = Path(output_dir)
    capability_json = directory / "finite_sample_capability.json"
    capability_markdown = directory / "finite_sample_capability.md"
    sensitivity_json = directory / "study_sensitivity.json"
    sensitivity_markdown = directory / "study_sensitivity.md"
    destinations = (
        capability_json,
        capability_markdown,
        sensitivity_json,
        sensitivity_markdown,
    )
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        raise FileExistsError(
            "study preflight output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    write_finite_sample_capability(
        preflight.capability,
        capability_json,
        markdown=capability_markdown,
    )
    write_study_sensitivity(
        preflight.sensitivity,
        sensitivity_json,
        markdown=sensitivity_markdown,
    )


def _coverage_argument(value: str) -> float:
    try:
        return coverage(float(value), name="coverage")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _power_argument(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("power must be a real number") from error
    if not 0.5 < result < 1.0:
        raise argparse.ArgumentTypeError("power must be strictly between 0.5 and 1.0")
    return result


def _scenario_argument(value: str) -> PairedDifferenceScenarioV1:
    scenario_id, separator, raw_standard_deviation = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("paired SD must use SCENARIO_ID=STANDARD_DEVIATION_MM")
    try:
        return PairedDifferenceScenarioV1(scenario_id, float(raw_standard_deviation))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run one high-level target-free study preflight."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--cohort-binding", type=Path)
    parser.add_argument("--source-summary-id", required=True)
    parser.add_argument("--source-metric", required=True)
    parser.add_argument(
        "--paired-sd",
        type=_scenario_argument,
        action="append",
        required=True,
        dest="paired_scenarios",
        metavar="SCENARIO=MM",
    )
    parser.add_argument("--coverage", type=_coverage_argument, action="append")
    parser.add_argument("--power", type=_power_argument, action="append")
    parser.add_argument("--confidence", type=_power_argument, default=0.95)
    parser.add_argument("--alternative", choices=ALTERNATIVES, default="two-sided")
    parser.add_argument(
        "--accepted-groups",
        type=int,
        action="append",
        dest="accepted_group_counts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-primary-finite", action="store_true")
    parser.add_argument("--require-strata-finite", action="store_true")
    arguments = parser.parse_args(argv)

    study = HeldoutProviderStudy.from_lock(arguments.lock)
    cohort_binding = (
        None
        if arguments.cohort_binding is None
        else load_deform360_cohort_binding(arguments.cohort_binding)
    )
    preflight = study.preflight(
        source_summary_id=arguments.source_summary_id,
        source_metric=arguments.source_metric,
        paired_difference_scenarios=arguments.paired_scenarios,
        coverage_levels=(
            DEFAULT_COVERAGE_LEVELS
            if arguments.coverage is None
            else arguments.coverage
        ),
        power_levels=DEFAULT_POWER_LEVELS if arguments.power is None else arguments.power,
        confidence_level=arguments.confidence,
        alternative=arguments.alternative,
        accepted_group_counts=arguments.accepted_group_counts,
        cohort_binding=cohort_binding,
    )
    write_study_preflight(preflight, arguments.output_dir)
    print(json.dumps(preflight.to_dict(), sort_keys=True))

    if arguments.require_primary_finite and not preflight.capability.primary_levels_finite:
        return 3
    if (
        arguments.require_strata_finite
        and preflight.capability.stratum_levels_finite is not True
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HeldoutProviderStudy",
    "StudyPreflightV1",
    "main",
    "write_study_preflight",
]
