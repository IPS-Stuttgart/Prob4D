"""Target-free sensitivity audit for held-out provider studies.

The audit binds source-derived paired-difference dispersion to one sealed
promotion lock and reports the effect sizes and harmful-update rates that the
frozen target group count can resolve. It never reads provider payloads, target
predictions, physical-query outcomes, or target metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Protocol, cast

from ._atomic_file import atomic_write_text
from ._heldout_promotion_lock import load_promotion_lock
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_json_number,
    require_mapping,
    require_sha256,
    require_string_sequence,
)

STUDY_SENSITIVITY_SCHEMA = "prob4d.study-sensitivity"
STUDY_SENSITIVITY_VERSION = 1
STUDY_SENSITIVITY_CLAIM_BOUNDARY = (
    "This target-free design audit uses source-declared paired standard deviations and "
    "a transparent normal approximation to describe statistical resolution. It does "
    "not open target outcomes, alter the frozen target decision, establish normality "
    "or exchangeability, prove provider competence, establish BayesianPhysTwin or "
    "Causal4D benefit, calibrate deployment uncertainty, or support a state-of-the-art "
    "claim."
)
DEFAULT_POWER_LEVELS = (0.80, 0.90)
ALTERNATIVES = ("one-sided", "two-sided")


class PromotionLockLike(Protocol):
    """Minimal target-free lock surface required by this audit."""

    promotion_lock_id: str
    target_group_ids: tuple[str, ...]
    minimum_target_group_count: int
    query_superiority_margin_mm: float
    harmful_update_margin_mm: float
    maximum_harmful_accepted_updates: int
    maximum_worst_group_regression_mm: float


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_positive(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _strict_nonnegative(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _strict_probability(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if not 0.5 < result < 1.0:
        raise ValueError(f"{name} must be strictly between 0.5 and 1.0")
    return result


def _canonical_group_ids(value: object, *, name: str) -> tuple[str, ...]:
    result = require_string_sequence(value, name=name)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _canonical_probabilities(value: object, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of probabilities")
    normalized = {
        _strict_probability(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    }
    result = tuple(sorted(normalized))
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _canonical_counts(
    value: object,
    *,
    name: str,
    maximum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of integers")
    result = tuple(
        sorted(
            {
                require_exact_integer(item, name=f"{name}[{index}]", minimum=1)
                for index, item in enumerate(value)
            }
        )
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if result[-1] > maximum:
        raise ValueError(f"{name} cannot exceed the frozen target group count")
    return result


def _normal_critical_value(confidence_level: float, alternative: str) -> float:
    alpha = 1.0 - confidence_level
    probability = 1.0 - alpha / 2.0 if alternative == "two-sided" else confidence_level
    return NormalDist().inv_cdf(probability)


def _binomial_cdf(events: int, trials: int, probability: float) -> float:
    """Return P[X <= events] with stable log-domain summation."""

    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if events >= trials else 0.0
    if events >= trials:
        return 1.0

    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_probability
        + (trials - index) * log_complement
        for index in range(events + 1)
    ]
    maximum = max(logs)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)


def one_sided_binomial_upper_bound(
    events: int,
    trials: int,
    confidence_level: float,
) -> float:
    """Return the exact Clopper--Pearson one-sided upper rate bound."""

    count = require_exact_integer(events, name="events", minimum=0)
    total = require_exact_integer(trials, name="trials", minimum=1)
    confidence = _strict_probability(confidence_level, name="confidence_level")
    if count > total:
        raise ValueError("events cannot exceed trials")
    if count == total:
        return 1.0
    alpha = 1.0 - confidence
    if count == 0:
        return 1.0 - alpha ** (1.0 / total)

    lower = count / total
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(count, total, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


@dataclass(frozen=True, slots=True)
class PairedDifferenceScenarioV1:
    """One source-derived paired-difference dispersion scenario."""

    scenario_id: str
    paired_standard_deviation_mm: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            require_exact_string(self.scenario_id, name="scenario_id"),
        )
        object.__setattr__(
            self,
            "paired_standard_deviation_mm",
            _strict_positive(
                self.paired_standard_deviation_mm,
                name="paired_standard_deviation_mm",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "paired_standard_deviation_mm": self.paired_standard_deviation_mm,
        }


@dataclass(frozen=True, slots=True)
class StudySensitivityV1:
    """Content-addressed target-free held-out-study sensitivity report."""

    promotion_lock_id: str
    source_summary_id: str
    source_metric: str
    target_group_ids: tuple[str, ...]
    paired_difference_scenarios: tuple[PairedDifferenceScenarioV1, ...]
    power_levels: tuple[float, ...] = DEFAULT_POWER_LEVELS
    confidence_level: float = 0.95
    alternative: str = "two-sided"
    accepted_group_counts: tuple[int, ...] = ()
    minimum_target_group_count: int = 1
    query_superiority_margin_mm: float = 0.0
    harmful_update_margin_mm: float = 0.0
    maximum_harmful_accepted_updates: int = 0
    maximum_worst_group_regression_mm: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "promotion_lock_id",
            require_sha256(self.promotion_lock_id, name="promotion_lock_id"),
        )
        object.__setattr__(
            self,
            "source_summary_id",
            require_sha256(self.source_summary_id, name="source_summary_id"),
        )
        object.__setattr__(
            self,
            "source_metric",
            require_exact_string(self.source_metric, name="source_metric"),
        )
        target_groups = _canonical_group_ids(self.target_group_ids, name="target_group_ids")
        object.__setattr__(self, "target_group_ids", target_groups)

        scenarios = self.paired_difference_scenarios
        if type(scenarios) is not tuple or not scenarios:
            raise ValueError("paired_difference_scenarios must be a nonempty tuple")
        if not all(isinstance(item, PairedDifferenceScenarioV1) for item in scenarios):
            raise ValueError(
                "paired_difference_scenarios must contain PairedDifferenceScenarioV1"
            )
        scenario_ids = tuple(item.scenario_id for item in scenarios)
        if scenario_ids != tuple(sorted(scenario_ids)) or len(set(scenario_ids)) != len(
            scenario_ids
        ):
            raise ValueError("paired_difference_scenarios must be sorted by unique scenario_id")

        object.__setattr__(
            self,
            "power_levels",
            _canonical_probabilities(self.power_levels, name="power_levels"),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _strict_probability(self.confidence_level, name="confidence_level"),
        )
        alternative = require_exact_string(self.alternative, name="alternative")
        if alternative not in ALTERNATIVES:
            raise ValueError(f"alternative must be one of {list(ALTERNATIVES)}")
        object.__setattr__(self, "alternative", alternative)

        accepted_counts = self.accepted_group_counts or (len(target_groups),)
        object.__setattr__(
            self,
            "accepted_group_counts",
            _canonical_counts(
                accepted_counts,
                name="accepted_group_counts",
                maximum=len(target_groups),
            ),
        )
        minimum_groups = require_exact_integer(
            self.minimum_target_group_count,
            name="minimum_target_group_count",
            minimum=1,
        )
        if len(target_groups) < minimum_groups:
            raise ValueError("target group count is below the frozen minimum")
        object.__setattr__(self, "minimum_target_group_count", minimum_groups)
        object.__setattr__(
            self,
            "query_superiority_margin_mm",
            _strict_nonnegative(
                self.query_superiority_margin_mm,
                name="query_superiority_margin_mm",
            ),
        )
        object.__setattr__(
            self,
            "harmful_update_margin_mm",
            _strict_nonnegative(
                self.harmful_update_margin_mm,
                name="harmful_update_margin_mm",
            ),
        )
        object.__setattr__(
            self,
            "maximum_harmful_accepted_updates",
            require_exact_integer(
                self.maximum_harmful_accepted_updates,
                name="maximum_harmful_accepted_updates",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_worst_group_regression_mm",
            _strict_nonnegative(
                self.maximum_worst_group_regression_mm,
                name="maximum_worst_group_regression_mm",
            ),
        )

    @property
    def target_group_count(self) -> int:
        return len(self.target_group_ids)

    @property
    def paired_effect_sensitivity(self) -> tuple[dict[str, object], ...]:
        critical = _normal_critical_value(self.confidence_level, self.alternative)
        records: list[dict[str, object]] = []
        for scenario in self.paired_difference_scenarios:
            standard_error = scenario.paired_standard_deviation_mm / math.sqrt(
                self.target_group_count
            )
            half_width = critical * standard_error
            for power in self.power_levels:
                detectable = (critical + NormalDist().inv_cdf(power)) * standard_error
                margin_detectable = (
                    None
                    if self.query_superiority_margin_mm == 0.0
                    else detectable <= self.query_superiority_margin_mm
                )
                records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "paired_standard_deviation_mm": (
                            scenario.paired_standard_deviation_mm
                        ),
                        "target_group_count": self.target_group_count,
                        "standard_error_mm": standard_error,
                        "confidence_interval_half_width_mm": half_width,
                        "power": power,
                        "minimum_detectable_effect_mm": detectable,
                        "standardized_minimum_detectable_effect": (
                            detectable / scenario.paired_standard_deviation_mm
                        ),
                        "query_superiority_margin_mm": self.query_superiority_margin_mm,
                        "query_margin_detectable": margin_detectable,
                    }
                )
        return tuple(records)

    @property
    def harmful_update_resolution(self) -> tuple[dict[str, object], ...]:
        records: list[dict[str, object]] = []
        for accepted_count in self.accepted_group_counts:
            allowed = min(self.maximum_harmful_accepted_updates, accepted_count)
            records.append(
                {
                    "accepted_group_count_scenario": accepted_count,
                    "one_event_rate_resolution": 1.0 / accepted_count,
                    "frozen_maximum_harmful_accepted_updates": (
                        self.maximum_harmful_accepted_updates
                    ),
                    "bounded_event_count": allowed,
                    "count_criterion_informative": allowed < accepted_count,
                    "zero_harm_one_sided_upper_rate_bound": (
                        one_sided_binomial_upper_bound(
                            0,
                            accepted_count,
                            self.confidence_level,
                        )
                    ),
                    "allowed_count_one_sided_upper_rate_bound": (
                        one_sided_binomial_upper_bound(
                            allowed,
                            accepted_count,
                            self.confidence_level,
                        )
                    ),
                }
            )
        return tuple(records)

    @property
    def target_design_resolution(self) -> dict[str, object]:
        count = self.target_group_count
        return {
            "target_group_count": count,
            "one_group_mass_resolution": 1.0 / count,
            "leave_one_group_out_replications": count,
            "smallest_one_sided_sign_probability": 0.5**count,
            "smallest_two_sided_sign_probability": min(1.0, 2.0 * 0.5**count),
        }

    @property
    def query_margin_detectable(self) -> bool | None:
        if self.query_superiority_margin_mm == 0.0:
            return None
        return all(
            cast(bool, record["query_margin_detectable"])
            for record in self.paired_effect_sensitivity
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": STUDY_SENSITIVITY_SCHEMA,
            "schema_version": STUDY_SENSITIVITY_VERSION,
            "promotion_lock_id": self.promotion_lock_id,
            "source_summary_id": self.source_summary_id,
            "source_metric": self.source_metric,
            "target_group_ids": list(self.target_group_ids),
            "paired_difference_scenarios": [
                scenario.to_dict() for scenario in self.paired_difference_scenarios
            ],
            "power_levels": list(self.power_levels),
            "confidence_level": self.confidence_level,
            "alternative": self.alternative,
            "accepted_group_counts": list(self.accepted_group_counts),
            "frozen_decision_thresholds": {
                "minimum_target_group_count": self.minimum_target_group_count,
                "query_superiority_margin_mm": self.query_superiority_margin_mm,
                "harmful_update_margin_mm": self.harmful_update_margin_mm,
                "maximum_harmful_accepted_updates": (
                    self.maximum_harmful_accepted_updates
                ),
                "maximum_worst_group_regression_mm": (
                    self.maximum_worst_group_regression_mm
                ),
            },
            "paired_effect_sensitivity": list(self.paired_effect_sensitivity),
            "harmful_update_resolution": list(self.harmful_update_resolution),
            "target_design_resolution": self.target_design_resolution,
            "query_margin_detectable": self.query_margin_detectable,
            "target_outcomes_opened": False,
            "claim_boundary": STUDY_SENSITIVITY_CLAIM_BOUNDARY,
        }

    @property
    def sensitivity_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "sensitivity_id": self.sensitivity_id}


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "promotion_lock_id",
        "source_summary_id",
        "source_metric",
        "target_group_ids",
        "paired_difference_scenarios",
        "power_levels",
        "confidence_level",
        "alternative",
        "accepted_group_counts",
        "frozen_decision_thresholds",
        "paired_effect_sensitivity",
        "harmful_update_resolution",
        "target_design_resolution",
        "query_margin_detectable",
        "target_outcomes_opened",
        "claim_boundary",
        "sensitivity_id",
    }
)
_THRESHOLD_FIELDS = frozenset(
    {
        "minimum_target_group_count",
        "query_superiority_margin_mm",
        "harmful_update_margin_mm",
        "maximum_harmful_accepted_updates",
        "maximum_worst_group_regression_mm",
    }
)
_SCENARIO_FIELDS = frozenset({"scenario_id", "paired_standard_deviation_mm"})


def _raw_sequence(value: object, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def build_study_sensitivity(
    lock: PromotionLockLike,
    *,
    source_summary_id: str,
    source_metric: str,
    paired_difference_scenarios: Sequence[PairedDifferenceScenarioV1],
    power_levels: Sequence[float] = DEFAULT_POWER_LEVELS,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
    accepted_group_counts: Sequence[int] | None = None,
) -> StudySensitivityV1:
    """Build one target-free report from a sealed promotion lock."""

    return StudySensitivityV1(
        promotion_lock_id=lock.promotion_lock_id,
        source_summary_id=source_summary_id,
        source_metric=source_metric,
        target_group_ids=tuple(lock.target_group_ids),
        paired_difference_scenarios=tuple(
            sorted(paired_difference_scenarios, key=lambda item: item.scenario_id)
        ),
        power_levels=tuple(power_levels),
        confidence_level=confidence_level,
        alternative=alternative,
        accepted_group_counts=(
            () if accepted_group_counts is None else tuple(accepted_group_counts)
        ),
        minimum_target_group_count=lock.minimum_target_group_count,
        query_superiority_margin_mm=lock.query_superiority_margin_mm,
        harmful_update_margin_mm=lock.harmful_update_margin_mm,
        maximum_harmful_accepted_updates=lock.maximum_harmful_accepted_updates,
        maximum_worst_group_regression_mm=lock.maximum_worst_group_regression_mm,
    )


def study_sensitivity_from_dict(value: Mapping[str, Any]) -> StudySensitivityV1:
    """Strictly reconstruct and replay a sensitivity report."""

    require_exact_fields(value, _TOP_LEVEL_FIELDS, name="study sensitivity report")
    if value["schema_name"] != STUDY_SENSITIVITY_SCHEMA:
        raise ValueError("unsupported study sensitivity schema")
    if value["schema_version"] != STUDY_SENSITIVITY_VERSION:
        raise ValueError("unsupported study sensitivity version")
    if value["target_outcomes_opened"] is not False:
        raise ValueError("study sensitivity report must remain target-free")
    if value["claim_boundary"] != STUDY_SENSITIVITY_CLAIM_BOUNDARY:
        raise ValueError("study sensitivity claim boundary changed")

    raw_scenarios = value["paired_difference_scenarios"]
    if isinstance(raw_scenarios, (str, bytes)) or not isinstance(raw_scenarios, Sequence):
        raise ValueError("paired_difference_scenarios must be a sequence")
    scenarios: list[PairedDifferenceScenarioV1] = []
    for index, raw in enumerate(raw_scenarios):
        mapping = require_mapping(raw, name=f"paired_difference_scenarios[{index}]")
        require_exact_fields(
            mapping,
            _SCENARIO_FIELDS,
            name=f"paired_difference_scenarios[{index}]",
        )
        scenarios.append(
            PairedDifferenceScenarioV1(
                scenario_id=mapping["scenario_id"],
                paired_standard_deviation_mm=mapping[
                    "paired_standard_deviation_mm"
                ],
            )
        )

    thresholds = require_mapping(
        value["frozen_decision_thresholds"],
        name="frozen_decision_thresholds",
    )
    require_exact_fields(
        thresholds,
        _THRESHOLD_FIELDS,
        name="frozen_decision_thresholds",
    )
    report = StudySensitivityV1(
        promotion_lock_id=value["promotion_lock_id"],
        source_summary_id=value["source_summary_id"],
        source_metric=value["source_metric"],
        target_group_ids=tuple(
            require_string_sequence(value["target_group_ids"], name="target_group_ids")
        ),
        paired_difference_scenarios=tuple(scenarios),
        power_levels=_raw_sequence(value["power_levels"], name="power_levels"),
        confidence_level=value["confidence_level"],
        alternative=value["alternative"],
        accepted_group_counts=_raw_sequence(
            value["accepted_group_counts"],
            name="accepted_group_counts",
        ),
        minimum_target_group_count=thresholds["minimum_target_group_count"],
        query_superiority_margin_mm=thresholds["query_superiority_margin_mm"],
        harmful_update_margin_mm=thresholds["harmful_update_margin_mm"],
        maximum_harmful_accepted_updates=thresholds[
            "maximum_harmful_accepted_updates"
        ],
        maximum_worst_group_regression_mm=thresholds[
            "maximum_worst_group_regression_mm"
        ],
    )
    if report.to_dict() != dict(value):
        raise ValueError("study sensitivity report failed deterministic replay")
    return report


def load_study_sensitivity(path: str | Path) -> StudySensitivityV1:
    """Load one strict content-addressed sensitivity report."""

    return study_sensitivity_from_dict(
        load_json_object(path, name="study sensitivity report")
    )


def render_study_sensitivity_markdown(report: StudySensitivityV1) -> str:
    """Render a compact deterministic target-free design summary."""

    lines = [
        "# Held-out study sensitivity preflight",
        "",
        f"- sensitivity ID: `{report.sensitivity_id}`",
        f"- promotion lock: `{report.promotion_lock_id}`",
        f"- source summary: `{report.source_summary_id}`",
        f"- source metric: `{report.source_metric}`",
        f"- target groups: **{report.target_group_count}**",
        f"- confidence: **{report.confidence_level:.3f}** ({report.alternative})",
        "",
        "## Paired effect resolution",
        "",
        "| Scenario | Paired SD [mm] | Power | CI half-width [mm] | MDE [mm] | "
        "Frozen margin detectable |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for record in report.paired_effect_sensitivity:
        detectable = record["query_margin_detectable"]
        display = "—" if detectable is None else ("yes" if detectable else "no")
        lines.append(
            f"| {record['scenario_id']} | "
            f"{cast(float, record['paired_standard_deviation_mm']):.6g} | "
            f"{cast(float, record['power']):.3f} | "
            f"{cast(float, record['confidence_interval_half_width_mm']):.6g} | "
            f"{cast(float, record['minimum_detectable_effect_mm']):.6g} | "
            f"{display} |"
        )

    lines.extend(
        [
            "",
            "## Harmful-update count resolution",
            "",
            "| Accepted groups (scenario) | One event | Allowed count | "
            "Zero-harm upper rate | Allowed-count upper rate |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in report.harmful_update_resolution:
        lines.append(
            f"| {record['accepted_group_count_scenario']} | "
            f"{cast(float, record['one_event_rate_resolution']):.3f} | "
            f"{record['bounded_event_count']} | "
            f"{cast(float, record['zero_harm_one_sided_upper_rate_bound']):.3f} | "
            f"{cast(float, record['allowed_count_one_sided_upper_rate_bound']):.3f} |"
        )

    resolution = report.target_design_resolution
    lines.extend(
        [
            "",
            "## Frozen decision context",
            "",
            f"- query superiority margin: `{report.query_superiority_margin_mm:.6g} mm`",
            f"- harmful-update margin: `{report.harmful_update_margin_mm:.6g} mm`",
            "- maximum harmful accepted updates: "
            f"`{report.maximum_harmful_accepted_updates}`",
            "- maximum worst-group regression: "
            f"`{report.maximum_worst_group_regression_mm:.6g} mm`",
            "- one target group mass: "
            f"`{cast(float, resolution['one_group_mass_resolution']):.6g}`",
            "- smallest one-sided all-favorable sign probability: "
            f"`{cast(float, resolution['smallest_one_sided_sign_probability']):.6g}`",
            "",
            STUDY_SENSITIVITY_CLAIM_BOUNDARY,
            "",
        ]
    )
    return "\n".join(lines)


def write_study_sensitivity(
    report: StudySensitivityV1,
    output: str | Path,
    *,
    markdown: str | Path | None = None,
) -> None:
    """Publish JSON and optional Markdown with no-clobber semantics."""

    output_path = Path(output)
    markdown_path = None if markdown is None else Path(markdown)
    destinations = [output_path] if markdown_path is None else [output_path, markdown_path]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "study sensitivity output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    encoded = json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n"
    atomic_write_text(output_path, encoded, overwrite=False)
    if markdown_path is not None:
        atomic_write_text(
            markdown_path,
            render_study_sensitivity_markdown(report),
            overwrite=False,
        )


def _scenario_argument(value: str) -> PairedDifferenceScenarioV1:
    scenario_id, separator, raw_standard_deviation = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("paired SD must use SCENARIO_ID=STANDARD_DEVIATION_MM")
    try:
        standard_deviation = float(raw_standard_deviation)
        return PairedDifferenceScenarioV1(scenario_id, standard_deviation)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _probability_argument(value: str) -> float:
    try:
        return _strict_probability(float(value), name="probability")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed target-free study-sensitivity command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--source-summary-id", required=True)
    parser.add_argument("--source-metric", required=True)
    parser.add_argument(
        "--paired-sd",
        type=_scenario_argument,
        action="append",
        required=True,
        dest="paired_scenarios",
        metavar="SCENARIO=MM",
        help="source-derived paired-difference SD scenario; may be repeated",
    )
    parser.add_argument(
        "--power",
        type=_probability_argument,
        action="append",
        dest="powers",
        help="requested power level; may be repeated (default: 0.80, 0.90)",
    )
    parser.add_argument(
        "--confidence",
        type=_probability_argument,
        default=0.95,
    )
    parser.add_argument("--alternative", choices=ALTERNATIVES, default="two-sided")
    parser.add_argument(
        "--accepted-groups",
        type=int,
        action="append",
        dest="accepted_group_counts",
        help="accepted-group denominator scenario; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args(argv)

    lock = load_promotion_lock(arguments.lock)
    report = build_study_sensitivity(
        lock,
        source_summary_id=arguments.source_summary_id,
        source_metric=arguments.source_metric,
        paired_difference_scenarios=arguments.paired_scenarios,
        power_levels=DEFAULT_POWER_LEVELS if arguments.powers is None else arguments.powers,
        confidence_level=arguments.confidence,
        alternative=arguments.alternative,
        accepted_group_counts=arguments.accepted_group_counts,
    )
    write_study_sensitivity(report, arguments.output, markdown=arguments.markdown)
    print(
        json.dumps(
            {
                "sensitivity_id": report.sensitivity_id,
                "target_group_count": report.target_group_count,
                "query_margin_detectable": report.query_margin_detectable,
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALTERNATIVES",
    "DEFAULT_POWER_LEVELS",
    "STUDY_SENSITIVITY_CLAIM_BOUNDARY",
    "STUDY_SENSITIVITY_SCHEMA",
    "STUDY_SENSITIVITY_VERSION",
    "PairedDifferenceScenarioV1",
    "StudySensitivityV1",
    "build_study_sensitivity",
    "load_study_sensitivity",
    "main",
    "one_sided_binomial_upper_bound",
    "render_study_sensitivity_markdown",
    "study_sensitivity_from_dict",
    "write_study_sensitivity",
]
