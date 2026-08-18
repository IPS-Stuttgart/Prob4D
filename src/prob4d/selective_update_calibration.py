"""Source-validation calibration after a guarded Bayesian update decision.

A proposal can be calibrated before a BayesianPhysTwin guard is applied while the
accepted subset is not.  This module records one equal-object/session validation
row per independent group, separates the unselected candidate, the accepted
subset, the exact fallback, and the deployed policy, and evaluates frozen gates
without inspecting target outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_real,
    _strict_string,
)

SELECTIVE_UPDATE_CALIBRATION_SCHEMA = "prob4d.selective-update-calibration"
SELECTIVE_UPDATE_CALIBRATION_VERSION = 1
SELECTIVE_UPDATE_EVIDENCE_PARTITION = "source-validation"
SELECTIVE_UPDATE_SCORE_DIRECTION = "lower-is-better"
SELECTIVE_UPDATE_USES_TARGET_OUTCOMES = False
SELECTIVE_UPDATE_CALIBRATION_CLAIM_BOUNDARY = (
    "This artifact certifies source-validation calibration after one frozen "
    "Bayesian update-selection guard. Complete physical objects or acquisition "
    "sessions are the independent groups. It does not establish target calibration, "
    "unseen-object benefit, Causal4D intervention benefit, deployment safety, or "
    "state of the art."
)

_CRITERION_NAMES = (
    "accepted_coverage",
    "complete_policy_coverage",
    "harmful_accepted_fraction",
    "mean_score_advantage",
    "mean_width_ratio",
    "minimum_accepted_group_count",
    "selection_coverage_drop",
    "worst_group_coverage",
)


def _probability(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _strict_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_real(value: Any, *, name: str) -> float | None:
    return None if value is None else _strict_real(value, name=name)


def _canonical_string_tuple(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    result = tuple(
        _strict_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _string_tuple_from_json(value: Any, *, name: str) -> tuple[str, ...]:
    return _canonical_string_tuple(_strict_list(value, name=name), name=name)


def _same_metric(observed: float, expected: float) -> bool:
    return observed == expected or math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: str | Path, *, name: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable or invalid JSON: {source}") from error
    mapping = _strict_mapping(value, name=name)
    if any(type(key) is not str for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    return mapping


@dataclass(frozen=True, slots=True)
class SelectiveUpdateGroupV1:
    """One equal-mass independent source-validation group."""

    group_id: str
    accepted: bool
    candidate_coverage: float
    candidate_width: float
    candidate_score: float
    fallback_coverage: float
    fallback_width: float
    fallback_score: float
    deployed_coverage: float
    deployed_width: float
    deployed_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _strict_string(self.group_id, name="group_id"))
        object.__setattr__(self, "accepted", _strict_bool(self.accepted, name="accepted"))
        for field_name in ("candidate_coverage", "fallback_coverage", "deployed_coverage"):
            object.__setattr__(
                self,
                field_name,
                _probability(getattr(self, field_name), name=field_name),
            )
        for field_name in ("candidate_width", "fallback_width", "deployed_width"):
            object.__setattr__(
                self,
                field_name,
                _positive_real(getattr(self, field_name), name=field_name),
            )
        for field_name in ("candidate_score", "fallback_score", "deployed_score"):
            object.__setattr__(
                self,
                field_name,
                _strict_real(getattr(self, field_name), name=field_name),
            )
        expected_prefix = "candidate" if self.accepted else "fallback"
        for suffix in ("coverage", "width", "score"):
            deployed_name = f"deployed_{suffix}"
            observed = float(getattr(self, deployed_name))
            expected = float(getattr(self, f"{expected_prefix}_{suffix}"))
            if not _same_metric(observed, expected):
                raise ValueError(
                    "deployed metrics must equal the accepted candidate or exact fallback"
                )
            object.__setattr__(self, deployed_name, expected)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="selective update group metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "accepted": self.accepted,
            "candidate_coverage": self.candidate_coverage,
            "candidate_width": self.candidate_width,
            "candidate_score": self.candidate_score,
            "fallback_coverage": self.fallback_coverage,
            "fallback_width": self.fallback_width,
            "fallback_score": self.fallback_score,
            "deployed_coverage": self.deployed_coverage,
            "deployed_width": self.deployed_width,
            "deployed_score": self.deployed_score,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> SelectiveUpdateGroupV1:
        mapping = _strict_mapping(value, name="selective update group")
        _exact_keys(
            mapping,
            {
                "group_id",
                "accepted",
                "candidate_coverage",
                "candidate_width",
                "candidate_score",
                "fallback_coverage",
                "fallback_width",
                "fallback_score",
                "deployed_coverage",
                "deployed_width",
                "deployed_score",
                "metadata",
            },
            name="selective update group",
        )
        return cls(
            group_id=mapping["group_id"],
            accepted=mapping["accepted"],
            candidate_coverage=mapping["candidate_coverage"],
            candidate_width=mapping["candidate_width"],
            candidate_score=mapping["candidate_score"],
            fallback_coverage=mapping["fallback_coverage"],
            fallback_width=mapping["fallback_width"],
            fallback_score=mapping["fallback_score"],
            deployed_coverage=mapping["deployed_coverage"],
            deployed_width=mapping["deployed_width"],
            deployed_score=mapping["deployed_score"],
            metadata=_strict_mapping(mapping["metadata"], name="group metadata"),
        )


@dataclass(frozen=True, slots=True)
class SelectiveUpdateThresholdsV1:
    """Frozen source-validation gates for the guarded deployment policy."""

    nominal_coverage: float
    maximum_accepted_coverage_shortfall: float
    maximum_complete_policy_coverage_shortfall: float
    maximum_selection_coverage_drop: float
    maximum_mean_width_ratio_vs_fallback: float
    minimum_mean_score_advantage_vs_fallback: float
    harmful_score_margin: float
    maximum_harmful_accepted_fraction: float
    minimum_accepted_group_count: int
    maximum_worst_group_coverage_shortfall: float

    def __post_init__(self) -> None:
        nominal = _probability(self.nominal_coverage, name="nominal_coverage")
        if nominal <= 0.0:
            raise ValueError("nominal_coverage must be positive")
        object.__setattr__(self, "nominal_coverage", nominal)
        for field_name in (
            "maximum_accepted_coverage_shortfall",
            "maximum_complete_policy_coverage_shortfall",
            "maximum_selection_coverage_drop",
            "maximum_harmful_accepted_fraction",
            "maximum_worst_group_coverage_shortfall",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "maximum_mean_width_ratio_vs_fallback",
            _positive_real(
                self.maximum_mean_width_ratio_vs_fallback,
                name="maximum_mean_width_ratio_vs_fallback",
            ),
        )
        object.__setattr__(
            self,
            "minimum_mean_score_advantage_vs_fallback",
            _nonnegative_real(
                self.minimum_mean_score_advantage_vs_fallback,
                name="minimum_mean_score_advantage_vs_fallback",
            ),
        )
        object.__setattr__(
            self,
            "harmful_score_margin",
            _nonnegative_real(self.harmful_score_margin, name="harmful_score_margin"),
        )
        object.__setattr__(
            self,
            "minimum_accepted_group_count",
            _strict_integer(
                self.minimum_accepted_group_count,
                name="minimum_accepted_group_count",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "nominal_coverage": self.nominal_coverage,
            "maximum_accepted_coverage_shortfall": (
                self.maximum_accepted_coverage_shortfall
            ),
            "maximum_complete_policy_coverage_shortfall": (
                self.maximum_complete_policy_coverage_shortfall
            ),
            "maximum_selection_coverage_drop": self.maximum_selection_coverage_drop,
            "maximum_mean_width_ratio_vs_fallback": (
                self.maximum_mean_width_ratio_vs_fallback
            ),
            "minimum_mean_score_advantage_vs_fallback": (
                self.minimum_mean_score_advantage_vs_fallback
            ),
            "harmful_score_margin": self.harmful_score_margin,
            "maximum_harmful_accepted_fraction": (
                self.maximum_harmful_accepted_fraction
            ),
            "minimum_accepted_group_count": self.minimum_accepted_group_count,
            "maximum_worst_group_coverage_shortfall": (
                self.maximum_worst_group_coverage_shortfall
            ),
        }

    @classmethod
    def from_dict(cls, value: Any) -> SelectiveUpdateThresholdsV1:
        mapping = _strict_mapping(value, name="selective update thresholds")
        _exact_keys(
            mapping,
            {
                "nominal_coverage",
                "maximum_accepted_coverage_shortfall",
                "maximum_complete_policy_coverage_shortfall",
                "maximum_selection_coverage_drop",
                "maximum_mean_width_ratio_vs_fallback",
                "minimum_mean_score_advantage_vs_fallback",
                "harmful_score_margin",
                "maximum_harmful_accepted_fraction",
                "minimum_accepted_group_count",
                "maximum_worst_group_coverage_shortfall",
            },
            name="selective update thresholds",
        )
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class SelectiveUpdateCalibrationReportV1:
    """Equal-group proposal, selected-subset, fallback, and deployed diagnostics."""

    group_count: int
    accepted_group_count: int
    rejected_group_count: int
    candidate_mean_coverage: float
    accepted_mean_coverage: float | None
    fallback_mean_coverage: float
    deployed_mean_coverage: float
    selection_coverage_drop: float | None
    candidate_mean_width: float
    accepted_mean_width: float | None
    accepted_fallback_mean_width: float | None
    accepted_to_fallback_width_ratio: float | None
    fallback_mean_width: float
    deployed_mean_width: float
    deployed_to_fallback_width_ratio: float
    candidate_mean_score: float
    accepted_candidate_mean_score: float | None
    accepted_fallback_mean_score: float | None
    accepted_score_advantage_vs_fallback: float | None
    fallback_mean_score: float
    deployed_mean_score: float
    deployed_score_advantage_vs_fallback: float
    harmful_accepted_count: int
    harmful_accepted_fraction: float
    worst_group_deployed_coverage_shortfall: float
    criteria: Mapping[str, Any]
    passed: bool

    def __post_init__(self) -> None:
        group_count = _strict_integer(self.group_count, name="group_count", minimum=1)
        accepted = _strict_integer(
            self.accepted_group_count,
            name="accepted_group_count",
            minimum=0,
        )
        rejected = _strict_integer(
            self.rejected_group_count,
            name="rejected_group_count",
            minimum=0,
        )
        if accepted + rejected != group_count:
            raise ValueError("accepted and rejected counts must sum to group_count")
        harmful = _strict_integer(
            self.harmful_accepted_count,
            name="harmful_accepted_count",
            minimum=0,
        )
        if harmful > accepted:
            raise ValueError("harmful accepted count exceeds accepted count")
        object.__setattr__(self, "group_count", group_count)
        object.__setattr__(self, "accepted_group_count", accepted)
        object.__setattr__(self, "rejected_group_count", rejected)
        object.__setattr__(self, "harmful_accepted_count", harmful)
        for field_name in (
            "candidate_mean_coverage",
            "fallback_mean_coverage",
            "deployed_mean_coverage",
            "harmful_accepted_fraction",
            "worst_group_deployed_coverage_shortfall",
        ):
            object.__setattr__(
                self,
                field_name,
                _probability(getattr(self, field_name), name=field_name),
            )
        accepted_coverage = _optional_real(
            self.accepted_mean_coverage,
            name="accepted_mean_coverage",
        )
        if accepted_coverage is not None:
            accepted_coverage = _probability(
                accepted_coverage,
                name="accepted_mean_coverage",
            )
        object.__setattr__(self, "accepted_mean_coverage", accepted_coverage)
        selection_drop = _optional_real(
            self.selection_coverage_drop,
            name="selection_coverage_drop",
        )
        object.__setattr__(self, "selection_coverage_drop", selection_drop)
        for field_name in (
            "candidate_mean_width",
            "fallback_mean_width",
            "deployed_mean_width",
            "deployed_to_fallback_width_ratio",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_real(getattr(self, field_name), name=field_name),
            )
        for field_name in (
            "accepted_mean_width",
            "accepted_fallback_mean_width",
            "accepted_to_fallback_width_ratio",
        ):
            value = _optional_real(getattr(self, field_name), name=field_name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{field_name} must be positive when present")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "candidate_mean_score",
            "fallback_mean_score",
            "deployed_mean_score",
            "deployed_score_advantage_vs_fallback",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_real(getattr(self, field_name), name=field_name),
            )
        for field_name in (
            "accepted_candidate_mean_score",
            "accepted_fallback_mean_score",
            "accepted_score_advantage_vs_fallback",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_real(getattr(self, field_name), name=field_name),
            )
        optional_fields = (
            self.accepted_mean_coverage,
            self.accepted_mean_width,
            self.accepted_fallback_mean_width,
            self.accepted_to_fallback_width_ratio,
            self.accepted_candidate_mean_score,
            self.accepted_fallback_mean_score,
            self.accepted_score_advantage_vs_fallback,
        )
        if accepted == 0 and any(value is not None for value in optional_fields):
            raise ValueError("accepted-subset diagnostics require accepted groups")
        if accepted > 0 and any(value is None for value in optional_fields):
            raise ValueError("accepted groups require complete accepted-subset diagnostics")
        criteria = _strict_mapping(self.criteria, name="selective update criteria")
        if set(criteria) != set(_CRITERION_NAMES):
            raise ValueError("selective update criteria names changed")
        normalized_criteria = {
            name: _strict_bool(criteria[name], name=f"criteria[{name!r}]")
            for name in _CRITERION_NAMES
        }
        passed = _strict_bool(self.passed, name="passed")
        if passed != all(normalized_criteria.values()):
            raise ValueError("passed must equal the conjunction of all criteria")
        object.__setattr__(
            self,
            "criteria",
            frozen_finite_json_mapping(normalized_criteria, name="criteria"),
        )
        object.__setattr__(self, "passed", passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_count": self.group_count,
            "accepted_group_count": self.accepted_group_count,
            "rejected_group_count": self.rejected_group_count,
            "candidate_mean_coverage": self.candidate_mean_coverage,
            "accepted_mean_coverage": self.accepted_mean_coverage,
            "fallback_mean_coverage": self.fallback_mean_coverage,
            "deployed_mean_coverage": self.deployed_mean_coverage,
            "selection_coverage_drop": self.selection_coverage_drop,
            "candidate_mean_width": self.candidate_mean_width,
            "accepted_mean_width": self.accepted_mean_width,
            "accepted_fallback_mean_width": self.accepted_fallback_mean_width,
            "accepted_to_fallback_width_ratio": (
                self.accepted_to_fallback_width_ratio
            ),
            "fallback_mean_width": self.fallback_mean_width,
            "deployed_mean_width": self.deployed_mean_width,
            "deployed_to_fallback_width_ratio": (
                self.deployed_to_fallback_width_ratio
            ),
            "candidate_mean_score": self.candidate_mean_score,
            "accepted_candidate_mean_score": (
                self.accepted_candidate_mean_score
            ),
            "accepted_fallback_mean_score": (
                self.accepted_fallback_mean_score
            ),
            "accepted_score_advantage_vs_fallback": (
                self.accepted_score_advantage_vs_fallback
            ),
            "fallback_mean_score": self.fallback_mean_score,
            "deployed_mean_score": self.deployed_mean_score,
            "deployed_score_advantage_vs_fallback": (
                self.deployed_score_advantage_vs_fallback
            ),
            "harmful_accepted_count": self.harmful_accepted_count,
            "harmful_accepted_fraction": self.harmful_accepted_fraction,
            "worst_group_deployed_coverage_shortfall": (
                self.worst_group_deployed_coverage_shortfall
            ),
            "criteria": plain_json(self.criteria),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SelectiveUpdateCalibrationReportV1:
        mapping = _strict_mapping(value, name="selective update report")
        _exact_keys(mapping, set(cls.__dataclass_fields__), name="selective update report")
        return cls(**mapping)


def _mean(rows: Sequence[SelectiveUpdateGroupV1], field_name: str) -> float:
    return math.fsum(float(getattr(row, field_name)) for row in rows) / len(rows)


def evaluate_selective_update_calibration(
    rows: Sequence[SelectiveUpdateGroupV1],
    thresholds: SelectiveUpdateThresholdsV1,
) -> SelectiveUpdateCalibrationReportV1:
    """Evaluate equal-group calibration and policy regret under frozen thresholds."""

    if not isinstance(thresholds, SelectiveUpdateThresholdsV1):
        raise ValueError("thresholds must be SelectiveUpdateThresholdsV1")
    ordered = tuple(rows)
    if not ordered or not all(isinstance(row, SelectiveUpdateGroupV1) for row in ordered):
        raise ValueError("rows must be a nonempty sequence of SelectiveUpdateGroupV1")
    group_ids = tuple(row.group_id for row in ordered)
    if group_ids != tuple(sorted(group_ids)) or len(group_ids) != len(set(group_ids)):
        raise ValueError("rows must be sorted by unique group_id")
    accepted_rows = tuple(row for row in ordered if row.accepted)
    accepted_count = len(accepted_rows)
    rejected_count = len(ordered) - accepted_count

    candidate_coverage = _mean(ordered, "candidate_coverage")
    fallback_coverage = _mean(ordered, "fallback_coverage")
    deployed_coverage = _mean(ordered, "deployed_coverage")
    accepted_coverage = (
        _mean(accepted_rows, "candidate_coverage") if accepted_rows else None
    )
    selection_drop = (
        candidate_coverage - accepted_coverage
        if accepted_coverage is not None
        else None
    )

    candidate_width = _mean(ordered, "candidate_width")
    fallback_width = _mean(ordered, "fallback_width")
    deployed_width = _mean(ordered, "deployed_width")
    accepted_width = _mean(accepted_rows, "candidate_width") if accepted_rows else None
    accepted_fallback_width = (
        _mean(accepted_rows, "fallback_width") if accepted_rows else None
    )
    accepted_width_ratio = (
        accepted_width / accepted_fallback_width
        if accepted_width is not None and accepted_fallback_width is not None
        else None
    )
    width_ratio = deployed_width / fallback_width

    candidate_score = _mean(ordered, "candidate_score")
    fallback_score = _mean(ordered, "fallback_score")
    deployed_score = _mean(ordered, "deployed_score")
    accepted_candidate_score = (
        _mean(accepted_rows, "candidate_score") if accepted_rows else None
    )
    accepted_fallback_score = (
        _mean(accepted_rows, "fallback_score") if accepted_rows else None
    )
    accepted_score_advantage = (
        accepted_fallback_score - accepted_candidate_score
        if accepted_candidate_score is not None and accepted_fallback_score is not None
        else None
    )
    score_advantage = fallback_score - deployed_score
    harmful_count = sum(
        row.candidate_score > row.fallback_score + thresholds.harmful_score_margin
        for row in accepted_rows
    )
    harmful_fraction = harmful_count / accepted_count if accepted_count else 0.0
    worst_shortfall = max(
        max(0.0, thresholds.nominal_coverage - row.deployed_coverage)
        for row in ordered
    )
    accepted_shortfall = (
        max(0.0, thresholds.nominal_coverage - accepted_coverage)
        if accepted_coverage is not None
        else None
    )
    deployed_shortfall = max(
        0.0,
        thresholds.nominal_coverage - deployed_coverage,
    )

    criteria = {
        "accepted_coverage": (
            accepted_shortfall is not None
            and accepted_shortfall <= thresholds.maximum_accepted_coverage_shortfall
        ),
        "complete_policy_coverage": (
            deployed_shortfall
            <= thresholds.maximum_complete_policy_coverage_shortfall
        ),
        "harmful_accepted_fraction": (
            harmful_fraction <= thresholds.maximum_harmful_accepted_fraction
        ),
        "mean_score_advantage": (
            accepted_score_advantage is not None
            and accepted_score_advantage
            >= thresholds.minimum_mean_score_advantage_vs_fallback
        ),
        "mean_width_ratio": (
            accepted_width_ratio is not None
            and accepted_width_ratio
            <= thresholds.maximum_mean_width_ratio_vs_fallback
        ),
        "minimum_accepted_group_count": (
            accepted_count >= thresholds.minimum_accepted_group_count
        ),
        "selection_coverage_drop": (
            selection_drop is not None
            and selection_drop <= thresholds.maximum_selection_coverage_drop
        ),
        "worst_group_coverage": (
            worst_shortfall
            <= thresholds.maximum_worst_group_coverage_shortfall
        ),
    }
    return SelectiveUpdateCalibrationReportV1(
        group_count=len(ordered),
        accepted_group_count=accepted_count,
        rejected_group_count=rejected_count,
        candidate_mean_coverage=candidate_coverage,
        accepted_mean_coverage=accepted_coverage,
        fallback_mean_coverage=fallback_coverage,
        deployed_mean_coverage=deployed_coverage,
        selection_coverage_drop=selection_drop,
        candidate_mean_width=candidate_width,
        accepted_mean_width=accepted_width,
        accepted_fallback_mean_width=accepted_fallback_width,
        accepted_to_fallback_width_ratio=accepted_width_ratio,
        fallback_mean_width=fallback_width,
        deployed_mean_width=deployed_width,
        deployed_to_fallback_width_ratio=width_ratio,
        candidate_mean_score=candidate_score,
        accepted_candidate_mean_score=accepted_candidate_score,
        accepted_fallback_mean_score=accepted_fallback_score,
        accepted_score_advantage_vs_fallback=accepted_score_advantage,
        fallback_mean_score=fallback_score,
        deployed_mean_score=deployed_score,
        deployed_score_advantage_vs_fallback=score_advantage,
        harmful_accepted_count=harmful_count,
        harmful_accepted_fraction=harmful_fraction,
        worst_group_deployed_coverage_shortfall=worst_shortfall,
        criteria=criteria,
        passed=all(criteria.values()),
    )


@dataclass(frozen=True, slots=True)
class SelectiveUpdateCalibrationV1:
    """Content-addressed source-validation certificate for a frozen guard."""

    protocol_id: str
    query_definition: str
    score_definition: str
    width_unit: str
    group_definition: str
    selection_lock_id: str
    guard_artifact_id: str
    candidate_artifact_id: str
    fallback_artifact_id: str
    guard_fit_group_ids: tuple[str, ...]
    guard_calibration_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    rows: tuple[SelectiveUpdateGroupV1, ...]
    thresholds: SelectiveUpdateThresholdsV1
    report: SelectiveUpdateCalibrationReportV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_id",
            "query_definition",
            "score_definition",
            "width_unit",
            "group_definition",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_string(getattr(self, field_name), name=field_name),
            )
        for field_name in (
            "selection_lock_id",
            "guard_artifact_id",
            "candidate_artifact_id",
            "fallback_artifact_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        fit_ids = _canonical_string_tuple(
            self.guard_fit_group_ids,
            name="guard_fit_group_ids",
        )
        calibration_ids = _canonical_string_tuple(
            self.guard_calibration_group_ids,
            name="guard_calibration_group_ids",
        )
        validation_ids = _canonical_string_tuple(
            self.validation_group_ids,
            name="validation_group_ids",
        )
        if set(fit_ids) & set(calibration_ids):
            raise ValueError("guard-fit and guard-calibration groups must be disjoint")
        if set(fit_ids) & set(validation_ids):
            raise ValueError("guard-fit and validation groups must be disjoint")
        if set(calibration_ids) & set(validation_ids):
            raise ValueError("guard-calibration and validation groups must be disjoint")
        if type(self.rows) is not tuple or not self.rows or not all(
            isinstance(row, SelectiveUpdateGroupV1) for row in self.rows
        ):
            raise ValueError("rows must be a nonempty tuple of SelectiveUpdateGroupV1")
        rows = tuple(self.rows)
        row_ids = tuple(row.group_id for row in rows)
        if row_ids != validation_ids:
            raise ValueError("rows must match the ordered validation_group_ids exactly")
        if not isinstance(self.thresholds, SelectiveUpdateThresholdsV1):
            raise ValueError("thresholds must be SelectiveUpdateThresholdsV1")
        if not isinstance(self.report, SelectiveUpdateCalibrationReportV1):
            raise ValueError("report must be SelectiveUpdateCalibrationReportV1")
        replayed = evaluate_selective_update_calibration(rows, self.thresholds)
        if self.report.to_dict() != replayed.to_dict():
            raise ValueError("selective update report does not match deterministic replay")
        object.__setattr__(self, "guard_fit_group_ids", fit_ids)
        object.__setattr__(self, "guard_calibration_group_ids", calibration_ids)
        object.__setattr__(self, "validation_group_ids", validation_ids)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="selective update calibration metadata",
            ),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": SELECTIVE_UPDATE_CALIBRATION_SCHEMA,
            "schema_version": SELECTIVE_UPDATE_CALIBRATION_VERSION,
            "protocol_id": self.protocol_id,
            "query_definition": self.query_definition,
            "score_definition": self.score_definition,
            "score_direction": SELECTIVE_UPDATE_SCORE_DIRECTION,
            "width_unit": self.width_unit,
            "group_definition": self.group_definition,
            "evidence_partition": SELECTIVE_UPDATE_EVIDENCE_PARTITION,
            "uses_target_outcomes": SELECTIVE_UPDATE_USES_TARGET_OUTCOMES,
            "selection_lock_id": self.selection_lock_id,
            "guard_artifact_id": self.guard_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "fallback_artifact_id": self.fallback_artifact_id,
            "guard_fit_group_ids": list(self.guard_fit_group_ids),
            "guard_calibration_group_ids": list(self.guard_calibration_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "rows": [row.to_dict() for row in self.rows],
            "thresholds": self.thresholds.to_dict(),
            "report": self.report.to_dict(),
            "metadata": plain_json(self.metadata),
            "claim_boundary": SELECTIVE_UPDATE_CALIBRATION_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_dict(cls, value: Any) -> SelectiveUpdateCalibrationV1:
        mapping = _strict_mapping(value, name="selective update calibration")
        _exact_keys(
            mapping,
            {
                "schema_name",
                "schema_version",
                "protocol_id",
                "query_definition",
                "score_definition",
                "score_direction",
                "width_unit",
                "group_definition",
                "evidence_partition",
                "uses_target_outcomes",
                "selection_lock_id",
                "guard_artifact_id",
                "candidate_artifact_id",
                "fallback_artifact_id",
                "guard_fit_group_ids",
                "guard_calibration_group_ids",
                "validation_group_ids",
                "rows",
                "thresholds",
                "report",
                "metadata",
                "claim_boundary",
                "artifact_id",
            },
            name="selective update calibration",
        )
        if mapping["schema_name"] != SELECTIVE_UPDATE_CALIBRATION_SCHEMA:
            raise ValueError("unsupported selective update calibration schema")
        if mapping["schema_version"] != SELECTIVE_UPDATE_CALIBRATION_VERSION:
            raise ValueError("unsupported selective update calibration version")
        if mapping["claim_boundary"] != SELECTIVE_UPDATE_CALIBRATION_CLAIM_BOUNDARY:
            raise ValueError("selective update calibration claim boundary changed")
        if mapping["score_direction"] != SELECTIVE_UPDATE_SCORE_DIRECTION:
            raise ValueError("selective update score direction changed")
        if mapping["evidence_partition"] != SELECTIVE_UPDATE_EVIDENCE_PARTITION:
            raise ValueError("selective update evidence partition changed")
        if _strict_bool(
            mapping["uses_target_outcomes"],
            name="uses_target_outcomes",
        ) is not SELECTIVE_UPDATE_USES_TARGET_OUTCOMES:
            raise ValueError("selective update artifact must not use target outcomes")
        rows = tuple(
            SelectiveUpdateGroupV1.from_dict(row)
            for row in _strict_list(mapping["rows"], name="rows")
        )
        artifact = cls(
            protocol_id=mapping["protocol_id"],
            query_definition=mapping["query_definition"],
            score_definition=mapping["score_definition"],
            width_unit=mapping["width_unit"],
            group_definition=mapping["group_definition"],
            selection_lock_id=mapping["selection_lock_id"],
            guard_artifact_id=mapping["guard_artifact_id"],
            candidate_artifact_id=mapping["candidate_artifact_id"],
            fallback_artifact_id=mapping["fallback_artifact_id"],
            guard_fit_group_ids=_string_tuple_from_json(
                mapping["guard_fit_group_ids"],
                name="guard_fit_group_ids",
            ),
            guard_calibration_group_ids=_string_tuple_from_json(
                mapping["guard_calibration_group_ids"],
                name="guard_calibration_group_ids",
            ),
            validation_group_ids=_string_tuple_from_json(
                mapping["validation_group_ids"],
                name="validation_group_ids",
            ),
            rows=rows,
            thresholds=SelectiveUpdateThresholdsV1.from_dict(mapping["thresholds"]),
            report=SelectiveUpdateCalibrationReportV1.from_dict(mapping["report"]),
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )
        supplied = _strict_digest(
            mapping["artifact_id"],
            name="artifact_id",
            pattern=_SHA256,
        )
        if supplied != artifact.artifact_id:
            raise ValueError("selective update calibration artifact_id mismatch")
        return artifact


def build_selective_update_calibration(
    *,
    protocol_id: str,
    query_definition: str,
    score_definition: str,
    width_unit: str,
    group_definition: str,
    selection_lock_id: str,
    guard_artifact_id: str,
    candidate_artifact_id: str,
    fallback_artifact_id: str,
    guard_fit_group_ids: Sequence[str],
    guard_calibration_group_ids: Sequence[str],
    rows: Sequence[SelectiveUpdateGroupV1],
    thresholds: SelectiveUpdateThresholdsV1,
    metadata: Mapping[str, Any] | None = None,
) -> SelectiveUpdateCalibrationV1:
    """Build a canonical certificate from one source-validation group table."""

    supplied_rows = tuple(rows)
    if not supplied_rows or not all(
        isinstance(row, SelectiveUpdateGroupV1) for row in supplied_rows
    ):
        raise ValueError("rows must be a nonempty sequence of SelectiveUpdateGroupV1")
    ordered_rows = tuple(sorted(supplied_rows, key=lambda row: row.group_id))
    validation_group_ids = tuple(row.group_id for row in ordered_rows)
    report = evaluate_selective_update_calibration(ordered_rows, thresholds)
    return SelectiveUpdateCalibrationV1(
        protocol_id=protocol_id,
        query_definition=query_definition,
        score_definition=score_definition,
        width_unit=width_unit,
        group_definition=group_definition,
        selection_lock_id=selection_lock_id,
        guard_artifact_id=guard_artifact_id,
        candidate_artifact_id=candidate_artifact_id,
        fallback_artifact_id=fallback_artifact_id,
        guard_fit_group_ids=tuple(sorted(guard_fit_group_ids)),
        guard_calibration_group_ids=tuple(sorted(guard_calibration_group_ids)),
        validation_group_ids=validation_group_ids,
        rows=ordered_rows,
        thresholds=thresholds,
        report=report,
        metadata={} if metadata is None else metadata,
    )


def selective_update_calibration_from_raw(value: Any) -> SelectiveUpdateCalibrationV1:
    mapping = _strict_mapping(value, name="raw selective update calibration")
    _exact_keys(
        mapping,
        {
            "protocol_id",
            "query_definition",
            "score_definition",
            "width_unit",
            "group_definition",
            "selection_lock_id",
            "guard_artifact_id",
            "candidate_artifact_id",
            "fallback_artifact_id",
            "guard_fit_group_ids",
            "guard_calibration_group_ids",
            "rows",
            "thresholds",
            "metadata",
        },
        name="raw selective update calibration",
    )
    return build_selective_update_calibration(
        protocol_id=mapping["protocol_id"],
        query_definition=mapping["query_definition"],
        score_definition=mapping["score_definition"],
        width_unit=mapping["width_unit"],
        group_definition=mapping["group_definition"],
        selection_lock_id=mapping["selection_lock_id"],
        guard_artifact_id=mapping["guard_artifact_id"],
        candidate_artifact_id=mapping["candidate_artifact_id"],
        fallback_artifact_id=mapping["fallback_artifact_id"],
        guard_fit_group_ids=_strict_list(
            mapping["guard_fit_group_ids"],
            name="guard_fit_group_ids",
        ),
        guard_calibration_group_ids=_strict_list(
            mapping["guard_calibration_group_ids"],
            name="guard_calibration_group_ids",
        ),
        rows=tuple(
            SelectiveUpdateGroupV1.from_dict(row)
            for row in _strict_list(mapping["rows"], name="rows")
        ),
        thresholds=SelectiveUpdateThresholdsV1.from_dict(mapping["thresholds"]),
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )


def write_selective_update_calibration(
    artifact: SelectiveUpdateCalibrationV1,
    path: str | Path,
) -> None:
    if not isinstance(artifact, SelectiveUpdateCalibrationV1):
        raise ValueError("artifact must be SelectiveUpdateCalibrationV1")
    destination = Path(path)
    payload = (
        json.dumps(
            artifact.to_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        atomic_write_text(destination, payload, overwrite=False)
    except FileExistsError:
        existing = load_selective_update_calibration(destination)
        if existing.to_dict() == artifact.to_dict():
            return
        raise FileExistsError(
            f"refusing to replace a different selective update calibration: {destination}"
        ) from None


def load_selective_update_calibration(
    path: str | Path,
) -> SelectiveUpdateCalibrationV1:
    return SelectiveUpdateCalibrationV1.from_dict(
        _load_json(path, name="selective update calibration")
    )


def _build_cli(arguments: argparse.Namespace) -> int:
    artifact = selective_update_calibration_from_raw(
        _load_json(arguments.input, name="raw selective update calibration")
    )
    write_selective_update_calibration(artifact, arguments.output)
    print(artifact.artifact_id)
    if arguments.require_pass and not artifact.report.passed:
        return 3
    return 0


def _verify_cli(arguments: argparse.Namespace) -> int:
    artifact = load_selective_update_calibration(arguments.artifact)
    print(artifact.artifact_id)
    if arguments.require_pass and not artifact.report.passed:
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify a source-validation selective-calibration certificate."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a content-addressed certificate")
    build.add_argument("input", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--require-pass", action="store_true")
    build.set_defaults(handler=_build_cli)
    verify = subparsers.add_parser("verify", help="verify and replay a certificate")
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--require-pass", action="store_true")
    verify.set_defaults(handler=_verify_cli)
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


__all__ = [
    "SELECTIVE_UPDATE_CALIBRATION_CLAIM_BOUNDARY",
    "SELECTIVE_UPDATE_CALIBRATION_SCHEMA",
    "SELECTIVE_UPDATE_CALIBRATION_VERSION",
    "SELECTIVE_UPDATE_EVIDENCE_PARTITION",
    "SELECTIVE_UPDATE_SCORE_DIRECTION",
    "SELECTIVE_UPDATE_USES_TARGET_OUTCOMES",
    "SelectiveUpdateCalibrationReportV1",
    "SelectiveUpdateCalibrationV1",
    "SelectiveUpdateGroupV1",
    "SelectiveUpdateThresholdsV1",
    "build_selective_update_calibration",
    "evaluate_selective_update_calibration",
    "load_selective_update_calibration",
    "main",
    "selective_update_calibration_from_raw",
    "write_selective_update_calibration",
]


if __name__ == "__main__":
    raise SystemExit(main())
