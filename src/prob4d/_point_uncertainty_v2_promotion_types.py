"""Threshold policy and per-group metrics for point uncertainty v2 promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ._scientific_scalars import require_finite_real, require_genuine_integer
from ._strict_json import (
    require_exact_fields,
    require_exact_string,
    require_json_number,
    require_mapping,
)

POINT_UNCERTAINTY_PROMOTION_SCHEMA = "prob4d.point-uncertainty-promotion"
POINT_UNCERTAINTY_PROMOTION_VERSION = 1
POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY = (
    "This artifact is disjoint source-validation evidence comparing the experimental "
    "point-uncertainty v2 conditional covariance with the production v1 point covariance. "
    "It does not change provider-v2 export, absorb shared Sim(3) gauge uncertainty, use "
    "protected target outcomes, establish fresh-object transfer, authorize a "
    "BayesianPhysTwin update, establish Causal4D benefit, deployment safety, or state of "
    "the art."
)
_NOMINAL_COVERAGE = 0.90
_CHI2_DF3_90 = 6.251388631170325
_LOG_2PI = math.log(2.0 * math.pi)

_POLICY_FIELDS = frozenset(
    {
        "minimum_group_count",
        "minimum_rows_per_group",
        "minimum_mean_nll_improvement",
        "minimum_group_win_fraction",
        "maximum_coverage_error_increase",
        "maximum_worst_group_coverage_error_increase",
        "maximum_mean_width_ratio",
        "maximum_worst_group_width_ratio",
        "maximum_worst_group_nll_regression",
    }
)

_GROUP_FIELDS = frozenset(
    {
        "group_id",
        "count",
        "baseline_mean_nll",
        "candidate_mean_nll",
        "baseline_coverage90",
        "candidate_coverage90",
        "baseline_mean_rms_width",
        "candidate_mean_rms_width",
        "baseline_normalized_energy",
        "candidate_normalized_energy",
    }
)


def _probability(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class PointUncertaintyPromotionPolicyV1:
    """Predeclared source-validation promotion thresholds."""

    minimum_group_count: int
    minimum_rows_per_group: int
    minimum_mean_nll_improvement: float
    minimum_group_win_fraction: float
    maximum_coverage_error_increase: float
    maximum_worst_group_coverage_error_increase: float
    maximum_mean_width_ratio: float
    maximum_worst_group_width_ratio: float
    maximum_worst_group_nll_regression: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_group_count",
            require_genuine_integer(
                self.minimum_group_count,
                name="minimum_group_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "minimum_rows_per_group",
            require_genuine_integer(
                self.minimum_rows_per_group,
                name="minimum_rows_per_group",
                minimum=1,
            ),
        )
        for name in (
            "minimum_mean_nll_improvement",
            "maximum_coverage_error_increase",
            "maximum_worst_group_coverage_error_increase",
            "maximum_worst_group_nll_regression",
        ):
            object.__setattr__(
                self,
                name,
                require_finite_real(getattr(self, name), name=name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "minimum_group_win_fraction",
            _probability(
                self.minimum_group_win_fraction,
                name="minimum_group_win_fraction",
            ),
        )
        for name in ("maximum_mean_width_ratio", "maximum_worst_group_width_ratio"):
            object.__setattr__(
                self,
                name,
                require_finite_real(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    minimum_inclusive=False,
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_group_count": self.minimum_group_count,
            "minimum_rows_per_group": self.minimum_rows_per_group,
            "minimum_mean_nll_improvement": self.minimum_mean_nll_improvement,
            "minimum_group_win_fraction": self.minimum_group_win_fraction,
            "maximum_coverage_error_increase": self.maximum_coverage_error_increase,
            "maximum_worst_group_coverage_error_increase": (
                self.maximum_worst_group_coverage_error_increase
            ),
            "maximum_mean_width_ratio": self.maximum_mean_width_ratio,
            "maximum_worst_group_width_ratio": self.maximum_worst_group_width_ratio,
            "maximum_worst_group_nll_regression": (
                self.maximum_worst_group_nll_regression
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> PointUncertaintyPromotionPolicyV1:
        mapping = require_mapping(value, name="point uncertainty promotion policy")
        require_exact_fields(
            mapping,
            _POLICY_FIELDS,
            name="point uncertainty promotion policy",
        )
        return cls(
            minimum_group_count=mapping["minimum_group_count"],
            minimum_rows_per_group=mapping["minimum_rows_per_group"],
            minimum_mean_nll_improvement=mapping["minimum_mean_nll_improvement"],
            minimum_group_win_fraction=mapping["minimum_group_win_fraction"],
            maximum_coverage_error_increase=mapping[
                "maximum_coverage_error_increase"
            ],
            maximum_worst_group_coverage_error_increase=mapping[
                "maximum_worst_group_coverage_error_increase"
            ],
            maximum_mean_width_ratio=mapping["maximum_mean_width_ratio"],
            maximum_worst_group_width_ratio=mapping[
                "maximum_worst_group_width_ratio"
            ],
            maximum_worst_group_nll_regression=mapping[
                "maximum_worst_group_nll_regression"
            ],
        )


@dataclass(frozen=True, slots=True)
class PointUncertaintyGroupMetricsV1:
    group_id: str
    count: int
    baseline_mean_nll: float
    candidate_mean_nll: float
    baseline_coverage90: float
    candidate_coverage90: float
    baseline_mean_rms_width: float
    candidate_mean_rms_width: float
    baseline_normalized_energy: float
    candidate_normalized_energy: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            require_exact_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "count",
            require_genuine_integer(self.count, name="count", minimum=1),
        )
        for name in (
            "baseline_mean_nll",
            "candidate_mean_nll",
            "baseline_mean_rms_width",
            "candidate_mean_rms_width",
            "baseline_normalized_energy",
            "candidate_normalized_energy",
        ):
            minimum = 0.0 if "width" in name or "energy" in name else None
            object.__setattr__(
                self,
                name,
                require_finite_real(getattr(self, name), name=name, minimum=minimum),
            )
        object.__setattr__(
            self,
            "baseline_coverage90",
            _probability(self.baseline_coverage90, name="baseline_coverage90"),
        )
        object.__setattr__(
            self,
            "candidate_coverage90",
            _probability(self.candidate_coverage90, name="candidate_coverage90"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "count": self.count,
            "baseline_mean_nll": self.baseline_mean_nll,
            "candidate_mean_nll": self.candidate_mean_nll,
            "baseline_coverage90": self.baseline_coverage90,
            "candidate_coverage90": self.candidate_coverage90,
            "baseline_mean_rms_width": self.baseline_mean_rms_width,
            "candidate_mean_rms_width": self.candidate_mean_rms_width,
            "baseline_normalized_energy": self.baseline_normalized_energy,
            "candidate_normalized_energy": self.candidate_normalized_energy,
        }

    @classmethod
    def from_dict(cls, value: object) -> PointUncertaintyGroupMetricsV1:
        mapping = require_mapping(value, name="point uncertainty group metrics")
        require_exact_fields(mapping, _GROUP_FIELDS, name="point uncertainty group metrics")
        return cls(**cast(dict[str, Any], dict(mapping)))


__all__ = [
    "POINT_UNCERTAINTY_PROMOTION_CLAIM_BOUNDARY",
    "POINT_UNCERTAINTY_PROMOTION_SCHEMA",
    "POINT_UNCERTAINTY_PROMOTION_VERSION",
    "PointUncertaintyGroupMetricsV1",
    "PointUncertaintyPromotionPolicyV1",
]
