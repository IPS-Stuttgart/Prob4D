"""Recording-group conformal lower bounds for candidate advantage.

For cases ``i`` in one exchangeable group, let ``b_i`` be any deterministic
base lower/nominal advantage computed without that group's outcomes and let
``a_i`` be the realized fallback-minus-candidate advantage.  The signed group
score

    S = max_i (b_i - a_i)

can be calibrated by a one-sided split-conformal order statistic.  On the event
``S_new <= tau``, every case in the new group simultaneously satisfies

    a_i >= b_i - tau.

Consequently, admitting a candidate only when ``b_i - tau`` exceeds the
registered margin prevents harmful accepted cases on that event.  Scores are
allowed to be negative: a negative threshold legitimately removes systematic
conservatism from a geometric base bound.

The statement is marginal over the next exchangeable group.  It is not
conditional coverage, target-shift robustness, provider competence, or a
deployment-safety authorization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_SCOPE = "recording-group-conformal-advantage-deficit-v1"


def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _probability(value: object, name: str) -> float:
    result = _scalar(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


@dataclass(frozen=True, slots=True)
class SignedGroupConformalUpperBound:
    """One-sided finite-sample upper bound for signed group scores."""

    threshold: float | None
    miscoverage: float
    calibration_group_count: int
    order_statistic: int
    finite: bool
    scope: str = _SCOPE

    def __post_init__(self) -> None:
        alpha = _probability(self.miscoverage, "miscoverage")
        count = self.calibration_group_count
        order = self.order_statistic
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("calibration_group_count must be a positive integer")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order_statistic must be a positive integer")
        if type(self.finite) is not bool:
            raise TypeError("finite must be a bool")
        if self.scope != _SCOPE:
            raise ValueError("calibration scope changed")
        if self.finite != (self.threshold is not None):
            raise ValueError("finite flag and threshold disagree")
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _scalar(self.threshold, "threshold"))
            if order > count:
                raise ValueError("finite conformal order exceeds sample count")
        elif order <= count:
            raise ValueError("missing threshold despite a finite conformal order")
        object.__setattr__(self, "miscoverage", alpha)

    @property
    def coverage_level(self) -> float:
        return 1.0 - self.miscoverage

    def lower_bound(self, base_advantage: object) -> float:
        """Return the calibrated lower bound for one case."""

        base = _scalar(base_advantage, "base_advantage")
        if self.threshold is None:
            raise ValueError("finite conformal threshold is unavailable")
        return base - self.threshold

    def admits(
        self,
        base_advantage: object,
        *,
        required_margin: object = 0.0,
        numerical_slack: object = 1e-12,
    ) -> bool:
        """Admit iff the calibrated lower bound is strictly positive."""

        margin = _scalar(required_margin, "required_margin", nonnegative=True)
        slack = _scalar(numerical_slack, "numerical_slack", nonnegative=True)
        return self.lower_bound(base_advantage) > margin + slack

    def summary(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "miscoverage": self.miscoverage,
            "coverage_level": self.coverage_level,
            "calibration_group_count": self.calibration_group_count,
            "order_statistic": self.order_statistic,
            "finite": self.finite,
            "signed_scores_allowed": True,
            "guarantee": (
                "marginal-over-next-exchangeable-group; all nested cases "
                "simultaneously satisfy realized_advantage >= "
                "base_advantage - threshold when its group score is covered"
            ),
        }


def calibrate_signed_group_upper_bound(
    scores: ArrayLike,
    *,
    miscoverage: float,
) -> SignedGroupConformalUpperBound:
    """Calibrate ``ceil((n+1)(1-alpha))`` for finite signed scores."""

    alpha = _probability(miscoverage, "miscoverage")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    order = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    if order > values.size:
        return SignedGroupConformalUpperBound(
            threshold=None,
            miscoverage=alpha,
            calibration_group_count=int(values.size),
            order_statistic=order,
            finite=False,
        )
    threshold = float(np.partition(values, order - 1)[order - 1])
    return SignedGroupConformalUpperBound(
        threshold=threshold,
        miscoverage=alpha,
        calibration_group_count=int(values.size),
        order_statistic=order,
        finite=True,
    )


def group_max_advantage_deficit(
    base_advantages: ArrayLike,
    realized_advantages: ArrayLike,
) -> float:
    """Return ``max(base - realized)`` over all cases in one group."""

    base = np.asarray(base_advantages, dtype=np.float64)
    realized = np.asarray(realized_advantages, dtype=np.float64)
    if base.ndim != 1 or base.size == 0:
        raise ValueError("base_advantages must be a nonempty vector")
    if realized.shape != base.shape:
        raise ValueError("realized_advantages must match base_advantages")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(realized)):
        raise ValueError("advantage arrays must be finite")
    return float(np.max(base - realized))


__all__ = [
    "SignedGroupConformalUpperBound",
    "calibrate_signed_group_upper_bound",
    "group_max_advantage_deficit",
]
