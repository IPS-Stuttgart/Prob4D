"""Finite-sample group-level control for nested selective loss families.

This module implements the conformal-risk-control correction

    R_plus(lambda) = n/(n+1) * R_hat_n(lambda) + B/(n+1),

for a fixed family of group losses in ``[0, B]`` ordered from least to most
conservative. Under exchangeability of the calibration groups and the next
group, and pointwise monotonicity of the loss family, selecting the least
conservative member with ``R_plus <= alpha`` controls the expected next-group
loss at level ``alpha``.

The result is not a high-probability risk bound, conditional coverage,
selective coverage conditional on acceptance, or a deployment-safety claim.
Any model, score, candidate grid, fallback and group definition must be frozen
before the calibration losses are opened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_SCOPE = "exchangeable-group-conformal-risk-control-v1"


def _scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_probability(value: object, name: str) -> float:
    result = _scalar(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _readonly_vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True, slots=True, eq=False)
class GroupRiskControlSelection:
    """Selection and corrected empirical risk for one nested loss family."""

    selected_index: int | None
    selected_parameter: float | None
    empirical_risk: float | None
    corrected_risk: float | None
    target_risk: float
    loss_bound: float
    calibration_group_count: int
    feasible: bool
    empirical_risk_curve: FloatArray
    corrected_risk_curve: FloatArray
    scope: str = _SCOPE

    def __post_init__(self) -> None:
        target = _positive_probability(self.target_risk, "target_risk")
        bound = _scalar(self.loss_bound, "loss_bound", nonnegative=True)
        if bound == 0.0:
            raise ValueError("loss_bound must be positive")
        count = self.calibration_group_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("calibration_group_count must be positive")
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be bool")
        empirical = _readonly_vector(self.empirical_risk_curve, "empirical_risk_curve")
        corrected = _readonly_vector(self.corrected_risk_curve, "corrected_risk_curve")
        if empirical.shape != corrected.shape:
            raise ValueError("risk curves must have equal shape")
        if np.any(empirical < 0.0) or np.any(empirical > bound):
            raise ValueError("empirical risk curve exceeds its declared bound")
        if np.any(corrected < 0.0) or np.any(corrected > bound + 1e-12):
            raise ValueError("corrected risk curve exceeds its declared bound")
        if np.any(np.diff(empirical) > 1e-12) or np.any(np.diff(corrected) > 1e-12):
            raise ValueError("risk curves must be nonincreasing with conservatism")
        if self.scope != _SCOPE:
            raise ValueError("risk-control scope changed")

        present = self.selected_index is not None
        if self.feasible != present:
            raise ValueError("feasible flag and selected index disagree")
        if present:
            index = self.selected_index
            assert index is not None
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("selected_index must be an integer")
            if not 0 <= index < empirical.size:
                raise ValueError("selected_index is outside the candidate family")
            parameter = _scalar(self.selected_parameter, "selected_parameter")
            selected_empirical = _scalar(self.empirical_risk, "empirical_risk")
            selected_corrected = _scalar(self.corrected_risk, "corrected_risk")
            if not math.isclose(selected_empirical, float(empirical[index]), abs_tol=1e-15):
                raise ValueError("selected empirical risk disagrees with its curve")
            if not math.isclose(selected_corrected, float(corrected[index]), abs_tol=1e-15):
                raise ValueError("selected corrected risk disagrees with its curve")
            if selected_corrected > target + 1e-12:
                raise ValueError("selected candidate does not satisfy target risk")
            if index > 0 and corrected[index - 1] <= target:
                raise ValueError("selection is not the least conservative feasible candidate")
            object.__setattr__(self, "selected_parameter", parameter)
        elif any(
            value is not None
            for value in (
                self.selected_parameter,
                self.empirical_risk,
                self.corrected_risk,
            )
        ):
            raise ValueError("infeasible result cannot contain selected values")

        object.__setattr__(self, "target_risk", target)
        object.__setattr__(self, "loss_bound", bound)
        object.__setattr__(self, "empirical_risk_curve", empirical)
        object.__setattr__(self, "corrected_risk_curve", corrected)

    @property
    def finite_sample_floor(self) -> float:
        return self.loss_bound / (self.calibration_group_count + 1)

    def summary(self) -> dict[str, object]:
        return {
            "selected_index": self.selected_index,
            "selected_parameter": self.selected_parameter,
            "empirical_risk": self.empirical_risk,
            "corrected_risk": self.corrected_risk,
            "target_risk": self.target_risk,
            "loss_bound": self.loss_bound,
            "calibration_group_count": self.calibration_group_count,
            "finite_sample_floor": self.finite_sample_floor,
            "feasible": self.feasible,
            "guarantee": (
                "expected-next-exchangeable-group-loss; not high-probability, conditional, "
                "or acceptance-conditional risk"
            ),
        }


def select_group_conformal_risk_control(
    losses: ArrayLike,
    parameters: ArrayLike,
    *,
    target_risk: float,
    loss_bound: float = 1.0,
    monotonicity_tolerance: float = 1e-12,
) -> GroupRiskControlSelection:
    """Select the least conservative candidate passing the CRC correction.

    ``losses[g, j]`` is the bounded loss of calibration group ``g`` under
    candidate ``j``. Columns must be ordered from least to most conservative,
    and every group's loss must be nonincreasing across columns. ``parameters``
    are retained for lineage only; the theorem concerns the nested loss family.
    """

    target = _positive_probability(target_risk, "target_risk")
    bound = _scalar(loss_bound, "loss_bound", nonnegative=True)
    tolerance = _scalar(
        monotonicity_tolerance,
        "monotonicity_tolerance",
        nonnegative=True,
    )
    if bound == 0.0:
        raise ValueError("loss_bound must be positive")
    matrix = np.asarray(losses, dtype=np.float64)
    candidates = np.asarray(parameters, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("losses must be a nonempty group-by-candidate matrix")
    if candidates.ndim != 1 or candidates.size != matrix.shape[1]:
        raise ValueError("parameters must match the candidate dimension")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(candidates)):
        raise ValueError("losses and parameters must be finite")
    if np.any(matrix < 0.0) or np.any(matrix > bound):
        raise ValueError("losses must lie in [0, loss_bound]")
    if np.any(np.diff(matrix, axis=1) > tolerance):
        raise ValueError("each group loss must be nonincreasing with conservatism")

    count = int(matrix.shape[0])
    empirical = np.mean(matrix, axis=0)
    corrected = count / (count + 1.0) * empirical + bound / (count + 1.0)
    feasible_indices = np.flatnonzero(corrected <= target + tolerance)
    if feasible_indices.size == 0:
        return GroupRiskControlSelection(
            selected_index=None,
            selected_parameter=None,
            empirical_risk=None,
            corrected_risk=None,
            target_risk=target,
            loss_bound=bound,
            calibration_group_count=count,
            feasible=False,
            empirical_risk_curve=empirical,
            corrected_risk_curve=corrected,
        )
    index = int(feasible_indices[0])
    return GroupRiskControlSelection(
        selected_index=index,
        selected_parameter=float(candidates[index]),
        empirical_risk=float(empirical[index]),
        corrected_risk=float(corrected[index]),
        target_risk=target,
        loss_bound=bound,
        calibration_group_count=count,
        feasible=True,
        empirical_risk_curve=empirical,
        corrected_risk_curve=corrected,
    )


def harmful_accepted_loss(
    accepted: ArrayLike,
    candidate_loss: ArrayLike,
    fallback_loss: ArrayLike,
) -> FloatArray:
    """Return the casewise binary loss ``accepted AND candidate>wfallback``."""

    admitted = np.asarray(accepted)
    candidate = np.asarray(candidate_loss, dtype=np.float64)
    fallback = np.asarray(fallback_loss, dtype=np.float64)
    if admitted.dtype.kind != "b":
        raise TypeError("accepted must be a boolean array")
    if admitted.shape != candidate.shape or admitted.shape != fallback.shape:
        raise ValueError("accepted and loss arrays must have equal shape")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(fallback)):
        raise ValueError("candidate and fallback losses must be finite")
    result = np.asarray(admitted & (candidate > fallback), dtype=np.float64)
    result.setflags(write=False)
    return result


__all__ = [
    "GroupRiskControlSelection",
    "harmful_accepted_loss",
    "select_group_conformal_risk_control",
]
