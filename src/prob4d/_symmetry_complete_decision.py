"""Robust finite-action decisions over unresolved compact-group orbits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._symmetry_complete_base import (
    _NUMERICAL_ATOL,
    FloatArray,
    SymmetryCompleteBeliefV1,
    _finite_nonnegative_vector,
    _genuine_bool,
    _immutable_float,
)

BoolArray: TypeAlias = NDArray[np.bool_]
DecisionCertificateStatus: TypeAlias = Literal[
    "certified-admissible",
    "certified-no-admissible-action",
    "undetermined",
    "scope-not-certified",
]


def _immutable_bool(value: ArrayLike, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    array = np.ascontiguousarray(raw, dtype=np.bool_)
    result: BoolArray = np.frombuffer(array.tobytes(order="C"), dtype=np.bool_)
    return result


@dataclass(frozen=True, slots=True)
class CompactGroupDecisionCertificateV1:
    """Lower and upper robust-regret bounds for finite actions."""

    status: DecisionCertificateStatus
    bounds_certified: bool
    cover_radius_certified: bool
    lipschitz_bound_certified: bool
    regret_tolerance: float
    sampled_pairwise_worst_case_loss_gap: FloatArray
    upper_pairwise_worst_case_loss_gap: FloatArray
    sampled_worst_case_regret: FloatArray
    upper_worst_case_regret: FloatArray
    tolerance_admissible_action_mask: BoolArray
    minimax_upper_action_index: int
    minimax_upper_worst_case_regret: float
    cover_radius_by_quotient: FloatArray
    action_loss_lipschitz_by_quotient: FloatArray

    def __post_init__(self) -> None:
        allowed: tuple[DecisionCertificateStatus, ...] = (
            "certified-admissible",
            "certified-no-admissible-action",
            "undetermined",
            "scope-not-certified",
        )
        if self.status not in allowed:
            raise ValueError("unsupported decision-certificate status")
        certified = _genuine_bool(self.bounds_certified, name="bounds_certified")
        cover_certified = _genuine_bool(
            self.cover_radius_certified,
            name="cover_radius_certified",
        )
        lipschitz_certified = _genuine_bool(
            self.lipschitz_bound_certified,
            name="lipschitz_bound_certified",
        )
        tolerance = float(self.regret_tolerance)
        minimum = float(self.minimax_upper_worst_case_regret)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("regret_tolerance must be finite and nonnegative")
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimax upper regret must be finite and nonnegative")

        sampled_pairwise = _immutable_float(
            self.sampled_pairwise_worst_case_loss_gap,
            name="sampled_pairwise_worst_case_loss_gap",
            ndim=2,
        )
        upper_pairwise = _immutable_float(
            self.upper_pairwise_worst_case_loss_gap,
            name="upper_pairwise_worst_case_loss_gap",
            ndim=2,
        )
        sampled_regret = _immutable_float(
            self.sampled_worst_case_regret,
            name="sampled_worst_case_regret",
            ndim=1,
        )
        upper_regret = _immutable_float(
            self.upper_worst_case_regret,
            name="upper_worst_case_regret",
            ndim=1,
        )
        admissible = _immutable_bool(
            self.tolerance_admissible_action_mask,
            name="tolerance_admissible_action_mask",
        )
        cover = _immutable_float(
            self.cover_radius_by_quotient,
            name="cover_radius_by_quotient",
            ndim=1,
        )
        lipschitz = _immutable_float(
            self.action_loss_lipschitz_by_quotient,
            name="action_loss_lipschitz_by_quotient",
            ndim=2,
        )
        action_count = sampled_regret.size
        if action_count < 2:
            raise ValueError("at least two actions are required")
        if sampled_pairwise.shape != (action_count, action_count):
            raise ValueError("sampled pairwise matrix has the wrong shape")
        if upper_pairwise.shape != sampled_pairwise.shape:
            raise ValueError("upper pairwise matrix has the wrong shape")
        if upper_regret.shape != sampled_regret.shape or admissible.shape != sampled_regret.shape:
            raise ValueError("action-wise decision arrays have inconsistent shapes")
        if lipschitz.shape != (cover.size, action_count):
            raise ValueError("action Lipschitz matrix has the wrong shape")
        if np.any(cover < 0.0) or np.any(lipschitz < 0.0):
            raise ValueError("cover radii and Lipschitz bounds must be nonnegative")
        if np.any(upper_pairwise + _NUMERICAL_ATOL < sampled_pairwise):
            raise ValueError("upper pairwise gap is below its sampled lower bound")
        if np.any(upper_regret + _NUMERICAL_ATOL < sampled_regret):
            raise ValueError("upper regret is below its sampled lower bound")
        if np.any(sampled_regret < -_NUMERICAL_ATOL):
            raise ValueError("sampled regret must be nonnegative")

        action_index = int(self.minimax_upper_action_index)
        if action_index < 0 or action_index >= action_count:
            raise ValueError("minimax_upper_action_index is out of range")
        if not math.isclose(
            minimum,
            float(np.min(upper_regret)),
            rel_tol=0.0,
            abs_tol=_NUMERICAL_ATOL,
        ) or not math.isclose(
            float(upper_regret[action_index]),
            minimum,
            rel_tol=0.0,
            abs_tol=_NUMERICAL_ATOL,
        ):
            raise ValueError("selected action is not an upper-regret minimizer")
        expected_admissible: BoolArray = certified & (
            upper_regret <= tolerance + _NUMERICAL_ATOL
        )
        if not np.array_equal(admissible, expected_admissible):
            raise ValueError("admissible action mask does not match certified bounds")
        if self.status == "scope-not-certified":
            if certified or np.any(admissible):
                raise ValueError("uncertified scope must reject every action")
        elif not certified:
            raise ValueError("classified decision bounds must be certified")
        if (self.status == "certified-admissible") != bool(np.any(admissible)):
            raise ValueError("certified-admissible status disagrees with action mask")
        lower_rejects_all = bool(
            np.all(sampled_regret > tolerance + _NUMERICAL_ATOL)
        )
        if (self.status == "certified-no-admissible-action") != (
            certified and not np.any(admissible) and lower_rejects_all
        ):
            raise ValueError("no-admissible-action status disagrees with regret bounds")
        if self.status == "undetermined" and (
            not certified or np.any(admissible) or lower_rejects_all
        ):
            raise ValueError("undetermined status disagrees with regret bounds")

        object.__setattr__(self, "bounds_certified", certified)
        object.__setattr__(self, "cover_radius_certified", cover_certified)
        object.__setattr__(self, "lipschitz_bound_certified", lipschitz_certified)
        object.__setattr__(self, "regret_tolerance", tolerance)
        object.__setattr__(self, "sampled_pairwise_worst_case_loss_gap", sampled_pairwise)
        object.__setattr__(self, "upper_pairwise_worst_case_loss_gap", upper_pairwise)
        object.__setattr__(self, "sampled_worst_case_regret", sampled_regret)
        object.__setattr__(self, "upper_worst_case_regret", upper_regret)
        object.__setattr__(self, "tolerance_admissible_action_mask", admissible)
        object.__setattr__(self, "minimax_upper_action_index", action_index)
        object.__setattr__(self, "minimax_upper_worst_case_regret", minimum)
        object.__setattr__(self, "cover_radius_by_quotient", cover)
        object.__setattr__(self, "action_loss_lipschitz_by_quotient", lipschitz)

    @property
    def action_count(self) -> int:
        return int(self.upper_worst_case_regret.size)

    @property
    def has_tolerance_admissible_action(self) -> bool:
        return bool(np.any(self.tolerance_admissible_action_mask))

    @property
    def uniquely_tolerance_identified(self) -> bool:
        return bool(np.count_nonzero(self.tolerance_admissible_action_mask) == 1)

    @property
    def fallback_required(self) -> bool:
        return not self.has_tolerance_admissible_action


def _loss_array(
    value: ArrayLike,
    *,
    quotient_count: int,
    group_count: int,
) -> FloatArray:
    losses = _immutable_float(
        value,
        name="loss_by_quotient_group_action",
        ndim=3,
    )
    if losses.shape[:2] != (quotient_count, group_count) or losses.shape[2] < 2:
        raise ValueError(
            "losses must have shape (quotient_count, group_count, action_count>=2)"
        )
    return losses


def _action_lipschitz(
    value: ArrayLike | float,
    *,
    quotient_count: int,
    action_count: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        raw = np.full((quotient_count, action_count), float(raw), dtype=np.float64)
    elif raw.ndim == 1 and raw.shape == (action_count,):
        raw = np.repeat(raw[None, :], quotient_count, axis=0)
    result = _immutable_float(
        raw,
        name="action_loss_lipschitz_by_quotient",
        ndim=2,
    )
    if result.shape != (quotient_count, action_count) or np.any(result < 0.0):
        raise ValueError(
            "action Lipschitz bounds must be a nonnegative scalar, action vector, "
            "or quotient-by-action matrix"
        )
    return result


def certify_compact_group_decision(
    belief: SymmetryCompleteBeliefV1,
    loss_by_quotient_group_action: ArrayLike,
    *,
    action_loss_lipschitz_by_quotient: ArrayLike | float,
    regret_tolerance: float = 0.0,
    cover_radius_by_quotient: ArrayLike | float | None = None,
    cover_radius_certified: bool | None = None,
    lipschitz_bound_certified: bool = False,
) -> CompactGroupDecisionCertificateV1:
    """Bound robust regret over every completion of the declared group orbit.

    Quotient masses remain fixed while each conditional may concentrate on any
    declared group element. On a certified ``rho``-net, sampled classwise loss-
    gap maxima are lower bounds; adding ``(L_a + L_b) rho`` gives upper bounds.
    Exact finite groups are recovered at ``rho=0``. The certificate does not use
    a selected representative or the numerical conditional group probabilities.
    """

    if not isinstance(belief, SymmetryCompleteBeliefV1):
        raise TypeError("belief must be SymmetryCompleteBeliefV1")
    losses = _loss_array(
        loss_by_quotient_group_action,
        quotient_count=belief.quotient_count,
        group_count=belief.quadrature.node_count,
    )
    action_count = int(losses.shape[2])
    lipschitz = _action_lipschitz(
        action_loss_lipschitz_by_quotient,
        quotient_count=belief.quotient_count,
        action_count=action_count,
    )
    cover_source: ArrayLike | float = (
        belief.quadrature.cover_radius
        if cover_radius_by_quotient is None
        else cover_radius_by_quotient
    )
    cover = _finite_nonnegative_vector(
        cover_source,
        name="cover_radius_by_quotient",
        size=belief.quotient_count,
    )
    if isinstance(regret_tolerance, (bool, np.bool_)):
        raise TypeError("regret_tolerance must be a real scalar")
    tolerance = float(regret_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("regret_tolerance must be finite and nonnegative")
    supplied_lipschitz_certified = _genuine_bool(
        lipschitz_bound_certified,
        name="lipschitz_bound_certified",
    )
    if cover_radius_certified is None:
        supplied_cover_certified = (
            belief.quadrature.cover_radius_certified
            if cover_radius_by_quotient is None
            else False
        )
    else:
        supplied_cover_certified = _genuine_bool(
            cover_radius_certified,
            name="cover_radius_certified",
        )
    needs_lipschitz = bool(np.any(cover[:, None] * lipschitz > 0.0))
    certified = supplied_cover_certified and (
        not needs_lipschitz or supplied_lipschitz_certified
    )

    difference: FloatArray = losses[:, :, :, None] - losses[:, :, None, :]
    sampled_class_max: FloatArray = np.max(difference, axis=1)
    pairwise_lipschitz: FloatArray = lipschitz[:, :, None] + lipschitz[:, None, :]
    diagonal = np.arange(action_count)
    pairwise_lipschitz[:, diagonal, diagonal] = 0.0
    upper_class_max: FloatArray = (
        sampled_class_max + pairwise_lipschitz * cover[:, None, None]
    )
    sampled_pairwise: FloatArray = np.tensordot(
        belief.quotient_weights,
        sampled_class_max,
        axes=(0, 0),
    )
    upper_pairwise: FloatArray = np.tensordot(
        belief.quotient_weights,
        upper_class_max,
        axes=(0, 0),
    )
    np.fill_diagonal(sampled_pairwise, 0.0)
    np.fill_diagonal(upper_pairwise, 0.0)
    sampled_regret: FloatArray = np.maximum(np.max(sampled_pairwise, axis=1), 0.0)
    upper_regret: FloatArray = np.maximum(np.max(upper_pairwise, axis=1), 0.0)
    minimum_upper = float(np.min(upper_regret))
    selected = int(
        np.flatnonzero(
            np.isclose(
                upper_regret,
                minimum_upper,
                rtol=0.0,
                atol=_NUMERICAL_ATOL,
            )
        )[0]
    )
    admissible: BoolArray = np.zeros(action_count, dtype=np.bool_)
    if certified:
        admissible = upper_regret <= tolerance + _NUMERICAL_ATOL
    lower_rejects_all = bool(np.all(sampled_regret > tolerance + _NUMERICAL_ATOL))
    if not certified:
        status: DecisionCertificateStatus = "scope-not-certified"
    elif np.any(admissible):
        status = "certified-admissible"
    elif lower_rejects_all:
        status = "certified-no-admissible-action"
    else:
        status = "undetermined"
    return CompactGroupDecisionCertificateV1(
        status=status,
        bounds_certified=certified,
        cover_radius_certified=supplied_cover_certified,
        lipschitz_bound_certified=supplied_lipschitz_certified,
        regret_tolerance=tolerance,
        sampled_pairwise_worst_case_loss_gap=sampled_pairwise,
        upper_pairwise_worst_case_loss_gap=upper_pairwise,
        sampled_worst_case_regret=sampled_regret,
        upper_worst_case_regret=upper_regret,
        tolerance_admissible_action_mask=admissible,
        minimax_upper_action_index=selected,
        minimax_upper_worst_case_regret=minimum_upper,
        cover_radius_by_quotient=cover,
        action_loss_lipschitz_by_quotient=lipschitz,
    )


__all__ = [
    "CompactGroupDecisionCertificateV1",
    "DecisionCertificateStatus",
    "certify_compact_group_decision",
]
