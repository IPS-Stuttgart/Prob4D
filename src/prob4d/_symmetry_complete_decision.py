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


def _immutable_bool(value: ArrayLike, *, name: str, ndim: int) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values")
    array = np.ascontiguousarray(raw, dtype=np.bool_)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional boolean array")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class CompactGroupDecisionCertificateV1:
    """Finite-action regret bounds over every declared group completion."""

    status: DecisionCertificateStatus
    bounds_certified: bool
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
        tolerance = float(self.regret_tolerance)
        minimum = float(self.minimax_upper_worst_case_regret)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("regret_tolerance must be finite and nonnegative")
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError(
                "minimax_upper_worst_case_regret must be finite and nonnegative"
            )
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
            ndim=1,
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
        if np.any(upper_pairwise + _NUMERICAL_ATOL < sampled_pairwise):
            raise ValueError("upper pairwise gap is below its sampled lower bound")
        if np.any(upper_regret + _NUMERICAL_ATOL < sampled_regret):
            raise ValueError("upper regret is below its sampled lower bound")
        if np.any(sampled_regret < -_NUMERICAL_ATOL) or np.any(
            upper_regret < -_NUMERICAL_ATOL
        ):
            raise ValueError("regret bounds must be nonnegative")
        action_index = int(self.minimax_upper_action_index)
        if action_index < 0 or action_index >= action_count:
            raise ValueError("minimax_upper_action_index is out of range")
        if not math.isclose(
            minimum,
            float(np.min(upper_regret)),
            rel_tol=0.0,
            abs_tol=_NUMERICAL_ATOL,
        ):
            raise ValueError("minimax upper regret does not match action bounds")
        if not math.isclose(
            float(upper_regret[action_index]),
            minimum,
            rel_tol=0.0,
            abs_tol=_NUMERICAL_ATOL,
        ):
            raise ValueError("selected action is not an upper-regret minimizer")
        expected_admissible = certified & (
            upper_regret <= tolerance + _NUMERICAL_ATOL
        )
        if not np.array_equal(admissible, expected_admissible):
            raise ValueError("admissible action mask does not match certified bounds")
        if self.status == "scope-not-certified":
            if certified or np.any(admissible):
                raise ValueError("uncertified scope must reject every action")
        elif not certified:
            raise ValueError("classified decision bounds must be certified")
        if self.status == "certified-admissible" and not np.any(admissible):
            raise ValueError("certified-admissible requires an admissible action")
        if self.status != "certified-admissible" and np.any(admissible):
            raise ValueError("only certified-admissible may admit an action")
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
        if lipschitz.shape != (cover.size, action_count):
            raise ValueError("action Lipschitz matrix has the wrong shape")
        object.__setattr__(self, "bounds_certified", certified)
        object.__setattr__(self, "regret_tolerance", tolerance)
        object.__setattr__(
            self,
            "sampled_pairwise_worst_case_loss_gap",
            sampled_pairwise,
        )
        object.__setattr__(
            self,
            "upper_pairwise_worst_case_loss_gap",
            upper_pairwise,
        )
        object.__setattr__(self, "sampled_worst_case_regret", sampled_regret)
        object.__setattr__(self, "upper_worst_case_regret", upper_regret)
        object.__setattr__(
            self,
            "tolerance_admissible_action_mask",
            admissible,
        )
        object.__setattr__(self, "minimax_upper_action_index", action_index)
        object.__setattr__(self, "minimax_upper_worst_case_regret", minimum)
        object.__setattr__(self, "cover_radius_by_quotient", cover)
        object.__setattr__(
            self,
            "action_loss_lipschitz_by_quotient",
            lipschitz,
        )

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
            "loss_by_quotient_group_action must have shape "
            "(quotient_count, group_count, action_count>=2)"
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
            "action_loss_lipschitz_by_quotient must be a nonnegative scalar, "
            "action vector, or quotient-by-action matrix"
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
) -> CompactGroupDecisionCertificateV1:
    """Bound robust regret over every completion of the declared group orbit.

    Quotient masses remain fixed, while the conditional distribution inside
    each active quotient class may concentrate on any declared group element.
    For actions ``a`` and ``b``, the exact classwise adversary is therefore

        sup_g [loss(g, a) - loss(g, b)].

    On a certified ``rho``-net, sampled maxima are lower bounds and adding
    ``(L_a + L_b) rho`` gives valid upper bounds when each action loss is
    ``L_a``-Lipschitz in the declared group metric. Summing classwise bounds with
    the fixed quotient masses and maximizing over benchmark actions yields lower
    and upper worst-case-regret bounds. Exact finite groups are recovered at
    ``rho=0``.

    The certificate deliberately does not rely on a selected group
    representative or on the numerical conditional group probabilities. It is
    robust over every conditional completion on the declared group domain. An
    uncertified cover rejects every action.
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
    certified = (
        belief.quadrature.cover_radius_certified
        if cover_radius_certified is None
        else _genuine_bool(
            cover_radius_certified,
            name="cover_radius_certified",
        )
    )

    pairwise_difference = losses[:, :, :, None] - losses[:, :, None, :]
    sampled_class_max: FloatArray = np.max(pairwise_difference, axis=1)
    pairwise_lipschitz = lipschitz[:, :, None] + lipschitz[:, None, :]
    diagonal = np.arange(action_count)
    pairwise_lipschitz[:, diagonal, diagonal] = 0.0
    upper_class_max = sampled_class_max + pairwise_lipschitz * cover[:, None, None]
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
    sampled_regret: FloatArray = np.maximum(
        np.max(sampled_pairwise, axis=1),
        0.0,
    )
    upper_regret: FloatArray = np.maximum(
        np.max(upper_pairwise, axis=1),
        0.0,
    )
    minimum_upper = float(np.min(upper_regret))
    minimizers = np.flatnonzero(
        np.isclose(
            upper_regret,
            minimum_upper,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        )
    )
    selected = int(minimizers[0])
    admissible: BoolArray = np.zeros(action_count, dtype=np.bool_)
    if certified:
        admissible = upper_regret <= tolerance + _NUMERICAL_ATOL
    if not certified:
        status: DecisionCertificateStatus = "scope-not-certified"
    elif np.any(admissible):
        status = "certified-admissible"
    elif np.all(sampled_regret > tolerance + _NUMERICAL_ATOL):
        status = "certified-no-admissible-action"
    else:
        status = "undetermined"
    return CompactGroupDecisionCertificateV1(
        status=status,
        bounds_certified=certified,
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
