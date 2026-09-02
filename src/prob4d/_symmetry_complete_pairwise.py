"""Pairwise-Lipschitz regret certificates for gauge-coupled action orbits.

The generic compact-group decision certificate bounds each absolute action loss
and then adds two Lipschitz margins to an action comparison.  This module works
directly with pairwise loss differences.  Action-independent gauge variation
therefore cancels before discretization error is bounded, which can make the
certificate substantially tighter under exact or approximate equivariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._symmetry_complete_action import GaugeCouplingReceiptV1
from ._symmetry_complete_base import (
    _NUMERICAL_ATOL,
    FloatArray,
    SymmetryCompleteBeliefV1,
    _finite_nonnegative_vector,
    _genuine_bool,
    _immutable_float,
)

BoolArray: TypeAlias = NDArray[np.bool_]
PairwiseDecisionStatus: TypeAlias = Literal[
    "certified-admissible",
    "certified-no-admissible-action",
    "undetermined",
    "scope-not-certified",
]

PAIRWISE_GAUGE_DECISION_CLAIM_BOUNDARY = (
    "The certificate is valid only for the supplied compact-group cover, coupled "
    "state/action loss table, certified pairwise loss-difference Lipschitz bounds, "
    "quotient masses, and execution-coupling receipt. It does not infer a symmetry, "
    "prove the physical loss or receipt, establish counterfactual outcomes for "
    "unexecuted actions, authorize deployment, or certify safety."
)


def _immutable_bool_vector(value: ArrayLike, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    contiguous = np.ascontiguousarray(raw, dtype=np.bool_)
    result: BoolArray = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)
    return result


def _pairwise_lipschitz(
    value: ArrayLike | float,
    *,
    quotient_count: int,
    action_count: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        expanded = np.full(
            (quotient_count, action_count, action_count),
            float(raw),
            dtype=np.float64,
        )
    elif raw.ndim == 2 and raw.shape == (action_count, action_count):
        expanded = np.repeat(raw[None, :, :], quotient_count, axis=0)
    else:
        expanded = raw
    result = np.array(
        _immutable_float(
            expanded,
            name="pairwise_difference_lipschitz_by_quotient_action",
            ndim=3,
        ),
        copy=True,
    )
    if result.shape != (quotient_count, action_count, action_count):
        raise ValueError(
            "pairwise Lipschitz bounds must be a nonnegative scalar, action-by-action "
            "matrix, or quotient-by-action-by-action tensor"
        )
    if np.any(result < 0.0):
        raise ValueError("pairwise Lipschitz bounds must be nonnegative")
    diagonal = np.arange(action_count)
    result[:, diagonal, diagonal] = 0.0
    return _immutable_float(
        result,
        name="pairwise_difference_lipschitz_by_quotient_action",
        ndim=3,
    )


@dataclass(frozen=True, slots=True)
class GaugeCoupledPairwiseDecisionCertificateV1:
    """Lower and upper regret bounds from pairwise difference regularity."""

    status: PairwiseDecisionStatus
    receipt: GaugeCouplingReceiptV1
    bounds_certified: bool
    cover_radius_certified: bool
    pairwise_lipschitz_bound_certified: bool
    regret_tolerance: float
    sampled_difference_range_by_quotient_action: FloatArray
    sampled_pairwise_worst_case_loss_gap: FloatArray
    upper_pairwise_worst_case_loss_gap: FloatArray
    sampled_worst_case_regret: FloatArray
    upper_worst_case_regret: FloatArray
    tolerance_admissible_action_mask: BoolArray
    minimax_upper_action_index: int
    minimax_upper_worst_case_regret: float
    cover_radius_by_quotient: FloatArray
    pairwise_difference_lipschitz_by_quotient_action: FloatArray
    pairwise_cover_correction_by_quotient_action: FloatArray

    def __post_init__(self) -> None:
        allowed: tuple[PairwiseDecisionStatus, ...] = (
            "certified-admissible",
            "certified-no-admissible-action",
            "undetermined",
            "scope-not-certified",
        )
        if self.status not in allowed:
            raise ValueError("unsupported pairwise decision status")
        if not isinstance(self.receipt, GaugeCouplingReceiptV1):
            raise TypeError("receipt must be GaugeCouplingReceiptV1")
        certified = _genuine_bool(self.bounds_certified, name="bounds_certified")
        cover_certified = _genuine_bool(
            self.cover_radius_certified,
            name="cover_radius_certified",
        )
        lipschitz_certified = _genuine_bool(
            self.pairwise_lipschitz_bound_certified,
            name="pairwise_lipschitz_bound_certified",
        )
        tolerance = float(self.regret_tolerance)
        minimum = float(self.minimax_upper_worst_case_regret)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("regret_tolerance must be finite and nonnegative")
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimax upper regret must be finite and nonnegative")

        difference_range = _immutable_float(
            self.sampled_difference_range_by_quotient_action,
            name="sampled_difference_range_by_quotient_action",
            ndim=3,
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
        admissible = _immutable_bool_vector(
            self.tolerance_admissible_action_mask,
            name="tolerance_admissible_action_mask",
        )
        cover = _immutable_float(
            self.cover_radius_by_quotient,
            name="cover_radius_by_quotient",
            ndim=1,
        )
        lipschitz = _immutable_float(
            self.pairwise_difference_lipschitz_by_quotient_action,
            name="pairwise_difference_lipschitz_by_quotient_action",
            ndim=3,
        )
        correction = _immutable_float(
            self.pairwise_cover_correction_by_quotient_action,
            name="pairwise_cover_correction_by_quotient_action",
            ndim=3,
        )
        action_count = sampled_regret.size
        quotient_count = cover.size
        pairwise_shape = (action_count, action_count)
        quotient_pairwise_shape = (quotient_count, action_count, action_count)
        if action_count < 2 or quotient_count < 1:
            raise ValueError("at least one quotient and two actions are required")
        if sampled_pairwise.shape != pairwise_shape or upper_pairwise.shape != pairwise_shape:
            raise ValueError("pairwise expected-gap matrices have the wrong shape")
        if sampled_regret.shape != (action_count,) or upper_regret.shape != (action_count,):
            raise ValueError("regret vectors have the wrong shape")
        if admissible.shape != (action_count,):
            raise ValueError("admissible action mask has the wrong shape")
        if difference_range.shape != quotient_pairwise_shape:
            raise ValueError("sampled pairwise ranges have the wrong shape")
        if (
            lipschitz.shape != quotient_pairwise_shape
            or correction.shape != quotient_pairwise_shape
        ):
            raise ValueError("pairwise regularity tensors have the wrong shape")
        if np.any(difference_range < 0.0) or np.any(cover < 0.0):
            raise ValueError("sample ranges and cover radii must be nonnegative")
        if np.any(lipschitz < 0.0) or np.any(correction < 0.0):
            raise ValueError("pairwise regularity bounds must be nonnegative")
        expected_correction = lipschitz * cover[:, None, None]
        if not np.allclose(
            correction,
            expected_correction,
            rtol=0.0,
            atol=_NUMERICAL_ATOL,
        ):
            raise ValueError("pairwise cover correction is inconsistent")
        if np.any(upper_pairwise + _NUMERICAL_ATOL < sampled_pairwise):
            raise ValueError("upper pairwise gap is below its sampled lower bound")
        if np.any(upper_regret + _NUMERICAL_ATOL < sampled_regret):
            raise ValueError("upper regret is below its sampled lower bound")
        if np.any(sampled_regret < -_NUMERICAL_ATOL):
            raise ValueError("sampled regret must be nonnegative")

        action_index = int(self.minimax_upper_action_index)
        if not 0 <= action_index < action_count:
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

        expected_admissible = certified & (upper_regret <= tolerance + _NUMERICAL_ATOL)
        if not np.array_equal(admissible, expected_admissible):
            raise ValueError("admissible mask does not match certified upper regret")
        lower_rejects_all = bool(np.all(sampled_regret > tolerance + _NUMERICAL_ATOL))
        if self.status == "scope-not-certified":
            if certified or np.any(admissible):
                raise ValueError("uncertified scope must reject every action")
        elif not certified:
            raise ValueError("classified pairwise bounds must be certified")
        if (self.status == "certified-admissible") != bool(np.any(admissible)):
            raise ValueError("certified-admissible status disagrees with action mask")
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
        object.__setattr__(
            self,
            "pairwise_lipschitz_bound_certified",
            lipschitz_certified,
        )
        object.__setattr__(self, "regret_tolerance", tolerance)
        object.__setattr__(
            self,
            "sampled_difference_range_by_quotient_action",
            difference_range,
        )
        object.__setattr__(
            self,
            "sampled_pairwise_worst_case_loss_gap",
            sampled_pairwise,
        )
        object.__setattr__(self, "upper_pairwise_worst_case_loss_gap", upper_pairwise)
        object.__setattr__(self, "sampled_worst_case_regret", sampled_regret)
        object.__setattr__(self, "upper_worst_case_regret", upper_regret)
        object.__setattr__(self, "tolerance_admissible_action_mask", admissible)
        object.__setattr__(self, "minimax_upper_action_index", action_index)
        object.__setattr__(self, "minimax_upper_worst_case_regret", minimum)
        object.__setattr__(self, "cover_radius_by_quotient", cover)
        object.__setattr__(
            self,
            "pairwise_difference_lipschitz_by_quotient_action",
            lipschitz,
        )
        object.__setattr__(
            self,
            "pairwise_cover_correction_by_quotient_action",
            correction,
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

    @property
    def maximum_pairwise_cover_correction(self) -> float:
        return float(np.max(self.pairwise_cover_correction_by_quotient_action))


def certify_gauge_coupled_pairwise_decision(
    belief: SymmetryCompleteBeliefV1,
    coupled_loss_by_quotient_group_action: ArrayLike,
    *,
    coupling_receipt: GaugeCouplingReceiptV1,
    pairwise_difference_lipschitz_by_quotient_action: ArrayLike | float,
    regret_tolerance: float = 0.0,
    cover_radius_by_quotient: ArrayLike | float | None = None,
    cover_radius_certified: bool | None = None,
    pairwise_lipschitz_bound_certified: bool = False,
) -> GaugeCoupledPairwiseDecisionCertificateV1:
    """Bound complete-orbit regret through pairwise loss-difference regularity.

    For class ``c`` and actions ``a,b``, let
    ``d_cab(g) = loss_c,a(g) - loss_c,b(g)``.  On a certified ``rho_c``-net, a
    certified Lipschitz constant ``L_cab`` gives

    ``sup_g d_cab(g) <= max_sample d_cab + L_cab * rho_c``.

    The sampled maximum is a lower bound on the same supremum.  Quotient masses
    then combine the classwise bounds exactly.  Unlike action-wise regularity,
    this construction removes every action-independent gauge term before the
    cover correction is applied.
    """

    if not isinstance(belief, SymmetryCompleteBeliefV1):
        raise TypeError("belief must be SymmetryCompleteBeliefV1")
    if not isinstance(coupling_receipt, GaugeCouplingReceiptV1):
        raise TypeError("coupling_receipt must be GaugeCouplingReceiptV1")
    if coupling_receipt.group_id != belief.quadrature.group_id:
        raise ValueError("coupling receipt and belief use different group identifiers")
    losses = _immutable_float(
        coupled_loss_by_quotient_group_action,
        name="coupled_loss_by_quotient_group_action",
        ndim=3,
    )
    expected_prefix = (
        belief.quotient_count,
        belief.quadrature.node_count,
    )
    if losses.shape[:2] != expected_prefix or losses.shape[2] < 2:
        raise ValueError(
            "coupled losses must have shape (quotient_count, group_count, action_count>=2)"
        )
    action_count = int(losses.shape[2])
    lipschitz = _pairwise_lipschitz(
        pairwise_difference_lipschitz_by_quotient_action,
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
        pairwise_lipschitz_bound_certified,
        name="pairwise_lipschitz_bound_certified",
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
    active = belief.quotient_weights > 0.0
    needs_lipschitz = bool(np.any(cover[active] > _NUMERICAL_ATOL))
    bounds_certified = (
        coupling_receipt.scope_certified
        and supplied_cover_certified
        and (not needs_lipschitz or supplied_lipschitz_certified)
    )

    difference: FloatArray = losses[:, :, :, None] - losses[:, :, None, :]
    sampled_class_max: FloatArray = np.max(difference, axis=1)
    sampled_class_min: FloatArray = np.min(difference, axis=1)
    sampled_range: FloatArray = sampled_class_max - sampled_class_min
    correction: FloatArray = lipschitz * cover[:, None, None]
    upper_class_max: FloatArray = sampled_class_max + correction
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
    if bounds_certified:
        admissible = upper_regret <= tolerance + _NUMERICAL_ATOL
    lower_rejects_all = bool(np.all(sampled_regret > tolerance + _NUMERICAL_ATOL))
    if not bounds_certified:
        status: PairwiseDecisionStatus = "scope-not-certified"
    elif np.any(admissible):
        status = "certified-admissible"
    elif lower_rejects_all:
        status = "certified-no-admissible-action"
    else:
        status = "undetermined"

    return GaugeCoupledPairwiseDecisionCertificateV1(
        status=status,
        receipt=coupling_receipt,
        bounds_certified=bounds_certified,
        cover_radius_certified=supplied_cover_certified,
        pairwise_lipschitz_bound_certified=supplied_lipschitz_certified,
        regret_tolerance=tolerance,
        sampled_difference_range_by_quotient_action=sampled_range,
        sampled_pairwise_worst_case_loss_gap=sampled_pairwise,
        upper_pairwise_worst_case_loss_gap=upper_pairwise,
        sampled_worst_case_regret=sampled_regret,
        upper_worst_case_regret=upper_regret,
        tolerance_admissible_action_mask=admissible,
        minimax_upper_action_index=selected,
        minimax_upper_worst_case_regret=minimum_upper,
        cover_radius_by_quotient=cover,
        pairwise_difference_lipschitz_by_quotient_action=lipschitz,
        pairwise_cover_correction_by_quotient_action=correction,
    )


__all__ = [
    "GaugeCoupledPairwiseDecisionCertificateV1",
    "PAIRWISE_GAUGE_DECISION_CLAIM_BOUNDARY",
    "PairwiseDecisionStatus",
    "certify_gauge_coupled_pairwise_decision",
]
