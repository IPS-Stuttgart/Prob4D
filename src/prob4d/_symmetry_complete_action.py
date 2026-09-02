"""Gauge-coupled equivariant action-orbit certificates.

A fixed world-frame action can remain unidentified when a physical state is known
only up to a compact-group coordinate.  This module certifies a different
object: an action *template* that is transformed by the same group element as
the state.  The caller must provide a provenance receipt that the state and
action branches share one group realization at execution time.

The decisive condition is weaker than invariance of every absolute loss.  Only
pairwise action-loss differences must be invariant along the coupled orbit.
Action-independent gauge terms may therefore vary arbitrarily without changing
the certified decision.
"""

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
    _canonical_string,
    _genuine_bool,
    _immutable_float,
)

BoolArray: TypeAlias = NDArray[np.bool_]
GaugeCoupledActionStatus: TypeAlias = Literal[
    "certified-admissible",
    "certified-no-admissible-action",
    "undetermined",
    "scope-not-certified",
]


def _immutable_bool_vector(value: ArrayLike, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    contiguous = np.ascontiguousarray(raw, dtype=np.bool_)
    result: BoolArray = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)
    return result


@dataclass(frozen=True, slots=True)
class GaugeCouplingReceiptV1:
    """Caller-owned receipt for shared state/action gauge realization.

    The receipt is a provenance contract, not an inferred fact.  In particular,
    ``shared_group_element_certified`` states that the action atom paired with a
    state atom uses the same group element, while
    ``execution_binding_certified`` states that the deployed command generator
    preserves that pairing rather than selecting an unrelated world-frame
    representative.
    """

    group_id: str
    coupling_id: str
    state_orbit_id: str
    action_orbit_id: str
    shared_group_element_certified: bool
    execution_binding_certified: bool

    def __post_init__(self) -> None:
        for name in ("group_id", "coupling_id", "state_orbit_id", "action_orbit_id"):
            object.__setattr__(
                self,
                name,
                _canonical_string(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "shared_group_element_certified",
            _genuine_bool(
                self.shared_group_element_certified,
                name="shared_group_element_certified",
            ),
        )
        object.__setattr__(
            self,
            "execution_binding_certified",
            _genuine_bool(
                self.execution_binding_certified,
                name="execution_binding_certified",
            ),
        )

    @property
    def scope_certified(self) -> bool:
        return self.shared_group_element_certified and self.execution_binding_certified


@dataclass(frozen=True, slots=True)
class GaugeCoupledActionCertificateV1:
    """Decision certificate for action templates coupled to an unresolved gauge."""

    status: GaugeCoupledActionStatus
    receipt: GaugeCouplingReceiptV1
    bounds_certified: bool
    sample_difference_invariance_verified: bool
    complete_group_difference_invariance_certified: bool
    difference_invariance_atol: float
    regret_tolerance: float
    absolute_loss_range_by_quotient_action: FloatArray
    pairwise_difference_range_by_quotient_action: FloatArray
    lower_pairwise_expected_loss_gap: FloatArray
    upper_pairwise_expected_loss_gap: FloatArray
    lower_worst_case_regret: FloatArray
    upper_worst_case_regret: FloatArray
    tolerance_admissible_action_mask: BoolArray
    minimax_upper_action_index: int
    minimax_upper_worst_case_regret: float

    def __post_init__(self) -> None:
        allowed: tuple[GaugeCoupledActionStatus, ...] = (
            "certified-admissible",
            "certified-no-admissible-action",
            "undetermined",
            "scope-not-certified",
        )
        if self.status not in allowed:
            raise ValueError("unsupported gauge-coupled action status")
        if not isinstance(self.receipt, GaugeCouplingReceiptV1):
            raise TypeError("receipt must be GaugeCouplingReceiptV1")
        bounds_certified = _genuine_bool(self.bounds_certified, name="bounds_certified")
        sample_verified = _genuine_bool(
            self.sample_difference_invariance_verified,
            name="sample_difference_invariance_verified",
        )
        complete_certified = _genuine_bool(
            self.complete_group_difference_invariance_certified,
            name="complete_group_difference_invariance_certified",
        )
        invariance_atol = float(self.difference_invariance_atol)
        regret_tolerance = float(self.regret_tolerance)
        minimum = float(self.minimax_upper_worst_case_regret)
        for name, value in (
            ("difference_invariance_atol", invariance_atol),
            ("regret_tolerance", regret_tolerance),
            ("minimax_upper_worst_case_regret", minimum),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        absolute_range = _immutable_float(
            self.absolute_loss_range_by_quotient_action,
            name="absolute_loss_range_by_quotient_action",
            ndim=2,
        )
        difference_range = _immutable_float(
            self.pairwise_difference_range_by_quotient_action,
            name="pairwise_difference_range_by_quotient_action",
            ndim=3,
        )
        lower_pairwise = _immutable_float(
            self.lower_pairwise_expected_loss_gap,
            name="lower_pairwise_expected_loss_gap",
            ndim=2,
        )
        upper_pairwise = _immutable_float(
            self.upper_pairwise_expected_loss_gap,
            name="upper_pairwise_expected_loss_gap",
            ndim=2,
        )
        lower_regret = _immutable_float(
            self.lower_worst_case_regret,
            name="lower_worst_case_regret",
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
        quotient_count, action_count = absolute_range.shape
        if quotient_count < 1 or action_count < 2:
            raise ValueError("at least one quotient and two actions are required")
        if difference_range.shape != (quotient_count, action_count, action_count):
            raise ValueError("pairwise difference range has the wrong shape")
        if lower_pairwise.shape != (action_count, action_count):
            raise ValueError("lower pairwise gap has the wrong shape")
        if upper_pairwise.shape != lower_pairwise.shape:
            raise ValueError("upper pairwise gap has the wrong shape")
        if lower_regret.shape != (action_count,) or upper_regret.shape != (action_count,):
            raise ValueError("regret vectors have the wrong shape")
        if admissible.shape != (action_count,):
            raise ValueError("admissible action mask has the wrong shape")
        if np.any(absolute_range < 0.0) or np.any(difference_range < 0.0):
            raise ValueError("loss ranges must be nonnegative")
        if np.any(upper_pairwise + _NUMERICAL_ATOL < lower_pairwise):
            raise ValueError("upper pairwise gap is below its lower bound")
        if np.any(upper_regret + _NUMERICAL_ATOL < lower_regret):
            raise ValueError("upper regret is below its lower bound")
        if np.any(lower_regret < -_NUMERICAL_ATOL):
            raise ValueError("lower regret must be nonnegative")

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

        expected_admissible = bounds_certified & (
            upper_regret <= regret_tolerance + _NUMERICAL_ATOL
        )
        if not np.array_equal(admissible, expected_admissible):
            raise ValueError("admissible mask does not match certified upper regret")
        lower_rejects_all = bool(np.all(lower_regret > regret_tolerance + _NUMERICAL_ATOL))
        if self.status == "scope-not-certified":
            if bounds_certified or np.any(admissible):
                raise ValueError("uncertified scope must reject every action")
        elif not bounds_certified:
            raise ValueError("classified action bounds must be certified")
        if (self.status == "certified-admissible") != bool(np.any(admissible)):
            raise ValueError("certified-admissible status disagrees with action mask")
        if (self.status == "certified-no-admissible-action") != (
            bounds_certified and not np.any(admissible) and lower_rejects_all
        ):
            raise ValueError("no-admissible-action status disagrees with regret bounds")
        if self.status == "undetermined" and (
            not bounds_certified or np.any(admissible) or lower_rejects_all
        ):
            raise ValueError("undetermined status disagrees with regret bounds")

        object.__setattr__(self, "bounds_certified", bounds_certified)
        object.__setattr__(
            self,
            "sample_difference_invariance_verified",
            sample_verified,
        )
        object.__setattr__(
            self,
            "complete_group_difference_invariance_certified",
            complete_certified,
        )
        object.__setattr__(self, "difference_invariance_atol", invariance_atol)
        object.__setattr__(self, "regret_tolerance", regret_tolerance)
        object.__setattr__(self, "absolute_loss_range_by_quotient_action", absolute_range)
        object.__setattr__(
            self,
            "pairwise_difference_range_by_quotient_action",
            difference_range,
        )
        object.__setattr__(self, "lower_pairwise_expected_loss_gap", lower_pairwise)
        object.__setattr__(self, "upper_pairwise_expected_loss_gap", upper_pairwise)
        object.__setattr__(self, "lower_worst_case_regret", lower_regret)
        object.__setattr__(self, "upper_worst_case_regret", upper_regret)
        object.__setattr__(self, "tolerance_admissible_action_mask", admissible)
        object.__setattr__(self, "minimax_upper_action_index", action_index)
        object.__setattr__(self, "minimax_upper_worst_case_regret", minimum)

    @property
    def action_count(self) -> int:
        return int(self.upper_worst_case_regret.size)

    @property
    def maximum_sample_absolute_loss_range(self) -> float:
        return float(np.max(self.absolute_loss_range_by_quotient_action))

    @property
    def maximum_sample_pairwise_difference_range(self) -> float:
        return float(np.max(self.pairwise_difference_range_by_quotient_action))

    @property
    def uniquely_tolerance_identified(self) -> bool:
        return bool(np.count_nonzero(self.tolerance_admissible_action_mask) == 1)

    @property
    def fallback_required(self) -> bool:
        return not bool(np.any(self.tolerance_admissible_action_mask))


def certify_gauge_coupled_action_orbit(
    belief: SymmetryCompleteBeliefV1,
    coupled_loss_by_quotient_group_action: ArrayLike,
    *,
    coupling_receipt: GaugeCouplingReceiptV1,
    whole_group_pairwise_difference_invariance_certified: bool = False,
    difference_invariance_atol: float = 1e-12,
    regret_tolerance: float = 0.0,
) -> GaugeCoupledActionCertificateV1:
    """Certify a co-transformed action template without selecting a gauge.

    ``coupled_loss_by_quotient_group_action[c, k, a]`` is the loss obtained when
    quotient state ``c`` and action template ``a`` are transformed by the same
    group node ``k``.  If every pairwise action-loss difference is invariant in
    ``k``, all Bayes comparisons are independent of the unresolved conditional
    group law.  Absolute losses may still contain arbitrary action-independent
    gauge terms.

    Finite groups require a certified zero-radius exhaustive quadrature.
    Continuous groups additionally require a caller-owned complete-group
    pairwise-difference invariance certificate.  Equality on sampled nodes is
    never promoted to a whole-group claim by this function.
    """

    if not isinstance(belief, SymmetryCompleteBeliefV1):
        raise TypeError("belief must be SymmetryCompleteBeliefV1")
    if not isinstance(coupling_receipt, GaugeCouplingReceiptV1):
        raise TypeError("coupling_receipt must be GaugeCouplingReceiptV1")
    if coupling_receipt.group_id != belief.quadrature.group_id:
        raise ValueError("coupling receipt and belief use different group identifiers")
    external_complete = _genuine_bool(
        whole_group_pairwise_difference_invariance_certified,
        name="whole_group_pairwise_difference_invariance_certified",
    )
    if isinstance(difference_invariance_atol, (bool, np.bool_)):
        raise TypeError("difference_invariance_atol must be a real scalar")
    if isinstance(regret_tolerance, (bool, np.bool_)):
        raise TypeError("regret_tolerance must be a real scalar")
    invariance_atol = float(difference_invariance_atol)
    tolerance = float(regret_tolerance)
    if not math.isfinite(invariance_atol) or invariance_atol < 0.0:
        raise ValueError("difference_invariance_atol must be finite and nonnegative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("regret_tolerance must be finite and nonnegative")

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
    absolute_range: FloatArray = np.max(losses, axis=1) - np.min(losses, axis=1)
    difference: FloatArray = losses[:, :, :, None] - losses[:, :, None, :]
    class_lower: FloatArray = np.min(difference, axis=1)
    class_upper: FloatArray = np.max(difference, axis=1)
    difference_range: FloatArray = class_upper - class_lower
    diagonal = np.arange(action_count)
    difference_range[:, diagonal, diagonal] = 0.0

    active = belief.quotient_weights > 0.0
    maximum_active_range = float(np.max(difference_range[active]))
    sample_verified = maximum_active_range <= invariance_atol + _NUMERICAL_ATOL
    finite_complete = (
        belief.quadrature.measure_kind == "finite-mass"
        and belief.quadrature.cover_radius_certified
        and belief.quadrature.cover_radius <= _NUMERICAL_ATOL
    )
    complete_certified = sample_verified and (finite_complete or external_complete)
    bounds_certified = coupling_receipt.scope_certified and complete_certified

    lower_pairwise: FloatArray = np.tensordot(
        belief.quotient_weights,
        class_lower,
        axes=(0, 0),
    )
    upper_pairwise: FloatArray = np.tensordot(
        belief.quotient_weights,
        class_upper,
        axes=(0, 0),
    )
    np.fill_diagonal(lower_pairwise, 0.0)
    np.fill_diagonal(upper_pairwise, 0.0)
    lower_regret: FloatArray = np.maximum(np.max(lower_pairwise, axis=1), 0.0)
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
    lower_rejects_all = bool(np.all(lower_regret > tolerance + _NUMERICAL_ATOL))
    if not bounds_certified:
        status: GaugeCoupledActionStatus = "scope-not-certified"
    elif np.any(admissible):
        status = "certified-admissible"
    elif lower_rejects_all:
        status = "certified-no-admissible-action"
    else:
        status = "undetermined"

    return GaugeCoupledActionCertificateV1(
        status=status,
        receipt=coupling_receipt,
        bounds_certified=bounds_certified,
        sample_difference_invariance_verified=sample_verified,
        complete_group_difference_invariance_certified=complete_certified,
        difference_invariance_atol=invariance_atol,
        regret_tolerance=tolerance,
        absolute_loss_range_by_quotient_action=absolute_range,
        pairwise_difference_range_by_quotient_action=difference_range,
        lower_pairwise_expected_loss_gap=lower_pairwise,
        upper_pairwise_expected_loss_gap=upper_pairwise,
        lower_worst_case_regret=lower_regret,
        upper_worst_case_regret=upper_regret,
        tolerance_admissible_action_mask=admissible,
        minimax_upper_action_index=selected,
        minimax_upper_worst_case_regret=minimum_upper,
    )


__all__ = [
    "GaugeCoupledActionCertificateV1",
    "GaugeCoupledActionStatus",
    "GaugeCouplingReceiptV1",
    "certify_gauge_coupled_action_orbit",
]
