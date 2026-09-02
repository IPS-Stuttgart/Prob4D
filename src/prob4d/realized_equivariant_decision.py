"""Gauge-coupled decision certificates with bounded intervention realization error.

The ideal shared-gauge certificate assumes that action template ``a`` is
executed as ``g . a`` under the same unresolved group element that acts on the
state.  A physical actuator instead realizes ``u_a(g)``.  If

    d_A(u_a(g), g . a) <= epsilon[c, a]

and the registered loss is ``K[c, a]``-Lipschitz in its action argument, then
for every action pair ``(a, b)`` the actual pairwise loss gap is at most the
ideal gap plus

    K[c, a] epsilon[c, a] + K[c, b] epsilon[c, b].

This module composes that explicit intervention margin with the complete-orbit
bound from :mod:`prob4d.equivariant_decision`.  The realization radii and
Lipschitz constants are caller-owned assumptions; this module audits their
algebraic consequence but does not infer or validate them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .equivariant_decision import (
    GaugeCoupledDecisionCertificate,
    STATUS_BOUNDED_REGRET,
    STATUS_EXACT_OPTIMAL,
    STATUS_FALLBACK,
    certify_gauge_coupled_actions,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

REALIZED_CERTIFICATE_VERSION: Final = 1


def _readonly_float(value: object, *, name: str, ndim: int) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_bool(value: object) -> BoolArray:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _class_action_array(
    value: object,
    *,
    name: str,
    class_count: int,
    action_count: int,
) -> FloatArray:
    array = np.asarray(value)
    if array.ndim == 0:
        result = np.full((class_count, action_count), float(array))
    elif array.shape == (action_count,):
        result = np.repeat(
            np.asarray(array, dtype=np.float64)[None, :],
            class_count,
            axis=0,
        )
    else:
        result = np.array(value, dtype=np.float64, copy=True)
    expected = (class_count, action_count)
    if result.shape != expected:
        raise ValueError(
            f"{name} must be a scalar, have shape ({action_count},), "
            f"or have shape {expected}"
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    result.setflags(write=False)
    return result


def _scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class RealizedGaugeCoupledDecisionCertificate:
    """Complete-orbit action certificate including realized-command mismatch."""

    schema_version: int
    ideal_certificate: GaugeCoupledDecisionCertificate
    realization_radius: FloatArray
    action_loss_lipschitz: FloatArray
    class_action_realization_loss_margin: FloatArray
    class_pairwise_realization_margin: FloatArray
    posterior_pairwise_realization_margin: FloatArray
    pairwise_realized_upper_bound: FloatArray
    worst_case_realized_regret_upper_bound: FloatArray
    robustly_optimal_under_realization: BoolArray
    epsilon_admissible_under_realization: BoolArray
    minimax_action: int
    fallback_action: int
    selected_action: int
    admitted: bool
    regret_tolerance: float
    status: str


def intervention_pairwise_realization_margin(
    realization_radius: object,
    action_loss_lipschitz: object,
    *,
    class_count: int,
    action_count: int,
) -> tuple[FloatArray, FloatArray]:
    """Return one-action and pairwise loss margins for bounded realization.

    Inputs can be scalars, one value per action, or one value per class/action.
    The returned pairwise matrix is symmetric and has a zero diagonal because
    comparing an action with itself introduces no regret uncertainty.
    """

    radius = _class_action_array(
        realization_radius,
        name="realization_radius",
        class_count=class_count,
        action_count=action_count,
    )
    lipschitz = _class_action_array(
        action_loss_lipschitz,
        name="action_loss_lipschitz",
        class_count=class_count,
        action_count=action_count,
    )
    one_action = np.array(radius * lipschitz, copy=True)
    pairwise = one_action[:, :, None] + one_action[:, None, :]
    diagonal = np.arange(action_count)
    pairwise[:, diagonal, diagonal] = 0.0
    one_action.setflags(write=False)
    pairwise.setflags(write=False)
    return one_action, pairwise


def certify_realized_gauge_coupled_actions(
    loss_samples: object,
    quotient_mass: object,
    *,
    cover_radius: object,
    pairwise_lipschitz: object,
    realization_radius: object,
    action_loss_lipschitz: object,
    fallback_action: int,
    regret_tolerance: float = 0.0,
    representative_index: object = 0,
    equivariance_tolerance: float = 1e-12,
    atol: float = 1e-12,
) -> RealizedGaugeCoupledDecisionCertificate:
    """Certify action templates under gauge and actuator uncertainty.

    The ideal complete-orbit calculation is delegated to
    :func:`certify_gauge_coupled_actions`.  The pairwise realization margin is
    then added before minimax selection.  If the resulting minimax regret exceeds
    the registered tolerance, the exact caller-owned fallback index is returned.
    """

    tolerance = _scalar(regret_tolerance, name="regret_tolerance")
    numerical_atol = _scalar(atol, name="atol")
    ideal = certify_gauge_coupled_actions(
        loss_samples,
        quotient_mass,
        cover_radius=cover_radius,
        pairwise_lipschitz=pairwise_lipschitz,
        fallback_action=fallback_action,
        regret_tolerance=tolerance,
        representative_index=representative_index,
        equivariance_tolerance=equivariance_tolerance,
        atol=numerical_atol,
    )
    class_count = ideal.quotient_mass.size
    action_count = ideal.pairwise_upper_bound.shape[0]
    radius = _class_action_array(
        realization_radius,
        name="realization_radius",
        class_count=class_count,
        action_count=action_count,
    )
    lipschitz = _class_action_array(
        action_loss_lipschitz,
        name="action_loss_lipschitz",
        class_count=class_count,
        action_count=action_count,
    )
    one_action, class_pairwise = intervention_pairwise_realization_margin(
        radius,
        lipschitz,
        class_count=class_count,
        action_count=action_count,
    )
    posterior_margin = np.einsum(
        "c,cab->ab",
        ideal.quotient_mass,
        class_pairwise,
    )
    pairwise_upper = np.array(
        ideal.pairwise_upper_bound + posterior_margin,
        copy=True,
    )
    diagonal = np.arange(action_count)
    pairwise_upper[diagonal, diagonal] = 0.0
    regret = np.max(pairwise_upper, axis=1)
    robust = np.all(pairwise_upper <= numerical_atol, axis=1)
    admissible_actions = regret <= tolerance + numerical_atol
    minimax = int(np.argmin(regret))
    admitted = bool(admissible_actions[minimax])
    selected = minimax if admitted else ideal.fallback_action
    if bool(robust[minimax]):
        status = STATUS_EXACT_OPTIMAL
    elif admitted:
        status = STATUS_BOUNDED_REGRET
    else:
        status = STATUS_FALLBACK

    posterior_margin.setflags(write=False)
    pairwise_upper.setflags(write=False)
    regret.setflags(write=False)
    robust.setflags(write=False)
    admissible_actions.setflags(write=False)
    return RealizedGaugeCoupledDecisionCertificate(
        schema_version=REALIZED_CERTIFICATE_VERSION,
        ideal_certificate=ideal,
        realization_radius=radius,
        action_loss_lipschitz=lipschitz,
        class_action_realization_loss_margin=one_action,
        class_pairwise_realization_margin=class_pairwise,
        posterior_pairwise_realization_margin=posterior_margin,
        pairwise_realized_upper_bound=pairwise_upper,
        worst_case_realized_regret_upper_bound=regret,
        robustly_optimal_under_realization=_readonly_bool(robust),
        epsilon_admissible_under_realization=_readonly_bool(admissible_actions),
        minimax_action=minimax,
        fallback_action=ideal.fallback_action,
        selected_action=selected,
        admitted=admitted,
        regret_tolerance=tolerance,
        status=status,
    )


__all__ = [
    "REALIZED_CERTIFICATE_VERSION",
    "RealizedGaugeCoupledDecisionCertificate",
    "certify_realized_gauge_coupled_actions",
    "intervention_pairwise_realization_margin",
]
