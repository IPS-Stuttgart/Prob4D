"""Decision certificates that preserve one shared compact-group gauge.

A symmetry-sensitive state or query need not force abstention when the action is
transported by the *same* unresolved group element.  This module treats that
shared latent gauge as part of the information contract instead of replacing it
by a point representative or independent per-object gauges.

For quotient class ``c``, sampled group element ``s`` and action template ``a``,
let ``loss[c, s, a]`` be the loss after transforming the state and action by the
same group element.  A finite cover with radius ``rho[c]`` and a Lipschitz bound
``K[c, a, b]`` for each pairwise loss difference yields

    sup_g (loss(c, g, a) - loss(c, g, b))
      <= max_s (loss[c, s, a] - loss[c, s, b]) + K[c, a, b] rho[c].

Weighting the classwise bounds by posterior quotient mass gives an upper bound
on worst-case regret over every unresolved within-orbit conditional belief.
When all cover radii are zero the finite-group result is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CERTIFICATE_VERSION: Final = 1
STATUS_EXACT_OPTIMAL: Final = "certified-exactly-optimal"
STATUS_BOUNDED_REGRET: Final = "certified-bounded-regret"
STATUS_FALLBACK: Final = "fallback-regret-unresolved"


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


def _probability_vector(value: object, *, name: str, length: int) -> FloatArray:
    result = _readonly_float(value, name=name, ndim=1)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return result


def _nonnegative_vector(value: object, *, name: str, length: int) -> FloatArray:
    array = np.asarray(value)
    if array.ndim == 0:
        result = np.full(length, float(array), dtype=np.float64)
    else:
        result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (length,):
        raise ValueError(f"{name} must be a scalar or have shape ({length},)")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    result.setflags(write=False)
    return result


def _representative_indices(
    value: object,
    *,
    class_count: int,
    sample_count: int,
) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim == 0:
        result = np.full(class_count, int(array), dtype=np.int64)
    elif array.shape == (class_count,) and array.dtype.kind in {"i", "u"}:
        result = np.array(array, dtype=np.int64, copy=True)
    else:
        raise ValueError(
            "representative_index must be an integer scalar or one integer per class"
        )
    if np.any(result < 0) or np.any(result >= sample_count):
        raise ValueError("representative_index is outside the sampled group grid")
    result.setflags(write=False)
    return result


def _pairwise_lipschitz(
    value: object,
    *,
    class_count: int,
    action_count: int,
    atol: float,
) -> FloatArray:
    result = _readonly_float(value, name="pairwise_lipschitz", ndim=3)
    expected = (class_count, action_count, action_count)
    if result.shape != expected:
        raise ValueError(f"pairwise_lipschitz must have shape {expected}")
    if np.any(result < 0.0):
        raise ValueError("pairwise_lipschitz must be nonnegative")
    if not np.allclose(result, np.swapaxes(result, 1, 2), rtol=0.0, atol=atol):
        raise ValueError("pairwise_lipschitz must be symmetric in the action pair")
    diagonal = np.diagonal(result, axis1=1, axis2=2)
    if not np.allclose(diagonal, 0.0, rtol=0.0, atol=atol):
        raise ValueError("same-action pairwise Lipschitz bounds must be zero")
    return result


def _validate_scalar(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _validate_action(value: object, *, name: str, action_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result < action_count:
        raise ValueError(f"{name} must index the action set")
    return result


@dataclass(frozen=True)
class GaugeCoupledDecisionCertificate:
    """Auditable finite-cover certificate for shared-gauge action templates."""

    schema_version: int
    quotient_mass: FloatArray
    cover_radius: FloatArray
    representative_index: NDArray[np.int64]
    class_pairwise_sampled_max: FloatArray
    class_pairwise_cover_margin: FloatArray
    pairwise_sampled_lower_bound: FloatArray
    pairwise_upper_bound: FloatArray
    worst_case_regret_lower_bound: FloatArray
    worst_case_regret_upper_bound: FloatArray
    class_decision_equivariance_defect_sampled: FloatArray
    class_decision_equivariance_defect_upper: FloatArray
    posterior_decision_equivariance_defect_upper: FloatArray
    exactly_decision_equivariant: bool
    posterior_gauge_irrelevant: bool
    robustly_optimal: BoolArray
    epsilon_admissible: BoolArray
    minimax_action: int
    fallback_action: int
    selected_action: int
    admitted: bool
    exact_finite_group: bool
    regret_tolerance: float
    equivariance_tolerance: float
    status: str


@dataclass(frozen=True)
class IndependentGaugeControl:
    """Diagnostic obtained after incorrectly decoupling state and action gauges."""

    pairwise_worst_case_gap: FloatArray
    worst_case_regret: FloatArray
    minimax_action: int
    fallback_action: int
    selected_action: int
    admitted: bool
    regret_tolerance: float


def certify_gauge_coupled_actions(
    loss_samples: object,
    quotient_mass: object,
    *,
    cover_radius: object,
    pairwise_lipschitz: object,
    fallback_action: int,
    regret_tolerance: float = 0.0,
    representative_index: object = 0,
    equivariance_tolerance: float = 1e-12,
    atol: float = 1e-12,
) -> GaugeCoupledDecisionCertificate:
    """Certify finite action templates over complete shared-gauge orbits.

    Parameters
    ----------
    loss_samples:
        Array of shape ``(C, S, A)``.  State and action template must be
        transformed by the same sampled group element in the second axis.
    quotient_mass:
        Posterior mass of each quotient class, shape ``(C,)``.
    cover_radius:
        Certified group-metric cover radius for each class, or one scalar.
        Zero means the supplied finite group is exhaustive.
    pairwise_lipschitz:
        Nonnegative symmetric bounds of shape ``(C, A, A)`` for the Lipschitz
        constants of pairwise loss differences over the group metric.
    fallback_action:
        Caller-owned action returned when no minimax template is within the
        registered regret tolerance.

    Notes
    -----
    The certificate does not validate the group action, the cover, the metric,
    the Lipschitz constants, the loss, or the actuator's ability to apply the
    same group transform.  Those are explicit caller-owned assumptions.
    """

    tolerance = _validate_scalar(regret_tolerance, name="regret_tolerance")
    equivariance_tol = _validate_scalar(
        equivariance_tolerance,
        name="equivariance_tolerance",
    )
    numerical_atol = _validate_scalar(atol, name="atol")
    losses = _readonly_float(loss_samples, name="loss_samples", ndim=3)
    class_count, sample_count, action_count = losses.shape
    if class_count < 1 or sample_count < 1 or action_count < 2:
        raise ValueError("loss_samples must have shape (C, S, A), with A >= 2")
    mass = _probability_vector(
        quotient_mass,
        name="quotient_mass",
        length=class_count,
    )
    radius = _nonnegative_vector(
        cover_radius,
        name="cover_radius",
        length=class_count,
    )
    lipschitz = _pairwise_lipschitz(
        pairwise_lipschitz,
        class_count=class_count,
        action_count=action_count,
        atol=numerical_atol,
    )
    fallback = _validate_action(
        fallback_action,
        name="fallback_action",
        action_count=action_count,
    )
    representative = _representative_indices(
        representative_index,
        class_count=class_count,
        sample_count=sample_count,
    )

    pairwise_samples = losses[:, :, :, None] - losses[:, :, None, :]
    sampled_max = np.max(pairwise_samples, axis=1)
    cover_margin = lipschitz * radius[:, None, None]
    class_upper = sampled_max + cover_margin
    pairwise_sampled = np.einsum("c,cab->ab", mass, sampled_max)
    pairwise_upper = np.einsum("c,cab->ab", mass, class_upper)

    same_action = np.arange(action_count)
    pairwise_sampled[same_action, same_action] = 0.0
    pairwise_upper[same_action, same_action] = 0.0
    regret_lower = np.max(pairwise_sampled, axis=1)
    regret_upper = np.max(pairwise_upper, axis=1)

    representative_pairwise = np.stack(
        [pairwise_samples[c, representative[c]] for c in range(class_count)],
        axis=0,
    )
    sampled_defect = np.max(
        np.abs(pairwise_samples - representative_pairwise[:, None, :, :]),
        axis=1,
    )
    complete_defect = sampled_defect + cover_margin
    posterior_defect = np.einsum("c,cab->ab", mass, complete_defect)
    exact_equivariance = bool(np.all(complete_defect <= numerical_atol))
    posterior_irrelevance = bool(np.all(posterior_defect <= equivariance_tol))

    robustly_optimal = np.all(pairwise_upper <= numerical_atol, axis=1)
    epsilon_admissible = regret_upper <= tolerance + numerical_atol
    minimax = int(np.argmin(regret_upper))
    admitted = bool(epsilon_admissible[minimax])
    selected = minimax if admitted else fallback
    exact_finite = bool(np.all(radius == 0.0))
    if bool(robustly_optimal[minimax]):
        status = STATUS_EXACT_OPTIMAL
    elif admitted:
        status = STATUS_BOUNDED_REGRET
    else:
        status = STATUS_FALLBACK

    arrays = (
        pairwise_samples,
        sampled_max,
        cover_margin,
        pairwise_sampled,
        pairwise_upper,
        regret_lower,
        regret_upper,
        sampled_defect,
        complete_defect,
        posterior_defect,
        robustly_optimal,
        epsilon_admissible,
    )
    for array in arrays:
        array.setflags(write=False)

    return GaugeCoupledDecisionCertificate(
        schema_version=CERTIFICATE_VERSION,
        quotient_mass=mass,
        cover_radius=radius,
        representative_index=representative,
        class_pairwise_sampled_max=sampled_max,
        class_pairwise_cover_margin=cover_margin,
        pairwise_sampled_lower_bound=pairwise_sampled,
        pairwise_upper_bound=pairwise_upper,
        worst_case_regret_lower_bound=regret_lower,
        worst_case_regret_upper_bound=regret_upper,
        class_decision_equivariance_defect_sampled=sampled_defect,
        class_decision_equivariance_defect_upper=complete_defect,
        posterior_decision_equivariance_defect_upper=posterior_defect,
        exactly_decision_equivariant=exact_equivariance,
        posterior_gauge_irrelevant=posterior_irrelevance,
        robustly_optimal=_readonly_bool(robustly_optimal),
        epsilon_admissible=_readonly_bool(epsilon_admissible),
        minimax_action=minimax,
        fallback_action=fallback,
        selected_action=selected,
        admitted=admitted,
        exact_finite_group=exact_finite,
        regret_tolerance=tolerance,
        equivariance_tolerance=equivariance_tol,
        status=status,
    )


def certify_independent_gauge_control(
    loss_samples: object,
    quotient_mass: object,
    *,
    fallback_action: int,
    regret_tolerance: float = 0.0,
    atol: float = 1e-12,
) -> IndependentGaugeControl:
    """Evaluate the conservative control that breaks the shared gauge.

    ``loss_samples`` has shape ``(C, S_state, S_action, A)``.  Maximization over
    the state and action group axes independently is intentionally *not* the
    symmetry-complete physical model.  It is a diagnostic for the information
    lost when a common latent transform is replaced by independent gauges.
    """

    tolerance = _validate_scalar(regret_tolerance, name="regret_tolerance")
    numerical_atol = _validate_scalar(atol, name="atol")
    losses = _readonly_float(
        loss_samples,
        name="independent_loss_samples",
        ndim=4,
    )
    class_count, state_count, action_gauge_count, action_count = losses.shape
    if (
        class_count < 1
        or state_count < 1
        or action_gauge_count < 1
        or action_count < 2
    ):
        raise ValueError(
            "independent_loss_samples must have shape (C, Sx, Sa, A), A >= 2"
        )
    mass = _probability_vector(
        quotient_mass,
        name="quotient_mass",
        length=class_count,
    )
    fallback = _validate_action(
        fallback_action,
        name="fallback_action",
        action_count=action_count,
    )
    pairwise = losses[:, :, :, :, None] - losses[:, :, :, None, :]
    class_worst = np.max(pairwise, axis=(1, 2))
    pairwise_gap = np.einsum("c,cab->ab", mass, class_worst)
    diagonal = np.arange(action_count)
    pairwise_gap[diagonal, diagonal] = 0.0
    regret = np.max(pairwise_gap, axis=1)
    minimax = int(np.argmin(regret))
    admitted = bool(regret[minimax] <= tolerance + numerical_atol)
    selected = minimax if admitted else fallback
    pairwise_gap.setflags(write=False)
    regret.setflags(write=False)
    return IndependentGaugeControl(
        pairwise_worst_case_gap=pairwise_gap,
        worst_case_regret=regret,
        minimax_action=minimax,
        fallback_action=fallback,
        selected_action=selected,
        admitted=admitted,
        regret_tolerance=tolerance,
    )


def squared_metric_shared_gauge_losses(
    state_samples: object,
    action_samples: object,
    *,
    metric: object | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    """Return squared losses with one shared group index.

    ``state_samples`` has shape ``(C, S, D)`` and ``action_samples`` has shape
    ``(C, S, A, D)``.  The same ``S`` index is used for state and action.
    """

    numerical_atol = _validate_scalar(atol, name="atol")
    states = _readonly_float(state_samples, name="state_samples", ndim=3)
    actions = _readonly_float(action_samples, name="action_samples", ndim=4)
    class_count, sample_count, dimension = states.shape
    if class_count < 1 or sample_count < 1 or dimension < 1:
        raise ValueError("state_samples must have nonempty shape (C, S, D)")
    if actions.shape[:2] != (class_count, sample_count):
        raise ValueError("state_samples and action_samples must share C and S")
    if actions.shape[2] < 2 or actions.shape[3] != dimension:
        raise ValueError("action_samples must have shape (C, S, A, D), A >= 2")
    weight = _metric_matrix(metric, dimension=dimension, atol=numerical_atol)
    difference = states[:, :, None, :] - actions
    result = np.einsum("csad,de,csae->csa", difference, weight, difference)
    result.setflags(write=False)
    return result


def squared_metric_independent_gauge_losses(
    state_samples: object,
    action_samples: object,
    *,
    metric: object | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    """Return a diagnostic loss grid after decoupling state/action gauges."""

    numerical_atol = _validate_scalar(atol, name="atol")
    states = _readonly_float(state_samples, name="state_samples", ndim=3)
    actions = _readonly_float(action_samples, name="action_samples", ndim=4)
    class_count, state_count, dimension = states.shape
    if class_count < 1 or state_count < 1 or dimension < 1:
        raise ValueError("state_samples must have nonempty shape (C, S, D)")
    if actions.shape[0] != class_count or actions.shape[3] != dimension:
        raise ValueError("state_samples and action_samples must share C and D")
    if actions.shape[1] < 1 or actions.shape[2] < 2:
        raise ValueError("action_samples must contain group samples and actions")
    weight = _metric_matrix(metric, dimension=dimension, atol=numerical_atol)
    difference = states[:, :, None, None, :] - actions[:, None, :, :, :]
    result = np.einsum("cstad,de,cstae->csta", difference, weight, difference)
    result.setflags(write=False)
    return result


def _metric_matrix(metric: object | None, *, dimension: int, atol: float) -> FloatArray:
    if metric is None:
        result = np.eye(dimension, dtype=np.float64)
    else:
        result = _readonly_float(metric, name="metric", ndim=2)
        if result.shape != (dimension, dimension):
            raise ValueError(f"metric must have shape ({dimension}, {dimension})")
        if not np.allclose(result, result.T, rtol=0.0, atol=atol):
            raise ValueError("metric must be symmetric")
        if float(np.min(np.linalg.eigvalsh(result))) < -atol:
            raise ValueError("metric must be positive semidefinite")
        result = np.array(0.5 * (result + result.T), copy=True)
    result.setflags(write=False)
    return result


def so2_covering_radius(angles: object) -> float:
    """Exact covering radius of a nonempty finite subset of ``SO(2)``.

    The metric is wrapped absolute angular distance.  The radius is half the
    largest circular gap and is invariant to ordering, duplicate removal and a
    common angular offset.
    """

    values = _readonly_float(angles, name="angles", ndim=1)
    if values.size < 1:
        raise ValueError("angles must contain at least one sample")
    wrapped = np.unique(np.mod(values, 2.0 * math.pi))
    if wrapped.size == 1:
        return math.pi
    ordered = np.sort(wrapped)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * math.pi)))
    return 0.5 * float(np.max(gaps))


def minimum_uniform_so2_samples(
    pairwise_lipschitz: float,
    maximum_cover_margin: float,
) -> int:
    """Sufficient uniform-grid size for ``L * pi / S <= margin``."""

    lipschitz = _validate_scalar(
        pairwise_lipschitz,
        name="pairwise_lipschitz",
    )
    margin = _validate_scalar(
        maximum_cover_margin,
        name="maximum_cover_margin",
    )
    if margin == 0.0:
        if lipschitz == 0.0:
            return 1
        raise ValueError("positive Lipschitz variation needs positive margin")
    return max(1, int(math.ceil(math.pi * lipschitz / margin)))


__all__ = [
    "CERTIFICATE_VERSION",
    "GaugeCoupledDecisionCertificate",
    "IndependentGaugeControl",
    "STATUS_BOUNDED_REGRET",
    "STATUS_EXACT_OPTIMAL",
    "STATUS_FALLBACK",
    "certify_gauge_coupled_actions",
    "certify_independent_gauge_control",
    "minimum_uniform_so2_samples",
    "so2_covering_radius",
    "squared_metric_independent_gauge_losses",
    "squared_metric_shared_gauge_losses",
]
