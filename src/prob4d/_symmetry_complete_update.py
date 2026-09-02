"""Internal symmetry-preserving and symmetry-breaking Bayesian updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._symmetry_complete_base import (
    _PROBABILITY_ATOL,
    EvidenceSemantics,
    FloatArray,
    SymmetryCompleteBeliefV1,
    _genuine_bool,
    _immutable_float,
    _kl_divergence,
)


@dataclass(frozen=True, slots=True)
class SymmetryInformationV1:
    """KL chain-rule audit for one symmetry-aware Bayesian update."""

    quotient_information_nats: float
    gauge_information_nats: float
    total_information_nats: float
    maximum_conditional_l1_change: float
    maximum_relative_orbit_likelihood_spread: float
    quadrature_invariance_verified: bool
    whole_group_invariance_certified: bool

    def __post_init__(self) -> None:
        for name in (
            "quotient_information_nats",
            "gauge_information_nats",
            "total_information_nats",
            "maximum_conditional_l1_change",
            "maximum_relative_orbit_likelihood_spread",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < -1e-13:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, max(0.0, value))
        object.__setattr__(
            self,
            "quadrature_invariance_verified",
            _genuine_bool(
                self.quadrature_invariance_verified,
                name="quadrature_invariance_verified",
            ),
        )
        object.__setattr__(
            self,
            "whole_group_invariance_certified",
            _genuine_bool(
                self.whole_group_invariance_certified,
                name="whole_group_invariance_certified",
            ),
        )
        if not math.isclose(
            self.total_information_nats,
            self.quotient_information_nats + self.gauge_information_nats,
            rel_tol=0.0,
            abs_tol=2e-11,
        ):
            raise ValueError("information decomposition does not satisfy the KL chain rule")


@dataclass(frozen=True, slots=True)
class SymmetryUpdateV1:
    """Posterior and audit from one declared symmetry-aware update."""

    prior: SymmetryCompleteBeliefV1
    posterior: SymmetryCompleteBeliefV1
    evidence_semantics: EvidenceSemantics
    information: SymmetryInformationV1

    def __post_init__(self) -> None:
        if not isinstance(self.prior, SymmetryCompleteBeliefV1):
            raise TypeError("prior must be SymmetryCompleteBeliefV1")
        if not isinstance(self.posterior, SymmetryCompleteBeliefV1):
            raise TypeError("posterior must be SymmetryCompleteBeliefV1")
        if self.prior.quadrature is not self.posterior.quadrature:
            raise ValueError("prior and posterior must share the identical quadrature object")
        if self.evidence_semantics not in ("orbit-invariant", "symmetry-breaking"):
            raise ValueError("unsupported evidence_semantics")
        if not isinstance(self.information, SymmetryInformationV1):
            raise TypeError("information must be SymmetryInformationV1")


def _relative_orbit_spread(
    likelihood: FloatArray,
    prior: SymmetryCompleteBeliefV1,
) -> tuple[FloatArray, float]:
    spreads = np.zeros(prior.quotient_count, dtype=np.float64)
    for quotient_index in range(prior.quotient_count):
        supported = prior.group_conditional_weights[quotient_index] > 0.0
        values = likelihood[quotient_index, supported]
        if values.size == 0:
            raise ValueError("every quotient conditional must contain prior support")
        maximum = float(np.max(np.abs(values)))
        if maximum == 0.0:
            spreads[quotient_index] = 0.0
        else:
            spreads[quotient_index] = (float(np.max(values)) - float(np.min(values))) / maximum
    return spreads, float(np.max(spreads))


def update_symmetry_complete_belief(
    prior: SymmetryCompleteBeliefV1,
    likelihood_by_quotient_group: ArrayLike,
    *,
    evidence_semantics: EvidenceSemantics,
    posterior_belief_id: str,
    invariance_atol: float = _PROBABILITY_ATOL,
    whole_group_invariance_certified: bool = False,
) -> SymmetryUpdateV1:
    """Apply one likelihood without inventing unsupported group information.

    In ``orbit-invariant`` mode the likelihood must be constant, up to
    ``invariance_atol``, over every positive-prior group node within each quotient
    class. For a continuous group, the caller must additionally certify that
    invariance holds over the complete group; equality on finitely many nodes is
    not enough. The implementation then preserves the prior conditional group
    law byte-for-byte and updates only quotient masses. A non-invariant
    likelihood is rejected rather than silently converted into gauge information.

    In ``symmetry-breaking`` mode a full likelihood is allowed to update both the
    quotient and conditional group laws. This mode must be used only for an
    explicitly justified symmetry-breaking observation.
    """

    if not isinstance(prior, SymmetryCompleteBeliefV1):
        raise TypeError("prior must be SymmetryCompleteBeliefV1")
    if evidence_semantics not in ("orbit-invariant", "symmetry-breaking"):
        raise ValueError("unsupported evidence_semantics")
    external_invariance = _genuine_bool(
        whole_group_invariance_certified,
        name="whole_group_invariance_certified",
    )
    if isinstance(invariance_atol, (bool, np.bool_)):
        raise TypeError("invariance_atol must be a real scalar")
    atol = float(invariance_atol)
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("invariance_atol must be finite and nonnegative")
    likelihood = _immutable_float(
        likelihood_by_quotient_group,
        name="likelihood_by_quotient_group",
        ndim=2,
    )
    if likelihood.shape != (
        prior.quotient_count,
        prior.quadrature.node_count,
    ):
        raise ValueError("likelihood has the wrong quotient/group shape")
    if np.any(likelihood < 0.0):
        raise ValueError("likelihood must be nonnegative")

    spreads, maximum_spread = _relative_orbit_spread(likelihood, prior)
    quadrature_invariant = bool(np.all(spreads <= atol))
    whole_group_certified = quadrature_invariant and (
        prior.quadrature.measure_kind == "finite-mass" or external_invariance
    )
    if evidence_semantics == "orbit-invariant" and not quadrature_invariant:
        offending = np.flatnonzero(spreads > atol).tolist()
        raise ValueError(
            "orbit-invariant evidence varies over prior-supported group nodes "
            f"for quotient classes {offending}"
        )
    if (
        evidence_semantics == "orbit-invariant"
        and prior.quadrature.measure_kind == "continuous-density"
        and not external_invariance
    ):
        raise ValueError(
            "continuous orbit-invariant evidence requires a complete-group "
            "invariance certificate; quadrature equality alone is insufficient"
        )

    prior_joint = prior.joint_weights
    if evidence_semantics == "orbit-invariant":
        quotient_likelihood = np.sum(
            prior.group_conditional_weights * likelihood,
            axis=1,
            dtype=np.float64,
        )
        unnormalized_quotient = prior.quotient_weights * quotient_likelihood
        evidence = float(np.sum(unnormalized_quotient, dtype=np.float64))
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise ValueError("likelihood has zero prior-supported evidence")
        posterior_quotient = unnormalized_quotient / evidence
        posterior_conditionals = prior.group_conditional_weights
    else:
        unnormalized_joint = prior_joint * likelihood
        evidence = float(np.sum(unnormalized_joint, dtype=np.float64))
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise ValueError("likelihood has zero prior-supported evidence")
        posterior_joint = unnormalized_joint / evidence
        posterior_quotient = np.sum(posterior_joint, axis=1, dtype=np.float64)
        posterior_conditionals = prior.group_conditional_weights.copy()
        for quotient_index, mass in enumerate(posterior_quotient):
            if mass > 0.0:
                posterior_conditionals[quotient_index] = posterior_joint[quotient_index] / mass

    posterior = SymmetryCompleteBeliefV1(
        quotient_weights=posterior_quotient,
        group_conditional_weights=posterior_conditionals,
        quadrature=prior.quadrature,
        belief_id=posterior_belief_id,
    )
    if evidence_semantics == "orbit-invariant":
        # Reuse the exact bytes-backed conditional array rather than normalizing
        # it a second time. This makes unsupported gauge information impossible
        # to introduce through numerical renormalization.
        object.__setattr__(
            posterior,
            "group_conditional_weights",
            prior.group_conditional_weights,
        )
    quotient_information = _kl_divergence(
        posterior.quotient_weights,
        prior.quotient_weights,
    )
    gauge_information = 0.0
    for quotient_index, mass in enumerate(posterior.quotient_weights):
        if mass > 0.0:
            gauge_information += float(mass) * _kl_divergence(
                posterior.group_conditional_weights[quotient_index],
                prior.group_conditional_weights[quotient_index],
            )
    total_information = _kl_divergence(
        posterior.joint_weights.reshape(-1),
        prior_joint.reshape(-1),
    )
    maximum_change = float(
        np.max(
            np.sum(
                np.abs(posterior.group_conditional_weights - prior.group_conditional_weights),
                axis=1,
            )
        )
    )
    information = SymmetryInformationV1(
        quotient_information_nats=quotient_information,
        gauge_information_nats=gauge_information,
        total_information_nats=total_information,
        maximum_conditional_l1_change=maximum_change,
        maximum_relative_orbit_likelihood_spread=maximum_spread,
        quadrature_invariance_verified=quadrature_invariant,
        whole_group_invariance_certified=whole_group_certified,
    )
    if evidence_semantics == "orbit-invariant":
        if not np.array_equal(
            posterior.group_conditional_weights,
            prior.group_conditional_weights,
        ):
            raise AssertionError("orbit-invariant update changed conditional group law")
        if information.gauge_information_nats > 1e-14:
            raise AssertionError("orbit-invariant update created gauge information")
    return SymmetryUpdateV1(prior, posterior, evidence_semantics, information)


__all__ = [
    "SymmetryInformationV1",
    "SymmetryUpdateV1",
    "update_symmetry_complete_belief",
]
