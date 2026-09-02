"""Recursive Bayesian filtering without invented finite-group gauge information.

The state is represented by a quotient class and one common cyclic-group index.
Equivariant prediction uses a quotient transition and a group-increment kernel.
Invariant evidence may update quotient masses but must leave the conditional
group law unchanged.  Symmetry-breaking evidence is handled by a separate API
and reports the group information it introduces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

FILTER_VERSION: Final = 1


def _readonly(value: object, *, name: str, ndim: int) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _probability_array(value: object, *, name: str, ndim: int) -> FloatArray:
    result = _readonly(value, name=name, ndim=ndim)
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return result


def _entropy(probability: FloatArray) -> float:
    positive = probability > 0.0
    return float(-np.sum(probability[positive] * np.log(probability[positive])))


def _kl_divergence(posterior: FloatArray, prior: FloatArray) -> float:
    positive = posterior > 0.0
    if np.any(positive & (prior <= 0.0)):
        raise ValueError("posterior assigns mass outside prior support")
    return float(
        np.sum(posterior[positive] * np.log(posterior[positive] / prior[positive]))
    )


@dataclass(frozen=True)
class RecursiveSymmetryBelief:
    """Joint quotient/group belief and its exact disintegration."""

    schema_version: int
    joint_probability: FloatArray
    quotient_mass: FloatArray
    conditional_group_probability: FloatArray
    joint_entropy: float
    quotient_entropy: float
    conditional_group_entropy: FloatArray
    expected_conditional_group_entropy: float


@dataclass(frozen=True)
class SymmetryUpdateAudit:
    """KL-chain and gauge-information audit for one Bayesian update."""

    schema_version: int
    prior: RecursiveSymmetryBelief
    posterior: RecursiveSymmetryBelief
    evidence_probability: float
    invariant_evidence: bool
    joint_information: float
    quotient_information: float
    conditional_group_information: float
    chain_rule_error: float
    maximum_conditional_group_change: float
    zero_group_information_verified: bool


def symmetry_belief(joint_probability: object) -> RecursiveSymmetryBelief:
    """Construct a validated finite quotient-by-cyclic-group belief."""

    joint = _probability_array(
        joint_probability,
        name="joint_probability",
        ndim=2,
    )
    class_count, group_size = joint.shape
    if class_count < 1 or group_size < 1:
        raise ValueError("joint_probability must have nonempty shape (C, K)")
    quotient = np.sum(joint, axis=1)
    conditional = np.zeros_like(joint)
    positive = quotient > 0.0
    conditional[positive] = joint[positive] / quotient[positive, None]
    conditional_entropy = np.array(
        [
            _entropy(conditional[class_index]) if positive[class_index] else 0.0
            for class_index in range(class_count)
        ],
        dtype=np.float64,
    )
    expected_entropy = float(np.dot(quotient, conditional_entropy))
    quotient.setflags(write=False)
    conditional.setflags(write=False)
    conditional_entropy.setflags(write=False)
    return RecursiveSymmetryBelief(
        schema_version=FILTER_VERSION,
        joint_probability=joint,
        quotient_mass=quotient,
        conditional_group_probability=conditional,
        joint_entropy=_entropy(joint.reshape(-1)),
        quotient_entropy=_entropy(quotient),
        conditional_group_entropy=conditional_entropy,
        expected_conditional_group_entropy=expected_entropy,
    )


def uniform_group_belief(
    quotient_mass: object,
    group_size: int,
) -> RecursiveSymmetryBelief:
    """Construct a belief with a Haar-uniform finite cyclic conditional law."""

    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)):
        raise ValueError("group_size must be an integer")
    size = int(group_size)
    if size < 1:
        raise ValueError("group_size must be positive")
    quotient = _probability_array(
        quotient_mass,
        name="quotient_mass",
        ndim=1,
    )
    return symmetry_belief(quotient[:, None] / size * np.ones((1, size)))


def predict_cyclic_equivariant(
    belief: RecursiveSymmetryBelief,
    quotient_transition: object,
    increment_probability: object,
) -> RecursiveSymmetryBelief:
    """Predict through an equivariant finite-cyclic transition.

    ``quotient_transition[c, d]`` is the probability of target quotient ``d``
    from source quotient ``c``. ``increment_probability[c, d, r]`` is the
    conditional probability of adding cyclic increment ``r``.  Dependence on the
    absolute group index is forbidden by this representation.
    """

    joint = belief.joint_probability
    class_count, group_size = joint.shape
    transition = _readonly(
        quotient_transition,
        name="quotient_transition",
        ndim=2,
    )
    if transition.shape != (class_count, class_count):
        raise ValueError(
            "quotient_transition must have shape "
            f"({class_count}, {class_count})"
        )
    if np.any(transition < 0.0) or not np.allclose(
        transition.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("quotient_transition rows must be probability vectors")
    increment = _readonly(
        increment_probability,
        name="increment_probability",
        ndim=3,
    )
    if increment.shape != (class_count, class_count, group_size):
        raise ValueError(
            "increment_probability must have shape "
            f"({class_count}, {class_count}, {group_size})"
        )
    if np.any(increment < 0.0) or not np.allclose(
        increment.sum(axis=2),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("every increment kernel must be a probability vector")

    predicted = np.zeros_like(joint)
    for source_class in range(class_count):
        source = joint[source_class]
        for target_class in range(class_count):
            weight = transition[source_class, target_class]
            if weight == 0.0:
                continue
            convolved = np.zeros(group_size, dtype=np.float64)
            for increment_index in range(group_size):
                probability = increment[source_class, target_class, increment_index]
                if probability != 0.0:
                    convolved += probability * np.roll(source, increment_index)
            predicted[target_class] += weight * convolved
    predicted /= float(predicted.sum())
    return symmetry_belief(predicted)


def update_invariant_evidence(
    belief: RecursiveSymmetryBelief,
    likelihood_by_quotient: object,
    *,
    atol: float = 1e-12,
) -> SymmetryUpdateAudit:
    """Update quotient masses while requiring zero within-orbit information."""

    likelihood = _readonly(
        likelihood_by_quotient,
        name="likelihood_by_quotient",
        ndim=1,
    )
    class_count, group_size = belief.joint_probability.shape
    if likelihood.shape != (class_count,):
        raise ValueError(f"likelihood_by_quotient must have shape ({class_count},)")
    if np.any(likelihood < 0.0):
        raise ValueError("likelihood_by_quotient must be nonnegative")
    tiled = np.repeat(likelihood[:, None], group_size, axis=1)
    audit = _update(belief, tiled, invariant_evidence=True, atol=atol)
    if not audit.zero_group_information_verified:
        raise RuntimeError("invariant evidence introduced group information")
    return audit


def update_symmetry_breaking_evidence(
    belief: RecursiveSymmetryBelief,
    likelihood_by_state: object,
    *,
    atol: float = 1e-12,
) -> SymmetryUpdateAudit:
    """Update with explicitly gauge-sensitive likelihood values."""

    likelihood = _readonly(
        likelihood_by_state,
        name="likelihood_by_state",
        ndim=2,
    )
    if likelihood.shape != belief.joint_probability.shape:
        raise ValueError(
            "likelihood_by_state must have the same shape as the joint belief"
        )
    if np.any(likelihood < 0.0):
        raise ValueError("likelihood_by_state must be nonnegative")
    return _update(belief, likelihood, invariant_evidence=False, atol=atol)


def _update(
    belief: RecursiveSymmetryBelief,
    likelihood: FloatArray,
    *,
    invariant_evidence: bool,
    atol: float,
) -> SymmetryUpdateAudit:
    if isinstance(atol, bool) or not isinstance(
        atol, (int, float, np.integer, np.floating)
    ):
        raise ValueError("atol must be a real scalar")
    numerical_atol = float(atol)
    if not math.isfinite(numerical_atol) or numerical_atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    unnormalized = belief.joint_probability * likelihood
    evidence = float(unnormalized.sum())
    if not math.isfinite(evidence) or evidence <= 0.0:
        raise ValueError("likelihood has zero or invalid evidence probability")
    posterior = symmetry_belief(unnormalized / evidence)
    joint_information = _kl_divergence(
        posterior.joint_probability.reshape(-1),
        belief.joint_probability.reshape(-1),
    )
    quotient_information = _kl_divergence(
        posterior.quotient_mass,
        belief.quotient_mass,
    )
    group_information = 0.0
    maximum_change = 0.0
    for class_index, posterior_mass in enumerate(posterior.quotient_mass):
        if posterior_mass == 0.0:
            continue
        prior_mass = belief.quotient_mass[class_index]
        if prior_mass <= 0.0:
            raise ValueError("posterior quotient mass lies outside prior support")
        posterior_conditional = posterior.conditional_group_probability[class_index]
        prior_conditional = belief.conditional_group_probability[class_index]
        group_information += float(posterior_mass) * _kl_divergence(
            posterior_conditional,
            prior_conditional,
        )
        maximum_change = max(
            maximum_change,
            float(np.max(np.abs(posterior_conditional - prior_conditional))),
        )
    chain_error = abs(joint_information - quotient_information - group_information)
    zero_verified = bool(
        invariant_evidence
        and group_information <= numerical_atol
        and maximum_change <= numerical_atol
        and chain_error <= 10.0 * numerical_atol
    )
    return SymmetryUpdateAudit(
        schema_version=FILTER_VERSION,
        prior=belief,
        posterior=posterior,
        evidence_probability=evidence,
        invariant_evidence=invariant_evidence,
        joint_information=joint_information,
        quotient_information=quotient_information,
        conditional_group_information=group_information,
        chain_rule_error=chain_error,
        maximum_conditional_group_change=maximum_change,
        zero_group_information_verified=zero_verified,
    )


__all__ = [
    "FILTER_VERSION",
    "RecursiveSymmetryBelief",
    "SymmetryUpdateAudit",
    "predict_cyclic_equivariant",
    "symmetry_belief",
    "uniform_group_belief",
    "update_invariant_evidence",
    "update_symmetry_breaking_evidence",
]
