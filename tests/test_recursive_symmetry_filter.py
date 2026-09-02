from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.equivariant_decision import certify_gauge_coupled_actions
from prob4d.recursive_symmetry_filter import (
    predict_cyclic_equivariant,
    symmetry_belief,
    uniform_group_belief,
    update_invariant_evidence,
    update_symmetry_breaking_evidence,
)


def _identity_quotient_transition(class_count: int) -> np.ndarray:
    return np.eye(class_count)


def _increment_kernel(class_count: int, group_size: int) -> np.ndarray:
    kernel = np.zeros((class_count, class_count, group_size))
    for source in range(class_count):
        for target in range(class_count):
            kernel[source, target, 1 % group_size] = 0.75
            kernel[source, target, -1 % group_size] += 0.25
    return kernel


def test_equivariant_prediction_preserves_uniform_group_conditionals() -> None:
    belief = uniform_group_belief([0.4, 0.6], 16)
    predicted = predict_cyclic_equivariant(
        belief,
        np.array([[0.8, 0.2], [0.1, 0.9]]),
        _increment_kernel(2, 16),
    )

    assert predicted.quotient_mass == pytest.approx([0.38, 0.62])
    assert np.allclose(predicted.conditional_group_probability, 1.0 / 16.0)
    assert predicted.expected_conditional_group_entropy == pytest.approx(math.log(16))


def test_invariant_evidence_identifies_quotient_without_gauge_information() -> None:
    belief = uniform_group_belief([0.5, 0.5], 32)
    audit = update_invariant_evidence(belief, [0.9, 0.1])

    assert audit.posterior.quotient_mass == pytest.approx([0.9, 0.1])
    assert np.array_equal(
        audit.posterior.conditional_group_probability,
        belief.conditional_group_probability,
    )
    assert audit.conditional_group_information == pytest.approx(0.0, abs=1e-15)
    assert audit.maximum_conditional_group_change == pytest.approx(0.0, abs=1e-15)
    assert audit.joint_information == pytest.approx(audit.quotient_information)
    assert audit.chain_rule_error < 1e-14
    assert audit.zero_group_information_verified


def test_repeated_invariant_updates_resolve_action_but_not_group_state() -> None:
    group_size = 64
    belief = uniform_group_belief([0.5, 0.5], group_size)
    cumulative_group_information = 0.0
    initial_quotient_entropy = belief.quotient_entropy

    for _ in range(5):
        belief = predict_cyclic_equivariant(
            belief,
            _identity_quotient_transition(2),
            _increment_kernel(2, group_size),
        )
        audit = update_invariant_evidence(belief, [0.8, 0.2])
        cumulative_group_information += audit.conditional_group_information
        belief = audit.posterior

    loss_by_class = np.array(
        [
            [[0.0, 2.0, 1.0]] * group_size,
            [[2.0, 0.0, 1.0]] * group_size,
        ]
    )
    certificate = certify_gauge_coupled_actions(
        loss_by_class,
        belief.quotient_mass,
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((2, 3, 3)),
        fallback_action=2,
        regret_tolerance=0.0,
    )

    assert belief.quotient_entropy < initial_quotient_entropy / 10.0
    assert belief.expected_conditional_group_entropy == pytest.approx(
        math.log(group_size)
    )
    assert np.allclose(belief.conditional_group_probability, 1.0 / group_size)
    assert cumulative_group_information == pytest.approx(0.0, abs=1e-14)
    assert certificate.posterior_gauge_irrelevant
    assert certificate.robustly_optimal.tolist() == [True, False, False]
    assert certificate.selected_action == 0


def test_symmetry_breaking_evidence_legitimately_adds_group_information() -> None:
    group_size = 64
    belief = uniform_group_belief([1.0], group_size)
    angles = 2.0 * math.pi * np.arange(group_size) / group_size
    likelihood = np.exp(4.0 * np.cos(angles))[None, :]
    audit = update_symmetry_breaking_evidence(belief, likelihood)

    assert not audit.invariant_evidence
    assert not audit.zero_group_information_verified
    assert audit.conditional_group_information > 0.4
    assert audit.maximum_conditional_group_change > 0.01
    assert (
        audit.posterior.expected_conditional_group_entropy
        < belief.expected_conditional_group_entropy
    )
    assert audit.joint_information == pytest.approx(
        audit.quotient_information + audit.conditional_group_information,
        abs=1e-12,
    )


def test_action_independent_invariant_likelihood_is_exactly_preserved() -> None:
    joint = np.array(
        [
            [0.02, 0.08, 0.1, 0.3],
            [0.25, 0.05, 0.15, 0.05],
        ]
    )
    belief = symmetry_belief(joint)
    audit = update_invariant_evidence(belief, [0.2, 0.9])

    assert np.array_equal(
        audit.posterior.conditional_group_probability,
        belief.conditional_group_probability,
    )
    assert audit.zero_group_information_verified


def test_invalid_transitions_and_updates_fail_closed() -> None:
    belief = uniform_group_belief([0.5, 0.5], 4)
    valid_kernel = _increment_kernel(2, 4)
    with pytest.raises(ValueError, match="rows must be probability"):
        predict_cyclic_equivariant(
            belief,
            np.array([[0.8, 0.3], [0.2, 0.8]]),
            valid_kernel,
        )
    broken_kernel = valid_kernel.copy()
    broken_kernel[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="increment kernel"):
        predict_cyclic_equivariant(
            belief,
            np.eye(2),
            broken_kernel,
        )
    with pytest.raises(ValueError, match="zero or invalid evidence"):
        update_invariant_evidence(belief, [0.0, 0.0])
    with pytest.raises(ValueError, match="same shape"):
        update_symmetry_breaking_evidence(belief, np.ones((2, 3)))


def test_recursive_outputs_are_immutable() -> None:
    belief = uniform_group_belief([1.0], 8)
    audit = update_invariant_evidence(belief, [1.0])

    with pytest.raises(ValueError):
        belief.joint_probability[0, 0] = 0.0
    with pytest.raises(ValueError):
        audit.posterior.conditional_group_probability[0, 0] = 0.0
