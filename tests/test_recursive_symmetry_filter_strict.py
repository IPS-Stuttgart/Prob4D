from __future__ import annotations

import numpy as np
import pytest

from prob4d.recursive_symmetry_filter import (
    factorized_symmetry_belief,
    predict_cyclic_equivariant,
    symmetry_belief,
    update_invariant_evidence,
)


def test_zero_mass_class_retains_explicit_conditional_law() -> None:
    conditional = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    belief = factorized_symmetry_belief([1.0, 0.0], conditional)

    assert np.array_equal(belief.conditional_group_probability, conditional)
    assert np.array_equal(belief.joint_probability[1], np.zeros(4))


def test_joint_disintegration_fails_closed_when_zero_mass_law_is_undefined() -> None:
    joint = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="zero-mass quotient classes"):
        symmetry_belief(joint)

    continuation = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    belief = symmetry_belief(joint, zero_mass_conditional=continuation)
    assert np.array_equal(
        belief.conditional_group_probability[1],
        continuation[1],
    )


def test_invariant_evidence_can_eliminate_class_without_replacing_its_gauge() -> None:
    conditional = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    prior = factorized_symmetry_belief([0.4, 0.6], conditional)
    audit = update_invariant_evidence(prior, [1.0, 0.0])

    assert audit.posterior.quotient_mass.tolist() == [1.0, 0.0]
    assert np.array_equal(
        audit.posterior.conditional_group_probability,
        prior.conditional_group_probability,
    )
    assert audit.zero_group_information_verified
    assert audit.conditional_group_information == pytest.approx(0.0, abs=1e-15)


def test_zero_target_mass_prediction_retains_declared_continuation_law() -> None:
    conditional = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    belief = factorized_symmetry_belief([1.0, 0.0], conditional)
    transition = np.array([[1.0, 0.0], [1.0, 0.0]])
    increment = np.zeros((2, 2, 4))
    increment[:, :, 0] = 1.0
    predicted = predict_cyclic_equivariant(belief, transition, increment)

    assert predicted.quotient_mass.tolist() == [1.0, 0.0]
    assert np.array_equal(
        predicted.conditional_group_probability[1],
        conditional[1],
    )
