from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    GaugeCouplingReceiptV1,
    SymmetryCompleteBeliefV1,
    certify_compact_group_decision,
    certify_gauge_coupled_action_orbit,
)


def _rotation(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def _receipt(*, group_id: str = "controlled-s1", valid: bool = True) -> GaugeCouplingReceiptV1:
    return GaugeCouplingReceiptV1(
        group_id=group_id,
        coupling_id="shared-object-frame-v1",
        state_orbit_id="state-orbit-v1",
        action_orbit_id="action-template-orbit-v1",
        shared_group_element_certified=valid,
        execution_binding_certified=valid,
    )


def _coupled_problem(
    node_count: int = 64,
) -> tuple[SymmetryCompleteBeliefV1, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        node_count,
        group_id="controlled-s1",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.75, 0.25],
        quadrature,
        belief_id="two-class-posterior",
    )
    base_states = np.array([[1.0, 0.0], [-1.0, 0.0]])
    action_templates = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    angles = quadrature.nodes[:, 0]
    state_atoms = np.empty((2, node_count, 2))
    action_atoms = np.empty((3, node_count, 2))
    for node_index, angle in enumerate(angles):
        rotation = _rotation(float(angle))
        state_atoms[:, node_index] = base_states @ rotation.T
        action_atoms[:, node_index] = action_templates @ rotation.T
    common_gauge_loss = 2.0 + 2.0 * np.cos(3.0 * angles)
    losses = np.empty((2, node_count, 3))
    for quotient_index in range(2):
        delta = state_atoms[quotient_index, :, None, :] - np.transpose(
            action_atoms,
            (1, 0, 2),
        )
        losses[quotient_index] = np.sum(delta * delta, axis=2) + common_gauge_loss[:, None]
    return belief, losses, state_atoms, action_atoms, common_gauge_loss


def test_gauge_coupled_action_is_identified_without_absolute_loss_invariance() -> None:
    belief, losses, _, _, _ = _coupled_problem()
    certificate = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=_receipt(),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )

    assert certificate.status == "certified-admissible"
    assert certificate.bounds_certified
    assert certificate.sample_difference_invariance_verified
    assert certificate.complete_group_difference_invariance_certified
    assert certificate.maximum_sample_absolute_loss_range > 3.9
    assert certificate.maximum_sample_pairwise_difference_range < 1e-12
    np.testing.assert_allclose(certificate.upper_worst_case_regret, [0.0, 2.0, 1.0], atol=1e-12)
    np.testing.assert_array_equal(
        certificate.tolerance_admissible_action_mask,
        [True, False, False],
    )
    assert certificate.minimax_upper_action_index == 0
    assert certificate.uniquely_tolerance_identified
    assert not certificate.fallback_required


def test_pairwise_equivariance_is_strictly_tighter_than_actionwise_lipschitz() -> None:
    belief, losses, _, _, _ = _coupled_problem(node_count=8)
    actionwise = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=6.0,
        regret_tolerance=0.1,
        lipschitz_bound_certified=True,
    )
    coupled = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=_receipt(),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=0.1,
    )

    assert actionwise.status == "undetermined"
    assert actionwise.fallback_required
    assert actionwise.minimax_upper_worst_case_regret > 4.0
    assert coupled.status == "certified-admissible"
    assert coupled.minimax_upper_worst_case_regret < 1e-12
    assert coupled.uniquely_tolerance_identified


def test_fixed_frame_action_fails_pairwise_equivariance() -> None:
    belief, _, state_atoms, _, common = _coupled_problem()
    fixed_actions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    losses = np.empty((2, belief.quadrature.node_count, 3))
    for quotient_index in range(2):
        delta = state_atoms[quotient_index, :, None, :] - fixed_actions[None, :, :]
        losses[quotient_index] = np.sum(delta * delta, axis=2) + common[:, None]

    certificate = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=_receipt(),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
    )

    assert not certificate.sample_difference_invariance_verified
    assert certificate.status == "scope-not-certified"
    assert certificate.fallback_required
    assert certificate.maximum_sample_pairwise_difference_range > 3.9


def test_shared_and_independent_gauges_have_same_marginals_but_different_decision() -> None:
    belief, _, state_atoms, action_atoms, common = _coupled_problem()
    weights = belief.quadrature.reference_weights
    quotient = belief.quotient_weights

    shared = np.empty(3)
    independent = np.empty(3)
    fixed = np.empty(3)
    fixed_actions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    for action_index in range(3):
        shared_by_class = []
        independent_by_class = []
        fixed_by_class = []
        for quotient_index in range(2):
            shared_squared = np.sum(
                np.square(
                    state_atoms[quotient_index]
                    - action_atoms[action_index]
                ),
                axis=1,
            )
            shared_by_class.append(float(weights @ (shared_squared + common)))

            pair_delta = (
                state_atoms[quotient_index, :, None, :]
                - action_atoms[action_index, None, :, :]
            )
            pair_loss = np.sum(pair_delta * pair_delta, axis=2) + common[:, None]
            independent_by_class.append(float(weights @ pair_loss @ weights))

            fixed_squared = np.sum(
                np.square(
                    state_atoms[quotient_index] - fixed_actions[action_index]
                ),
                axis=1,
            )
            fixed_by_class.append(float(weights @ (fixed_squared + common)))
        shared[action_index] = quotient @ np.asarray(shared_by_class)
        independent[action_index] = quotient @ np.asarray(independent_by_class)
        fixed[action_index] = quotient @ np.asarray(fixed_by_class)

    np.testing.assert_allclose(shared, [3.0, 5.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(independent, [4.0, 4.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(fixed, [4.0, 4.0, 4.0], atol=1e-12)
    assert np.count_nonzero(np.isclose(shared, np.min(shared))) == 1
    assert np.count_nonzero(np.isclose(independent, np.min(independent))) == 3
    assert np.count_nonzero(np.isclose(fixed, np.min(fixed))) == 3


def test_missing_execution_coupling_fails_closed() -> None:
    belief, losses, _, _, _ = _coupled_problem()
    certificate = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=_receipt(valid=False),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )

    assert certificate.sample_difference_invariance_verified
    assert certificate.complete_group_difference_invariance_certified
    assert not certificate.bounds_certified
    assert certificate.status == "scope-not-certified"
    assert certificate.fallback_required


def test_continuous_sample_equality_does_not_prove_whole_group_equivariance() -> None:
    belief, losses, _, _, _ = _coupled_problem()
    certificate = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=_receipt(),
        whole_group_pairwise_difference_invariance_certified=False,
        difference_invariance_atol=1e-10,
    )

    assert certificate.sample_difference_invariance_verified
    assert not certificate.complete_group_difference_invariance_certified
    assert certificate.status == "scope-not-certified"
    assert certificate.fallback_required


def test_exhaustive_finite_group_needs_no_external_whole_group_receipt() -> None:
    angles = np.arange(4, dtype=np.float64) * (0.5 * math.pi)
    quadrature = CompactGroupQuadratureV1(
        group_id="c4",
        metric_id="discrete-c4",
        nodes=angles[:, None],
        reference_weights=np.full(4, 0.25),
        cover_radius=0.0,
        cover_radius_certified=True,
        measure_kind="finite-mass",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="finite-c4",
    )
    common = np.array([0.0, 1.0, 0.0, 1.0])
    losses = np.stack((common, common + 1.0), axis=1)[None, :, :]
    receipt = GaugeCouplingReceiptV1(
        group_id="c4",
        coupling_id="finite-shared",
        state_orbit_id="finite-state",
        action_orbit_id="finite-action",
        shared_group_element_certified=True,
        execution_binding_certified=True,
    )

    certificate = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=receipt,
        difference_invariance_atol=0.0,
    )

    assert certificate.complete_group_difference_invariance_certified
    assert certificate.status == "certified-admissible"
    np.testing.assert_array_equal(certificate.tolerance_admissible_action_mask, [True, False])


def test_gauge_coupled_certificate_arrays_are_immutable() -> None:
    belief, losses, _, _, _ = _coupled_problem()
    certificate = certify_gauge_coupled_action_orbit(
        belief,
        losses,
        coupling_receipt=_receipt(),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )

    with pytest.raises(ValueError, match="read-only"):
        certificate.upper_worst_case_regret[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        certificate.tolerance_admissible_action_mask[0] = False


def test_coupling_receipt_must_match_belief_group() -> None:
    belief, losses, _, _, _ = _coupled_problem()
    with pytest.raises(ValueError, match="different group identifiers"):
        certify_gauge_coupled_action_orbit(
            belief,
            losses,
            coupling_receipt=_receipt(group_id="wrong-group"),
            whole_group_pairwise_difference_invariance_certified=True,
            difference_invariance_atol=1e-10,
        )
