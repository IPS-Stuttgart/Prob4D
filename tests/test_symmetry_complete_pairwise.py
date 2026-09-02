from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    GaugeCouplingReceiptV1,
    SymmetryCompleteBeliefV1,
    certify_compact_group_decision,
    certify_gauge_coupled_pairwise_decision,
)


def _rotation(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def _receipt(*, valid: bool = True) -> GaugeCouplingReceiptV1:
    return GaugeCouplingReceiptV1(
        group_id="approximate-action-s1",
        coupling_id="approximate-shared-object-frame-v1",
        state_orbit_id="approximate-state-orbit-v1",
        action_orbit_id="approximate-action-orbit-v1",
        shared_group_element_certified=valid,
        execution_binding_certified=valid,
    )


def _approximate_problem(
    node_count: int = 8,
) -> tuple[
    SymmetryCompleteBeliefV1,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cover_radius = math.pi / node_count
    angles = (np.arange(node_count, dtype=np.float64) + 0.5) * (
        2.0 * math.pi / node_count
    )
    quadrature = CompactGroupQuadratureV1(
        group_id="approximate-action-s1",
        metric_id="wrapped-angle-radians-v1",
        nodes=angles[:, None],
        reference_weights=np.full(node_count, 1.0 / node_count),
        cover_radius=cover_radius,
        cover_radius_certified=True,
        measure_kind="continuous-density",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.75, 0.25],
        quadrature,
        belief_id="approximate-two-class-posterior",
    )
    base_states = np.array([[1.0, 0.0], [-1.0, 0.0]])
    action_templates = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    action_perturbation = np.array([0.0, 0.1, -0.05])
    losses = np.empty((2, node_count, 3))
    for node_index, angle in enumerate(angles):
        rotation = _rotation(float(angle))
        states = base_states @ rotation.T
        actions = action_templates @ rotation.T
        common = 2.0 + 2.0 * math.cos(3.0 * float(angle))
        for quotient_index in range(2):
            losses[quotient_index, node_index] = (
                np.sum(np.square(states[quotient_index] - actions), axis=1)
                + common
                + action_perturbation * math.cos(float(angle))
            )
    pairwise_lipschitz = np.abs(
        action_perturbation[:, None] - action_perturbation[None, :]
    )
    actionwise_lipschitz = 6.0 + np.abs(action_perturbation)
    return (
        belief,
        losses,
        pairwise_lipschitz,
        actionwise_lipschitz,
        action_perturbation,
    )


def _dense_true_regret(action_perturbation: np.ndarray) -> np.ndarray:
    dense_angles = np.linspace(0.0, 2.0 * math.pi, 200_001, endpoint=False)
    quotient_weights = np.array([0.75, 0.25])
    base_states = np.array([[1.0, 0.0], [-1.0, 0.0]])
    action_templates = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    class_supremum = np.full((2, 3, 3), -np.inf)
    for angle in dense_angles:
        rotation = _rotation(float(angle))
        states = base_states @ rotation.T
        actions = action_templates @ rotation.T
        common = 2.0 + 2.0 * math.cos(3.0 * float(angle))
        losses = np.empty((2, 3))
        for quotient_index in range(2):
            losses[quotient_index] = (
                np.sum(np.square(states[quotient_index] - actions), axis=1)
                + common
                + action_perturbation * math.cos(float(angle))
            )
        differences = losses[:, :, None] - losses[:, None, :]
        class_supremum = np.maximum(class_supremum, differences)
    pairwise = np.tensordot(quotient_weights, class_supremum, axes=(0, 0))
    np.fill_diagonal(pairwise, 0.0)
    return np.maximum(np.max(pairwise, axis=1), 0.0)


def test_pairwise_lipschitz_cover_certifies_approximate_equivariance() -> None:
    belief, losses, pairwise_lipschitz, _, perturbation = _approximate_problem()
    certificate = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        regret_tolerance=0.0,
        pairwise_lipschitz_bound_certified=True,
    )
    true_regret = _dense_true_regret(perturbation)

    assert certificate.status == "certified-admissible"
    assert certificate.bounds_certified
    assert certificate.uniquely_tolerance_identified
    assert certificate.minimax_upper_action_index == 0
    np.testing.assert_array_equal(
        certificate.tolerance_admissible_action_mask,
        [True, False, False],
    )
    assert certificate.maximum_pairwise_cover_correction > 0.05
    assert np.max(certificate.sampled_difference_range_by_quotient_action) > 0.25
    assert np.all(
        certificate.sampled_worst_case_regret <= true_regret + 2e-6
    )
    assert np.all(true_regret <= certificate.upper_worst_case_regret + 2e-6)
    assert certificate.upper_worst_case_regret[0] == 0.0


def test_pairwise_regularization_is_strictly_tighter_than_actionwise_bounds() -> None:
    belief, losses, pairwise_lipschitz, actionwise_lipschitz, _ = _approximate_problem()
    actionwise = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=actionwise_lipschitz,
        regret_tolerance=0.1,
        lipschitz_bound_certified=True,
    )
    pairwise = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=True,
    )

    assert actionwise.status == "undetermined"
    assert actionwise.minimax_upper_worst_case_regret > 3.7
    assert pairwise.status == "certified-admissible"
    assert pairwise.minimax_upper_worst_case_regret == 0.0
    assert (
        np.max(pairwise.pairwise_cover_correction_by_quotient_action)
        < 0.04
        * np.max(
            (
                actionwise.action_loss_lipschitz_by_quotient[:, :, None]
                + actionwise.action_loss_lipschitz_by_quotient[:, None, :]
            )
            * actionwise.cover_radius_by_quotient[:, None, None]
        )
    )


def test_pairwise_cover_requires_certified_regularities() -> None:
    belief, losses, pairwise_lipschitz, _, _ = _approximate_problem()
    certificate = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=False,
    )

    assert not certificate.bounds_certified
    assert certificate.status == "scope-not-certified"
    assert certificate.fallback_required


def test_pairwise_cover_requires_execution_coupling() -> None:
    belief, losses, pairwise_lipschitz, _, _ = _approximate_problem()
    certificate = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(valid=False),
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=True,
    )

    assert not certificate.bounds_certified
    assert certificate.status == "scope-not-certified"
    assert certificate.fallback_required


def test_zero_radius_finite_group_is_exact_without_lipschitz_receipt() -> None:
    angles = np.arange(4, dtype=np.float64) * (0.5 * math.pi)
    quadrature = CompactGroupQuadratureV1(
        group_id="approximate-action-s1",
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
        belief_id="finite-pairwise",
    )
    losses = np.array(
        [
            [
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [0.0, 4.0],
            ]
        ]
    )
    certificate = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=0.0,
        pairwise_lipschitz_bound_certified=False,
    )

    assert certificate.bounds_certified
    assert certificate.status == "certified-admissible"
    np.testing.assert_allclose(
        certificate.sampled_worst_case_regret,
        certificate.upper_worst_case_regret,
        atol=0.0,
    )
    np.testing.assert_array_equal(certificate.tolerance_admissible_action_mask, [True, False])


def test_pairwise_lipschitz_input_forms_and_diagonal_are_canonical() -> None:
    belief, losses, _, _, _ = _approximate_problem()
    scalar = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=0.2,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=True,
    )
    matrix = np.full((3, 3), 0.2)
    tensor = np.repeat(matrix[None, :, :], 2, axis=0)
    matrix_result = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=matrix,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=True,
    )
    tensor_result = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=tensor,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=True,
    )

    np.testing.assert_allclose(
        scalar.pairwise_difference_lipschitz_by_quotient_action,
        matrix_result.pairwise_difference_lipschitz_by_quotient_action,
    )
    np.testing.assert_allclose(
        matrix_result.pairwise_difference_lipschitz_by_quotient_action,
        tensor_result.pairwise_difference_lipschitz_by_quotient_action,
    )
    diagonal = np.arange(3)
    np.testing.assert_array_equal(
        scalar.pairwise_difference_lipschitz_by_quotient_action[:, diagonal, diagonal],
        0.0,
    )


def test_pairwise_certificate_arrays_are_immutable() -> None:
    belief, losses, pairwise_lipschitz, _, _ = _approximate_problem()
    certificate = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        pairwise_lipschitz_bound_certified=True,
    )

    with pytest.raises(ValueError, match="read-only"):
        certificate.upper_worst_case_regret[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        certificate.pairwise_cover_correction_by_quotient_action[0, 0, 1] = 1.0


def test_overridden_cover_is_uncertified_by_default() -> None:
    belief, losses, pairwise_lipschitz, _, _ = _approximate_problem()
    certificate = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=_receipt(),
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        cover_radius_by_quotient=0.1,
        pairwise_lipschitz_bound_certified=True,
    )

    assert not certificate.cover_radius_certified
    assert certificate.status == "scope-not-certified"
