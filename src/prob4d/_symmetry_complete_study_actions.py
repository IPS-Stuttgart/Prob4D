"""Gauge-coupled action panel for the symmetry-complete controlled study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .symmetry_complete_belief import (
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


def _problem(
    angles: np.ndarray,
    quadrature: CompactGroupQuadratureV1,
) -> tuple[SymmetryCompleteBeliefV1, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.75, 0.25],
        quadrature,
        belief_id="gauge-coupled-action-study",
    )
    base_states = np.array([[1.0, 0.0], [-1.0, 0.0]])
    action_templates = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    node_count = angles.size
    states = np.empty((2, node_count, 2))
    actions = np.empty((3, node_count, 2))
    for node_index, angle in enumerate(angles):
        rotation = _rotation(float(angle))
        states[:, node_index] = base_states @ rotation.T
        actions[:, node_index] = action_templates @ rotation.T
    common = 2.0 + 2.0 * np.cos(3.0 * angles)
    losses = np.empty((2, node_count, 3))
    action_by_node = np.transpose(actions, (1, 0, 2))
    for quotient_index in range(2):
        delta = states[quotient_index, :, None, :] - action_by_node
        losses[quotient_index] = np.sum(delta * delta, axis=2) + common[:, None]
    return belief, losses, states, actions, common


def _receipt(*, valid: bool) -> GaugeCouplingReceiptV1:
    return GaugeCouplingReceiptV1(
        group_id="controlled-action-s1",
        coupling_id="controlled-shared-object-frame-v1",
        state_orbit_id="controlled-state-orbit-v1",
        action_orbit_id="controlled-action-orbit-v1",
        shared_group_element_certified=valid,
        execution_binding_certified=valid,
    )


def _equivariant_action_coupling_study() -> dict[str, Any]:
    node_count = 64
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        node_count,
        group_id="controlled-action-s1",
    )
    angles = quadrature.nodes[:, 0]
    belief, coupled_losses, states, actions, common = _problem(angles, quadrature)
    certificate = certify_gauge_coupled_action_orbit(
        belief,
        coupled_losses,
        coupling_receipt=_receipt(valid=True),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )
    missing_receipt = certify_gauge_coupled_action_orbit(
        belief,
        coupled_losses,
        coupling_receipt=_receipt(valid=False),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )

    coarse_count = 8
    coarse = CompactGroupQuadratureV1.uniform_circle(
        coarse_count,
        group_id="controlled-action-s1",
    )
    coarse_belief, coarse_losses, _, _, _ = _problem(coarse.nodes[:, 0], coarse)
    actionwise = certify_compact_group_decision(
        coarse_belief,
        coarse_losses,
        action_loss_lipschitz_by_quotient=6.0,
        regret_tolerance=0.1,
        lipschitz_bound_certified=True,
    )
    pairwise = certify_gauge_coupled_action_orbit(
        coarse_belief,
        coarse_losses,
        coupling_receipt=_receipt(valid=True),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=0.1,
    )

    fixed_templates = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    fixed_losses = np.empty_like(coupled_losses)
    for quotient_index in range(2):
        delta = states[quotient_index, :, None, :] - fixed_templates[None, :, :]
        fixed_losses[quotient_index] = np.sum(delta * delta, axis=2) + common[:, None]
    fixed_certificate = certify_gauge_coupled_action_orbit(
        belief,
        fixed_losses,
        coupling_receipt=_receipt(valid=True),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )

    group_weights = quadrature.reference_weights
    quotient_weights = belief.quotient_weights
    shared_expected = np.empty(3)
    independent_expected = np.empty(3)
    fixed_expected = np.empty(3)
    for action_index in range(3):
        shared_by_class = np.empty(2)
        independent_by_class = np.empty(2)
        fixed_by_class = np.empty(2)
        for quotient_index in range(2):
            shared_squared = np.sum(
                np.square(states[quotient_index] - actions[action_index]),
                axis=1,
            )
            shared_by_class[quotient_index] = group_weights @ (shared_squared + common)

            independent_delta = (
                states[quotient_index, :, None, :] - actions[action_index, None, :, :]
            )
            independent_loss = (
                np.sum(independent_delta * independent_delta, axis=2) + common[:, None]
            )
            independent_by_class[quotient_index] = group_weights @ independent_loss @ group_weights

            fixed_squared = np.sum(
                np.square(states[quotient_index] - fixed_templates[action_index]),
                axis=1,
            )
            fixed_by_class[quotient_index] = group_weights @ (fixed_squared + common)
        shared_expected[action_index] = quotient_weights @ shared_by_class
        independent_expected[action_index] = quotient_weights @ independent_by_class
        fixed_expected[action_index] = quotient_weights @ fixed_by_class

    offset = 0.371
    offset_angles = angles + offset
    offset_quadrature = CompactGroupQuadratureV1(
        group_id="controlled-action-s1",
        metric_id=quadrature.metric_id,
        nodes=offset_angles[:, None],
        reference_weights=quadrature.reference_weights,
        cover_radius=quadrature.cover_radius,
        cover_radius_certified=True,
        measure_kind="continuous-density",
    )
    offset_belief, offset_losses, _, _, _ = _problem(
        offset_angles,
        offset_quadrature,
    )
    offset_certificate = certify_gauge_coupled_action_orbit(
        offset_belief,
        offset_losses,
        coupling_receipt=_receipt(valid=True),
        whole_group_pairwise_difference_invariance_certified=True,
        difference_invariance_atol=1e-10,
        regret_tolerance=1e-10,
    )

    return {
        "node_count": node_count,
        "quotient_weights": quotient_weights.tolist(),
        "coupled_action_status": certificate.status,
        "coupled_selected_action_template": certificate.minimax_upper_action_index,
        "coupled_upper_regret": certificate.upper_worst_case_regret.tolist(),
        "coupled_admissible_action_count": int(
            np.count_nonzero(certificate.tolerance_admissible_action_mask)
        ),
        "maximum_absolute_loss_range": certificate.maximum_sample_absolute_loss_range,
        "maximum_pairwise_difference_range": (certificate.maximum_sample_pairwise_difference_range),
        "missing_coupling_receipt_status": missing_receipt.status,
        "actionwise_lipschitz_status": actionwise.status,
        "actionwise_lipschitz_minimum_upper_regret": (actionwise.minimax_upper_worst_case_regret),
        "pairwise_equivariance_status": pairwise.status,
        "pairwise_equivariance_minimum_upper_regret": (pairwise.minimax_upper_worst_case_regret),
        "fixed_frame_status": fixed_certificate.status,
        "fixed_frame_pairwise_difference_range": (
            fixed_certificate.maximum_sample_pairwise_difference_range
        ),
        "shared_gauge_expected_loss": shared_expected.tolist(),
        "independent_gauge_expected_loss": independent_expected.tolist(),
        "fixed_frame_expected_loss": fixed_expected.tolist(),
        "shared_gauge_optimal_action_count": int(
            np.count_nonzero(np.isclose(shared_expected, np.min(shared_expected)))
        ),
        "independent_gauge_optimal_action_count": int(
            np.count_nonzero(np.isclose(independent_expected, np.min(independent_expected)))
        ),
        "fixed_frame_optimal_action_count": int(
            np.count_nonzero(np.isclose(fixed_expected, np.min(fixed_expected)))
        ),
        "maximum_regret_change_under_group_coordinate_offset": float(
            np.max(
                np.abs(
                    certificate.upper_worst_case_regret - offset_certificate.upper_worst_case_regret
                )
            )
        ),
        "claim_boundary": (
            "Controlled coupled state/action geometry only. The receipt is supplied, "
            "not inferred, and no physical command is executed."
        ),
    }


__all__ = ["_equivariant_action_coupling_study"]
