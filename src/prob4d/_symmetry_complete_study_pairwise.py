"""Approximate pairwise-equivariance panel for the controlled study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .symmetry_complete_belief import (
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


def _approximate_pairwise_action_study() -> dict[str, Any]:
    node_count = 8
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
        belief_id="approximate-pairwise-action-study",
    )
    receipt = GaugeCouplingReceiptV1(
        group_id="approximate-action-s1",
        coupling_id="approximate-shared-object-frame-v1",
        state_orbit_id="approximate-state-orbit-v1",
        action_orbit_id="approximate-action-orbit-v1",
        shared_group_element_certified=True,
        execution_binding_certified=True,
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
    pairwise = certify_gauge_coupled_pairwise_decision(
        belief,
        losses,
        coupling_receipt=receipt,
        pairwise_difference_lipschitz_by_quotient_action=pairwise_lipschitz,
        regret_tolerance=0.1,
        pairwise_lipschitz_bound_certified=True,
    )
    actionwise = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=actionwise_lipschitz,
        regret_tolerance=0.1,
        lipschitz_bound_certified=True,
    )

    dense_angles = np.linspace(0.0, 2.0 * math.pi, 131_072, endpoint=False)
    class_supremum = np.full((2, 3, 3), -np.inf)
    for angle in dense_angles:
        rotation = _rotation(float(angle))
        states = base_states @ rotation.T
        actions = action_templates @ rotation.T
        common = 2.0 + 2.0 * math.cos(3.0 * float(angle))
        dense_losses = np.empty((2, 3))
        for quotient_index in range(2):
            dense_losses[quotient_index] = (
                np.sum(np.square(states[quotient_index] - actions), axis=1)
                + common
                + action_perturbation * math.cos(float(angle))
            )
        dense_difference = dense_losses[:, :, None] - dense_losses[:, None, :]
        class_supremum = np.maximum(class_supremum, dense_difference)
    dense_pairwise = np.tensordot(
        belief.quotient_weights,
        class_supremum,
        axes=(0, 0),
    )
    np.fill_diagonal(dense_pairwise, 0.0)
    dense_regret = np.maximum(np.max(dense_pairwise, axis=1), 0.0)

    actionwise_correction = (
        (
            actionwise.action_loss_lipschitz_by_quotient[:, :, None]
            + actionwise.action_loss_lipschitz_by_quotient[:, None, :]
        )
        * actionwise.cover_radius_by_quotient[:, None, None]
    )
    diagonal = np.arange(3)
    actionwise_correction[:, diagonal, diagonal] = 0.0
    return {
        "node_count": node_count,
        "cover_radius": cover_radius,
        "pairwise_status": pairwise.status,
        "pairwise_selected_action": pairwise.minimax_upper_action_index,
        "pairwise_sampled_regret": pairwise.sampled_worst_case_regret.tolist(),
        "pairwise_upper_regret": pairwise.upper_worst_case_regret.tolist(),
        "dense_reference_regret": dense_regret.tolist(),
        "minimum_dense_minus_sampled_regret": float(
            np.min(dense_regret - pairwise.sampled_worst_case_regret)
        ),
        "minimum_upper_minus_dense_regret": float(
            np.min(pairwise.upper_worst_case_regret - dense_regret)
        ),
        "maximum_sampled_pairwise_difference_range": float(
            np.max(pairwise.sampled_difference_range_by_quotient_action)
        ),
        "maximum_pairwise_cover_correction": (
            pairwise.maximum_pairwise_cover_correction
        ),
        "maximum_actionwise_cover_correction": float(
            np.max(actionwise_correction)
        ),
        "pairwise_to_actionwise_correction_ratio": float(
            pairwise.maximum_pairwise_cover_correction
            / np.max(actionwise_correction)
        ),
        "actionwise_status": actionwise.status,
        "actionwise_minimum_upper_regret": (
            actionwise.minimax_upper_worst_case_regret
        ),
        "claim_boundary": (
            "Controlled approximate SO(2) regularity with supplied valid pairwise "
            "Lipschitz bounds and execution coupling; no learned or physical receipt."
        ),
    }


__all__ = ["_approximate_pairwise_action_study"]
