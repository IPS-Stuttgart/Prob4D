from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.equivariant_decision import (
    STATUS_EXACT_OPTIMAL,
    STATUS_FALLBACK,
    certify_gauge_coupled_actions,
    certify_independent_gauge_control,
    minimum_uniform_so2_samples,
    so2_covering_radius,
    squared_metric_independent_gauge_losses,
    squared_metric_shared_gauge_losses,
)


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )


def _equivariant_orbits(sample_count: int = 16) -> tuple[np.ndarray, np.ndarray]:
    angles = 2.0 * math.pi * (np.arange(sample_count) + 0.5) / sample_count
    state = np.stack([_rotation(angle) @ np.array([1.0, 0.0]) for angle in angles])
    actions = np.stack(
        [
            np.stack(
                (
                    _rotation(angle) @ np.array([1.0, 0.0]),
                    _rotation(angle) @ np.array([0.0, 1.0]),
                    np.zeros(2),
                )
            )
            for angle in angles
        ]
    )
    return state[None, :, :], actions[None, :, :, :]


def test_equivariant_action_is_identified_without_state_representative() -> None:
    state, actions = _equivariant_orbits()
    shared_losses = squared_metric_shared_gauge_losses(state, actions)
    certificate = certify_gauge_coupled_actions(
        shared_losses,
        [1.0],
        cover_radius=so2_covering_radius(
            2.0 * math.pi * (np.arange(16) + 0.5) / 16
        ),
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        fallback_action=2,
        regret_tolerance=0.0,
    )

    assert np.ptp(state[0, :, 0]) > 1.9
    assert np.allclose(shared_losses[0, :, 0], 0.0)
    assert np.allclose(shared_losses[0, :, 1], 2.0)
    assert np.allclose(shared_losses[0, :, 2], 1.0)
    assert certificate.exactly_decision_equivariant
    assert certificate.posterior_gauge_irrelevant
    assert certificate.robustly_optimal.tolist() == [True, False, False]
    assert certificate.minimax_action == 0
    assert certificate.selected_action == 0
    assert certificate.admitted
    assert certificate.status == STATUS_EXACT_OPTIMAL


def test_independent_gauge_control_destroys_the_shared_cancellation() -> None:
    state, actions = _equivariant_orbits(32)
    independent_losses = squared_metric_independent_gauge_losses(state, actions)
    control = certify_independent_gauge_control(
        independent_losses,
        [1.0],
        fallback_action=2,
        regret_tolerance=0.0,
    )

    assert control.worst_case_regret[0] > 2.9
    assert control.minimax_action == 2
    assert not control.admitted
    assert control.selected_action == 2


def test_action_independent_loss_offsets_preserve_decisions() -> None:
    base = np.array([0.0, 1.0, 2.0])
    offsets = np.array([-4.0, 0.5, 9.0, -1.5])
    losses = base[None, None, :] + offsets[None, :, None]
    certificate = certify_gauge_coupled_actions(
        losses,
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        fallback_action=2,
    )

    assert certificate.exactly_decision_equivariant
    assert certificate.posterior_gauge_irrelevant
    assert certificate.robustly_optimal.tolist() == [True, False, False]
    assert certificate.worst_case_regret_upper_bound[0] == pytest.approx(0.0)


def test_cover_margin_prevents_false_continuous_optimality() -> None:
    sample_count = 4
    angles = 2.0 * math.pi * (np.arange(sample_count) + 0.5) / sample_count
    losses = np.stack((1.0 + np.cos(angles), np.full(sample_count, 1.8)), axis=1)
    sampled_only = certify_gauge_coupled_actions(
        losses[None, :, :],
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 2, 2)),
        fallback_action=1,
    )
    pairwise_lipschitz = np.zeros((1, 2, 2))
    pairwise_lipschitz[0, 0, 1] = 1.0
    pairwise_lipschitz[0, 1, 0] = 1.0
    covered = certify_gauge_coupled_actions(
        losses[None, :, :],
        [1.0],
        cover_radius=so2_covering_radius(angles),
        pairwise_lipschitz=pairwise_lipschitz,
        fallback_action=1,
    )

    dense = np.linspace(0.0, 2.0 * math.pi, 100_001)
    true_gap = float(np.max((1.0 + np.cos(dense)) - 1.8))
    assert sampled_only.robustly_optimal[0]
    assert true_gap > 0.19
    assert covered.pairwise_upper_bound[0, 1] >= true_gap
    assert not covered.robustly_optimal[0]
    assert not covered.admitted
    assert covered.selected_action == 1
    assert covered.status == STATUS_FALLBACK


def test_zero_cover_radius_recovers_exact_finite_group_regret() -> None:
    rng = np.random.default_rng(8)
    losses = rng.normal(size=(2, 5, 4))
    mass = np.array([0.3, 0.7])
    certificate = certify_gauge_coupled_actions(
        losses,
        mass,
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((2, 4, 4)),
        fallback_action=3,
        regret_tolerance=100.0,
    )

    expected_pairwise = np.zeros((4, 4))
    for class_index in range(2):
        differences = (
            losses[class_index, :, :, None]
            - losses[class_index, :, None, :]
        )
        expected_pairwise += mass[class_index] * np.max(differences, axis=0)
    np.fill_diagonal(expected_pairwise, 0.0)

    assert certificate.exact_finite_group
    assert np.allclose(certificate.pairwise_upper_bound, expected_pairwise)
    assert np.array_equal(
        certificate.pairwise_upper_bound,
        certificate.pairwise_sampled_lower_bound,
    )
    assert np.allclose(
        certificate.worst_case_regret_upper_bound,
        np.max(expected_pairwise, axis=1),
    )


def test_posterior_specific_gauge_irrelevance_ignores_zero_mass_class() -> None:
    losses = np.array(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [3.0, 0.0]],
        ]
    )
    certificate = certify_gauge_coupled_actions(
        losses,
        [1.0, 0.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((2, 2, 2)),
        fallback_action=1,
    )

    assert not certificate.exactly_decision_equivariant
    assert certificate.posterior_gauge_irrelevant
    assert certificate.selected_action == 0


def test_so2_covering_radius_and_sample_requirement() -> None:
    for sample_count in (1, 2, 5, 32):
        phase = 0.37
        angles = phase + 2.0 * math.pi * np.arange(sample_count) / sample_count
        assert so2_covering_radius(angles) == pytest.approx(
            math.pi / sample_count
        )
        assert so2_covering_radius(angles[::-1]) == pytest.approx(
            math.pi / sample_count
        )
    assert minimum_uniform_so2_samples(2.0, 0.1) == 63
    assert minimum_uniform_so2_samples(0.0, 0.0) == 1
    with pytest.raises(ValueError, match="positive Lipschitz"):
        minimum_uniform_so2_samples(1.0, 0.0)


def test_metric_helpers_and_certificate_outputs_are_immutable() -> None:
    state, actions = _equivariant_orbits(8)
    metric = np.diag([2.0, 0.5])
    losses = squared_metric_shared_gauge_losses(state, actions, metric=metric)
    certificate = certify_gauge_coupled_actions(
        losses,
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        fallback_action=2,
        regret_tolerance=10.0,
    )

    with pytest.raises(ValueError):
        losses[0, 0, 0] = 4.0
    with pytest.raises(ValueError):
        certificate.pairwise_upper_bound[0, 0] = 1.0
    with pytest.raises(ValueError):
        certificate.robustly_optimal[0] = False


def test_invalid_contracts_fail_closed() -> None:
    losses = np.zeros((1, 2, 2))
    valid_lipschitz = np.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="sum to one"):
        certify_gauge_coupled_actions(
            losses,
            [0.5],
            cover_radius=0.0,
            pairwise_lipschitz=valid_lipschitz,
            fallback_action=0,
        )
    with pytest.raises(ValueError, match="symmetric"):
        certify_gauge_coupled_actions(
            losses,
            [1.0],
            cover_radius=0.0,
            pairwise_lipschitz=np.array([[[0.0, 1.0], [0.0, 0.0]]]),
            fallback_action=0,
        )
    with pytest.raises(ValueError, match="same-action"):
        certify_gauge_coupled_actions(
            losses,
            [1.0],
            cover_radius=0.0,
            pairwise_lipschitz=np.ones((1, 2, 2)),
            fallback_action=0,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        squared_metric_shared_gauge_losses(
            np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2, 2)),
            metric=np.diag([1.0, -1.0]),
        )
    with pytest.raises(ValueError, match="at least one"):
        so2_covering_radius([])
