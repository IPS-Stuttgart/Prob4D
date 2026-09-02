from __future__ import annotations

import numpy as np
import pytest

from prob4d.equivariant_decision import (
    STATUS_BOUNDED_REGRET,
    STATUS_EXACT_OPTIMAL,
    STATUS_FALLBACK,
)
from prob4d.realized_equivariant_decision import (
    certify_realized_gauge_coupled_actions,
    intervention_pairwise_realization_margin,
)


def _invariant_losses() -> np.ndarray:
    losses = np.empty((1, 8, 3), dtype=np.float64)
    losses[0, :, :] = np.array([0.0, 2.0, 1.0])
    return losses


def test_zero_realization_error_recovers_ideal_certificate() -> None:
    certificate = certify_realized_gauge_coupled_actions(
        _invariant_losses(),
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        realization_radius=0.0,
        action_loss_lipschitz=2.0,
        fallback_action=2,
    )

    assert np.array_equal(
        certificate.pairwise_realized_upper_bound,
        certificate.ideal_certificate.pairwise_upper_bound,
    )
    assert certificate.robustly_optimal_under_realization.tolist() == [
        True,
        False,
        False,
    ]
    assert certificate.selected_action == 0
    assert certificate.status == STATUS_EXACT_OPTIMAL


def test_small_realization_error_preserves_exact_action_optimality() -> None:
    certificate = certify_realized_gauge_coupled_actions(
        _invariant_losses(),
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        realization_radius=[0.2, 0.2, 0.0],
        action_loss_lipschitz=[1.0, 1.0, 1.0],
        fallback_action=2,
    )

    assert certificate.posterior_pairwise_realization_margin[0, 2] == pytest.approx(
        0.2
    )
    assert certificate.pairwise_realized_upper_bound[0, 2] == pytest.approx(-0.8)
    assert certificate.robustly_optimal_under_realization[0]
    assert certificate.selected_action == 0
    assert certificate.status == STATUS_EXACT_OPTIMAL


def test_large_realization_error_forces_exact_fallback() -> None:
    certificate = certify_realized_gauge_coupled_actions(
        _invariant_losses(),
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        realization_radius=[1.2, 0.0, 0.0],
        action_loss_lipschitz=1.0,
        fallback_action=2,
        regret_tolerance=0.0,
    )

    assert not certificate.robustly_optimal_under_realization[0]
    assert certificate.worst_case_realized_regret_upper_bound[0] == pytest.approx(
        0.2
    )
    assert not certificate.admitted
    assert certificate.selected_action == 2
    assert certificate.status == STATUS_FALLBACK


def test_positive_tolerance_admits_bounded_regret_action() -> None:
    certificate = certify_realized_gauge_coupled_actions(
        _invariant_losses(),
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        realization_radius=[1.2, 0.0, 0.0],
        action_loss_lipschitz=1.0,
        fallback_action=2,
        regret_tolerance=0.25,
    )

    assert not certificate.robustly_optimal_under_realization[0]
    assert certificate.epsilon_admissible_under_realization[0]
    assert certificate.admitted
    assert certificate.selected_action == 0
    assert certificate.status == STATUS_BOUNDED_REGRET


def test_class_weighting_and_pairwise_margin_are_exact() -> None:
    one_action, pairwise = intervention_pairwise_realization_margin(
        [[0.1, 0.2], [0.3, 0.4]],
        [[2.0, 3.0], [4.0, 5.0]],
        class_count=2,
        action_count=2,
    )
    assert one_action == pytest.approx([[0.2, 0.6], [1.2, 2.0]])
    assert pairwise[:, 0, 1] == pytest.approx([0.8, 3.2])
    assert np.array_equal(np.diagonal(pairwise, axis1=1, axis2=2), np.zeros((2, 2)))


def test_invalid_realization_contracts_fail_closed() -> None:
    common = dict(
        loss_samples=_invariant_losses(),
        quotient_mass=[1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        fallback_action=2,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        certify_realized_gauge_coupled_actions(
            **common,
            realization_radius=-0.1,
            action_loss_lipschitz=1.0,
        )
    with pytest.raises(ValueError, match="must be a scalar"):
        certify_realized_gauge_coupled_actions(
            **common,
            realization_radius=np.zeros((2, 2)),
            action_loss_lipschitz=1.0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        certify_realized_gauge_coupled_actions(
            **common,
            realization_radius=0.1,
            action_loss_lipschitz=[1.0, -1.0, 1.0],
        )


def test_realized_certificate_outputs_are_immutable() -> None:
    certificate = certify_realized_gauge_coupled_actions(
        _invariant_losses(),
        [1.0],
        cover_radius=0.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        realization_radius=0.1,
        action_loss_lipschitz=1.0,
        fallback_action=2,
    )
    with pytest.raises(ValueError):
        certificate.realization_radius[0, 0] = 0.0
    with pytest.raises(ValueError):
        certificate.pairwise_realized_upper_bound[0, 1] = 0.0
