from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    SymmetryCompleteBeliefV1,
    certify_compact_group_decision,
)


def test_finite_group_decision_certificate_is_exact() -> None:
    quadrature = CompactGroupQuadratureV1(
        group_id="c2",
        metric_id="discrete",
        nodes=np.array([[0.0], [1.0]]),
        reference_weights=np.array([0.5, 0.5]),
        cover_radius=0.0,
        cover_radius_certified=True,
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.25, 0.75],
        quadrature,
        belief_id="posterior",
    )
    losses = np.array(
        [
            [[0.0, 1.0], [0.0, 2.0]],
            [[0.0, 0.5], [0.0, 3.0]],
        ]
    )
    certificate = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=0.0,
    )

    np.testing.assert_allclose(certificate.sampled_worst_case_regret, [0.0, 2.75])
    np.testing.assert_allclose(
        certificate.upper_worst_case_regret,
        certificate.sampled_worst_case_regret,
    )
    np.testing.assert_array_equal(
        certificate.tolerance_admissible_action_mask,
        [True, False],
    )
    assert certificate.status == "certified-admissible"
    assert certificate.minimax_upper_action_index == 0
    assert certificate.uniquely_tolerance_identified
    assert not certificate.fallback_required


def test_continuous_group_can_certify_no_uniformly_good_action() -> None:
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        32,
        group_id="s1",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    losses = np.stack((np.zeros_like(angle), np.cos(angle)), axis=1)[None, :, :]
    certificate = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=[0.0, 1.0],
        regret_tolerance=0.5,
    )

    np.testing.assert_allclose(certificate.sampled_worst_case_regret, [1.0, 1.0])
    assert certificate.status == "certified-no-admissible-action"
    assert certificate.fallback_required
    assert not certificate.has_tolerance_admissible_action


def test_continuous_group_certifies_bounded_regret_without_point_completion() -> None:
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        64,
        group_id="s1",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    # Action 0 is never worse than action 1. The finite cover makes the
    # continuous robust-regret claim with a small nonzero tolerance.
    losses = np.stack((np.zeros_like(angle), 1.0 + np.cos(angle)), axis=1)[None, :, :]
    certificate = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=[0.0, 1.0],
        regret_tolerance=math.pi / 64.0 + 1e-12,
    )

    assert certificate.status == "certified-admissible"
    assert certificate.tolerance_admissible_action_mask[0]
    assert not certificate.tolerance_admissible_action_mask[1]
    assert certificate.minimax_upper_action_index == 0
    assert certificate.minimax_upper_worst_case_regret == pytest.approx(math.pi / 64.0)


def test_coarse_cover_can_leave_decision_undetermined() -> None:
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        3,
        group_id="s1",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    losses = np.stack((np.zeros_like(angle), 1.0 + np.cos(angle)), axis=1)[None, :, :]
    certificate = certify_compact_group_decision(
        belief,
        losses,
        action_loss_lipschitz_by_quotient=[0.0, 1.0],
        regret_tolerance=0.9,
    )

    assert certificate.sampled_worst_case_regret[0] == pytest.approx(0.0)
    assert certificate.upper_worst_case_regret[0] > 0.9
    assert certificate.sampled_worst_case_regret[1] > 0.9
    assert certificate.status == "undetermined"
    assert certificate.fallback_required


def test_uncertified_decision_cover_fails_closed() -> None:
    quadrature = CompactGroupQuadratureV1(
        group_id="unknown",
        metric_id="unknown",
        nodes=np.array([[0.0], [1.0]]),
        reference_weights=np.array([0.5, 0.5]),
        cover_radius=0.25,
        cover_radius_certified=False,
        measure_kind="continuous-density",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    certificate = certify_compact_group_decision(
        belief,
        np.zeros((1, 2, 2)),
        action_loss_lipschitz_by_quotient=0.0,
        regret_tolerance=1.0,
    )

    assert certificate.status == "scope-not-certified"
    assert certificate.fallback_required
    assert not np.any(certificate.tolerance_admissible_action_mask)


def test_decision_certificate_arrays_are_immutable() -> None:
    quadrature = CompactGroupQuadratureV1(
        group_id="c2",
        metric_id="discrete",
        nodes=np.array([[0.0], [1.0]]),
        reference_weights=np.array([0.5, 0.5]),
        cover_radius=0.0,
        cover_radius_certified=True,
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0], quadrature, belief_id="posterior"
    )
    certificate = certify_compact_group_decision(
        belief,
        np.array([[[0.0, 1.0], [0.0, 2.0]]]),
        action_loss_lipschitz_by_quotient=0.0,
    )
    with pytest.raises(ValueError, match="read-only"):
        certificate.upper_worst_case_regret[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        certificate.tolerance_admissible_action_mask[0] = False
