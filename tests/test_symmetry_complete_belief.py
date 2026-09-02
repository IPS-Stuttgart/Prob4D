from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    SymmetryCompleteBeliefV1,
    audit_point_completion,
    certify_compact_group_query,
    pushforward_shared_group_query,
    update_symmetry_complete_belief,
)


def circle(count: int = 32) -> CompactGroupQuadratureV1:
    return CompactGroupQuadratureV1.uniform_circle(
        count,
        group_id="shared-axial-so2-v1",
    )


def test_orbit_invariant_likelihood_preserves_conditional_law_exactly() -> None:
    quadrature = circle(16)
    skew = np.linspace(1.0, 2.0, quadrature.node_count)
    skew /= np.sum(skew)
    prior = SymmetryCompleteBeliefV1(
        quotient_weights=np.array([0.2, 0.3, 0.5]),
        group_conditional_weights=np.array(
            [
                quadrature.reference_weights,
                np.roll(quadrature.reference_weights, 1),
                skew,
            ]
        ),
        quadrature=quadrature,
        belief_id="prior",
    )
    likelihood = np.repeat(
        np.array([[0.5], [2.0], [4.0]]),
        quadrature.node_count,
        axis=1,
    )

    result = update_symmetry_complete_belief(
        prior,
        likelihood,
        evidence_semantics="orbit-invariant",
        posterior_belief_id="posterior",
        whole_group_invariance_certified=True,
    )

    np.testing.assert_array_equal(
        result.posterior.group_conditional_weights,
        result.prior.group_conditional_weights,
    )
    assert result.posterior.group_conditional_weights is result.prior.group_conditional_weights
    np.testing.assert_allclose(
        result.posterior.quotient_weights,
        [1.0 / 27.0, 2.0 / 9.0, 20.0 / 27.0],
    )
    assert result.information.gauge_information_nats == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert result.information.maximum_conditional_l1_change == 0.0
    assert result.information.total_information_nats == pytest.approx(
        result.information.quotient_information_nats,
        abs=1e-14,
    )
    assert result.information.quadrature_invariance_verified
    assert result.information.whole_group_invariance_certified


def test_orbit_invariant_mode_rejects_hidden_symmetry_breaking() -> None:
    quadrature = circle(8)
    prior = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="prior",
    )
    likelihood = np.ones((1, quadrature.node_count))
    likelihood[0, 3] = 1.01
    with pytest.raises(ValueError, match="varies over prior-supported group nodes"):
        update_symmetry_complete_belief(
            prior,
            likelihood,
            evidence_semantics="orbit-invariant",
            posterior_belief_id="bad",
            whole_group_invariance_certified=True,
        )


def test_continuous_invariant_update_requires_complete_group_certificate() -> None:
    quadrature = circle(8)
    prior = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="prior",
    )
    with pytest.raises(ValueError, match="quadrature equality alone"):
        update_symmetry_complete_belief(
            prior,
            np.ones((1, quadrature.node_count)),
            evidence_semantics="orbit-invariant",
            posterior_belief_id="bad",
        )


def test_finite_group_invariant_update_is_exhaustive_without_external_flag() -> None:
    quadrature = CompactGroupQuadratureV1(
        group_id="c2",
        metric_id="discrete",
        nodes=np.array([[0.0], [1.0]]),
        reference_weights=np.array([0.5, 0.5]),
        cover_radius=0.0,
        cover_radius_certified=True,
    )
    prior = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.25, 0.75],
        quadrature,
        belief_id="prior",
    )
    result = update_symmetry_complete_belief(
        prior,
        np.array([[1.0, 1.0], [2.0, 2.0]]),
        evidence_semantics="orbit-invariant",
        posterior_belief_id="posterior",
    )
    assert result.information.quadrature_invariance_verified
    assert result.information.whole_group_invariance_certified
    assert result.information.gauge_information_nats == 0.0


def test_explicit_symmetry_breaking_updates_gauge_and_chain_rule() -> None:
    quadrature = circle(64)
    prior = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.4, 0.6],
        quadrature,
        belief_id="prior",
    )
    angle = quadrature.nodes[:, 0]
    likelihood = np.stack(
        (
            np.exp(2.0 * np.cos(angle)),
            0.5 * np.exp(-1.5 * np.sin(angle)),
        ),
        axis=0,
    )
    result = update_symmetry_complete_belief(
        prior,
        likelihood,
        evidence_semantics="symmetry-breaking",
        posterior_belief_id="posterior",
    )

    assert result.information.gauge_information_nats > 0.1
    assert result.information.maximum_conditional_l1_change > 0.1
    assert not result.information.quadrature_invariance_verified
    assert not result.information.whole_group_invariance_certified
    assert result.information.total_information_nats == pytest.approx(
        result.information.quotient_information_nats + result.information.gauge_information_nats,
        abs=1e-12,
    )
    assert np.argmax(result.posterior.group_conditional_weights[0]) == quadrature.node_count // 2


def test_uniform_point_completion_adds_log_node_count_specificity() -> None:
    quadrature = circle(32)
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [0.25, 0.75],
        quadrature,
        belief_id="posterior",
    )
    audit = audit_point_completion(belief, [0, 17])
    assert audit.supported_by_quadrature
    assert not audit.physical_point_completion_has_finite_kl
    assert audit.status == "continuous-singular"
    assert audit.discretized_specificity_nats == pytest.approx(math.log(32.0))


def test_finite_group_point_completion_has_finite_kl() -> None:
    quadrature = CompactGroupQuadratureV1(
        group_id="c4",
        metric_id="discrete",
        nodes=np.arange(4, dtype=np.float64)[:, None],
        reference_weights=np.full(4, 0.25),
        cover_radius=0.0,
        cover_radius_certified=True,
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    audit = audit_point_completion(belief, [2])
    assert audit.status == "finite-supported"
    assert audit.physical_point_completion_has_finite_kl
    assert audit.discretized_specificity_nats == pytest.approx(math.log(4.0))


def test_point_completion_outside_prior_support_is_not_given_finite_penalty() -> None:
    quadrature = circle(4)
    belief = SymmetryCompleteBeliefV1(
        [1.0],
        [[0.5, 0.5, 0.0, 0.0]],
        quadrature,
        "posterior",
    )
    audit = audit_point_completion(belief, [3])
    assert not audit.supported_by_quadrature
    assert audit.status == "outside-support"
    assert audit.discretized_specificity_nats is None


def test_shared_group_pushforward_preserves_joint_cancellation() -> None:
    quadrature = circle(128)
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    first = np.column_stack((np.cos(angle), np.sin(angle)))
    second = -first
    atoms = np.concatenate((first, second), axis=1)[None, :, :]
    law = pushforward_shared_group_query(belief, atoms)

    np.testing.assert_allclose(law.mean, np.zeros(4), atol=1e-15)
    difference = law.atoms[:, :2] - law.atoms[:, 2:]
    np.testing.assert_allclose(np.linalg.norm(difference, axis=1), 2.0, atol=1e-14)
    # Shared gauge creates exact anticorrelation; independent pointwise gauges do not.
    assert law.covariance[0, 2] == pytest.approx(-0.5, abs=1e-14)
    assert law.covariance[1, 3] == pytest.approx(-0.5, abs=1e-14)


def test_compact_group_query_certificate_covers_continuous_circle() -> None:
    quadrature = circle(16)
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    atoms = np.column_stack((np.cos(angle), np.sin(angle)))[None, :, :]
    certificate = certify_compact_group_query(
        belief,
        atoms,
        query_id="unit-circle-position",
        lipschitz_by_quotient=1.0,
        tolerance=1.5,
        lipschitz_bound_certified=True,
    )

    exact_diameter = 2.0
    assert certificate.maximum_sample_diameter <= exact_diameter + 1e-14
    assert certificate.maximum_upper_diameter >= exact_diameter
    assert certificate.status == "certified-variant"
    assert certificate.bounds_certified
    assert certificate.cover_radius_certified
    assert certificate.lipschitz_bound_certified
    assert not certificate.admitted


def test_invariant_query_is_admitted_and_local_stationarity_is_not_enough() -> None:
    quadrature = circle(32)
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    invariant = np.full((1, quadrature.node_count, 1), 3.0)
    invariant_certificate = certify_compact_group_query(
        belief,
        invariant,
        query_id="axis-coordinate",
        lipschitz_by_quotient=0.0,
        tolerance=0.0,
    )
    assert invariant_certificate.admitted
    assert invariant_certificate.status == "certified-invariant"

    deceptive = np.cos(angle)[None, :, None]
    # dq/dtheta at theta=0 is zero, but the continuous orbit diameter is two.
    assert -math.sin(0.0) == 0.0
    deceptive_certificate = certify_compact_group_query(
        belief,
        deceptive,
        query_id="representative-x-coordinate",
        lipschitz_by_quotient=1.0,
        tolerance=0.1,
        lipschitz_bound_certified=True,
    )
    assert deceptive_certificate.status == "certified-variant"
    assert not deceptive_certificate.admitted


def test_custom_query_cover_requires_explicit_certificate() -> None:
    quadrature = circle(8)
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    angle = quadrature.nodes[:, 0]
    atoms = np.column_stack((np.cos(angle), np.sin(angle)))[None, :, :]

    uncertified = certify_compact_group_query(
        belief,
        atoms,
        query_id="custom-cover",
        lipschitz_by_quotient=1.0,
        cover_radius_by_quotient=0.01,
        tolerance=3.0,
        lipschitz_bound_certified=True,
    )
    assert uncertified.status == "scope-not-certified"
    assert not uncertified.cover_radius_certified
    assert not uncertified.admitted

    certified = certify_compact_group_query(
        belief,
        atoms,
        query_id="custom-cover",
        lipschitz_by_quotient=1.0,
        cover_radius_by_quotient=0.01,
        tolerance=3.0,
        cover_radius_certified=True,
        lipschitz_bound_certified=True,
    )
    assert certified.status == "certified-invariant"
    assert certified.bounds_certified
    assert certified.admitted


def test_uncertified_cover_fails_closed() -> None:
    quadrature = CompactGroupQuadratureV1(
        group_id="unknown-compact-group",
        metric_id="caller-metric",
        nodes=np.array([[0.0], [1.0]]),
        reference_weights=np.array([0.5, 0.5]),
        cover_radius=0.25,
        cover_radius_certified=False,
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    certificate = certify_compact_group_query(
        belief,
        np.zeros((1, 2, 1)),
        query_id="constant-on-samples-only",
        lipschitz_by_quotient=1.0,
        tolerance=1.0,
        lipschitz_bound_certified=True,
    )
    assert certificate.status == "scope-not-certified"
    assert not certificate.bounds_certified
    assert not certificate.admitted


def test_outputs_are_irreversibly_immutable() -> None:
    quadrature = circle(8)
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="posterior",
    )
    with pytest.raises(ValueError, match="read-only"):
        belief.quotient_weights[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        belief.group_conditional_weights[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        belief.quadrature.nodes[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        belief.quadrature.reference_weights[0] = 1.0


@pytest.mark.parametrize(
    ("quotient", "conditional", "match"),
    [
        ([0.4, 0.5], [[0.5, 0.5], [0.5, 0.5]], "sum to one"),
        ([1.0], [[0.4, 0.4]], "row"),
        ([1.0], [[0.5, -0.5]], "nonnegative"),
    ],
)
def test_invalid_beliefs_fail_closed(
    quotient: object,
    conditional: object,
    match: str,
) -> None:
    quadrature = CompactGroupQuadratureV1(
        "c2",
        "discrete",
        [[0.0], [1.0]],
        [0.5, 0.5],
        0.0,
        True,
    )
    with pytest.raises(ValueError, match=match):
        SymmetryCompleteBeliefV1(quotient, conditional, quadrature, "bad")
