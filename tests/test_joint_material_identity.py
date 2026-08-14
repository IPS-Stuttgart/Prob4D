import numpy as np
import pytest

from prob4d.joint_material_identity import (
    assignment_components,
    build_joint_material_identity_posterior,
    joint_candidate_marginals,
    marginalize_joint_assignment_log_likelihoods,
)
from prob4d.material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
)

RULE_ID = "a" * 64
CALIBRATION_ID = "b" * 64
TRACKLET_REVISION = "c" * 40
ASSOCIATION_REVISION = "d" * 40
RESULT_ID = "e" * 64


def mixture(
    *,
    target_window: str,
    target_track: int,
    window_order: tuple[str, ...],
    source_endpoint: LocalTrackEndpoint | None,
    linked_weight: float = 9.0,
    null_weight: float = 1.0,
    result_id: str = RESULT_ID,
) -> MaterialIdentityMixtureV1:
    candidates = [
        MaterialIdentityCandidateV1(
            source_endpoint=None,
            association_result_id=None,
            source_score=None,
            calibrated_log_weight=np.log(null_weight),
            metadata={"role": "exact-local-fallback"},
        )
    ]
    if source_endpoint is not None:
        candidates.append(
            MaterialIdentityCandidateV1(
                source_endpoint=source_endpoint,
                association_result_id=result_id,
                source_score=0.9,
                calibrated_log_weight=np.log(linked_weight),
                metadata={"source": "calibration-only"},
            )
        )
    return MaterialIdentityMixtureV1(
        target_endpoint=LocalTrackEndpoint(target_window, target_track),
        window_order=window_order,
        causal_frame_stop=20 + window_order.index(target_window),
        association_rule_id=RULE_ID,
        calibration_id=CALIBRATION_ID,
        tracklet_producer_revision=TRACKLET_REVISION,
        association_revision=ASSOCIATION_REVISION,
        candidates=tuple(candidates),
        metadata={"stage": "source-only"},
    )


def conflicting_pair() -> tuple[MaterialIdentityMixtureV1, ...]:
    source = LocalTrackEndpoint("window-0", 0)
    return (
        mixture(
            target_window="window-1",
            target_track=0,
            window_order=("window-0", "window-1"),
            source_endpoint=source,
        ),
        mixture(
            target_window="window-1",
            target_track=1,
            window_order=("window-0", "window-1"),
            source_endpoint=source,
        ),
    )


def test_global_constraint_removes_one_source_splitting_into_two_target_tracks() -> None:
    posterior = build_joint_material_identity_posterior(
        conflicting_pair(),
        window_order=("window-0", "window-1"),
    )

    assert posterior.unconstrained_assignment_count == 4
    assert posterior.feasible_assignment_count == 3
    assert posterior.rejected_assignment_count == 1
    assert posterior.constraint_rejection_fraction == pytest.approx(0.25)
    np.testing.assert_allclose(
        posterior.probabilities,
        [1.0 / 19.0, 9.0 / 19.0, 9.0 / 19.0],
    )
    for marginal in posterior.marginals:
        np.testing.assert_allclose(
            marginal.probabilities,
            [10.0 / 19.0, 9.0 / 19.0],
        )
        assert marginal.null_probability == pytest.approx(10.0 / 19.0)


def test_transitive_component_cannot_contain_two_endpoints_from_one_window() -> None:
    source = LocalTrackEndpoint("window-0", 0)
    middle = LocalTrackEndpoint("window-1", 0)
    mixtures = (
        mixture(
            target_window="window-1",
            target_track=0,
            window_order=("window-0", "window-1"),
            source_endpoint=source,
        ),
        mixture(
            target_window="window-2",
            target_track=0,
            window_order=("window-0", "window-1", "window-2"),
            source_endpoint=middle,
        ),
        mixture(
            target_window="window-2",
            target_track=1,
            window_order=("window-0", "window-1", "window-2"),
            source_endpoint=source,
            result_id="f" * 64,
        ),
    )

    posterior = build_joint_material_identity_posterior(
        mixtures,
        window_order=("window-0", "window-1", "window-2"),
    )

    assert posterior.unconstrained_assignment_count == 8
    assert posterior.feasible_assignment_count == 7
    all_linked = tuple(value.candidate_ids[1] for value in posterior.mixtures)
    assert all_linked not in {assignment.candidate_ids for assignment in posterior.assignments}


def test_input_order_is_canonical_and_content_identity_is_stable() -> None:
    mixtures = conflicting_pair()
    first = build_joint_material_identity_posterior(
        mixtures,
        window_order=("window-0", "window-1"),
        metadata={"experiment": "joint-source-prior"},
    )
    second = build_joint_material_identity_posterior(
        tuple(reversed(mixtures)),
        window_order=("window-0", "window-1"),
        metadata={"experiment": "joint-source-prior"},
    )

    assert first.posterior_id == second.posterior_id
    assert first.mixture_ids == second.mixture_ids
    assert first.assignment_ids == second.assignment_ids


def test_assignment_bound_fails_closed_before_combinatorial_growth() -> None:
    mixtures = tuple(
        mixture(
            target_window="window-1",
            target_track=index,
            window_order=("window-0", "window-1"),
            source_endpoint=LocalTrackEndpoint("window-0", index),
        )
        for index in range(8)
    )
    with pytest.raises(ValueError, match="exceeds maximum_joint_assignments"):
        build_joint_material_identity_posterior(
            mixtures,
            window_order=("window-0", "window-1"),
            maximum_joint_assignments=100,
        )


def test_joint_likelihood_marginalization_and_candidate_projection() -> None:
    posterior = build_joint_material_identity_posterior(
        conflicting_pair(),
        window_order=("window-0", "window-1"),
    )
    log_likelihoods = np.array([-5.0, -1.0, -3.0])
    result = marginalize_joint_assignment_log_likelihoods(
        posterior,
        posterior.assignment_ids,
        log_likelihoods,
    )

    expected_terms = posterior.probabilities * np.exp(
        log_likelihoods - np.max(log_likelihoods)
    )
    expected_probabilities = expected_terms / np.sum(expected_terms)
    np.testing.assert_allclose(
        result.posterior_probabilities,
        expected_probabilities,
    )
    marginals = joint_candidate_marginals(
        posterior,
        assignment_probabilities=result.posterior_probabilities,
    )
    assert len(marginals) == 2
    np.testing.assert_allclose(
        marginals[0].probabilities,
        [expected_probabilities[0] + expected_probabilities[1], expected_probabilities[2]],
    )

    prior_only = marginalize_joint_assignment_log_likelihoods(
        posterior,
        posterior.assignment_ids,
        np.array([-np.inf, -3.0, -np.inf]),
        likelihood_power=0.0,
    )
    assert prior_only.log_marginal_likelihood == pytest.approx(0.0)
    np.testing.assert_allclose(
        prior_only.posterior_probabilities,
        posterior.probabilities,
    )


def test_assignment_components_keep_local_endpoints_without_global_id_rewrite() -> None:
    posterior = build_joint_material_identity_posterior(
        conflicting_pair(),
        window_order=("window-0", "window-1"),
    )
    linked_assignment = next(
        assignment
        for assignment in posterior.assignments
        if assignment.candidate_ids[0] == posterior.mixtures[0].candidate_ids[1]
    )
    components = assignment_components(posterior, linked_assignment.assignment_id)

    assert any(
        component
        == (
            LocalTrackEndpoint("window-0", 0),
            LocalTrackEndpoint("window-1", 0),
        )
        for component in components
    )
    assert any(
        component == (LocalTrackEndpoint("window-1", 1),)
        for component in components
    )

