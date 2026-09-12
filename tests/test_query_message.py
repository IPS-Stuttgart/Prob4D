from __future__ import annotations

import numpy as np
import pytest

from prob4d.query_message import (
    GaussianQueryMessage,
    apply_gaussian_query_message,
    compress_gaussian_query_posterior,
    fuse_gaussian_query_messages_covariance_intersection,
    select_pairwise_covariance_intersection,
)
from prob4d.query_posterior import GaussianQueryPosterior


def _posterior(
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    information_increment: np.ndarray,
    natural_increment: np.ndarray,
) -> GaussianQueryPosterior:
    prior_precision = np.linalg.inv(prior_covariance)
    posterior_precision = prior_precision + information_increment
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ (
        prior_precision @ prior_mean + natural_increment
    )
    reduction = 0.5 * (
        prior_covariance - posterior_covariance
        + (prior_covariance - posterior_covariance).T
    )
    return GaussianQueryPosterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        mean_shift=posterior_mean - prior_mean,
        covariance_reduction=reduction,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        innovation_precision_quadratic=0.0,
        innovation_log_determinant=0.0,
        innovation_negative_log_likelihood=0.0,
        observation_dimension=12,
    )


def _messages() -> tuple[GaussianQueryMessage, GaussianQueryMessage]:
    prior_mean = np.asarray([0.3, -0.2, 0.5], dtype=np.float64)
    prior_covariance = np.asarray(
        [
            [2.0, 0.3, -0.1],
            [0.3, 1.5, 0.2],
            [-0.1, 0.2, 1.2],
        ],
        dtype=np.float64,
    )
    first_root = np.asarray(
        [[0.8, 0.1, -0.2], [0.0, 0.5, 0.1], [0.2, -0.1, 0.6]],
        dtype=np.float64,
    )
    second_root = np.asarray(
        [[0.3, -0.2, 0.1], [0.4, 0.7, -0.1], [-0.2, 0.1, 0.9]],
        dtype=np.float64,
    )
    first = _posterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        information_increment=first_root @ first_root.T,
        natural_increment=np.asarray([0.2, -0.4, 0.1]),
    )
    second = _posterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        information_increment=second_root @ second_root.T,
        natural_increment=np.asarray([-0.3, 0.2, 0.5]),
    )
    return (
        compress_gaussian_query_posterior(
            first,
            query_id="rope-endpoint",
            prior_id="prior-17",
            evidence_ids=("window-1",),
        ),
        compress_gaussian_query_posterior(
            second,
            query_id="rope-endpoint",
            prior_id="prior-17",
            evidence_ids=("window-2",),
        ),
    )


def test_message_reproduces_query_posterior_and_is_immutable() -> None:
    message, _ = _messages()
    belief = apply_gaussian_query_message(message)
    prior_precision = np.linalg.inv(message.anchor_prior_covariance)
    expected_covariance = np.linalg.inv(
        prior_precision + message.information_increment
    )
    expected_mean = expected_covariance @ (
        prior_precision @ message.anchor_prior_mean
        + message.natural_parameter_increment
    )

    np.testing.assert_allclose(belief.mean, expected_mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        belief.covariance,
        expected_covariance,
        atol=1e-12,
        rtol=1e-12,
    )
    assert message.query_dimension == 3
    assert message.payload_nbytes == 12 * np.dtype(np.float64).itemsize
    assert len(message.message_id) == 64
    assert message.evidence_ids == ("window-1",)
    assert not message.information_increment.flags.writeable
    assert not belief.covariance.flags.writeable
    with pytest.raises(ValueError):
        message.information_increment[0, 0] = 0.0


def test_explicit_anchor_must_be_complete_and_byte_identical() -> None:
    message, _ = _messages()
    explicit = apply_gaussian_query_message(
        message,
        prior_id=message.prior_id,
        prior_mean=message.anchor_prior_mean.copy(),
        prior_covariance=message.anchor_prior_covariance.copy(),
    )
    implicit = apply_gaussian_query_message(message)
    np.testing.assert_array_equal(explicit.mean, implicit.mean)
    with pytest.raises(ValueError, match="supplied together"):
        apply_gaussian_query_message(message, prior_id=message.prior_id)
    changed = message.anchor_prior_mean.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    with pytest.raises(ValueError, match="byte-identical"):
        apply_gaussian_query_message(
            message,
            prior_id=message.prior_id,
            prior_mean=changed,
            prior_covariance=message.anchor_prior_covariance,
        )


def test_covariance_intersection_matches_weighted_component_posteriors() -> None:
    first, second = _messages()
    fused = fuse_gaussian_query_messages_covariance_intersection(
        (first, second),
        weights=np.asarray([0.25, 0.75]),
    )
    belief = apply_gaussian_query_message(fused)
    first_belief = apply_gaussian_query_message(first)
    second_belief = apply_gaussian_query_message(second)
    expected_precision = (
        0.25 * np.linalg.inv(first_belief.covariance)
        + 0.75 * np.linalg.inv(second_belief.covariance)
    )
    expected_natural = (
        0.25 * np.linalg.solve(first_belief.covariance, first_belief.mean)
        + 0.75 * np.linalg.solve(second_belief.covariance, second_belief.mean)
    )
    expected_covariance = np.linalg.inv(expected_precision)
    expected_mean = expected_covariance @ expected_natural

    np.testing.assert_allclose(belief.covariance, expected_covariance)
    np.testing.assert_allclose(belief.mean, expected_mean)
    assert fused.prior_weight == 0.0
    assert fused.evidence_ids == ("window-1", "window-2")
    assert sum(fused.component_weights) == pytest.approx(1.0)


def test_unused_weight_is_assigned_to_anchor_prior_once() -> None:
    first, second = _messages()
    fused = fuse_gaussian_query_messages_covariance_intersection(
        (first, second),
        weights=[0.2, 0.3],
    )
    belief = apply_gaussian_query_message(fused)
    prior_precision = np.linalg.inv(first.anchor_prior_covariance)
    first_belief = apply_gaussian_query_message(first)
    second_belief = apply_gaussian_query_message(second)
    expected_precision = (
        0.5 * prior_precision
        + 0.2 * np.linalg.inv(first_belief.covariance)
        + 0.3 * np.linalg.inv(second_belief.covariance)
    )
    expected_natural = (
        0.5 * prior_precision @ first.anchor_prior_mean
        + 0.2 * np.linalg.solve(first_belief.covariance, first_belief.mean)
        + 0.3 * np.linalg.solve(second_belief.covariance, second_belief.mean)
    )

    np.testing.assert_allclose(belief.covariance, np.linalg.inv(expected_precision))
    np.testing.assert_allclose(
        belief.mean,
        np.linalg.solve(expected_precision, expected_natural),
    )
    assert fused.prior_weight == pytest.approx(0.5)


def test_duplicate_message_cannot_create_additional_confidence() -> None:
    message, _ = _messages()
    duplicate = fuse_gaussian_query_messages_covariance_intersection(
        (message, message),
        weights=[0.5, 0.5],
    )
    reference = apply_gaussian_query_message(message)
    actual = apply_gaussian_query_message(duplicate)
    np.testing.assert_allclose(actual.mean, reference.mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        actual.covariance,
        reference.covariance,
        atol=1e-12,
        rtol=1e-12,
    )
    assert duplicate.component_message_ids == (message.message_id,)
    assert duplicate.component_weights == pytest.approx((1.0,))

    prior_precision = np.linalg.inv(message.anchor_prior_covariance)
    naive_covariance = np.linalg.inv(
        prior_precision + 2.0 * message.information_increment
    )
    assert np.linalg.det(naive_covariance) < np.linalg.det(reference.covariance)


def test_fusion_is_canonical_under_input_reordering() -> None:
    first, second = _messages()
    forward = fuse_gaussian_query_messages_covariance_intersection(
        (first, second),
        weights=[0.4, 0.6],
    )
    reverse = fuse_gaussian_query_messages_covariance_intersection(
        (second, first),
        weights=[0.6, 0.4],
    )
    assert forward.message_id == reverse.message_id
    np.testing.assert_array_equal(
        forward.information_increment,
        reverse.information_increment,
    )
    assert forward.component_message_ids == reverse.component_message_ids
    assert forward.component_weights == reverse.component_weights


def test_pairwise_selector_chooses_the_more_informative_dominating_message() -> None:
    prior_mean = np.zeros(2, dtype=np.float64)
    prior_covariance = np.eye(2, dtype=np.float64)
    weak = _posterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        information_increment=0.2 * np.eye(2),
        natural_increment=np.asarray([0.1, -0.1]),
    )
    strong = _posterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        information_increment=2.0 * np.eye(2),
        natural_increment=np.asarray([0.2, -0.2]),
    )
    weak_message = compress_gaussian_query_posterior(
        weak,
        query_id="centroid",
        prior_id="prior",
        evidence_ids=("weak",),
    )
    strong_message = compress_gaussian_query_posterior(
        strong,
        query_id="centroid",
        prior_id="prior",
        evidence_ids=("strong",),
    )
    selected = select_pairwise_covariance_intersection(
        weak_message,
        strong_message,
        grid_size=101,
        objective="logdet",
    )
    selected_belief = apply_gaussian_query_message(selected)
    strong_belief = apply_gaussian_query_message(strong_message)
    np.testing.assert_allclose(selected_belief.covariance, strong_belief.covariance)
    assert max(selected.component_weights) == pytest.approx(1.0)


def test_fusion_fails_closed_for_incompatible_anchors_and_weights() -> None:
    first, second = _messages()
    incompatible = GaussianQueryMessage(
        query_id="different-query",
        prior_id=second.prior_id,
        evidence_ids=second.evidence_ids,
        anchor_prior_mean=second.anchor_prior_mean,
        anchor_prior_covariance=second.anchor_prior_covariance,
        information_increment=second.information_increment,
        natural_parameter_increment=second.natural_parameter_increment,
    )
    with pytest.raises(ValueError, match="same query_id"):
        fuse_gaussian_query_messages_covariance_intersection(
            (first, incompatible),
            weights=[0.5, 0.5],
        )
    with pytest.raises(ValueError, match="at most one"):
        fuse_gaussian_query_messages_covariance_intersection(
            (first, second),
            weights=[0.7, 0.7],
        )
    with pytest.raises(ValueError, match="nonnegative"):
        fuse_gaussian_query_messages_covariance_intersection(
            (first, second),
            weights=[-0.1, 0.5],
        )
    with pytest.raises(ValueError, match="one entry"):
        fuse_gaussian_query_messages_covariance_intersection(
            (first, second),
            weights=[1.0],
        )


def test_singular_or_information_increasing_query_covariance_is_rejected() -> None:
    prior_mean = np.zeros(2)
    singular = GaussianQueryPosterior(
        prior_mean=prior_mean,
        prior_covariance=np.diag([1.0, 0.0]),
        mean_shift=np.zeros(2),
        covariance_reduction=np.zeros((2, 2)),
        posterior_mean=prior_mean,
        posterior_covariance=np.diag([1.0, 0.0]),
        innovation_precision_quadratic=0.0,
        innovation_log_determinant=0.0,
        innovation_negative_log_likelihood=0.0,
        observation_dimension=1,
    )
    with pytest.raises(ValueError, match="positive definite"):
        compress_gaussian_query_posterior(
            singular,
            query_id="q",
            prior_id="p",
            evidence_ids=("e",),
        )

    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianQueryMessage(
            query_id="q",
            prior_id="p",
            evidence_ids=("e",),
            anchor_prior_mean=prior_mean,
            anchor_prior_covariance=np.eye(2),
            information_increment=-0.5 * np.eye(2),
            natural_parameter_increment=np.zeros(2),
        )
