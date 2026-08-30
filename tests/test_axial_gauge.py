from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.axial_gauge import (
    AngularRule,
    GaussianQueryMixture,
    angular_rule,
    evaluate_axial_queries,
    first_harmonic_range,
    propagated_query_mixture,
)


def _expected_mean_covariance(
    points: np.ndarray,
    weights: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = []
    for angle in angles:
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed.append(points @ rotation.T)
    stacked = np.asarray(transformed)
    mean = np.tensordot(weights, stacked, axes=(0, 0))
    delta = stacked - mean[None, :, :]
    covariance = np.einsum("k,kni,kmj->nimj", weights, delta, delta).reshape(
        points.size, points.size
    )
    return mean, covariance


def test_discrete_rule_rejects_non_normalized_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        AngularRule(np.array([0.0, 1.0]), np.array([0.4, 0.4]))


def test_uniform_rule_moments_match_the_continuous_orbit() -> None:
    rule = angular_rule({"kind": "uniform"}, 512)
    assert abs(rule.moment(1)) < 1e-13
    assert abs(rule.moment(2)) < 1e-13
    assert abs(rule.moment(3)) < 1e-13


def test_propagated_mixture_matches_direct_shared_angle_samples() -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
    point_covariance = np.diag([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    angles = np.array([-0.8, 0.1, 1.2])
    weights = np.array([0.2, 0.5, 0.3])
    rule = AngularRule(angles, weights)

    mixture = propagated_query_mixture(
        points,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        rule,
        point_covariance=point_covariance,
    )
    expected_mean, expected_between = _expected_mean_covariance(points, weights, angles)
    expected_within = np.zeros_like(point_covariance)
    for angle, weight in zip(angles, weights, strict=True):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        block_rotation = np.kron(np.eye(2), rotation)
        expected_within += weight * (
            block_rotation @ point_covariance @ block_rotation.T
        )

    np.testing.assert_allclose(mixture.mean(), expected_mean.reshape(-1), atol=1e-12)
    np.testing.assert_allclose(
        mixture.covariance(), expected_between + expected_within, atol=1e-12
    )
    assert np.linalg.norm(mixture.covariance()[:3, 3:]) > 0.05


def test_shared_angle_law_differs_from_independent_point_angles() -> None:
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rule = angular_rule({"kind": "uniform"}, 256)
    mixture = propagated_query_mixture(
        points,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        rule,
    )
    covariance = mixture.covariance()
    assert covariance[0, 3] == pytest.approx(1.0, abs=1e-12)
    assert covariance[1, 4] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("std", [0.05, 0.6, 1.2])
def test_wrapped_normal_moments_against_continuous_analytic_formula(std: float) -> None:
    rule = angular_rule({"kind": "wrapped-normal", "std_rad": std}, 512)
    for order in [1, 2, 3]:
        assert rule.moment(order) == pytest.approx(
            np.exp(-0.5 * order**2 * std**2), abs=1e-12
        )


def test_threefold_and_uniform_continuous_moments_are_matched() -> None:
    uniform = angular_rule({"kind": "uniform"}, 512)
    threefold = angular_rule({"kind": "threefold", "std_rad": 0.12}, 512)
    for order in [1, 2]:
        assert abs(uniform.moment(order) - threefold.moment(order)) < 1e-13
    assert abs(threefold.moment(3)) > 0.9
    assert abs(uniform.moment(3)) < 1e-13


def test_single_component_logpdf_matches_gaussian_and_batches() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(3, 3))
    covariance = matrix @ matrix.T + np.eye(3)
    mean = np.array([1.0, 2.0, 3.0])
    query = GaussianQueryMixture(mean[None, :], np.ones(1), covariance)
    observations = rng.normal(size=(23, 3))
    delta = observations - mean
    expected = -0.5 * (
        3 * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + np.einsum("ij,jk,ik->i", delta, np.linalg.inv(covariance), delta)
    )
    np.testing.assert_allclose(
        query.logpdf(observations, batch_size=7), expected, atol=1e-13
    )
    tolerance = 8 * np.finfo(float).eps
    np.testing.assert_allclose(
        query.logpdf(observations, batch_size=1),
        query.logpdf(observations, batch_size=7),
        rtol=tolerance,
        atol=tolerance,
    )


def test_zero_weight_components_and_tail_logpdf_stay_finite() -> None:
    query = GaussianQueryMixture(
        np.array([[0.0], [1e6]]), np.array([1.0, 0.0]), np.eye(1)
    )
    result = query.logpdf(np.array([[1e3]]))
    assert np.isfinite(result[0])
    assert result[0] == pytest.approx(-0.5e6 - 0.5 * np.log(2 * np.pi))
    assert query.halfspace_probability(np.array([1.0]), 0.0) == pytest.approx(0.5)
    assert query.halfspace_probability(
        np.array([1.0]), 1.6448536269514722
    ) == pytest.approx(0.05)


def test_discrete_law_refuses_lebesgue_log_density() -> None:
    law = GaussianQueryMixture(
        np.array([[0.0], [1.0]]), np.ones(2), np.zeros((1, 1))
    )
    assert law.halfspace_probability(np.ones(1), 0.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="zero continuous covariance"):
        law.logpdf(np.array([[0.25]]))


def test_tail_probability_matches_normal_for_single_component() -> None:
    law = GaussianQueryMixture(np.array([[2.0]]), np.ones(1), np.array([[4.0]]))
    assert law.halfspace_probability(np.array([1.0]), 2.0) == pytest.approx(0.5)
    assert law.halfspace_probability(np.array([1.0]), 4.0) == pytest.approx(
        0.15865525393145707
    )


def test_first_harmonic_range_matches_dense_sampling() -> None:
    offset = 0.7
    cosine = np.array([0.8, -0.2, 0.5])
    sine = np.array([-0.4, 0.9, 0.1])
    direction = np.array([0.3, -0.5, 0.8])
    lower, upper = first_harmonic_range(
        offset, cosine, sine, direction, (-1.2, 0.9)
    )
    theta = np.linspace(-1.2, 0.9, 100_001)
    values = offset + np.outer(np.cos(theta), cosine) + np.outer(
        np.sin(theta), sine
    )
    projected = values @ direction
    assert lower == pytest.approx(projected.min(), abs=1e-10)
    assert upper == pytest.approx(projected.max(), abs=1e-10)


def test_evaluate_axial_queries_reports_bounded_metrics() -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    direction = np.array([1.0, -0.5, 0.25, -0.2, 0.1, 0.4])
    result = evaluate_axial_queries(
        points=points,
        origin=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        rule=angular_rule({"kind": "wrapped-normal", "std_rad": 0.7}, 128),
        query_direction=direction,
        query_threshold=0.1,
        point_covariance=np.eye(points.size) * 0.02,
        source_id="unit-source",
        model_id="unit-model",
    )
    assert result["source_id"] == "unit-source"
    assert result["model_id"] == "unit-model"
    assert result["mixture"]["component_count"] == 128
    assert 0.0 <= result["halfspace_probability"] <= 1.0
    assert result["mean"].shape == (6,)
    assert result["covariance"].shape == (6, 6)
    assert result["first_harmonic_range"][0] <= result["first_harmonic_range"][1]


def test_evaluate_axial_queries_rejects_bad_source_and_model_ids() -> None:
    common = dict(
        points=np.array([[1.0, 0.0, 0.0]]),
        origin=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        rule=angular_rule({"kind": "uniform"}, 32),
        query_direction=np.array([1.0, 0.0, 0.0]),
        query_threshold=0.0,
    )
    with pytest.raises(ValueError, match="source_id"):
        evaluate_axial_queries(**common, source_id="", model_id="model")
    with pytest.raises(ValueError, match="model_id"):
        evaluate_axial_queries(**common, source_id="source", model_id="")


def test_batched_logpdf_matches_unbatched_for_multiple_components() -> None:
    means = np.array([[0.0, 0.0], [1.0, -0.5], [-1.0, 1.5]])
    weights = np.array([0.2, 0.5, 0.3])
    covariance = np.array([[0.8, 0.1], [0.1, 1.3]])
    query = GaussianQueryMixture(means, weights, covariance)
    rng = np.random.default_rng(18)
    observations = rng.normal(size=(29, 2))
    expected = query.logpdf(observations, batch_size=29)
    for batch_size in [1, 2, 7, 13]:
        np.testing.assert_allclose(
            query.logpdf(observations, batch_size=batch_size), expected, atol=1e-13
        )


def test_moment_matching_preserves_mean_and_covariance() -> None:
    means = np.array([[0.0, 0.0], [2.0, -1.0], [-1.0, 3.0]])
    weights = np.array([0.2, 0.5, 0.3])
    covariance = np.array([[0.7, 0.2], [0.2, 1.1]])
    mixture = GaussianQueryMixture(means, weights, covariance)
    matched = mixture.moment_matched()
    np.testing.assert_allclose(matched.mean(), mixture.mean(), atol=1e-12)
    np.testing.assert_allclose(matched.covariance(), mixture.covariance(), atol=1e-12)


def test_sampling_reproduces_mixture_moments() -> None:
    means = np.array([[0.0, 0.0], [2.0, -1.0]])
    weights = np.array([0.4, 0.6])
    covariance = np.array([[0.5, 0.1], [0.1, 0.8]])
    mixture = GaussianQueryMixture(means, weights, covariance)
    samples = mixture.sample(200_000, np.random.default_rng(77))
    np.testing.assert_allclose(samples.mean(axis=0), mixture.mean(), atol=0.01)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False), mixture.covariance(), atol=0.02
    )


def test_mean_and_covariance_are_finite_for_large_means() -> None:
    means = np.array([[1e150, -1e150], [1e150 + 1e135, -1e150 - 1e135]])
    weights = np.array([0.5, 0.5])
    covariance = np.eye(2)
    mixture = GaussianQueryMixture(means, weights, covariance)
    assert np.all(np.isfinite(mixture.mean()))
    assert np.all(np.isfinite(mixture.covariance()))


def test_constructor_validates_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        GaussianQueryMixture(np.zeros(3), np.ones(1), np.eye(3))
    with pytest.raises(ValueError, match="one-dimensional"):
        GaussianQueryMixture(np.zeros((1, 3)), np.ones((1, 1)), np.eye(3))
    with pytest.raises(ValueError, match="component dimension"):
        GaussianQueryMixture(np.zeros((2, 3)), np.ones(1), np.eye(3))
    with pytest.raises(ValueError, match="nonnegative"):
        GaussianQueryMixture(np.zeros((2, 3)), np.array([1.0, -1.0]), np.eye(3))
    with pytest.raises(ValueError, match="sum to one"):
        GaussianQueryMixture(np.zeros((2, 3)), np.array([0.4, 0.4]), np.eye(3))
    with pytest.raises(ValueError, match="square"):
        GaussianQueryMixture(np.zeros((1, 3)), np.ones(1), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="match the query dimension"):
        GaussianQueryMixture(np.zeros((1, 3)), np.ones(1), np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        GaussianQueryMixture(
            np.zeros((1, 2)), np.ones(1), np.array([[1.0, 0.2], [0.0, 1.0]])
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianQueryMixture(np.zeros((1, 2)), np.ones(1), np.diag([1.0, -0.1]))


def test_first_harmonic_range_validates_inputs() -> None:
    with pytest.raises(ValueError, match="matching one-dimensional"):
        first_harmonic_range(0.0, np.zeros(2), np.zeros(3), np.zeros(2), None)
    with pytest.raises(ValueError, match="finite"):
        first_harmonic_range(
            float("nan"), np.zeros(2), np.zeros(2), np.zeros(2), None
        )
    with pytest.raises(ValueError, match="support"):
        first_harmonic_range(
            0.0, np.zeros(2), np.zeros(2), np.zeros(2), (1.0, -1.0)
        )


def test_angular_rule_rejects_invalid_support() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        AngularRule(np.zeros((2, 2)), np.ones(4) / 4)
    with pytest.raises(ValueError, match="matching shape"):
        AngularRule(np.zeros(3), np.ones(2) / 2)
    with pytest.raises(ValueError, match="nonnegative"):
        AngularRule(np.zeros(2), np.array([1.1, -0.1]))
    with pytest.raises(ValueError, match="finite"):
        AngularRule(np.array([0.0, np.nan]), np.array([0.5, 0.5]))


def test_angular_rule_validates_moment_order() -> None:
    rule = angular_rule({"kind": "uniform"}, 32)
    with pytest.raises(ValueError, match="positive integer"):
        rule.moment(0)
    with pytest.raises(ValueError, match="positive integer"):
        rule.moment(-1)
    with pytest.raises(ValueError, match="positive integer"):
        rule.moment(1.5)  # type: ignore[arg-type]


def test_angular_rule_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown angular rule"):
        angular_rule({"kind": "unsupported"}, 32)


def test_angular_rule_requires_positive_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        angular_rule({"kind": "uniform"}, 0)


def test_wrapped_normal_requires_positive_std() -> None:
    with pytest.raises(ValueError, match="positive"):
        angular_rule({"kind": "wrapped-normal", "std_rad": 0.0}, 32)


def test_threefold_requires_positive_std() -> None:
    with pytest.raises(ValueError, match="positive"):
        angular_rule({"kind": "threefold", "std_rad": 0.0}, 32)


def test_axial_query_math_is_rotation_equivariant() -> None:
    points = np.array([[1.2, -0.3, 0.4], [0.2, 1.1, -0.7]])
    origin = np.array([0.1, -0.2, 0.3])
    axis = np.array([0.2, -0.1, 1.0])
    rule = angular_rule({"kind": "wrapped-normal", "std_rad": 0.4}, 128)
    query_direction = np.array([0.7, -0.2, 0.4, -0.1, 0.6, 0.3])

    angle = 0.73
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed_points = points @ rotation.T
    transformed_origin = origin @ rotation.T
    transformed_axis = axis @ rotation.T
    transformed_direction = query_direction.reshape(-1, 3) @ rotation.T

    original = evaluate_axial_queries(
        points=points,
        origin=origin,
        axis=axis,
        rule=rule,
        query_direction=query_direction,
        query_threshold=0.1,
        source_id="source",
        model_id="model",
    )
    transformed = evaluate_axial_queries(
        points=transformed_points,
        origin=transformed_origin,
        axis=transformed_axis,
        rule=rule,
        query_direction=transformed_direction.reshape(-1),
        query_threshold=0.1,
        source_id="source",
        model_id="model",
    )

    np.testing.assert_allclose(original["mean"], transformed["mean"].reshape(-1, 3) @ rotation, atol=1e-12)
    np.testing.assert_allclose(
        original["covariance"],
        np.kron(np.eye(points.shape[0]), rotation.T)
        @ transformed["covariance"]
        @ np.kron(np.eye(points.shape[0]), rotation),
        atol=1e-12,
    )
    assert original["halfspace_probability"] == pytest.approx(
        transformed["halfspace_probability"], abs=1e-12
    )
