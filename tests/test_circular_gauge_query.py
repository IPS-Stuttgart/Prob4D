"""Focused analytic and adversarial tests; no provider or held-out data access."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.circular_gauge_query import (
    AffineCircularQuery,
    CircularPrior,
    ProbabilityBounds,
    QueryMoments,
    bounded_risk_admissible,
    path_violation_probability,
    point_rotation_orbit,
    validate_declared_line_support,
    violation_arcs,
)


def normal(mu: float = 0.0, sigma: float = 1.0) -> CircularPrior:
    return CircularPrior.wrapped_normal(mu, sigma, prior_id="test-explicit-conditional-prior")


def scalar_event(c: float, a: float, b: float = 0.0) -> AffineCircularQuery:
    return AffineCircularQuery(np.array([c]), np.array([a]), np.array([b]))


def test_wrapped_normal_characteristic_and_conjugacy() -> None:
    prior = normal(0.4, 0.8)
    for order in range(-5, 6):
        assert prior.trigonometric_moment(order) == pytest.approx(
            np.exp(1j * order * 0.4 - 0.5 * (order * 0.8) ** 2), abs=1e-14
        )


def test_uniform_moments_and_complete_explicit_prior() -> None:
    prior = CircularPrior.uniform(prior_id="explicit-uniform")
    mean, covariance = prior.trigonometric_mean_covariance()
    np.testing.assert_array_equal(mean, np.zeros(2))
    np.testing.assert_array_equal(covariance, np.eye(2) / 2)
    assert prior.trigonometric_moment(0) == 1
    assert prior.trigonometric_moment(1) == 0


def test_stable_narrow_prior_variance() -> None:
    prior = normal(0.0, 1e-8)
    _, covariance = prior.trigonometric_mean_covariance()
    assert covariance[0, 0] == pytest.approx(0.5e-32, rel=1e-12, abs=0)
    assert covariance[1, 1] == pytest.approx(1e-16, rel=1e-12, abs=0)


def test_analytic_radial_counterexample() -> None:
    prior = normal()
    radius = 0.1
    query = scalar_event(0, radius)
    result = query.moments(prior)
    assert result.mean[0] == pytest.approx(radius * np.exp(-0.5), abs=1e-15)
    assert result.covariance[0, 0] == pytest.approx(
        0.5 * radius**2 * (-np.expm1(-1.0)) ** 2, abs=1e-16
    )
    # Both a first derivative and the documented rank-one +/-sigma rule miss it.
    derivative = -radius * np.sin(0.0)
    sigma_samples = query.evaluate(np.array([-1.0, 1.0]))[:, 0]
    assert derivative**2 == 0.0
    assert np.var(sigma_samples) == 0.0
    assert result.covariance[0, 0] > 0.001


def test_mixture_moments_equal_first_two_fourier_moments() -> None:
    prior = CircularPrior(
        np.array([0.25, 0.5]), np.array([0.2, 2.7]), np.array([0.3, 1.1]), 0.25, "mixture"
    )
    mean, covariance = prior.trigonometric_mean_covariance()
    first, second = prior.trigonometric_moment(1), prior.trigonometric_moment(2)
    expected_mean = np.array([first.real, first.imag])
    second_moment = np.array(
        [[(1 + second.real) / 2, second.imag / 2],
         [second.imag / 2, (1 - second.real) / 2]]
    )
    np.testing.assert_allclose(mean, expected_mean, atol=1e-15)
    np.testing.assert_allclose(covariance, second_moment - np.outer(mean, mean), atol=1e-15)


def test_joint_point_covariance_preserves_shared_phase() -> None:
    query = point_rotation_orbit(
        np.array([[0.1, 0, 0], [0.1, 0, 0.2]]),
        axis_origin=np.zeros(3), axis_direction=[0, 0, 1],
    )
    result = query.moments(normal())
    np.testing.assert_allclose(result.covariance[0:3, 3:6], result.covariance[0:3, 0:3])
    assert np.linalg.matrix_rank(result.covariance, tol=1e-12) == 2
    np.testing.assert_allclose(query.evaluate([0])[0], [0.1, 0, 0, 0.1, 0, 0.2])


def test_point_orbit_matches_independent_rodrigues_formula() -> None:
    rng = np.random.default_rng(123)
    points, origin, axis = rng.normal(size=(7, 3)), rng.normal(size=3), rng.normal(size=3)
    unit = axis / np.linalg.norm(axis)
    skew = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
    query = point_rotation_orbit(points, axis_origin=origin, axis_direction=axis)
    for angle in [-2.8, -0.1, 0.0, 0.4, 2.9]:
        rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
        expected = (points - origin) @ rotation.T + origin
        np.testing.assert_allclose(query.evaluate(angle).reshape(-1, 3), expected, atol=2e-15)


def test_invariant_line_has_zero_uncertainty_and_retains_prior() -> None:
    points = np.array([[0, 0, -2], [0, 0, 0], [0, 0, 3]], dtype=float)
    prior = normal(0.4, 2.0)
    before = (prior.weights.copy(), prior.means.copy(), prior.stddevs.copy())
    assert validate_declared_line_support(points, axis_origin=[0, 0, 0], axis_direction=[0, 0, 3]) == 0
    result = point_rotation_orbit(points, axis_origin=[0, 0, 0], axis_direction=[0, 0, 3]).moments(prior)
    np.testing.assert_array_equal(result.mean, points.reshape(-1))
    np.testing.assert_array_equal(result.covariance, np.zeros((9, 9)))
    for original, actual in zip(before, (prior.weights, prior.means, prior.stddevs)):
        np.testing.assert_array_equal(original, actual)


def test_projection_and_coordinate_rescaling() -> None:
    query = AffineCircularQuery(np.array([1.0, 2.0]), np.array([0.3, 0.1]), np.array([0.2, -0.4]))
    matrix, shift = np.array([[2.0, -3.0]]), np.array([0.7])
    transformed = query.project(matrix, offset=shift)
    moments, projected = query.moments(normal()), transformed.moments(normal())
    np.testing.assert_allclose(projected.mean, matrix @ moments.mean + shift)
    np.testing.assert_allclose(projected.covariance, matrix @ moments.covariance @ matrix.T)
    original = path_violation_probability(query, normal())
    scaled = path_violation_probability(AffineCircularQuery(query.offset * 1000, query.cosine * 1000, query.sine * 1000), normal())
    assert original.lower == pytest.approx(scaled.lower, abs=1e-14)


def test_probability_against_independent_unwrapped_normal_formula() -> None:
    probability = path_violation_probability(scalar_event(0.05, -0.1), normal())
    # x<radius/2 iff phi lies in (pi/3+2k*pi, 5pi/3+2k*pi).
    cdf = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    expected = math.fsum(
        cdf(5 * math.pi / 3 + 2 * k * math.pi) - cdf(math.pi / 3 + 2 * k * math.pi)
        for k in range(-5, 6)
    )
    assert probability.lower == pytest.approx(expected, abs=2e-15)
    assert probability.upper - probability.lower <= 1e-12
    assert probability.lower == pytest.approx(0.2950083103791666, abs=1e-15)
    assert not bounded_risk_admissible(probability, maximum_risk=0.10)


def test_wraparound_and_rotation_equivariance() -> None:
    query = AffineCircularQuery(np.array([-0.4, -0.7]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    prior = normal(0.8, 0.7)
    for shift in [0.2, 2.0, 6.0, -1.7]:
        shifted = AffineCircularQuery(
            query.offset,
            query.cosine * np.cos(shift) - query.sine * np.sin(shift),
            query.cosine * np.sin(shift) + query.sine * np.cos(shift),
        )
        actual = path_violation_probability(shifted, normal(0.8 + shift, 0.7))
        expected = path_violation_probability(query, prior)
        assert actual.lower == pytest.approx(expected.lower, abs=2e-14)


def test_uniform_arc_length() -> None:
    result = path_violation_probability(scalar_event(-0.5, 1), CircularPrior.uniform(prior_id="uniform"))
    assert result.lower == pytest.approx(1 / 3, abs=1e-15)
    assert result.upper == result.lower
    assert result.omitted_tail_bound == 0


def test_repeated_constraint_is_not_new_independent_risk() -> None:
    prior = CircularPrior.uniform(prior_id="uniform")
    threshold = math.cos(math.pi / 10)
    single = path_violation_probability(scalar_event(-threshold, 1), prior)
    repeated = AffineCircularQuery(np.full(5, -threshold), np.ones(5), np.zeros(5))
    result = path_violation_probability(repeated, prior)
    assert result.lower == pytest.approx(0.1, abs=1e-15)
    assert result.lower == single.lower
    assert 1 - (1 - single.lower) ** 5 == pytest.approx(0.40951)


def test_independence_can_also_underestimate_risk() -> None:
    phases = np.arange(5) * (2 * math.pi / 5)
    query = AffineCircularQuery(
        np.full(5, -math.cos(math.pi / 10)), np.cos(phases), np.sin(phases)
    )
    result = path_violation_probability(query, CircularPrior.uniform(prior_id="uniform"))
    assert result.lower == pytest.approx(0.5, abs=1e-15)
    assert 1 - 0.9**5 < 0.45 < result.lower


@pytest.mark.parametrize("c,a,expected", [(1, 0, 1), (0, 0, 0), (-1, 0, 0), (1, 1, 1), (-1, 1, 0), (2, 1, 1), (-2, 1, 0)])
def test_constant_and_tangent_events(c: float, a: float, expected: float) -> None:
    result = path_violation_probability(scalar_event(c, a), normal())
    assert result.lower == result.upper == expected


def test_arc_partition_matches_direct_evaluation() -> None:
    rng = np.random.default_rng(654)
    angles = (np.arange(10001) + 0.321) / 10001 * 2 * math.pi
    for _ in range(30):
        query = AffineCircularQuery(rng.uniform(-1.5, 0.0, 5), rng.normal(size=5), rng.normal(size=5))
        arcs = violation_arcs(query)
        analytic_mask = np.zeros(angles.shape, dtype=bool)
        for left, right in arcs:
            analytic_mask |= (angles > left) & (angles < right)
        np.testing.assert_array_equal(analytic_mask, np.any(query.evaluate(angles) > 0, axis=1))


def test_probabilities_match_independent_density_integration() -> None:
    scipy = pytest.importorskip("scipy.integrate")
    prior = CircularPrior(np.array([0.3, 0.45]), np.array([0.2, 3.9]), np.array([0.4, 1.8]), 0.25, "mixed")
    query = AffineCircularQuery(np.array([-0.8, -0.9]), np.array([1, -0.5]), np.array([0.0, 0.9]))
    def density(phi: float) -> float:
        total = prior.uniform_weight / (2 * math.pi)
        for weight, mean, std in zip(prior.weights, prior.means, prior.stddevs):
            total += weight * sum(
                math.exp(-0.5 * ((phi + 2 * k * math.pi - mean) / std) ** 2)
                / (math.sqrt(2 * math.pi) * std) for k in range(-12, 13)
            )
        return float(total)
    expected = sum(scipy.quad(density, left, right, epsabs=1e-13, epsrel=1e-13)[0] for left, right in violation_arcs(query))
    result = path_violation_probability(query, prior)
    assert result.lower == pytest.approx(expected, abs=3e-14)


def test_tail_bound_encloses_tighter_sum() -> None:
    prior = normal(0.8, 4.0)
    query = scalar_event(-0.3, 0.7, 0.8)
    loose = path_violation_probability(query, prior, tail_tolerance=0.01)
    tight = path_violation_probability(query, prior, tail_tolerance=1e-15)
    assert loose.lower <= tight.lower + 1e-15
    assert tight.upper <= loose.upper + 1e-15
    assert loose.omitted_tail_bound <= 0.01


def test_probability_is_monotonic_as_failure_threshold_increases() -> None:
    prior = normal(0.5, 1.2)
    values = [path_violation_probability(scalar_event(-threshold, 1), prior).lower for threshold in np.linspace(-1, 1, 31)]
    assert np.all(np.diff(values) <= 1e-14)


def test_arrays_are_immutable_copies() -> None:
    weights = np.array([1.0])
    prior = CircularPrior(weights, np.array([0.0]), np.array([1.0]), 0, "immutable")
    weights[0] = 0.1
    assert prior.weights[0] == 1
    with pytest.raises(ValueError):
        prior.means[0] = 1
    query = scalar_event(0.0, 0.1)
    with pytest.raises(ValueError):
        query.cosine[0] = 7
    with pytest.raises(ValueError):
        query.moments(prior).covariance[0, 0] = -1


@pytest.mark.parametrize("stddev", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_standard_deviation_is_rejected(stddev: float) -> None:
    with pytest.raises(ValueError):
        normal(0, stddev)


def test_invalid_prior_and_shapes_are_rejected() -> None:
    for args in [([0.4], [0], [1], 0, "x"), ([-1], [0], [1], 0, "x"), ([1], [], [1], 0, "x"), ([1], [0], [1], 0, "")]:
        with pytest.raises(ValueError):
            CircularPrior(*args)
    with pytest.raises(ValueError):
        AffineCircularQuery(np.array([0, 0]), np.array([1]), np.array([0]))
    with pytest.raises(ValueError):
        AffineCircularQuery(np.array([0]), np.array([np.nan]), np.array([0]))
    with pytest.raises(ValueError):
        scalar_event(0, 1).project(np.eye(2))
    with pytest.raises(ValueError):
        QueryMoments(np.array([0.0]), np.array([[-1.0]]))


def test_bent_or_unsupported_geometry_is_not_promoted() -> None:
    for points in [np.array([[0, 0, 0], [1e-3, 0, 1]]), np.zeros((2, 3)), np.zeros((1, 3))]:
        with pytest.raises(ValueError):
            validate_declared_line_support(points, axis_origin=[0, 0, 0], axis_direction=[0, 0, 1])
    with pytest.raises(ValueError):
        point_rotation_orbit([[0, 0, 0]], axis_origin=[0, 0, 0], axis_direction=[0, 0, 0])


def test_bad_risk_parameters_fail_closed() -> None:
    for tolerance in [0, -1, 0.3, np.nan, True]:
        with pytest.raises(ValueError):
            path_violation_probability(scalar_event(0, 1), normal(), tail_tolerance=tolerance)
    for count in [0, -1, True, 2.3]:
        with pytest.raises(ValueError):
            path_violation_probability(scalar_event(0, 1), normal(), max_periods=count)
    with pytest.raises(ValueError):
        path_violation_probability(scalar_event(0, 1), normal(0, 1e20), max_periods=10)
    for risk in [-0.1, 1.1, np.nan, True]:
        with pytest.raises(ValueError):
            bounded_risk_admissible(ProbabilityBounds(0, 0, 0, 0), maximum_risk=risk)


def test_complete_belief_fallback_can_preserve_caller_identity() -> None:
    fallback, candidate = object(), object()
    bound = path_violation_probability(scalar_event(0.05, -0.1), normal())
    selected = candidate if bounded_risk_admissible(bound, maximum_risk=0.1) else fallback
    assert selected is fallback
    # This is caller-level behavior only, not a claim to have run BayesianPhysTwin.


def test_low_rank_factor_preserves_exact_dense_moments() -> None:
    query = AffineCircularQuery(np.zeros(3), np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]))
    factor = query.low_rank_moments(normal())
    dense = query.moments(normal())
    np.testing.assert_allclose(factor.factor @ factor.factor.T, dense.covariance, atol=1e-15)
    np.testing.assert_allclose(factor.marginal_variance, np.diag(dense.covariance), atol=1e-15)
    assert factor.factor.shape == (3, 2)


def test_large_query_stays_linear_memory_and_dense_fails_closed() -> None:
    query = AffineCircularQuery(np.zeros(10000), np.ones(10000), np.zeros(10000))
    factor = query.low_rank_moments(normal())
    assert factor.factor.shape == (10000, 2)
    assert factor.factor.nbytes == 160000
    with pytest.raises(ValueError):
        factor.dense()
