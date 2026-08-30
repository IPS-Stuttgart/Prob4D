"""Independent geometry, analytic-moment, and decision-law controls."""

from __future__ import annotations

import numpy as np
import pytest

from prob4d.axial_gauge import AxialGaugeOrbit, CircularQuadrature, GaussianQueryMixture
from prob4d.axial_gauge_study import PROTOCOL, angular_rule


def uniform_rule(nodes: int = 256) -> CircularQuadrature:
    return CircularQuadrature((np.arange(nodes) + 0.5) * 2 * np.pi / nodes, np.ones(nodes))


def test_exact_line_is_fixed_but_off_axis_probe_moves() -> None:
    center = np.array([4.0, 5.0, 6.0])
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    line = center + np.linspace(-2.0, 2.0, 9)[:, None] * axis
    orbit = AxialGaugeOrbit.from_line(line)
    atoms = orbit.positions(line, uniform_rule().angles)
    np.testing.assert_allclose(atoms, np.broadcast_to(line, atoms.shape), atol=1e-13)
    probe = orbit.positions(np.array([[4.0, 6.0, 6.0]]), uniform_rule().angles)
    assert np.ptp(probe[:, 0, 0]) > 0.1


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_relative_geometry_check_and_zero_extent(scale: float) -> None:
    line = scale * np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(AxialGaugeOrbit.from_line(line).axis, [1.0, 0.0, 0.0])
    line[1, 1] = scale * 1e-5
    with pytest.raises(ValueError, match="not an exact line"):
        AxialGaugeOrbit.from_line(line)
    with pytest.raises(ValueError, match="zero-extent"):
        AxialGaugeOrbit.from_line(np.zeros((3, 3)))


def test_rigid_frame_equivariance() -> None:
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    translation = rng.normal(size=3)
    orbit = AxialGaugeOrbit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
    transformed = AxialGaugeOrbit(rotation @ orbit.center + translation, rotation @ orbit.axis)
    points = rng.normal(size=(4, 3))
    angles = uniform_rule().angles
    expected = orbit.positions(points, angles) @ rotation.T + translation
    actual = transformed.positions(points @ rotation.T + translation, angles)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_shared_angle_preserves_distance_and_joint_covariance() -> None:
    orbit = AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    points = np.array([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]])
    law = orbit.pushforward(points, uniform_rule())
    positions = law.atoms.reshape(-1, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1), 4.0)
    np.testing.assert_allclose(law.covariance[1, 4], -2.0, atol=1e-13)
    sum_y = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert abs(float(sum_y @ law.covariance @ sum_y)) < 1e-12
    # Discarding cross-point covariance would invent variance four here.
    assert float(sum_y @ np.diag(np.diag(law.covariance)) @ sum_y) == pytest.approx(4.0)


def test_harmonic_moments_equal_direct_weighted_atoms() -> None:
    rng = np.random.default_rng(8)
    orbit = AxialGaugeOrbit(rng.normal(size=3), rng.normal(size=3))
    rule = CircularQuadrature(rng.uniform(-np.pi, np.pi, 61), rng.uniform(size=61))
    queries = rng.normal(size=(5, 3))
    mean, covariance = orbit.moments(queries, rule)
    mixture = orbit.pushforward(queries, rule)
    np.testing.assert_allclose(mean.ravel(), mixture.mean, atol=1e-13)
    np.testing.assert_allclose(covariance, mixture.covariance, atol=1e-13)


def test_same_first_two_moments_do_not_determine_query_probability() -> None:
    orbit = AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    query = np.array([[0.0, 1.0, 0.0]])
    uniform = orbit.pushforward(query, uniform_rule())
    threefold = orbit.pushforward(
        query, CircularQuadrature(np.array([0.0, 2 * np.pi / 3, -2 * np.pi / 3]), np.ones(3))
    )
    np.testing.assert_allclose(uniform.mean, threefold.mean, atol=1e-14)
    np.testing.assert_allclose(uniform.covariance, threefold.covariance, atol=1e-14)
    normal = np.array([0.0, 1.0, 0.0])
    assert uniform.halfspace_probability(normal, 0.0) == pytest.approx(0.5)
    assert threefold.halfspace_probability(normal, 0.0) == pytest.approx(1 / 3)


@pytest.mark.parametrize("std", [0.05, 0.6, 1.2])
def test_wrapped_normal_moments_against_continuous_analytic_formula(std: float) -> None:
    rule = angular_rule({"kind": "wrapped-normal", "std_rad": std}, 512)
    for order in [1, 2, 3]:
        assert rule.moment(order) == pytest.approx(np.exp(-0.5 * order**2 * std**2), abs=1e-12)


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
        3 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
        + np.einsum("ij,jk,ik->i", delta, np.linalg.inv(covariance), delta)
    )
    np.testing.assert_allclose(query.logpdf(observations, batch_size=7), expected, atol=1e-13)
    np.testing.assert_array_equal(
        query.logpdf(observations, batch_size=1), query.logpdf(observations, batch_size=7)
    )


def test_zero_weight_components_and_tail_logpdf_stay_finite() -> None:
    query = GaussianQueryMixture(np.array([[0.0], [1e6]]), np.array([1.0, 0.0]), np.eye(1))
    result = query.logpdf(np.array([[1e3]]))
    assert np.isfinite(result[0])
    assert result[0] == pytest.approx(-0.5e6 - 0.5 * np.log(2 * np.pi))
    assert query.halfspace_probability(np.array([1.0]), 0.0) == pytest.approx(0.5)
    assert query.halfspace_probability(np.array([1.0]), 1.6448536269514722) == pytest.approx(0.05)


def test_discrete_law_refuses_lebesgue_log_density() -> None:
    law = GaussianQueryMixture(np.array([[0.0], [1.0]]), np.ones(2), np.zeros((1, 1)))
    assert law.halfspace_probability(np.ones(1), 0.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="positive-definite"):
        law.logpdf(np.zeros((1, 1)))


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 1], [np.inf, 1]])
def test_bad_mass_is_rejected(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        CircularQuadrature(np.array([0.0, 1.0]), np.asarray(weights))


def test_mass_normalization_is_overflow_safe_and_copies_inputs() -> None:
    weights = np.array([1e308, 1e308])
    rule = CircularQuadrature(np.array([0.0, np.pi]), weights)
    weights[:] = 0.0
    np.testing.assert_array_equal(rule.weights, [0.5, 0.5])
    assert not rule.weights.flags.writeable
    assert rule.moment(0) == pytest.approx(1.0)
    assert rule.moment(-1) == pytest.approx(rule.moment(1).conjugate())
    with pytest.raises(TypeError):
        rule.moment(True)


@pytest.mark.parametrize("covariance", [np.array([[-1.0]]), np.eye(2), np.array([[np.nan]])])
def test_invalid_noise_is_rejected(covariance: np.ndarray) -> None:
    with pytest.raises(ValueError):
        GaussianQueryMixture(np.zeros((2, 1)), np.ones(2), covariance)


def test_query_dimension_and_argument_checks() -> None:
    with pytest.raises(ValueError):
        AxialGaugeOrbit(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        CircularQuadrature(np.zeros(3), np.ones(2))
    law = GaussianQueryMixture(np.zeros((1, 3)), np.ones(1), np.eye(3))
    with pytest.raises(ValueError):
        law.logpdf(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        law.halfspace_probability(np.ones(2), 0.0)
    with pytest.raises(ValueError):
        law.halfspace_probability(np.ones(3), np.inf)
    with pytest.raises(TypeError):
        law.logpdf(np.zeros((2, 3)), batch_size=True)
    assert PROTOCOL["statistical_unit"].startswith("one independent")


def test_exact_line_likelihood_preserves_nonuniform_angular_prior() -> None:
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    orbit = AxialGaugeOrbit.from_line(points)
    prior = angular_rule({"kind": "threefold", "std_rad": 0.12}, 512)
    observed = points + np.array([[0.01, 0.02, -0.01], [-0.02, 0.01, 0.03]])
    posterior = orbit.condition_on_correspondences(
        points, observed, prior, noise_covariance=0.01 * np.eye(6)
    )
    np.testing.assert_allclose(posterior.weights, prior.weights, atol=1e-15)


def test_off_axis_evidence_can_resolve_twist_instead_of_erasing_it() -> None:
    orbit = AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.03, 0.0]])
    truth = 1.0
    observed = np.array([[-1.0, 0.0, 0.0], [1.0, 0.03 * np.cos(truth), 0.03 * np.sin(truth)]])
    posterior = orbit.condition_on_correspondences(
        points, observed, uniform_rule(1024), noise_covariance=1e-6 * np.eye(6)
    )
    assert np.angle(posterior.moment(1)) == pytest.approx(truth, abs=1e-8)
    assert abs(posterior.moment(1)) > 0.999
    with pytest.raises(ValueError, match="not an exact line"):
        AxialGaugeOrbit.from_line(np.vstack((points, np.zeros(3))))


def test_correspondence_likelihood_handles_cross_point_covariance() -> None:
    orbit = AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    points = np.array([[0.0, 1.0, 0.0], [0.0, 0.8, 0.1]])
    observed = np.array([[0.01, 0.9, 0.2], [-0.01, 0.7, 0.3]])
    prior = uniform_rule(64)
    covariance = np.eye(6) * 0.1 + np.ones((6, 6)) * 0.02
    posterior = orbit.condition_on_correspondences(
        points, observed, prior, noise_covariance=covariance
    )
    residual = observed.ravel() - orbit.positions(points, prior.angles).reshape(64, 6)
    log_likelihood = -0.5 * np.einsum(
        "ij,jk,ik->i", residual, np.linalg.inv(covariance), residual
    )
    expected = np.exp(log_likelihood - np.max(log_likelihood))
    expected /= np.sum(expected)
    np.testing.assert_allclose(posterior.weights, expected, atol=1e-14)
