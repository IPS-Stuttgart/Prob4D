"""Independent numerical and adversarial controls for exact axial moments."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.axial_gauge_moment_study import build_report, main
from prob4d.axial_gauge_moments import AxialGaugeOrbit, CircularMoments2

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "axial-gauge-moment-closure-v1.json"


def _direct_rodrigues(points, axis, pivot, angles, weights):
    """Independent full rotations, rather than the implementation's factorization."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    values = []
    for angle in angles:
        rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        values.append(((points - pivot) @ rotation.T + pivot).reshape(-1))
    values = np.asarray(values)
    probability = np.asarray(weights) / np.sum(weights)
    mean = probability @ values
    centered = values - mean
    return mean, (centered.T * probability) @ centered


def _joint(result):
    factor = result.shared_factors.reshape(-1, 2)
    return factor @ factor.T


def test_uniform_ring_is_rank_two_and_preserves_cross_point_dependence():
    points = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.5]])
    result = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.zeros(3)).point_moments(
        points, CircularMoments2.uniform()
    )
    np.testing.assert_allclose(result.mean, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(result.marginal_covariance[0], np.diag([0.005, 0.005, 0.0]))
    np.testing.assert_allclose(_joint(result)[:3, 3:], np.diag([-0.005, -0.005, 0.0]))
    assert np.linalg.matrix_rank(_joint(result), tol=1e-14) == 2


@pytest.mark.parametrize("mean,variance", [(0.0, 0.0), (0.7, 0.01), (-2.0, 1.0), (2.9, 9.0)])
def test_wrapped_normal_matches_independent_rodrigues_quadrature(mean, variance):
    points = np.array([[0.2, -0.1, 0.3], [-0.7, 0.5, 0.1], [0.0, 0.0, 0.0]])
    axis, pivot = np.array([1.0, 2.0, -3.0]), np.array([0.1, 0.2, 0.3])
    angular = CircularMoments2.wrapped_normal(mean, variance)
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    expected_mean, expected_covariance = _direct_rodrigues(
        points, axis, pivot, mean + np.sqrt(2.0 * variance) * nodes, weights
    )
    actual = AxialGaugeOrbit(axis, pivot).point_moments(points, angular)
    np.testing.assert_allclose(actual.mean.reshape(-1), expected_mean, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(_joint(actual), expected_covariance, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(angular.first_moment, np.exp(1j * mean - variance / 2), atol=1e-15)
    np.testing.assert_allclose(angular.second_moment, np.exp(2j * mean - 2 * variance), atol=1e-15)


def test_multimodal_atoms_and_linear_query_match_dense_reference():
    rng = np.random.default_rng(48271)
    points = rng.normal(size=(7, 3))
    axis, pivot = np.array([2.0, 1.0, 3.0]), np.array([-0.3, 0.7, 1.2])
    angles, weights = np.array([-2.8, -0.4, 0.5, 2.2]), np.array([1.0, 5.0, 3.0, 7.0])
    angular = CircularMoments2.from_atoms(angles, weights)
    actual = AxialGaugeOrbit(axis, pivot).point_moments(points, angular)
    mean, covariance = _direct_rodrigues(points, axis, pivot, angles, weights)
    np.testing.assert_allclose(actual.mean.reshape(-1), mean, atol=1e-14)
    np.testing.assert_allclose(_joint(actual), covariance, atol=1e-14)
    matrix = rng.normal(size=(4, 7, 3))
    query = actual.project(matrix)
    flat = matrix.reshape(4, -1)
    np.testing.assert_allclose(query.mean, flat @ mean, atol=1e-14)
    np.testing.assert_allclose(query.covariance, flat @ covariance @ flat.T, atol=1e-12)


def test_small_variance_retains_second_order_radial_variance():
    variance = 1e-12
    angular = CircularMoments2.wrapped_normal(0.0, variance)
    np.testing.assert_allclose(angular.covariance[0, 0], 0.5 * variance**2, rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(angular.covariance[1, 1], variance, rtol=2e-12, atol=0.0)
    assert np.all(CircularMoments2.wrapped_normal(0.3, 0.0).covariance == 0.0)


def test_very_broad_wrapped_normal_has_uniform_limit():
    actual = CircularMoments2.wrapped_normal(1.2, 1e300)
    np.testing.assert_allclose(actual.mean, [0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(actual.covariance, 0.5 * np.eye(2), atol=1e-15)


def test_axis_sign_and_angle_reflection_preserve_moments():
    points = np.array([[0.1, 0.2, -0.3], [0.6, 0.2, 0.1]])
    axis, pivot = np.array([1.0, 3.0, -2.0]), np.array([0.2, -0.5, 0.1])
    first = AxialGaugeOrbit(axis, pivot).point_moments(
        points, CircularMoments2.wrapped_normal(0.9, 0.8)
    )
    second = AxialGaugeOrbit(-axis, pivot).point_moments(
        points, CircularMoments2.wrapped_normal(-0.9, 0.8)
    )
    np.testing.assert_allclose(first.mean, second.mean, atol=1e-15)
    np.testing.assert_allclose(_joint(first), _joint(second), atol=1e-15)


def test_similarity_frame_equivariance():
    points = np.array([[0.2, -0.1, 0.3], [-0.7, 0.5, 0.1]])
    axis, pivot = np.array([1.0, 2.0, -3.0]), np.array([0.1, 0.2, 0.3])
    angular = CircularMoments2.wrapped_normal(0.4, 2.0)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scale, shift = 3.7, np.array([5.0, -7.0, 2.0])
    first = AxialGaugeOrbit(axis, pivot).point_moments(points, angular)
    second = AxialGaugeOrbit(rotation @ axis, scale * (rotation @ pivot) + shift).point_moments(
        scale * (points @ rotation.T) + shift, angular
    )
    np.testing.assert_allclose(second.mean, scale * (first.mean @ rotation.T) + shift, atol=1e-14)
    transform = np.kron(np.eye(2), scale * rotation)
    np.testing.assert_allclose(_joint(second), transform @ _joint(first) @ transform.T, atol=1e-13)


def test_shared_uncertainty_does_not_average_away_and_identical_contrast_cancels():
    copies = 64
    points = np.tile([0.1, 0.0, 0.0], (copies, 1))
    moments = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.zeros(3)).point_moments(
        points, CircularMoments2.wrapped_normal(0.0, 1.0)
    )
    weights = np.zeros((copies, 3))
    weights[:, 0] = 1.0 / copies
    np.testing.assert_allclose(
        moments.project(weights).covariance[0, 0], moments.marginal_covariance[0, 0, 0]
    )
    weights[:] = 0.0
    weights[0, 0], weights[1, 0] = 1.0, -1.0
    np.testing.assert_array_equal(moments.project(weights).covariance, [[0.0]])


def test_zero_local_derivative_is_not_global_query_invariance():
    orbit = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    moments = orbit.point_moments([[0.1, 0.0, 0.0]], CircularMoments2.wrapped_normal(0.0, 1.0))
    query = moments.project([[1.0, 0.0, 0.0]])
    assert query.sine_coefficients[0] == 0.0  # dq/dtheta at theta = 0.
    assert query.orbit_amplitude[0] == 0.1
    assert query.covariance[0, 0] > 0.0019
    np.testing.assert_allclose(query.full_orbit_bounds, [[-0.1, 0.1]])
    # Two equally weighted +/-sigma points have the same cosine: no radial variance.
    cubature = CircularMoments2.from_atoms([-1.0, 1.0], [0.5, 0.5])
    cubature_query = orbit.point_moments([[0.1, 0.0, 0.0]], cubature).project([[1.0, 0.0, 0.0]])
    assert cubature_query.covariance[0, 0] == 0.0


def test_collinear_support_is_unchanged_but_near_collinearity_is_not_exact():
    orbit = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    support = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.3]])
    np.testing.assert_array_equal(orbit.support_orbit_diameter(support), np.zeros(3))
    moments = orbit.point_moments(support, CircularMoments2.uniform())
    np.testing.assert_array_equal(moments.mean, support)
    np.testing.assert_array_equal(moments.shared_factors, np.zeros((3, 3, 2)))
    support[0, 0] = 1e-8
    assert orbit.support_orbit_diameter(support)[0] == 2e-8


def test_full_orbit_bounds_are_componentwise_and_tight():
    orbit = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    points = np.array([[2.0, 4.0, 5.0]])
    moments = orbit.point_moments(points, CircularMoments2.uniform())
    query = moments.project(np.eye(3).reshape(3, 1, 3))
    for index in range(3):
        angle = np.arctan2(query.sine_coefficients[index], query.cosine_coefficients[index])
        upper = (
            query.orbit_center[index]
            + query.cosine_coefficients[index] * np.cos(angle)
            + query.sine_coefficients[index] * np.sin(angle)
        )
        np.testing.assert_allclose(upper, query.full_orbit_bounds[index, 1], atol=1e-14)


def test_inputs_are_copied_and_outputs_are_readonly():
    axis, pivot = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    orbit = AxialGaugeOrbit(axis, pivot)
    axis[:] = 7.0
    pivot[:] = 9.0
    result = orbit.point_moments([[0.1, 0.0, 0.0]], CircularMoments2.uniform())
    for array in (
        orbit.axis,
        orbit.pivot,
        result.mean,
        result.shared_factors,
        result.marginal_covariance,
    ):
        assert not array.flags.writeable
    np.testing.assert_allclose(orbit.axis, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(orbit.pivot, [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "mean,covariance",
    [
        ([0.0, 0.0], [[1.0, 0.0], [0.0, -1.0]]),
        ([0.0, 0.0], [[0.5, 0.2], [0.0, 0.5]]),
        ([1.0, 1.0], [[0.0, 0.0], [0.0, 0.0]]),
        ([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]),
        ([np.nan, 0.0], [[0.5, 0.0], [0.0, 0.5]]),
    ],
)
def test_invalid_circular_moments_rejected(mean, covariance):
    with pytest.raises(ValueError):
        CircularMoments2(np.asarray(mean), np.asarray(covariance))


@pytest.mark.parametrize(
    "angles,weights",
    [
        ([], []),
        ([0.0], [0.0]),
        ([0.0, 1.0], [1.0, -1.0]),
        ([0.0], [np.nan]),
        ([np.inf], [1.0]),
        ([0.0], [1.0, 1.0]),
    ],
)
def test_invalid_atoms_rejected(angles, weights):
    with pytest.raises(ValueError):
        CircularMoments2.from_atoms(angles, weights)


@pytest.mark.parametrize("mean,variance", [(np.nan, 1.0), (0.0, np.inf), (0.0, -1.0), (True, 1.0)])
def test_invalid_wrapped_normals_rejected(mean, variance):
    with pytest.raises(ValueError):
        CircularMoments2.wrapped_normal(mean, variance)


def test_invalid_geometry_and_query_shapes_rejected():
    with pytest.raises(ValueError):
        AxialGaugeOrbit(np.zeros(3), np.zeros(3))
    orbit = AxialGaugeOrbit(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    for bad in ([[0.0, np.inf, 0.0]], [[1j, 0.0, 0.0]], [], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            orbit.point_moments(bad, CircularMoments2.uniform())
    points = orbit.point_moments([[0.1, 0.0, 0.0]], CircularMoments2.uniform())
    for bad in ([1.0, 2.0, 3.0], np.ones((1, 2, 3)), np.ones((0, 1, 3))):
        with pytest.raises(ValueError):
            points.project(bad)


def test_protocol_reproduces_counterexample_and_retains_strong_quadrature_control():
    report = build_report(json.loads(PROTOCOL.read_text()))
    primary = next(row for row in report["cases"] if row["angular_std_radians"] == 1.0)
    methods = {row["method"]: row for row in primary["methods"]}
    assert methods["first-order"]["illustrative_std_screen_passes"]
    assert not methods["exact-circular-moments"]["illustrative_std_screen_passes"]
    np.testing.assert_allclose(
        methods["exact-circular-moments"]["query_std_m"], 0.04497646055959317
    )
    np.testing.assert_allclose(
        methods["gauss-hermite-32"]["query_std_m"],
        methods["exact-circular-moments"]["query_std_m"],
        atol=1e-14,
    )
    for case in report["cases"]:
        assert case["reference_mean_absolute_error_m"] < 1e-13
        assert case["reference_variance_absolute_error_m2"] < 1e-13
    np.testing.assert_allclose(
        report["shared_covariance_control"]["variance_understatement_factor"], 64.0
    )


def test_study_rejects_target_access_and_changed_arm_set():
    protocol = json.loads(PROTOCOL.read_text())
    for mutation in ("target", "methods", "unknown"):
        changed = copy.deepcopy(protocol)
        if mutation == "target":
            changed["information_boundary"]["target_outcomes"] = True
        elif mutation == "methods":
            changed["methods"] = ["first-order", "exact-circular-moments"]
        else:
            changed["unregistered"] = 1
        with pytest.raises(ValueError):
            build_report(changed)


def test_study_cli_does_not_overwrite_retained_evidence(tmp_path):
    output = tmp_path / "result.json"
    args = ["--protocol", str(PROTOCOL), "--output", str(output)]
    assert main(args) == 0
    retained = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == retained
