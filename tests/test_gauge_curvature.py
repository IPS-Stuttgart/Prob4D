"""Analytic, algebraic, finite-difference, and upstream-Sim3 regression tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.gauge_curvature import (
    SharedGaugeMoments,
    finite_difference_gauge_moments,
    quadratic_gaussian_moments,
    sim3_chain_gauge_moments,
)
from prob4d.sim3 import Sim3


def _quadratic_data(rank: int = 4, outputs: int = 6):
    rng = np.random.default_rng(703)
    c = rng.normal(size=outputs)
    a = rng.normal(size=(outputs, rank))
    b = rng.normal(size=(outputs, rank, rank))
    b = 0.5 * (b + b.swapaxes(1, 2))
    return c, a, b


def _quadratic(c, a, b):
    return lambda z: c + a @ z + 0.5 * np.einsum("i,pij,j->p", z, b, z)


def _axis_rule(function, rank):
    samples = [
        function(sign * math.sqrt(rank) * np.eye(rank)[i])
        for i in range(rank)
        for sign in (-1.0, 1.0)
    ]
    values = np.asarray(samples)
    return values.mean(axis=0), np.atleast_2d(np.cov(values.T, bias=True))


def test_exact_quadratic_moments_against_gaussian_identity():
    c, a, b = _quadratic_data()
    result = quadratic_gaussian_moments(c, a, b)
    expected_mean = c + 0.5 * np.trace(b, axis1=1, axis2=2)
    expected_cov = a @ a.T + 0.5 * np.einsum("pij,qji->pq", b, b)
    np.testing.assert_allclose(result.mean, expected_mean, atol=1e-14)
    np.testing.assert_allclose(result.covariance(), expected_cov, atol=1e-13)
    assert np.linalg.eigvalsh(result.covariance()).min() >= -1e-12


def test_affine_map_exact_and_zero_curvature():
    c, a, _ = _quadratic_data()
    result = quadratic_gaussian_moments(c, a, np.zeros((len(c), a.shape[1], a.shape[1])))
    np.testing.assert_array_equal(result.mean, c)
    np.testing.assert_array_equal(result.curvature_factor, 0)
    np.testing.assert_allclose(result.covariance(), a @ a.T)


def test_mixed_product_is_invisible_to_axis_rule():
    b = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    result = quadratic_gaussian_moments([0.0], np.zeros((1, 2)), b)
    axis_mean, axis_cov = _axis_rule(lambda z: np.array([z[0] * z[1]]), 2)
    np.testing.assert_array_equal(axis_mean, [0.0])
    np.testing.assert_array_equal(axis_cov, [[0.0]])
    np.testing.assert_array_equal(result.mean, [0.0])
    np.testing.assert_array_equal(result.covariance(), [[1.0]])


def test_square_variance_invisible_to_rank_one_axis_rule():
    result = quadratic_gaussian_moments([0.0], [[0.0]], [[[2.0]]])
    mean, covariance = _axis_rule(lambda z: z**2, 1)
    np.testing.assert_array_equal(mean, [1.0])
    np.testing.assert_array_equal(covariance, [[0.0]])
    np.testing.assert_allclose(result.covariance(), [[2.0]])


def test_radial_square_variance_invisible_to_axis_rule():
    rank = 7
    result = quadratic_gaussian_moments([0.0], np.zeros((1, rank)), [2.0 * np.eye(rank)])
    _, covariance = _axis_rule(lambda z: np.array([z @ z]), rank)
    np.testing.assert_allclose(covariance, 0, atol=1e-28)
    np.testing.assert_allclose(result.mean, [rank])
    np.testing.assert_allclose(result.covariance(), [[2.0 * rank]])


def test_orthogonal_whitening_basis_invariance_for_exact_derivatives():
    c, a, b = _quadratic_data()
    rotation, _ = np.linalg.qr(np.random.default_rng(444).normal(size=(4, 4)))
    transformed_b = np.einsum("ia,pij,jb->pab", rotation, b, rotation)
    original = quadratic_gaussian_moments(c, a, b)
    transformed = quadratic_gaussian_moments(c, a @ rotation, transformed_b)
    np.testing.assert_allclose(transformed.mean, original.mean, atol=1e-13)
    np.testing.assert_allclose(transformed.covariance(), original.covariance(), atol=1e-12)


def test_projection_preserves_cross_point_and_query_covariance():
    weights = np.array([1.0, -2.0, 4.0])
    mixed = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = quadratic_gaussian_moments(
        np.zeros(3), np.zeros((3, 2)), weights[:, None, None] * mixed,
    )
    np.testing.assert_allclose(result.covariance(), np.outer(weights, weights))
    contrast = np.array([[2.0, 1.0, 0.0]])
    np.testing.assert_allclose(result.project(contrast).covariance(), [[0.0]], atol=1e-25)
    # Replacing the shared covariance with its diagonal destroys cancellation.
    assert (contrast @ np.diag(result.marginal_variance) @ contrast.T)[0, 0] == 8.0


def test_covariance_action_and_marginal_variance():
    c, a, b = _quadratic_data()
    result = quadratic_gaussian_moments(c, a, b)
    v = np.arange(len(c), dtype=float)
    np.testing.assert_allclose(result.covariance_action(v), result.covariance() @ v)
    np.testing.assert_allclose(result.marginal_variance, np.diag(result.covariance()))
    np.testing.assert_allclose(
        result.covariance_factor @ result.covariance_factor.T, result.covariance(),
    )


def test_input_cross_covariance():
    c, a, b = _quadratic_data()
    root = np.random.default_rng(441).normal(size=(9, 4))
    result = quadratic_gaussian_moments(c, a, b)
    np.testing.assert_allclose(result.input_cross_covariance(root), root @ a.T)


@pytest.mark.parametrize("rank", [1, 2, 4, 7])
def test_finite_difference_quadratic_exactness_and_call_count(rank):
    c, a, b = _quadratic_data(rank, 5)
    calls = []
    function = _quadratic(c, a, b)

    def record(z):
        calls.append(z.copy())
        return function(z)

    result = finite_difference_gauge_moments(record, np.zeros(rank), np.eye(rank), step=0.02)
    expected = quadratic_gaussian_moments(c, a, b)
    assert result.evaluation_count == len(calls) == 1 + 2 * rank**2
    np.testing.assert_allclose(result.mean, expected.mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.covariance(), expected.covariance(), rtol=1e-10, atol=1e-10)


def test_singular_correlated_root_and_nonzero_mean():
    rng = np.random.default_rng(202)
    c, a, b = _quadratic_data(5, 3)
    center = rng.normal(size=5)
    root = rng.normal(size=(5, 2))
    function = _quadratic(c, a, b)
    j = a + np.einsum("pij,j->pi", b, center)
    white_h = np.einsum("ia,pij,jb->pab", root, b, root)
    expected = quadratic_gaussian_moments(function(center), j @ root, white_h)
    actual = finite_difference_gauge_moments(function, center, root, step=0.01)
    np.testing.assert_allclose(actual.mean, expected.mean, atol=1e-9)
    np.testing.assert_allclose(actual.covariance(), expected.covariance(), rtol=1e-9)


def test_zero_rank_evaluates_once_and_keeps_exact_mean():
    result = finite_difference_gauge_moments(
        lambda x: np.array([x.sum()]), [1.0, 2.0], np.empty((2, 0)),
    )
    np.testing.assert_array_equal(result.mean, [3.0])
    np.testing.assert_array_equal(result.covariance(), [[0.0]])
    assert result.covariance_factor.shape == (1, 0)
    assert result.evaluation_count == 1


def _mixed_sim3_case(sigma=0.1, lever=0.5):
    vectors = np.zeros((2, 7))
    root = np.zeros((14, 2))
    root[3, 0] = sigma  # T0: z-axis rotation alpha.
    root[9, 1] = sigma  # T1: y-axis rotation beta.
    return vectors, root, np.array([[0.0, 0.0, lever]])


def test_upstream_sim3_chain_matches_independent_analytic_map():
    alpha, beta, lever = 0.13, -0.21, 0.5
    first, second = np.zeros(7), np.zeros(7)
    first[3], second[2] = alpha, beta
    composed = Sim3.from_vector(first).compose(Sim3.from_vector(second))
    actual = composed.transform_points([[0, 0, lever]])[0]
    expected = lever * np.array([
        np.cos(alpha) * np.sin(beta), np.sin(alpha) * np.sin(beta), np.cos(beta),
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_sim3_mixed_rotation_query_recovers_missing_variance():
    vectors, root, points = _mixed_sim3_case()
    result = sim3_chain_gauge_moments(vectors, root, points, query_matrix=[[0.0, 1.0, 0.0]])
    true_variance = 0.5**2 * ((1 - np.exp(-2 * 0.1**2)) / 2)**2
    np.testing.assert_allclose(result.mean, [0.0], atol=1e-16)
    np.testing.assert_allclose(result.marginal_variance, [0.5**2 * 0.1**4], rtol=2e-8)
    assert 0 < result.marginal_variance[0] / true_variance - 1 < 0.021
    assert result.evaluation_count == 9
    np.testing.assert_array_equal(result.linear_factor, 0)


def test_project_before_or_after_sim3_transform():
    vectors, root, points = _mixed_sim3_case()
    points = np.concatenate((points, 2.0 * points))
    query = np.array([[0, 1, 0, 0, 0, 0], [0, -2, 0, 0, 1, 0]], dtype=float)
    before = sim3_chain_gauge_moments(vectors, root, points, query_matrix=query)
    after = sim3_chain_gauge_moments(vectors, root, points).project(query)
    np.testing.assert_allclose(before.mean, after.mean, atol=1e-10)
    np.testing.assert_allclose(before.covariance(), after.covariance(), atol=1e-10)
    np.testing.assert_allclose(before.marginal_variance[1], 0, atol=1e-24)


def test_common_translation_shared_across_points():
    root = np.zeros((7, 1))
    root[5, 0] = 0.03
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    query = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
    result = sim3_chain_gauge_moments(np.zeros((1, 7)), root, points, query_matrix=query)
    np.testing.assert_allclose(result.covariance(), np.full((2, 2), 0.03**2), atol=1e-14)


def test_defensive_copy_and_readonly_storage():
    c, a, b = _quadratic_data()
    result = quadratic_gaussian_moments(c, a, b)
    original = result.mean.copy()
    c[:] = 123
    a[:] = 321
    b[:] = 222
    np.testing.assert_array_equal(result.mean, original)
    assert not result.mean.flags.writeable
    assert not result.linear_factor.flags.writeable
    assert not result.curvature_factor.flags.writeable
    with pytest.raises(ValueError):
        result.mean[0] = 1


@pytest.mark.parametrize("step", [0.0, -0.1, np.inf, np.nan, True])
def test_invalid_derivative_step_fails(step):
    with pytest.raises(ValueError):
        finite_difference_gauge_moments(lambda x: x, [0.0], [[1.0]], step=step)


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_nonfinite_inputs_fail(bad):
    with pytest.raises(ValueError):
        quadratic_gaussian_moments([bad], [[0.0]], [[[0.0]]])
    with pytest.raises(ValueError):
        finite_difference_gauge_moments(lambda x: x, [0.0], [[bad]])
    with pytest.raises(ValueError):
        finite_difference_gauge_moments(lambda x: np.array([bad]), [0.0], [[1.0]])


def test_asymmetric_hessian_fails():
    with pytest.raises(ValueError, match="symmetric"):
        quadratic_gaussian_moments([0.0], [[0.0, 0.0]], [[[0, 1], [0, 0]]])


def test_rank_limit_never_silently_truncates():
    with pytest.raises(ValueError, match="not truncated"):
        finite_difference_gauge_moments(lambda x: x, np.zeros(4), np.eye(4), max_rank=3)


@pytest.mark.parametrize("max_rank", [-1, 2.5, True])
def test_invalid_rank_limit_fails(max_rank):
    with pytest.raises(ValueError):
        finite_difference_gauge_moments(lambda x: x, [0.0], [[1.0]], max_rank=max_rank)


def test_changed_callback_shape_fails():
    with pytest.raises(ValueError, match="shape changed"):
        finite_difference_gauge_moments(lambda x: np.zeros(1 if x[0] == 0 else 2), [0.0], [[1.0]])


def test_invalid_shapes_fail():
    with pytest.raises(ValueError):
        quadratic_gaussian_moments([0.0], [[0.0]], np.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        finite_difference_gauge_moments(lambda x: x, [0.0, 0.0], [[1.0]])
    with pytest.raises(ValueError):
        sim3_chain_gauge_moments(np.zeros((1, 6)), np.zeros((6, 0)), [[0, 0, 0]])
    with pytest.raises(ValueError):
        sim3_chain_gauge_moments(np.zeros((1, 7)), np.zeros((7, 0)), [[0, 0]])
    with pytest.raises(ValueError):
        sim3_chain_gauge_moments(
            np.zeros((1, 7)), np.zeros((7, 0)), [[0, 0, 0]], query_matrix=[[1.0]],
        )


def test_missing_curvature_features_fail():
    with pytest.raises(ValueError, match="all diagonal and mixed"):
        SharedGaugeMoments(np.zeros(2), np.zeros((2, 3)), np.zeros((2, 3)))


def test_invalid_projection_and_covariance_operands_fail():
    result = quadratic_gaussian_moments([0.0], [[1.0]], [[[0.0]]])
    with pytest.raises(ValueError):
        result.project([[1.0, 2.0]])
    with pytest.raises(ValueError):
        result.covariance_action([1.0, 2.0])
    with pytest.raises(ValueError):
        result.input_cross_covariance([[1.0, 2.0]])


def test_covariance_overflow_is_not_reported_as_finite():
    result = quadratic_gaussian_moments([0.0], [[1e200]], [[[0.0]]])
    with pytest.raises(ValueError, match="non-finite"):
        result.covariance()
    with pytest.raises(ValueError, match="non-finite"):
        _ = result.marginal_variance


def test_general_nonlinear_covariance_is_not_claimed_second_order_exact():
    # For sin(sigma*z), the Hessian vanishes. The missing linear-cubic
    # covariance term is O(sigma**4); the method must not claim to capture it.
    sigma = 0.2
    result = finite_difference_gauge_moments(lambda x: np.sin(x), [0.0], [[sigma]])
    truth = (1 - np.exp(-2 * sigma**2)) / 2
    assert result.marginal_variance[0] > truth
    np.testing.assert_allclose(result.marginal_variance, [sigma**2], rtol=2e-8)
