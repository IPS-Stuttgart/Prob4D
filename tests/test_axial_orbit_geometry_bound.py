from __future__ import annotations

import numpy as np
import pytest

from prob4d.axial_orbit_geometry_bound import (
    axial_point_orbit_coefficients,
    bound_axial_query_coefficient_error,
    project_axial_query_coefficients,
)


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _rotate_about_line(
    points: np.ndarray,
    axis: np.ndarray,
    pivot: np.ndarray,
    angle: float,
) -> np.ndarray:
    unit = _unit(axis)
    x, y, z = unit
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (
        skew @ skew
    )
    return (points - pivot) @ rotation.T + pivot


def test_projected_coefficients_match_independent_rodrigues_queries() -> None:
    rng = np.random.default_rng(982451653)
    points = rng.normal(size=(9, 3))
    axis = rng.normal(size=3)
    pivot = rng.normal(size=3)
    weights = rng.normal(size=(4, 9, 3))
    coefficients = project_axial_query_coefficients(
        points,
        axis=axis,
        pivot=pivot,
        query_weights=weights,
    )
    reference_query = np.einsum("dnc,nc->d", weights, points)
    for angle in np.linspace(-np.pi, np.pi, 19):
        rotated = _rotate_about_line(points, axis, pivot, float(angle))
        actual = np.einsum("dnc,nc->d", weights, rotated) - reference_query
        harmonic_difference = np.array([np.cos(angle) - 1.0, np.sin(angle)])
        expected = coefficients @ harmonic_difference
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)


def test_geometric_bound_contains_random_projected_coefficient_errors() -> None:
    rng = np.random.default_rng(32452843)
    maximum_ratio = 0.0
    for _ in range(400):
        point_count = int(rng.integers(3, 16))
        query_dimension = int(rng.integers(1, 8))
        estimated_points = rng.normal(size=(point_count, 3))
        estimated_axis = _unit(rng.normal(size=3))
        estimated_pivot = rng.normal(size=3)
        weights = rng.normal(size=(query_dimension, point_count, 3))

        point_limits = rng.uniform(0.0, 0.08, size=point_count)
        point_directions = rng.normal(size=(point_count, 3))
        point_directions /= np.linalg.norm(point_directions, axis=1)[:, None]
        point_magnitudes = rng.uniform(0.0, 1.0, size=point_count) * point_limits
        true_points = estimated_points + point_directions * point_magnitudes[:, None]

        pivot_limit = float(rng.uniform(0.0, 0.08))
        pivot_direction = _unit(rng.normal(size=3))
        true_pivot = estimated_pivot + pivot_direction * float(
            rng.uniform(0.0, pivot_limit)
        )

        raw_axis_change = rng.normal(size=3)
        raw_axis_change -= estimated_axis * float(raw_axis_change @ estimated_axis)
        if np.linalg.norm(raw_axis_change) == 0.0:
            raw_axis_change = np.roll(estimated_axis, 1)
            raw_axis_change -= estimated_axis * float(
                raw_axis_change @ estimated_axis
            )
        raw_axis_change = _unit(raw_axis_change)
        true_axis = _unit(
            estimated_axis
            + raw_axis_change * float(rng.uniform(0.0, 0.35))
        )
        axis_error = float(np.linalg.norm(true_axis - estimated_axis))

        estimate = project_axial_query_coefficients(
            estimated_points,
            axis=estimated_axis,
            pivot=estimated_pivot,
            query_weights=weights,
        )
        truth = project_axial_query_coefficients(
            true_points,
            axis=true_axis,
            pivot=true_pivot,
            query_weights=weights,
        )
        actual_error = float(np.linalg.svd(truth - estimate, compute_uv=False)[0])
        bound = bound_axial_query_coefficient_error(
            estimated_points,
            estimated_axis=estimated_axis,
            estimated_pivot=estimated_pivot,
            query_weights=weights,
            point_position_error_bounds=point_limits,
            axis_vector_error_bound=axis_error,
            pivot_position_error_bound=pivot_limit,
        )
        assert actual_error <= bound.coefficient_operator_error_bound + 5e-13
        if bound.coefficient_operator_error_bound > 0.0:
            maximum_ratio = max(
                maximum_ratio,
                actual_error / bound.coefficient_operator_error_bound,
            )
    assert maximum_ratio > 0.0
    assert maximum_ratio <= 1.0 + 1e-12


def test_zero_geometric_error_produces_zero_coefficient_error_bound() -> None:
    points = np.array([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.7]])
    weights = np.eye(6).reshape(6, 2, 3)
    bound = bound_axial_query_coefficient_error(
        points,
        estimated_axis=[0.0, 0.0, 1.0],
        estimated_pivot=[0.0, 0.0, 0.0],
        query_weights=weights,
        point_position_error_bounds=[0.0, 0.0],
        axis_vector_error_bound=0.0,
        pivot_position_error_bound=0.0,
    )
    assert bound.coefficient_operator_error_bound == 0.0
    assert bound.stacked_coefficient_frobenius_bound == 0.0
    np.testing.assert_array_equal(bound.cosine_coefficient_error_bounds, [0.0, 0.0])
    np.testing.assert_array_equal(bound.sine_coefficient_error_bounds, [0.0, 0.0])


def test_point_coefficients_are_frame_equivariant() -> None:
    points = np.array([[0.3, -0.2, 0.7], [-0.4, 0.5, 0.2]])
    axis = _unit(np.array([1.0, 2.0, -1.0]))
    pivot = np.array([0.2, -0.3, 0.1])
    cosine, sine = axial_point_orbit_coefficients(points, axis=axis, pivot=pivot)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 3.5
    translation = np.array([2.0, -1.0, 4.0])
    transformed_cosine, transformed_sine = axial_point_orbit_coefficients(
        scale * (points @ rotation.T) + translation,
        axis=rotation @ axis,
        pivot=scale * (rotation @ pivot) + translation,
    )
    np.testing.assert_allclose(
        transformed_cosine,
        scale * (cosine @ rotation.T),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        transformed_sine,
        scale * (sine @ rotation.T),
        atol=1e-14,
    )


def test_inputs_are_copied_and_outputs_are_readonly() -> None:
    points = np.array([[1.0, 0.0, 0.0]])
    axis = np.array([0.0, 0.0, 1.0])
    pivot = np.zeros(3)
    weights = np.array([[[1.0, 0.0, 0.0]]])
    point_limits = np.array([0.1])
    bound = bound_axial_query_coefficient_error(
        points,
        estimated_axis=axis,
        estimated_pivot=pivot,
        query_weights=weights,
        point_position_error_bounds=point_limits,
        axis_vector_error_bound=0.1,
        pivot_position_error_bound=0.1,
    )
    coefficients = project_axial_query_coefficients(
        points,
        axis=axis,
        pivot=pivot,
        query_weights=weights,
    )
    points[:] = 9.0
    axis[:] = 9.0
    pivot[:] = 9.0
    weights[:] = 9.0
    point_limits[:] = 9.0
    for value in (
        coefficients,
        bound.point_offset_error_bounds,
        bound.cosine_coefficient_error_bounds,
        bound.sine_coefficient_error_bounds,
    ):
        assert not value.flags.writeable
    assert bound.point_offset_error_bounds[0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "points,axis,pivot,weights,point_errors,axis_error,pivot_error",
    [
        ([], [0, 0, 1], [0, 0, 0], np.ones((1, 1, 3)), [], 0.0, 0.0),
        ([[1, 2]], [0, 0, 1], [0, 0, 0], np.ones((1, 1, 3)), [0.0], 0.0, 0.0),
        ([[1, 2, 3]], [0, 0, 0], [0, 0, 0], np.ones((1, 1, 3)), [0.0], 0.0, 0.0),
        ([[1, 2, 3]], [0, 0, 1], [0, 0], np.ones((1, 1, 3)), [0.0], 0.0, 0.0),
        ([[1, 2, 3]], [0, 0, 1], [0, 0, 0], np.ones((1, 2, 3)), [0.0], 0.0, 0.0),
        ([[1, 2, 3]], [0, 0, 1], [0, 0, 0], np.ones((1, 1, 3)), [-1.0], 0.0, 0.0),
        ([[1, 2, 3]], [0, 0, 1], [0, 0, 0], np.ones((1, 1, 3)), [0.0], 2.1, 0.0),
        ([[1, 2, 3]], [0, 0, 1], [0, 0, 0], np.ones((1, 1, 3)), [0.0], 0.0, -1.0),
        ([[np.nan, 2, 3]], [0, 0, 1], [0, 0, 0], np.ones((1, 1, 3)), [0.0], 0.0, 0.0),
    ],
)
def test_invalid_geometric_error_contracts_are_rejected(
    points,
    axis,
    pivot,
    weights,
    point_errors,
    axis_error,
    pivot_error,
) -> None:
    with pytest.raises(ValueError):
        bound_axial_query_coefficient_error(
            points,
            estimated_axis=axis,
            estimated_pivot=pivot,
            query_weights=weights,
            point_position_error_bounds=point_errors,
            axis_vector_error_bound=axis_error,
            pivot_position_error_bound=pivot_error,
        )
