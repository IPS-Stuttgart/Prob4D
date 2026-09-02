from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.bounded_axial_orbit import (
    bounded_axial_radius,
    point_to_line_coordinates,
)


def _ball_error(rng: np.random.Generator, radius: float) -> np.ndarray:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return radius * rng.random() ** (1.0 / 3.0) * direction


def test_zero_error_bound_recovers_observed_radius_exactly() -> None:
    first = np.array([0.2, -0.3, 0.4])
    second = np.array([1.2, 0.1, -0.2])
    probe = np.array([-0.4, 0.8, 1.7])
    _, radius = point_to_line_coordinates(first, second, probe)
    result = bounded_axial_radius(first, second, probe, 0.0)
    assert result.informative
    assert result.direction_difference_bound == 0.0
    assert result.observed_radius == radius
    assert result.radius_upper_bound == radius
    assert result.outer_full_orbit_width == result.observed_full_orbit_width


def test_outer_bound_contains_random_true_geometries() -> None:
    rng = np.random.default_rng(20260902)
    for _ in range(2048):
        true_a = rng.normal(size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        true_b = true_a + rng.uniform(0.5, 3.0) * direction
        true_probe = true_a + rng.normal(size=3)
        epsilon = rng.uniform(0.0, 0.08) * np.linalg.norm(true_b - true_a)
        observed_a = true_a + _ball_error(rng, epsilon)
        observed_b = true_b + _ball_error(rng, epsilon)
        observed_probe = true_probe + _ball_error(rng, epsilon)
        result = bounded_axial_radius(
            observed_a,
            observed_b,
            observed_probe,
            epsilon,
        )
        _, true_radius = point_to_line_coordinates(true_a, true_b, true_probe)
        assert result.radius_upper_bound + 1e-12 >= true_radius


def test_plugin_radius_can_underestimate_but_outer_bound_does_not() -> None:
    true_a = np.array([0.0, 0.0, 0.0])
    true_b = np.array([1.0, 0.0, 0.0])
    true_probe = np.array([0.5, 1.0, 0.0])
    epsilon = 0.1
    observed_probe = np.array([0.5, 0.9, 0.0])
    result = bounded_axial_radius(true_a, true_b, observed_probe, epsilon)
    _, true_radius = point_to_line_coordinates(true_a, true_b, true_probe)
    assert result.observed_radius < true_radius
    assert result.radius_upper_bound >= true_radius
    assert result.accepts_width(1.9) is False


def test_ambiguous_anchor_direction_fails_closed() -> None:
    result = bounded_axial_radius(
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.05,
    )
    assert not result.informative
    assert math.isinf(result.radius_upper_bound)
    assert result.accepts_width(1e12) is False


def test_similarity_scaling_preserves_dimensionless_bound() -> None:
    first = np.array([0.2, -0.3, 0.4])
    second = np.array([1.2, 0.1, -0.2])
    probe = np.array([-0.4, 0.8, 1.7])
    epsilon = 0.03
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 7.5
    shift = np.array([4.0, -8.0, 2.0])
    original = bounded_axial_radius(first, second, probe, epsilon)
    transformed = bounded_axial_radius(
        scale * (rotation @ first) + shift,
        scale * (rotation @ second) + shift,
        scale * (rotation @ probe) + shift,
        scale * epsilon,
    )
    assert transformed.direction_difference_bound == pytest.approx(
        original.direction_difference_bound
    )
    assert transformed.observed_radius == pytest.approx(scale * original.observed_radius)
    assert transformed.radius_upper_bound == pytest.approx(scale * original.radius_upper_bound)


@pytest.mark.parametrize(
    "points,error",
    [
        (([0, 0, 0], [0, 0, 0], [1, 0, 0]), 0.0),
        (([0, 0], [1, 0, 0], [0, 1, 0]), 0.0),
        (([0, 0, 0], [1, 0, 0], [0, np.nan, 0]), 0.0),
        (([0, 0, 0], [1, 0, 0], [0, 1, 0]), -1.0),
        (([0, 0, 0], [1, 0, 0], [0, 1, 0]), math.inf),
    ],
)
def test_invalid_inputs_are_rejected(points, error) -> None:
    with pytest.raises(ValueError):
        bounded_axial_radius(*points, error)
