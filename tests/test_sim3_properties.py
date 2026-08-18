from __future__ import annotations

import numpy as np

from prob4d.sim3 import Sim3, so3_exp, so3_log


def _random_sim3(rng: np.random.Generator) -> Sim3:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-2.8, 2.8)
    return Sim3.from_vector(
        np.concatenate(
            (
                [rng.uniform(-1.5, 1.5)],
                axis * angle,
                rng.normal(scale=4.0, size=3),
            )
        )
    )


def test_sim3_composition_matches_sequential_point_action() -> None:
    rng = np.random.default_rng(20260818)
    for _ in range(64):
        first = _random_sim3(rng)
        second = _random_sim3(rng)
        points = rng.normal(size=(32, 3))
        sequential = first.transform_points(second.transform_points(points))
        composed = first.compose(second).transform_points(points)
        np.testing.assert_allclose(composed, sequential, rtol=2e-12, atol=2e-12)


def test_sim3_inverse_and_associativity_hold_on_actions() -> None:
    rng = np.random.default_rng(90210)
    for _ in range(48):
        first = _random_sim3(rng)
        second = _random_sim3(rng)
        third = _random_sim3(rng)
        points = rng.normal(size=(16, 3))

        identity_action = first.compose(first.inverse()).transform_points(points)
        np.testing.assert_allclose(identity_action, points, rtol=3e-12, atol=3e-12)

        left = first.compose(second).compose(third).transform_points(points)
        right = first.compose(second.compose(third)).transform_points(points)
        np.testing.assert_allclose(left, right, rtol=4e-12, atol=4e-12)


def test_sim3_covariance_action_matches_sample_transform() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(24):
        transform = _random_sim3(rng)
        factor = rng.normal(size=(3, 3))
        covariance = factor @ factor.T + 0.05 * np.eye(3)
        transformed = transform.transform_covariances(covariance)
        linear = transform.scale * transform.rotation
        expected = linear @ covariance @ linear.T
        np.testing.assert_allclose(transformed, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(transformed, transformed.T, rtol=0.0, atol=2e-12)
        assert float(np.linalg.eigvalsh(transformed)[0]) >= -1e-10


def test_so3_round_trip_covers_small_and_near_pi_rotations() -> None:
    vectors = (
        np.array([1e-12, -2e-12, 3e-12]),
        np.array([0.2, -0.3, 0.4]),
        np.array([np.pi - 2e-6, 0.0, 0.0]),
    )
    for vector in vectors:
        rotation = so3_exp(vector)
        reconstructed = so3_exp(so3_log(rotation))
        np.testing.assert_allclose(reconstructed, rotation, rtol=1e-9, atol=1e-9)
