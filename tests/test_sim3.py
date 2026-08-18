import numpy as np
import pytest

from prob4d.sim3 import Sim3, so3_exp, so3_log, so3_right_jacobian


def test_so3_round_trip() -> None:
    rotation_vector = np.array([0.2, -0.1, 0.05])
    np.testing.assert_allclose(so3_log(so3_exp(rotation_vector)), rotation_vector, atol=1e-10)


def test_exact_pi_log_is_canonical_and_round_trips() -> None:
    axes = (
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
        np.asarray([-1.0, -2.0, 0.5]) / np.linalg.norm([-1.0, -2.0, 0.5]),
    )
    for axis in axes:
        rotation = so3_exp(np.pi * axis)
        value = so3_log(rotation)
        significant = np.flatnonzero(np.abs(value) > 1e-12)
        assert significant.size
        assert value[int(significant[0])] > 0.0
        assert np.linalg.norm(value) == pytest.approx(np.pi)
        np.testing.assert_allclose(so3_exp(value), rotation, atol=2e-12, rtol=2e-12)


def test_opposite_exact_pi_vectors_have_identical_log() -> None:
    axis = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)

    first = so3_log(so3_exp(np.pi * axis))
    second = so3_log(so3_exp(-np.pi * axis))

    np.testing.assert_allclose(first, second, atol=2e-12, rtol=2e-12)


def test_sim3_vector_is_canonical_at_exact_pi() -> None:
    axis = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    transform = Sim3.from_vector(
        np.concatenate(([0.2], np.pi * axis, [2.0, -1.0, 0.5]))
    )

    vector = transform.as_vector()

    assert vector[1] == pytest.approx(np.pi)
    np.testing.assert_allclose(vector[2:4], np.zeros(2), atol=1e-15)
    np.testing.assert_allclose(vector[4:], np.asarray([2.0, -1.0, 0.5]))


def test_near_pi_log_uses_antisymmetric_orientation() -> None:
    axis = np.asarray([-0.3, 0.7, 0.2], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    for offset in (1e-12, 1e-10, 1e-7, 1e-4):
        vector = (np.pi - offset) * axis
        recovered = so3_log(so3_exp(vector))
        np.testing.assert_allclose(recovered, vector, atol=5e-8, rtol=5e-8)


def test_so3_right_jacobian_matches_point_finite_difference() -> None:
    rotation_vector = np.array([0.5, -0.3, 0.2])
    point = np.array([2.0, -0.5, 3.0])
    rotation = so3_exp(rotation_vector)
    analytic = (
        -rotation
        @ np.array(
            [
                [0.0, -point[2], point[1]],
                [point[2], 0.0, -point[0]],
                [-point[1], point[0], 0.0],
            ]
        )
        @ so3_right_jacobian(rotation_vector)
    )
    numerical = np.empty((3, 3))
    baseline = rotation @ point
    for index in range(3):
        perturbed = rotation_vector.copy()
        perturbed[index] += 1e-6
        numerical[:, index] = (so3_exp(perturbed) @ point - baseline) / 1e-6

    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-6)


def test_sim3_inverse_and_composition() -> None:
    transform = Sim3.from_vector(np.array([0.2, 0.1, -0.2, 0.05, 2.0, -1.0, 0.5]))
    points = np.array([[0.0, 1.0, 2.0], [3.0, -1.0, 4.0]])

    recovered = transform.inverse().transform_points(transform.transform_points(points))

    np.testing.assert_allclose(recovered, points, atol=1e-10)
    np.testing.assert_allclose(
        transform.compose(transform.inverse()).as_vector(),
        np.zeros(7),
        atol=1e-10,
    )


def test_covariance_transform_matches_samples() -> None:
    transform = Sim3.from_vector(np.array([-0.1, 0.1, 0.2, -0.1, 0.0, 0.0, 0.0]))
    covariance = np.diag([1.0, 2.0, 3.0])
    expected = transform.scale**2 * transform.rotation @ covariance @ transform.rotation.T

    np.testing.assert_allclose(transform.transform_covariances(covariance), expected)


def _random_sim3(rng: np.random.Generator) -> Sim3:
    return Sim3(
        scale=float(np.exp(rng.normal(scale=0.4))),
        rotation=so3_exp(rng.normal(size=3) * 0.6),
        translation=rng.normal(size=3),
    )


def test_randomized_sim3_group_action_properties() -> None:
    rng = np.random.default_rng(20260814)
    for _ in range(64):
        first = _random_sim3(rng)
        second = _random_sim3(rng)
        third = _random_sim3(rng)
        points = rng.normal(size=(17, 3))

        np.testing.assert_allclose(
            first.compose(second).transform_points(points),
            first.transform_points(second.transform_points(points)),
            atol=2e-12,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            first.inverse().transform_points(first.transform_points(points)),
            points,
            atol=3e-12,
            rtol=3e-12,
        )
        np.testing.assert_allclose(
            first.compose(second).compose(third).transform_points(points),
            first.compose(second.compose(third)).transform_points(points),
            atol=6e-12,
            rtol=6e-12,
        )


def test_randomized_covariance_transform_preserves_psd_and_composition() -> None:
    rng = np.random.default_rng(314159)
    for _ in range(64):
        first = _random_sim3(rng)
        second = _random_sim3(rng)
        root = rng.normal(size=(3, 3))
        covariance = root @ root.T + np.eye(3) * 1e-5

        direct = first.compose(second).transform_covariances(covariance)
        sequential = first.transform_covariances(
            second.transform_covariances(covariance)
        )
        np.testing.assert_allclose(direct, sequential, atol=5e-12, rtol=5e-12)
        np.testing.assert_allclose(direct, direct.T, atol=5e-12, rtol=0.0)
        assert float(np.linalg.eigvalsh(direct).min()) >= -1e-10
