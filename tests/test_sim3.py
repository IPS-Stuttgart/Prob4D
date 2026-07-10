import numpy as np

from prob4d.sim3 import Sim3, so3_exp, so3_log, so3_right_jacobian


def test_so3_round_trip() -> None:
    rotation_vector = np.array([0.2, -0.1, 0.05])
    np.testing.assert_allclose(so3_log(so3_exp(rotation_vector)), rotation_vector, atol=1e-10)


def test_so3_right_jacobian_matches_point_finite_difference() -> None:
    rotation_vector = np.array([0.5, -0.3, 0.2])
    point = np.array([2.0, -0.5, 3.0])
    rotation = so3_exp(rotation_vector)
    analytic = (
        -rotation
        @ np.array(
            [[0.0, -point[2], point[1]], [point[2], 0.0, -point[0]], [-point[1], point[0], 0.0]]
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
        transform.compose(transform.inverse()).as_vector(), np.zeros(7), atol=1e-10
    )


def test_covariance_transform_matches_samples() -> None:
    transform = Sim3.from_vector(np.array([-0.1, 0.1, 0.2, -0.1, 0.0, 0.0, 0.0]))
    covariance = np.diag([1.0, 2.0, 3.0])
    expected = transform.scale**2 * transform.rotation @ covariance @ transform.rotation.T

    np.testing.assert_allclose(transform.transform_covariances(covariance), expected)
