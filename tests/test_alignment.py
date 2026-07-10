import numpy as np

from prob4d.alignment import _alignment_covariance, align_windows, estimate_sim3_robust
from prob4d.data import PredictionWindow
from prob4d.sim3 import Sim3


def test_robust_sim3_recovers_transform_with_outliers() -> None:
    generator = np.random.default_rng(4)
    source = generator.normal(size=(500, 3))
    truth = Sim3.from_vector(np.array([0.15, 0.1, -0.05, 0.08, 1.0, -0.5, 0.2]))
    target = truth.transform_points(source) + generator.normal(scale=0.003, size=source.shape)
    target[:50] += generator.normal(scale=2.0, size=(50, 3))

    result = estimate_sim3_robust(source, target)

    np.testing.assert_allclose(result.transform.scale, truth.scale, rtol=3e-3)
    np.testing.assert_allclose(result.transform.rotation, truth.rotation, atol=3e-3)
    np.testing.assert_allclose(result.transform.translation, truth.translation, atol=5e-3)
    assert result.inlier_fraction > 0.85
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0)


def test_window_alignment_uses_absolute_overlap_frames() -> None:
    generator = np.random.default_rng(10)
    global_points = generator.normal(size=(10, 4, 6, 3))
    moving_to_reference = Sim3.from_vector(np.array([-0.08, 0.03, 0.02, -0.04, 0.3, -0.1, 0.2]))
    reference = PredictionWindow(
        "reference",
        np.arange(3, 8),
        global_points[3:8],
        np.ones((5, 4, 6), dtype=bool),
    )
    moving = PredictionWindow(
        "moving",
        np.arange(5, 10),
        moving_to_reference.inverse().transform_points(global_points[5:10]),
        np.ones((5, 4, 6), dtype=bool),
    )

    alignment = align_windows(reference, moving)

    np.testing.assert_array_equal(alignment.common_frames, [5, 6, 7])
    np.testing.assert_allclose(
        alignment.result.transform.as_vector(), moving_to_reference.as_vector(), atol=1e-9
    )


def test_alignment_covariance_matches_parameter_finite_difference() -> None:
    generator = np.random.default_rng(12)
    source = generator.normal(size=(30, 3))
    transform = Sim3.from_vector(np.array([0.2, 0.5, -0.3, 0.2, 1.0, 2.0, -1.0]))
    target = transform.transform_points(source) + generator.normal(scale=0.01, size=source.shape)
    weights = np.ones(source.shape[0])

    covariance = _alignment_covariance(source, target, weights, transform)

    vector = transform.as_vector()
    information = np.zeros((7, 7))
    baseline = transform.transform_points(source)
    for point_index in range(source.shape[0]):
        jacobian = np.empty((3, 7))
        for parameter_index in range(7):
            perturbed = vector.copy()
            perturbed[parameter_index] += 1e-6
            jacobian[:, parameter_index] = (
                Sim3.from_vector(perturbed).transform_points(source[point_index])
                - baseline[point_index]
            ) / 1e-6
        information += jacobian.T @ jacobian
    residuals = target - baseline
    variance = float(np.sum(residuals**2) / (3 * source.shape[0] - 7))
    floor = np.diag([1e-10] * 4 + [1e-12] * 3)
    expected = variance * np.linalg.pinv(information, rcond=1e-10) + floor

    np.testing.assert_allclose(covariance, expected, rtol=3e-6, atol=1e-10)
