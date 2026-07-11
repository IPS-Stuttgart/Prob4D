import numpy as np

from prob4d.phystwin_experiment import ManualFlowSamples
from prob4d.phystwin_uncertainty import (
    calibrate_covariance_scale,
    gaussian_product,
    intersect_manual_flow_samples,
    sample_covariances,
)


def _manual_samples(frames: list[int], tracks: list[int], offset: float) -> ManualFlowSamples:
    count = len(frames)
    truth_current = np.zeros((count, 3))
    truth_next = np.ones((count, 3))
    return ManualFlowSamples(
        frame_indices=np.asarray(frames),
        track_indices=np.asarray(tracks),
        visual_current_world=truth_current + offset,
        visual_flow_world=truth_next + offset,
        truth_current_world=truth_current,
        truth_next_world=truth_next,
    )


def test_intersection_uses_frame_and_track_identity() -> None:
    first = _manual_samples([1, 1, 2], [0, 1, 0], 0.1)
    second = _manual_samples([2, 1, 3], [0, 1, 0], 0.2)

    result = intersect_manual_flow_samples([first, second])

    np.testing.assert_array_equal(result.frame_indices, [1, 2])
    np.testing.assert_array_equal(result.track_indices, [1, 0])
    np.testing.assert_allclose(result.visual_flow_samples[:, :, 0], [[1.1, 1.1], [1.2, 1.2]])


def test_sample_covariance_is_positive_definite_and_train_scalable() -> None:
    samples = np.array(
        [
            [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )

    covariance = sample_covariances(samples, variance_floor_m2=1e-3)
    scale = calibrate_covariance_scale(np.array([[1.0, 1.0, 1.0]]), covariance[:1])

    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert scale > 0.0


def test_gaussian_product_matches_scalar_precision_weighting() -> None:
    mean_a = np.zeros((2, 3))
    mean_b = np.full((2, 3), 2.0)
    covariance_a = np.eye(3)
    covariance_b = 3.0 * np.eye(3)

    mean, covariance = gaussian_product(mean_a, covariance_a, mean_b, covariance_b)

    np.testing.assert_allclose(mean, 0.5)
    np.testing.assert_allclose(covariance, np.broadcast_to(0.75 * np.eye(3), (2, 3, 3)))
