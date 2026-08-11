import numpy as np
import pytest

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


def _inverse_gaussian_product_oracle(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = mean_a.shape[0]
    cov_a = np.broadcast_to(covariance_a, (count, 3, 3))
    cov_b = np.broadcast_to(covariance_b, (count, 3, 3))
    precision_a = np.linalg.inv(cov_a)
    precision_b = np.linalg.inv(cov_b)
    covariance = np.linalg.inv(precision_a + precision_b)
    information = (
        np.einsum("nij,nj->ni", precision_a, mean_a)
        + np.einsum("nij,nj->ni", precision_b, mean_b)
    )
    mean = np.einsum("nij,nj->ni", covariance, information)
    return mean, covariance


def test_intersection_uses_frame_and_track_identity() -> None:
    first = _manual_samples([1, 1, 2], [0, 1, 0], 0.1)
    second = _manual_samples([2, 1, 3], [0, 1, 0], 0.2)

    result = intersect_manual_flow_samples([first, second])

    np.testing.assert_array_equal(result.frame_indices, [1, 2])
    np.testing.assert_array_equal(result.track_indices, [1, 0])
    np.testing.assert_allclose(
        result.visual_flow_samples[:, :, 0],
        [[1.1, 1.1], [1.2, 1.2]],
    )


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
    np.testing.assert_allclose(
        covariance,
        np.broadcast_to(0.75 * np.eye(3), (2, 3, 3)),
    )


def test_gaussian_product_matches_inverse_oracle_for_noncommuting_batches() -> None:
    mean_a = np.array([[0.5, -1.0, 2.0], [-0.2, 0.4, 1.1]])
    mean_b = np.array([[1.2, 0.3, -0.7], [1.0, -0.5, 0.2]])
    covariance_a = np.array(
        [
            [[2.0, 0.3, 0.1], [0.3, 1.2, 0.2], [0.1, 0.2, 0.8]],
            [[1.1, -0.2, 0.05], [-0.2, 1.6, 0.25], [0.05, 0.25, 0.9]],
        ]
    )
    covariance_b = np.array(
        [
            [[0.7, -0.15, 0.05], [-0.15, 2.2, 0.4], [0.05, 0.4, 1.4]],
            [[2.0, 0.1, -0.3], [0.1, 0.8, 0.05], [-0.3, 0.05, 1.3]],
        ]
    )
    expected_mean, expected_covariance = _inverse_gaussian_product_oracle(
        mean_a,
        covariance_a,
        mean_b,
        covariance_b,
    )

    mean, covariance = gaussian_product(mean_a, covariance_a, mean_b, covariance_b)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        covariance,
        expected_covariance,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(covariance, covariance.swapaxes(1, 2), atol=1e-14)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_gaussian_product_does_not_use_explicit_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_inverse(*args: object, **kwargs: object) -> None:
        raise AssertionError("explicit inverse was called")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)

    gaussian_product(
        np.zeros((1, 3)),
        np.eye(3),
        np.ones((1, 3)),
        2.0 * np.eye(3),
    )


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, 1.0, 0.0]), "positive definite"),
        (np.diag([1.0, 1.0, np.nan]), "finite"),
    ],
)
def test_gaussian_product_rejects_invalid_covariance(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        gaussian_product(
            np.zeros((1, 3)),
            covariance,
            np.ones((1, 3)),
            np.eye(3),
        )


def test_gaussian_product_rejects_nonfinite_mean() -> None:
    with pytest.raises(ValueError, match="means must be finite"):
        gaussian_product(
            np.array([[np.nan, 0.0, 0.0]]),
            np.eye(3),
            np.ones((1, 3)),
            np.eye(3),
        )
