from itertools import permutations

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.fusion import (
    fuse_gaussians_covariance_intersection,
    fuse_gaussians_generalized_covariance_intersection,
    fuse_windows,
)
from prob4d.sim3 import Sim3
from prob4d.uncertainty import StructuredCovariance


def _window(
    window_id: str,
    points: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    flow: np.ndarray | None = None,
    deform_mask: np.ndarray | None = None,
) -> PredictionWindow:
    point_map = np.asarray(points, dtype=np.float64)
    if point_map.ndim == 3:
        point_map = point_map[None]
    valid = (
        np.ones(point_map.shape[:-1], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    return PredictionWindow(
        window_id=window_id,
        frame_indices=np.arange(point_map.shape[0]),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform_mask,
    )


def _uncertainty(window: PredictionWindow, variance: float) -> StructuredCovariance:
    rays = np.zeros_like(window.point_map)
    rays[..., 2] = 1.0
    return StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.full(window.shape, variance),
        lateral_variance=np.full(window.shape, variance),
    )


def _random_covariances(
    generator: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    factors = generator.normal(size=shape + (3, 3))
    return (
        np.einsum("...ij,...kj->...ik", factors, factors)
        + 0.2 * np.eye(3)
    )


def test_generalized_ci_preserves_equal_covariance_symmetry() -> None:
    means = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )
    covariances = np.broadcast_to(np.eye(3), (3, 1, 3, 3)).copy()

    mean, covariance, weights = (
        fuse_gaussians_generalized_covariance_intersection(means, covariances)
    )

    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0), atol=1e-10)
    np.testing.assert_allclose(mean, [[1.0, 0.0, 0.0]], atol=1e-10)
    np.testing.assert_allclose(covariance, np.eye(3)[None], atol=1e-10)


def test_generalized_ci_retains_exact_pairwise_path() -> None:
    generator = np.random.default_rng(2)
    means = generator.normal(size=(2, 15, 3))
    covariances = _random_covariances(generator, (2, 15))

    expected_mean, expected_covariance, expected_weight = (
        fuse_gaussians_covariance_intersection(
            means[0],
            covariances[0],
            means[1],
            covariances[1],
        )
    )
    actual_mean, actual_covariance, actual_weights = (
        fuse_gaussians_generalized_covariance_intersection(means, covariances)
    )

    value = float(np.ravel(expected_weight)[0])
    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_covariance, expected_covariance)
    np.testing.assert_allclose(actual_weights, [value, 1.0 - value])


def test_generalized_ci_is_invariant_to_contributor_permutation() -> None:
    generator = np.random.default_rng(3)
    means = generator.normal(size=(4, 23, 3))
    covariances = _random_covariances(generator, (4, 23))
    reference_mean, reference_covariance, reference_weights = (
        fuse_gaussians_generalized_covariance_intersection(means, covariances)
    )

    for permutation in permutations(range(4)):
        ordering = np.asarray(permutation)
        mean, covariance, weights = (
            fuse_gaussians_generalized_covariance_intersection(
                means[ordering],
                covariances[ordering],
            )
        )
        np.testing.assert_allclose(mean, reference_mean, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            covariance,
            reference_covariance,
            rtol=1e-9,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            weights[np.argsort(ordering)],
            reference_weights,
            rtol=1e-8,
            atol=1e-9,
        )


def test_generalized_ci_matches_a_dense_simplex_grid() -> None:
    means = np.zeros((3, 7, 2))
    covariances = np.empty((3, 7, 2, 2))
    covariances[0] = np.array([[1.0, 0.7], [0.7, 2.0]])
    covariances[1] = np.array([[2.0, -0.4], [-0.4, 0.6]])
    covariances[2] = np.array([[0.8, 0.1], [0.1, 3.0]])

    _, output_covariance, weights = (
        fuse_gaussians_generalized_covariance_intersection(means, covariances)
    )
    information = np.linalg.inv(covariances[:, 0])

    def score(candidate: np.ndarray) -> float:
        combined = np.einsum("k,kij->ij", candidate, information)
        return -float(np.linalg.slogdet(combined)[1])

    best_grid_score = np.inf
    for first in range(101):
        for second in range(101 - first):
            candidate = np.array([first, second, 100 - first - second]) / 100.0
            best_grid_score = min(best_grid_score, score(candidate))

    assert score(weights) <= best_grid_score + 2e-4
    assert np.min(np.linalg.eigvalsh(output_covariance[0])) > 0.0


def test_joint_window_ci_handles_masks_and_matches_direct_fusion() -> None:
    point_maps = [
        np.array([[[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [20.0, 0.0, 1.0]]]),
        np.array([[[2.0, 0.0, 1.0], [12.0, 0.0, 1.0], [22.0, 0.0, 1.0]]]),
        np.array([[[4.0, 0.0, 1.0], [14.0, 0.0, 1.0], [24.0, 0.0, 1.0]]]),
    ]
    masks = [
        np.array([[[True, True, True]]]),
        np.array([[[True, True, False]]]),
        np.array([[[True, False, False]]]),
    ]
    windows = [
        _window(chr(ord("c") - index), points, mask=masks[index])
        for index, points in enumerate(point_maps)
    ]
    gauges = {window.window_id: Sim3.identity() for window in windows}
    uncertainties = {
        window.window_id: _uncertainty(window, float(index + 1))
        for index, window in enumerate(windows)
    }

    result = fuse_windows(
        windows,
        gauges,
        uncertainties,
        method="covariance_intersection",
    )
    direct_mean, direct_covariance, _ = (
        fuse_gaussians_generalized_covariance_intersection(
            np.stack([points[0, 0] for points in point_maps])[:, None],
            np.stack([np.eye(3) * value for value in (1.0, 2.0, 3.0)])[:, None],
        )
    )

    np.testing.assert_array_equal(result.contributors[0, 0], [3, 2, 1])
    np.testing.assert_allclose(result.point_map[0, 0, 0], direct_mean[0])
    np.testing.assert_allclose(
        result.point_covariance[0, 0, 0],
        direct_covariance[0],
    )
    for ordering in permutations(windows):
        candidate = fuse_windows(
            list(ordering),
            gauges,
            uncertainties,
            method="covariance_intersection",
        )
        np.testing.assert_array_equal(candidate.contributors, result.contributors)
        np.testing.assert_allclose(candidate.point_map, result.point_map)
        np.testing.assert_allclose(
            candidate.point_covariance,
            result.point_covariance,
        )


def test_scene_flow_uses_the_same_joint_ci_semantics() -> None:
    windows: list[PredictionWindow] = []
    for index in range(3):
        points = np.array([[[[0.0, 0.0, 1.0]]]])
        flow = np.array([[[[float(2 * index), 0.0, 0.0]]]])
        mask = np.ones((1, 1, 1), dtype=bool)
        windows.append(
            PredictionWindow(
                window_id=str(index),
                frame_indices=np.array([0]),
                point_map=points,
                valid_mask=mask,
                scene_flow=flow,
                deform_mask=mask,
            )
        )
    gauges = {window.window_id: Sim3.identity() for window in windows}
    uncertainties = {
        window.window_id: _uncertainty(window, float(index + 1))
        for index, window in enumerate(windows)
    }

    result = fuse_windows(
        windows,
        gauges,
        uncertainties,
        method="covariance_intersection",
    )
    direct_mean, direct_covariance, _ = (
        fuse_gaussians_generalized_covariance_intersection(
            np.stack([window.scene_flow[0, 0, 0] for window in windows])[:, None],
            np.stack([np.eye(3) * value for value in (1.0, 2.0, 3.0)])[:, None],
        )
    )

    np.testing.assert_allclose(result.scene_flow[0, 0, 0], direct_mean[0])
    np.testing.assert_allclose(
        result.flow_covariance[0, 0, 0],
        direct_covariance[0],
    )


def test_generalized_ci_rejects_invalid_covariance() -> None:
    means = np.zeros((3, 1, 3))
    covariances = np.broadcast_to(np.eye(3), (3, 1, 3, 3)).copy()
    covariances[2, 0, 2, 2] = -0.1

    with pytest.raises(ValueError, match="positive semidefinite"):
        fuse_gaussians_generalized_covariance_intersection(means, covariances)


def test_generalized_ci_large_batch_smoke() -> None:
    generator = np.random.default_rng(5)
    means = generator.normal(size=(3, 4_096, 3))
    covariances = _random_covariances(generator, (3, 4_096))

    mean, covariance, weights = (
        fuse_gaussians_generalized_covariance_intersection(means, covariances)
    )

    assert mean.shape == (4_096, 3)
    assert covariance.shape == (4_096, 3, 3)
    assert weights.shape == (3,)
    np.testing.assert_allclose(np.sum(weights), 1.0)
