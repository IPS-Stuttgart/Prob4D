import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.sim3 import Sim3
from prob4d.stratified_alignment import (
    align_windows_stratified,
    stratified_overlapping_correspondences,
)


def windows(
    *,
    frames: int = 2,
    height: int = 16,
    width: int = 16,
    transform: Sim3 | None = None,
) -> tuple[PredictionWindow, PredictionWindow]:
    frame_indices = np.arange(10, 10 + frames)
    rows, columns = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )
    base = np.stack(
        (
            0.03 * columns,
            0.03 * rows,
            0.5 + 0.01 * rows + 0.02 * columns + 0.001 * rows * columns,
        ),
        axis=-1,
    )
    source = np.stack(
        [base + np.array([0.0, 0.0, 0.05 * index]) for index in range(frames)]
    )
    transform = transform or Sim3.identity()
    target = transform.transform_points(source)
    mask = np.ones((frames, height, width), dtype=bool)
    return (
        PredictionWindow("reference", frame_indices, target, mask),
        PredictionWindow("moving", frame_indices, source, mask),
    )


def test_stratified_sample_is_deterministic_and_covers_all_tiles() -> None:
    reference, moving = windows()
    first = stratified_overlapping_correspondences(
        reference,
        moving,
        max_correspondences=40,
        spatial_tile_size=4,
    )
    second = stratified_overlapping_correspondences(
        reference,
        moving,
        max_correspondences=40,
        spatial_tile_size=4,
    )

    assert first.available_count == 512
    assert first.sample_count == 40
    assert first.cluster_count == 32
    np.testing.assert_array_equal(first.source_points, second.source_points)
    np.testing.assert_array_equal(first.target_points, second.target_points)
    np.testing.assert_array_equal(first.frame_ids, second.frame_ids)
    np.testing.assert_array_equal(first.rows, second.rows)
    np.testing.assert_array_equal(first.columns, second.columns)
    np.testing.assert_array_equal(first.cluster_ids, second.cluster_ids)


def test_stratified_sample_spans_the_depth_range_within_one_tile() -> None:
    reference, moving = windows(frames=1, height=1, width=20)
    sample = stratified_overlapping_correspondences(
        reference,
        moving,
        max_correspondences=5,
        spatial_tile_size=32,
    )
    depths = np.linalg.norm(sample.source_points, axis=1)
    all_depths = np.linalg.norm(moving.point_map.reshape(-1, 3), axis=1)
    assert depths.min() == pytest.approx(all_depths.min())
    assert depths.max() == pytest.approx(all_depths.max())


def test_stratified_alignment_recovers_exact_similarity_transform() -> None:
    truth = Sim3.from_vector(
        np.array([0.04, 0.05, -0.03, 0.02, 0.3, -0.1, 0.2])
    )
    reference, moving = windows(transform=truth)
    alignment = align_windows_stratified(
        reference,
        moving,
        max_correspondences=80,
        spatial_tile_size=4,
    )

    np.testing.assert_allclose(
        alignment.result.transform.as_vector(),
        truth.as_vector(),
        atol=1e-10,
    )
    assert alignment.result.num_correspondences == 80
    assert alignment.result.num_covariance_clusters == 32
    assert alignment.result.covariance_fallback is None


def test_stratified_alignment_fails_closed_on_too_few_clusters() -> None:
    reference, moving = windows(frames=1, height=2, width=4)
    with pytest.raises(ValueError, match="fewer than eight"):
        align_windows_stratified(
            reference,
            moving,
            max_correspondences=8,
            spatial_tile_size=32,
        )

    alignment = align_windows_stratified(
        reference,
        moving,
        max_correspondences=8,
        spatial_tile_size=32,
        fallback_policy="pointwise",
    )
    assert alignment.result.covariance_fallback.endswith("pointwise_v1")
