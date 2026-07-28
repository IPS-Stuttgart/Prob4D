import numpy as np

from prob4d.alignment import WindowAlignment, estimate_sim3_robust
from prob4d.cross_fitted_disagreement import (
    accumulate_cross_fitted_disagreement,
)
from prob4d.data import PredictionWindow
from prob4d.uncertainty import accumulate_disagreement


def _tile_bias_fixture() -> tuple[
    dict[str, PredictionWindow],
    list[WindowAlignment],
]:
    rows, columns = np.mgrid[0:4, 0:4]
    reference_points = np.stack(
        (
            0.30 * columns,
            0.25 * rows,
            1.0 + 0.20 * rows + 0.10 * columns + 0.05 * rows * columns,
        ),
        axis=-1,
    )
    tile_offsets = np.asarray(
        [
            [0.08, 0.00, 0.02],
            [-0.05, 0.04, -0.01],
            [0.02, -0.07, 0.03],
            [-0.04, -0.02, -0.04],
        ]
    )
    moving_points = reference_points.copy()
    offset_index = 0
    for tile_row in range(2):
        for tile_column in range(2):
            moving_points[
                2 * tile_row : 2 * (tile_row + 1),
                2 * tile_column : 2 * (tile_column + 1),
            ] -= tile_offsets[offset_index]
            offset_index += 1

    valid = np.ones((1, 4, 4), dtype=bool)
    reference = PredictionWindow(
        "reference",
        np.asarray([0]),
        reference_points[None],
        valid,
    )
    moving = PredictionWindow(
        "moving",
        np.asarray([0]),
        moving_points[None],
        valid,
    )
    result = estimate_sim3_robust(
        moving_points.reshape(-1, 3),
        reference_points.reshape(-1, 3),
    )
    alignment = WindowAlignment(
        reference_id="reference",
        moving_id="moving",
        common_frames=np.asarray([0]),
        result=result,
    )
    return (
        {"reference": reference, "moving": moving},
        [alignment],
    )


def _two_alignment_fixture() -> tuple[
    dict[str, PredictionWindow],
    list[WindowAlignment],
]:
    windows, alignments = _tile_bias_fixture()
    reference = windows["reference"]
    second_points = reference.point_map[0].copy()
    second_offsets = np.asarray(
        [
            [-0.03, 0.06, 0.01],
            [0.07, -0.01, -0.03],
            [-0.02, -0.05, 0.04],
            [0.01, 0.03, -0.02],
        ]
    )
    offset_index = 0
    for tile_row in range(2):
        for tile_column in range(2):
            second_points[
                2 * tile_row : 2 * (tile_row + 1),
                2 * tile_column : 2 * (tile_column + 1),
            ] -= second_offsets[offset_index]
            offset_index += 1
    second = PredictionWindow(
        "moving-second",
        np.asarray([0]),
        second_points[None],
        np.ones((1, 4, 4), dtype=bool),
    )
    result = estimate_sim3_robust(
        second_points.reshape(-1, 3),
        reference.point_map[0].reshape(-1, 3),
    )
    windows["moving-second"] = second
    alignments.append(
        WindowAlignment(
            reference_id="reference",
            moving_id="moving-second",
            common_frames=np.asarray([0]),
            result=result,
        )
    )
    return windows, alignments


def test_cross_fitted_disagreement_exposes_fit_optimism() -> None:
    windows, alignments = _tile_bias_fixture()
    in_sample = accumulate_disagreement(windows, alignments)
    cross_fitted, report = accumulate_cross_fitted_disagreement(
        windows,
        alignments,
        folds=4,
        cluster_size=2,
        seed=7,
    )

    in_sample_energy = (
        in_sample["reference"].parallel_mean
        + in_sample["reference"].lateral_mean
    )
    cross_fitted_energy = (
        cross_fitted["reference"].parallel_mean
        + cross_fitted["reference"].lateral_mean
    )

    assert report.fitted_folds == 4
    assert report.skipped_folds == 0
    assert report.evaluated_points == 16
    assert report.evaluated_fraction == 1.0
    np.testing.assert_allclose(
        cross_fitted["reference"].count,
        np.ones((1, 4, 4)),
    )
    assert np.mean(cross_fitted_energy) > 1.5 * np.mean(in_sample_energy)


def test_cross_fitted_disagreement_is_deterministic() -> None:
    windows, alignments = _tile_bias_fixture()
    first, first_report = accumulate_cross_fitted_disagreement(
        windows,
        alignments,
        folds=4,
        cluster_size=2,
        seed=19,
    )
    second, second_report = accumulate_cross_fitted_disagreement(
        windows,
        alignments,
        folds=4,
        cluster_size=2,
        seed=19,
    )

    assert first_report == second_report
    for window_id in windows:
        np.testing.assert_array_equal(
            first[window_id].parallel_sum,
            second[window_id].parallel_sum,
        )
        np.testing.assert_array_equal(
            first[window_id].lateral_sum,
            second[window_id].lateral_sum,
        )
        np.testing.assert_array_equal(
            first[window_id].count,
            second[window_id].count,
        )


def test_cross_fitted_disagreement_is_alignment_order_invariant() -> None:
    windows, alignments = _two_alignment_fixture()
    forward, forward_report = accumulate_cross_fitted_disagreement(
        windows,
        alignments,
        folds=2,
        cluster_size=2,
        seed=31,
    )
    reverse, reverse_report = accumulate_cross_fitted_disagreement(
        windows,
        list(reversed(alignments)),
        folds=2,
        cluster_size=2,
        seed=31,
    )

    assert forward_report == reverse_report
    for window_id in windows:
        np.testing.assert_array_equal(
            forward[window_id].parallel_sum,
            reverse[window_id].parallel_sum,
        )
        np.testing.assert_array_equal(
            forward[window_id].lateral_sum,
            reverse[window_id].lateral_sum,
        )
        np.testing.assert_array_equal(
            forward[window_id].count,
            reverse[window_id].count,
        )


def test_cross_fitted_disagreement_fails_closed_without_two_clusters() -> None:
    windows, alignments = _tile_bias_fixture()
    evidence, report = accumulate_cross_fitted_disagreement(
        windows,
        alignments,
        folds=4,
        cluster_size=16,
    )

    assert report.candidate_folds == 0
    assert report.fitted_folds == 0
    assert report.skipped_alignments == 1
    assert report.overlap_points == 16
    assert report.evaluated_points == 0
    for item in evidence.values():
        assert not np.any(item.count)
        assert not np.any(item.parallel_sum)
        assert not np.any(item.lateral_sum)
