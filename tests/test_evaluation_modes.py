from __future__ import annotations

import numpy as np
import pytest

from prob4d.evaluation_modes import evaluate_sequence_modes
from prob4d.fusion import FusedSequence
from prob4d.metrics import TruthSequence


def _prediction(
    points: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    scene_flow: np.ndarray | None = None,
    deform_mask: np.ndarray | None = None,
) -> FusedSequence:
    shape = points.shape[:-1]
    valid = (
        np.ones(shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    covariance = np.broadcast_to(
        np.eye(3) * 0.01,
        points.shape + (3,),
    ).copy()
    return FusedSequence(
        frame_indices=np.arange(points.shape[0]),
        point_map=points,
        valid_mask=valid,
        point_covariance=covariance,
        contributors=np.ones(shape, dtype=np.uint16),
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        flow_covariance=(
            None
            if scene_flow is None
            else np.broadcast_to(
                np.eye(3) * 0.01,
                points.shape + (3,),
            ).copy()
        ),
    )


def _truth(
    points: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    scene_flow: np.ndarray | None = None,
    deform_mask: np.ndarray | None = None,
) -> TruthSequence:
    shape = points.shape[:-1]
    valid = (
        np.ones(shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    return TruthSequence(
        frame_indices=np.arange(points.shape[0]),
        point_map=points,
        valid_mask=valid,
        scene_flow=scene_flow,
        deform_mask=deform_mask,
    )


def _two_frame_problem() -> tuple[FusedSequence, TruthSequence]:
    truth_points = np.asarray(
        [
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]],
            [[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]],
        ]
    )
    translation = np.asarray([1.0, -0.5, 0.25])
    predicted_points = (truth_points - translation) / 2.0
    predicted_points[1, ..., 0] += 0.5
    return _prediction(predicted_points), _truth(truth_points)


def test_modes_separate_metric_prefix_and_oracle_alignment() -> None:
    prediction, truth = _two_frame_problem()

    result = evaluate_sequence_modes(
        prediction,
        truth,
        prefix_frame_stop_exclusive=1,
    )

    assert result.prefix_aligned is not None
    np.testing.assert_allclose(result.prefix_aligned.fitted_scale, 2.0)
    np.testing.assert_allclose(
        result.prefix_aligned.fitted_translation,
        [1.0, -0.5, 0.25],
    )
    assert result.prefix_aligned.fit_frame_count == 1
    assert result.prefix_aligned.fit_point_count == 3
    np.testing.assert_allclose(
        result.prefix_aligned.metrics.metric_endpoint_point_rmse,
        1.0,
    )
    assert (
        result.oracle_aligned.metrics.metric_point_rmse
        < result.prefix_aligned.metrics.metric_point_rmse
    )
    assert (
        result.metric.metrics.metric_point_rmse
        > result.oracle_aligned.metrics.metric_point_rmse
    )
    assert result.prefix_aligned.metrics.fitted_alignment_scale == 2.0


def test_prefix_fit_is_append_invariant() -> None:
    prediction, truth = _two_frame_problem()
    base = evaluate_sequence_modes(
        prediction,
        truth,
        prefix_frame_stop_exclusive=1,
    )

    appended_prediction_points = np.concatenate(
        [
            prediction.point_map,
            prediction.point_map[-1:] + np.asarray([[[[8.0, -3.0, 2.0]]]]),
        ],
        axis=0,
    )
    appended_truth_points = np.concatenate(
        [
            truth.point_map,
            truth.point_map[-1:] + np.asarray([[[[0.0, 2.0, 0.0]]]]),
        ],
        axis=0,
    )
    appended = evaluate_sequence_modes(
        _prediction(appended_prediction_points),
        _truth(appended_truth_points),
        prefix_frame_stop_exclusive=1,
    )

    assert base.prefix_aligned is not None
    assert appended.prefix_aligned is not None
    np.testing.assert_allclose(
        appended.prefix_aligned.fitted_scale,
        base.prefix_aligned.fitted_scale,
    )
    np.testing.assert_allclose(
        appended.prefix_aligned.fitted_translation,
        base.prefix_aligned.fitted_translation,
    )
    assert appended.prefix_aligned.fit_point_count == base.prefix_aligned.fit_point_count


def test_flow_evaluation_excludes_invalid_geometry() -> None:
    points = np.asarray(
        [[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]]]
    )
    valid = np.asarray([[[True, True, False]]])
    predicted_flow = np.asarray(
        [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [999.0, 0.0, 0.0]]]]
    )
    truth_flow = np.asarray(
        [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    )
    deform = np.ones(valid.shape, dtype=bool)

    result = evaluate_sequence_modes(
        _prediction(
            points,
            valid_mask=valid,
            scene_flow=predicted_flow,
            deform_mask=deform,
        ),
        _truth(
            points,
            valid_mask=valid,
            scene_flow=truth_flow,
            deform_mask=deform,
        ),
        prefix_frame_stop_exclusive=1,
    )

    assert result.metric.metrics.flow_epe == 0.0
    assert result.prefix_aligned is not None
    assert result.prefix_aligned.metrics.flow_epe == 0.0
    assert result.oracle_aligned.metrics.flow_epe == 0.0


def test_prefix_alignment_requires_preboundary_support() -> None:
    prediction, truth = _two_frame_problem()

    with pytest.raises(ValueError, match="no jointly valid points"):
        evaluate_sequence_modes(
            prediction,
            truth,
            prefix_frame_stop_exclusive=0,
        )


def test_mode_results_are_json_ready() -> None:
    prediction, truth = _two_frame_problem()
    result = evaluate_sequence_modes(
        prediction,
        truth,
        prefix_frame_stop_exclusive=1,
    )

    payload = result.to_dict()

    assert payload["metric"]["mode"] == "metric"
    assert payload["prefix_aligned"]["fit_frame_stop_exclusive"] == 1
    assert payload["oracle_aligned"]["mode"] == "oracle_aligned"
