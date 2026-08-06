from __future__ import annotations

import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.fusion import FusedSequence
from prob4d.metrics import TruthSequence, evaluate_sequence
from prob4d.truth_view import (
    PredictionWindowTruthView,
    prediction_window_truth_view,
)


def _window() -> PredictionWindow:
    frame_indices = np.arange(3, dtype=np.int64)
    rows, columns = np.indices((3, 4), dtype=np.float32)
    points = np.empty((3, 3, 4, 3), dtype=np.float32)
    for frame in range(3):
        points[frame, ..., 0] = columns + 0.1 * frame
        points[frame, ..., 1] = rows - 0.05 * frame
        points[frame, ..., 2] = 2.0 + 0.2 * frame
    valid = np.ones(points.shape[:-1], dtype=bool)
    valid[1, 0, 0] = False
    flow = np.full_like(points, 0.025)
    deform = valid.copy()
    deform[0, -1, -1] = False
    return PredictionWindow(
        window_id="reference-window",
        frame_indices=frame_indices,
        point_map=points,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
        dense_storage_dtype="float32",
    )


def _prediction(window: PredictionWindow) -> FusedSequence:
    shape = window.shape
    point_map = np.asarray(window.point_map, dtype=np.float64).copy()
    point_map[..., 0] += 0.01
    covariance = np.zeros(shape + (3, 3), dtype=np.float64)
    diagonal = np.arange(3)
    covariance[..., diagonal, diagonal] = 0.02
    assert window.scene_flow is not None
    assert window.deform_mask is not None
    scene_flow = np.asarray(window.scene_flow, dtype=np.float64).copy()
    scene_flow[..., 1] -= 0.002
    flow_covariance = np.zeros_like(covariance)
    flow_covariance[..., diagonal, diagonal] = 0.005
    return FusedSequence(
        frame_indices=window.frame_indices,
        point_map=point_map,
        valid_mask=window.valid_mask,
        point_covariance=covariance,
        contributors=np.ones(shape, dtype=np.uint16),
        scene_flow=scene_flow,
        deform_mask=window.deform_mask,
        flow_covariance=flow_covariance,
    )


def test_prediction_window_truth_view_reuses_every_reference_array() -> None:
    window = _window()
    view = prediction_window_truth_view(window)

    assert isinstance(view, TruthSequence)
    assert isinstance(view, PredictionWindowTruthView)
    assert view.source_window_id == window.window_id
    assert view.frame_indices is window.frame_indices
    assert view.point_map is window.point_map
    assert view.valid_mask is window.valid_mask
    assert view.scene_flow is window.scene_flow
    assert view.deform_mask is window.deform_mask
    assert view.point_map.dtype == np.dtype(np.float32)
    assert view.retained_array_bytes == sum(
        value.nbytes
        for value in (
            window.frame_indices,
            window.point_map,
            window.valid_mask,
            window.scene_flow,
            window.deform_mask,
        )
        if value is not None
    )

    with pytest.raises(ValueError):
        view.point_map[0, 0, 0, 0] = 123.0
    with pytest.raises(ValueError):
        view.point_map.setflags(write=True)


def test_prediction_window_truth_view_preserves_evaluation_results() -> None:
    window = _window()
    prediction = _prediction(window)
    copied = TruthSequence(
        frame_indices=window.frame_indices,
        point_map=window.point_map,
        valid_mask=window.valid_mask,
        scene_flow=window.scene_flow,
        deform_mask=window.deform_mask,
    )
    viewed = PredictionWindowTruthView(window)

    copied_metrics = evaluate_sequence(
        prediction,
        copied,
        boundary_frames=[2],
        align_scale_translation=True,
        evaluation_chunk_size=5,
    )
    viewed_metrics = evaluate_sequence(
        prediction,
        viewed,
        boundary_frames=[2],
        align_scale_translation=True,
        evaluation_chunk_size=5,
    )

    copied_record = copied_metrics.to_dict()
    viewed_record = viewed_metrics.to_dict()
    assert viewed_record.keys() == copied_record.keys()
    numeric_differences: dict[str, float] = {}
    for name, copied_value in copied_record.items():
        viewed_value = viewed_record[name]
        if copied_value is None or isinstance(copied_value, int):
            assert viewed_value == copied_value
            continue
        assert viewed_value is not None
        numeric_differences[name] = abs(float(viewed_value) - float(copied_value))
    assert max(numeric_differences.values(), default=0.0) <= 1e-20

    assert viewed.point_map.dtype == np.dtype(np.float32)
    assert copied.point_map.dtype == np.dtype(np.float64)
    assert not np.shares_memory(copied.point_map, window.point_map)
    assert np.shares_memory(viewed.point_map, window.point_map)


def test_prediction_window_truth_view_rejects_unvalidated_values() -> None:
    with pytest.raises(TypeError, match="validated PredictionWindow"):
        PredictionWindowTruthView(object())  # type: ignore[arg-type]
