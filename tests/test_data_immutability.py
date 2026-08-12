from __future__ import annotations

import numpy as np
import pytest

import prob4d.fusion as fusion_module
from prob4d.data import FusedSequence, PredictionWindow


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray([3, 4], dtype=np.int64)
    points = np.asarray(
        [
            [[[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]]],
            [[[1.5, 0.0, 1.0], [0.0, 2.5, 1.0]]],
        ]
    )
    valid = np.ones(points.shape[:-1], dtype=bool)
    return frames, points, valid


def test_prediction_window_defensively_copies_and_freezes_arrays() -> None:
    frames, points, valid = _inputs()
    window = PredictionWindow(
        window_id="window-a",
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
    )

    frames[0] = 99
    points[0, 0, 0] = 99.0
    valid[0, 0, 0] = False

    np.testing.assert_array_equal(window.frame_indices, [3, 4])
    np.testing.assert_allclose(window.point_map[0, 0, 0], [1.0, 0.0, 1.0])
    assert bool(window.valid_mask[0, 0, 0])
    assert not window.frame_indices.flags.writeable
    assert not window.point_map.flags.writeable
    assert not window.valid_mask.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        window.point_map[0, 0, 0, 0] = 2.0

    for array in (window.frame_indices, window.point_map, window.valid_mask):
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            array.setflags(write=True)


def test_prediction_window_normalizes_and_freezes_optional_arrays() -> None:
    frames, points, valid = _inputs()
    flow = np.ones_like(points)
    deform = valid.copy()
    rays = 2.0 * points
    window = PredictionWindow(
        window_id="window-a",
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
        scene_flow=flow,
        deform_mask=deform,
        ray_directions=rays,
    )

    np.testing.assert_allclose(
        np.linalg.norm(window.ray_directions[window.valid_mask], axis=-1),
        1.0,
    )
    assert not window.scene_flow.flags.writeable
    assert not window.deform_mask.flags.writeable
    assert not window.ray_directions.flags.writeable

    flow[...] = 7.0
    deform[...] = False
    rays[...] = 0.0
    np.testing.assert_allclose(window.scene_flow, 1.0)
    assert np.all(window.deform_mask)
    assert np.all(np.linalg.norm(window.ray_directions[window.valid_mask], axis=-1) > 0.0)

    for array in (window.scene_flow, window.deform_mask, window.ray_directions):
        assert array is not None
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            array.setflags(write=True)


def test_prediction_window_rejects_nonfinite_active_optional_values() -> None:
    frames, points, valid = _inputs()
    flow = np.ones_like(points)
    flow[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="active scene_flow"):
        PredictionWindow(
            window_id="window-a",
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            scene_flow=flow,
            deform_mask=valid,
        )

    rays = points.copy()
    rays[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="must be nonzero"):
        PredictionWindow(
            window_id="window-a",
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            ray_directions=rays,
        )


def _fused_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    frames = np.asarray([3, 4], dtype=np.int32)
    points = np.asarray(
        [
            [[[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]]],
            [[[1.5, 0.0, 1.0], [0.0, 2.5, 1.0]]],
        ],
        dtype=np.float32,
    )
    valid = np.ones(points.shape[:-1], dtype=bool)
    covariance = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        points.shape + (3,),
    ).copy()
    contributors = np.ones(valid.shape, dtype=np.int32)
    flow = np.full(points.shape, 0.25, dtype=np.float32)
    deform = valid.copy()
    flow_covariance = 2.0 * covariance
    return (
        frames,
        points,
        valid,
        covariance,
        contributors,
        flow,
        deform,
        flow_covariance,
    )


def test_fused_sequence_defensively_copies_normalizes_and_freezes_arrays() -> None:
    (
        frames,
        points,
        valid,
        covariance,
        contributors,
        flow,
        deform,
        flow_covariance,
    ) = _fused_inputs()
    sequence = FusedSequence(
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
        point_covariance=covariance,
        contributors=contributors,
        scene_flow=flow,
        deform_mask=deform,
        flow_covariance=flow_covariance,
    )

    frames[0] = 99
    points[...] = 99.0
    valid[...] = False
    covariance[...] = 99.0
    contributors[...] = 7
    flow[...] = 99.0
    deform[...] = False
    flow_covariance[...] = 99.0

    np.testing.assert_array_equal(sequence.frame_indices, [3, 4])
    np.testing.assert_allclose(sequence.point_map[0, 0, 0], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(sequence.point_covariance[0, 0, 0], np.eye(3))
    np.testing.assert_array_equal(sequence.contributors, 1)
    np.testing.assert_allclose(sequence.scene_flow, 0.25)
    np.testing.assert_allclose(sequence.flow_covariance[0, 0, 0], 2.0 * np.eye(3))
    assert np.all(sequence.valid_mask)
    assert np.all(sequence.deform_mask)

    assert sequence.frame_indices.dtype == np.int64
    assert sequence.point_map.dtype == np.float64
    assert sequence.point_covariance.dtype == np.float64
    assert sequence.contributors.dtype == np.uint16
    assert sequence.scene_flow.dtype == np.float64
    assert sequence.flow_covariance.dtype == np.float64

    arrays = (
        sequence.frame_indices,
        sequence.point_map,
        sequence.valid_mask,
        sequence.point_covariance,
        sequence.contributors,
        sequence.scene_flow,
        sequence.deform_mask,
        sequence.flow_covariance,
    )
    assert all(not array.flags.writeable for array in arrays)
    with pytest.raises(ValueError, match="read-only"):
        sequence.point_covariance[0, 0, 0, 0, 0] = 2.0


def test_fused_sequence_rejects_incomplete_flow_and_invalid_contributors() -> None:
    frames, points, valid, covariance, contributors, flow, deform, _ = _fused_inputs()

    with pytest.raises(ValueError, match="all be present or absent"):
        FusedSequence(
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            point_covariance=covariance,
            contributors=contributors,
            scene_flow=flow,
            deform_mask=deform,
        )

    invalid_contributors = contributors.copy()
    invalid_contributors[0, 0, 0] = 0
    with pytest.raises(ValueError, match="at least one contributor"):
        FusedSequence(
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            point_covariance=covariance,
            contributors=invalid_contributors,
        )


def test_fused_sequence_rejects_nonfinite_active_geometry() -> None:
    frames, points, valid, covariance, contributors, *_ = _fused_inputs()
    points[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="valid point_map"):
        FusedSequence(
            frame_indices=frames,
            point_map=points,
            valid_mask=valid,
            point_covariance=covariance,
            contributors=contributors,
        )


def test_fused_sequence_private_owned_path_avoids_second_dense_copy() -> None:
    frames, points, valid, covariance, contributors, *_ = _fused_inputs()
    frames = frames.astype(np.int64)
    points = points.astype(np.float64)
    covariance = covariance.astype(np.float64)
    contributors = contributors.astype(np.uint16)

    sequence = FusedSequence._from_owned_arrays(
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
        point_covariance=covariance,
        contributors=contributors,
    )

    assert np.shares_memory(sequence.frame_indices, frames)
    assert np.shares_memory(sequence.point_map, points)
    assert np.shares_memory(sequence.valid_mask, valid)
    assert np.shares_memory(sequence.point_covariance, covariance)
    assert np.shares_memory(sequence.contributors, contributors)
    assert all(
        not array.flags.writeable
        for array in (
            frames,
            points,
            valid,
            covariance,
            contributors,
        )
    )


def test_fused_sequence_psd_fast_path_avoids_eigendecomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, points, valid, covariance, contributors, *_ = _fused_inputs()

    def forbidden_eigendecomposition(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("well-conditioned PSD matrices should stay on the minor fast path")

    monkeypatch.setattr(
        fusion_module,
        "validated_covariance_psd",
        forbidden_eigendecomposition,
    )
    sequence = FusedSequence(
        frame_indices=frames,
        point_map=points,
        valid_mask=valid,
        point_covariance=covariance,
        contributors=contributors,
    )

    np.testing.assert_allclose(sequence.point_covariance[0, 0, 0], np.eye(3))
