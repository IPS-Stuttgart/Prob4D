from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prob4d.dual_gripper_surface_action import (
    DUAL_GRIPPER_ACTION_SEMANTICS,
    DUAL_GRIPPER_ARM_ORDER,
    DualGripperSurfaceActionWindow,
    dual_gripper_surface_action_from_tracker_poses,
)


def _action(
    *,
    action_id: str = "flatnfold-action-000",
    tracker_calibration_id: str = "b" * 64,
) -> DualGripperSurfaceActionWindow:
    surface_points = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    surface_normals = np.asarray(
        [
            [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]],
        ],
        dtype=np.float32,
    )
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    quaternions = np.zeros((2, 2, 4), dtype=np.float32)
    quaternions[..., 0] = 2.0  # deliberately non-unit wxyz identity
    gripper_open = np.asarray([[True, False], [False, True]], dtype=bool)
    return dual_gripper_surface_action_from_tracker_poses(
        action_id=action_id,
        frame_indices=np.asarray([20, 21], dtype=np.int64),
        surface_points_tracker=surface_points,
        surface_normals_tracker=surface_normals,
        positions_world_from_tracker=positions,
        quaternions_world_from_tracker_wxyz=quaternions,
        gripper_open=gripper_open,
        template_id="a" * 64,
        tracker_calibration_id=tracker_calibration_id,
        pose_stream_id="c" * 64,
        timestamp_association_id="d" * 64,
    )


def test_action_transforms_right_then_left_and_normalizes_quaternions() -> None:
    action = _action()

    assert DUAL_GRIPPER_ARM_ORDER == ("right", "left")
    assert action.action_semantics == DUAL_GRIPPER_ACTION_SEMANTICS
    assert action.frame_count == 2
    assert action.point_count == 4
    assert action.points_per_arm == 2
    assert np.array_equal(action.arm_ids, [0, 0, 1, 1])
    assert np.array_equal(action.template_point_indices, [0, 1, 0, 1])
    assert np.allclose(
        action.robot_positions[0],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
    )
    assert np.allclose(action.robot_positions[1] - action.robot_positions[0], [0, 1, 0])
    assert np.allclose(np.linalg.norm(action.robot_normals, axis=-1), 1.0)
    assert np.allclose(action.robot_colors, [1.0, 0.0, 1.0])
    assert np.all(action.robot_exists)
    assert not action.robot_positions.flags.writeable


def test_point_ids_are_template_and_calibration_scoped_not_action_scoped() -> None:
    first = _action(action_id="first")
    second = _action(action_id="second")
    recalibrated = _action(action_id="third", tracker_calibration_id="e" * 64)

    assert np.array_equal(first.point_ids, second.point_ids)
    assert not np.any(np.isin(first.point_ids, recalibrated.point_ids))


def test_pointworld_sample_exposes_bimanual_fields_with_owned_arrays() -> None:
    action = _action()
    sample = action.pointworld_sample()

    assert sample["__has_right_gripper__"] is True
    assert sample["__has_left_gripper__"] is True
    assert np.asarray(sample["robot_flows"]).shape == (2, 4, 3)
    assert np.asarray(sample["robot_normals"]).shape == (2, 4, 3)
    assert np.asarray(sample["robot_colors"]).shape == (2, 4, 3)
    assert np.array_equal(
        sample["right_gripper_open"],
        np.asarray([[1.0], [0.0]], dtype=np.float32),
    )
    assert np.array_equal(
        sample["left_gripper_open"],
        np.asarray([[0.0], [1.0]], dtype=np.float32),
    )

    robot_flows = np.asarray(sample["robot_flows"])
    robot_flows[0, 0] = 99.0
    assert not np.allclose(robot_flows, action.robot_positions)


def test_action_archive_roundtrip(tmp_path: Path) -> None:
    action = _action()
    archive = tmp_path / "action.npz"
    action.to_npz(archive)
    restored = DualGripperSurfaceActionWindow.from_npz(archive)

    assert restored.action_id == action.action_id
    assert restored.storage_dtype == "float32"
    assert restored.template_id == action.template_id
    assert restored.tracker_calibration_id == action.tracker_calibration_id
    assert restored.pose_stream_id == action.pose_stream_id
    assert restored.timestamp_association_id == action.timestamp_association_id
    assert np.array_equal(restored.frame_indices, action.frame_indices)
    assert np.array_equal(restored.point_ids, action.point_ids)
    assert np.array_equal(restored.robot_positions, action.robot_positions)
    assert np.array_equal(restored.robot_normals, action.robot_normals)
    assert np.array_equal(restored.gripper_open, action.gripper_open)


def test_rotation_is_applied_to_points_and_normals() -> None:
    points = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    normals = np.array(points, copy=True)
    translations = np.zeros((1, 2, 3), dtype=np.float64)
    angle = np.pi / 2.0
    quaternions = np.asarray(
        [[[np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]] * 2],
        dtype=np.float64,
    )
    action = dual_gripper_surface_action_from_tracker_poses(
        action_id="rotated",
        frame_indices=np.asarray([0], dtype=np.int64),
        surface_points_tracker=points,
        surface_normals_tracker=normals,
        positions_world_from_tracker=translations,
        quaternions_world_from_tracker_wxyz=quaternions,
        gripper_open=np.asarray([[False, False]], dtype=bool),
        template_id="1" * 64,
        tracker_calibration_id="2" * 64,
        pose_stream_id="3" * 64,
        timestamp_association_id="4" * 64,
        storage_dtype="float64",
    )

    assert np.allclose(action.robot_positions[0], [[0, 1, 0], [0, 1, 0]], atol=1e-12)
    assert np.allclose(action.robot_normals[0], [[0, 1, 0], [0, 1, 0]], atol=1e-12)


def test_action_rejects_nonboolean_gripper_state() -> None:
    action = _action()
    with pytest.raises(TypeError, match="gripper_open must be bool"):
        dual_gripper_surface_action_from_tracker_poses(
            action_id="bad-open",
            frame_indices=action.frame_indices,
            surface_points_tracker=np.zeros((2, 1, 3), dtype=np.float32),
            surface_normals_tracker=np.asarray(
                [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            positions_world_from_tracker=np.zeros((2, 2, 3), dtype=np.float32),
            quaternions_world_from_tracker_wxyz=np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]] * 2] * 2,
                dtype=np.float32,
            ),
            gripper_open=np.zeros((2, 2), dtype=np.int64),
            template_id="a" * 64,
            tracker_calibration_id="b" * 64,
            pose_stream_id="c" * 64,
            timestamp_association_id="d" * 64,
        )


def test_action_rejects_missing_or_misaligned_rigid_support() -> None:
    action = _action()
    with pytest.raises(ValueError, match="must have shape \(2, M, 3\)"):
        dual_gripper_surface_action_from_tracker_poses(
            action_id="one-arm-template",
            frame_indices=action.frame_indices,
            surface_points_tracker=np.zeros((1, 2, 3), dtype=np.float32),
            surface_normals_tracker=np.ones((1, 2, 3), dtype=np.float32),
            positions_world_from_tracker=np.zeros((2, 2, 3), dtype=np.float32),
            quaternions_world_from_tracker_wxyz=np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]] * 2] * 2,
                dtype=np.float32,
            ),
            gripper_open=np.zeros((2, 2), dtype=bool),
            template_id="a" * 64,
            tracker_calibration_id="b" * 64,
            pose_stream_id="c" * 64,
            timestamp_association_id="d" * 64,
        )

    with pytest.raises(ValueError, match="quaternion and translation trajectories must align"):
        dual_gripper_surface_action_from_tracker_poses(
            action_id="misaligned-poses",
            frame_indices=action.frame_indices,
            surface_points_tracker=np.zeros((2, 1, 3), dtype=np.float32),
            surface_normals_tracker=np.asarray(
                [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            positions_world_from_tracker=np.zeros((2, 2, 3), dtype=np.float32),
            quaternions_world_from_tracker_wxyz=np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]] * 2],
                dtype=np.float32,
            ),
            gripper_open=np.zeros((2, 2), dtype=bool),
            template_id="a" * 64,
            tracker_calibration_id="b" * 64,
            pose_stream_id="c" * 64,
            timestamp_association_id="d" * 64,
        )


def test_archive_rejects_unknown_field(tmp_path: Path) -> None:
    action = _action()
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        schema_name=np.asarray("prob4d.dual-gripper-surface-action-window-npz"),
        schema_version=np.asarray(1, dtype=np.int64),
        storage_dtype=np.asarray("float32"),
        action_id=np.asarray(action.action_id),
        frame_indices=action.frame_indices,
        point_ids=action.point_ids,
        template_point_indices=action.template_point_indices,
        arm_ids=action.arm_ids,
        robot_positions=action.robot_positions,
        robot_normals=action.robot_normals,
        robot_colors=action.robot_colors,
        robot_exists=action.robot_exists,
        gripper_open=action.gripper_open,
        template_id=np.asarray(action.template_id),
        tracker_calibration_id=np.asarray(action.tracker_calibration_id),
        pose_stream_id=np.asarray(action.pose_stream_id),
        timestamp_association_id=np.asarray(action.timestamp_association_id),
        action_semantics=np.asarray(action.action_semantics),
        point_identity_semantics=np.asarray(action.point_identity_semantics),
        coordinate_semantics=np.asarray(action.coordinate_semantics),
        unexpected=np.asarray(1),
    )

    with pytest.raises(ValueError, match=r"extra=\['unexpected'\]"):
        DualGripperSurfaceActionWindow.from_npz(path)
