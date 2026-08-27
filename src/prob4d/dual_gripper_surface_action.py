"""Rigid dual-gripper surface trajectories for action-conditioned providers.

Flat'n'Fold exposes two end-effector pose streams and one gripper mesh, whereas
PointWorld consumes sampled robot surface-point trajectories. This module
provides a strict additive bridge between those representations without claiming
that a gripper-only Baxter representation is equivalent to PointWorld's released
DROID or BEHAVIOR robot inputs.

The caller owns mesh loading and deterministic surface sampling. Prob4D binds the
resulting template, tracker calibration, pose stream, and timestamp association
by SHA-256 before transforming the fixed surface points through time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.integer[Any]]
StorageDType = Literal["float32", "float64"]

DUAL_GRIPPER_ACTION_NPZ_SCHEMA: Final = (
    "prob4d.dual-gripper-surface-action-window-npz"
)
DUAL_GRIPPER_ACTION_NPZ_VERSION: Final = 1
DUAL_GRIPPER_ACTION_STORAGE_DTYPES: Final[tuple[StorageDType, ...]] = (
    "float32",
    "float64",
)
DUAL_GRIPPER_ACTION_SEMANTICS: Final = (
    "dual-rigid-gripper-surface-points-from-wxyz-tracker-poses-v1"
)
DUAL_GRIPPER_POINT_IDENTITY_SEMANTICS: Final = (
    "arm-template-calibration-scoped-surface-point-hash-v1"
)
DUAL_GRIPPER_COORDINATE_SEMANTICS: Final = "metric-world-frame-v1"
DUAL_GRIPPER_ARM_ORDER: Final[tuple[str, str]] = ("right", "left")

_REQUIRED_MEMBERS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "storage_dtype",
        "action_id",
        "frame_indices",
        "point_ids",
        "template_point_indices",
        "arm_ids",
        "robot_positions",
        "robot_normals",
        "robot_colors",
        "robot_exists",
        "gripper_open",
        "template_id",
        "tracker_calibration_id",
        "pose_stream_id",
        "timestamp_association_id",
        "action_semantics",
        "point_identity_semantics",
        "coordinate_semantics",
    }
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _strict_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be one nonempty literal string")
    return value


def _strict_sha256(value: Any, *, name: str) -> str:
    result = _strict_text(value, name=name)
    if len(result) != 64 or result != result.lower():
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(result)
    except ValueError as error:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest") from error
    if len(decoded) != 32:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _storage_dtype(value: Any) -> StorageDType:
    normalized = str(value)
    if normalized not in DUAL_GRIPPER_ACTION_STORAGE_DTYPES:
        raise ValueError(
            "storage_dtype must be one of "
            + ", ".join(DUAL_GRIPPER_ACTION_STORAGE_DTYPES)
        )
    return cast(StorageDType, normalized)


def _numpy_dtype(value: StorageDType) -> np.dtype[Any]:
    return np.dtype(np.float32 if value == "float32" else np.float64)


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be one scalar string")
    return _strict_text(str(value.item()), name=name)


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be one scalar integer")
    return int(value.item())


def _integer_vector(value: Any, *, name: str, nonempty: bool) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one vector")
    if nonempty and raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain genuine integers")
    if raw.dtype.kind == "u" and raw.size and int(np.max(raw)) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} values must fit in int64")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _unit_normals(value: Any, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if result.ndim != 3 or result.shape[0] != 2 or result.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (2, M, 3)")
    if result.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one point per arm")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{name} entries must be nonzero")
    result /= norms
    return result


def _rotation_matrices_from_wxyz(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    quaternions = np.asarray(value, dtype=dtype).copy()
    if quaternions.ndim != 3 or quaternions.shape[1:] != (2, 4):
        raise ValueError("quaternions_world_from_tracker_wxyz must have shape (T, 2, 4)")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError("quaternions_world_from_tracker_wxyz must be finite")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("quaternions_world_from_tracker_wxyz must be nonzero")
    quaternions /= norms

    w = quaternions[..., 0]
    x = quaternions[..., 1]
    y = quaternions[..., 2]
    z = quaternions[..., 3]
    matrices = np.empty((*quaternions.shape[:2], 3, 3), dtype=dtype)
    matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[..., 0, 1] = 2.0 * (x * y - z * w)
    matrices[..., 0, 2] = 2.0 * (x * z + y * w)
    matrices[..., 1, 0] = 2.0 * (x * y + z * w)
    matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[..., 1, 2] = 2.0 * (y * z - x * w)
    matrices[..., 2, 0] = 2.0 * (x * z - y * w)
    matrices[..., 2, 1] = 2.0 * (y * z + x * w)
    matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _surface_point_ids(
    template_id: str,
    tracker_calibration_id: str,
    points_per_arm: int,
) -> np.ndarray:
    point_ids = np.empty(2 * points_per_arm, dtype=np.int64)
    for arm_id, arm_name in enumerate(DUAL_GRIPPER_ARM_ORDER):
        for source_index in range(points_per_arm):
            payload = (
                f"{template_id}\x00{tracker_calibration_id}\x00"
                f"{arm_name}\x00{source_index}"
            ).encode()
            digest = hashlib.sha256(payload).digest()
            point_ids[arm_id * points_per_arm + source_index] = (
                int.from_bytes(digest[:8]) & ((1 << 63) - 1)
            )
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError("dual-gripper surface point identity collision")
    return point_ids


@dataclass(frozen=True, slots=True)
class DualGripperSurfaceActionWindow:
    """One bimanual rigid-surface action trajectory.

    Arm order is always right then left to match PointWorld's bimanual feature
    allocation. Surface identity is stable across action windows only when the
    exact template and tracker-calibration IDs match.
    """

    action_id: str
    frame_indices: IntArray
    point_ids: IntArray
    template_point_indices: IntArray
    arm_ids: IntArray
    robot_positions: FloatArray
    robot_normals: FloatArray
    robot_colors: FloatArray
    robot_exists: BoolArray
    gripper_open: BoolArray
    template_id: str
    tracker_calibration_id: str
    pose_stream_id: str
    timestamp_association_id: str
    action_semantics: str = DUAL_GRIPPER_ACTION_SEMANTICS
    point_identity_semantics: str = DUAL_GRIPPER_POINT_IDENTITY_SEMANTICS
    coordinate_semantics: str = DUAL_GRIPPER_COORDINATE_SEMANTICS
    storage_dtype: StorageDType = "float32"

    def __post_init__(self) -> None:
        action_id = _strict_text(self.action_id, name="action_id")
        storage_dtype = _storage_dtype(self.storage_dtype)
        dtype = _numpy_dtype(storage_dtype)
        frame_indices = _integer_vector(
            self.frame_indices,
            name="frame_indices",
            nonempty=True,
        )
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        point_ids = _integer_vector(self.point_ids, name="point_ids", nonempty=True)
        template_indices = _integer_vector(
            self.template_point_indices,
            name="template_point_indices",
            nonempty=True,
        )
        arm_ids = _integer_vector(self.arm_ids, name="arm_ids", nonempty=True)
        point_count = len(point_ids)
        if len(template_indices) != point_count or len(arm_ids) != point_count:
            raise ValueError("point IDs, template indices, and arm IDs must align")
        if len(np.unique(point_ids)) != point_count:
            raise ValueError("point_ids must be unique")
        if set(np.unique(arm_ids)) != {0, 1}:
            raise ValueError("arm_ids must contain both right=0 and left=1")
        right_count = int(np.sum(arm_ids == 0))
        left_count = int(np.sum(arm_ids == 1))
        if right_count != left_count:
            raise ValueError("right and left arms must retain equal template support")
        if not np.array_equal(
            template_indices[:right_count],
            np.arange(right_count, dtype=np.int64),
        ) or not np.array_equal(
            template_indices[right_count:],
            np.arange(left_count, dtype=np.int64),
        ):
            raise ValueError("template_point_indices must be canonical within each arm")
        if not np.all(arm_ids[:right_count] == 0) or not np.all(arm_ids[right_count:] == 1):
            raise ValueError("point order must be right arm followed by left arm")

        expected = (len(frame_indices), point_count)
        positions = np.asarray(self.robot_positions, dtype=dtype).copy()
        normals = np.asarray(self.robot_normals, dtype=dtype).copy()
        colors = np.asarray(self.robot_colors, dtype=dtype).copy()
        if positions.shape != (*expected, 3):
            raise ValueError("robot_positions must have shape (T, N, 3)")
        if normals.shape != positions.shape:
            raise ValueError("robot_normals must match robot_positions")
        if colors.shape != positions.shape:
            raise ValueError("robot_colors must match robot_positions")
        if not np.all(np.isfinite(positions)):
            raise ValueError("robot_positions must be finite")
        if not np.all(np.isfinite(normals)):
            raise ValueError("robot_normals must be finite")
        normal_norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        if np.any(normal_norms <= np.finfo(np.float64).eps):
            raise ValueError("robot_normals must be nonzero")
        normals /= normal_norms
        if not np.all(np.isfinite(colors)) or np.any((colors < 0.0) | (colors > 1.0)):
            raise ValueError("robot_colors must be finite and lie in [0, 1]")

        raw_exists = np.asarray(self.robot_exists)
        raw_open = np.asarray(self.gripper_open)
        if raw_exists.dtype != np.dtype(bool) or raw_exists.shape != expected:
            raise TypeError("robot_exists must be bool with shape (T, N)")
        if raw_open.dtype != np.dtype(bool) or raw_open.shape != (len(frame_indices), 2):
            raise TypeError("gripper_open must be bool with shape (T, 2)")
        if not np.all(raw_exists):
            raise ValueError("version 1 requires complete rigid-template support")

        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "storage_dtype", storage_dtype)
        object.__setattr__(
            self,
            "template_id",
            _strict_sha256(self.template_id, name="template_id"),
        )
        object.__setattr__(
            self,
            "tracker_calibration_id",
            _strict_sha256(
                self.tracker_calibration_id,
                name="tracker_calibration_id",
            ),
        )
        object.__setattr__(
            self,
            "pose_stream_id",
            _strict_sha256(self.pose_stream_id, name="pose_stream_id"),
        )
        object.__setattr__(
            self,
            "timestamp_association_id",
            _strict_sha256(
                self.timestamp_association_id,
                name="timestamp_association_id",
            ),
        )
        for field_name in (
            "action_semantics",
            "point_identity_semantics",
            "coordinate_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_text(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(self, "frame_indices", _readonly(frame_indices))
        object.__setattr__(self, "point_ids", _readonly(point_ids))
        object.__setattr__(
            self,
            "template_point_indices",
            _readonly(template_indices),
        )
        object.__setattr__(self, "arm_ids", _readonly(arm_ids))
        object.__setattr__(self, "robot_positions", _readonly(positions))
        object.__setattr__(self, "robot_normals", _readonly(normals))
        object.__setattr__(self, "robot_colors", _readonly(colors))
        object.__setattr__(self, "robot_exists", _readonly(raw_exists))
        object.__setattr__(self, "gripper_open", _readonly(raw_open))

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    @property
    def point_count(self) -> int:
        return len(self.point_ids)

    @property
    def points_per_arm(self) -> int:
        return self.point_count // 2

    def pointworld_sample(self) -> dict[str, object]:
        """Return the released PointWorld sample fields owned by this artifact."""

        return {
            "robot_flows": np.array(self.robot_positions, copy=True),
            "robot_normals": np.array(self.robot_normals, copy=True),
            "robot_colors": np.array(self.robot_colors, copy=True),
            "right_gripper_open": np.asarray(
                self.gripper_open[:, 0:1],
                dtype=np.float32,
            ),
            "left_gripper_open": np.asarray(
                self.gripper_open[:, 1:2],
                dtype=np.float32,
            ),
            "__has_right_gripper__": True,
            "__has_left_gripper__": True,
        }

    def to_npz(
        self,
        path: str | Path,
        *,
        storage_dtype: StorageDType | None = None,
    ) -> None:
        selected_dtype = (
            self.storage_dtype
            if storage_dtype is None
            else _storage_dtype(storage_dtype)
        )
        dtype = _numpy_dtype(selected_dtype)
        np.savez_compressed(
            Path(path),
            schema_name=np.asarray(DUAL_GRIPPER_ACTION_NPZ_SCHEMA),
            schema_version=np.asarray(DUAL_GRIPPER_ACTION_NPZ_VERSION, dtype=np.int64),
            storage_dtype=np.asarray(selected_dtype),
            action_id=np.asarray(self.action_id),
            frame_indices=np.asarray(self.frame_indices, dtype=np.int64),
            point_ids=np.asarray(self.point_ids, dtype=np.int64),
            template_point_indices=np.asarray(
                self.template_point_indices,
                dtype=np.int64,
            ),
            arm_ids=np.asarray(self.arm_ids, dtype=np.int64),
            robot_positions=np.asarray(self.robot_positions, dtype=dtype),
            robot_normals=np.asarray(self.robot_normals, dtype=dtype),
            robot_colors=np.asarray(self.robot_colors, dtype=dtype),
            robot_exists=np.asarray(self.robot_exists, dtype=bool),
            gripper_open=np.asarray(self.gripper_open, dtype=bool),
            template_id=np.asarray(self.template_id),
            tracker_calibration_id=np.asarray(self.tracker_calibration_id),
            pose_stream_id=np.asarray(self.pose_stream_id),
            timestamp_association_id=np.asarray(self.timestamp_association_id),
            action_semantics=np.asarray(self.action_semantics),
            point_identity_semantics=np.asarray(self.point_identity_semantics),
            coordinate_semantics=np.asarray(self.coordinate_semantics),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> DualGripperSurfaceActionWindow:
        with np.load(Path(path), allow_pickle=False) as data:
            files = set(data.files)
            missing = sorted(_REQUIRED_MEMBERS - files)
            extra = sorted(files - _REQUIRED_MEMBERS)
            if missing or extra:
                raise ValueError(
                    "dual-gripper action archive fields changed; "
                    f"missing={missing}, extra={extra}"
                )
            schema = _scalar_text(data["schema_name"], name="schema_name")
            version = _scalar_integer(data["schema_version"], name="schema_version")
            if schema != DUAL_GRIPPER_ACTION_NPZ_SCHEMA:
                raise ValueError("unsupported dual-gripper action archive schema")
            if version != DUAL_GRIPPER_ACTION_NPZ_VERSION:
                raise ValueError("unsupported dual-gripper action archive version")
            storage_dtype = _storage_dtype(
                _scalar_text(data["storage_dtype"], name="storage_dtype")
            )
            expected_dtype = _numpy_dtype(storage_dtype)
            for field in ("robot_positions", "robot_normals", "robot_colors"):
                if data[field].dtype != expected_dtype:
                    raise ValueError(f"{field} dtype disagrees with storage_dtype")
            for field in (
                "frame_indices",
                "point_ids",
                "template_point_indices",
                "arm_ids",
            ):
                if data[field].dtype != np.dtype(np.int64):
                    raise ValueError(f"{field} must use int64")
            for field in ("robot_exists", "gripper_open"):
                if data[field].dtype != np.dtype(bool):
                    raise ValueError(f"{field} must use bool")

            return cls(
                action_id=_scalar_text(data["action_id"], name="action_id"),
                frame_indices=data["frame_indices"],
                point_ids=data["point_ids"],
                template_point_indices=data["template_point_indices"],
                arm_ids=data["arm_ids"],
                robot_positions=data["robot_positions"],
                robot_normals=data["robot_normals"],
                robot_colors=data["robot_colors"],
                robot_exists=data["robot_exists"],
                gripper_open=data["gripper_open"],
                template_id=_scalar_text(data["template_id"], name="template_id"),
                tracker_calibration_id=_scalar_text(
                    data["tracker_calibration_id"],
                    name="tracker_calibration_id",
                ),
                pose_stream_id=_scalar_text(
                    data["pose_stream_id"],
                    name="pose_stream_id",
                ),
                timestamp_association_id=_scalar_text(
                    data["timestamp_association_id"],
                    name="timestamp_association_id",
                ),
                action_semantics=_scalar_text(
                    data["action_semantics"],
                    name="action_semantics",
                ),
                point_identity_semantics=_scalar_text(
                    data["point_identity_semantics"],
                    name="point_identity_semantics",
                ),
                coordinate_semantics=_scalar_text(
                    data["coordinate_semantics"],
                    name="coordinate_semantics",
                ),
                storage_dtype=storage_dtype,
            )


def dual_gripper_surface_action_from_tracker_poses(
    *,
    action_id: str,
    frame_indices: Any,
    surface_points_tracker: Any,
    surface_normals_tracker: Any,
    positions_world_from_tracker: Any,
    quaternions_world_from_tracker_wxyz: Any,
    gripper_open: Any,
    template_id: str,
    tracker_calibration_id: str,
    pose_stream_id: str,
    timestamp_association_id: str,
    storage_dtype: StorageDType = "float32",
) -> DualGripperSurfaceActionWindow:
    """Transform fixed right/left gripper templates through timestamped poses.

    ``surface_points_tracker`` and ``surface_normals_tracker`` use shape
    ``(2, M, 3)`` in right-then-left tracker frames. Quaternions use Flat'n'Fold's
    documented parser order ``[w, x, y, z]`` and are deterministically normalized
    before use, accommodating the public parser's three-decimal rounding.
    """

    selected_dtype = _storage_dtype(storage_dtype)
    dtype = _numpy_dtype(selected_dtype)
    frames = _integer_vector(frame_indices, name="frame_indices", nonempty=True)
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    template = np.asarray(surface_points_tracker, dtype=dtype).copy()
    if template.ndim != 3 or template.shape[0] != 2 or template.shape[-1] != 3:
        raise ValueError("surface_points_tracker must have shape (2, M, 3)")
    if template.shape[1] == 0 or not np.all(np.isfinite(template)):
        raise ValueError("surface_points_tracker must be finite and nonempty")
    template_normals = _unit_normals(
        surface_normals_tracker,
        name="surface_normals_tracker",
        dtype=dtype,
    )
    if template_normals.shape != template.shape:
        raise ValueError("surface points and normals must align")

    translations = np.asarray(positions_world_from_tracker, dtype=dtype).copy()
    if translations.shape != (len(frames), 2, 3):
        raise ValueError("positions_world_from_tracker must have shape (T, 2, 3)")
    if not np.all(np.isfinite(translations)):
        raise ValueError("positions_world_from_tracker must be finite")
    rotations = _rotation_matrices_from_wxyz(
        quaternions_world_from_tracker_wxyz,
        dtype=dtype,
    )
    if rotations.shape[:2] != translations.shape[:2]:
        raise ValueError("quaternion and translation trajectories must align")
    raw_open = np.asarray(gripper_open)
    if raw_open.dtype != np.dtype(bool) or raw_open.shape != (len(frames), 2):
        raise TypeError("gripper_open must be bool with shape (T, 2)")

    transformed_points = np.einsum("taij,amj->tami", rotations, template)
    transformed_points += translations[:, :, None, :]
    transformed_normals = np.einsum("taij,amj->tami", rotations, template_normals)
    transformed_normals /= np.linalg.norm(
        transformed_normals,
        axis=-1,
        keepdims=True,
    )
    points_per_arm = template.shape[1]
    robot_positions = transformed_points.reshape(len(frames), 2 * points_per_arm, 3)
    robot_normals = transformed_normals.reshape(len(frames), 2 * points_per_arm, 3)
    robot_colors = np.empty_like(robot_positions)
    robot_colors[..., 0] = 1.0
    robot_colors[..., 1] = 0.0
    robot_colors[..., 2] = 1.0
    point_ids = _surface_point_ids(
        _strict_sha256(template_id, name="template_id"),
        _strict_sha256(
            tracker_calibration_id,
            name="tracker_calibration_id",
        ),
        points_per_arm,
    )
    template_indices = np.tile(
        np.arange(points_per_arm, dtype=np.int64),
        2,
    )
    arm_ids = np.repeat(np.arange(2, dtype=np.int64), points_per_arm)

    return DualGripperSurfaceActionWindow(
        action_id=action_id,
        frame_indices=frames,
        point_ids=point_ids,
        template_point_indices=template_indices,
        arm_ids=arm_ids,
        robot_positions=robot_positions,
        robot_normals=robot_normals,
        robot_colors=robot_colors,
        robot_exists=np.ones((len(frames), 2 * points_per_arm), dtype=bool),
        gripper_open=raw_open,
        template_id=template_id,
        tracker_calibration_id=tracker_calibration_id,
        pose_stream_id=pose_stream_id,
        timestamp_association_id=timestamp_association_id,
        storage_dtype=selected_dtype,
    )


__all__ = [
    "DUAL_GRIPPER_ACTION_NPZ_SCHEMA",
    "DUAL_GRIPPER_ACTION_NPZ_VERSION",
    "DUAL_GRIPPER_ACTION_SEMANTICS",
    "DUAL_GRIPPER_ARM_ORDER",
    "DUAL_GRIPPER_COORDINATE_SEMANTICS",
    "DUAL_GRIPPER_POINT_IDENTITY_SEMANTICS",
    "DualGripperSurfaceActionWindow",
    "dual_gripper_surface_action_from_tracker_poses",
]
