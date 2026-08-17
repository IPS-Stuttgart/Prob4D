"""Resource limits and per-frame geometry for CUT3R imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from ._strict_json import require_exact_integer
from .data import DenseStorageDType

_GIB: Final = 1024**3


@dataclass(frozen=True)
class Cut3RImportLimits:
    """Fail-closed resource limits for one CUT3R source-tree import."""

    max_frames: int = 4096
    max_height: int = 4096
    max_width: int = 4096
    max_source_bytes: int = 128 * _GIB
    max_dense_bytes: int = 128 * _GIB

    def __post_init__(self) -> None:
        for name in (
            "max_frames",
            "max_height",
            "max_width",
            "max_source_bytes",
            "max_dense_bytes",
        ):
            require_exact_integer(getattr(self, name), name=name, minimum=1)


DEFAULT_CUT3R_IMPORT_LIMITS: Final = Cut3RImportLimits()


def _validate_camera(pose: np.ndarray, intrinsics: np.ndarray) -> None:
    if pose.shape != (4, 4):
        raise ValueError("CUT3R camera pose must have shape (4, 4)")
    if intrinsics.shape != (3, 3):
        raise ValueError("CUT3R camera intrinsics must have shape (3, 3)")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("CUT3R camera pose and intrinsics must be finite")
    if not np.allclose(pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("CUT3R camera pose must be a homogeneous rigid transform")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError("CUT3R camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("CUT3R camera rotation must be proper")
    if not np.allclose(intrinsics[2], np.asarray([0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("CUT3R camera intrinsics must use the standard final row")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("CUT3R focal lengths must be strictly positive")


def _validated_grid_shape(
    depth: np.ndarray,
    confidence: np.ndarray,
    *,
    limits: Cut3RImportLimits,
    expected_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if depth.ndim != 2:
        raise ValueError("CUT3R depth members must have shape (H, W)")
    if confidence.shape != depth.shape:
        raise ValueError("CUT3R confidence must match the depth grid")
    if depth.dtype.kind not in {"f", "i", "u"} or confidence.dtype.kind not in {
        "f",
        "i",
        "u",
    }:
        raise ValueError("CUT3R depth and confidence must be real numeric arrays")
    height, width = cast(tuple[int, int], depth.shape)
    if height > limits.max_height:
        raise ValueError(
            f"CUT3R frame height {height} exceeds max_height={limits.max_height}"
        )
    if width > limits.max_width:
        raise ValueError(f"CUT3R frame width {width} exceeds max_width={limits.max_width}")
    shape = (height, width)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("CUT3R frames must share one spatial prediction grid")
    return shape


def _unproject_world_points(
    depth: np.ndarray,
    confidence: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_camera(pose, intrinsics)

    depth64 = np.asarray(depth, dtype=np.float64)
    confidence64 = np.asarray(confidence, dtype=np.float64)
    valid = (
        np.isfinite(depth64)
        & np.isfinite(confidence64)
        & (depth64 > 0.0)
        & (confidence64 >= confidence_threshold)
    )

    height, width = depth64.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    homogeneous_pixels = np.stack(
        (columns, rows, np.ones((height, width), dtype=np.float64)),
        axis=-1,
    )
    try:
        camera_rays = np.linalg.solve(
            intrinsics,
            homogeneous_pixels.reshape(-1, 3).T,
        ).T.reshape(height, width, 3)
    except np.linalg.LinAlgError as error:
        raise ValueError("CUT3R camera intrinsics must be nonsingular") from error
    camera_points = camera_rays * depth64[..., None]
    world = np.einsum("ij,hwj->hwi", pose[:3, :3], camera_points) + pose[:3, 3]
    valid &= np.all(np.isfinite(world), axis=-1)
    world[~valid] = 0.0
    return world, valid


def _dense_array_byte_count(
    frame_count: int,
    height: int,
    width: int,
    storage_dtype: DenseStorageDType,
) -> int:
    dense_itemsize = np.dtype(np.float32 if storage_dtype == "float32" else np.float64).itemsize
    point_bytes = frame_count * height * width * 3 * dense_itemsize
    mask_bytes = frame_count * height * width * np.dtype(bool).itemsize
    frame_index_bytes = frame_count * np.dtype(np.int64).itemsize
    return point_bytes + mask_bytes + frame_index_bytes


__all__ = [
    "Cut3RImportLimits",
    "DEFAULT_CUT3R_IMPORT_LIMITS",
]
