"""Bounded-memory canonical windows from direct CUT3R XYZ point maps."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ._cut3r_direct_source import (
    _direct_source_tree_byte_count,
    _validated_direct_source_members,
)
from ._cut3r_limits import Cut3RImportLimits, _dense_array_byte_count, _validate_camera
from ._cut3r_source import _SourceMemberDescriptor, _load_camera, _load_npy
from ._cut3r_window import _close_memmap
from .data import DenseStorageDType
from .prediction_store import MMapPredictionWindow


def _validated_direct_grid_shape(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    limits: Cut3RImportLimits,
    expected_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("CUT3R direct point maps must have shape (H, W, 3)")
    if confidence.shape != points.shape[:-1]:
        raise ValueError("CUT3R confidence must match the direct point-map grid")
    if points.dtype.kind not in {"f", "i", "u"} or confidence.dtype.kind not in {
        "f",
        "i",
        "u",
    }:
        raise ValueError("CUT3R direct points and confidence must be real numeric arrays")
    height = int(points.shape[0])
    width = int(points.shape[1])
    if height > limits.max_height:
        raise ValueError(f"CUT3R frame height {height} exceeds max_height={limits.max_height}")
    if width > limits.max_width:
        raise ValueError(f"CUT3R frame width {width} exceeds max_width={limits.max_width}")
    shape = (height, width)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("CUT3R direct point maps must share one spatial grid")
    return shape


def _direct_world_points(
    points: np.ndarray,
    confidence: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_camera(pose, intrinsics)
    points64 = np.asarray(points, dtype=np.float64)
    confidence64 = np.asarray(confidence, dtype=np.float64)
    valid = (
        np.all(np.isfinite(points64), axis=-1)
        & np.isfinite(confidence64)
        & (points64[..., 2] > 0.0)
        & (confidence64 >= confidence_threshold)
    )
    world = np.einsum("ij,hwj->hwi", pose[:3, :3], points64) + pose[:3, 3]
    valid &= np.all(np.isfinite(world), axis=-1)
    world[~valid] = 0.0
    return world, valid


@contextmanager
def _canonical_direct_window(
    root: Path,
    *,
    frame_start: int,
    window_id: str,
    confidence_threshold: float,
    storage_dtype: DenseStorageDType,
    limits: Cut3RImportLimits,
) -> Iterator[
    tuple[
        MMapPredictionWindow,
        tuple[_SourceMemberDescriptor, ...],
        int,
        int,
    ]
]:
    point_members, confidence_members, camera_members = _validated_direct_source_members(root)
    frame_count = len(point_members)
    if frame_count > limits.max_frames:
        raise ValueError(f"CUT3R frame count {frame_count} exceeds max_frames={limits.max_frames}")
    source_bytes = _direct_source_tree_byte_count(
        point_members,
        confidence_members,
        camera_members,
    )
    if source_bytes > limits.max_source_bytes:
        raise ValueError(
            "CUT3R direct source tree byte count "
            f"{source_bytes} exceeds max_source_bytes={limits.max_source_bytes}"
        )

    workspace = Path(tempfile.mkdtemp(prefix=".prob4d-cut3r-direct-import."))
    point_writer: np.memmap | None = None
    mask_writer: np.memmap | None = None
    frame_writer: np.memmap | None = None
    read_only_maps: list[np.ndarray] = []
    try:
        descriptors: list[_SourceMemberDescriptor] = []
        shape: tuple[int, int] | None = None
        dense_bytes = 0
        any_valid = False
        dense_dtype = np.float32 if storage_dtype == "float32" else np.float64
        point_path = workspace / "point_map.npy"
        mask_path = workspace / "valid_mask.npy"
        frame_path = workspace / "frame_indices.npy"

        for index in range(frame_count):
            points, point_descriptor = _load_npy(point_members[index], label="direct points")
            confidence, confidence_descriptor = _load_npy(
                confidence_members[index],
                label="confidence",
            )
            try:
                frame_shape = _validated_direct_grid_shape(
                    points,
                    confidence,
                    limits=limits,
                    expected_shape=shape,
                )
                if shape is None:
                    shape = frame_shape
                    dense_bytes = _dense_array_byte_count(
                        frame_count,
                        shape[0],
                        shape[1],
                        storage_dtype,
                    )
                    if dense_bytes > limits.max_dense_bytes:
                        raise ValueError(
                            "CUT3R direct dense array byte count "
                            f"{dense_bytes} exceeds max_dense_bytes={limits.max_dense_bytes}"
                        )
                    point_writer = np.lib.format.open_memmap(
                        point_path,
                        mode="w+",
                        dtype=dense_dtype,
                        shape=(frame_count, shape[0], shape[1], 3),
                    )
                    mask_writer = np.lib.format.open_memmap(
                        mask_path,
                        mode="w+",
                        dtype=bool,
                        shape=(frame_count, shape[0], shape[1]),
                    )
                    frame_writer = np.lib.format.open_memmap(
                        frame_path,
                        mode="w+",
                        dtype=np.int64,
                        shape=(frame_count,),
                    )
                    frame_writer[:] = np.arange(
                        frame_start,
                        frame_start + frame_count,
                        dtype=np.int64,
                    )

                pose, intrinsics, camera_descriptor = _load_camera(camera_members[index])
                world, valid = _direct_world_points(
                    points,
                    confidence,
                    pose,
                    intrinsics,
                    confidence_threshold=confidence_threshold,
                )
                assert point_writer is not None
                assert mask_writer is not None
                point_writer[index] = world
                mask_writer[index] = valid
                any_valid = any_valid or bool(np.any(valid))
                descriptors.extend(
                    (point_descriptor, confidence_descriptor, camera_descriptor)
                )
            finally:
                _close_memmap(points)
                _close_memmap(confidence)

        if not any_valid:
            raise ValueError(
                "CUT3R direct output contains no point above the frozen support threshold"
            )
        assert shape is not None
        assert point_writer is not None
        assert mask_writer is not None
        assert frame_writer is not None
        for active_writer in (point_writer, mask_writer, frame_writer):
            active_writer.flush()
            _close_memmap(active_writer)
        point_writer = None
        mask_writer = None
        frame_writer = None

        described_source_bytes = sum(member["byte_count"] for member in descriptors)
        if described_source_bytes > limits.max_source_bytes:
            raise ValueError(
                "CUT3R described direct source byte count "
                f"{described_source_bytes} exceeds max_source_bytes={limits.max_source_bytes}"
            )

        frame_indices = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        point_map = np.load(point_path, mmap_mode="r", allow_pickle=False)
        valid_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        read_only_maps.extend((frame_indices, point_map, valid_mask))
        window = MMapPredictionWindow(
            window_id=window_id,
            frame_indices=frame_indices,
            point_map=point_map,
            valid_mask=valid_mask,
            dense_storage_dtype=storage_dtype,
        )
        yield window, tuple(descriptors), described_source_bytes, dense_bytes
    finally:
        for pending_writer in (point_writer, mask_writer, frame_writer):
            if pending_writer is not None:
                _close_memmap(pending_writer)
        for value in read_only_maps:
            _close_memmap(value)
        shutil.rmtree(workspace, ignore_errors=True)


__all__ = ["_canonical_direct_window", "_direct_world_points"]
