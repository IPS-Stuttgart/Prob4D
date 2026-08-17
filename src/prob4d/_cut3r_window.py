"""Bounded-memory canonical prediction windows for CUT3R imports."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

from ._atomic_file import publish_temporary_file
from ._cut3r_limits import (
    Cut3RImportLimits,
    _dense_array_byte_count,
    _unproject_world_points,
    _validated_grid_shape,
)
from ._cut3r_source import (
    _load_camera,
    _load_npy,
    _source_tree_byte_count,
    _SourceMemberDescriptor,
    _validated_source_members,
)
from .data import DenseStorageDType, PredictionWindow
from .prediction_store import MMapPredictionWindow


def _close_memmap(value: np.ndarray) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _load_read_only_npy(path: Path, *, name: str) -> np.ndarray:
    try:
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot reopen canonical CUT3R {name}") from error
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"canonical CUT3R {name} must be one NPY array")
    return loaded


@contextmanager
def _canonical_window(
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
    depth_members, confidence_members, camera_members = _validated_source_members(root)
    frame_count = len(depth_members)
    if frame_count > limits.max_frames:
        raise ValueError(f"CUT3R frame count {frame_count} exceeds max_frames={limits.max_frames}")
    source_bytes = _source_tree_byte_count((depth_members, confidence_members, camera_members))
    if source_bytes > limits.max_source_bytes:
        raise ValueError(
            "CUT3R source tree byte count "
            f"{source_bytes} exceeds max_source_bytes={limits.max_source_bytes}"
        )

    workspace = Path(tempfile.mkdtemp(prefix=".prob4d-cut3r-import."))
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
            depth, depth_descriptor = _load_npy(depth_members[index], label="depth")
            confidence, confidence_descriptor = _load_npy(
                confidence_members[index],
                label="confidence",
            )
            try:
                frame_shape = _validated_grid_shape(
                    depth,
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
                            "CUT3R dense array byte count "
                            f"{dense_bytes} exceeds max_dense_bytes="
                            f"{limits.max_dense_bytes}"
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
                world, valid = _unproject_world_points(
                    depth,
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
                descriptors.extend((depth_descriptor, confidence_descriptor, camera_descriptor))
            finally:
                _close_memmap(depth)
                _close_memmap(confidence)

        if not any_valid:
            raise ValueError("CUT3R output contains no point above the frozen support threshold")
        assert shape is not None
        assert point_writer is not None
        assert mask_writer is not None
        assert frame_writer is not None
        for writer in (point_writer, mask_writer, frame_writer):
            writer.flush()
            _close_memmap(writer)
        point_writer = None
        mask_writer = None
        frame_writer = None

        described_source_bytes = sum(member["byte_count"] for member in descriptors)
        if described_source_bytes > limits.max_source_bytes:
            raise ValueError(
                "CUT3R described source byte count "
                f"{described_source_bytes} exceeds max_source_bytes={limits.max_source_bytes}"
            )

        frame_indices = _load_read_only_npy(frame_path, name="frame indices")
        point_map = _load_read_only_npy(point_path, name="point map")
        valid_mask = _load_read_only_npy(mask_path, name="valid mask")
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
        for writer in (point_writer, mask_writer, frame_writer):
            if writer is not None:
                _close_memmap(writer)
        for value in read_only_maps:
            _close_memmap(value)
        shutil.rmtree(workspace, ignore_errors=True)


def _windows_equal(first: PredictionWindow, second: PredictionWindow) -> bool:
    return (
        first.window_id == second.window_id
        and first.dense_storage_dtype == second.dense_storage_dtype
        and np.array_equal(first.frame_indices, second.frame_indices)
        and np.array_equal(first.point_map, second.point_map)
        and np.array_equal(first.valid_mask, second.valid_mask)
        and first.scene_flow is None
        and second.scene_flow is None
        and first.ray_directions is None
        and second.ray_directions is None
    )


def _write_window_atomically(path: Path, window: PredictionWindow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        window.to_npz(temporary)
        try:
            publish_temporary_file(temporary, path, overwrite=False)
        except FileExistsError:
            existing = PredictionWindow.from_npz(
                path,
                dense_storage_dtype=window.dense_storage_dtype,
            )
            if not _windows_equal(existing, window):
                raise ValueError(
                    f"refusing to replace different canonical CUT3R payload {path.name!r}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _relative_member(path: Path, *, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            "canonical CUT3R payload must lie inside the manifest directory"
        ) from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("canonical CUT3R payload path is not confined")
    return PurePosixPath(*relative.parts).as_posix()
