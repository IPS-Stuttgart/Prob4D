"""Bounded-memory export for existing fused-prediction NPZ artifacts.

The ordinary :func:`prob4d.io.save_fused_prediction` path is convenient for
small and medium sequences, but NumPy materializes full float32 point/flow
copies and full packed covariance arrays before writing. This module preserves
the same field schema while staging large NPY members in bounded chunks and
atomically assembling the final NPZ archive.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import DTypeLike

from ._immutable_json import plain_json
from .fusion import FusedSequence
from .io import FusedPredictionMetadata, fusion_covariance_semantics

DEFAULT_EXPORT_CHUNK_ROWS: Final = 262_144
_PACKED_COVARIANCE_WIDTH: Final = 6


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _normalized_destination(path: str | Path) -> Path:
    destination = Path(path)
    if not str(destination).endswith(".npz"):
        destination = Path(str(destination) + ".npz")
    return destination


def _write_small_array(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _stage_cast_array(
    path: Path,
    value: np.ndarray,
    *,
    dtype: DTypeLike,
    chunk_rows: int,
) -> None:
    source = np.asarray(value)
    target = open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=source.shape,
        fortran_order=False,
    )
    source_rows = source.reshape(-1)
    target_rows = target.reshape(-1)
    for start in range(0, source_rows.size, chunk_rows):
        stop = min(start + chunk_rows, source_rows.size)
        target_rows[start:stop] = source_rows[start:stop]
    target.flush()
    del target
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _stage_packed_covariance(
    path: Path,
    covariance: np.ndarray,
    *,
    chunk_rows: int,
) -> None:
    source = np.asarray(covariance)
    if source.shape[-2:] != (3, 3):
        raise ValueError("covariance must end in shape (3, 3)")
    packed_shape = source.shape[:-2] + (_PACKED_COVARIANCE_WIDTH,)
    target = open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=packed_shape,
        fortran_order=False,
    )
    source_rows = source.reshape(-1, 3, 3)
    target_rows = target.reshape(-1, _PACKED_COVARIANCE_WIDTH)
    for start in range(0, source_rows.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source_rows.shape[0])
        values = source_rows[start:stop]
        output = target_rows[start:stop]
        output[:, 0] = values[:, 0, 0]
        output[:, 1] = values[:, 0, 1]
        output[:, 2] = values[:, 0, 2]
        output[:, 3] = values[:, 1, 1]
        output[:, 4] = values[:, 1, 2]
        output[:, 5] = values[:, 2, 2]
    target.flush()
    del target
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _archive_members(
    destination: Path,
    members: Mapping[str, Path],
    *,
    compressed: bool,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=compression,
            allowZip64=True,
        ) as archive:
            for field, member in members.items():
                archive.write(member, arcname=f"{field}.npy")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_fused_prediction_bounded(
    path: str | Path,
    sequence: FusedSequence,
    *,
    method_id: str,
    fusion_method: str,
    include_covariance: bool = True,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = False,
    chunk_rows: int = DEFAULT_EXPORT_CHUNK_ROWS,
) -> FusedPredictionMetadata:
    """Atomically write the existing fused-prediction schema with bounded RAM.

    Large point, flow, and covariance fields are staged as NPY files in the
    destination directory. Casting and symmetric-covariance packing operate on
    at most ``chunk_rows`` scalar values or covariance matrices at a time. The
    final archive remains readable by :func:`prob4d.io.load_fused_prediction` and
    contains the same field names, dtypes, shapes, and values as the ordinary
    writer. ZIP container bytes are not promised to match another writer.

    Temporary disk usage can approach the sum of the staged NPY members and the
    final NPZ archive. Existing destinations are replaced only after every
    member and the complete archive have been written successfully.
    """

    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    if not isinstance(sequence, FusedSequence):
        raise TypeError("sequence must be a validated FusedSequence")

    covariance_semantics, correlation_assumption = fusion_covariance_semantics(fusion_method)
    artifact_metadata = FusedPredictionMetadata(
        method_id=method_id,
        fusion_method=fusion_method,
        covariance_semantics=covariance_semantics,
        correlation_assumption=correlation_assumption,
        metadata={} if metadata is None else metadata,
    )
    destination = _normalized_destination(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.members.",
            dir=destination.parent,
        )
    )

    members: dict[str, Path] = {}

    def stage_small(field: str, value: np.ndarray) -> None:
        member = workspace / f"{field}.npy"
        _write_small_array(member, value)
        members[field] = member

    def stage_cast(field: str, value: np.ndarray, dtype: DTypeLike) -> None:
        member = workspace / f"{field}.npy"
        _stage_cast_array(
            member,
            value,
            dtype=dtype,
            chunk_rows=chunk_rows,
        )
        members[field] = member

    def stage_covariance(field: str, value: np.ndarray) -> None:
        member = workspace / f"{field}.npy"
        _stage_packed_covariance(member, value, chunk_rows=chunk_rows)
        members[field] = member

    try:
        stage_small("artifact_schema", np.asarray(artifact_metadata.schema_name))
        stage_small(
            "artifact_version",
            np.asarray(artifact_metadata.schema_version, dtype=np.int64),
        )
        stage_small("method_id", np.asarray(artifact_metadata.method_id))
        stage_small("fusion_method", np.asarray(artifact_metadata.fusion_method))
        stage_small(
            "covariance_semantics",
            np.asarray(artifact_metadata.covariance_semantics),
        )
        stage_small(
            "correlation_assumption",
            np.asarray(artifact_metadata.correlation_assumption),
        )
        stage_small(
            "artifact_metadata_json",
            np.asarray(
                json.dumps(
                    plain_json(artifact_metadata.metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        )
        stage_cast("point_map", sequence.point_map, np.dtype(np.float32))
        stage_cast("valid_mask", sequence.valid_mask, np.dtype(bool))
        stage_cast("frame_indices", sequence.frame_indices, np.dtype(np.int64))

        if include_covariance:
            stage_covariance("point_covariance_packed", sequence.point_covariance)
            stage_cast(
                "contributors",
                sequence.contributors,
                sequence.contributors.dtype,
            )

        if sequence.scene_flow is not None:
            if sequence.deform_mask is None:
                raise ValueError("scene-flow deformation mask is missing")
            stage_cast("scene_flow", sequence.scene_flow, np.dtype(np.float32))
            stage_cast("deform_mask", sequence.deform_mask, np.dtype(bool))
            if include_covariance:
                if sequence.flow_covariance is None:
                    raise ValueError("scene-flow covariance is missing from fused sequence")
                stage_covariance(
                    "flow_covariance_packed",
                    sequence.flow_covariance,
                )

        _archive_members(destination, members, compressed=compressed)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return artifact_metadata


__all__ = ["DEFAULT_EXPORT_CHUNK_ROWS", "save_fused_prediction_bounded"]
