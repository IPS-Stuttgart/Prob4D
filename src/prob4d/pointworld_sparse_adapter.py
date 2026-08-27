"""Source-only export of PointWorld persistent scene-point trajectories."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from .persistent_point_prediction import (
    PERSISTENT_POINT_WINDOW_NPZ_SCHEMA,
    STORAGE_DTYPES,
    PersistentPointPredictionWindow,
    StorageDType,
)

POINTWORLD_SPARSE_SOURCE_SCHEMA: Final = (
    "prob4d.pointworld-persistent-point-source-npz"
)
POINTWORLD_SPARSE_SOURCE_VERSION: Final = 1
POINTWORLD_UNCERTAINTY_SEMANTICS: Final = (
    "pointworld-normalized-relative-log-variance-v1"
)

_SOURCE_REQUIRED_MEMBERS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "frame_indices",
        "scene_flows",
        "scene_exists",
        "log_var",
        "context_frame_count",
    }
)
_SOURCE_OPTIONAL_MEMBERS: Final = frozenset({"point_ids"})


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be one scalar string")
    return str(value.item())


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be one scalar integer")
    return int(value.item())


def _without_singleton_batch(
    value: object,
    *,
    name: str,
    unbatched_ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == unbatched_ndim:
        return array
    if array.ndim == unbatched_ndim + 1 and array.shape[0] == 1:
        return array[0]
    raise ValueError(
        f"{name} must have {unbatched_ndim} dimensions or one singleton "
        "batch dimension"
    )


def pointworld_output_to_persistent_window(
    *,
    window_id: str,
    frame_indices: object,
    scene_flows: object,
    scene_exists: object,
    log_var: object,
    context_frame_count: int = 1,
    point_ids: object | None = None,
    storage_dtype: StorageDType = "float32",
) -> PersistentPointPredictionWindow:
    """Convert one PointWorld output while preserving seeded point identity.

    PointWorld's released evaluation pipeline samples scene points at the first
    timestep and applies exactly the same selected indices to all timesteps.
    ``scene_flows`` therefore represents absolute trajectories of persistent
    window-local point identities, not a dense image grid.

    ``log_var`` is retained under its released normalized-relative semantics.
    This function does not reinterpret it as calibrated metric covariance.
    """

    trajectory = _without_singleton_batch(
        scene_flows,
        name="scene_flows",
        unbatched_ndim=3,
    )
    exists = _without_singleton_batch(
        scene_exists,
        name="scene_exists",
        unbatched_ndim=2,
    )
    uncertainty = _without_singleton_batch(
        log_var,
        name="log_var",
        unbatched_ndim=3,
    )
    frames = np.asarray(frame_indices)
    if frames.dtype != np.dtype(np.int64) or frames.ndim != 1:
        raise ValueError("frame_indices must be a one-dimensional int64 array")
    if trajectory.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("scene_flows must use float32 or float64")
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("scene_flows must have shape (T, N, 3)")
    if exists.dtype != np.dtype(bool):
        raise ValueError("scene_exists must use bool dtype")
    if exists.shape != trajectory.shape[:2]:
        raise ValueError("scene_exists must have shape (T, N)")
    if uncertainty.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("log_var must use float32 or float64")
    if (
        uncertainty.ndim != 3
        or uncertainty.shape[:2] != trajectory.shape[:2]
        or uncertainty.shape[-1] not in {1, 3}
    ):
        raise ValueError("log_var must have shape (T, N, 1) or (T, N, 3)")
    if frames.size != trajectory.shape[0]:
        raise ValueError("frame_indices length must match PointWorld output time")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("scene_flows must contain only finite values")
    if not np.all(np.isfinite(uncertainty)):
        raise ValueError("log_var must contain only finite values")

    source_point_mask = np.asarray(exists[0], dtype=bool)
    if not np.any(source_point_mask):
        raise ValueError("PointWorld source frame contains no valid scene points")

    point_count = int(trajectory.shape[1])
    if point_ids is None:
        identifiers = np.arange(point_count, dtype=np.int64)
    else:
        identifiers = np.asarray(point_ids)
        if identifiers.dtype != np.dtype(np.int64) or identifiers.ndim != 1:
            raise ValueError("point_ids must be a one-dimensional int64 array")
        if identifiers.size != point_count:
            raise ValueError("point_ids length must match PointWorld point count")
        if np.any(identifiers < 0) or len(set(map(int, identifiers))) != point_count:
            raise ValueError("point_ids must be non-negative and unique")
        identifiers = identifiers.astype(np.int64, copy=False)

    retained_ids = identifiers[source_point_mask]
    retained_trajectory = trajectory[:, source_point_mask]
    retained_exists = exists[:, source_point_mask]
    retained_uncertainty = uncertainty[:, source_point_mask]

    order = np.argsort(retained_ids, kind="stable")
    retained_ids = retained_ids[order]
    retained_trajectory = retained_trajectory[:, order]
    retained_exists = retained_exists[:, order]
    retained_uncertainty = retained_uncertainty[:, order]

    return PersistentPointPredictionWindow(
        window_id=window_id,
        frame_indices=frames,
        point_ids=retained_ids,
        point_trajectory=retained_trajectory,
        valid_mask=retained_exists,
        context_frame_count=context_frame_count,
        uncertainty=retained_uncertainty,
        uncertainty_semantics=POINTWORLD_UNCERTAINTY_SEMANTICS,
        storage_dtype=storage_dtype,
    )


def load_pointworld_source_snapshot(
    path: str | Path,
    *,
    window_id: str,
    storage_dtype: StorageDType = "float32",
) -> PersistentPointPredictionWindow:
    """Load one strict, runtime-exported PointWorld source snapshot."""

    source_path = Path(path)
    if source_path.is_symlink():
        raise ValueError("PointWorld source snapshot is a symbolic link")
    with np.load(source_path, allow_pickle=False) as data:
        files = set(data.files)
        missing = sorted(_SOURCE_REQUIRED_MEMBERS - files)
        extra = sorted(
            files - (_SOURCE_REQUIRED_MEMBERS | _SOURCE_OPTIONAL_MEMBERS)
        )
        if missing or extra:
            raise ValueError(
                "PointWorld source snapshot fields changed; "
                f"missing={missing}, extra={extra}"
            )
        schema_name = _scalar_text(data["schema_name"], name="schema_name")
        schema_version = _scalar_integer(
            data["schema_version"],
            name="schema_version",
        )
        if schema_name != POINTWORLD_SPARSE_SOURCE_SCHEMA:
            raise ValueError("unsupported PointWorld source snapshot schema")
        if schema_version != POINTWORLD_SPARSE_SOURCE_VERSION:
            raise ValueError("unsupported PointWorld source snapshot version")
        return pointworld_output_to_persistent_window(
            window_id=window_id,
            frame_indices=data["frame_indices"],
            scene_flows=data["scene_flows"],
            scene_exists=data["scene_exists"],
            log_var=data["log_var"],
            context_frame_count=_scalar_integer(
                data["context_frame_count"],
                name="context_frame_count",
            ),
            point_ids=data["point_ids"] if "point_ids" in data else None,
            storage_dtype=storage_dtype,
        )


def export_pointworld_source_snapshot(
    source_path: str | Path,
    output_path: str | Path,
    *,
    window_id: str,
    storage_dtype: StorageDType = "float32",
) -> PersistentPointPredictionWindow:
    """Convert one strict source snapshot into a no-clobber canonical archive."""

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "persistent-point output already exists; refusing to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    window = load_pointworld_source_snapshot(
        source_path,
        window_id=window_id,
        storage_dtype=storage_dtype,
    )
    try:
        window.to_npz(output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    verified = PersistentPointPredictionWindow.from_npz(output)
    arrays_match = (
        np.array_equal(verified.frame_indices, window.frame_indices)
        and np.array_equal(verified.point_ids, window.point_ids)
        and np.array_equal(verified.point_trajectory, window.point_trajectory)
        and np.array_equal(verified.valid_mask, window.valid_mask)
        and (
            (verified.uncertainty is None and window.uncertainty is None)
            or (
                verified.uncertainty is not None
                and window.uncertainty is not None
                and np.array_equal(verified.uncertainty, window.uncertainty)
            )
        )
    )
    if verified.summary() != window.summary() or not arrays_match:
        output.unlink(missing_ok=True)
        raise RuntimeError("persistent-point archive verification changed content")
    return verified


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a strict PointWorld scene-point snapshot into a "
            "persistent sparse Prob4D prediction window."
        )
    )
    parser.add_argument("source_snapshot")
    parser.add_argument("output")
    parser.add_argument("--window-id", required=True)
    parser.add_argument(
        "--storage-dtype",
        choices=STORAGE_DTYPES,
        default="float32",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(
        list(argv) if argv is not None else None
    )
    window = export_pointworld_source_snapshot(
        arguments.source_snapshot,
        arguments.output,
        window_id=arguments.window_id,
        storage_dtype=arguments.storage_dtype,
    )
    print(
        json.dumps(
            {
                "output": str(Path(arguments.output)),
                "payload_schema": PERSISTENT_POINT_WINDOW_NPZ_SCHEMA,
                **window.summary(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "POINTWORLD_SPARSE_SOURCE_SCHEMA",
    "POINTWORLD_SPARSE_SOURCE_VERSION",
    "POINTWORLD_UNCERTAINTY_SEMANTICS",
    "export_pointworld_source_snapshot",
    "load_pointworld_source_snapshot",
    "main",
    "pointworld_output_to_persistent_window",
]


if __name__ == "__main__":
    raise SystemExit(main())
