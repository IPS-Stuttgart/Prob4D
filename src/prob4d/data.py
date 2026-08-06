"""Validated data contracts for decoded 4-D prediction windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array

FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.integer[Any]]
DenseStorageDType = Literal["float32", "float64"]
DENSE_STORAGE_DTYPES: Final[tuple[DenseStorageDType, ...]] = (
    "float32",
    "float64",
)
PREDICTION_WINDOW_NPZ_SCHEMA: Final[str] = "prob4d.prediction-window-npz"
PREDICTION_WINDOW_NPZ_VERSION: Final[int] = 2
_PREDICTION_WINDOW_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_name", "schema_version", "dense_storage_dtype"}
)
_PREDICTION_WINDOW_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "dense_storage_dtype",
        "window_id",
        "frame_indices",
        "point_map",
        "valid_mask",
    }
)
_PREDICTION_WINDOW_OPTIONAL_MEMBERS: Final[frozenset[str]] = frozenset(
    {"scene_flow", "deform_mask", "ray_directions"}
)


def _readonly_owned(value: np.ndarray) -> np.ndarray:
    """Move an already validated array onto irreversible read-only storage."""

    return immutable_array(value)


def _validated_dense_storage_dtype(value: object) -> DenseStorageDType:
    normalized = str(value)
    if normalized not in DENSE_STORAGE_DTYPES:
        raise ValueError("dense_storage_dtype must be one of " + ", ".join(DENSE_STORAGE_DTYPES))
    return cast(DenseStorageDType, normalized)


def _numpy_dense_dtype(value: DenseStorageDType) -> np.dtype[Any]:
    return np.dtype(np.float32 if value == "float32" else np.float64)


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be one scalar string")
    return str(value.item())


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be one scalar integer")
    return int(value.item())


def _validated_versioned_archive(data: Any) -> DenseStorageDType:
    files = set(data.files)
    allowed = _PREDICTION_WINDOW_REQUIRED_MEMBERS | _PREDICTION_WINDOW_OPTIONAL_MEMBERS
    missing = sorted(_PREDICTION_WINDOW_REQUIRED_MEMBERS - files)
    extra = sorted(files - allowed)
    if missing or extra:
        raise ValueError(
            f"prediction-window archive fields changed; missing={missing}, extra={extra}"
        )
    if ("scene_flow" in files) != ("deform_mask" in files):
        raise ValueError(
            "versioned prediction-window archives require scene_flow and deform_mask together"
        )

    schema_name = _scalar_text(data["schema_name"], name="schema_name")
    schema_version = _scalar_integer(data["schema_version"], name="schema_version")
    if schema_name != PREDICTION_WINDOW_NPZ_SCHEMA:
        raise ValueError("unsupported prediction-window archive schema")
    if schema_version != PREDICTION_WINDOW_NPZ_VERSION:
        raise ValueError("unsupported prediction-window archive version")
    stored_dtype = _validated_dense_storage_dtype(
        _scalar_text(data["dense_storage_dtype"], name="dense_storage_dtype")
    )
    _scalar_text(data["window_id"], name="window_id")

    expected_dtype = _numpy_dense_dtype(stored_dtype)
    for field in ("point_map", "scene_flow", "ray_directions"):
        if field in data and data[field].dtype != expected_dtype:
            raise ValueError(f"{field} dtype disagrees with dense_storage_dtype metadata")
    if data["frame_indices"].dtype != np.dtype(np.int64):
        raise ValueError("versioned frame_indices must use int64")
    if data["valid_mask"].dtype != np.dtype(bool):
        raise ValueError("versioned valid_mask must use bool")
    if "deform_mask" in data and data["deform_mask"].dtype != np.dtype(bool):
        raise ValueError("versioned deform_mask must use bool")
    return stored_dtype


@dataclass(frozen=True)
class PredictionWindow:
    """Decoded predictions for one local temporal window.

    Point maps and scene flow use the window's local world gauge. Absolute
    ``frame_indices`` identify duplicate frames across overlapping windows.
    Every NumPy field is defensively copied and made read-only so a validated
    window cannot be mutated after it has entered a content-addressed workflow.
    """

    window_id: str
    frame_indices: IntArray
    point_map: FloatArray
    valid_mask: BoolArray
    scene_flow: FloatArray | None = None
    deform_mask: BoolArray | None = None
    ray_directions: FloatArray | None = None
    dense_storage_dtype: DenseStorageDType = "float64"

    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        dense_storage_dtype = _validated_dense_storage_dtype(self.dense_storage_dtype)
        dense_dtype = _numpy_dense_dtype(dense_storage_dtype)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64).copy()
        point_map = np.asarray(self.point_map, dtype=dense_dtype).copy()
        valid_mask = np.asarray(self.valid_mask, dtype=bool).copy()

        if not window_id:
            raise ValueError("window_id must not be empty")
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError("frame_indices must be a non-empty one-dimensional array")
        if np.any(frame_indices < 0):
            raise ValueError("frame_indices must be non-negative")
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if point_map.ndim != 4 or point_map.shape[-1] != 3:
            raise ValueError("point_map must have shape (T, H, W, 3)")
        if valid_mask.shape != point_map.shape[:-1]:
            raise ValueError("valid_mask must have shape (T, H, W)")
        if point_map.shape[0] != frame_indices.size:
            raise ValueError("frame_indices length must match point_map time dimension")
        if not np.all(np.isfinite(point_map[valid_mask])):
            raise ValueError("valid point_map entries must be finite")

        scene_flow = self._optional_vector_field("scene_flow", self.scene_flow, point_map)
        deform_mask = self._optional_mask("deform_mask", self.deform_mask, valid_mask)
        rays = self._optional_vector_field("ray_directions", self.ray_directions, point_map)

        if (scene_flow is None) != (deform_mask is None):
            raise ValueError("scene_flow and deform_mask must either both be present or absent")
        if deform_mask is not None:
            assert scene_flow is not None
            if np.any(deform_mask & ~valid_mask):
                raise ValueError("deform_mask must be a subset of valid_mask")
            if not np.all(np.isfinite(scene_flow[deform_mask])):
                raise ValueError("active scene_flow entries must be finite")
        if rays is not None:
            if not np.all(np.isfinite(rays[valid_mask])):
                raise ValueError("valid ray_directions entries must be finite")
            ray_norm = np.linalg.norm(rays, axis=-1)
            if np.any(valid_mask & (ray_norm <= np.finfo(np.float64).eps)):
                raise ValueError("valid ray_directions entries must be nonzero")
            rays[valid_mask] /= ray_norm[valid_mask, None]

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "dense_storage_dtype", dense_storage_dtype)
        object.__setattr__(self, "frame_indices", _readonly_owned(frame_indices))
        object.__setattr__(self, "point_map", _readonly_owned(point_map))
        object.__setattr__(self, "valid_mask", _readonly_owned(valid_mask))
        object.__setattr__(
            self,
            "scene_flow",
            None if scene_flow is None else _readonly_owned(scene_flow),
        )
        object.__setattr__(
            self,
            "deform_mask",
            None if deform_mask is None else _readonly_owned(deform_mask),
        )
        object.__setattr__(
            self,
            "ray_directions",
            None if rays is None else _readonly_owned(rays),
        )

    @staticmethod
    def _optional_vector_field(
        name: str, value: FloatArray | None, reference: FloatArray
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=reference.dtype).copy()
        if array.shape != reference.shape:
            raise ValueError(f"{name} must have shape {reference.shape}")
        return array

    @staticmethod
    def _optional_mask(
        name: str, value: BoolArray | None, reference: BoolArray
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=bool).copy()
        if array.shape != reference.shape:
            raise ValueError(f"{name} must have shape {reference.shape}")
        return array

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the ``(T, H, W)`` sample grid shape."""

        return self.point_map.shape[:-1]

    @property
    def start_frame(self) -> int:
        return int(self.frame_indices[0])

    @property
    def stop_frame(self) -> int:
        """Return the exclusive final frame index for unit-stride windows."""

        return int(self.frame_indices[-1]) + 1

    def local_index(self, frame_index: int) -> int:
        position = int(np.searchsorted(self.frame_indices, frame_index))
        if position == self.frame_indices.size or self.frame_indices[position] != frame_index:
            raise KeyError(f"frame {frame_index} is not in window {self.window_id!r}")
        return position

    def common_frames(self, other: PredictionWindow) -> IntArray:
        return np.intersect1d(self.frame_indices, other.frame_indices, assume_unique=True)

    @property
    def dense_vector_storage_bytes(self) -> int:
        """Return bytes retained by dense three-vector prediction fields."""

        arrays = (self.point_map, self.scene_flow, self.ray_directions)
        return sum(array.nbytes for array in arrays if array is not None)

    def rays_at(
        self,
        local_index: int,
        *,
        dtype: type[np.floating[Any]] | np.dtype[np.floating[Any]] | None = None,
    ) -> FloatArray:
        """Return normalized rays for one frame without copying the full window."""

        index = int(local_index)
        if index != local_index or not 0 <= index < self.shape[0]:
            raise IndexError("local ray frame index is out of range")
        target_dtype = self.point_map.dtype if dtype is None else np.dtype(dtype)
        if target_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("ray dtype must be float32 or float64")
        if self.ray_directions is not None:
            return np.array(
                self.ray_directions[index],
                dtype=target_dtype,
                copy=True,
            )

        points = np.asarray(self.point_map[index], dtype=target_dtype)
        norms = np.linalg.norm(points, axis=-1, keepdims=True)
        return np.divide(
            points,
            norms,
            out=np.zeros(points.shape, dtype=target_dtype),
            where=norms > np.finfo(np.float64).eps,
        )

    def rays(
        self,
        *,
        dtype: type[np.floating[Any]] | np.dtype[np.floating[Any]] | None = None,
    ) -> FloatArray:
        """Return all normalized rays, using frame-local temporary storage."""

        if self.ray_directions is not None:
            target_dtype = self.ray_directions.dtype if dtype is None else np.dtype(dtype)
            if target_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
                raise ValueError("ray dtype must be float32 or float64")
            return np.array(self.ray_directions, dtype=target_dtype, copy=True)
        target_dtype = self.point_map.dtype if dtype is None else np.dtype(dtype)
        output = np.empty(self.point_map.shape, dtype=target_dtype)
        for local_index in range(self.shape[0]):
            output[local_index] = self.rays_at(local_index, dtype=target_dtype)
        return output

    def to_npz(
        self,
        path: str | Path,
        *,
        storage_dtype: DenseStorageDType | None = None,
    ) -> None:
        """Write a versioned archive without silently changing dense precision.

        By default, the archive preserves ``dense_storage_dtype``. Callers that
        intentionally want a compact float32 archive must pass
        ``storage_dtype="float32"`` explicitly; the selected on-disk dtype is
        recorded and validated when the archive is loaded.
        """

        selected_dtype = (
            self.dense_storage_dtype
            if storage_dtype is None
            else _validated_dense_storage_dtype(storage_dtype)
        )
        numpy_dtype = _numpy_dense_dtype(selected_dtype)
        payload: dict[str, Any] = {
            "schema_name": np.asarray(PREDICTION_WINDOW_NPZ_SCHEMA),
            "schema_version": np.asarray(PREDICTION_WINDOW_NPZ_VERSION, dtype=np.int64),
            "dense_storage_dtype": np.asarray(selected_dtype),
            "window_id": np.asarray(self.window_id),
            "frame_indices": self.frame_indices,
            "point_map": self.point_map.astype(numpy_dtype, copy=False),
            "valid_mask": self.valid_mask,
        }
        if self.scene_flow is not None:
            payload["scene_flow"] = self.scene_flow.astype(numpy_dtype, copy=False)
            payload["deform_mask"] = self.deform_mask
        if self.ray_directions is not None:
            payload["ray_directions"] = self.ray_directions.astype(
                numpy_dtype,
                copy=False,
            )
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        start_frame: int | None = None,
        window_id: str | None = None,
        dense_storage_dtype: DenseStorageDType | None = None,
    ) -> PredictionWindow:
        """Read a versioned or legacy window with explicit precision semantics."""

        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            present_metadata = _PREDICTION_WINDOW_METADATA_FIELDS.intersection(data.files)
            stored_dtype: DenseStorageDType | None = None
            if present_metadata:
                if present_metadata != _PREDICTION_WINDOW_METADATA_FIELDS:
                    missing = sorted(_PREDICTION_WINDOW_METADATA_FIELDS - present_metadata)
                    raise ValueError(
                        "prediction-window archive has incomplete storage metadata: "
                        + ", ".join(missing)
                    )
                stored_dtype = _validated_versioned_archive(data)
            elif "point_map" not in data or "valid_mask" not in data:
                raise ValueError(f"{path} does not contain point_map and valid_mask")

            target_dtype = (
                _validated_dense_storage_dtype(dense_storage_dtype)
                if dense_storage_dtype is not None
                else stored_dtype or "float64"
            )
            time_steps = int(data["point_map"].shape[0])
            if stored_dtype is not None:
                frame_indices = data["frame_indices"]
                stored_id = _scalar_text(data["window_id"], name="window_id")
            else:
                if "frame_indices" in data:
                    frame_indices = data["frame_indices"]
                elif start_frame is not None:
                    frame_indices = np.arange(start_frame, start_frame + time_steps)
                else:
                    raise ValueError(
                        "prediction files without frame_indices require start_frame explicitly"
                    )
                stored_id = str(data["window_id"].item()) if "window_id" in data else path.stem

            return cls(
                window_id=window_id or stored_id,
                frame_indices=frame_indices,
                point_map=data["point_map"],
                valid_mask=data["valid_mask"],
                scene_flow=data["scene_flow"] if "scene_flow" in data else None,
                deform_mask=data["deform_mask"] if "deform_mask" in data else None,
                ray_directions=data["ray_directions"] if "ray_directions" in data else None,
                dense_storage_dtype=target_dtype,
            )


__all__ = [
    "DENSE_STORAGE_DTYPES",
    "PREDICTION_WINDOW_NPZ_SCHEMA",
    "PREDICTION_WINDOW_NPZ_VERSION",
    "DenseStorageDType",
    "PredictionWindow",
]
