"""Validated sparse prediction windows with persistent point identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array, immutable_integer_array

FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.integer[Any]]
StorageDType = Literal["float32", "float64"]

PERSISTENT_POINT_WINDOW_NPZ_SCHEMA: Final = "prob4d.persistent-point-prediction-window-npz"
PERSISTENT_POINT_WINDOW_NPZ_VERSION: Final = 1
PERSISTENT_POINT_IDENTITY_SEMANTICS: Final = (
    "window-local-point-id-persistent-across-output-frames-v1"
)
PERSISTENT_POINT_TRAJECTORY_SEMANTICS: Final = "absolute-point-position-per-output-frame-v1"
UNCERTAINTY_ABSENT: Final = "absent"
STORAGE_DTYPES: Final[tuple[StorageDType, ...]] = ("float32", "float64")

_REQUIRED_MEMBERS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "storage_dtype",
        "window_id",
        "frame_indices",
        "point_ids",
        "point_trajectory",
        "valid_mask",
        "context_frame_count",
        "point_identity_semantics",
        "trajectory_semantics",
        "uncertainty_semantics",
    }
)
_OPTIONAL_MEMBERS: Final = frozenset({"uncertainty"})


def _validated_storage_dtype(value: object) -> StorageDType:
    normalized = str(value)
    if normalized not in STORAGE_DTYPES:
        raise ValueError("storage_dtype must be one of " + ", ".join(STORAGE_DTYPES))
    return cast(StorageDType, normalized)


def _numpy_dtype(value: StorageDType) -> np.dtype[Any]:
    return np.dtype(np.float32 if value == "float32" else np.float64)


def _strict_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be one scalar string")
    return str(value.item())


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be one scalar integer")
    return int(value.item())


def _strict_bool_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must use bool dtype")
    return np.array(array, dtype=bool, copy=True)


@dataclass(frozen=True)
class PersistentPointPredictionWindow:
    """Sparse point trajectories whose point identity is stable within a window.

    ``point_trajectory[t, n]`` is the absolute position of ``point_ids[n]`` at
    ``frame_indices[t]`` in the provider's declared coordinate system. The
    first output frame seeds every retained point identity. Optional
    ``uncertainty`` is preserved as an uninterpreted provider quantity whose
    exact meaning is named by ``uncertainty_semantics``; it is not promoted to
    calibrated covariance by this contract.
    """

    window_id: str
    frame_indices: IntArray
    point_ids: IntArray
    point_trajectory: FloatArray
    valid_mask: BoolArray
    context_frame_count: int
    uncertainty: FloatArray | None = None
    uncertainty_semantics: str = UNCERTAINTY_ABSENT
    storage_dtype: StorageDType = "float64"
    point_identity_semantics: str = PERSISTENT_POINT_IDENTITY_SEMANTICS
    trajectory_semantics: str = PERSISTENT_POINT_TRAJECTORY_SEMANTICS

    def __post_init__(self) -> None:
        window_id = _strict_text(self.window_id, name="window_id")
        storage_dtype = _validated_storage_dtype(self.storage_dtype)
        dense_dtype = _numpy_dtype(storage_dtype)

        frame_indices = immutable_integer_array(
            self.frame_indices,
            name="frame_indices",
        )
        point_ids = immutable_integer_array(self.point_ids, name="point_ids")
        trajectory = np.asarray(self.point_trajectory, dtype=dense_dtype).copy()
        valid_mask = _strict_bool_array(self.valid_mask, name="valid_mask")
        context_frame_count = _strict_integer(
            self.context_frame_count,
            name="context_frame_count",
        )
        uncertainty_semantics = _strict_text(
            self.uncertainty_semantics,
            name="uncertainty_semantics",
        )
        point_identity_semantics = _strict_text(
            self.point_identity_semantics,
            name="point_identity_semantics",
        )
        trajectory_semantics = _strict_text(
            self.trajectory_semantics,
            name="trajectory_semantics",
        )

        if point_identity_semantics != PERSISTENT_POINT_IDENTITY_SEMANTICS:
            raise ValueError("unsupported point_identity_semantics")
        if trajectory_semantics != PERSISTENT_POINT_TRAJECTORY_SEMANTICS:
            raise ValueError("unsupported trajectory_semantics")
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError("frame_indices must be a non-empty one-dimensional array")
        if np.any(frame_indices < 0) or np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be non-negative and strictly increasing")
        if point_ids.ndim != 1 or point_ids.size == 0:
            raise ValueError("point_ids must be a non-empty one-dimensional array")
        if np.any(point_ids < 0) or np.any(np.diff(point_ids) <= 0):
            raise ValueError("point_ids must be non-negative and strictly increasing")
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError("point_trajectory must have shape (T, N, 3)")
        if trajectory.shape[:2] != (
            frame_indices.size,
            point_ids.size,
        ):
            raise ValueError("point_trajectory dimensions must match frame_indices and point_ids")
        if valid_mask.shape != trajectory.shape[:2]:
            raise ValueError("valid_mask must have shape (T, N)")
        if not 1 <= context_frame_count <= frame_indices.size:
            raise ValueError("context_frame_count must lie in [1, number of frames]")
        if not np.all(valid_mask[0]):
            raise ValueError(
                "every retained persistent point must be valid in the first context frame"
            )
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("point_trajectory must contain only finite values")

        uncertainty: np.ndarray | None
        if self.uncertainty is None:
            if uncertainty_semantics != UNCERTAINTY_ABSENT:
                raise ValueError(
                    "uncertainty_semantics must be 'absent' when uncertainty is absent"
                )
            uncertainty = None
        else:
            if uncertainty_semantics == UNCERTAINTY_ABSENT:
                raise ValueError("uncertainty_semantics must describe present uncertainty")
            uncertainty = np.asarray(
                self.uncertainty,
                dtype=dense_dtype,
            ).copy()
            if (
                uncertainty.ndim != 3
                or uncertainty.shape[:2] != trajectory.shape[:2]
                or uncertainty.shape[-1] not in {1, 3}
            ):
                raise ValueError("uncertainty must have shape (T, N, 1) or (T, N, 3)")
            if not np.all(np.isfinite(uncertainty)):
                raise ValueError("uncertainty must contain only finite values")

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "storage_dtype", storage_dtype)
        object.__setattr__(
            self,
            "context_frame_count",
            context_frame_count,
        )
        object.__setattr__(
            self,
            "uncertainty_semantics",
            uncertainty_semantics,
        )
        object.__setattr__(
            self,
            "point_identity_semantics",
            point_identity_semantics,
        )
        object.__setattr__(
            self,
            "trajectory_semantics",
            trajectory_semantics,
        )
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(
            self,
            "point_trajectory",
            immutable_array(trajectory),
        )
        object.__setattr__(
            self,
            "valid_mask",
            immutable_array(valid_mask),
        )
        object.__setattr__(
            self,
            "uncertainty",
            None if uncertainty is None else immutable_array(uncertainty),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the ``(T, N)`` trajectory grid shape."""

        return self.point_trajectory.shape[:2]

    @property
    def start_frame(self) -> int:
        return int(self.frame_indices[0])

    @property
    def stop_frame(self) -> int:
        """Return the exclusive final frame index for unit-stride output."""

        return int(self.frame_indices[-1]) + 1

    @property
    def prediction_frame_indices(self) -> IntArray:
        return self.frame_indices[self.context_frame_count :]

    def local_index(self, frame_index: int) -> int:
        position = int(np.searchsorted(self.frame_indices, frame_index))
        if position == self.frame_indices.size or self.frame_indices[position] != frame_index:
            raise KeyError(f"frame {frame_index} is not in window {self.window_id!r}")
        return position

    def common_frames(
        self,
        other: PersistentPointPredictionWindow,
    ) -> IntArray:
        return np.intersect1d(
            self.frame_indices,
            other.frame_indices,
            assume_unique=True,
        )

    def summary(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "start_frame": self.start_frame,
            "stop_frame_exclusive": self.stop_frame,
            "frame_count": int(self.shape[0]),
            "context_frame_count": self.context_frame_count,
            "prediction_frame_count": int(self.shape[0] - self.context_frame_count),
            "point_count": int(self.shape[1]),
            "uncertainty_dimension": (
                0 if self.uncertainty is None else int(self.uncertainty.shape[-1])
            ),
            "uncertainty_semantics": self.uncertainty_semantics,
            "storage_dtype": self.storage_dtype,
        }

    def to_npz(
        self,
        path: str | Path,
        *,
        storage_dtype: StorageDType | None = None,
    ) -> None:
        """Write the strict versioned archive."""

        selected_dtype = (
            self.storage_dtype if storage_dtype is None else _validated_storage_dtype(storage_dtype)
        )
        numpy_dtype = _numpy_dtype(selected_dtype)
        payload: dict[str, Any] = {
            "schema_name": np.asarray(PERSISTENT_POINT_WINDOW_NPZ_SCHEMA),
            "schema_version": np.asarray(
                PERSISTENT_POINT_WINDOW_NPZ_VERSION,
                dtype=np.int64,
            ),
            "storage_dtype": np.asarray(selected_dtype),
            "window_id": np.asarray(self.window_id),
            "frame_indices": self.frame_indices,
            "point_ids": self.point_ids,
            "point_trajectory": self.point_trajectory.astype(
                numpy_dtype,
                copy=False,
            ),
            "valid_mask": self.valid_mask,
            "context_frame_count": np.asarray(
                self.context_frame_count,
                dtype=np.int64,
            ),
            "point_identity_semantics": np.asarray(self.point_identity_semantics),
            "trajectory_semantics": np.asarray(self.trajectory_semantics),
            "uncertainty_semantics": np.asarray(self.uncertainty_semantics),
        }
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty.astype(
                numpy_dtype,
                copy=False,
            )
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **payload)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
    ) -> PersistentPointPredictionWindow:
        """Read one strict versioned persistent-point archive."""

        with np.load(Path(path), allow_pickle=False) as data:
            files = set(data.files)
            missing = sorted(_REQUIRED_MEMBERS - files)
            extra = sorted(files - (_REQUIRED_MEMBERS | _OPTIONAL_MEMBERS))
            if missing or extra:
                raise ValueError(
                    f"persistent-point archive fields changed; missing={missing}, extra={extra}"
                )
            schema_name = _scalar_text(
                data["schema_name"],
                name="schema_name",
            )
            schema_version = _scalar_integer(
                data["schema_version"],
                name="schema_version",
            )
            if schema_name != PERSISTENT_POINT_WINDOW_NPZ_SCHEMA:
                raise ValueError("unsupported persistent-point archive schema")
            if schema_version != PERSISTENT_POINT_WINDOW_NPZ_VERSION:
                raise ValueError("unsupported persistent-point archive version")
            storage_dtype = _validated_storage_dtype(
                _scalar_text(
                    data["storage_dtype"],
                    name="storage_dtype",
                )
            )
            expected_dtype = _numpy_dtype(storage_dtype)
            if data["point_trajectory"].dtype != expected_dtype:
                raise ValueError("point_trajectory dtype disagrees with storage_dtype")
            if "uncertainty" in data and data["uncertainty"].dtype != expected_dtype:
                raise ValueError("uncertainty dtype disagrees with storage_dtype")
            for scalar_name in ("schema_version", "context_frame_count"):
                if data[scalar_name].dtype != np.dtype(np.int64):
                    raise ValueError(f"{scalar_name} must use int64")
            if data["frame_indices"].dtype != np.dtype(np.int64):
                raise ValueError("frame_indices must use int64")
            if data["point_ids"].dtype != np.dtype(np.int64):
                raise ValueError("point_ids must use int64")
            if data["valid_mask"].dtype != np.dtype(bool):
                raise ValueError("valid_mask must use bool")
            uncertainty_semantics = _scalar_text(
                data["uncertainty_semantics"],
                name="uncertainty_semantics",
            )
            return cls(
                window_id=_scalar_text(data["window_id"], name="window_id"),
                frame_indices=data["frame_indices"],
                point_ids=data["point_ids"],
                point_trajectory=data["point_trajectory"],
                valid_mask=data["valid_mask"],
                context_frame_count=_scalar_integer(
                    data["context_frame_count"],
                    name="context_frame_count",
                ),
                uncertainty=(data["uncertainty"] if "uncertainty" in data else None),
                uncertainty_semantics=uncertainty_semantics,
                storage_dtype=storage_dtype,
                point_identity_semantics=_scalar_text(
                    data["point_identity_semantics"],
                    name="point_identity_semantics",
                ),
                trajectory_semantics=_scalar_text(
                    data["trajectory_semantics"],
                    name="trajectory_semantics",
                ),
            )


__all__ = [
    "PERSISTENT_POINT_IDENTITY_SEMANTICS",
    "PERSISTENT_POINT_TRAJECTORY_SEMANTICS",
    "PERSISTENT_POINT_WINDOW_NPZ_SCHEMA",
    "PERSISTENT_POINT_WINDOW_NPZ_VERSION",
    "STORAGE_DTYPES",
    "UNCERTAINTY_ABSENT",
    "PersistentPointPredictionWindow",
    "StorageDType",
]
