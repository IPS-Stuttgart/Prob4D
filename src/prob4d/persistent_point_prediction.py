"""Versioned persistent-point prediction windows for sparse 4-D providers.

The dense :class:`prob4d.data.PredictionWindow` contract is intentionally tied to
an image-grid point map. Action-conditioned point-world models instead retain a
fixed scene-point axis and predict the positions of those seeded points over a
future horizon. This module preserves that representation without rasterizing it
onto a target-dependent image grid.

Provider-reported log variance is retained as an explicitly uncalibrated field.
It is not converted into metric covariance by this contract.
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

PERSISTENT_POINT_WINDOW_NPZ_SCHEMA: Final = (
    "prob4d.persistent-point-prediction-window-npz"
)
PERSISTENT_POINT_WINDOW_NPZ_VERSION: Final = 1
PERSISTENT_POINT_STORAGE_DTYPES: Final[tuple[StorageDType, ...]] = (
    "float32",
    "float64",
)
POINTWORLD_POSITION_SEMANTICS: Final = (
    "absolute-position-of-seeded-scene-point-v1"
)
POINTWORLD_POINT_IDENTITY_SEMANTICS: Final = (
    "window-scoped-persistent-source-axis-hash-v1"
)
POINTWORLD_REPORTED_UNCERTAINTY_SEMANTICS: Final = (
    "provider-log-variance-of-normalized-initial-frame-relative-displacement-v1"
)

_REQUIRED_MEMBERS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "storage_dtype",
        "window_id",
        "frame_indices",
        "source_point_indices",
        "point_ids",
        "point_positions",
        "valid_mask",
        "position_semantics",
        "point_identity_semantics",
    }
)
_REPORTED_UNCERTAINTY_MEMBERS: Final = frozenset(
    {
        "reported_log_variance",
        "reported_uncertainty_semantics",
        "reported_uncertainty_reference_id",
    }
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _validated_storage_dtype(value: Any) -> StorageDType:
    normalized = str(value)
    if normalized not in PERSISTENT_POINT_STORAGE_DTYPES:
        raise ValueError(
            "storage_dtype must be one of "
            + ", ".join(PERSISTENT_POINT_STORAGE_DTYPES)
        )
    return cast(StorageDType, normalized)


def _numpy_dtype(value: StorageDType) -> np.dtype[Any]:
    return np.dtype(np.float32 if value == "float32" else np.float64)


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be one scalar string")
    result = str(value.item())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be one scalar integer")
    return int(value.item())


def _nonempty_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be one nonempty literal string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    result = _nonempty_text(value, name=name)
    if len(result) != 64:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(result)
    except ValueError as error:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest") from error
    if len(decoded) != 32 or result != result.lower():
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return result


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


def window_scoped_point_ids(
    window_id: str,
    source_point_indices: Any,
) -> IntArray:
    """Derive deterministic identities without asserting cross-window association."""

    identifier = _nonempty_text(window_id, name="window_id")
    source_indices = _integer_vector(
        source_point_indices,
        name="source_point_indices",
        nonempty=True,
    )
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("source_point_indices must be unique")

    point_ids = np.empty(len(source_indices), dtype=np.int64)
    for position, source_index in enumerate(source_indices):
        payload = f"{identifier}\x00{int(source_index)}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        point_ids[position] = int.from_bytes(digest[:8]) & ((1 << 63) - 1)
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError("window-scoped point identity collision")
    return cast(IntArray, _readonly(point_ids))


@dataclass(frozen=True, slots=True)
class PersistentPointPredictionWindow:
    """One sparse prediction window with persistent point identities.

    ``point_positions[t, n]`` describes the same provider seed identified by
    ``point_ids[n]`` at every frame in the window. The identity is persistent
    only under the declared ``point_identity_semantics``; the PointWorld adapter
    deliberately assigns different identities to points from different windows.

    ``reported_log_variance`` is raw provider output tied to
    ``reported_uncertainty_reference_id``. It is not metric covariance and must
    be calibrated before it is used as ``ObservationFactor.local_covariance_m2``.
    """

    window_id: str
    frame_indices: IntArray
    source_point_indices: IntArray
    point_ids: IntArray
    point_positions: FloatArray
    valid_mask: BoolArray
    position_semantics: str
    point_identity_semantics: str
    reported_log_variance: FloatArray | None = None
    reported_uncertainty_semantics: str | None = None
    reported_uncertainty_reference_id: str | None = None
    storage_dtype: StorageDType = "float64"

    def __post_init__(self) -> None:
        window_id = _nonempty_text(self.window_id, name="window_id")
        position_semantics = _nonempty_text(
            self.position_semantics,
            name="position_semantics",
        )
        point_identity_semantics = _nonempty_text(
            self.point_identity_semantics,
            name="point_identity_semantics",
        )
        storage_dtype = _validated_storage_dtype(self.storage_dtype)
        dtype = _numpy_dtype(storage_dtype)

        frame_indices = _integer_vector(
            self.frame_indices,
            name="frame_indices",
            nonempty=True,
        )
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        source_indices = _integer_vector(
            self.source_point_indices,
            name="source_point_indices",
            nonempty=True,
        )
        point_ids = _integer_vector(
            self.point_ids,
            name="point_ids",
            nonempty=True,
        )
        if len(source_indices) != len(point_ids):
            raise ValueError("source_point_indices and point_ids must have equal length")
        if len(np.unique(source_indices)) != len(source_indices):
            raise ValueError("source_point_indices must be unique")
        if len(np.unique(point_ids)) != len(point_ids):
            raise ValueError("point_ids must be unique")

        positions = np.asarray(self.point_positions, dtype=dtype).copy()
        raw_valid = np.asarray(self.valid_mask)
        if raw_valid.dtype != np.dtype(bool):
            raise TypeError("valid_mask must have bool dtype")
        valid = raw_valid.copy()
        expected_shape = (len(frame_indices), len(point_ids))
        if positions.shape != (*expected_shape, 3):
            raise ValueError(
                "point_positions must have shape "
                f"({expected_shape[0]}, {expected_shape[1]}, 3)"
            )
        if valid.shape != expected_shape:
            raise ValueError(
                "valid_mask must have shape "
                f"({expected_shape[0]}, {expected_shape[1]})"
            )
        if not np.all(np.isfinite(positions[valid])):
            raise ValueError("valid point positions must be finite")

        uncertainty_fields = (
            self.reported_log_variance,
            self.reported_uncertainty_semantics,
            self.reported_uncertainty_reference_id,
        )
        present = tuple(item is not None for item in uncertainty_fields)
        if any(present) and not all(present):
            raise ValueError(
                "reported uncertainty payload, semantics, and reference ID "
                "must either all be present or all be absent"
            )

        reported_log_variance: np.ndarray | None = None
        reported_uncertainty_semantics: str | None = None
        reported_uncertainty_reference_id: str | None = None
        if all(present):
            assert self.reported_log_variance is not None
            assert self.reported_uncertainty_semantics is not None
            assert self.reported_uncertainty_reference_id is not None
            reported_log_variance = np.asarray(
                self.reported_log_variance,
                dtype=dtype,
            ).copy()
            if reported_log_variance.shape not in {
                (*expected_shape, 1),
                (*expected_shape, 3),
            }:
                raise ValueError(
                    "reported_log_variance must have shape (T, N, 1) or (T, N, 3)"
                )
            active_uncertainty = np.broadcast_to(
                valid[..., None],
                reported_log_variance.shape,
            )
            if not np.all(np.isfinite(reported_log_variance[active_uncertainty])):
                raise ValueError("valid reported log variances must be finite")
            reported_uncertainty_semantics = _nonempty_text(
                self.reported_uncertainty_semantics,
                name="reported_uncertainty_semantics",
            )
            reported_uncertainty_reference_id = _sha256(
                self.reported_uncertainty_reference_id,
                name="reported_uncertainty_reference_id",
            )

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "position_semantics", position_semantics)
        object.__setattr__(
            self,
            "point_identity_semantics",
            point_identity_semantics,
        )
        object.__setattr__(self, "storage_dtype", storage_dtype)
        object.__setattr__(self, "frame_indices", _readonly(frame_indices))
        object.__setattr__(
            self,
            "source_point_indices",
            _readonly(source_indices),
        )
        object.__setattr__(self, "point_ids", _readonly(point_ids))
        object.__setattr__(self, "point_positions", _readonly(positions))
        object.__setattr__(self, "valid_mask", _readonly(valid))
        object.__setattr__(
            self,
            "reported_log_variance",
            None if reported_log_variance is None else _readonly(reported_log_variance),
        )
        object.__setattr__(
            self,
            "reported_uncertainty_semantics",
            reported_uncertainty_semantics,
        )
        object.__setattr__(
            self,
            "reported_uncertainty_reference_id",
            reported_uncertainty_reference_id,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(frame_count, point_count)``."""

        return self.valid_mask.shape

    @property
    def frame_count(self) -> int:
        return self.shape[0]

    @property
    def point_count(self) -> int:
        return self.shape[1]

    @property
    def has_reported_uncertainty(self) -> bool:
        return self.reported_log_variance is not None

    def local_index(self, frame_index: int) -> int:
        position = int(np.searchsorted(self.frame_indices, frame_index))
        if position == self.frame_count or self.frame_indices[position] != frame_index:
            raise KeyError(f"frame {frame_index} is not in window {self.window_id!r}")
        return position

    def to_npz(
        self,
        path: str | Path,
        *,
        storage_dtype: StorageDType | None = None,
    ) -> None:
        """Write a strict versioned archive while preserving semantic labels."""

        selected_dtype = (
            self.storage_dtype
            if storage_dtype is None
            else _validated_storage_dtype(storage_dtype)
        )
        dtype = _numpy_dtype(selected_dtype)
        payload: dict[str, Any] = {
            "schema_name": np.asarray(PERSISTENT_POINT_WINDOW_NPZ_SCHEMA),
            "schema_version": np.asarray(
                PERSISTENT_POINT_WINDOW_NPZ_VERSION,
                dtype=np.int64,
            ),
            "storage_dtype": np.asarray(selected_dtype),
            "window_id": np.asarray(self.window_id),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int64),
            "source_point_indices": np.asarray(
                self.source_point_indices,
                dtype=np.int64,
            ),
            "point_ids": np.asarray(self.point_ids, dtype=np.int64),
            "point_positions": np.asarray(self.point_positions, dtype=dtype),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
            "position_semantics": np.asarray(self.position_semantics),
            "point_identity_semantics": np.asarray(self.point_identity_semantics),
        }
        if self.reported_log_variance is not None:
            assert self.reported_uncertainty_semantics is not None
            assert self.reported_uncertainty_reference_id is not None
            payload.update(
                {
                    "reported_log_variance": np.asarray(
                        self.reported_log_variance,
                        dtype=dtype,
                    ),
                    "reported_uncertainty_semantics": np.asarray(
                        self.reported_uncertainty_semantics
                    ),
                    "reported_uncertainty_reference_id": np.asarray(
                        self.reported_uncertainty_reference_id
                    ),
                }
            )
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def from_npz(cls, path: str | Path) -> PersistentPointPredictionWindow:
        """Load and revalidate one strict persistent-point archive."""

        with np.load(Path(path), allow_pickle=False) as data:
            files = set(data.files)
            allowed = _REQUIRED_MEMBERS | _REPORTED_UNCERTAINTY_MEMBERS
            missing = sorted(_REQUIRED_MEMBERS - files)
            extra = sorted(files - allowed)
            if missing or extra:
                raise ValueError(
                    "persistent-point archive fields changed; "
                    f"missing={missing}, extra={extra}"
                )
            uncertainty_present = _REPORTED_UNCERTAINTY_MEMBERS.intersection(files)
            if uncertainty_present and uncertainty_present != _REPORTED_UNCERTAINTY_MEMBERS:
                raise ValueError("persistent-point uncertainty members must be complete")
            schema = _scalar_text(data["schema_name"], name="schema_name")
            version = _scalar_integer(data["schema_version"], name="schema_version")
            if schema != PERSISTENT_POINT_WINDOW_NPZ_SCHEMA:
                raise ValueError("unsupported persistent-point archive schema")
            if version != PERSISTENT_POINT_WINDOW_NPZ_VERSION:
                raise ValueError("unsupported persistent-point archive version")
            storage_dtype = _validated_storage_dtype(
                _scalar_text(data["storage_dtype"], name="storage_dtype")
            )
            expected_dtype = _numpy_dtype(storage_dtype)
            for field in ("point_positions", "reported_log_variance"):
                if field in files and data[field].dtype != expected_dtype:
                    raise ValueError(f"{field} dtype disagrees with storage_dtype")
            for field in ("frame_indices", "source_point_indices", "point_ids"):
                if data[field].dtype != np.dtype(np.int64):
                    raise ValueError(f"{field} must use int64")
            if data["valid_mask"].dtype != np.dtype(bool):
                raise ValueError("valid_mask must use bool")

            return cls(
                window_id=_scalar_text(data["window_id"], name="window_id"),
                frame_indices=data["frame_indices"],
                source_point_indices=data["source_point_indices"],
                point_ids=data["point_ids"],
                point_positions=data["point_positions"],
                valid_mask=data["valid_mask"],
                position_semantics=_scalar_text(
                    data["position_semantics"],
                    name="position_semantics",
                ),
                point_identity_semantics=_scalar_text(
                    data["point_identity_semantics"],
                    name="point_identity_semantics",
                ),
                reported_log_variance=(
                    data["reported_log_variance"]
                    if "reported_log_variance" in files
                    else None
                ),
                reported_uncertainty_semantics=(
                    _scalar_text(
                        data["reported_uncertainty_semantics"],
                        name="reported_uncertainty_semantics",
                    )
                    if "reported_uncertainty_semantics" in files
                    else None
                ),
                reported_uncertainty_reference_id=(
                    _scalar_text(
                        data["reported_uncertainty_reference_id"],
                        name="reported_uncertainty_reference_id",
                    )
                    if "reported_uncertainty_reference_id" in files
                    else None
                ),
                storage_dtype=storage_dtype,
            )


def persistent_point_window_from_pointworld(
    *,
    window_id: str,
    frame_indices: Any,
    scene_positions: Any,
    scene_valid_mask: Any,
    reported_log_variance: Any,
    normalization_id: str,
    source_point_indices: Any | None = None,
    storage_dtype: StorageDType = "float32",
) -> PersistentPointPredictionWindow:
    """Convert one released PointWorld output without dense rasterization.

    The caller supplies NumPy-compatible snapshots of PointWorld's
    ``out['scene_flows']``, validity mask, and ``out['log_var']`` after selecting
    one batch member. The raw log variance remains tied to the exact
    normalization-statistics digest and is deliberately not promoted to metric
    covariance.
    """

    positions = np.asarray(scene_positions)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("PointWorld scene_positions must have shape (T, N, 3)")
    point_count = positions.shape[1]
    source_indices = (
        np.arange(point_count, dtype=np.int64)
        if source_point_indices is None
        else _integer_vector(
            source_point_indices,
            name="source_point_indices",
            nonempty=True,
        )
    )
    if len(source_indices) != point_count:
        raise ValueError("source_point_indices length must match PointWorld point axis")
    raw_log_variance = np.asarray(reported_log_variance)
    if raw_log_variance.shape != (*positions.shape[:2], 1):
        raise ValueError("PointWorld reported_log_variance must have shape (T, N, 1)")

    return PersistentPointPredictionWindow(
        window_id=window_id,
        frame_indices=frame_indices,
        source_point_indices=source_indices,
        point_ids=window_scoped_point_ids(window_id, source_indices),
        point_positions=positions,
        valid_mask=scene_valid_mask,
        position_semantics=POINTWORLD_POSITION_SEMANTICS,
        point_identity_semantics=POINTWORLD_POINT_IDENTITY_SEMANTICS,
        reported_log_variance=raw_log_variance,
        reported_uncertainty_semantics=(
            POINTWORLD_REPORTED_UNCERTAINTY_SEMANTICS
        ),
        reported_uncertainty_reference_id=normalization_id,
        storage_dtype=storage_dtype,
    )


__all__ = [
    "PERSISTENT_POINT_STORAGE_DTYPES",
    "PERSISTENT_POINT_WINDOW_NPZ_SCHEMA",
    "PERSISTENT_POINT_WINDOW_NPZ_VERSION",
    "POINTWORLD_POINT_IDENTITY_SEMANTICS",
    "POINTWORLD_POSITION_SEMANTICS",
    "POINTWORLD_REPORTED_UNCERTAINTY_SEMANTICS",
    "PersistentPointPredictionWindow",
    "persistent_point_window_from_pointworld",
    "window_scoped_point_ids",
]
