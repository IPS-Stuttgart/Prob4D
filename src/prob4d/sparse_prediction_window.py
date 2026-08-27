"""Portable sparse persistent-point prediction windows.

This module is an upstream provider contract.  It preserves one point identity
across a finite forecast window without forcing sparse predictions onto an image
grid.  Provider-native uncertainty is retained as an explicitly labelled raw
quantity; it is not interpreted as calibrated metric covariance.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._immutable_array import immutable_array
from ._immutable_json import frozen_finite_json_mapping, plain_json
from .data import BoolArray, DenseStorageDType, FloatArray, IntArray

SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA = "prob4d.sparse-prediction-window-npz"
SPARSE_PREDICTION_WINDOW_NPZ_VERSION = 1
SPARSE_UNCERTAINTY_ABSENT = "absent"

_REQUIRED_ARCHIVE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "dense_storage_dtype",
        "window_id",
        "coordinate_semantics",
        "identity_semantics",
        "uncertainty_semantics",
        "frame_indices",
        "point_ids",
        "position",
        "valid_mask",
        "metadata_json",
    }
)
_OPTIONAL_ARCHIVE_FIELDS = frozenset(
    {"provider_uncertainty", "uncertainty_valid_mask"}
)


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _storage_dtype(value: object) -> DenseStorageDType:
    if type(value) is not str or value not in {"float32", "float64"}:
        raise ValueError("dense_storage_dtype must be 'float32' or 'float64'")
    return value


def _numpy_dtype(value: DenseStorageDType) -> np.dtype[Any]:
    return np.dtype(np.float32 if value == "float32" else np.float64)


def _integer_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must have an integer dtype")
    if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds the int64 range")
    return np.array(value, dtype=np.int64, copy=True)


def _bool_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be a Boolean NumPy array")
    return np.array(value, dtype=bool, copy=True)


def _float_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"{name} must have float32 or float64 dtype")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype must match dense_storage_dtype")
    return np.array(value, dtype=dtype, copy=True)


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be one scalar string")
    return str(value.item())


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be one scalar integer")
    return int(value.item())


def _metadata_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        plain_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_metadata_json(value: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"metadata_json contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"metadata_json contains non-finite number {token!r}")

    try:
        loaded = json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("metadata_json must contain valid JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError("metadata_json must contain one JSON object")
    return loaded


def _update_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little", signed=False))
    digest.update(encoded)


def _update_array(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    _update_text(digest, array.dtype.str)
    digest.update(len(array.shape).to_bytes(8, "little", signed=False))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class SparsePredictionWindow:
    """One sparse trajectory forecast with persistent within-window point IDs.

    ``position[t, n]`` refers to the same seeded point ``point_ids[n]`` at every
    forecast frame.  The identity scope and coordinate system are explicit
    strings because neither cross-window identity nor a metric world frame may be
    inferred from array shape.

    ``provider_uncertainty`` is deliberately generic.  Its interpretation is
    fixed only by ``uncertainty_semantics`` and it must not be treated as a
    calibrated covariance without a separate source/calibration-only mapping.
    """

    window_id: str
    frame_indices: IntArray
    point_ids: IntArray
    position: FloatArray
    valid_mask: BoolArray
    coordinate_semantics: str
    identity_semantics: str
    uncertainty_semantics: str = SPARSE_UNCERTAINTY_ABSENT
    provider_uncertainty: FloatArray | None = None
    uncertainty_valid_mask: BoolArray | None = None
    dense_storage_dtype: DenseStorageDType = "float64"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        window_id = _nonempty_string(self.window_id, name="window_id")
        coordinate_semantics = _nonempty_string(
            self.coordinate_semantics,
            name="coordinate_semantics",
        )
        identity_semantics = _nonempty_string(
            self.identity_semantics,
            name="identity_semantics",
        )
        uncertainty_semantics = _nonempty_string(
            self.uncertainty_semantics,
            name="uncertainty_semantics",
        )
        storage_dtype = _storage_dtype(self.dense_storage_dtype)
        numpy_dtype = _numpy_dtype(storage_dtype)

        frame_indices = _integer_array(self.frame_indices, name="frame_indices")
        point_ids = _integer_array(self.point_ids, name="point_ids")
        position = _float_array(
            self.position,
            name="position",
            dtype=numpy_dtype,
        )
        valid_mask = _bool_array(self.valid_mask, name="valid_mask")

        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError("frame_indices must be nonempty and one-dimensional")
        if np.any(frame_indices < 0) or np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be nonnegative and strictly increasing")
        if point_ids.ndim != 1 or point_ids.size == 0:
            raise ValueError("point_ids must be nonempty and one-dimensional")
        if np.any(point_ids < 0) or len(np.unique(point_ids)) != point_ids.size:
            raise ValueError("point_ids must be nonnegative and unique")
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("position must have shape (T, N, 3)")
        if position.shape[:2] != (frame_indices.size, point_ids.size):
            raise ValueError("position dimensions must match frame_indices and point_ids")
        if valid_mask.shape != position.shape[:2]:
            raise ValueError("valid_mask must have shape (T, N)")
        if not np.all(valid_mask[0]):
            raise ValueError("every persistent point ID must be valid at the seed frame")
        if not np.all(np.isfinite(position)):
            raise ValueError("position must contain only finite values")

        uncertainty = self.provider_uncertainty
        uncertainty_mask = self.uncertainty_valid_mask
        if (uncertainty is None) != (uncertainty_mask is None):
            raise ValueError(
                "provider_uncertainty and uncertainty_valid_mask must be present together"
            )
        if uncertainty is None:
            if uncertainty_semantics != SPARSE_UNCERTAINTY_ABSENT:
                raise ValueError(
                    "uncertainty_semantics must be 'absent' when uncertainty is absent"
                )
            validated_uncertainty = None
            validated_uncertainty_mask = None
        else:
            if uncertainty_semantics == SPARSE_UNCERTAINTY_ABSENT:
                raise ValueError(
                    "uncertainty_semantics must describe present provider uncertainty"
                )
            validated_uncertainty = _float_array(
                uncertainty,
                name="provider_uncertainty",
                dtype=numpy_dtype,
            )
            validated_uncertainty_mask = _bool_array(
                uncertainty_mask,
                name="uncertainty_valid_mask",
            )
            if validated_uncertainty.ndim != 3:
                raise ValueError(
                    "provider_uncertainty must have shape (T, N, 1) or (T, N, 3)"
                )
            if validated_uncertainty.shape[:2] != position.shape[:2]:
                raise ValueError(
                    "provider_uncertainty dimensions must match position"
                )
            if validated_uncertainty.shape[-1] not in {1, 3}:
                raise ValueError(
                    "provider_uncertainty final dimension must be one or three"
                )
            if validated_uncertainty_mask.shape != position.shape[:2]:
                raise ValueError("uncertainty_valid_mask must have shape (T, N)")
            if np.any(validated_uncertainty_mask & ~valid_mask):
                raise ValueError(
                    "uncertainty_valid_mask must be a subset of valid_mask"
                )
            if not np.all(np.isfinite(validated_uncertainty)):
                raise ValueError("provider_uncertainty must contain only finite values")

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "coordinate_semantics", coordinate_semantics)
        object.__setattr__(self, "identity_semantics", identity_semantics)
        object.__setattr__(self, "uncertainty_semantics", uncertainty_semantics)
        object.__setattr__(self, "dense_storage_dtype", storage_dtype)
        object.__setattr__(self, "frame_indices", immutable_array(frame_indices))
        object.__setattr__(self, "point_ids", immutable_array(point_ids))
        object.__setattr__(self, "position", immutable_array(position))
        object.__setattr__(self, "valid_mask", immutable_array(valid_mask))
        object.__setattr__(
            self,
            "provider_uncertainty",
            None
            if validated_uncertainty is None
            else immutable_array(validated_uncertainty),
        )
        object.__setattr__(
            self,
            "uncertainty_valid_mask",
            None
            if validated_uncertainty_mask is None
            else immutable_array(validated_uncertainty_mask),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(T, N)`` for the sparse forecast grid."""

        return self.valid_mask.shape

    @property
    def start_frame(self) -> int:
        return int(self.frame_indices[0])

    @property
    def stop_frame(self) -> int:
        return int(self.frame_indices[-1]) + 1

    @property
    def content_id(self) -> str:
        """Return a canonical content ID independent of NPZ ZIP metadata."""

        digest = hashlib.sha256()
        _update_text(digest, SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA)
        digest.update(
            SPARSE_PREDICTION_WINDOW_NPZ_VERSION.to_bytes(
                8,
                "little",
                signed=False,
            )
        )
        for value in (
            self.dense_storage_dtype,
            self.window_id,
            self.coordinate_semantics,
            self.identity_semantics,
            self.uncertainty_semantics,
            _metadata_json(self.metadata),
        ):
            _update_text(digest, value)
        for value in (
            self.frame_indices,
            self.point_ids,
            self.position,
            self.valid_mask,
        ):
            _update_array(digest, value)
        digest.update(b"1" if self.provider_uncertainty is not None else b"0")
        if self.provider_uncertainty is not None:
            assert self.uncertainty_valid_mask is not None
            _update_array(digest, self.provider_uncertainty)
            _update_array(digest, self.uncertainty_valid_mask)
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "content_id": self.content_id,
            "frame_count": int(self.frame_indices.size),
            "point_count": int(self.point_ids.size),
            "valid_position_count": int(np.count_nonzero(self.valid_mask)),
            "coordinate_semantics": self.coordinate_semantics,
            "identity_semantics": self.identity_semantics,
            "uncertainty_semantics": self.uncertainty_semantics,
            "uncertainty_present": self.provider_uncertainty is not None,
            "dense_storage_dtype": self.dense_storage_dtype,
        }

    def to_npz(
        self,
        path: str | Path,
        *,
        storage_dtype: DenseStorageDType | None = None,
    ) -> None:
        """Write a versioned no-clobber NPZ artifact.

        A precision change is allowed only through the explicit ``storage_dtype``
        argument and is recorded in the archive metadata.
        """

        target = Path(path)
        if target.is_symlink():
            raise ValueError("sparse prediction output path is a symbolic link")
        target.parent.mkdir(parents=True, exist_ok=True)
        selected = (
            self.dense_storage_dtype
            if storage_dtype is None
            else _storage_dtype(storage_dtype)
        )
        dtype = _numpy_dtype(selected)
        payload: dict[str, np.ndarray] = {
            "schema_name": np.asarray(SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA),
            "schema_version": np.asarray(
                SPARSE_PREDICTION_WINDOW_NPZ_VERSION,
                dtype=np.int64,
            ),
            "dense_storage_dtype": np.asarray(selected),
            "window_id": np.asarray(self.window_id),
            "coordinate_semantics": np.asarray(self.coordinate_semantics),
            "identity_semantics": np.asarray(self.identity_semantics),
            "uncertainty_semantics": np.asarray(self.uncertainty_semantics),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int64),
            "point_ids": np.asarray(self.point_ids, dtype=np.int64),
            "position": np.asarray(self.position, dtype=dtype),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
            "metadata_json": np.asarray(_metadata_json(self.metadata)),
        }
        if self.provider_uncertainty is not None:
            assert self.uncertainty_valid_mask is not None
            payload["provider_uncertainty"] = np.asarray(
                self.provider_uncertainty,
                dtype=dtype,
            )
            payload["uncertainty_valid_mask"] = np.asarray(
                self.uncertainty_valid_mask,
                dtype=bool,
            )
        try:
            with target.open("xb") as stream:
                np.savez(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    @classmethod
    def from_npz(cls, path: str | Path) -> SparsePredictionWindow:
        """Load and strictly validate one versioned sparse artifact."""

        source = Path(path)
        if source.is_symlink():
            raise ValueError("sparse prediction input path is a symbolic link")
        try:
            with np.load(source, allow_pickle=False) as archive:
                fields = set(archive.files)
                allowed = _REQUIRED_ARCHIVE_FIELDS | _OPTIONAL_ARCHIVE_FIELDS
                missing = sorted(_REQUIRED_ARCHIVE_FIELDS - fields)
                extra = sorted(fields - allowed)
                if missing or extra:
                    raise ValueError(
                        "sparse prediction archive fields changed; "
                        f"missing={missing}, extra={extra}"
                    )
                uncertainty_present = "provider_uncertainty" in fields
                mask_present = "uncertainty_valid_mask" in fields
                if uncertainty_present != mask_present:
                    raise ValueError(
                        "sparse prediction uncertainty fields must be present together"
                    )
                schema = _scalar_text(archive["schema_name"], name="schema_name")
                version = _scalar_integer(
                    archive["schema_version"],
                    name="schema_version",
                )
                if schema != SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA:
                    raise ValueError("unsupported sparse prediction archive schema")
                if version != SPARSE_PREDICTION_WINDOW_NPZ_VERSION:
                    raise ValueError("unsupported sparse prediction archive version")
                storage = _storage_dtype(
                    _scalar_text(
                        archive["dense_storage_dtype"],
                        name="dense_storage_dtype",
                    )
                )
                dtype = _numpy_dtype(storage)
                if archive["position"].dtype != dtype:
                    raise ValueError(
                        "position dtype disagrees with dense_storage_dtype"
                    )
                if uncertainty_present and archive["provider_uncertainty"].dtype != dtype:
                    raise ValueError(
                        "provider_uncertainty dtype disagrees with dense_storage_dtype"
                    )
                if archive["frame_indices"].dtype != np.dtype(np.int64):
                    raise ValueError("frame_indices must use int64")
                if archive["point_ids"].dtype != np.dtype(np.int64):
                    raise ValueError("point_ids must use int64")
                if archive["valid_mask"].dtype != np.dtype(bool):
                    raise ValueError("valid_mask must use bool")
                if mask_present and archive["uncertainty_valid_mask"].dtype != np.dtype(bool):
                    raise ValueError("uncertainty_valid_mask must use bool")
                metadata = _load_metadata_json(
                    _scalar_text(archive["metadata_json"], name="metadata_json")
                )
                return cls(
                    window_id=_scalar_text(archive["window_id"], name="window_id"),
                    frame_indices=np.array(archive["frame_indices"], copy=True),
                    point_ids=np.array(archive["point_ids"], copy=True),
                    position=np.array(archive["position"], copy=True),
                    valid_mask=np.array(archive["valid_mask"], copy=True),
                    coordinate_semantics=_scalar_text(
                        archive["coordinate_semantics"],
                        name="coordinate_semantics",
                    ),
                    identity_semantics=_scalar_text(
                        archive["identity_semantics"],
                        name="identity_semantics",
                    ),
                    uncertainty_semantics=_scalar_text(
                        archive["uncertainty_semantics"],
                        name="uncertainty_semantics",
                    ),
                    provider_uncertainty=(
                        np.array(archive["provider_uncertainty"], copy=True)
                        if uncertainty_present
                        else None
                    ),
                    uncertainty_valid_mask=(
                        np.array(archive["uncertainty_valid_mask"], copy=True)
                        if mask_present
                        else None
                    ),
                    dense_storage_dtype=storage,
                    metadata=metadata,
                )
        except OSError as error:
            raise ValueError("cannot read sparse prediction archive") from error


__all__ = [
    "SPARSE_PREDICTION_WINDOW_NPZ_SCHEMA",
    "SPARSE_PREDICTION_WINDOW_NPZ_VERSION",
    "SPARSE_UNCERTAINTY_ABSENT",
    "SparsePredictionWindow",
]
