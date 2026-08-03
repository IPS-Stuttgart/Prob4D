"""Content-addressed memory-mapped execution storage for prediction bundles.

Portable MotionCrafter and provider artifacts remain unchanged. This module
materializes verified NPZ prediction bundles into uncompressed NPY members that
NumPy can open read-only with ``mmap_mode='r'``. The store is an explicit
execution cache: each member and manifest is hashed, source-manifest identity is
retained, and existing fusion/alignment code can consume the resulting
``MMapPredictionWindow`` because it preserves the ``PredictionWindow`` interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from ._immutable_json import plain_json
from .data import DENSE_STORAGE_DTYPES, DenseStorageDType, PredictionWindow

PREDICTION_WINDOW_STORE_SCHEMA: Final = "prob4d.prediction-window-store"
PREDICTION_WINDOW_STORE_VERSION: Final = 1
PREDICTION_BUNDLE_STORE_SCHEMA: Final = "prob4d.prediction-bundle-store"
PREDICTION_BUNDLE_STORE_VERSION: Final = 1
PREDICTION_STORE_MANIFEST: Final = "manifest.json"

_WINDOW_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "store_id",
        "window_id",
        "dense_storage_dtype",
        "shape",
        "fields",
        "metadata",
    }
)
_BUNDLE_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "store_id",
        "source_manifest_sha256",
        "dense_storage_dtype",
        "source_metadata",
        "overlap_windows",
        "disjoint_baseline",
        "latent_linear_baseline",
    }
)
_FIELD_DESCRIPTOR_FIELDS: Final = frozenset(
    {"path", "sha256", "bytes", "dtype", "shape"}
)
_REQUIRED_WINDOW_FIELDS: Final = frozenset(
    {"frame_indices", "point_map", "valid_mask"}
)
_OPTIONAL_WINDOW_FIELDS: Final = frozenset(
    {"scene_flow", "deform_mask", "ray_directions"}
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validated_dense_storage_dtype(value: object) -> DenseStorageDType:
    normalized = str(value)
    if normalized not in DENSE_STORAGE_DTYPES:
        raise ValueError(
            "dense_storage_dtype must be one of " + ", ".join(DENSE_STORAGE_DTYPES)
        )
    return cast(DenseStorageDType, normalized)


def _dense_dtype(value: DenseStorageDType) -> np.dtype[np.floating]:
    return np.dtype(np.float32 if value == "float32" else np.float64)


def _safe_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return path.as_posix()


def _resolve_member(root: Path, value: object, *, name: str) -> Path:
    relative = _safe_relative_path(value, name=name)
    root_resolved = root.resolve()
    if root.is_symlink():
        raise ValueError(f"{name} root must not be a symlink")
    candidate = root_resolved.joinpath(*PurePosixPath(relative).parts)
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes the prediction store") from error
    return resolved


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _field_descriptor(path: Path, *, root: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _compute_store_id(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("store_id", None)
    return _sha256_bytes(_canonical_json(payload))


def _validate_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, name: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _validate_finite_vectors(
    value: np.ndarray,
    active_mask: np.ndarray,
    *,
    name: str,
    chunk_size: int = 65_536,
) -> None:
    flat_value = value.reshape(-1, value.shape[-1])
    flat_mask = active_mask.reshape(-1)
    for start in range(0, flat_value.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_value.shape[0])
        selected = flat_mask[start:stop]
        if np.any(selected) and not np.all(np.isfinite(flat_value[start:stop][selected])):
            raise ValueError(f"active {name} entries must be finite")


class MMapPredictionWindow(PredictionWindow):
    """Validated read-only prediction window backed by NPY memory maps.

    Unlike the public ``PredictionWindow`` constructor, this execution-cache
    subclass does not defensively copy dense arrays. Files are hash-verified
    before opening, arrays are read-only, and validation is chunked. Portable
    artifacts and content-addressed provider semantics remain owned by the
    original NPZ prediction bundle.
    """

    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        dense_storage_dtype = _validated_dense_storage_dtype(self.dense_storage_dtype)
        expected_dense_dtype = _dense_dtype(dense_storage_dtype)
        frame_indices = np.asarray(self.frame_indices)
        point_map = np.asarray(self.point_map)
        valid_mask = np.asarray(self.valid_mask)

        if not window_id:
            raise ValueError("window_id must not be empty")
        if frame_indices.dtype != np.dtype(np.int64):
            raise ValueError("stored frame_indices must use int64")
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError("frame_indices must be a non-empty one-dimensional array")
        if np.any(frame_indices < 0) or np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be non-negative and strictly increasing")
        if point_map.dtype != expected_dense_dtype:
            raise ValueError("stored point_map dtype differs from dense_storage_dtype")
        if point_map.ndim != 4 or point_map.shape[-1] != 3:
            raise ValueError("point_map must have shape (T, H, W, 3)")
        if valid_mask.dtype != np.dtype(bool) or valid_mask.shape != point_map.shape[:-1]:
            raise ValueError("valid_mask must be Boolean with shape (T, H, W)")
        if point_map.shape[0] != frame_indices.size:
            raise ValueError("frame_indices length must match point_map time dimension")
        _validate_finite_vectors(point_map, valid_mask, name="point_map")

        scene_flow = None if self.scene_flow is None else np.asarray(self.scene_flow)
        deform_mask = None if self.deform_mask is None else np.asarray(self.deform_mask)
        rays = None if self.ray_directions is None else np.asarray(self.ray_directions)
        if (scene_flow is None) != (deform_mask is None):
            raise ValueError("scene_flow and deform_mask must either both be present or absent")
        if scene_flow is not None and deform_mask is not None:
            if scene_flow.dtype != expected_dense_dtype or scene_flow.shape != point_map.shape:
                raise ValueError("stored scene_flow must match point_map shape and dtype")
            if deform_mask.dtype != np.dtype(bool) or deform_mask.shape != valid_mask.shape:
                raise ValueError("stored deform_mask must match valid_mask")
            if np.any(deform_mask & ~valid_mask):
                raise ValueError("deform_mask must be a subset of valid_mask")
            _validate_finite_vectors(scene_flow, deform_mask, name="scene_flow")
        if rays is not None:
            if rays.dtype != expected_dense_dtype or rays.shape != point_map.shape:
                raise ValueError("stored ray_directions must match point_map shape and dtype")
            _validate_finite_vectors(rays, valid_mask, name="ray_directions")
            flat_rays = rays.reshape(-1, 3)
            flat_valid = valid_mask.reshape(-1)
            for start in range(0, flat_rays.shape[0], 65_536):
                stop = min(start + 65_536, flat_rays.shape[0])
                selected = flat_valid[start:stop]
                if not np.any(selected):
                    continue
                norms = np.linalg.norm(flat_rays[start:stop][selected], axis=-1)
                if not np.allclose(norms, 1.0, atol=1e-6, rtol=1e-5):
                    raise ValueError("stored ray_directions must already be normalized")

        arrays = (frame_indices, point_map, valid_mask, scene_flow, deform_mask, rays)
        for array in arrays:
            if array is not None:
                array.setflags(write=False)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "dense_storage_dtype", dense_storage_dtype)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "point_map", point_map)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "scene_flow", scene_flow)
        object.__setattr__(self, "deform_mask", deform_mask)
        object.__setattr__(self, "ray_directions", rays)


def _window_arrays(
    window: PredictionWindow, dense_storage_dtype: DenseStorageDType
) -> dict[str, np.ndarray]:
    dense_dtype = _dense_dtype(dense_storage_dtype)
    arrays: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(window.frame_indices, dtype=np.int64),
        "point_map": np.asarray(window.point_map, dtype=dense_dtype),
        "valid_mask": np.asarray(window.valid_mask, dtype=bool),
    }
    if window.scene_flow is not None:
        assert window.deform_mask is not None
        arrays["scene_flow"] = np.asarray(window.scene_flow, dtype=dense_dtype)
        arrays["deform_mask"] = np.asarray(window.deform_mask, dtype=bool)
    if window.ray_directions is not None:
        arrays["ray_directions"] = np.asarray(window.ray_directions, dtype=dense_dtype)
    return arrays


def write_prediction_window_store(
    window: PredictionWindow,
    destination: str | Path,
    *,
    dense_storage_dtype: DenseStorageDType | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically materialize one prediction window as hash-bound NPY members."""

    target = Path(destination)
    if target.exists():
        raise ValueError(f"prediction-window store already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    storage_dtype = _validated_dense_storage_dtype(
        window.dense_storage_dtype if dense_storage_dtype is None else dense_storage_dtype
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        descriptors: dict[str, dict[str, object]] = {}
        for field, array in _window_arrays(window, storage_dtype).items():
            path = temporary / f"{field}.npy"
            _write_npy(path, array)
            descriptors[field] = _field_descriptor(path, root=temporary)
        payload: dict[str, Any] = {
            "schema_name": PREDICTION_WINDOW_STORE_SCHEMA,
            "schema_version": PREDICTION_WINDOW_STORE_VERSION,
            "window_id": window.window_id,
            "dense_storage_dtype": storage_dtype,
            "shape": list(window.shape),
            "fields": descriptors,
            "metadata": {
                **({} if metadata is None else plain_json(metadata)),
                "execution_cache_only": True,
                "portable_provider_artifact_semantics_unchanged": True,
            },
        }
        payload["store_id"] = _compute_store_id(payload)
        _atomic_write_json(temporary / PREDICTION_STORE_MANIFEST, payload)
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / PREDICTION_STORE_MANIFEST


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _load_field(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    name: str,
    verify_hashes: bool,
) -> np.ndarray:
    _validate_exact_fields(descriptor, _FIELD_DESCRIPTOR_FIELDS, name=name)
    path = _resolve_member(root, descriptor.get("path"), name=f"{name} path")
    if not path.is_file():
        raise ValueError(f"{name} file is missing")
    expected_bytes = _require_nonnegative_integer(
        descriptor.get("bytes"), name=f"{name} bytes"
    )
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{name} byte count mismatch")
    expected_sha = _require_sha256(descriptor.get("sha256"), name=f"{name} sha256")
    if verify_hashes and _sha256_file(path) != expected_sha:
        raise ValueError(f"{name} SHA-256 mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_dtype = str(descriptor.get("dtype", ""))
    if array.dtype.str != expected_dtype:
        raise ValueError(f"{name} dtype differs from its descriptor")
    expected_shape = descriptor.get("shape")
    if not isinstance(expected_shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expected_shape
    ):
        raise ValueError(f"{name} shape descriptor is invalid")
    if list(array.shape) != expected_shape:
        raise ValueError(f"{name} shape differs from its descriptor")
    array.setflags(write=False)
    return array


def load_prediction_window_store(
    directory: str | Path, *, verify_hashes: bool = True
) -> MMapPredictionWindow:
    """Verify and open one prediction-window execution store read-only."""

    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prediction-window store is not a regular directory: {root}")
    manifest_path = root / PREDICTION_STORE_MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("prediction-window store manifest must not be a symlink")
    manifest = _load_json_object(
        manifest_path, name="prediction-window store manifest"
    )
    _validate_exact_fields(
        manifest, _WINDOW_MANIFEST_FIELDS, name="prediction-window store manifest"
    )
    if manifest.get("schema_name") != PREDICTION_WINDOW_STORE_SCHEMA:
        raise ValueError("unsupported prediction-window store schema")
    if manifest.get("schema_version") != PREDICTION_WINDOW_STORE_VERSION:
        raise ValueError("unsupported prediction-window store version")
    declared_id = _require_sha256(manifest.get("store_id"), name="window store_id")
    if _compute_store_id(manifest) != declared_id:
        raise ValueError("prediction-window store ID does not match its manifest")
    storage_dtype = _validated_dense_storage_dtype(manifest.get("dense_storage_dtype"))
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("prediction-window store metadata must be a mapping")
    if metadata.get("execution_cache_only") is not True:
        raise ValueError("prediction-window store must declare execution-cache-only semantics")
    if metadata.get("portable_provider_artifact_semantics_unchanged") is not True:
        raise ValueError("prediction-window store must preserve provider artifact semantics")
    fields = manifest.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("prediction-window store fields must be a mapping")
    field_names = set(fields)
    if not _REQUIRED_WINDOW_FIELDS.issubset(field_names):
        raise ValueError("prediction-window store lacks required fields")
    unknown = field_names - _REQUIRED_WINDOW_FIELDS - _OPTIONAL_WINDOW_FIELDS
    if unknown:
        raise ValueError(f"prediction-window store has unknown fields: {sorted(unknown)}")
    if ("scene_flow" in fields) != ("deform_mask" in fields):
        raise ValueError("stored scene_flow and deform_mask must be present together")
    paths: list[str] = []
    for name in fields:
        descriptor = fields[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"prediction field {name} descriptor must be a mapping")
        paths.append(
            _safe_relative_path(
                descriptor.get("path"),
                name=f"prediction field {name} path",
            )
        )
    if len(paths) != len(set(paths)):
        raise ValueError("prediction-window store field paths must be unique")
    arrays = {
        name: _load_field(
            root,
            cast(Mapping[str, Any], fields[name]),
            name=f"prediction field {name}",
            verify_hashes=verify_hashes,
        )
        for name in fields
    }
    shape = manifest.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("prediction-window store shape must contain T, H, W")
    window = MMapPredictionWindow(
        window_id=str(manifest.get("window_id", "")),
        frame_indices=arrays["frame_indices"],
        point_map=arrays["point_map"],
        valid_mask=arrays["valid_mask"],
        scene_flow=arrays.get("scene_flow"),
        deform_mask=arrays.get("deform_mask"),
        ray_directions=arrays.get("ray_directions"),
        dense_storage_dtype=storage_dtype,
    )
    if list(window.shape) != shape:
        raise ValueError("prediction-window store shape differs from loaded arrays")
    return window


def _store_reference(path: Path, *, root: Path, window: PredictionWindow) -> dict[str, Any]:
    manifest = _load_json_object(path / PREDICTION_STORE_MANIFEST, name="window manifest")
    return {
        "window_id": window.window_id,
        "path": path.relative_to(root).as_posix(),
        "store_id": manifest["store_id"],
    }


def materialize_prediction_bundle_store(
    source_manifest: str | Path,
    destination: str | Path,
    *,
    dense_storage_dtype: DenseStorageDType = "float32",
) -> Path:
    """Convert one verified NPZ bundle into an atomic memory-mapped execution store."""

    from .motioncrafter_integrity import (
        resolve_motioncrafter_member,
        verify_motioncrafter_prediction_manifest,
    )

    source = Path(source_manifest).resolve()
    verify_motioncrafter_prediction_manifest(source, verify_hashes=True)
    source_bytes = source.read_bytes()
    source_metadata = _load_json_object(source, name="prediction source manifest")
    root = source.parent
    target = Path(destination)
    if target.exists():
        raise ValueError(f"prediction-bundle store already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        overlap_values = source_metadata.get("overlap_windows")
        if not isinstance(overlap_values, list) or not overlap_values:
            raise ValueError("prediction source manifest requires overlap windows")
        overlap_references: list[dict[str, Any]] = []
        for index, item in enumerate(overlap_values):
            if not isinstance(item, Mapping):
                raise ValueError(f"prediction overlap window {index} must be a mapping")
            window_id = str(item.get("window_id", ""))
            source_path = resolve_motioncrafter_member(
                root,
                item.get("path"),
                name=f"overlap window {window_id!r} path",
            )
            window = PredictionWindow.from_npz(
                source_path,
                start_frame=item.get("start_frame"),
                window_id=window_id,
                dense_storage_dtype=dense_storage_dtype,
            )
            relative = Path("overlap") / f"{index:04d}"
            write_prediction_window_store(
                window,
                temporary / relative,
                dense_storage_dtype=dense_storage_dtype,
                metadata={
                    "source_manifest_sha256": _sha256_bytes(source_bytes),
                    "source_role": "overlap_window",
                    "source_index": index,
                },
            )
            overlap_references.append(
                _store_reference(temporary / relative, root=temporary, window=window)
            )
            del window
        baseline_references: dict[str, dict[str, Any]] = {}
        baseline_specs = (
            ("disjoint_baseline", "baseline_disjoint"),
            ("latent_linear_baseline", "baseline_latent_linear"),
        )
        for role, window_id in baseline_specs:
            source_path = resolve_motioncrafter_member(
                root,
                source_metadata.get(role),
                name=f"{role} path",
            )
            window = PredictionWindow.from_npz(
                source_path,
                start_frame=0,
                window_id=window_id,
                dense_storage_dtype=dense_storage_dtype,
            )
            relative = Path(role)
            write_prediction_window_store(
                window,
                temporary / relative,
                dense_storage_dtype=dense_storage_dtype,
                metadata={
                    "source_manifest_sha256": _sha256_bytes(source_bytes),
                    "source_role": role,
                },
            )
            baseline_references[role] = _store_reference(
                temporary / relative, root=temporary, window=window
            )
            del window
        payload: dict[str, Any] = {
            "schema_name": PREDICTION_BUNDLE_STORE_SCHEMA,
            "schema_version": PREDICTION_BUNDLE_STORE_VERSION,
            "source_manifest_sha256": _sha256_bytes(source_bytes),
            "dense_storage_dtype": _validated_dense_storage_dtype(
                dense_storage_dtype
            ),
            "source_metadata": plain_json(source_metadata),
            "overlap_windows": overlap_references,
            "disjoint_baseline": baseline_references["disjoint_baseline"],
            "latent_linear_baseline": baseline_references[
                "latent_linear_baseline"
            ],
        }
        payload["store_id"] = _compute_store_id(payload)
        _atomic_write_json(temporary / PREDICTION_STORE_MANIFEST, payload)
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / PREDICTION_STORE_MANIFEST


def _load_store_reference(
    root: Path,
    value: object,
    *,
    name: str,
    verify_hashes: bool,
) -> MMapPredictionWindow:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    _validate_exact_fields(
        value, frozenset({"window_id", "path", "store_id"}), name=name
    )
    path = _resolve_member(root, value.get("path"), name=f"{name} path")
    window = load_prediction_window_store(path, verify_hashes=verify_hashes)
    if window.window_id != value.get("window_id"):
        raise ValueError(f"{name} window ID differs from its store")
    manifest = _load_json_object(path / PREDICTION_STORE_MANIFEST, name=name)
    if manifest.get("store_id") != _require_sha256(
        value.get("store_id"), name=f"{name} store_id"
    ):
        raise ValueError(f"{name} store ID differs from its reference")
    return window


def load_prediction_bundle_store(
    directory: str | Path, *, verify_hashes: bool = True
):
    """Verify and open a complete bundle through read-only NPY memory maps."""

    from .io import PredictionBundle

    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prediction-bundle store is not a regular directory: {root}")
    manifest_path = root / PREDICTION_STORE_MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("prediction-bundle store manifest must not be a symlink")
    manifest = _load_json_object(manifest_path, name="prediction-bundle store manifest")
    _validate_exact_fields(
        manifest, _BUNDLE_MANIFEST_FIELDS, name="prediction-bundle store manifest"
    )
    if manifest.get("schema_name") != PREDICTION_BUNDLE_STORE_SCHEMA:
        raise ValueError("unsupported prediction-bundle store schema")
    if manifest.get("schema_version") != PREDICTION_BUNDLE_STORE_VERSION:
        raise ValueError("unsupported prediction-bundle store version")
    declared_id = _require_sha256(manifest.get("store_id"), name="bundle store_id")
    if _compute_store_id(manifest) != declared_id:
        raise ValueError("prediction-bundle store ID does not match its manifest")
    source_sha = _require_sha256(
        manifest.get("source_manifest_sha256"), name="source manifest sha256"
    )
    storage_dtype = _validated_dense_storage_dtype(manifest.get("dense_storage_dtype"))
    overlap_values = manifest.get("overlap_windows")
    if not isinstance(overlap_values, list) or not overlap_values:
        raise ValueError("prediction-bundle store requires overlap windows")
    overlap = [
        _load_store_reference(
            root,
            value,
            name=f"overlap window {index}",
            verify_hashes=verify_hashes,
        )
        for index, value in enumerate(overlap_values)
    ]
    overlap.sort(key=lambda window: window.start_frame)
    disjoint = _load_store_reference(
        root,
        manifest.get("disjoint_baseline"),
        name="disjoint baseline",
        verify_hashes=verify_hashes,
    )
    latent = _load_store_reference(
        root,
        manifest.get("latent_linear_baseline"),
        name="latent-linear baseline",
        verify_hashes=verify_hashes,
    )
    source_metadata = manifest.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("prediction-bundle source_metadata must be a mapping")
    metadata = dict(plain_json(source_metadata))
    metadata["prediction_execution_store"] = {
        "schema_name": PREDICTION_BUNDLE_STORE_SCHEMA,
        "schema_version": PREDICTION_BUNDLE_STORE_VERSION,
        "store_id": declared_id,
        "source_manifest_sha256": source_sha,
        "dense_storage_dtype": storage_dtype,
        "verify_hashes": bool(verify_hashes),
    }
    return PredictionBundle(
        manifest_path=manifest_path,
        overlap_windows=overlap,
        disjoint_baseline=disjoint,
        latent_linear_baseline=latent,
        metadata=metadata,
    )


def prediction_bundle_store_summary(directory: str | Path) -> dict[str, object]:
    """Return validated execution-store identity and retained-array accounting."""

    bundle = load_prediction_bundle_store(directory, verify_hashes=True)
    store = bundle.metadata["prediction_execution_store"]
    return {
        **store,
        **bundle.dense_storage_summary(),
        "manifest_path": str(bundle.manifest_path),
    }


__all__ = [
    "MMapPredictionWindow",
    "PREDICTION_BUNDLE_STORE_SCHEMA",
    "PREDICTION_BUNDLE_STORE_VERSION",
    "PREDICTION_STORE_MANIFEST",
    "PREDICTION_WINDOW_STORE_SCHEMA",
    "PREDICTION_WINDOW_STORE_VERSION",
    "load_prediction_bundle_store",
    "load_prediction_window_store",
    "materialize_prediction_bundle_store",
    "prediction_bundle_store_summary",
    "write_prediction_window_store",
]
