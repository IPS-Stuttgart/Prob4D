"""Internal validation helpers for observation artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

_REQUIRED_PROVENANCE = {
    "producer",
    "producer_revision",
    "source_model",
    "source_model_revision",
    "source_manifest_sha256",
    "method",
}


def _validate_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")
    return text


def _json_value(value: Any, *, path: str = "provenance") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            output[key] = _json_value(item, path=f"{path}.{key}")
        return output
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def _integer_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(raw.dtype, np.integer):
        integer_valued = (
            np.issubdtype(raw.dtype, np.floating)
            and np.all(np.isfinite(raw))
            and np.all(np.equal(raw, np.floor(raw)))
        )
        if not integer_valued:
            raise ValueError(f"{name} must contain integers")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite")
    return np.asarray(raw, dtype=np.int64)


def _integer_scalar(value: Any, *, name: str) -> int:
    array = _integer_array([value], name=name, ndim=1)
    return int(array[0])


def _validate_covariances(
    covariance: np.ndarray,
    *,
    name: str,
    active: np.ndarray | None = None,
    chunk_size: int = 65_536,
) -> None:
    matrices = np.asarray(covariance)
    if not np.issubdtype(matrices.dtype, np.floating):
        raise ValueError(f"{name} must use a floating-point dtype")
    if matrices.shape[-2:] not in {(3, 3), (7, 7)}:
        raise ValueError(f"{name} must end in shape (3, 3) or (7, 7)")
    flattened = matrices.reshape(-1, matrices.shape[-1], matrices.shape[-1])
    active_flat = None if active is None else np.asarray(active, dtype=bool).reshape(-1)
    if active_flat is not None and active_flat.size != flattened.shape[0]:
        raise ValueError(f"{name} active mask has incompatible shape")

    for start in range(0, flattened.shape[0], chunk_size):
        stop = min(start + chunk_size, flattened.shape[0])
        chunk = np.asarray(flattened[start:stop], dtype=np.float64)
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"{name} contains non-finite values")
        if active_flat is not None:
            chunk = chunk[active_flat[start:stop]]
        if chunk.size == 0:
            continue
        transpose = np.swapaxes(chunk, -1, -2)
        if np.max(np.abs(chunk - transpose)) > 1e-9:
            raise ValueError(f"{name} must be symmetric")
        eigenvalues = np.linalg.eigvalsh(0.5 * (chunk + transpose))
        scale = np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
        if np.any(np.min(eigenvalues, axis=1) < -1e-10 * scale):
            raise ValueError(f"{name} must be positive semidefinite")


def pack_symmetric_covariance(covariance: np.ndarray) -> np.ndarray:
    """Pack the upper triangle of dense 3x3 covariance matrices into six values."""

    covariance = np.asarray(covariance)
    if covariance.shape[-2:] != (3, 3):
        raise ValueError("covariance must end in shape (3, 3)")
    return covariance[..., (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def unpack_symmetric_covariance(packed: np.ndarray) -> np.ndarray:
    """Restore dense 3x3 covariance matrices from six upper-triangle values."""

    packed = np.asarray(packed)
    if packed.shape[-1] != 6:
        raise ValueError("packed covariance must end in six values")
    covariance = np.empty(packed.shape[:-1] + (3, 3), dtype=packed.dtype)
    covariance[..., 0, 0] = packed[..., 0]
    covariance[..., 0, 1] = covariance[..., 1, 0] = packed[..., 1]
    covariance[..., 0, 2] = covariance[..., 2, 0] = packed[..., 2]
    covariance[..., 1, 1] = packed[..., 3]
    covariance[..., 1, 2] = covariance[..., 2, 1] = packed[..., 4]
    covariance[..., 2, 2] = packed[..., 5]
    return covariance
