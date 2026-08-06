"""Irreversibly read-only NumPy arrays for validated artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np


def immutable_array(
    values: Any,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    """Return a defensive array whose write flag cannot be re-enabled.

    Clearing the write flag on an owned NumPy allocation is reversible through
    ``array.setflags(write=True)``.  Claim-bearing values instead use an
    immutable ``bytes`` buffer, preserving the exact dtype, shape, and C-order
    values while making both direct writes and write-flag restoration fail.
    """

    array = np.asarray(values, dtype=dtype)
    if array.dtype.hasobject:
        raise ValueError("immutable arrays must not contain Python objects")
    original_shape = array.shape
    contiguous = np.ascontiguousarray(array)
    payload = contiguous.tobytes(order="C")
    result = np.frombuffer(payload, dtype=contiguous.dtype).reshape(original_shape)
    if result.flags.writeable:
        raise RuntimeError("immutable array construction produced writable storage")
    return result


def immutable_integer_array(values: Any, *, name: str) -> np.ndarray:
    """Return immutable int64 data without coercing non-integer inputs."""

    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers")
    if array.dtype.kind == "u" and array.size:
        maximum = int(np.max(array))
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains an integer outside int64 range")
    return immutable_array(array, dtype=np.int64)


__all__ = ["immutable_array", "immutable_integer_array"]
