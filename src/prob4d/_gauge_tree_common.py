"""Validation and identity helpers for sparse gauge-tree priors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array, immutable_integer_array

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]

GAUGE_DIMENSION: Final = 7
GAUGE_TREE_PRIOR_SCHEMA: Final = "prob4d.gauge-tree-square-root-prior"
GAUGE_TREE_PRIOR_VERSION: Final = 1
GAUGE_TREE_PRIOR_SEMANTICS: Final = (
    "zero-mean-linearized-causal-tree-independent-innovations-v1"
)


def canonical_array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sha256(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def validate_gauge_ids(values: Sequence[str]) -> tuple[str, ...]:
    gauge_ids = tuple(str(value) for value in values)
    if not gauge_ids or any(not value for value in gauge_ids):
        raise ValueError("gauge_ids must contain nonempty strings")
    if len(set(gauge_ids)) != len(gauge_ids):
        raise ValueError("gauge_ids must be unique")
    return gauge_ids


def validate_parent_indices(values: Any, *, gauge_count: int) -> IntArray:
    parent_indices = immutable_integer_array(values, name="parent_indices")
    if parent_indices.shape != (gauge_count,):
        raise ValueError("parent_indices must have shape (K,)")
    if int(parent_indices[0]) != -1:
        raise ValueError("the first gauge must be the unique root with parent -1")
    if gauge_count > 1 and np.any(parent_indices[1:] < 0):
        raise ValueError("only the first gauge may be a root")
    for index in range(1, gauge_count):
        if int(parent_indices[index]) >= index:
            raise ValueError("every parent must precede its child in causal order")
    return parent_indices


def strict_scale_tril(covariance: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    expected = (GAUGE_DIMENSION, GAUGE_DIMENSION)
    if matrix.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(matrix, symmetric, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    try:
        return np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be strictly positive definite") from error


def joint_covariance_sha256(value: Any) -> str:
    return str(canonical_array_descriptor(np.asarray(value, dtype=np.float64))["sha256"])


def validate_factor_arrays(
    gauge_ids: Sequence[str],
    parent_indices: Any,
    transition_matrices: Any,
    innovation_scale_tril: Any,
    source_digest: str | None,
    semantics: str,
) -> tuple[tuple[str, ...], IntArray, FloatArray, FloatArray, str | None]:
    ids = validate_gauge_ids(gauge_ids)
    parents = validate_parent_indices(parent_indices, gauge_count=len(ids))
    transitions = np.asarray(transition_matrices, dtype=np.float64)
    scales = np.asarray(innovation_scale_tril, dtype=np.float64)
    expected = (len(ids), GAUGE_DIMENSION, GAUGE_DIMENSION)
    if transitions.shape != expected:
        raise ValueError(f"transition_matrices must have shape {expected}")
    if scales.shape != expected:
        raise ValueError(f"innovation_scale_tril must have shape {expected}")
    if not np.all(np.isfinite(transitions)) or not np.all(np.isfinite(scales)):
        raise ValueError("gauge-tree factors must be finite")
    if np.any(transitions[0] != 0.0):
        raise ValueError("the root transition matrix must be exactly zero")
    if not np.array_equal(scales, np.tril(scales)):
        raise ValueError("innovation_scale_tril must be exactly lower triangular")
    if np.any(np.diagonal(scales, axis1=1, axis2=2) <= 0.0):
        raise ValueError("innovation_scale_tril must have positive diagonal entries")
    if semantics != GAUGE_TREE_PRIOR_SEMANTICS:
        raise ValueError("gauge-tree representation semantics changed")
    digest = validate_sha256(source_digest, name="source_joint_covariance_sha256")
    return (
        ids,
        parents,
        immutable_array(transitions, dtype=np.float64),
        immutable_array(scales, dtype=np.float64),
        digest,
    )


def coerce_gauge_blocks(
    value: Any,
    *,
    gauge_count: int,
    name: str,
) -> tuple[np.ndarray, str]:
    dimension = GAUGE_DIMENSION * gauge_count
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (dimension,):
        blocks = array.reshape(gauge_count, GAUGE_DIMENSION, 1)
        layout = "flat-vector"
    elif array.shape == (gauge_count, GAUGE_DIMENSION):
        blocks = array[:, :, None]
        layout = "block-vector"
    elif array.ndim == 2 and array.shape[0] == dimension:
        blocks = array.reshape(gauge_count, GAUGE_DIMENSION, array.shape[1])
        layout = "flat-matrix"
    elif array.ndim == 3 and array.shape[:2] == (gauge_count, GAUGE_DIMENSION):
        blocks = array
        layout = "block-matrix"
    else:
        raise ValueError(
            f"{name} must have shape ({dimension},), ({gauge_count}, 7), "
            f"({dimension}, R), or ({gauge_count}, 7, R)"
        )
    if blocks.shape[2] < 1 or not np.all(np.isfinite(blocks)):
        raise ValueError(f"{name} must contain finite values and at least one right-hand side")
    return blocks, layout


def restore_gauge_layout(blocks: np.ndarray, layout: str) -> FloatArray:
    if layout == "flat-vector":
        return blocks[:, :, 0].reshape(-1)
    if layout == "block-vector":
        return blocks[:, :, 0]
    if layout == "flat-matrix":
        return blocks.reshape(-1, blocks.shape[2])
    if layout == "block-matrix":
        return blocks
    raise RuntimeError("unknown internal gauge-array layout")


def validate_observation_design(
    local_gauge_jacobian: Any,
    gauge_indices: Any,
    *,
    gauge_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = np.asarray(local_gauge_jacobian, dtype=np.float64)
    if jacobian.ndim != 3 or jacobian.shape[1:] != (3, GAUGE_DIMENSION):
        raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("local_gauge_jacobian must be finite")
    raw = np.asarray(gauge_indices)
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("gauge_indices must contain genuine integers")
    indices = np.asarray(raw, dtype=np.int64)
    if indices.shape != (len(jacobian),):
        raise ValueError("gauge_indices must have shape (M,)")
    if np.any(indices < 0) or np.any(indices >= gauge_count):
        raise ValueError("gauge_indices reference an unknown gauge")
    return jacobian, indices
