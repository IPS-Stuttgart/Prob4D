"""Shared contracts for joint shared/camera-specific visual-bias calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array
from ._immutable_json import frozen_finite_json_mapping, plain_json

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

JOINT_VISUAL_BIAS_LAYOUT_SCHEMA: Final = "prob4d.joint-visual-bias-layout"
JOINT_VISUAL_BIAS_LAYOUT_VERSION: Final = 1
JOINT_VISUAL_BIAS_BASIS_ORDER: Final = "shared-prefix-then-camera-mode-major-v1"
JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS: Final = "complete-joint-selected-coefficient-covariance-v1"
JOINT_VISUAL_BIAS_METADATA_KEY: Final = "joint_visual_bias_layout_v1"
JOINT_VISUAL_BIAS_CLAIM_BOUNDARY: Final = (
    "Source/calibration-group joint visual-bias design and covariance only. "
    "The selected nested basis need not be complete, target calibration is not "
    "established, no physical-state update is accepted here, and tactile, depth, "
    "LiDAR, or force evidence must remain separate factors."
)

_LAYOUT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "layout_id",
        "camera_ids",
        "shared_basis_names",
        "camera_basis_names",
        "expanded_basis_names",
        "basis_order_semantics",
        "claim_boundary",
    }
)
_CALIBRATION_METADATA_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "layout",
        "layout_id",
        "group_artifact_ids",
        "allow_partial_camera_mode",
        "uses_target_outcomes",
        "uses_downstream_physical_innovation",
        "covariance_semantics",
        "claim_boundary",
    }
)
_RESERVED_METADATA_KEYS = frozenset(
    {
        JOINT_VISUAL_BIAS_METADATA_KEY,
        "uses_target_outcomes",
        "uses_downstream_physical_innovation",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _layout_component(value: object, *, name: str) -> str:
    result = _nonempty_string(value, name=name)
    if "::" in result:
        raise ValueError(f"{name} must not contain the reserved delimiter '::'")
    return result


def _sha256(value: object, *, name: str) -> str:
    digest = _nonempty_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _string_tuple(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    require_sorted: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    result = tuple(
        _layout_component(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    if require_sorted and result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted")
    return result


def _json_string_tuple(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    require_sorted: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a JSON array of strings")
    return _string_tuple(
        tuple(value),
        name=name,
        minimum=minimum,
        require_sorted=require_sorted,
    )


def _json_nonempty_string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a JSON array of strings")
    result = tuple(
        _nonempty_string(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _float64_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have dtype float64")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return cast(FloatArray, immutable_array(array, dtype=np.float64))


def _metadata(
    value: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    def reserved_keys(current: object) -> set[str]:
        if isinstance(current, Mapping):
            result = _RESERVED_METADATA_KEYS & set(current)
            for nested in current.values():
                result |= reserved_keys(nested)
            return result
        if isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            result: set[str] = set()
            for nested in current:
                result |= reserved_keys(nested)
            return result
        return set()

    collisions = sorted(reserved_keys(value))
    if collisions:
        raise ValueError(f"{name} uses reserved keys: {collisions}")
    return frozen_finite_json_mapping(value, name=name)


__all__ = [
    "FloatArray",
    "IntArray",
    "JOINT_VISUAL_BIAS_BASIS_ORDER",
    "JOINT_VISUAL_BIAS_CLAIM_BOUNDARY",
    "JOINT_VISUAL_BIAS_COVARIANCE_SEMANTICS",
    "JOINT_VISUAL_BIAS_LAYOUT_SCHEMA",
    "JOINT_VISUAL_BIAS_LAYOUT_VERSION",
    "JOINT_VISUAL_BIAS_METADATA_KEY",
]
