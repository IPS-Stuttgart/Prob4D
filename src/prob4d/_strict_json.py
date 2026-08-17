"""Strict portable-JSON validation for content-addressed artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ._immutable_json import plain_json


class _StrictJsonValueError(ValueError):
    """Internal marker for already contextualized strict-JSON failures."""


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={sorted(missing)}, extra={sorted(extra)}")


def require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def require_exact_string(value: Any, *, name: str) -> str:
    """Return one non-empty built-in string without silently normalizing it."""

    if type(value) is not str:
        raise ValueError(f"{name} must be a genuine string")
    if not value:
        raise ValueError(f"{name} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    return value


def require_exact_integer(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def require_json_number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_revision(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be an exact lowercase 40- or 64-character revision")
    return value


def require_string_sequence(
    value: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    values = tuple(value)
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain nonempty strings")
    return cast(tuple[str, ...], values)


def require_finite_json_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    mapping = require_mapping(value, name=name)
    try:
        normalized = json.loads(
            json.dumps(
                plain_json(mapping),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be a JSON object")
    return mapping


def loads_json_object(content: str, *, name: str) -> dict[str, Any]:
    """Parse one finite JSON object while rejecting duplicate object keys."""

    if type(content) is not str:
        raise ValueError(f"{name} must be UTF-8 text")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise _StrictJsonValueError(f"{name} contains duplicate JSON object key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise _StrictJsonValueError(f"{name} contains non-finite JSON number {token!r}")

    try:
        value = json.loads(
            content,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except _StrictJsonValueError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def load_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    """Load one finite JSON object while rejecting duplicate object keys."""

    return loads_json_object(Path(path).read_text(encoding="utf-8"), name=name)


__all__ = [
    "load_json_object",
    "loads_json_object",
    "require_exact_fields",
    "require_exact_integer",
    "require_exact_string",
    "require_finite_json_mapping",
    "require_json_number",
    "require_mapping",
    "require_nonempty_string",
    "require_revision",
    "require_sha256",
    "require_string_sequence",
]
