"""Strict JSON and path helpers for portable gauge-tree prior artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys instead of silently accepting the last one."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_nonfinite_constant(value: str) -> None:
    """Reject NaN and infinite JSON extensions."""

    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}")


def string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def exact_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def sha256(value: Any, *, name: str) -> str:
    digest = string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def json_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    """Load strict UTF-8 JSON with duplicate and non-finite rejection."""

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"gauge-tree prior manifest is unreadable or invalid JSON: {path}"
        ) from error
    return mapping(value, name="gauge-tree prior manifest")


def relative_payload_path(manifest: Path, payload: Path) -> str:
    """Return a portable payload path constrained below the manifest directory."""

    root = manifest.parent.resolve()
    resolved = payload.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "gauge-tree prior payload must remain below the manifest directory"
        ) from error
    if relative == Path("."):
        raise ValueError("gauge-tree prior payload must differ from the manifest")
    return relative.as_posix()


def resolve_payload_path(manifest: Path, value: Any) -> Path:
    """Resolve a manifest payload path without permitting directory traversal."""

    text = string(value, name="payload.path")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "payload.path must be a relative path below the manifest directory"
        )
    root = manifest.parent.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("payload.path escapes the manifest directory") from error
    return resolved
