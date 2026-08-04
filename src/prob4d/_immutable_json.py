"""Finite JSON normalization with recursively immutable dict/list values."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _plain_json(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON object keys must be strings at {path}; "
                    f"got {type(key).__name__}"
                )
            result[key] = _plain_json(item, path=f"{path}[{key!r}]")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def plain_json(value: Any) -> Any:
    """Return ordinary JSON-compatible containers without coercing object keys."""

    return _plain_json(value, path="$")


class FrozenJsonDict(dict[str, Any]):
    """A ``dict``-compatible mapping that rejects every in-place mutation."""

    __slots__ = ()
    _MUTATORS = frozenset({"clear", "pop", "popitem", "setdefault", "update"})

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("metadata is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        self._immutable(key, value)

    def __delitem__(self, key: str) -> None:
        self._immutable(key)

    def __ior__(self, other: object) -> FrozenJsonDict:
        self._immutable(other)
        return self

    def __copy__(self) -> dict[str, Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return plain_json(self)


class FrozenJsonList(list[Any]):
    """A ``list``-compatible sequence that rejects every in-place mutation."""

    __slots__ = ()
    _MUTATORS = frozenset(
        {"append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort"}
    )

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("metadata is immutable")

    def __setitem__(self, key: int | slice, value: Any) -> None:
        self._immutable(key, value)

    def __delitem__(self, key: int | slice) -> None:
        self._immutable(key)

    def __iadd__(self, other: object) -> FrozenJsonList:
        self._immutable(other)
        return self

    def __imul__(self, other: object) -> FrozenJsonList:
        self._immutable(other)
        return self

    def __copy__(self) -> list[Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        del memo
        return plain_json(self)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenJsonDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenJsonList(_freeze_json(item) for item in value)
    return value


def frozen_finite_json_mapping(
    values: Mapping[str, Any],
    *,
    name: str = "metadata",
) -> Mapping[str, Any]:
    """Normalize finite JSON data and freeze every nested mapping and sequence."""

    try:
        normalized = json.loads(
            json.dumps(
                plain_json(values),
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be a JSON object")
    return _freeze_json(normalized)


__all__ = [
    "FrozenJsonDict",
    "FrozenJsonList",
    "frozen_finite_json_mapping",
    "plain_json",
]
