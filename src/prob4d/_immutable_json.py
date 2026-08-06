"""Finite JSON normalization with recursively immutable protocol values."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NoReturn, overload


class FrozenJsonDict(Mapping[str, Any]):
    """A read-only JSON object without a mutable ``dict`` base class."""

    __slots__ = ("_values",)
    _values: Mapping[str, Any]

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(dict(self._values))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise TypeError("metadata is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        del name
        raise TypeError("metadata is immutable")

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("metadata is immutable")

    def __setitem__(self, key: object, value: object) -> NoReturn:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> NoReturn:
        self._immutable(key)

    def clear(self) -> NoReturn:
        self._immutable()

    def pop(self, *args: object) -> NoReturn:
        self._immutable(*args)

    def popitem(self) -> NoReturn:
        self._immutable()

    def setdefault(self, *args: object) -> NoReturn:
        self._immutable(*args)

    def update(self, *args: object, **kwargs: object) -> NoReturn:
        self._immutable(*args, **kwargs)

    def __ior__(self, other: object) -> NoReturn:
        self._immutable(other)

    def copy(self) -> dict[str, Any]:
        return plain_json(self)

    def __copy__(self) -> dict[str, Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return plain_json(self)


class FrozenJsonList(Sequence[Any]):
    """A read-only JSON array without a mutable ``list`` base class."""

    __slots__ = ("_values",)
    _values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        object.__setattr__(self, "_values", tuple(values))

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Any, ...]: ...

    def __getitem__(self, index: int | slice) -> Any | tuple[Any, ...]:
        return self._values[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(list(self._values))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenJsonList):
            return self._values == other._values
        if isinstance(other, (list, tuple)):
            return self._values == tuple(other)
        return False

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise TypeError("metadata is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        del name
        raise TypeError("metadata is immutable")

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("metadata is immutable")

    def __setitem__(self, key: object, value: object) -> NoReturn:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> NoReturn:
        self._immutable(key)

    def append(self, value: object) -> NoReturn:
        self._immutable(value)

    def clear(self) -> NoReturn:
        self._immutable()

    def extend(self, values: object) -> NoReturn:
        self._immutable(values)

    def insert(self, index: object, value: object) -> NoReturn:
        self._immutable(index, value)

    def pop(self, *args: object) -> NoReturn:
        self._immutable(*args)

    def remove(self, value: object) -> NoReturn:
        self._immutable(value)

    def reverse(self) -> NoReturn:
        self._immutable()

    def sort(self, *args: object, **kwargs: object) -> NoReturn:
        self._immutable(*args, **kwargs)

    def __iadd__(self, other: object) -> NoReturn:
        self._immutable(other)

    def __imul__(self, other: object) -> NoReturn:
        self._immutable(other)

    def copy(self) -> list[Any]:
        return plain_json(self)

    def __copy__(self) -> list[Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        del memo
        return plain_json(self)


_JSON_ARRAY_TYPES = (list, tuple, FrozenJsonList)


def _plain_json(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON object keys must be strings at {path}; got {type(key).__name__}"
                )
            result[key] = _plain_json(item, path=f"{path}[{key!r}]")
        return result
    if isinstance(value, _JSON_ARRAY_TYPES):
        return [_plain_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def plain_json(value: Any) -> Any:
    """Return ordinary JSON-compatible containers without coercing object keys."""

    return _plain_json(value, path="$")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenJsonDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenJsonList(tuple(_freeze_json(item) for item in value))
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
