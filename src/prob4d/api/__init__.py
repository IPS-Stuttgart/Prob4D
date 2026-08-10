"""Versioned public API façades for downstream Prob4D consumers."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from . import v1

__all__ = ["v1", "v2"]


def __getattr__(name: str) -> ModuleType:
    if name == "v2":
        module = import_module(".v2", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
