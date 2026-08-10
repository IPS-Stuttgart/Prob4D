"""Resolve the installed Prob4D version without duplicating release state."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

UNKNOWN_VERSION = "0+unknown"


def package_version() -> str:
    """Return installed distribution metadata or an explicit source-tree sentinel."""

    try:
        return version("prob4d")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = package_version()

__all__ = ["UNKNOWN_VERSION", "__version__", "package_version"]
