"""Prob4D package metadata.

Use :mod:`prob4d.api.v2` for current downstream integrations. The package root is
intentionally minimal so importing ``prob4d`` does not create an accidental,
unversioned API surface.
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]
