"""Fail-closed validation for scalar scientific parameters and reports."""

from __future__ import annotations

import numpy as np


def require_genuine_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return one genuine integer while rejecting coercible scalar aliases."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def require_finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Return one finite real scalar without accepting strings or booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a genuine real scalar")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        below = normalized < minimum if minimum_inclusive else normalized <= minimum
        if below:
            relation = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        above = normalized > maximum if maximum_inclusive else normalized >= maximum
        if above:
            relation = "at most" if maximum_inclusive else "less than"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return normalized


__all__ = ["require_finite_real", "require_genuine_integer"]
