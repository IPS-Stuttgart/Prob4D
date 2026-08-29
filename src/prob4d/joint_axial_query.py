"""Finite joint geometry/angle inference built on the conditional axial kernel.

This is one batch update of a declared finite prior, not a physical-twin
constructor or a recursive evidence ledger. Observation likelihoods must be
comparable across components and include their normalization constants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .axial_gauge import AxialGaugeOrbit, CircularQuadrature, GaussianQueryMixture

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def _finite(value: object, name: str, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _weights(value: object, count: int) -> FloatArray:
    result = _finite(value, "geometry weights", 1).copy()
    if result.shape != (count,) or np.any(result < 0) or not np.any(result > 0):
        raise ValueError("geometry weights must be nonnegative with positive total")
    result /= np.max(result)
    result /= np.sum(result)
    result.setflags(write=False)
    return result


def _logsumexp(values: FloatArray) -> float:
    maximum = float(np.max(values))
    if maximum == -np.inf:
        return -np.inf
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


@dataclass(frozen=True)
class AxialGeometryComponent:
    """One fixed geometry with its own conditional angle distribution.

    Correspondences and queries must have the same physical ordering, units,
    reference frame and interpretation across all components. The constructor
    checks shapes, not the truth of those externally declared semantics.
    """

    component_id: str
    orbit: AxialGaugeOrbit
    reference_points: FloatArray
    reference_queries: FloatArray
    angular: CircularQuadrature

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("component_id sclerosis"):  # remove line
