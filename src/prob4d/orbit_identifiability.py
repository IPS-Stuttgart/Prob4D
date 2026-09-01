"""Robust query-identifiability certificates for estimated equivalence orbits.

Let ``O`` be the unknown true orbit, ``O_hat`` an estimated orbit, and ``q`` an
``L``-Lipschitz query. If the orbit-set Hausdorff distance is at most ``delta``,
then

    diameter(q(O)) <= diameter(q(O_hat)) + 2 * L * delta.

The result follows by approximating both endpoints of an arbitrary true-orbit
query pair with estimated-orbit points and applying the triangle inequality.
The certificate is conditional on a valid orbit-set error bound; this module
does not estimate that bound from data or convert a confidence quantile into a
deterministic guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def query_set_diameter(values: ArrayLike) -> float:
    """Return the Euclidean diameter of a finite nonempty query-value set."""
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("values must have finite nonempty shape (N,) or (N,D)")
    maximum_squared = 0.0
    for start in range(0, points.shape[0], 1024):
        difference = points[start : start + 1024, None, :] - points[None, :, :]
        maximum_squared = max(
            maximum_squared,
            float(np.max(np.einsum("ijk,ijk->ij", difference, difference))),
        )
    return float(np.sqrt(maximum_squared))


def paired_orbit_error_bound(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Upper-bound orbit Hausdorff error from a shared parameterization.

    For paired samples ``reference[t]`` and ``estimate[t]`` from the same orbit
    parameter, ``max_t ||reference[t]-estimate[t]||`` upper-bounds the Hausdorff
    distance between those sampled sets. It need not be the tightest bound.
    """
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(estimate, dtype=np.float64)
    if (
        first.ndim != 2
        or first.shape[0] == 0
        or first.shape != second.shape
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("paired orbits must have equal finite nonempty shape (N,D)")
    return float(np.max(np.linalg.norm(first - second, axis=1)))


@dataclass(frozen=True)
class QueryOrbitCertificate:
    """Conditional upper bound on true query variation over an orbit."""

    estimated_query_diameter: float
    query_lipschitz_constant: float
    orbit_hausdorff_radius: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimated_query_diameter",
            _nonnegative(
                self.estimated_query_diameter,
                name="estimated_query_diameter",
            ),
        )
        object.__setattr__(
            self,
            "query_lipschitz_constant",
            _nonnegative(
                self.query_lipschitz_constant,
                name="query_lipschitz_constant",
            ),
        )
        object.__setattr__(
            self,
            "orbit_hausdorff_radius",
            _nonnegative(
                self.orbit_hausdorff_radius,
                name="orbit_hausdorff_radius",
            ),
        )

    @property
    def true_query_diameter_upper_bound(self) -> float:
        return float(
            self.estimated_query_diameter
            + 2.0 * self.query_lipschitz_constant * self.orbit_hausdorff_radius
        )

    def admitted(self, maximum_query_diameter: float) -> bool:
        tolerance = _nonnegative(
            maximum_query_diameter,
            name="maximum_query_diameter",
        )
        return self.true_query_diameter_upper_bound <= tolerance
