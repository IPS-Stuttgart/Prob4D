"""Marginal-preserving tempering of shared Gaussian dependence.

The helper in this module deliberately changes only cross-output covariance.
It is useful when two candidate covariance models have the same diagonal but
different dependence structure.  It does not fit the tempering strength and it
does not claim that a convex blend is a calibrated posterior.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _covariance(value: NDArray[np.floating], *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    scale = max(1.0, float(np.max(np.abs(array))))
    if not np.allclose(array, array.T, rtol=0.0, atol=1.0e-10 * scale):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (array + array.T)
    minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
    if minimum < -1.0e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def temper_shared_dependence(
    marginal_covariance: NDArray[np.floating],
    shared_covariance: NDArray[np.floating],
    strength: float,
    *,
    diagonal_rtol: float = 1.0e-10,
    diagonal_atol: float = 1.0e-12,
) -> FloatArray:
    """Blend dependence while preserving every marginal variance.

    Returns

    ``marginal + strength * (shared - marginal)``

    for ``0 <= strength <= 1`` after requiring that the endpoint covariances
    have matching diagonals.  Convexity preserves positive semidefiniteness;
    the diagonal is then restored exactly from ``marginal_covariance`` to avoid
    round-off drift.

    ``strength=0`` is the marginal endpoint and ``strength=1`` is the shared
    endpoint.  No calibration or safety interpretation is attached to either
    endpoint or to an intermediate value.
    """

    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and lie in [0, 1]")
    marginal = _covariance(marginal_covariance, name="marginal_covariance")
    shared = _covariance(shared_covariance, name="shared_covariance")
    if marginal.shape != shared.shape:
        raise ValueError("covariance endpoints must have matching shape")
    diagonal = np.diag(marginal).copy()
    if not np.allclose(
        diagonal,
        np.diag(shared),
        rtol=diagonal_rtol,
        atol=diagonal_atol,
    ):
        raise ValueError("covariance endpoints must have matching marginal variances")
    result = marginal + float(strength) * (shared - marginal)
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, diagonal)
    scale = max(1.0, float(np.max(np.abs(result))))
    minimum = float(np.min(np.linalg.eigvalsh(result)))
    if minimum < -1.0e-9 * scale:
        raise RuntimeError("convex dependence tempering unexpectedly lost PSD")
    return result
