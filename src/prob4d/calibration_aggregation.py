"""Explicit aggregation semantics for robust covariance-scale calibration."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

UPPER_WINSORIZED_MEAN_V1: Final = "upper-quantile-winsorized-mean-v1"
GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2: Final = (
    "equal-group-mean-of-within-group-upper-winsorized-ratios-v2"
)
LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1: Final = (
    "equal-group-mean-of-within-group-trimmed-ratios-v1"
)


def upper_winsorized_mean(
    values: FloatArray,
    *,
    quantile: float,
    minimum: float = 1e-6,
    canonicalize: bool = False,
) -> float:
    """Return a mean after clipping values above one empirical quantile.

    This is upper winsorization, not trimming: observations above the selected
    quantile remain in the sample with their values replaced by the quantile.
    ``canonicalize=True`` sorts first so equal input multisets have identical
    floating-point accumulation order.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("upper winsorization requires one nonempty value vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("upper winsorization values must be finite")
    if isinstance(quantile, (bool, np.bool_)):
        raise ValueError("winsor quantile must be a finite number in (0, 1]")
    normalized_quantile = float(quantile)
    if not np.isfinite(normalized_quantile) or not 0.0 < normalized_quantile <= 1.0:
        raise ValueError("winsor quantile must be a finite number in (0, 1]")
    if isinstance(minimum, (bool, np.bool_)):
        raise ValueError("winsorized-mean minimum must be finite and nonnegative")
    normalized_minimum = float(minimum)
    if not np.isfinite(normalized_minimum) or normalized_minimum < 0.0:
        raise ValueError("winsorized-mean minimum must be finite and nonnegative")

    ordered = np.sort(array) if canonicalize else array
    upper = float(np.quantile(ordered, normalized_quantile))
    return max(float(np.mean(np.minimum(ordered, upper))), normalized_minimum)


__all__ = [
    "GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2",
    "LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1",
    "UPPER_WINSORIZED_MEAN_V1",
    "upper_winsorized_mean",
]
