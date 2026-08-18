"""Preview version-1 geometry façade backed by :mod:`prob4d.api.v2`."""

from __future__ import annotations

from typing import Final

from .v2 import (
    GAUGE_DIMENSION,
    GAUGE_PARAMETERIZATION,
    GaugeEstimate,
    Sim3,
    sim3_point_jacobian,
)

FACADE_VERSION: Final = 1
LIFECYCLE: Final = "preview"

__all__ = [
    "FACADE_VERSION",
    "GAUGE_DIMENSION",
    "GAUGE_PARAMETERIZATION",
    "GaugeEstimate",
    "LIFECYCLE",
    "Sim3",
    "sim3_point_jacobian",
]
