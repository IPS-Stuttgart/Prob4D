"""Fail-closed axial-orbit bounds under bounded point-position errors.

The local finite-orbit examples use two anchor points to define a rotation axis
and one probe point whose distance from that axis determines the full radial
query orbit.  In practice those three points are estimated.  This module gives
a deterministic outer bound on the true probe radius when every supplied point
is known only within one Euclidean error ball.

The result is conditional: the caller owns the point-error bound.  If the
observed anchors are too close to identify a direction under that bound, the
result is uninformative and its radius upper bound is infinite.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _vector(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result


def _nonnegative_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class AxialRadiusBound:
    """Observed line geometry and an outer bound on the true probe radius."""

    anchor_distance: float
    observed_axial_coordinate: float
    observed_radius: float
    point_error_bound: float
    direction_difference_bound: float
    radius_upper_bound: float
    informative: bool

    @property
    def observed_full_orbit_width(self) -> float:
        return 2.0 * self.observed_radius

    @property
    def outer_full_orbit_width(self) -> float:
        return 2.0 * self.radius_upper_bound

    def accepts_width(self, maximum_width: float) -> bool:
        """Return whether the outer orbit is no wider than ``maximum_width``."""

        threshold = _nonnegative_scalar(maximum_width, name="maximum_width")
        return self.informative and self.outer_full_orbit_width <= threshold


def point_to_line_coordinates(
    anchor_a: ArrayLike,
    anchor_b: ArrayLike,
    probe: ArrayLike,
) -> tuple[float, float]:
    """Return axial coordinate and radius relative to an oriented anchor line."""

    first = _vector(anchor_a, name="anchor_a")
    second = _vector(anchor_b, name="anchor_b")
    point = _vector(probe, name="probe")
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        raise ValueError("anchor points must be distinct")
    axis = delta / distance
    offset = point - first
    axial = float(offset @ axis)
    radial = offset - axial * axis
    return axial, float(np.linalg.norm(radial))


def bounded_axial_radius(
    observed_anchor_a: ArrayLike,
    observed_anchor_b: ArrayLike,
    observed_probe: ArrayLike,
    point_error_bound: float,
) -> AxialRadiusBound:
    """Bound the true probe-to-axis radius under equal point-error balls.

    Let the unobserved true points ``a``, ``b`` and ``p`` lie within distance
    ``epsilon`` of their observed counterparts.  If ``d_hat`` is the observed
    anchor separation and ``d_hat > 2*epsilon``, normalized-vector perturbation
    gives

    ``||u - u_hat|| <= 4*epsilon / (d_hat - 2*epsilon)``.

    Choosing the observed axial coordinate as a feasible point on the true line
    then yields

    ``r_true <= r_hat + 2*epsilon + |t_hat|*||u-u_hat||``.

    The implementation caps the direction difference at its universal maximum
    of two.  When the anchors cannot identify a direction under the declared
    error bound, the method fails closed with an infinite upper bound.
    """

    first = _vector(observed_anchor_a, name="observed_anchor_a")
    second = _vector(observed_anchor_b, name="observed_anchor_b")
    point = _vector(observed_probe, name="observed_probe")
    epsilon = _nonnegative_scalar(point_error_bound, name="point_error_bound")

    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        raise ValueError("observed anchor points must be distinct")
    axis = delta / distance
    offset = point - first
    axial = float(offset @ axis)
    radius = float(np.linalg.norm(offset - axial * axis))

    anchor_vector_error = 2.0 * epsilon
    if distance <= anchor_vector_error:
        return AxialRadiusBound(
            anchor_distance=distance,
            observed_axial_coordinate=axial,
            observed_radius=radius,
            point_error_bound=epsilon,
            direction_difference_bound=2.0,
            radius_upper_bound=math.inf,
            informative=False,
        )

    direction_bound = min(
        2.0,
        2.0 * anchor_vector_error / (distance - anchor_vector_error),
    )
    upper = radius + 2.0 * epsilon + abs(axial) * direction_bound
    return AxialRadiusBound(
        anchor_distance=distance,
        observed_axial_coordinate=axial,
        observed_radius=radius,
        point_error_bound=epsilon,
        direction_difference_bound=direction_bound,
        radius_upper_bound=upper,
        informative=True,
    )
