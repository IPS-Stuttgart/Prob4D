"""Exact finite-orbit bounds for affine queries of axial-gauge point families.

Experimental, NumPy-only analysis. This is not a posterior or an admission
certificate for a provider. The caller must register the axial symmetry,
point identities, shared gauge groups, and the domain of allowed angles.
A small local Jacobian or a rank-six factor does not establish that domain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _finite_array(value: object, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _vector(value: object, *, name: str) -> FloatArray:
    result = _finite_array(value, name=name)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return result


def _points(value: object, *, name: str) -> FloatArray:
    result = _finite_array(value, name=name)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), N >= 1")
    return result


def _group_id(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("gauge group IDs must be nonempty, unpadded strings")
    return value


def _nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class AxialGaugeOrbit:
    """All rotations about a registered metric-space line, with fixed scale.

    ``axis`` is normalized on construction. ``pivot`` lies on the line.
    The class defines an orbit; it does not infer a symmetry from observations.
    """

    pivot: FloatArray
    axis: FloatArray

    def __post_init__(self) -> None:
        pivot = _vector(self.pivot, name="pivot")
        axis = _vector(self.axis, name="axis")
        scale = float(np.max(np.abs(axis)))
        if scale == 0.0:
            raise ValueError("axis must be nonzero")
        normalized = axis / scale
        normalized = normalized / np.linalg.norm(normalized)
        normalized.setflags(write=False)
        object.__setattr__(self, "pivot", pivot)
        object.__setattr__(self, "axis", normalized)

    def components(self, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return constant, cosine, and sine point components (Rodrigues)."""
        values = _points(points, name="points")
        relative = values - self.pivot
        parallel = (relative @ self.axis)[:, None] * self.axis
        constant = self.pivot + parallel
        cosine = relative - parallel
        sine = np.cross(self.axis, cosine)
        return (
            _finite_array(constant, name="constant orbit component"),
            _finite_array(cosine, name="cosine orbit component"),
            _finite_array(sine, name="sine orbit component"),
        )

    def transform(self, points: FloatArray, angle: float) -> FloatArray:
        """Apply a finite rotation, without a local Gaussian approximation."""
        angle = float(angle)
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        constant, cosine, sine = self.components(points)
        return _finite_array(
            constant + np.cos(angle) * cosine + np.sin(angle) * sine,
            name="transformed points",
        )

    def maximum_support_motion(self, support_points: FloatArray) -> float:
        """Sharp maximum point displacement over the full orbit, in input units.

        This is twice the largest distance to the axis. Zero means geometric
        invariance of the supplied support. A small nonzero value is only a
        geometric bound: it is NOT equal likelihood or statistical equivalence.
        """
        _, cosine, _ = self.components(support_points)
        bound = float(2.0 * np.max(np.linalg.norm(cosine, axis=1)))
        if not np.isfinite(bound):
            raise ValueError("support motion is not finite")
        return bound


@dataclass(frozen=True)
class AxialQueryFamily:
    """q_k(phi) = constant_k + sum_g (cosine_kg cos phi_g + sine_kg sin phi_g).

    All queries share the SAME angle for each group. Bounds are sharp when
    group angles range freely over a product of full circles. For a smaller
    coupled angle domain they remain conservative, not necessarily sharp.
    """

    group_ids: tuple[str, ...]
    constant: FloatArray
    cosine: FloatArray
    sine: FloatArray

    def __post_init__(self) -> None:
        if isinstance(self.group_ids, (str, bytes)):
            raise ValueError("group_ids must be a sequence of group IDs")
        groups = tuple(_group_id(group) for group in self.group_ids)
        if len(groups) != len(set(groups)):
            raise ValueError("group_ids must be unique")
        constant = _finite_array(self.constant, name="constant")
        if constant.ndim != 1 or constant.size < 1:
            raise ValueError("constant must have shape (Q,), Q >= 1")
        cosine = _finite_array(self.cosine, name="cosine")
        sine = _finite_array(self.sine, name="sine")
        expected = (constant.size, len(groups))
        if cosine.shape != expected or sine.shape != expected:
            raise ValueError("cosine and sine must have shape (Q, number of groups)")
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "cosine", cosine)
        object.__setattr__(self, "sine", sine)

    def evaluate(self, angles: FloatArray) -> FloatArray:
        angles = _finite_array(angles, name="angles")
        if angles.shape != (len(self.group_ids),):
            raise ValueError("angles must have shape (number of groups,)")
        return _finite_array(
            self.constant + self.cosine @ np.cos(angles) + self.sine @ np.sin(angles),
            name="query values",
        )

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        """Sharp componentwise intervals; not a joint rectangular feasible set."""
        radius = np.sum(np.hypot(self.cosine, self.sine), axis=1)
        return (
            _finite_array(self.constant - radius, name="lower bounds"),
            _finite_array(self.constant + radius, name="upper bounds"),
        )

    def contrast_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Sharp bounds on q_i - q_j, preserving shared-angle cancellation."""
        center = self.constant[:, None] - self.constant[None, :]
        cosine = self.cosine[:, None, :] - self.cosine[None, :, :]
        sine = self.sine[:, None, :] - self.sine[None, :, :]
        radius = np.sum(np.hypot(cosine, sine), axis=2)
        return (
            _finite_array(center - radius, name="lower contrast bounds"),
            _finite_array(center + radius, name="upper contrast bounds"),
        )

    def worst_case_regrets(self) -> FloatArray:
        """Sharp regrets when queries are actual lower-is-better action losses.

        The caller must supply those loss functions. This does not convert
        position uncertainty into a simulator-loss or robot-safety guarantee.
        """
        _, upper = self.contrast_bounds()
        return _finite_array(np.max(upper, axis=1), name="worst-case regrets")

    def _action_index(self, action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer index")
        action = int(action)
        if not 0 <= action < self.constant.size:
            raise ValueError("action is outside the query family")
        return action

    def regret_witness(self, action: int) -> tuple[int, FloatArray]:
        """Return a competing action and angle vector attaining the worst regret."""
        action = self._action_index(action)
        _, upper = self.contrast_bounds()
        competitor = int(np.argmax(upper[action]))
        angles = np.arctan2(
            self.sine[action] - self.sine[competitor],
            self.cosine[action] - self.cosine[competitor],
        )
        return competitor, _finite_array(angles, name="witness angles")

    def within_regret_budget(
        self,
        action: int,
        *,
        maximum_regret: float,
        numerical_margin: float = 0.0,
    ) -> bool:
        """Check a caller-selected action against a caller-frozen regret budget.

        No action is executed and no belief is modified. Rejection must be
        routed to the caller's existing complete-belief fallback. These are
        float64 evaluations of analytic bounds, not interval-arithmetic proofs.
        """
        action = self._action_index(action)
        budget = _nonnegative(maximum_regret, name="maximum_regret")
        margin = _nonnegative(numerical_margin, name="numerical_margin")
        return bool(float(self.worst_case_regrets()[action]) + margin <= budget)


def affine_axial_queries(
    points: FloatArray,
    weights: FloatArray,
    *,
    point_group_ids: Sequence[str],
    orbits: Mapping[str, AxialGaugeOrbit],
    offsets: FloatArray | None = None,
) -> AxialQueryFamily:
    """Build affine queries ``offsets[k] + sum_i weights[k,i] dot points_i``.

    Points are already in the aligned metric frame. Their group labels encode
    a declared shared finite rotation, not merely similar geometry or an
    arbitrary dependence label. Contributions are summed INSIDE each group
    before an amplitude is computed. Incorrect group merging can fabricate
    cancellation, just as incorrect splitting can discard real cancellation.
    """
    values = _points(points, name="points")
    weights = _finite_array(weights, name="weights")
    if weights.ndim != 3 or weights.shape[0] < 1 or weights.shape[1:] != values.shape:
        raise ValueError("weights must have shape (Q, N, 3), Q >= 1")
    if isinstance(point_group_ids, (str, bytes)):
        raise ValueError("point_group_ids must be a sequence of group IDs")
    labels = tuple(_group_id(group) for group in point_group_ids)
    if len(labels) != values.shape[0]:
        raise ValueError("one gauge group ID is required per point")
    keys = tuple(_group_id(group) for group in orbits)
    if set(keys) != set(labels):
        raise ValueError("orbits must contain exactly the referenced gauge groups")
    if any(not isinstance(orbit, AxialGaugeOrbit) for orbit in orbits.values()):
        raise TypeError("every orbit must be an AxialGaugeOrbit")
    groups = tuple(sorted(keys))
    query_count = weights.shape[0]
    constant = (
        np.zeros(query_count, dtype=np.float64)
        if offsets is None
        else _finite_array(offsets, name="offsets").copy()
    )
    if constant.shape != (query_count,):
        raise ValueError("offsets must have shape (Q,)")
    cosine = np.zeros((query_count, len(groups)), dtype=np.float64)
    sine = np.zeros_like(cosine)
    for index, group in enumerate(groups):
        mask = np.array([label == group for label in labels])
        base, cos_part, sin_part = orbits[group].components(values[mask])
        group_weights = weights[:, mask, :]
        constant += np.einsum("qnc,nc->q", group_weights, base)
        cosine[:, index] = np.einsum("qnc,nc->q", group_weights, cos_part)
        sine[:, index] = np.einsum("qnc,nc->q", group_weights, sin_part)
    return AxialQueryFamily(groups, constant, cosine, sine)


__all__ = ["AxialGaugeOrbit", "AxialQueryFamily", "affine_axial_queries"]
