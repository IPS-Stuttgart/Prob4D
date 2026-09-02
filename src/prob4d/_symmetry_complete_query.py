"""Certified finite-cover bounds for compact-group query orbits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from ._symmetry_complete_base import (
    _NUMERICAL_ATOL,
    FloatArray,
    SymmetryCompleteBeliefV1,
    _canonical_string,
    _finite_nonnegative_vector,
    _genuine_bool,
    _immutable_float,
)

CertificateStatus: TypeAlias = Literal[
    "certified-invariant",
    "certified-variant",
    "undetermined",
    "scope-not-certified",
]


@dataclass(frozen=True, slots=True)
class OrbitInvarianceCertificateV1:
    """Sampled lower and Lipschitz-cover upper bounds on group-orbit diameter."""

    query_id: str
    group_id: str
    metric_id: str
    status: CertificateStatus
    admitted: bool
    bounds_certified: bool
    sample_diameter_by_quotient: FloatArray
    upper_diameter_by_quotient: FloatArray
    maximum_sample_diameter: float
    maximum_upper_diameter: float
    tolerance: float
    cover_radius_by_quotient: FloatArray
    lipschitz_by_quotient: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_id",
            _canonical_string(self.query_id, name="query_id"),
        )
        object.__setattr__(
            self,
            "group_id",
            _canonical_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "metric_id",
            _canonical_string(self.metric_id, name="metric_id"),
        )
        allowed: tuple[CertificateStatus, ...] = (
            "certified-invariant",
            "certified-variant",
            "undetermined",
            "scope-not-certified",
        )
        if self.status not in allowed:
            raise ValueError("unsupported certificate status")
        admitted = _genuine_bool(self.admitted, name="admitted")
        certified = _genuine_bool(self.bounds_certified, name="bounds_certified")
        if admitted != (self.status == "certified-invariant"):
            raise ValueError("admitted must match certified-invariant status")
        if self.status == "scope-not-certified" and certified:
            raise ValueError("scope-not-certified cannot have certified bounds")
        if self.status != "scope-not-certified" and not certified:
            raise ValueError("a classified certificate must have certified bounds")
        sample = _immutable_float(
            self.sample_diameter_by_quotient,
            name="sample_diameter_by_quotient",
            ndim=1,
        )
        upper = _immutable_float(
            self.upper_diameter_by_quotient,
            name="upper_diameter_by_quotient",
            ndim=1,
        )
        cover = _immutable_float(
            self.cover_radius_by_quotient,
            name="cover_radius_by_quotient",
            ndim=1,
        )
        lipschitz = _immutable_float(
            self.lipschitz_by_quotient,
            name="lipschitz_by_quotient",
            ndim=1,
        )
        if not (sample.shape == upper.shape == cover.shape == lipschitz.shape):
            raise ValueError("per-quotient certificate arrays must have identical shapes")
        if (
            np.any(sample < 0.0)
            or np.any(upper < sample)
            or np.any(cover < 0.0)
            or np.any(lipschitz < 0.0)
        ):
            raise ValueError("certificate arrays violate nonnegative bound ordering")
        tolerance = float(self.tolerance)
        maximum_sample = float(self.maximum_sample_diameter)
        maximum_upper = float(self.maximum_upper_diameter)
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (tolerance, maximum_sample, maximum_upper)
        ):
            raise ValueError("certificate scalars must be finite and nonnegative")
        if maximum_upper + _NUMERICAL_ATOL < maximum_sample:
            raise ValueError("maximum upper diameter is below sample diameter")
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "bounds_certified", certified)
        object.__setattr__(self, "sample_diameter_by_quotient", sample)
        object.__setattr__(self, "upper_diameter_by_quotient", upper)
        object.__setattr__(self, "cover_radius_by_quotient", cover)
        object.__setattr__(self, "lipschitz_by_quotient", lipschitz)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "maximum_sample_diameter", maximum_sample)
        object.__setattr__(self, "maximum_upper_diameter", maximum_upper)


def _sample_diameter(points: FloatArray) -> float:
    maximum = 0.0
    for index in range(points.shape[0]):
        delta = points[index + 1 :] - points[index]
        if delta.size:
            maximum = max(
                maximum,
                float(np.max(np.linalg.norm(delta, axis=1))),
            )
    return maximum


def certify_compact_group_query(
    belief: SymmetryCompleteBeliefV1,
    query_atoms: ArrayLike,
    *,
    query_id: str,
    lipschitz_by_quotient: ArrayLike | float,
    cover_radius_by_quotient: ArrayLike | float | None = None,
    tolerance: float = 0.0,
    cover_radius_certified: bool | None = None,
) -> OrbitInvarianceCertificateV1:
    """Certify whether a vector query is invariant over each active group orbit.

    Let ``S`` be the supplied group nodes with cover radius ``rho`` in the
    declared metric, and let the query be ``L``-Lipschitz. If ``D_S`` is the
    maximum Euclidean distance between sampled query values, then

        D_S <= diam(q(G)) <= D_S + 2 L rho.

    The maximum is taken over quotient classes with positive posterior mass.
    An upper bound below ``tolerance`` certifies gauge invariance. A sampled
    lower bound above it certifies variation. An overlapping interval is
    undetermined and must fall back.
    """

    if not isinstance(belief, SymmetryCompleteBeliefV1):
        raise TypeError("belief must be SymmetryCompleteBeliefV1")
    atoms = _immutable_float(query_atoms, name="query_atoms", ndim=3)
    if (
        atoms.shape[:2]
        != (
            belief.quotient_count,
            belief.quadrature.node_count,
        )
        or atoms.shape[2] == 0
    ):
        raise ValueError("query_atoms have the wrong quotient/group/query shape")
    lipschitz = _finite_nonnegative_vector(
        lipschitz_by_quotient,
        name="lipschitz_by_quotient",
        size=belief.quotient_count,
    )
    cover_source: ArrayLike | float = (
        belief.quadrature.cover_radius
        if cover_radius_by_quotient is None
        else cover_radius_by_quotient
    )
    cover = _finite_nonnegative_vector(
        cover_source,
        name="cover_radius_by_quotient",
        size=belief.quotient_count,
    )
    if isinstance(tolerance, (bool, np.bool_)):
        raise TypeError("tolerance must be a real scalar")
    threshold = float(tolerance)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    certified = (
        belief.quadrature.cover_radius_certified
        if cover_radius_certified is None
        else _genuine_bool(
            cover_radius_certified,
            name="cover_radius_certified",
        )
    )
    sample: FloatArray = np.zeros(belief.quotient_count, dtype=np.float64)
    upper: FloatArray = np.zeros(belief.quotient_count, dtype=np.float64)
    active = belief.quotient_weights > 0.0
    for quotient_index in np.flatnonzero(active):
        sample[quotient_index] = _sample_diameter(atoms[quotient_index])
        upper[quotient_index] = (
            sample[quotient_index] + 2.0 * lipschitz[quotient_index] * cover[quotient_index]
        )
    maximum_sample = float(np.max(sample[active]))
    maximum_upper = float(np.max(upper[active]))
    if not certified:
        status: CertificateStatus = "scope-not-certified"
    elif maximum_upper <= threshold + _NUMERICAL_ATOL:
        status = "certified-invariant"
    elif maximum_sample > threshold + _NUMERICAL_ATOL:
        status = "certified-variant"
    else:
        status = "undetermined"
    return OrbitInvarianceCertificateV1(
        query_id=query_id,
        group_id=belief.quadrature.group_id,
        metric_id=belief.quadrature.metric_id,
        status=status,
        admitted=status == "certified-invariant",
        bounds_certified=certified,
        sample_diameter_by_quotient=sample,
        upper_diameter_by_quotient=upper,
        maximum_sample_diameter=maximum_sample,
        maximum_upper_diameter=maximum_upper,
        tolerance=threshold,
        cover_radius_by_quotient=cover,
        lipschitz_by_quotient=lipschitz,
    )


__all__ = [
    "CertificateStatus",
    "OrbitInvarianceCertificateV1",
    "certify_compact_group_query",
]
