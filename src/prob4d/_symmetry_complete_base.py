"""Internal quotient-times-group belief representation and update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
EvidenceSemantics: TypeAlias = Literal["orbit-invariant", "symmetry-breaking"]
GroupMeasureKind: TypeAlias = Literal["finite-mass", "continuous-density"]

SYMMETRY_COMPLETE_BELIEF_VERSION = 1
SYMMETRY_COMPLETE_BELIEF_CLAIM_BOUNDARY = (
    "The representation is exact only for the supplied finite quotient, group "
    "quadrature, prior support, likelihood values, and declared symmetry "
    "semantics. A continuous-group certificate additionally requires a valid "
    "metric cover radius and query Lipschitz bound. The module does not infer or "
    "validate a physical symmetry, calibrate those bounds, establish provider "
    "competence, prove target transport, authorize deployment, or certify safety."
)

_PROBABILITY_ATOL = 1e-12
_NUMERICAL_ATOL = 1e-12


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _immutable_float(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        array.shape
    )
    return result


def _probability_vector(value: ArrayLike, *, name: str) -> FloatArray:
    array = _immutable_float(value, name=name, ndim=1)
    if array.size == 0 or np.any(array < 0.0):
        raise ValueError(f"{name} must be a nonempty nonnegative vector")
    total = float(np.sum(array, dtype=np.float64))
    if not math.isfinite(total) or not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_ATOL
    ):
        raise ValueError(f"{name} must sum to one")
    normalized = np.ascontiguousarray(array / total, dtype=np.float64)
    return np.frombuffer(normalized.tobytes(order="C"), dtype=np.dtype("<f8"))


def _conditional_probability_matrix(
    value: ArrayLike,
    *,
    quotient_count: int,
    group_count: int,
    name: str,
) -> FloatArray:
    array = _immutable_float(value, name=name, ndim=2)
    if array.shape != (quotient_count, group_count):
        raise ValueError(
            f"{name} must have shape ({quotient_count}, {group_count})"
        )
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    totals = np.sum(array, axis=1, dtype=np.float64)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError(f"every row of {name} must sum to one")
    normalized = np.ascontiguousarray(array / totals[:, None], dtype=np.float64)
    return np.frombuffer(normalized.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        normalized.shape
    )


def _finite_nonnegative_vector(
    value: ArrayLike | float,
    *,
    name: str,
    size: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        raw = np.full(size, float(raw), dtype=np.float64)
    array = _immutable_float(raw, name=name, ndim=1)
    if array.shape != (size,) or np.any(array < 0.0):
        raise ValueError(f"{name} must be a nonnegative scalar or length-{size} vector")
    return array


def _genuine_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _kl_divergence(posterior: FloatArray, prior: FloatArray) -> float:
    positive = posterior > 0.0
    if np.any(prior[positive] <= 0.0):
        raise ValueError("posterior must remain absolutely continuous with the prior")
    return float(
        np.sum(
            posterior[positive]
            * (np.log(posterior[positive]) - np.log(prior[positive])),
            dtype=np.float64,
        )
    )


@dataclass(frozen=True, slots=True)
class CompactGroupQuadratureV1:
    """Finite quadrature for one declared compact group and metric.

    ``nodes`` are caller-defined coordinates; this class does not verify group
    closure, Haar exactness, or the cover radius. ``cover_radius`` is the largest
    group-metric distance from any admissible group element to its nearest node.
    A positive radius is useful only when ``cover_radius_certified`` is true and
    the downstream query has a valid Lipschitz bound in the same metric.
    """

    group_id: str
    metric_id: str
    nodes: FloatArray
    reference_weights: FloatArray
    cover_radius: float
    cover_radius_certified: bool
    measure_kind: GroupMeasureKind = "finite-mass"

    def __post_init__(self) -> None:
        group_id = _canonical_string(self.group_id, name="group_id")
        metric_id = _canonical_string(self.metric_id, name="metric_id")
        nodes = _immutable_float(self.nodes, name="nodes", ndim=2)
        if nodes.shape[0] == 0 or nodes.shape[1] == 0:
            raise ValueError("nodes must have shape (positive count, positive dimension)")
        weights = _probability_vector(self.reference_weights, name="reference_weights")
        if weights.shape != (nodes.shape[0],):
            raise ValueError("reference_weights must match the group-node count")
        if isinstance(self.cover_radius, (bool, np.bool_)):
            raise TypeError("cover_radius must be a real scalar")
        radius = float(self.cover_radius)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("cover_radius must be finite and nonnegative")
        certified = _genuine_bool(
            self.cover_radius_certified,
            name="cover_radius_certified",
        )
        if self.measure_kind not in ("finite-mass", "continuous-density"):
            raise ValueError("unsupported measure_kind")
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "reference_weights", weights)
        object.__setattr__(self, "cover_radius", radius)
        object.__setattr__(self, "cover_radius_certified", certified)

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @classmethod
    def uniform_circle(
        cls,
        node_count: int,
        *,
        group_id: str,
        metric_id: str = "wrapped-angle-radians-v1",
    ) -> CompactGroupQuadratureV1:
        """Return an exactly covered uniform quadrature for the circle group.

        The nodes use the half-open interval ``[-pi, pi)`` and geodesic angular
        distance. The exact cover radius is ``pi / node_count``.
        """

        if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
            raise TypeError("node_count must be an integer")
        count = int(node_count)
        if count < 1:
            raise ValueError("node_count must be positive")
        angles = -math.pi + (2.0 * math.pi / count) * np.arange(count)
        return cls(
            group_id=group_id,
            metric_id=metric_id,
            nodes=angles[:, None],
            reference_weights=np.full(count, 1.0 / count),
            cover_radius=math.pi / count,
            cover_radius_certified=True,
            measure_kind="continuous-density",
        )


@dataclass(frozen=True, slots=True)
class SymmetryCompleteBeliefV1:
    """One quotient belief with an explicit conditional group law."""

    quotient_weights: FloatArray
    group_conditional_weights: FloatArray
    quadrature: CompactGroupQuadratureV1
    belief_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, CompactGroupQuadratureV1):
            raise TypeError("quadrature must be CompactGroupQuadratureV1")
        quotient = _probability_vector(self.quotient_weights, name="quotient_weights")
        conditionals = _conditional_probability_matrix(
            self.group_conditional_weights,
            quotient_count=quotient.size,
            group_count=self.quadrature.node_count,
            name="group_conditional_weights",
        )
        belief_id = _canonical_string(self.belief_id, name="belief_id")
        object.__setattr__(self, "quotient_weights", quotient)
        object.__setattr__(self, "group_conditional_weights", conditionals)
        object.__setattr__(self, "belief_id", belief_id)

    @property
    def quotient_count(self) -> int:
        return int(self.quotient_weights.size)

    @property
    def joint_weights(self) -> FloatArray:
        joint = np.ascontiguousarray(
            self.quotient_weights[:, None] * self.group_conditional_weights,
            dtype=np.float64,
        )
        return np.frombuffer(joint.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
            joint.shape
        )

    @classmethod
    def with_reference_group_law(
        cls,
        quotient_weights: ArrayLike,
        quadrature: CompactGroupQuadratureV1,
        *,
        belief_id: str,
    ) -> SymmetryCompleteBeliefV1:
        quotient = _probability_vector(quotient_weights, name="quotient_weights")
        conditionals = np.repeat(
            quadrature.reference_weights[None, :],
            quotient.size,
            axis=0,
        )
        return cls(quotient, conditionals, quadrature, belief_id)


__all__ = [
    "CompactGroupQuadratureV1",
    "EvidenceSemantics",
    "FloatArray",
    "GroupMeasureKind",
    "SYMMETRY_COMPLETE_BELIEF_CLAIM_BOUNDARY",
    "SYMMETRY_COMPLETE_BELIEF_VERSION",
    "SymmetryCompleteBeliefV1",
    "_NUMERICAL_ATOL",
    "_PROBABILITY_ATOL",
    "_canonical_string",
    "_finite_nonnegative_vector",
    "_genuine_bool",
    "_immutable_float",
    "_kl_divergence",
]
