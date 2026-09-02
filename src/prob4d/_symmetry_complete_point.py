"""Point-completion and shared-query audits for group beliefs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._symmetry_complete_base import SymmetryCompleteBeliefV1, _immutable_float
from .axial_gauge import GaussianQueryMixture

PointCompletionStatus: TypeAlias = Literal[
    "finite-supported",
    "continuous-singular",
    "outside-support",
]


@dataclass(frozen=True, slots=True)
class PointCompletionAuditV1:
    """Audit specificity introduced by selecting one group representative."""

    selected_group_node_by_quotient: NDArray[np.int64]
    discretized_specificity_nats: float | None
    status: PointCompletionStatus

    def __post_init__(self) -> None:
        indices = np.asarray(self.selected_group_node_by_quotient)
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise ValueError("selected_group_node_by_quotient must be an integer vector")
        immutable = np.frombuffer(
            np.ascontiguousarray(indices, dtype=np.int64).tobytes(order="C"),
            dtype=np.dtype("<i8"),
        )
        allowed: tuple[PointCompletionStatus, ...] = (
            "finite-supported",
            "continuous-singular",
            "outside-support",
        )
        if self.status not in allowed:
            raise ValueError("unsupported point-completion status")
        if self.discretized_specificity_nats is None:
            if self.status != "outside-support":
                raise ValueError("supported numerical completion requires specificity")
            specificity: float | None = None
        else:
            specificity = float(self.discretized_specificity_nats)
            if not math.isfinite(specificity) or specificity < 0.0:
                raise ValueError(
                    "discretized_specificity_nats must be finite and nonnegative"
                )
            if self.status == "outside-support":
                raise ValueError("outside-support completion must not report specificity")
        object.__setattr__(self, "selected_group_node_by_quotient", immutable)
        object.__setattr__(self, "discretized_specificity_nats", specificity)

    @property
    def supported_by_quadrature(self) -> bool:
        return self.status != "outside-support"

    @property
    def physical_point_completion_has_finite_kl(self) -> bool:
        return self.status == "finite-supported"


def audit_point_completion(
    belief: SymmetryCompleteBeliefV1,
    selected_group_node_by_quotient: ArrayLike,
) -> PointCompletionAuditV1:
    """Audit a delta completion of each unresolved conditional group law.

    For a finite group law, the added KL specificity is the quotient-weighted
    ``-log`` probability of the selected nodes. For a continuous group law, a
    point mass is singular with respect to the physical conditional density; the
    returned discretized value is only a resolution-dependent diagnostic. A
    selected numerical node outside positive quadrature support fails closed.
    """

    if not isinstance(belief, SymmetryCompleteBeliefV1):
        raise TypeError("belief must be SymmetryCompleteBeliefV1")
    raw = np.asarray(selected_group_node_by_quotient)
    if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.shape != (
        belief.quotient_count,
    ):
        raise ValueError(
            "selected_group_node_by_quotient must contain one integer per quotient"
        )
    selected = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= belief.quadrature.node_count):
        raise ValueError("selected group-node index is out of range")
    specificity = 0.0
    for quotient_index, quotient_mass in enumerate(belief.quotient_weights):
        if quotient_mass <= 0.0:
            continue
        node_mass = float(
            belief.group_conditional_weights[
                quotient_index,
                selected[quotient_index],
            ]
        )
        if node_mass <= 0.0:
            return PointCompletionAuditV1(selected, None, "outside-support")
        specificity += float(quotient_mass) * -math.log(node_mass)
    status: PointCompletionStatus = (
        "continuous-singular"
        if belief.quadrature.measure_kind == "continuous-density"
        else "finite-supported"
    )
    return PointCompletionAuditV1(selected, specificity, status)


def pushforward_shared_group_query(
    belief: SymmetryCompleteBeliefV1,
    query_atoms: ArrayLike,
    *,
    noise_covariance: ArrayLike | None = None,
) -> GaussianQueryMixture:
    """Push one shared quotient/group draw into a complete vector query.

    ``query_atoms[c, k]`` must be the complete query vector under quotient class
    ``c`` and group node ``k``. All vector coordinates in one row share that
    same group node; the function never constructs independent per-coordinate
    gauge draws.
    """

    if not isinstance(belief, SymmetryCompleteBeliefV1):
        raise TypeError("belief must be SymmetryCompleteBeliefV1")
    atoms = _immutable_float(query_atoms, name="query_atoms", ndim=3)
    expected_prefix = (belief.quotient_count, belief.quadrature.node_count)
    if atoms.shape[:2] != expected_prefix or atoms.shape[2] == 0:
        raise ValueError(
            "query_atoms must have shape (quotient_count, group_count, positive dimension)"
        )
    dimension = int(atoms.shape[2])
    noise = (
        np.zeros((dimension, dimension), dtype=np.float64)
        if noise_covariance is None
        else noise_covariance
    )
    return GaussianQueryMixture(
        atoms=atoms.reshape(-1, dimension),
        weights=belief.joint_weights.reshape(-1),
        noise_covariance=noise,
    )


__all__ = [
    "PointCompletionAuditV1",
    "PointCompletionStatus",
    "audit_point_completion",
    "pushforward_shared_group_query",
]
