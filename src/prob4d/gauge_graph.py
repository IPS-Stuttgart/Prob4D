"""Order-invariant graph estimation and diagnostics for uncertain ``Sim(3)`` gauges.

The portable Prob4D observation stream keeps its frozen causal-tree default. This
module provides additive generalized-CI sequential inference, deterministic
global tree selection, full joint covariance propagation, and held-out loop-edge
diagnostics for controlled ablations.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .covariance import (
    covariance_eigendecomposition,
    covariance_statistics,
    regularized_inverse_psd,
)
from .gauge_ci import (
    GaugeCandidate,
    GeneralizedCIFusionResult,
    _compose_with_covariance,
    _inverse_with_covariance,
    _numerical_jacobian,
    _readonly_covariance,
    constraint_residual,
    fuse_sim3_generalized_ci,
    generalized_ci_weights,
    right_invariant_residual,
)
from .sim3 import Sim3

FloatArray = NDArray[np.floating]


class RelativeGaugeConstraintLike(Protocol):
    """Structural contract shared with :class:`prob4d.gauge.RelativeGaugeConstraint`."""

    reference_id: str
    moving_id: str
    reference_from_moving: Sim3
    covariance: FloatArray
    residual_rms: float
    num_correspondences: int


@dataclass(frozen=True)
class GaugeGraphEstimate:
    """One estimated gauge plus the candidate weights used to obtain it."""

    window_id: str
    global_from_local: Sim3
    covariance: FloatArray
    source_labels: tuple[str, ...] = ()
    source_weights: FloatArray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        if not window_id:
            raise ValueError("window_id must be nonempty")
        labels = tuple(map(str, self.source_labels))
        weights = np.asarray(self.source_weights, dtype=np.float64).copy()
        if weights.shape != (len(labels),):
            raise ValueError("source weights changed shape")
        if labels:
            if len(set(labels)) != len(labels):
                raise ValueError("source labels must be unique")
            if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-10):
                raise ValueError("source weights must form a probability vector")
        elif weights.size:
            raise ValueError("root estimate cannot carry source weights")
        weights.setflags(write=False)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "source_labels", labels)
        object.__setattr__(self, "source_weights", weights)
        object.__setattr__(
            self,
            "covariance",
            _readonly_covariance(self.covariance, name=f"gauge {window_id!r} covariance"),
        )


def _canonical_constraint_parts(
    constraint: RelativeGaugeConstraintLike,
) -> tuple[str, str, Sim3, FloatArray]:
    reference_id = str(constraint.reference_id)
    moving_id = str(constraint.moving_id)
    if not reference_id or not moving_id or reference_id == moving_id:
        raise ValueError("constraint window IDs must be nonempty and distinct")
    covariance = np.asarray(constraint.covariance, dtype=np.float64)
    if covariance.shape != (7, 7):
        raise ValueError("relative gauge covariance must have shape (7, 7)")
    covariance = _readonly_covariance(covariance, name="relative gauge covariance")
    left, right = sorted((reference_id, moving_id))
    if reference_id == left:
        return left, right, constraint.reference_from_moving, covariance
    inverse, inverse_covariance = _inverse_with_covariance(
        constraint.reference_from_moving,
        covariance,
    )
    return left, right, inverse, inverse_covariance


def constraint_content_id(constraint: RelativeGaugeConstraintLike) -> str:
    """Return an orientation-independent content ID for a relative constraint."""

    left, right, left_from_right, covariance = _canonical_constraint_parts(constraint)
    digest = hashlib.sha256()
    digest.update(left.encode("utf-8"))
    digest.update(b"\0")
    digest.update(right.encode("utf-8"))
    digest.update(b"\0")
    transform_values = np.round(
        np.asarray(left_from_right.as_vector(), dtype=np.float64),
        decimals=12,
    )
    covariance_values = np.round(
        np.asarray(covariance, dtype=np.float64),
        decimals=12,
    )
    transform_values[transform_values == 0.0] = 0.0
    covariance_values[covariance_values == 0.0] = 0.0
    digest.update(np.asarray(transform_values, dtype="<f8").tobytes())
    digest.update(np.asarray(covariance_values, dtype="<f8").tobytes())
    residual_rms = float(getattr(constraint, "residual_rms", 0.0))
    correspondences = int(getattr(constraint, "num_correspondences", 0))
    if not np.isfinite(residual_rms) or residual_rms < 0.0:
        raise ValueError("constraint residual_rms must be finite and non-negative")
    if correspondences < 0:
        raise ValueError("constraint num_correspondences must be non-negative")
    digest.update(np.asarray([residual_rms], dtype="<f8").tobytes())
    digest.update(np.asarray([correspondences], dtype="<i8").tobytes())
    return digest.hexdigest()


class OrderInvariantSequentialGaugeEstimator:
    """Sequential gauge estimator with manifold-local generalized CI.

    Temporal window order remains explicit, but the result is invariant to the
    order in which constraints are supplied. Multiple already-initialized parent
    candidates are fused jointly rather than through order-dependent pairwise CI.
    """

    def __init__(
        self,
        *,
        maximum_manifold_iterations: int = 12,
        maximum_weight_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> None:
        self.maximum_manifold_iterations = maximum_manifold_iterations
        self.maximum_weight_iterations = maximum_weight_iterations
        self.tolerance = tolerance

    def estimate(
        self,
        ordered_window_ids: Sequence[str],
        constraints: Sequence[RelativeGaugeConstraintLike],
        *,
        initial_transform: Sim3 | None = None,
        initial_covariance: FloatArray | None = None,
    ) -> dict[str, GaugeGraphEstimate]:
        window_ids = tuple(map(str, ordered_window_ids))
        if not window_ids or any(not value for value in window_ids):
            raise ValueError("ordered_window_ids must be nonempty")
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("window IDs must be unique")
        first_covariance = (
            np.diag([1e-10] * 7)
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=np.float64)
        )
        estimates: dict[str, GaugeGraphEstimate] = {
            window_ids[0]: GaugeGraphEstimate(
                window_id=window_ids[0],
                global_from_local=initial_transform or Sim3.identity(),
                covariance=first_covariance,
            )
        }
        known_ids = set(window_ids)
        for constraint in constraints:
            if (
                constraint.reference_id not in known_ids
                or constraint.moving_id not in known_ids
            ):
                raise ValueError(
                    "constraint references a window outside ordered_window_ids"
                )

        for window_id in window_ids[1:]:
            candidates_by_label: dict[str, GaugeCandidate] = {}
            for constraint in constraints:
                parent_id: str | None = None
                relative: Sim3 | None = None
                relative_covariance: FloatArray | None = None
                if (
                    constraint.moving_id == window_id
                    and constraint.reference_id in estimates
                ):
                    parent_id = constraint.reference_id
                    relative = constraint.reference_from_moving
                    relative_covariance = np.asarray(
                        constraint.covariance,
                        dtype=np.float64,
                    )
                elif (
                    constraint.reference_id == window_id
                    and constraint.moving_id in estimates
                ):
                    parent_id = constraint.moving_id
                    relative, relative_covariance = _inverse_with_covariance(
                        constraint.reference_from_moving,
                        np.asarray(constraint.covariance, dtype=np.float64),
                    )
                if parent_id is None or relative is None or relative_covariance is None:
                    continue
                parent = estimates[parent_id]
                transform, covariance = _compose_with_covariance(
                    parent.global_from_local,
                    parent.covariance,
                    relative,
                    relative_covariance,
                )
                label = f"{parent_id}:{constraint_content_id(constraint)}"
                candidates_by_label.setdefault(
                    label,
                    GaugeCandidate(
                        label=label,
                        global_from_local=transform,
                        covariance=covariance,
                    ),
                )
            if not candidates_by_label:
                raise ValueError(
                    f"window {window_id!r} has no constraint to an initialized gauge"
                )
            fusion = fuse_sim3_generalized_ci(
                tuple(candidates_by_label.values()),
                maximum_manifold_iterations=self.maximum_manifold_iterations,
                maximum_weight_iterations=self.maximum_weight_iterations,
                tolerance=self.tolerance,
            )
            estimates[window_id] = GaugeGraphEstimate(
                window_id=window_id,
                global_from_local=fusion.global_from_local,
                covariance=fusion.covariance,
                source_labels=fusion.candidate_labels,
                source_weights=fusion.weights,
            )
        return estimates


@dataclass(frozen=True)
class GaugeGraphEdge:
    """Canonical undirected edge storing the transform from right to left."""

    edge_id: str
    left_id: str
    right_id: str
    left_from_right: Sim3
    covariance: FloatArray
    residual_rms: float
    num_correspondences: int

    def __post_init__(self) -> None:
        if (
            not self.edge_id
            or not self.left_id
            or not self.right_id
            or self.left_id >= self.right_id
        ):
            raise ValueError("canonical gauge edge identity is invalid")
        if len(self.edge_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.edge_id
        ):
            raise ValueError("edge_id must be a lowercase SHA-256 digest")
        if not np.isfinite(self.residual_rms) or self.residual_rms < 0.0:
            raise ValueError("edge residual_rms must be finite and non-negative")
        if self.num_correspondences < 0:
            raise ValueError("edge num_correspondences must be non-negative")
        object.__setattr__(
            self,
            "covariance",
            _readonly_covariance(self.covariance, name=f"edge {self.edge_id} covariance"),
        )

    @classmethod
    def from_constraint(cls, constraint: RelativeGaugeConstraintLike) -> GaugeGraphEdge:
        left, right, transform, covariance = _canonical_constraint_parts(constraint)
        return cls(
            edge_id=constraint_content_id(constraint),
            left_id=left,
            right_id=right,
            left_from_right=transform,
            covariance=covariance,
            residual_rms=float(getattr(constraint, "residual_rms", 0.0)),
            num_correspondences=int(getattr(constraint, "num_correspondences", 0)),
        )

    def oriented(self, parent_id: str, child_id: str) -> tuple[Sim3, FloatArray]:
        """Return ``parent_from_child`` and its covariance."""

        if parent_id == self.left_id and child_id == self.right_id:
            return self.left_from_right, self.covariance
        if parent_id == self.right_id and child_id == self.left_id:
            return _inverse_with_covariance(self.left_from_right, self.covariance)
        raise ValueError("requested orientation does not match the edge endpoints")

    @property
    def quality_key(self) -> tuple[float, int, float, str, str, str]:
        _, _, log_determinant = covariance_statistics(
            self.covariance,
            name=f"edge {self.edge_id} covariance",
        )
        return (
            float(log_determinant),
            -self.num_correspondences,
            self.residual_rms,
            self.left_id,
            self.right_id,
            self.edge_id,
        )


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return True


def select_uncertainty_volume_spanning_tree(
    window_ids: Sequence[str],
    constraints: Sequence[RelativeGaugeConstraintLike],
) -> tuple[GaugeGraphEdge, ...]:
    """Select a deterministic minimum-uncertainty-volume spanning tree."""

    identifiers = tuple(map(str, window_ids))
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("window_ids must be nonempty and unique")
    known = set(identifiers)
    edges_by_id: dict[str, GaugeGraphEdge] = {}
    for constraint in constraints:
        edge = GaugeGraphEdge.from_constraint(constraint)
        if edge.left_id not in known or edge.right_id not in known:
            raise ValueError("constraint references a window outside window_ids")
        edges_by_id.setdefault(edge.edge_id, edge)
    disjoint = _DisjointSet(identifiers)
    selected: list[GaugeGraphEdge] = []
    for edge in sorted(edges_by_id.values(), key=lambda value: value.quality_key):
        if disjoint.union(edge.left_id, edge.right_id):
            selected.append(edge)
            if len(selected) == len(identifiers) - 1:
                break
    if len(selected) != len(identifiers) - 1:
        components = sorted({disjoint.find(value) for value in identifiers})
        raise ValueError(
            "relative-gauge graph is disconnected; components are "
            + ", ".join(components)
        )
    return tuple(selected)


@dataclass(frozen=True)
class GaugeGraphPosterior:
    """Gauge means and full cross-window covariance from a selected graph tree."""

    window_ids: tuple[str, ...]
    estimates: Mapping[str, Sim3]
    joint_covariance: FloatArray
    root_window_id: str
    parent_window_ids: tuple[str | None, ...]
    parent_edge_ids: tuple[str | None, ...]
    selected_edge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        window_ids = tuple(map(str, self.window_ids))
        if not window_ids or len(set(window_ids)) != len(window_ids):
            raise ValueError("posterior window_ids must be nonempty and unique")
        if self.root_window_id not in window_ids:
            raise ValueError("root_window_id is not in window_ids")
        if set(self.estimates) != set(window_ids):
            raise ValueError("posterior estimates do not match window_ids")
        if (
            len(self.parent_window_ids) != len(window_ids)
            or len(self.parent_edge_ids) != len(window_ids)
        ):
            raise ValueError("posterior parent lineage changed length")
        dimension = 7 * len(window_ids)
        covariance = np.asarray(self.joint_covariance, dtype=np.float64)
        if covariance.shape != (dimension, dimension):
            raise ValueError("joint gauge covariance changed shape")
        covariance = _readonly_covariance(covariance, name="joint gauge covariance")
        if len(self.selected_edge_ids) != len(window_ids) - 1:
            raise ValueError("selected edge count does not form a spanning tree")
        if len(set(self.selected_edge_ids)) != len(self.selected_edge_ids):
            raise ValueError("selected edge IDs must be unique")
        root_position = window_ids.index(self.root_window_id)
        if (
            self.parent_window_ids[root_position] is not None
            or self.parent_edge_ids[root_position] is not None
        ):
            raise ValueError("root gauge must not have a parent")
        object.__setattr__(self, "window_ids", window_ids)
        object.__setattr__(self, "joint_covariance", covariance)
        object.__setattr__(self, "estimates", dict(self.estimates))


def estimate_joint_gauge_tree(
    window_ids: Sequence[str],
    constraints: Sequence[RelativeGaugeConstraintLike],
    *,
    root_window_id: str,
    initial_transform: Sim3,
    initial_covariance: FloatArray,
) -> GaugeGraphPosterior:
    """Propagate a globally selected tree into one full joint covariance."""

    identifiers = tuple(map(str, window_ids))
    if root_window_id not in identifiers:
        raise ValueError("root_window_id must be included in window_ids")
    selected = select_uncertainty_volume_spanning_tree(identifiers, constraints)
    adjacency: dict[str, list[GaugeGraphEdge]] = {value: [] for value in identifiers}
    for edge in selected:
        adjacency[edge.left_id].append(edge)
        adjacency[edge.right_id].append(edge)
    for value in adjacency.values():
        value.sort(key=lambda edge: (edge.left_id, edge.right_id, edge.edge_id))

    positions = {window_id: index for index, window_id in enumerate(identifiers)}
    dimension = 7 * len(identifiers)
    joint = np.zeros((dimension, dimension), dtype=np.float64)
    root_covariance = _readonly_covariance(
        initial_covariance,
        name="initial gauge covariance",
    )
    root_position = positions[root_window_id]
    root_slice = slice(7 * root_position, 7 * (root_position + 1))
    joint[root_slice, root_slice] = root_covariance
    estimates: dict[str, Sim3] = {root_window_id: initial_transform}
    parent_ids: dict[str, str | None] = {root_window_id: None}
    parent_edges: dict[str, str | None] = {root_window_id: None}
    processed: list[str] = [root_window_id]
    queue: deque[str] = deque([root_window_id])

    while queue:
        parent_id = queue.popleft()
        for edge in adjacency[parent_id]:
            child_id = edge.right_id if parent_id == edge.left_id else edge.left_id
            if child_id in estimates:
                continue
            relative, relative_covariance = edge.oriented(parent_id, child_id)
            parent = estimates[parent_id]
            child = parent.compose(relative)
            parent_jacobian = _numerical_jacobian(
                lambda value: Sim3.from_vector(value).compose(relative).as_vector(),
                parent.as_vector(),
            )
            relative_jacobian = _numerical_jacobian(
                lambda value: parent.compose(Sim3.from_vector(value)).as_vector(),
                relative.as_vector(),
            )
            parent_position = positions[parent_id]
            child_position = positions[child_id]
            parent_slice = slice(7 * parent_position, 7 * (parent_position + 1))
            child_slice = slice(7 * child_position, 7 * (child_position + 1))
            for previous_id in processed:
                previous_position = positions[previous_id]
                previous_slice = slice(
                    7 * previous_position,
                    7 * (previous_position + 1),
                )
                cross = parent_jacobian @ joint[parent_slice, previous_slice]
                joint[child_slice, previous_slice] = cross
                joint[previous_slice, child_slice] = cross.T
            child_covariance = (
                parent_jacobian
                @ joint[parent_slice, parent_slice]
                @ parent_jacobian.T
                + relative_jacobian
                @ relative_covariance
                @ relative_jacobian.T
            )
            joint[child_slice, child_slice] = 0.5 * (
                child_covariance + child_covariance.T
            )
            estimates[child_id] = child
            parent_ids[child_id] = parent_id
            parent_edges[child_id] = edge.edge_id
            processed.append(child_id)
            queue.append(child_id)

    if len(estimates) != len(identifiers):
        raise RuntimeError("selected spanning tree traversal did not reach every window")
    _, eigenvalues, eigenvectors = covariance_eigendecomposition(
        0.5 * (joint + joint.T),
        name="propagated joint gauge covariance",
        eigenvalue_floor=1e-15,
    )
    joint = (eigenvectors * eigenvalues) @ eigenvectors.T
    return GaugeGraphPosterior(
        window_ids=identifiers,
        estimates=estimates,
        joint_covariance=joint,
        root_window_id=root_window_id,
        parent_window_ids=tuple(parent_ids[window_id] for window_id in identifiers),
        parent_edge_ids=tuple(parent_edges[window_id] for window_id in identifiers),
        selected_edge_ids=tuple(edge.edge_id for edge in selected),
    )


@dataclass(frozen=True)
class LoopClosureDiagnostic:
    """Independent non-tree edge check against a tree-derived posterior."""

    edge_id: str
    left_id: str
    right_id: str
    residual: FloatArray
    residual_norm: float
    normalized_innovation_squared: float
    threshold: float
    suspicious: bool

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual, dtype=np.float64).copy()
        if residual.shape != (7,) or not np.all(np.isfinite(residual)):
            raise ValueError("loop residual must be a finite seven-vector")
        if not np.isfinite(self.residual_norm) or self.residual_norm < 0.0:
            raise ValueError("loop residual norm must be finite and non-negative")
        if (
            not np.isfinite(self.normalized_innovation_squared)
            or self.normalized_innovation_squared < 0.0
        ):
            raise ValueError("loop NIS must be finite and non-negative")
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("loop NIS threshold must be finite and positive")
        residual.setflags(write=False)
        object.__setattr__(self, "residual", residual)


def loop_closure_diagnostics(
    posterior: GaugeGraphPosterior,
    constraints: Sequence[RelativeGaugeConstraintLike],
    *,
    nis_threshold: float = 24.321886347856854,
) -> tuple[LoopClosureDiagnostic, ...]:
    """Evaluate non-tree overlap edges without refitting the selected tree.

    The threshold defaults to the 0.999 quantile of a seven-dimensional
    chi-squared reference distribution. The calculation treats each held-out
    edge as independent of the selected tree; correlated MotionCrafter residuals
    should therefore be interpreted as diagnostics rather than calibrated tests.
    """

    if not np.isfinite(nis_threshold) or nis_threshold <= 0.0:
        raise ValueError("nis_threshold must be finite and positive")
    selected = set(posterior.selected_edge_ids)
    positions = {window_id: index for index, window_id in enumerate(posterior.window_ids)}
    edges_by_id: dict[str, GaugeGraphEdge] = {}
    for constraint in constraints:
        edge = GaugeGraphEdge.from_constraint(constraint)
        edges_by_id.setdefault(edge.edge_id, edge)
    diagnostics: list[LoopClosureDiagnostic] = []
    for edge in sorted(edges_by_id.values(), key=lambda value: value.edge_id):
        if edge.edge_id in selected:
            continue
        left = posterior.estimates[edge.left_id]
        right = posterior.estimates[edge.right_id]
        predicted = left.inverse().compose(right)
        residual = constraint_residual(edge.left_from_right, predicted)

        left_vector = left.as_vector()
        right_vector = right.as_vector()
        left_jacobian = _numerical_jacobian(
            lambda value: Sim3.from_vector(value).inverse().compose(right).as_vector(),
            left_vector,
        )
        right_jacobian = _numerical_jacobian(
            lambda value: left.inverse().compose(Sim3.from_vector(value)).as_vector(),
            right_vector,
        )
        left_position = positions[edge.left_id]
        right_position = positions[edge.right_id]
        left_slice = slice(7 * left_position, 7 * (left_position + 1))
        right_slice = slice(7 * right_position, 7 * (right_position + 1))
        joint = posterior.joint_covariance
        predicted_covariance = (
            left_jacobian @ joint[left_slice, left_slice] @ left_jacobian.T
            + right_jacobian @ joint[right_slice, right_slice] @ right_jacobian.T
            + left_jacobian @ joint[left_slice, right_slice] @ right_jacobian.T
            + right_jacobian @ joint[right_slice, left_slice] @ left_jacobian.T
        )
        innovation_covariance = _readonly_covariance(
            predicted_covariance + edge.covariance,
            name=f"loop edge {edge.edge_id} innovation covariance",
        )
        precision = regularized_inverse_psd(
            innovation_covariance,
            name=f"loop edge {edge.edge_id} innovation covariance",
        )
        nis = float(residual @ precision @ residual)
        diagnostics.append(
            LoopClosureDiagnostic(
                edge_id=edge.edge_id,
                left_id=edge.left_id,
                right_id=edge.right_id,
                residual=residual,
                residual_norm=float(np.linalg.norm(residual)),
                normalized_innovation_squared=nis,
                threshold=nis_threshold,
                suspicious=nis > nis_threshold,
            )
        )
    return tuple(diagnostics)


__all__ = [
    "GaugeCandidate",
    "GaugeGraphEdge",
    "GaugeGraphEstimate",
    "GaugeGraphPosterior",
    "GeneralizedCIFusionResult",
    "LoopClosureDiagnostic",
    "OrderInvariantSequentialGaugeEstimator",
    "RelativeGaugeConstraintLike",
    "constraint_content_id",
    "constraint_residual",
    "estimate_joint_gauge_tree",
    "fuse_sim3_generalized_ci",
    "generalized_ci_weights",
    "loop_closure_diagnostics",
    "right_invariant_residual",
    "select_uncertainty_volume_spanning_tree",
]
