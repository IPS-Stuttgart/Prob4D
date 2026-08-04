"""Source-only admission guard for the experimental causal gauge graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .alignment import WindowAlignment
from .alignment_cycles import AlignmentCycleAudit, audit_alignment_cycles
from .causal_gauge_graph import (
    CAUSAL_GAUGE_GRAPH_MODE,
    CausalGaugeGraphReport,
    estimate_causal_multi_edge_gauge_graph,
)
from .composition_jacobian import composition_jacobian_mode
from .data import PredictionWindow
from .observation_export import JointGaugePosterior, estimate_joint_gauge_tree
from .sim3 import Sim3

FloatArray = NDArray[np.floating]
GUARDED_CAUSAL_GAUGE_GRAPH_MODE: Final = "guarded_causal_full_joint_ci_graph_v1"
GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE: Final = (
    "source_only_cycle_gate_then_full_joint_ci_with_exact_tree_fallback_v1"
)
_PRODUCTION_GAUGE_TREE_MODE: Final = "sequential_joint_spanning_tree_v1"


@dataclass(frozen=True)
class GuardedCausalGaugeGraphReport:
    """Source-only cycle gate and the estimator actually returned."""

    cycle_audit: AlignmentCycleAudit
    window_ids: tuple[str, ...]
    multi_edge_child_ids: tuple[str, ...]
    graph_report: CausalGaugeGraphReport | None
    fallback_applied: bool
    returned_posterior_mode: str
    minimum_cycles_per_multi_edge_child: int = 1
    dependence_semantics: str = GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE
    claim_bearing_provider_export: bool = False

    def __post_init__(self) -> None:
        if self.cycle_audit.maximum_representative_displacement is None:
            raise ValueError("guarded gauge graph requires a declared cycle threshold")
        if self.cycle_audit.passed is None:
            raise ValueError("guarded gauge graph cycle audit must have a pass decision")
        window_ids = tuple(str(value) for value in self.window_ids)
        if not window_ids or len(set(window_ids)) != len(window_ids):
            raise ValueError("guarded gauge graph requires unique window IDs")
        child_ids = tuple(str(value) for value in self.multi_edge_child_ids)
        if any(not value for value in child_ids) or child_ids != tuple(
            sorted(set(child_ids))
        ):
            raise ValueError("multi_edge_child_ids must be unique, nonempty, and sorted")
        if not set(child_ids).issubset(window_ids[1:]):
            raise ValueError("multi_edge_child_ids must identify non-anchor windows")
        normalized_minimum = _validated_minimum_cycle_count(
            self.minimum_cycles_per_multi_edge_child
        )
        if not isinstance(self.fallback_applied, (bool, np.bool_)):
            raise ValueError("fallback_applied must be boolean")
        if (
            not isinstance(self.returned_posterior_mode, str)
            or not self.returned_posterior_mode
        ):
            raise ValueError("returned_posterior_mode must be nonempty text")
        expected_fallback = (
            not self.cycle_support_sufficient or self.cycle_audit.passed is False
        )
        if bool(self.fallback_applied) != expected_fallback:
            raise ValueError("guarded gauge-graph fallback differs from the cycle gate")
        if expected_fallback:
            if self.graph_report is not None:
                raise ValueError(
                    "rejected guarded graph cannot retain an admitted graph report"
                )
            if self.returned_posterior_mode != _PRODUCTION_GAUGE_TREE_MODE:
                raise ValueError("rejected guarded graph must return the production tree")
        else:
            if self.graph_report is None:
                raise ValueError("accepted guarded graph requires its graph report")
            if self.graph_report.window_ids != window_ids:
                raise ValueError("guarded graph report window order changed")
            if self.returned_posterior_mode != CAUSAL_GAUGE_GRAPH_MODE:
                raise ValueError("accepted guarded graph must return the graph posterior")
        if self.dependence_semantics != GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE:
            raise ValueError("guarded gauge-graph dependence semantics changed")
        if self.claim_bearing_provider_export is not False:
            raise ValueError("guarded causal gauge graph cannot be claim-bearing")
        object.__setattr__(self, "window_ids", window_ids)
        object.__setattr__(self, "multi_edge_child_ids", child_ids)
        object.__setattr__(self, "fallback_applied", bool(self.fallback_applied))
        object.__setattr__(
            self,
            "minimum_cycles_per_multi_edge_child",
            normalized_minimum,
        )

    @property
    def cycle_count_by_child(self) -> dict[str, int]:
        counts = {child_id: 0 for child_id in self.multi_edge_child_ids}
        for cycle in self.cycle_audit.cycles:
            if cycle.moving_id in counts:
                counts[cycle.moving_id] += 1
        return counts

    @property
    def unsupported_multi_edge_child_ids(self) -> tuple[str, ...]:
        return tuple(
            child_id
            for child_id, count in self.cycle_count_by_child.items()
            if count < self.minimum_cycles_per_multi_edge_child
        )

    @property
    def cycle_support_sufficient(self) -> bool:
        return not self.unsupported_multi_edge_child_ids

    @property
    def fallback_reason(self) -> str | None:
        if not self.cycle_support_sufficient:
            return "insufficient_cycle_support"
        if self.cycle_audit.passed is False:
            return "cycle_inconsistency"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": GUARDED_CAUSAL_GAUGE_GRAPH_MODE,
            "dependence_semantics": self.dependence_semantics,
            "cycle_audit": self.cycle_audit.to_dict(),
            "window_ids": list(self.window_ids),
            "multi_edge_child_ids": list(self.multi_edge_child_ids),
            "cycle_count_by_child": self.cycle_count_by_child,
            "minimum_cycles_per_multi_edge_child": (
                self.minimum_cycles_per_multi_edge_child
            ),
            "unsupported_multi_edge_child_ids": list(
                self.unsupported_multi_edge_child_ids
            ),
            "cycle_support_sufficient": self.cycle_support_sufficient,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "returned_posterior_mode": self.returned_posterior_mode,
            "graph": None if self.graph_report is None else self.graph_report.to_dict(),
            "claim_bearing_provider_export": self.claim_bearing_provider_export,
        }


def _validated_minimum_cycle_count(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "minimum_cycles_per_multi_edge_child must be a positive integer"
        )
    normalized = int(value)
    if normalized != value or normalized < 1:
        raise ValueError(
            "minimum_cycles_per_multi_edge_child must be a positive integer"
        )
    return normalized


def _validated_causal_alignments(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
) -> tuple[WindowAlignment, ...]:
    window_ids = tuple(window.window_id for window in windows)
    if not window_ids or len(set(window_ids)) != len(window_ids):
        raise ValueError("guarded gauge graph requires unique ordered window IDs")
    position = {window_id: index for index, window_id in enumerate(window_ids)}
    normalized = tuple(alignments)
    for alignment in normalized:
        if alignment.reference_id not in position or alignment.moving_id not in position:
            raise ValueError("guarded gauge graph alignment references an unknown window")
        if position[alignment.reference_id] >= position[alignment.moving_id]:
            raise ValueError(
                "guarded gauge graph requires prefix-valid directed alignments"
            )
    return normalized


def _multi_edge_child_ids(
    alignments: Sequence[WindowAlignment],
) -> tuple[str, ...]:
    parents_by_child: dict[str, set[str]] = {}
    for alignment in alignments:
        parents_by_child.setdefault(alignment.moving_id, set()).add(
            alignment.reference_id
        )
    return tuple(
        sorted(
            child_id
            for child_id, parent_ids in parents_by_child.items()
            if len(parent_ids) > 1
        )
    )


def estimate_guarded_causal_multi_edge_gauge_graph(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
    *,
    initial_transform: Sim3,
    initial_covariance: FloatArray,
    maximum_cycle_displacement: float,
    representative_radius: float = 1.0,
    minimum_cycles_per_multi_edge_child: int = 1,
    minimum_edge_weight: float = 0.0,
) -> tuple[JointGaugePosterior, GuardedCausalGaugeGraphReport]:
    """Apply a source-only cycle gate before admitting the experimental graph.

    The threshold must be frozen from source or calibration data. A failed gate
    returns the exact analytic-Jacobian production spanning tree; it does not
    discard the case, select an edge from target truth, or partially retain graph
    updates. A passing gate runs the unchanged full-joint CI graph.
    """

    normalized_alignments = _validated_causal_alignments(windows, alignments)
    audit = audit_alignment_cycles(
        normalized_alignments,
        representative_radius=representative_radius,
        maximum_representative_displacement=maximum_cycle_displacement,
    )
    normalized_minimum = _validated_minimum_cycle_count(
        minimum_cycles_per_multi_edge_child
    )
    multi_edge_child_ids = _multi_edge_child_ids(normalized_alignments)
    cycle_count_by_child = {child_id: 0 for child_id in multi_edge_child_ids}
    for cycle in audit.cycles:
        if cycle.moving_id in cycle_count_by_child:
            cycle_count_by_child[cycle.moving_id] += 1
    support_sufficient = all(
        count >= normalized_minimum for count in cycle_count_by_child.values()
    )
    fallback_required = not support_sufficient or audit.passed is False
    if fallback_required:
        with composition_jacobian_mode("analytic"):
            posterior = estimate_joint_gauge_tree(
                windows,
                normalized_alignments,
                initial_transform=initial_transform,
                initial_covariance=initial_covariance,
            )
        if posterior.mode != _PRODUCTION_GAUGE_TREE_MODE:
            raise RuntimeError("guarded gauge-graph fallback tree mode changed")
        return posterior, GuardedCausalGaugeGraphReport(
            cycle_audit=audit,
            window_ids=tuple(window.window_id for window in windows),
            multi_edge_child_ids=multi_edge_child_ids,
            graph_report=None,
            fallback_applied=True,
            returned_posterior_mode=posterior.mode,
            minimum_cycles_per_multi_edge_child=normalized_minimum,
        )

    posterior, graph_report = estimate_causal_multi_edge_gauge_graph(
        windows,
        normalized_alignments,
        initial_transform=initial_transform,
        initial_covariance=initial_covariance,
        minimum_edge_weight=minimum_edge_weight,
    )
    return posterior, GuardedCausalGaugeGraphReport(
        cycle_audit=audit,
        window_ids=tuple(window.window_id for window in windows),
        multi_edge_child_ids=multi_edge_child_ids,
        graph_report=graph_report,
        fallback_applied=False,
        returned_posterior_mode=posterior.mode,
        minimum_cycles_per_multi_edge_child=normalized_minimum,
    )


__all__ = [
    "GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE",
    "GUARDED_CAUSAL_GAUGE_GRAPH_MODE",
    "GuardedCausalGaugeGraphReport",
    "estimate_guarded_causal_multi_edge_gauge_graph",
]
