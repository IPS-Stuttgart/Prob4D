"""Uncertainty-normalized source guard for the experimental causal gauge graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .alignment import WindowAlignment
from .causal_gauge_graph import (
    CAUSAL_GAUGE_GRAPH_MODE,
    CausalGaugeGraphReport,
    estimate_causal_multi_edge_gauge_graph,
)
from .composition_jacobian import composition_jacobian_mode
from .data import PredictionWindow
from .guarded_causal_gauge_graph import (
    _multi_edge_child_ids,
    _validated_causal_alignments,
    _validated_minimum_cycle_count,
)
from .observation_export import JointGaugePosterior, estimate_joint_gauge_tree
from .sim3 import Sim3
from .uncertainty_normalized_cycles import (
    UncertaintyNormalizedCycleAudit,
    audit_uncertainty_normalized_alignment_cycles,
)

FloatArray = NDArray[np.floating]
UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_MODE: Final = (
    "uncertainty_guarded_causal_full_joint_ci_graph_v1"
)
UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE: Final = (
    "source_only_minkowski_normalized_cycle_gate_then_full_joint_ci_with_"
    "exact_tree_fallback_v1"
)
_PRODUCTION_GAUGE_TREE_MODE: Final = "sequential_joint_spanning_tree_v1"


@dataclass(frozen=True)
class UncertaintyGuardedCausalGaugeGraphReport:
    """Normalized cycle gate and the estimator actually returned."""

    cycle_audit: UncertaintyNormalizedCycleAudit
    window_ids: tuple[str, ...]
    multi_edge_child_ids: tuple[str, ...]
    graph_report: CausalGaugeGraphReport | None
    fallback_applied: bool
    returned_posterior_mode: str
    minimum_cycles_per_multi_edge_child: int = 1
    dependence_semantics: str = UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE
    claim_bearing_provider_export: bool = False

    def __post_init__(self) -> None:
        if self.cycle_audit.maximum_normalized_score is None:
            raise ValueError(
                "uncertainty-guarded graph requires a normalized cycle threshold"
            )
        if self.cycle_audit.passed is None:
            raise ValueError("normalized cycle audit must have a pass decision")
        window_ids = tuple(str(value) for value in self.window_ids)
        if not window_ids or len(set(window_ids)) != len(window_ids):
            raise ValueError("uncertainty-guarded graph requires unique window IDs")
        child_ids = tuple(str(value) for value in self.multi_edge_child_ids)
        if any(not value for value in child_ids) or child_ids != tuple(
            sorted(set(child_ids))
        ):
            raise ValueError(
                "multi_edge_child_ids must be unique, nonempty, and sorted"
            )
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
            raise ValueError("fallback differs from the normalized cycle gate")
        if expected_fallback:
            if self.graph_report is not None:
                raise ValueError("rejected graph cannot retain a graph report")
            if self.returned_posterior_mode != _PRODUCTION_GAUGE_TREE_MODE:
                raise ValueError("rejected graph must return the production tree")
        else:
            if self.graph_report is None:
                raise ValueError("accepted normalized guard requires a graph report")
            if self.graph_report.window_ids != window_ids:
                raise ValueError("normalized guarded graph window order changed")
            if self.returned_posterior_mode != CAUSAL_GAUGE_GRAPH_MODE:
                raise ValueError("accepted normalized guard must return graph posterior")
        if (
            self.dependence_semantics
            != UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE
        ):
            raise ValueError("normalized guard dependence semantics changed")
        if self.claim_bearing_provider_export is not False:
            raise ValueError("uncertainty-normalized guarded graph is experimental")
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
            return "uncertainty_normalized_cycle_inconsistency"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_MODE,
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


def estimate_uncertainty_guarded_causal_multi_edge_gauge_graph(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
    *,
    initial_transform: Sim3,
    initial_covariance: FloatArray,
    maximum_normalized_cycle_score: float,
    representative_radius: float = 1.0,
    minimum_uncertainty_scale: float = 1e-12,
    minimum_cycles_per_multi_edge_child: int = 1,
    minimum_edge_weight: float = 0.0,
) -> tuple[JointGaugePosterior, UncertaintyGuardedCausalGaugeGraphReport]:
    """Gate the graph with an empirically calibrated, source-normalized score.

    A failed gate returns the exact analytic-Jacobian production tree. The
    normalized score uses only alignment transforms and their source-side
    covariances; it does not consume target truth or downstream innovations.
    """

    normalized_alignments = _validated_causal_alignments(windows, alignments)
    audit = audit_uncertainty_normalized_alignment_cycles(
        normalized_alignments,
        representative_radius=representative_radius,
        minimum_uncertainty_scale=minimum_uncertainty_scale,
        maximum_normalized_score=maximum_normalized_cycle_score,
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
            raise RuntimeError("normalized guard fallback tree mode changed")
        return posterior, UncertaintyGuardedCausalGaugeGraphReport(
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
    return posterior, UncertaintyGuardedCausalGaugeGraphReport(
        cycle_audit=audit,
        window_ids=tuple(window.window_id for window in windows),
        multi_edge_child_ids=multi_edge_child_ids,
        graph_report=graph_report,
        fallback_applied=False,
        returned_posterior_mode=posterior.mode,
        minimum_cycles_per_multi_edge_child=normalized_minimum,
    )


__all__ = [
    "UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_DEPENDENCE",
    "UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_MODE",
    "UncertaintyGuardedCausalGaugeGraphReport",
    "estimate_uncertainty_guarded_causal_multi_edge_gauge_graph",
]
