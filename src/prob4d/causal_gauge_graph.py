"""Correlation-aware causal multi-edge gauge graph by full-joint CI.

The claim-bearing provider-v2 estimator deliberately retains a transparent
single-parent spanning tree. This module is a separately labelled experimental
mode: every prefix-valid overlap edge predicts an augmented joint state
containing all previously admitted gauges and the new child gauge. Covariance
intersection fuses those augmented distributions while treating their unknown
shared-backbone and shared-frame correlation conservatively.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .alignment import WindowAlignment
from .composition_jacobian import analytic_sim3_compose_jacobians
from .covariance import regularized_inverse_psd, validated_covariance_psd
from .data import PredictionWindow
from .observation_export import JointGaugePosterior
from .sim3 import Sim3

FloatArray = NDArray[np.floating]
CAUSAL_GAUGE_GRAPH_MODE: Final = "causal_full_joint_ci_graph_v1"
CAUSAL_GAUGE_GRAPH_DEPENDENCE: Final = (
    "unknown_cross_edge_correlation_full_joint_covariance_intersection_v1"
)


@dataclass(frozen=True)
class CausalGaugeGraphStep:
    """One child admission and the conservative edge weights used for it."""

    child_window_id: str
    candidate_parent_ids: tuple[str, ...]
    candidate_alignment_indices: tuple[int, ...]
    covariance_intersection_weights: FloatArray

    def __post_init__(self) -> None:
        child = str(self.child_window_id)
        parents = tuple(str(value) for value in self.candidate_parent_ids)
        indices = tuple(int(value) for value in self.candidate_alignment_indices)
        weights = np.asarray(
            self.covariance_intersection_weights,
            dtype=np.float64,
        ).copy()
        if not child or not parents or any(not value for value in parents):
            raise ValueError("causal gauge-graph step requires nonempty window IDs")
        if len(parents) != len(indices) or weights.shape != (len(parents),):
            raise ValueError("causal gauge-graph step fields changed length")
        if len(set(indices)) != len(indices) or any(value < 0 for value in indices):
            raise ValueError("causal gauge-graph alignment indices must be unique")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "causal gauge-graph CI weights must be finite and nonnegative"
            )
        if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("causal gauge-graph CI weights must sum to one")
        weights.setflags(write=False)
        object.__setattr__(self, "child_window_id", child)
        object.__setattr__(self, "candidate_parent_ids", parents)
        object.__setattr__(self, "candidate_alignment_indices", indices)
        object.__setattr__(self, "covariance_intersection_weights", weights)

    def to_dict(self) -> dict[str, object]:
        return {
            "child_window_id": self.child_window_id,
            "candidate_parent_ids": list(self.candidate_parent_ids),
            "candidate_alignment_indices": list(self.candidate_alignment_indices),
            "covariance_intersection_weights": [
                float(value) for value in self.covariance_intersection_weights
            ],
        }


@dataclass(frozen=True)
class CausalGaugeGraphReport:
    """Auditable edge admission and dependence semantics for one graph run."""

    window_ids: tuple[str, ...]
    steps: tuple[CausalGaugeGraphStep, ...]
    dependence_semantics: str = CAUSAL_GAUGE_GRAPH_DEPENDENCE
    composition_jacobian_mode: str = "analytic"
    claim_bearing_provider_export: bool = False

    def __post_init__(self) -> None:
        windows = tuple(str(value) for value in self.window_ids)
        if not windows or len(set(windows)) != len(windows):
            raise ValueError("causal gauge-graph report requires unique window IDs")
        if len(self.steps) != len(windows) - 1:
            raise ValueError("causal gauge-graph report requires one step per child")
        if tuple(step.child_window_id for step in self.steps) != windows[1:]:
            raise ValueError("causal gauge-graph steps differ from window order")
        if self.dependence_semantics != CAUSAL_GAUGE_GRAPH_DEPENDENCE:
            raise ValueError("causal gauge-graph dependence semantics changed")
        if self.composition_jacobian_mode != "analytic":
            raise ValueError(
                "causal gauge graph requires analytic composition Jacobians"
            )
        if self.claim_bearing_provider_export is not False:
            raise ValueError("experimental causal gauge graph cannot be claim-bearing")
        object.__setattr__(self, "window_ids", windows)
        object.__setattr__(self, "steps", tuple(self.steps))

    @property
    def admitted_edge_count(self) -> int:
        return sum(len(step.candidate_alignment_indices) for step in self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": CAUSAL_GAUGE_GRAPH_MODE,
            "window_ids": list(self.window_ids),
            "steps": [step.to_dict() for step in self.steps],
            "admitted_edge_count": self.admitted_edge_count,
            "dependence_semantics": self.dependence_semantics,
            "composition_jacobian_mode": self.composition_jacobian_mode,
            "claim_bearing_provider_export": self.claim_bearing_provider_export,
        }


def _validate_rotation_coordinates(vector: FloatArray, *, name: str) -> None:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or values.size % 7 != 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite ordered Sim(3) coordinates")
    rotations = values.reshape(-1, 7)[:, 1:4]
    if np.any(np.linalg.norm(rotations, axis=1) >= np.pi - 1e-7):
        raise ValueError(f"{name} reaches the SO(3) axis-angle branch cut")


def _candidate_joint_distribution(
    *,
    previous_window_ids: tuple[str, ...],
    estimates: dict[str, Sim3],
    joint_covariance: FloatArray,
    child_id: str,
    alignment: WindowAlignment,
) -> tuple[FloatArray, FloatArray, Sim3]:
    parent_id = alignment.reference_id
    parent_index = previous_window_ids.index(parent_id)
    parent = estimates[parent_id]
    relative = alignment.result.transform
    child = parent.compose(relative)
    parent_jacobian, relative_jacobian = analytic_sim3_compose_jacobians(
        parent,
        relative,
    )
    previous_dimension = 7 * len(previous_window_ids)
    parent_slice = slice(7 * parent_index, 7 * (parent_index + 1))
    cross = parent_jacobian @ joint_covariance[parent_slice, :]
    relative_covariance = validated_covariance_psd(
        alignment.result.covariance,
        name=f"relative gauge covariance {parent_id}->{child_id}",
        shape=(7, 7),
        readonly=False,
    )
    child_covariance = (
        parent_jacobian
        @ joint_covariance[parent_slice, parent_slice]
        @ parent_jacobian.T
        + relative_jacobian @ relative_covariance @ relative_jacobian.T
    )
    covariance = np.empty(
        (previous_dimension + 7, previous_dimension + 7),
        dtype=np.float64,
    )
    covariance[:previous_dimension, :previous_dimension] = joint_covariance
    covariance[previous_dimension:, :previous_dimension] = cross
    covariance[:previous_dimension, previous_dimension:] = cross.T
    covariance[previous_dimension:, previous_dimension:] = child_covariance
    covariance = validated_covariance_psd(
        covariance,
        name=f"augmented gauge candidate {parent_id}->{child_id}",
        readonly=False,
    )
    mean = np.concatenate(
        [
            *(estimates[window_id].as_vector() for window_id in previous_window_ids),
            child.as_vector(),
        ]
    )
    _validate_rotation_coordinates(mean, name="causal gauge-graph candidate mean")
    return mean, covariance, child


def _symmetric_inverse(value: FloatArray, *, name: str) -> FloatArray:
    inverse = regularized_inverse_psd(
        value,
        name=name,
        eigenvalue_floor=1e-12,
    )
    return 0.5 * (inverse + inverse.T)


def _ci_objective(weights: FloatArray, information: FloatArray) -> float:
    combined = np.einsum("k,kij->ij", weights, information, optimize=True)
    combined = 0.5 * (combined + combined.T)
    covariance = _symmetric_inverse(
        combined,
        name="causal gauge-graph CI information",
    )
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("causal gauge-graph CI lost positive definiteness")
    return float(log_determinant)


def _optimize_ci_weights(
    information: FloatArray,
    *,
    minimum_weight: float,
    grid_size: int = 21,
    maximum_sweeps: int = 32,
) -> FloatArray:
    count = information.shape[0]
    if count == 1:
        return np.ones(1, dtype=np.float64)
    if not 0.0 <= minimum_weight < 1.0 / count:
        raise ValueError("minimum_edge_weight must lie in [0, 1 / edge_count)")
    if grid_size < 3 or maximum_sweeps < 1:
        raise ValueError("CI grid size and sweep count must be positive")
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    score = _ci_objective(weights, information)
    for _ in range(maximum_sweeps):
        improved = False
        for first in range(count - 1):
            for second in range(first + 1, count):
                pair_mass = float(weights[first] + weights[second])
                lower = minimum_weight / pair_mass
                upper = 1.0 - lower
                candidates = np.linspace(lower, upper, grid_size)
                candidates = candidates[
                    np.argsort(
                        np.abs(candidates - weights[first] / pair_mass),
                        kind="stable",
                    )
                ]
                best_weights = weights
                best_score = score
                for fraction in candidates:
                    candidate = weights.copy()
                    candidate[first] = pair_mass * float(fraction)
                    candidate[second] = pair_mass * (1.0 - float(fraction))
                    candidate_score = _ci_objective(candidate, information)
                    if candidate_score < best_score - 1e-12:
                        best_weights = candidate
                        best_score = candidate_score
                if best_score < score - 1e-12:
                    weights = best_weights
                    score = best_score
                    improved = True
        if not improved:
            break
    weights /= float(np.sum(weights))
    return weights


def _fuse_augmented_candidates(
    means: FloatArray,
    covariances: FloatArray,
    *,
    minimum_edge_weight: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    count = means.shape[0]
    if count == 1:
        return means[0].copy(), covariances[0].copy(), np.ones(1, dtype=np.float64)
    information = np.stack(
        [
            _symmetric_inverse(
                covariance,
                name=f"causal gauge-graph candidate covariance {index}",
            )
            for index, covariance in enumerate(covariances)
        ]
    )
    weights = _optimize_ci_weights(
        information,
        minimum_weight=minimum_edge_weight,
    )
    combined_information = np.einsum(
        "k,kij->ij",
        weights,
        information,
        optimize=True,
    )
    combined_information = 0.5 * (combined_information + combined_information.T)
    covariance = _symmetric_inverse(
        combined_information,
        name="causal gauge-graph combined information",
    )
    information_vector = np.einsum(
        "k,kij,kj->i",
        weights,
        information,
        means,
        optimize=True,
    )
    mean = covariance @ information_vector
    return mean, covariance, weights


def estimate_causal_multi_edge_gauge_graph(
    windows: Sequence[PredictionWindow],
    alignments: Sequence[WindowAlignment],
    *,
    initial_transform: Sim3,
    initial_covariance: FloatArray,
    minimum_edge_weight: float = 0.0,
) -> tuple[JointGaugePosterior, CausalGaugeGraphReport]:
    """Fuse all prefix-valid gauge edges with full-joint covariance intersection.

    Every candidate contains the same complete previously admitted joint gauge
    posterior plus one child prediction from a different overlap edge. CI is
    applied to that augmented state, so shared prior, frame, and backbone
    dependence is never treated as independence. The method is experimental and
    is not an accepted claim-bearing provider-v2 gauge model.
    """

    if not windows:
        raise ValueError("causal gauge graph requires at least one window")
    window_ids = tuple(window.window_id for window in windows)
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("causal gauge-graph window IDs must be unique")
    position = {window_id: index for index, window_id in enumerate(window_ids)}
    covariance = validated_covariance_psd(
        initial_covariance,
        name="causal gauge-graph anchor covariance",
        shape=(7, 7),
        readonly=False,
    )
    joint = covariance.copy()
    estimates: dict[str, Sim3] = {window_ids[0]: initial_transform}
    steps: list[CausalGaugeGraphStep] = []

    for child_index, child_id in enumerate(window_ids[1:], start=1):
        candidates = [
            (index, alignment)
            for index, alignment in enumerate(alignments)
            if alignment.moving_id == child_id
            and alignment.reference_id in estimates
            and position[alignment.reference_id] < child_index
        ]
        candidates.sort(
            key=lambda item: (
                position[item[1].reference_id],
                item[0],
            )
        )
        if not candidates:
            raise ValueError(
                f"window {child_id!r} has no causal overlap with an earlier window"
            )
        previous_ids = window_ids[:child_index]
        candidate_means: list[FloatArray] = []
        candidate_covariances: list[FloatArray] = []
        for _, alignment in candidates:
            mean, candidate_covariance, _ = _candidate_joint_distribution(
                previous_window_ids=previous_ids,
                estimates=estimates,
                joint_covariance=joint,
                child_id=child_id,
                alignment=alignment,
            )
            candidate_means.append(mean)
            candidate_covariances.append(candidate_covariance)
        fused_mean, fused_covariance, weights = _fuse_augmented_candidates(
            np.stack(candidate_means),
            np.stack(candidate_covariances),
            minimum_edge_weight=minimum_edge_weight,
        )
        _validate_rotation_coordinates(
            fused_mean,
            name="causal gauge-graph fused mean",
        )
        joint = validated_covariance_psd(
            fused_covariance,
            name=f"causal gauge-graph posterior through {child_id}",
            readonly=False,
        )
        for index, window_id in enumerate(window_ids[: child_index + 1]):
            estimates[window_id] = Sim3.from_vector(
                fused_mean[7 * index : 7 * (index + 1)]
            )
        steps.append(
            CausalGaugeGraphStep(
                child_window_id=child_id,
                candidate_parent_ids=tuple(
                    alignment.reference_id for _, alignment in candidates
                ),
                candidate_alignment_indices=tuple(index for index, _ in candidates),
                covariance_intersection_weights=weights,
            )
        )

    posterior = JointGaugePosterior(
        window_ids=window_ids,
        estimates=estimates,
        joint_covariance=joint,
        mode=CAUSAL_GAUGE_GRAPH_MODE,
        cross_window_covariance_preserved=True,
        parent_window_ids=tuple(None for _ in window_ids),
        selected_alignment_indices=tuple(None for _ in window_ids),
    )
    report = CausalGaugeGraphReport(window_ids=window_ids, steps=tuple(steps))
    return posterior, report


__all__ = [
    "CAUSAL_GAUGE_GRAPH_DEPENDENCE",
    "CAUSAL_GAUGE_GRAPH_MODE",
    "CausalGaugeGraphReport",
    "CausalGaugeGraphStep",
    "estimate_causal_multi_edge_gauge_graph",
]
