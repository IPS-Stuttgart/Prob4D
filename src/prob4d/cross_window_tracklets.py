"""Source-only association diagnostics for tracklets from overlapping windows.

The routines in this module do not rewrite Prob4D observation identities. They
score geometrically compatible tracklets from two causally sealed windows and
admit only unambiguous mutual-best links. Cross-window material identity remains
experimental until its gates are calibrated on independent data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .causal_tracklets import CausalTrackletSet
from .sim3 import Sim3

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(
    value: Any,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if (strictly_positive and result <= minimum) or (
        not strictly_positive and result < minimum
    ):
        relation = "greater than" if strictly_positive else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


@dataclass(frozen=True)
class CrossWindowAssociationConfig:
    """Frozen source-only gates for pairwise cross-window association."""

    minimum_shared_frames: int = 2
    minimum_effective_support: float = 1.0
    isotropic_distance_scale_m: float = 0.02
    covariance_floor_m2: float = 1e-10
    maximum_weighted_rms_m: float = 0.05
    minimum_compatibility_score: float = 0.05
    minimum_score_margin: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_shared_frames",
            _integer(self.minimum_shared_frames, name="minimum_shared_frames", minimum=1),
        )
        for name in (
            "minimum_effective_support",
            "isotropic_distance_scale_m",
            "covariance_floor_m2",
            "maximum_weighted_rms_m",
        ):
            object.__setattr__(
                self,
                name,
                _real(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    strictly_positive=True,
                ),
            )
        for name in ("minimum_compatibility_score", "minimum_score_margin"):
            object.__setattr__(
                self,
                name,
                _real(getattr(self, name), name=name, minimum=0.0, maximum=1.0),
            )


@dataclass(frozen=True)
class CrossWindowAssociationCandidate:
    """One source-only compatibility score between two window-local tracks."""

    left_track_id: int
    right_track_id: int
    shared_frame_indices: tuple[int, ...]
    effective_support: float
    weighted_rms_m: float
    maximum_distance_m: float
    normalized_rms: float
    compatibility_score: float
    used_covariance: bool


@dataclass(frozen=True)
class CrossWindowAssociationLink:
    """An admitted unambiguous mutual-best cross-window association."""

    left_track_id: int
    right_track_id: int
    shared_frame_indices: tuple[int, ...]
    compatibility_score: float
    left_score_margin: float
    right_score_margin: float


@dataclass(frozen=True)
class CrossWindowAssociationResult:
    """Candidates, admitted links, unmatched tracks, and rejection accounting."""

    left_window_id: str
    right_window_id: str
    causal_frame_stop: int
    configuration: CrossWindowAssociationConfig
    candidates: tuple[CrossWindowAssociationCandidate, ...]
    links: tuple[CrossWindowAssociationLink, ...]
    unmatched_left_track_ids: tuple[int, ...]
    unmatched_right_track_ids: tuple[int, ...]
    evaluated_track_pair_count: int
    insufficient_shared_frame_pair_count: int
    zero_support_pair_count: int
    low_support_pair_count: int
    non_mutual_best_count: int
    ambiguous_mutual_best_count: int
    threshold_rejected_mutual_best_count: int

    @property
    def accepted_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((link.left_track_id, link.right_track_id) for link in self.links)


def _global_covariances(
    value: FloatArray,
    *,
    observation_count: int,
    name: str,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    covariance = np.asarray(raw, dtype=np.float64).copy()
    expected = (observation_count, 3, 3)
    if covariance.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
    if np.any(np.min(eigenvalues, axis=1) < -1e-10 * scale):
        raise ValueError(f"{name} must be positive semidefinite")
    symmetric.setflags(write=False)
    return symmetric


def _track_rows(tracklets: CausalTrackletSet) -> dict[int, IntArray]:
    return {
        track_id: np.flatnonzero(tracklets.track_ids == track_id)
        for track_id in range(tracklets.track_count)
    }


def _frame_rows(tracklets: CausalTrackletSet, rows: IntArray) -> dict[int, int]:
    return {int(tracklets.frame_indices[row]): int(row) for row in rows}


def _normalized_square(
    residual: FloatArray,
    *,
    left_covariance: FloatArray | None,
    right_covariance: FloatArray | None,
    config: CrossWindowAssociationConfig,
) -> float:
    if left_covariance is None or right_covariance is None:
        return float(residual @ residual / config.isotropic_distance_scale_m**2)
    covariance = (
        left_covariance
        + right_covariance
        + config.covariance_floor_m2 * np.eye(3)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, config.covariance_floor_m2)
    coordinates = eigenvectors.T @ residual
    return float(np.sum(coordinates**2 / eigenvalues) / 3.0)


def _candidate_rank(
    candidate: CrossWindowAssociationCandidate,
    *,
    side: Literal["left", "right"],
) -> tuple[float, float, float, int]:
    other_id = (
        candidate.right_track_id if side == "left" else candidate.left_track_id
    )
    return (
        -candidate.compatibility_score,
        candidate.weighted_rms_m,
        -candidate.effective_support,
        other_id,
    )


def _best_by_side(
    candidates: tuple[CrossWindowAssociationCandidate, ...],
    *,
    side: Literal["left", "right"],
) -> tuple[
    dict[int, CrossWindowAssociationCandidate],
    dict[tuple[int, int], float],
]:
    grouped: dict[int, list[CrossWindowAssociationCandidate]] = {}
    for candidate in candidates:
        track_id = (
            candidate.left_track_id if side == "left" else candidate.right_track_id
        )
        grouped.setdefault(track_id, []).append(candidate)
    best: dict[int, CrossWindowAssociationCandidate] = {}
    margins: dict[tuple[int, int], float] = {}
    for track_id, group in grouped.items():
        ordered = sorted(group, key=lambda item: _candidate_rank(item, side=side))
        selected = ordered[0]
        second = ordered[1].compatibility_score if len(ordered) > 1 else 0.0
        best[track_id] = selected
        margins[(selected.left_track_id, selected.right_track_id)] = max(
            0.0, selected.compatibility_score - second
        )
    return best, margins


def associate_cross_window_tracklets(
    left: CausalTrackletSet,
    right: CausalTrackletSet,
    *,
    left_global_from_local: Sim3,
    right_global_from_local: Sim3,
    configuration: CrossWindowAssociationConfig | None = None,
    left_global_covariance_m2: FloatArray | None = None,
    right_global_covariance_m2: FloatArray | None = None,
) -> CrossWindowAssociationResult:
    """Score and conservatively admit source-only links between two windows.

    Optional covariance arrays must already be in the global frame and align with
    the flattened observations in each tracklet set. They may contain local point
    uncertainty, gauge uncertainty, or both. Supplying one side only is rejected.
    """

    if not isinstance(left, CausalTrackletSet) or not isinstance(
        right, CausalTrackletSet
    ):
        raise TypeError("left and right must be CausalTrackletSet instances")
    if left.window_id == right.window_id:
        raise ValueError("cross-window association requires distinct window IDs")
    if not isinstance(left_global_from_local, Sim3) or not isinstance(
        right_global_from_local, Sim3
    ):
        raise TypeError("window gauges must be Sim3 instances")
    config = configuration or CrossWindowAssociationConfig()
    if not isinstance(config, CrossWindowAssociationConfig):
        raise TypeError("configuration must be CrossWindowAssociationConfig")
    if (left_global_covariance_m2 is None) != (right_global_covariance_m2 is None):
        raise ValueError("global covariance must be supplied for both windows or neither")
    left_covariance = (
        None
        if left_global_covariance_m2 is None
        else _global_covariances(
            left_global_covariance_m2,
            observation_count=left.observation_count,
            name="left_global_covariance_m2",
        )
    )
    right_covariance = (
        None
        if right_global_covariance_m2 is None
        else _global_covariances(
            right_global_covariance_m2,
            observation_count=right.observation_count,
            name="right_global_covariance_m2",
        )
    )
    used_covariance = left_covariance is not None

    candidates: list[CrossWindowAssociationCandidate] = []
    insufficient_shared_frames = 0
    zero_support_pairs = 0
    low_support_pairs = 0
    left_tracks = _track_rows(left)
    right_tracks = _track_rows(right)

    for left_track_id, left_rows in left_tracks.items():
        left_by_frame = _frame_rows(left, left_rows)
        for right_track_id, right_rows in right_tracks.items():
            right_by_frame = _frame_rows(right, right_rows)
            shared_frames = tuple(sorted(left_by_frame.keys() & right_by_frame.keys()))
            if len(shared_frames) < config.minimum_shared_frames:
                insufficient_shared_frames += 1
                continue
            left_indices = np.asarray(
                [left_by_frame[frame] for frame in shared_frames], dtype=np.int64
            )
            right_indices = np.asarray(
                [right_by_frame[frame] for frame in shared_frames], dtype=np.int64
            )
            left_points = left_global_from_local.transform_points(
                left.points_local[left_indices]
            )
            right_points = right_global_from_local.transform_points(
                right.points_local[right_indices]
            )
            residuals = left_points - right_points
            distances = np.linalg.norm(residuals, axis=1)
            weights = (
                left.association_probability[left_indices]
                * right.association_probability[right_indices]
            )
            support = float(np.sum(weights))
            if support <= 0.0:
                zero_support_pairs += 1
                continue
            normalized_squares = np.asarray(
                [
                    _normalized_square(
                        residual,
                        left_covariance=(
                            None
                            if left_covariance is None
                            else left_covariance[left_index]
                        ),
                        right_covariance=(
                            None
                            if right_covariance is None
                            else right_covariance[right_index]
                        ),
                        config=config,
                    )
                    for residual, left_index, right_index in zip(
                        residuals, left_indices, right_indices, strict=True
                    )
                ],
                dtype=np.float64,
            )
            weighted_rms = float(np.sqrt(np.sum(weights * distances**2) / support))
            normalized_rms = float(
                np.sqrt(np.sum(weights * normalized_squares) / support)
            )
            support_fraction = min(1.0, support / config.minimum_effective_support)
            score = float(support_fraction * np.exp(-0.5 * normalized_rms**2))
            if support < config.minimum_effective_support:
                low_support_pairs += 1
            candidates.append(
                CrossWindowAssociationCandidate(
                    left_track_id=left_track_id,
                    right_track_id=right_track_id,
                    shared_frame_indices=shared_frames,
                    effective_support=support,
                    weighted_rms_m=weighted_rms,
                    maximum_distance_m=float(np.max(distances)),
                    normalized_rms=normalized_rms,
                    compatibility_score=score,
                    used_covariance=used_covariance,
                )
            )

    candidate_tuple = tuple(
        sorted(candidates, key=lambda item: (item.left_track_id, item.right_track_id))
    )
    left_best, left_margins = _best_by_side(candidate_tuple, side="left")
    right_best, right_margins = _best_by_side(candidate_tuple, side="right")
    links: list[CrossWindowAssociationLink] = []
    non_mutual = 0
    ambiguous = 0
    threshold_rejected = 0

    for left_track_id, candidate in sorted(left_best.items()):
        right_candidate = right_best.get(candidate.right_track_id)
        if right_candidate is None or right_candidate.left_track_id != left_track_id:
            non_mutual += 1
            continue
        pair = (candidate.left_track_id, candidate.right_track_id)
        left_margin = left_margins[pair]
        right_margin = right_margins[pair]
        if (
            left_margin < config.minimum_score_margin
            or right_margin < config.minimum_score_margin
        ):
            ambiguous += 1
            continue
        if (
            candidate.effective_support < config.minimum_effective_support
            or candidate.weighted_rms_m > config.maximum_weighted_rms_m
            or candidate.compatibility_score < config.minimum_compatibility_score
        ):
            threshold_rejected += 1
            continue
        links.append(
            CrossWindowAssociationLink(
                left_track_id=candidate.left_track_id,
                right_track_id=candidate.right_track_id,
                shared_frame_indices=candidate.shared_frame_indices,
                compatibility_score=candidate.compatibility_score,
                left_score_margin=left_margin,
                right_score_margin=right_margin,
            )
        )

    link_tuple = tuple(links)
    linked_left = {link.left_track_id for link in link_tuple}
    linked_right = {link.right_track_id for link in link_tuple}
    return CrossWindowAssociationResult(
        left_window_id=left.window_id,
        right_window_id=right.window_id,
        causal_frame_stop=min(left.causal_frame_stop, right.causal_frame_stop),
        configuration=config,
        candidates=candidate_tuple,
        links=link_tuple,
        unmatched_left_track_ids=tuple(
            track_id for track_id in left_tracks if track_id not in linked_left
        ),
        unmatched_right_track_ids=tuple(
            track_id for track_id in right_tracks if track_id not in linked_right
        ),
        evaluated_track_pair_count=left.track_count * right.track_count,
        insufficient_shared_frame_pair_count=insufficient_shared_frames,
        zero_support_pair_count=zero_support_pairs,
        low_support_pair_count=low_support_pairs,
        non_mutual_best_count=non_mutual,
        ambiguous_mutual_best_count=ambiguous,
        threshold_rejected_mutual_best_count=threshold_rejected,
    )


__all__ = [
    "CrossWindowAssociationCandidate",
    "CrossWindowAssociationConfig",
    "CrossWindowAssociationLink",
    "CrossWindowAssociationResult",
    "associate_cross_window_tracklets",
]
