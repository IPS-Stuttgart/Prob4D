"""Source-only association diagnostics for tracklets from overlapping windows.

The routines in this module do not rewrite Prob4D observation identities. They
score geometrically compatible tracklets from two causally sealed windows and
admit only unambiguous mutual-best links. Cross-window material identity remains
experimental until its gates are calibrated on independent data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .causal_tracklets import CausalTrackletSet
from .sim3 import Sim3

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]

_RESIDUAL_DIMENSION = 3
_ASSOCIATION_SCHEMA_VERSION = 2


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
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
    if type(value) not in {int, float} and not isinstance(value, (np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if (strictly_positive and result <= minimum) or (not strictly_positive and result < minimum):
        relation = "greater than" if strictly_positive else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _integer_tuple(
    value: Any,
    *,
    name: str,
    nonempty: bool = False,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple of integers")
    result = tuple(_integer(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(next_value <= value for value, next_value in zip(result, result[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _ordered_unique_pairs(
    values: tuple[Any, ...],
    *,
    name: str,
) -> tuple[tuple[int, int], ...]:
    pairs = tuple((value.left_track_id, value.right_track_id) for value in values)
    if pairs != tuple(sorted(pairs)) or len(set(pairs)) != len(pairs):
        raise ValueError(f"{name} must contain sorted unique track pairs")
    return pairs


@dataclass(frozen=True)
class CrossWindowAssociationConfig:
    """Frozen source-only gates for pairwise cross-window association."""

    minimum_shared_frames: int = 2
    minimum_effective_support: float = 1.0
    isotropic_distance_scale_m: float = 0.02
    covariance_floor_m2: float = 1e-10
    maximum_weighted_rms_m: float = 0.05
    maximum_shared_frame_distance_m: float | None = 0.10
    maximum_spatial_candidate_pairs: int | None = 1_000_000
    minimum_compatibility_score: float = 0.05
    minimum_score_margin: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_shared_frames",
            _integer(
                self.minimum_shared_frames,
                name="minimum_shared_frames",
                minimum=1,
            ),
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
        if self.maximum_shared_frame_distance_m is not None:
            maximum_shared_distance = _real(
                self.maximum_shared_frame_distance_m,
                name="maximum_shared_frame_distance_m",
                minimum=0.0,
                strictly_positive=True,
            )
            if maximum_shared_distance < self.maximum_weighted_rms_m:
                raise ValueError(
                    "maximum_shared_frame_distance_m must not be smaller than "
                    "maximum_weighted_rms_m"
                )
            object.__setattr__(
                self,
                "maximum_shared_frame_distance_m",
                maximum_shared_distance,
            )
        if self.maximum_spatial_candidate_pairs is not None:
            object.__setattr__(
                self,
                "maximum_spatial_candidate_pairs",
                _integer(
                    self.maximum_spatial_candidate_pairs,
                    name="maximum_spatial_candidate_pairs",
                    minimum=1,
                ),
            )
        for name in ("minimum_compatibility_score", "minimum_score_margin"):
            object.__setattr__(
                self,
                name,
                _real(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "minimum_shared_frames": self.minimum_shared_frames,
            "minimum_effective_support": self.minimum_effective_support,
            "isotropic_distance_scale_m": self.isotropic_distance_scale_m,
            "covariance_floor_m2": self.covariance_floor_m2,
            "maximum_weighted_rms_m": self.maximum_weighted_rms_m,
            "maximum_shared_frame_distance_m": self.maximum_shared_frame_distance_m,
            "maximum_spatial_candidate_pairs": self.maximum_spatial_candidate_pairs,
            "minimum_compatibility_score": self.minimum_compatibility_score,
            "minimum_score_margin": self.minimum_score_margin,
        }


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_track_id",
            _integer(self.left_track_id, name="left_track_id"),
        )
        object.__setattr__(
            self,
            "right_track_id",
            _integer(self.right_track_id, name="right_track_id"),
        )
        object.__setattr__(
            self,
            "shared_frame_indices",
            _integer_tuple(
                self.shared_frame_indices,
                name="shared_frame_indices",
                nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "effective_support",
            _real(
                self.effective_support,
                name="effective_support",
                strictly_positive=True,
            ),
        )
        for name in ("weighted_rms_m", "maximum_distance_m", "normalized_rms"):
            object.__setattr__(
                self,
                name,
                _real(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "compatibility_score",
            _real(
                self.compatibility_score,
                name="compatibility_score",
                maximum=1.0,
            ),
        )
        if type(self.used_covariance) is not bool:
            raise ValueError("used_covariance must be a Boolean")
        tolerance = 1e-12 * max(1.0, self.maximum_distance_m)
        if self.weighted_rms_m > self.maximum_distance_m + tolerance:
            raise ValueError("weighted_rms_m must not exceed maximum_distance_m")

    def to_dict(self) -> dict[str, object]:
        return {
            "left_track_id": self.left_track_id,
            "right_track_id": self.right_track_id,
            "shared_frame_indices": list(self.shared_frame_indices),
            "effective_support": self.effective_support,
            "weighted_rms_m": self.weighted_rms_m,
            "maximum_distance_m": self.maximum_distance_m,
            "normalized_rms": self.normalized_rms,
            "compatibility_score": self.compatibility_score,
            "used_covariance": self.used_covariance,
        }


@dataclass(frozen=True)
class CrossWindowAssociationLink:
    """An admitted unambiguous mutual-best cross-window association."""

    left_track_id: int
    right_track_id: int
    shared_frame_indices: tuple[int, ...]
    compatibility_score: float
    left_score_margin: float
    right_score_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_track_id",
            _integer(self.left_track_id, name="left_track_id"),
        )
        object.__setattr__(
            self,
            "right_track_id",
            _integer(self.right_track_id, name="right_track_id"),
        )
        object.__setattr__(
            self,
            "shared_frame_indices",
            _integer_tuple(
                self.shared_frame_indices,
                name="shared_frame_indices",
                nonempty=True,
            ),
        )
        for name in (
            "compatibility_score",
            "left_score_margin",
            "right_score_margin",
        ):
            object.__setattr__(
                self,
                name,
                _real(getattr(self, name), name=name, maximum=1.0),
            )
        tolerance = 1e-15
        if (
            self.left_score_margin > self.compatibility_score + tolerance
            or self.right_score_margin > self.compatibility_score + tolerance
        ):
            raise ValueError("score margins must not exceed compatibility_score")

    def to_dict(self) -> dict[str, object]:
        return {
            "left_track_id": self.left_track_id,
            "right_track_id": self.right_track_id,
            "shared_frame_indices": list(self.shared_frame_indices),
            "compatibility_score": self.compatibility_score,
            "left_score_margin": self.left_score_margin,
            "right_score_margin": self.right_score_margin,
        }


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
    possible_track_pair_count: int
    spatial_candidate_pair_count: int
    spatially_rejected_pair_count: int
    evaluated_track_pair_count: int
    shared_gate_frame_count: int
    insufficient_shared_frame_pair_count: int
    zero_support_pair_count: int
    low_support_pair_count: int
    non_mutual_best_count: int
    ambiguous_mutual_best_count: int
    threshold_rejected_mutual_best_count: int

    def __post_init__(self) -> None:
        left_window_id = _strict_string(self.left_window_id, name="left_window_id")
        right_window_id = _strict_string(
            self.right_window_id,
            name="right_window_id",
        )
        if left_window_id == right_window_id:
            raise ValueError("association result requires distinct window IDs")
        object.__setattr__(self, "left_window_id", left_window_id)
        object.__setattr__(self, "right_window_id", right_window_id)
        object.__setattr__(
            self,
            "causal_frame_stop",
            _integer(
                self.causal_frame_stop,
                name="causal_frame_stop",
                minimum=1,
            ),
        )
        if not isinstance(self.configuration, CrossWindowAssociationConfig):
            raise TypeError("configuration must be CrossWindowAssociationConfig")
        if type(self.candidates) is not tuple or not all(
            isinstance(candidate, CrossWindowAssociationCandidate) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of association candidates")
        if type(self.links) is not tuple or not all(
            isinstance(link, CrossWindowAssociationLink) for link in self.links
        ):
            raise TypeError("links must be a tuple of association links")
        candidates = tuple(self.candidates)
        links = tuple(self.links)
        candidate_pairs = _ordered_unique_pairs(candidates, name="candidates")
        link_pairs = _ordered_unique_pairs(links, name="links")
        if len({link.left_track_id for link in links}) != len(links):
            raise ValueError("links must be one-to-one on the left")
        if len({link.right_track_id for link in links}) != len(links):
            raise ValueError("links must be one-to-one on the right")

        unmatched_left = _integer_tuple(
            self.unmatched_left_track_ids,
            name="unmatched_left_track_ids",
        )
        unmatched_right = _integer_tuple(
            self.unmatched_right_track_ids,
            name="unmatched_right_track_ids",
        )
        linked_left = {link.left_track_id for link in links}
        linked_right = {link.right_track_id for link in links}
        left_domain = tuple(sorted(linked_left | set(unmatched_left)))
        right_domain = tuple(sorted(linked_right | set(unmatched_right)))
        if not left_domain or left_domain != tuple(range(len(left_domain))):
            raise ValueError("left track IDs must form one non-empty contiguous domain")
        if not right_domain or right_domain != tuple(range(len(right_domain))):
            raise ValueError("right track IDs must form one non-empty contiguous domain")
        expected_unmatched_left = tuple(
            track_id for track_id in left_domain if track_id not in linked_left
        )
        expected_unmatched_right = tuple(
            track_id for track_id in right_domain if track_id not in linked_right
        )
        if unmatched_left != expected_unmatched_left:
            raise ValueError("unmatched_left_track_ids are inconsistent with links")
        if unmatched_right != expected_unmatched_right:
            raise ValueError("unmatched_right_track_ids are inconsistent with links")
        object.__setattr__(self, "unmatched_left_track_ids", unmatched_left)
        object.__setattr__(self, "unmatched_right_track_ids", unmatched_right)

        candidate_by_pair = dict(zip(candidate_pairs, candidates, strict=True))
        for pair, link in zip(link_pairs, links, strict=True):
            candidate = candidate_by_pair.get(pair)
            if candidate is None:
                raise ValueError("every link must reference an existing candidate")
            if link.shared_frame_indices != candidate.shared_frame_indices:
                raise ValueError("link shared frames differ from its candidate")
            if link.compatibility_score != candidate.compatibility_score:
                raise ValueError("link compatibility score differs from its candidate")
        if any(
            left_id not in left_domain or right_id not in right_domain
            for left_id, right_id in candidate_pairs
        ):
            raise ValueError("candidate track IDs lie outside the result domains")

        count_names = (
            "possible_track_pair_count",
            "spatial_candidate_pair_count",
            "spatially_rejected_pair_count",
            "evaluated_track_pair_count",
            "shared_gate_frame_count",
            "insufficient_shared_frame_pair_count",
            "zero_support_pair_count",
            "low_support_pair_count",
            "non_mutual_best_count",
            "ambiguous_mutual_best_count",
            "threshold_rejected_mutual_best_count",
        )
        for name in count_names:
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name),
            )
        expected_possible = len(left_domain) * len(right_domain)
        if self.possible_track_pair_count != expected_possible:
            raise ValueError("possible_track_pair_count is inconsistent with track domains")
        if self.spatial_candidate_pair_count > self.possible_track_pair_count:
            raise ValueError("spatial candidate count exceeds possible pair count")
        if (
            self.spatially_rejected_pair_count
            != self.possible_track_pair_count - self.spatial_candidate_pair_count
        ):
            raise ValueError("spatial rejection accounting is inconsistent")
        if self.evaluated_track_pair_count != len(candidates):
            raise ValueError("evaluated_track_pair_count must equal scored candidates")
        if (
            self.spatial_candidate_pair_count
            != len(candidates)
            + self.insufficient_shared_frame_pair_count
            + self.zero_support_pair_count
        ):
            raise ValueError("spatial candidate disposition accounting is inconsistent")
        if self.low_support_pair_count > len(candidates):
            raise ValueError("low_support_pair_count exceeds scored candidates")
        if self.spatial_candidate_pair_count and not self.shared_gate_frame_count:
            raise ValueError("spatial candidates require at least one shared gate frame")
        left_best_count = len({candidate.left_track_id for candidate in candidates})
        if (
            self.non_mutual_best_count
            + self.ambiguous_mutual_best_count
            + self.threshold_rejected_mutual_best_count
            + len(links)
            != left_best_count
        ):
            raise ValueError("mutual-best disposition accounting is inconsistent")

    @property
    def accepted_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((link.left_track_id, link.right_track_id) for link in self.links)

    def descriptor(self) -> dict[str, object]:
        """Return the complete portable result descriptor without its self-ID."""

        return {
            "schema_name": "prob4d.cross-window-tracklet-association",
            "schema_version": _ASSOCIATION_SCHEMA_VERSION,
            "left_window_id": self.left_window_id,
            "right_window_id": self.right_window_id,
            "causal_frame_stop": self.causal_frame_stop,
            "configuration": self.configuration.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "links": [link.to_dict() for link in self.links],
            "unmatched_left_track_ids": list(self.unmatched_left_track_ids),
            "unmatched_right_track_ids": list(self.unmatched_right_track_ids),
            "possible_track_pair_count": self.possible_track_pair_count,
            "spatial_candidate_pair_count": self.spatial_candidate_pair_count,
            "spatially_rejected_pair_count": self.spatially_rejected_pair_count,
            "evaluated_track_pair_count": self.evaluated_track_pair_count,
            "shared_gate_frame_count": self.shared_gate_frame_count,
            "insufficient_shared_frame_pair_count": (self.insufficient_shared_frame_pair_count),
            "zero_support_pair_count": self.zero_support_pair_count,
            "low_support_pair_count": self.low_support_pair_count,
            "non_mutual_best_count": self.non_mutual_best_count,
            "ambiguous_mutual_best_count": self.ambiguous_mutual_best_count,
            "threshold_rejected_mutual_best_count": (self.threshold_rejected_mutual_best_count),
        }

    @property
    def result_id(self) -> str:
        """Return a deterministic SHA-256 identity for the semantic result."""

        encoded = json.dumps(
            self.descriptor(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["result_id"] = self.result_id
        return result


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
    expected = (observation_count, _RESIDUAL_DIMENSION, _RESIDUAL_DIMENSION)
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


def _spatial_candidate_pairs(
    left: CausalTrackletSet,
    right: CausalTrackletSet,
    *,
    left_global_from_local: Sim3,
    right_global_from_local: Sim3,
    maximum_distance_m: float | None,
    maximum_candidate_pairs: int | None,
    chunk_size: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    shared_frames = tuple(
        sorted(
            set(int(value) for value in np.unique(left.frame_indices))
            & set(int(value) for value in np.unique(right.frame_indices))
        )
    )
    if not shared_frames:
        return (), ()

    possible_pairs = left.track_count * right.track_count
    if maximum_distance_m is None:
        if maximum_candidate_pairs is not None and possible_pairs > maximum_candidate_pairs:
            raise ValueError(
                "exhaustive spatial candidate count exceeds maximum_spatial_candidate_pairs"
            )
        return shared_frames, tuple(
            (left_track_id, right_track_id)
            for left_track_id in range(left.track_count)
            for right_track_id in range(right.track_count)
        )

    maximum_square = maximum_distance_m**2
    pairs: set[tuple[int, int]] = set()
    for frame in shared_frames:
        left_rows = np.flatnonzero(left.frame_indices == frame)
        right_rows = np.flatnonzero(right.frame_indices == frame)
        if not len(left_rows) or not len(right_rows):
            continue
        left_points = left_global_from_local.transform_points(left.points_local[left_rows])
        right_points = right_global_from_local.transform_points(right.points_local[right_rows])
        left_track_ids = left.track_ids[left_rows]
        right_track_ids = right.track_ids[right_rows]
        for left_start in range(0, len(left_rows), chunk_size):
            left_stop = min(left_start + chunk_size, len(left_rows))
            for right_start in range(0, len(right_rows), chunk_size):
                right_stop = min(right_start + chunk_size, len(right_rows))
                differences = (
                    left_points[left_start:left_stop, None, :]
                    - right_points[None, right_start:right_stop, :]
                )
                squared = np.einsum("...i,...i->...", differences, differences)
                local_left, local_right = np.nonzero(squared <= maximum_square)
                for left_offset, right_offset in zip(
                    local_left,
                    local_right,
                    strict=True,
                ):
                    pairs.add(
                        (
                            int(left_track_ids[left_start + int(left_offset)]),
                            int(right_track_ids[right_start + int(right_offset)]),
                        )
                    )
                if maximum_candidate_pairs is not None and len(pairs) > maximum_candidate_pairs:
                    raise ValueError(
                        "spatial candidate count exceeds maximum_spatial_candidate_pairs"
                    )
    return shared_frames, tuple(sorted(pairs))


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
        + config.covariance_floor_m2 * np.eye(_RESIDUAL_DIMENSION)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, config.covariance_floor_m2)
    coordinates = eigenvectors.T @ residual
    mahalanobis_square = float(np.sum(coordinates**2 / eigenvalues))
    return mahalanobis_square / _RESIDUAL_DIMENSION


def _candidate_rank(
    candidate: CrossWindowAssociationCandidate,
    *,
    side: Literal["left", "right"],
) -> tuple[float, float, float, int]:
    other_id = candidate.right_track_id if side == "left" else candidate.left_track_id
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
        track_id = candidate.left_track_id if side == "left" else candidate.right_track_id
        grouped.setdefault(track_id, []).append(candidate)
    best: dict[int, CrossWindowAssociationCandidate] = {}
    margins: dict[tuple[int, int], float] = {}
    for track_id, group in grouped.items():
        ordered = sorted(group, key=lambda item: _candidate_rank(item, side=side))
        selected = ordered[0]
        second = ordered[1].compatibility_score if len(ordered) > 1 else 0.0
        best[track_id] = selected
        margins[(selected.left_track_id, selected.right_track_id)] = max(
            0.0,
            selected.compatibility_score - second,
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
    candidate_chunk_size: int = 256,
) -> CrossWindowAssociationResult:
    """Score and conservatively admit source-only links between two windows.

    Optional covariance arrays must already be in the global frame and align with
    the flattened observations in each tracklet set. They may contain local point
    uncertainty, gauge uncertainty, or both. Supplying one side only is rejected.
    """

    if not isinstance(left, CausalTrackletSet) or not isinstance(
        right,
        CausalTrackletSet,
    ):
        raise TypeError("left and right must be CausalTrackletSet instances")
    if left.window_id == right.window_id:
        raise ValueError("cross-window association requires distinct window IDs")
    if not isinstance(left_global_from_local, Sim3) or not isinstance(
        right_global_from_local,
        Sim3,
    ):
        raise TypeError("window gauges must be Sim3 instances")
    config = CrossWindowAssociationConfig() if configuration is None else configuration
    if not isinstance(config, CrossWindowAssociationConfig):
        raise TypeError("configuration must be CrossWindowAssociationConfig")
    chunk_size = _integer(
        candidate_chunk_size,
        name="candidate_chunk_size",
        minimum=1,
    )
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
    shared_gate_frames, pair_candidates = _spatial_candidate_pairs(
        left,
        right,
        left_global_from_local=left_global_from_local,
        right_global_from_local=right_global_from_local,
        maximum_distance_m=config.maximum_shared_frame_distance_m,
        maximum_candidate_pairs=config.maximum_spatial_candidate_pairs,
        chunk_size=chunk_size,
    )

    left_frame_rows = {track_id: _frame_rows(left, rows) for track_id, rows in left_tracks.items()}
    right_frame_rows = {
        track_id: _frame_rows(right, rows) for track_id, rows in right_tracks.items()
    }
    for left_track_id, right_track_id in pair_candidates:
        left_by_frame = left_frame_rows[left_track_id]
        right_by_frame = right_frame_rows[right_track_id]
        shared_frames = tuple(sorted(left_by_frame.keys() & right_by_frame.keys()))
        if len(shared_frames) < config.minimum_shared_frames:
            insufficient_shared_frames += 1
            continue
        left_indices = np.asarray(
            [left_by_frame[frame] for frame in shared_frames],
            dtype=np.int64,
        )
        right_indices = np.asarray(
            [right_by_frame[frame] for frame in shared_frames],
            dtype=np.int64,
        )
        left_points = left_global_from_local.transform_points(left.points_local[left_indices])
        right_points = right_global_from_local.transform_points(right.points_local[right_indices])
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
                        None if left_covariance is None else left_covariance[left_index]
                    ),
                    right_covariance=(
                        None if right_covariance is None else right_covariance[right_index]
                    ),
                    config=config,
                )
                for residual, left_index, right_index in zip(
                    residuals,
                    left_indices,
                    right_indices,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        weighted_rms = float(np.sqrt(np.sum(weights * distances**2) / support))
        normalized_rms = float(np.sqrt(np.sum(weights * normalized_squares) / support))
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
        if left_margin < config.minimum_score_margin or right_margin < config.minimum_score_margin:
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
    possible_pairs = left.track_count * right.track_count
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
        possible_track_pair_count=possible_pairs,
        spatial_candidate_pair_count=len(pair_candidates),
        spatially_rejected_pair_count=possible_pairs - len(pair_candidates),
        evaluated_track_pair_count=len(candidate_tuple),
        shared_gate_frame_count=len(shared_gate_frames),
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
