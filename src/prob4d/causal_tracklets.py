"""Causal scene-flow tracklets for persistent unfused observation identities.

The builder uses only prediction rows strictly before an exclusive causal cutoff.
It seeds a deterministic sparse grid in the first retained frame, predicts each
point with the window's local scene flow, and associates it to a nearby valid
point in the next retained frame. Association confidence remains separate from
source-side prior reliability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping
from .data import PredictionWindow
from .observation_factors import ObservationFactor
from .uncertainty import StructuredCovariance

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CausalTrackletSet:
    """Flattened persistent observations from one local prediction window."""

    window_id: str
    causal_frame_stop: int
    source_shape: tuple[int, int, int]
    seed_frame_index: int
    track_ids: IntArray
    frame_indices: IntArray
    local_frame_indices: IntArray
    rows: IntArray
    columns: IntArray
    points_local: FloatArray
    link_probability: FloatArray
    association_probability: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        if not window_id:
            raise ValueError("window_id must not be empty")
        causal_frame_stop = int(self.causal_frame_stop)
        seed_frame_index = int(self.seed_frame_index)
        raw_source_shape = tuple(self.source_shape)
        source_shape = tuple(int(value) for value in raw_source_shape)
        if (
            len(source_shape) != 3
            or any(value < 1 for value in source_shape)
            or any(
                normalized != original
                for normalized, original in zip(
                    source_shape,
                    raw_source_shape,
                    strict=True,
                )
            )
        ):
            raise ValueError("source_shape must contain positive integer (T, H, W)")
        if causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be positive")
        if seed_frame_index < 0 or seed_frame_index >= causal_frame_stop:
            raise ValueError("seed_frame_index must precede causal_frame_stop")

        track_ids = np.asarray(self.track_ids, dtype=np.int64)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        local_indices = np.asarray(self.local_frame_indices, dtype=np.int64)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        points = np.asarray(self.points_local, dtype=np.float64)
        link = np.asarray(self.link_probability, dtype=np.float64)
        association = np.asarray(self.association_probability, dtype=np.float64)
        count = len(track_ids)
        vectors = (frame_indices, local_indices, rows, columns, link, association)
        if count == 0 or any(value.shape != (count,) for value in vectors):
            raise ValueError("tracklet row arrays must share one non-empty length")
        if points.shape != (count, 3):
            raise ValueError("points_local must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("tracklet points must be finite")
        if np.any(track_ids < 0) or np.any(frame_indices < 0):
            raise ValueError("track and frame indices must be non-negative")
        if np.any(frame_indices >= causal_frame_stop):
            raise ValueError("tracklet observations cross the causal frame stop")
        if np.any(local_indices < 0) or np.any(local_indices >= source_shape[0]):
            raise ValueError("local frame indices lie outside source_shape")
        if np.any(rows < 0) or np.any(rows >= source_shape[1]):
            raise ValueError("tracklet rows lie outside source_shape")
        if np.any(columns < 0) or np.any(columns >= source_shape[2]):
            raise ValueError("tracklet columns lie outside source_shape")
        for name, value in (
            ("link_probability", link),
            ("association_probability", association),
        ):
            if not np.all(np.isfinite(value)) or np.any(
                (value < 0.0) | (value > 1.0)
            ):
                raise ValueError(f"{name} must lie in [0, 1]")

        unique_tracks = np.unique(track_ids)
        expected_tracks = np.arange(len(unique_tracks), dtype=np.int64)
        if not np.array_equal(unique_tracks, expected_tracks):
            raise ValueError("track_ids must be contiguous from zero")
        order = np.lexsort((frame_indices, track_ids))
        if not np.array_equal(order, np.arange(count)):
            raise ValueError("tracklet rows must be ordered by track and frame")
        pairs = np.column_stack((track_ids, frame_indices))
        if len(np.unique(pairs, axis=0)) != count:
            raise ValueError("a track may contain at most one row per frame")

        for track_id in unique_tracks:
            selected = np.flatnonzero(track_ids == track_id)
            track_frames = frame_indices[selected]
            track_local = local_indices[selected]
            if track_frames[0] != seed_frame_index:
                raise ValueError("every retained track must start at seed_frame_index")
            if np.any(np.diff(track_frames) <= 0):
                raise ValueError("frames must increase strictly within each track")
            if np.any(np.diff(track_local) <= 0):
                raise ValueError(
                    "local frame indices must increase strictly within each track"
                )
            if not np.isclose(link[selected[0]], 1.0):
                raise ValueError("the first link probability must equal one")
            if not np.isclose(association[selected[0]], 1.0):
                raise ValueError("the first association probability must equal one")
            expected = np.cumprod(link[selected])
            if not np.allclose(
                association[selected],
                expected,
                atol=1e-15,
                rtol=1e-12,
            ):
                raise ValueError(
                    "association_probability must equal cumulative link probability"
                )

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(self, "source_shape", source_shape)
        object.__setattr__(self, "seed_frame_index", seed_frame_index)
        object.__setattr__(self, "track_ids", _readonly(track_ids, dtype=np.int64))
        object.__setattr__(
            self, "frame_indices", _readonly(frame_indices, dtype=np.int64)
        )
        object.__setattr__(
            self,
            "local_frame_indices",
            _readonly(local_indices, dtype=np.int64),
        )
        object.__setattr__(self, "rows", _readonly(rows, dtype=np.int64))
        object.__setattr__(self, "columns", _readonly(columns, dtype=np.int64))
        object.__setattr__(
            self, "points_local", _readonly(points, dtype=np.float64)
        )
        object.__setattr__(
            self, "link_probability", _readonly(link, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "association_probability",
            _readonly(association, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="tracklet metadata"),
        )

    @property
    def track_count(self) -> int:
        return int(np.max(self.track_ids)) + 1

    @property
    def observation_count(self) -> int:
        return int(len(self.track_ids))

    @property
    def frame_count(self) -> int:
        return int(len(np.unique(self.frame_indices)))

    @property
    def track_lengths(self) -> IntArray:
        return np.bincount(self.track_ids, minlength=self.track_count)

    def summary(self) -> dict[str, object]:
        lengths = self.track_lengths
        return {
            "window_id": self.window_id,
            "causal_frame_stop": self.causal_frame_stop,
            "seed_frame_index": self.seed_frame_index,
            "track_count": self.track_count,
            "observation_count": self.observation_count,
            "frame_count": self.frame_count,
            "minimum_track_length": int(np.min(lengths)),
            "median_track_length": float(np.median(lengths)),
            "maximum_track_length": int(np.max(lengths)),
        }


@dataclass(frozen=True)
class CausalTrackletReport:
    """Termination and retention audit for one scene-flow tracklet build."""

    seed_count: int
    retained_track_count: int
    observation_count: int
    dropped_short_tracks: int
    terminated_invalid_source: int
    terminated_no_candidate: int
    terminated_step_error: int
    terminated_low_probability: int
    collision_rejections: int
    seed_stride: int
    search_radius_pixels: int
    maximum_step_error_local: float
    association_sigma_local: float
    minimum_link_probability: float
    minimum_track_length: int

    def __post_init__(self) -> None:
        integer_fields = (
            "seed_count",
            "retained_track_count",
            "observation_count",
            "dropped_short_tracks",
            "terminated_invalid_source",
            "terminated_no_candidate",
            "terminated_step_error",
            "terminated_low_probability",
            "collision_rejections",
            "seed_stride",
            "search_radius_pixels",
            "minimum_track_length",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        if self.seed_stride < 1:
            raise ValueError("seed_stride must be positive")
        if self.minimum_track_length < 1:
            raise ValueError("minimum_track_length must be positive")
        positive = (
            self.maximum_step_error_local,
            self.association_sigma_local,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("tracklet distance scales must be finite and positive")
        if not np.isfinite(self.minimum_link_probability) or not (
            0.0 < self.minimum_link_probability <= 1.0
        ):
            raise ValueError("minimum_link_probability must lie in (0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class _TrackObservation:
    local_index: int
    frame_index: int
    row: int
    column: int
    point: np.ndarray
    link_probability: float
    association_probability: float


@dataclass(frozen=True)
class _Proposal:
    track_id: int
    row: int
    column: int
    error: float
    link_probability: float
    association_probability: float


def _link_probability(
    best_error: float,
    second_error: float | None,
    *,
    sigma: float,
) -> float:
    error_score = float(np.exp(-0.5 * (best_error / sigma) ** 2))
    if second_error is None:
        uniqueness_score = 1.0
    else:
        margin = max(second_error - best_error, 0.0)
        uniqueness_score = float(
            1.0 - np.exp(-0.5 * (margin / sigma) ** 2)
        )
    return float(np.clip(error_score * uniqueness_score, 0.0, 1.0))


def build_causal_scene_flow_tracklets(
    window: PredictionWindow,
    *,
    causal_frame_stop: int,
    seed_stride: int = 8,
    search_radius_pixels: int = 4,
    maximum_step_error_local: float = 0.05,
    association_sigma_local: float | None = None,
    minimum_link_probability: float = 0.05,
    minimum_track_length: int = 2,
) -> tuple[CausalTrackletSet, CausalTrackletReport]:
    """Build deterministic prefix-only tracklets from a window's 3-D scene flow."""

    scene_flow = window.scene_flow
    deform_mask = window.deform_mask
    if scene_flow is None or deform_mask is None:
        raise ValueError("scene-flow tracklets require scene_flow and deform_mask")
    causal_frame_stop = int(causal_frame_stop)
    seed_stride = int(seed_stride)
    search_radius_pixels = int(search_radius_pixels)
    minimum_track_length = int(minimum_track_length)
    if causal_frame_stop < 1:
        raise ValueError("causal_frame_stop must be positive")
    if seed_stride < 1:
        raise ValueError("seed_stride must be positive")
    if search_radius_pixels < 0:
        raise ValueError("search_radius_pixels must be non-negative")
    if not np.isfinite(maximum_step_error_local) or (
        maximum_step_error_local <= 0.0
    ):
        raise ValueError("maximum_step_error_local must be finite and positive")
    sigma = (
        maximum_step_error_local / 3.0
        if association_sigma_local is None
        else float(association_sigma_local)
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("association_sigma_local must be finite and positive")
    if not np.isfinite(minimum_link_probability) or not (
        0.0 < minimum_link_probability <= 1.0
    ):
        raise ValueError("minimum_link_probability must lie in (0, 1]")
    if minimum_track_length < 1:
        raise ValueError("minimum_track_length must be positive")

    eligible = np.flatnonzero(window.frame_indices < causal_frame_stop)
    if not len(eligible):
        raise ValueError("window has no frame before causal_frame_stop")
    first_local = int(eligible[0])
    seed_frame = int(window.frame_indices[first_local])
    height, width = window.shape[1:]
    seed_mask = window.valid_mask[first_local].copy()
    seed_grid = np.zeros((height, width), dtype=bool)
    seed_grid[::seed_stride, ::seed_stride] = True
    seed_rows, seed_columns = np.nonzero(seed_mask & seed_grid)
    if not len(seed_rows):
        raise ValueError("the first retained frame has no valid tracklet seed")

    tracks: dict[int, list[_TrackObservation]] = {}
    active: dict[int, tuple[int, int, float]] = {}
    for track_id, (row, column) in enumerate(
        zip(seed_rows, seed_columns, strict=True)
    ):
        point = window.point_map[first_local, row, column].copy()
        tracks[track_id] = [
            _TrackObservation(
                local_index=first_local,
                frame_index=seed_frame,
                row=int(row),
                column=int(column),
                point=point,
                link_probability=1.0,
                association_probability=1.0,
            )
        ]
        active[track_id] = (int(row), int(column), 1.0)

    terminated_invalid_source = 0
    terminated_no_candidate = 0
    terminated_step_error = 0
    terminated_low_probability = 0
    collision_rejections = 0

    for current_local, next_local in zip(
        eligible[:-1], eligible[1:], strict=True
    ):
        current_local = int(current_local)
        next_local = int(next_local)
        proposals: list[_Proposal] = []
        for track_id, (row, column, cumulative) in active.items():
            if not (
                window.valid_mask[current_local, row, column]
                and deform_mask[current_local, row, column]
            ):
                terminated_invalid_source += 1
                continue

            predicted = (
                window.point_map[current_local, row, column]
                + scene_flow[current_local, row, column]
            )
            row_start = max(0, row - search_radius_pixels)
            row_stop = min(height, row + search_radius_pixels + 1)
            column_start = max(0, column - search_radius_pixels)
            column_stop = min(width, column + search_radius_pixels + 1)
            local_mask = window.valid_mask[
                next_local,
                row_start:row_stop,
                column_start:column_stop,
            ]
            candidate_rows, candidate_columns = np.nonzero(local_mask)
            if not len(candidate_rows):
                terminated_no_candidate += 1
                continue
            candidate_rows = candidate_rows + row_start
            candidate_columns = candidate_columns + column_start
            candidate_points = window.point_map[
                next_local, candidate_rows, candidate_columns
            ]
            errors = np.linalg.norm(candidate_points - predicted, axis=1)
            order = np.argsort(errors, kind="stable")
            best_index = int(order[0])
            best_error = float(errors[best_index])
            if best_error > maximum_step_error_local:
                terminated_step_error += 1
                continue
            second_error = (
                float(errors[int(order[1])]) if len(order) > 1 else None
            )
            link = _link_probability(best_error, second_error, sigma=sigma)
            if link < minimum_link_probability:
                terminated_low_probability += 1
                continue
            proposals.append(
                _Proposal(
                    track_id=track_id,
                    row=int(candidate_rows[best_index]),
                    column=int(candidate_columns[best_index]),
                    error=best_error,
                    link_probability=link,
                    association_probability=cumulative * link,
                )
            )

        winners: dict[tuple[int, int], _Proposal] = {}
        for proposal in proposals:
            key = (proposal.row, proposal.column)
            previous = winners.get(key)
            rank = (
                -proposal.link_probability,
                proposal.error,
                proposal.track_id,
            )
            if previous is None:
                winners[key] = proposal
                continue
            previous_rank = (
                -previous.link_probability,
                previous.error,
                previous.track_id,
            )
            collision_rejections += 1
            if rank < previous_rank:
                winners[key] = proposal

        next_active: dict[int, tuple[int, int, float]] = {}
        next_frame = int(window.frame_indices[next_local])
        for proposal in sorted(winners.values(), key=lambda item: item.track_id):
            point = window.point_map[
                next_local, proposal.row, proposal.column
            ].copy()
            tracks[proposal.track_id].append(
                _TrackObservation(
                    local_index=next_local,
                    frame_index=next_frame,
                    row=proposal.row,
                    column=proposal.column,
                    point=point,
                    link_probability=proposal.link_probability,
                    association_probability=proposal.association_probability,
                )
            )
            next_active[proposal.track_id] = (
                proposal.row,
                proposal.column,
                proposal.association_probability,
            )
        active = next_active
        if not active:
            break

    retained_ids = [
        track_id
        for track_id, observations in sorted(tracks.items())
        if len(observations) >= minimum_track_length
    ]
    if not retained_ids:
        raise ValueError("no tracklet satisfies minimum_track_length")
    id_map = {
        original: replacement
        for replacement, original in enumerate(retained_ids)
    }
    flattened: list[tuple[int, _TrackObservation]] = []
    for original in retained_ids:
        flattened.extend(
            (id_map[original], observation) for observation in tracks[original]
        )

    result = CausalTrackletSet(
        window_id=window.window_id,
        causal_frame_stop=causal_frame_stop,
        source_shape=window.shape,
        seed_frame_index=seed_frame,
        track_ids=np.asarray(
            [track_id for track_id, _ in flattened], dtype=np.int64
        ),
        frame_indices=np.asarray(
            [observation.frame_index for _, observation in flattened],
            dtype=np.int64,
        ),
        local_frame_indices=np.asarray(
            [observation.local_index for _, observation in flattened],
            dtype=np.int64,
        ),
        rows=np.asarray(
            [observation.row for _, observation in flattened], dtype=np.int64
        ),
        columns=np.asarray(
            [observation.column for _, observation in flattened],
            dtype=np.int64,
        ),
        points_local=np.asarray(
            [observation.point for _, observation in flattened],
            dtype=np.float64,
        ),
        link_probability=np.asarray(
            [observation.link_probability for _, observation in flattened],
            dtype=np.float64,
        ),
        association_probability=np.asarray(
            [
                observation.association_probability
                for _, observation in flattened
            ],
            dtype=np.float64,
        ),
        metadata={
            "method": "local-scene-flow-neighborhood-association-v1",
            "source_window_id": window.window_id,
            "source_frame_indices": [
                int(window.frame_indices[index]) for index in eligible
            ],
            "seed_stride": seed_stride,
            "search_radius_pixels": search_radius_pixels,
            "maximum_step_error_local": float(maximum_step_error_local),
            "association_sigma_local": sigma,
            "minimum_link_probability": float(minimum_link_probability),
            "minimum_track_length": minimum_track_length,
            "association_semantics": (
                "cumulative product of deterministic local link probabilities"
            ),
        },
    )
    report = CausalTrackletReport(
        seed_count=len(seed_rows),
        retained_track_count=result.track_count,
        observation_count=result.observation_count,
        dropped_short_tracks=len(tracks) - len(retained_ids),
        terminated_invalid_source=terminated_invalid_source,
        terminated_no_candidate=terminated_no_candidate,
        terminated_step_error=terminated_step_error,
        terminated_low_probability=terminated_low_probability,
        collision_rejections=collision_rejections,
        seed_stride=seed_stride,
        search_radius_pixels=search_radius_pixels,
        maximum_step_error_local=float(maximum_step_error_local),
        association_sigma_local=sigma,
        minimum_link_probability=float(minimum_link_probability),
        minimum_track_length=minimum_track_length,
    )
    return result, report


def tracklets_to_observation_factors(
    tracklets: CausalTrackletSet,
    covariance: StructuredCovariance | FloatArray,
    *,
    view_id: str,
    prior_reliability: FloatArray | None = None,
    prior_nominal_probability: float = 1.0,
    effective_samples_per_group: float = 64.0,
    correlation_group_prefix: str = "scene-flow-tracklets",
    factor_id_prefix: str | None = None,
) -> tuple[ObservationFactor, ...]:
    """Convert persistent tracklet rows into one unfused factor per frame."""

    view_id = str(view_id)
    correlation_group_prefix = str(correlation_group_prefix)
    if not view_id or not correlation_group_prefix:
        raise ValueError("view_id and correlation_group_prefix must be non-empty")
    if not np.isfinite(effective_samples_per_group) or (
        effective_samples_per_group <= 0.0
    ):
        raise ValueError("effective_samples_per_group must be positive")

    rays: np.ndarray | None
    if isinstance(covariance, StructuredCovariance):
        if covariance.parallel_variance.shape != tracklets.source_shape:
            raise ValueError("structured covariance shape differs from tracklet source")
        covariance_grid = covariance.matrices()
        rays = covariance.ray_directions
    else:
        covariance_grid = np.asarray(covariance, dtype=np.float64)
        expected = tracklets.source_shape + (3, 3)
        if covariance_grid.shape != expected:
            raise ValueError(f"covariance must have shape {expected}")
        rays = None
    if not np.all(np.isfinite(covariance_grid)):
        raise ValueError("tracklet covariance must be finite")

    if prior_reliability is None:
        reliability_grid = np.ones(tracklets.source_shape, dtype=np.float64)
    else:
        reliability_grid = np.asarray(prior_reliability, dtype=np.float64)
        if reliability_grid.shape != tracklets.source_shape:
            raise ValueError("prior_reliability must match tracklet source_shape")
        if not np.all(np.isfinite(reliability_grid)) or np.any(
            (reliability_grid < 0.0) | (reliability_grid > 1.0)
        ):
            raise ValueError("prior_reliability must lie in [0, 1]")

    prefix = (
        f"{tracklets.window_id}:{view_id}:tracklets"
        if factor_id_prefix is None
        else str(factor_id_prefix)
    )
    if not prefix:
        raise ValueError("factor_id_prefix must be non-empty")

    factors: list[ObservationFactor] = []
    for frame in np.unique(tracklets.frame_indices):
        selected = np.flatnonzero(tracklets.frame_indices == frame)
        local_indices = np.unique(tracklets.local_frame_indices[selected])
        if len(local_indices) != 1:
            raise ValueError("one absolute frame maps to multiple local frame indices")
        local_index = int(local_indices[0])
        rows = tracklets.rows[selected]
        columns = tracklets.columns[selected]
        count = len(selected)
        composite_weight = min(
            1.0, float(effective_samples_per_group) / count
        )
        factor = ObservationFactor(
            factor_id=f"{prefix}:frame-{int(frame)}",
            frame_index=int(frame),
            view_id=view_id,
            window_id=tracklets.window_id,
            gauge_id=tracklets.window_id,
            point_ids=tracklets.track_ids[selected],
            points_local_m=tracklets.points_local[selected],
            valid_mask=np.ones(count, dtype=bool),
            local_covariance_m2=covariance_grid[local_index, rows, columns],
            association_probability=tracklets.association_probability[selected],
            prior_reliability=reliability_grid[local_index, rows, columns],
            prior_nominal_probability=prior_nominal_probability,
            composite_weight=composite_weight,
            correlation_group_id=(
                f"{correlation_group_prefix}:frame-{int(frame)}"
            ),
            causal_frame_stop=tracklets.causal_frame_stop,
            ray_directions_local=(
                None if rays is None else rays[local_index, rows, columns]
            ),
        )
        factors.append(factor)
    return tuple(factors)


__all__ = [
    "CausalTrackletReport",
    "CausalTrackletSet",
    "build_causal_scene_flow_tracklets",
    "tracklets_to_observation_factors",
]
