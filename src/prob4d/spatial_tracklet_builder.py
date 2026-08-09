"""Spatially stratified wrapper for causal scene-flow tracklets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .causal_tracklets import (
    CausalTrackletReport,
    CausalTrackletSet,
    TargetDeformMaskPolicy,
    build_causal_scene_flow_tracklets,
)
from .data import PredictionWindow
from .spatial_seed_selection import (
    SPATIAL_TRACKLET_CLAIM_BOUNDARY,
    SeedSelectionPolicy,
    SpatialSeedSelection,
    select_spatial_tracklet_seeds,
)

IntArray: TypeAlias = NDArray[np.integer[Any]]


def _strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class SpatialTrackletReport:
    """Base association audit plus selected and retained spatial support."""

    base_report: CausalTrackletReport
    seed_selection: SpatialSeedSelection
    retained_seed_cell_ids: tuple[int, ...]
    claim_boundary: str = SPATIAL_TRACKLET_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not isinstance(self.base_report, CausalTrackletReport):
            raise TypeError("base_report must be a CausalTrackletReport")
        if not isinstance(self.seed_selection, SpatialSeedSelection):
            raise TypeError("seed_selection must be a SpatialSeedSelection")
        if type(self.retained_seed_cell_ids) is not tuple:
            raise ValueError("retained_seed_cell_ids must be a tuple")
        retained = tuple(
            _strict_integer(value, name="retained_seed_cell_id")
            for value in self.retained_seed_cell_ids
        )
        if len(retained) != self.base_report.retained_track_count:
            raise ValueError("retained_seed_cell_ids must contain one ID per retained track")
        if not set(retained).issubset(set(self.seed_selection.cell_ids.tolist())):
            raise ValueError("retained seed cells were not present in the selected seeds")
        if self.claim_boundary != SPATIAL_TRACKLET_CLAIM_BOUNDARY:
            raise ValueError("spatial tracklet claim boundary changed")
        object.__setattr__(self, "retained_seed_cell_ids", retained)

    @property
    def retained_seed_cell_count(self) -> int:
        return len(set(self.retained_seed_cell_ids))

    @property
    def dropped_seed_cell_count(self) -> int:
        return self.seed_selection.selected_cell_count - self.retained_seed_cell_count

    def to_dict(self) -> dict[str, object]:
        return {
            "base_report": self.base_report.to_dict(),
            "seed_selection": self.seed_selection.summary(),
            "retained_seed_cell_ids": list(self.retained_seed_cell_ids),
            "retained_seed_cell_count": self.retained_seed_cell_count,
            "dropped_seed_cell_count": self.dropped_seed_cell_count,
            "claim_boundary": self.claim_boundary,
        }


def _copy_window_with_seed_mask(
    window: PredictionWindow,
    *,
    first_local_index: int,
    selection: SpatialSeedSelection,
) -> PredictionWindow:
    if window.deform_mask is None:
        raise ValueError("spatial tracklets require deform_mask")
    deform_mask = np.asarray(window.deform_mask, dtype=bool).copy()
    deform_mask[first_local_index] = False
    deform_mask[first_local_index, selection.rows, selection.columns] = True
    return PredictionWindow(
        window_id=window.window_id,
        frame_indices=window.frame_indices,
        point_map=window.point_map,
        valid_mask=window.valid_mask,
        scene_flow=window.scene_flow,
        deform_mask=deform_mask,
        ray_directions=window.ray_directions,
        dense_storage_dtype=window.dense_storage_dtype,
    )


def seed_cell_ids_by_track(tracklets: CausalTrackletSet) -> IntArray:
    """Return strict per-track spatial cell IDs retained in tracklet metadata."""

    if not isinstance(tracklets, CausalTrackletSet):
        raise TypeError("tracklets must be a CausalTrackletSet")
    raw = tracklets.metadata.get("seed_cell_ids_by_track")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("tracklets do not contain seed_cell_ids_by_track metadata")
    values = np.asarray(
        [
            _strict_integer(value, name=f"seed_cell_ids_by_track[{index}]")
            for index, value in enumerate(raw)
        ],
        dtype=np.int64,
    )
    if values.shape != (tracklets.track_count,):
        raise ValueError("seed_cell_ids_by_track length differs from track_count")
    grid = tracklets.metadata.get("seed_cell_grid_shape")
    if (
        isinstance(grid, (str, bytes))
        or not isinstance(grid, Sequence)
        or len(grid) != 2
    ):
        raise ValueError("tracklets do not contain a valid seed_cell_grid_shape")
    rows = _strict_integer(grid[0], name="seed_cell_grid_shape[0]", minimum=1)
    columns = _strict_integer(grid[1], name="seed_cell_grid_shape[1]", minimum=1)
    if np.any(values >= rows * columns):
        raise ValueError("seed_cell_ids_by_track lie outside seed_cell_grid_shape")
    values.setflags(write=False)
    return values


def build_spatially_stratified_scene_flow_tracklets(
    window: PredictionWindow,
    *,
    causal_frame_stop: int,
    seed_stride: int = 8,
    seed_selection_policy: SeedSelectionPolicy = "spatial-stratified",
    cell_grid_rows: int = 4,
    cell_grid_columns: int = 4,
    maximum_seeds_per_cell: int | None = None,
    search_radius_pixels: int = 4,
    maximum_step_error_local: float = 0.05,
    association_sigma_local: float | None = None,
    minimum_link_probability: float = 0.05,
    minimum_track_length: int = 2,
    target_deform_mask_policy: TargetDeformMaskPolicy = "allow",
) -> tuple[CausalTrackletSet, SpatialTrackletReport]:
    """Build causal tracklets with deterministic spatially balanced seeds."""

    if not isinstance(window, PredictionWindow):
        raise TypeError("window must be a PredictionWindow")
    if window.scene_flow is None or window.deform_mask is None:
        raise ValueError("spatial scene-flow tracklets require scene_flow and deform_mask")
    cutoff = _strict_integer(
        causal_frame_stop,
        name="causal_frame_stop",
        minimum=1,
    )
    eligible = np.flatnonzero(window.frame_indices < cutoff)
    if not len(eligible):
        raise ValueError("window has no frame before causal_frame_stop")
    first_local = int(eligible[0])
    seed_mask = window.valid_mask[first_local] & window.deform_mask[first_local]
    selection = select_spatial_tracklet_seeds(
        seed_mask,
        seed_stride=seed_stride,
        seed_selection_policy=seed_selection_policy,
        cell_grid_rows=cell_grid_rows,
        cell_grid_columns=cell_grid_columns,
        maximum_seeds_per_cell=maximum_seeds_per_cell,
    )
    seeded_window = _copy_window_with_seed_mask(
        window,
        first_local_index=first_local,
        selection=selection,
    )
    tracklets, base_report = build_causal_scene_flow_tracklets(
        seeded_window,
        causal_frame_stop=cutoff,
        seed_stride=1,
        search_radius_pixels=search_radius_pixels,
        maximum_step_error_local=maximum_step_error_local,
        association_sigma_local=association_sigma_local,
        minimum_link_probability=minimum_link_probability,
        minimum_track_length=minimum_track_length,
        target_deform_mask_policy=target_deform_mask_policy,
    )
    base_report = replace(base_report, seed_stride=selection.seed_stride)
    coordinate_to_cell = {
        (int(row), int(column)): int(cell_id)
        for row, column, cell_id in zip(
            selection.rows,
            selection.columns,
            selection.cell_ids,
            strict=True,
        )
    }
    retained_cell_ids: list[int] = []
    for track_id in range(tracklets.track_count):
        first = int(np.flatnonzero(tracklets.track_ids == track_id)[0])
        coordinate = (int(tracklets.rows[first]), int(tracklets.columns[first]))
        try:
            retained_cell_ids.append(coordinate_to_cell[coordinate])
        except KeyError as error:
            raise ValueError("retained track does not begin at a selected seed") from error

    metadata = dict(tracklets.metadata)
    metadata.update(
        {
            "method": "spatially-stratified-local-scene-flow-association-v1",
            "base_method": tracklets.metadata.get("method"),
            "seed_selection_policy": selection.seed_selection_policy,
            "seed_selection_summary": selection.summary(),
            "seed_cell_grid_shape": list(selection.cell_grid_shape),
            "seed_cell_ids_by_track": retained_cell_ids,
            "selected_seed_cell_count": selection.selected_cell_count,
            "retained_seed_cell_count": len(set(retained_cell_ids)),
            "spatial_support_claim_boundary": SPATIAL_TRACKLET_CLAIM_BOUNDARY,
        }
    )
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("spatial tracklet metadata must be finite JSON") from error
    enriched = CausalTrackletSet(
        window_id=tracklets.window_id,
        causal_frame_stop=tracklets.causal_frame_stop,
        source_shape=tracklets.source_shape,
        seed_frame_index=tracklets.seed_frame_index,
        track_ids=tracklets.track_ids,
        frame_indices=tracklets.frame_indices,
        local_frame_indices=tracklets.local_frame_indices,
        rows=tracklets.rows,
        columns=tracklets.columns,
        points_local=tracklets.points_local,
        link_probability=tracklets.link_probability,
        association_probability=tracklets.association_probability,
        metadata=metadata,
    )
    report = SpatialTrackletReport(
        base_report=base_report,
        seed_selection=selection,
        retained_seed_cell_ids=tuple(retained_cell_ids),
    )
    return enriched, report


__all__ = [
    "SpatialTrackletReport",
    "build_spatially_stratified_scene_flow_tracklets",
    "seed_cell_ids_by_track",
]
