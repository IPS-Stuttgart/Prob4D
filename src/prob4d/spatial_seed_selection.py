"""Deterministic spatially stratified seed selection for causal tracklets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

IntArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
SeedSelectionPolicy = Literal["regular-grid", "spatial-stratified"]

SPATIAL_TRACKLET_CLAIM_BOUNDARY = (
    "This source-only diagnostic uses causal-prefix provider support and association "
    "outputs without truth residuals or target outcomes. Spatial coverage or panel "
    "support does not establish provider competence, uncertainty calibration, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the art."
)


def _strict_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_positive_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _strict_integer(value, name=name, minimum=1)


def _selection_policy(value: object) -> SeedSelectionPolicy:
    policy = _strict_string(value, name="seed_selection_policy")
    if policy not in {"regular-grid", "spatial-stratified"}:
        raise ValueError(
            "seed_selection_policy must be 'regular-grid' or 'spatial-stratified'"
        )
    return cast(SeedSelectionPolicy, policy)


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _effective_cell_grid(
    image_shape: tuple[int, int],
    requested: tuple[int, int],
) -> tuple[int, int]:
    height, width = image_shape
    requested_rows, requested_columns = requested
    return min(height, requested_rows), min(width, requested_columns)


def _cell_boundaries(length: int, count: int) -> np.ndarray:
    boundaries = np.linspace(0, length, count + 1, dtype=np.int64)
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("spatial cell grid contains an empty cell")
    return boundaries


def _cell_ids_for_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    image_shape: tuple[int, int],
    cell_grid_shape: tuple[int, int],
) -> np.ndarray:
    row_boundaries = _cell_boundaries(image_shape[0], cell_grid_shape[0])
    column_boundaries = _cell_boundaries(image_shape[1], cell_grid_shape[1])
    row_cells = np.searchsorted(row_boundaries[1:], rows, side="right")
    column_cells = np.searchsorted(column_boundaries[1:], columns, side="right")
    return row_cells * cell_grid_shape[1] + column_cells


def _cell_anchors(start: int, stop: int, stride: int) -> tuple[int, ...]:
    center = start + (stop - start - 1) // 2
    offset = min(stride // 2, max(stop - start - 1, 0))
    regular = tuple(range(start + offset, stop, stride))
    return (center, *tuple(value for value in regular if value != center))


def _select_nearest_unique(
    candidates: np.ndarray,
    anchors: Sequence[tuple[int, int]],
    *,
    maximum_count: int | None,
) -> np.ndarray:
    selected: list[int] = []
    available = np.ones(len(candidates), dtype=bool)
    for anchor_row, anchor_column in anchors:
        if maximum_count is not None and len(selected) >= maximum_count:
            break
        remaining = np.flatnonzero(available)
        if not len(remaining):
            break
        remaining_points = candidates[remaining]
        distance = (
            (remaining_points[:, 0] - anchor_row) ** 2
            + (remaining_points[:, 1] - anchor_column) ** 2
        )
        order = np.lexsort(
            (
                remaining_points[:, 1],
                remaining_points[:, 0],
                distance,
            )
        )
        chosen = int(remaining[int(order[0])])
        selected.append(chosen)
        available[chosen] = False
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SpatialSeedSelection:
    """Deterministic causal-prefix seed coordinates and spatial-cell lineage."""

    image_shape: tuple[int, int]
    requested_cell_grid_shape: tuple[int, int]
    cell_grid_shape: tuple[int, int]
    seed_stride: int
    seed_selection_policy: SeedSelectionPolicy
    maximum_seeds_per_cell: int | None
    rows: IntArray
    columns: IntArray
    cell_ids: IntArray
    occupied_cell_ids: IntArray

    def __post_init__(self) -> None:
        if type(self.image_shape) is not tuple or len(self.image_shape) != 2:
            raise ValueError("image_shape must be a two-element tuple")
        image_shape = tuple(
            _strict_integer(value, name=f"image_shape[{index}]", minimum=1)
            for index, value in enumerate(self.image_shape)
        )
        for field_name in ("requested_cell_grid_shape", "cell_grid_shape"):
            raw = getattr(self, field_name)
            if type(raw) is not tuple or len(raw) != 2:
                raise ValueError(f"{field_name} must be a two-element tuple")
            normalized = tuple(
                _strict_integer(value, name=f"{field_name}[{index}]", minimum=1)
                for index, value in enumerate(raw)
            )
            object.__setattr__(self, field_name, normalized)
        expected_grid = _effective_cell_grid(image_shape, self.requested_cell_grid_shape)
        if self.cell_grid_shape != expected_grid:
            raise ValueError("cell_grid_shape differs from the effective requested grid")
        stride = _strict_integer(self.seed_stride, name="seed_stride", minimum=1)
        policy = _selection_policy(self.seed_selection_policy)
        maximum = _optional_positive_integer(
            self.maximum_seeds_per_cell,
            name="maximum_seeds_per_cell",
        )
        rows = np.asarray(self.rows)
        columns = np.asarray(self.columns)
        cell_ids = np.asarray(self.cell_ids)
        occupied = np.asarray(self.occupied_cell_ids)
        for name, value in (
            ("rows", rows),
            ("columns", columns),
            ("cell_ids", cell_ids),
            ("occupied_cell_ids", occupied),
        ):
            if value.dtype.kind not in {"i", "u"} or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional integer array")
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        occupied = np.asarray(occupied, dtype=np.int64)
        if not len(rows) or columns.shape != rows.shape or cell_ids.shape != rows.shape:
            raise ValueError("selected seed arrays must share one non-empty length")
        if np.any(rows < 0) or np.any(rows >= image_shape[0]):
            raise ValueError("selected seed rows lie outside image_shape")
        if np.any(columns < 0) or np.any(columns >= image_shape[1]):
            raise ValueError("selected seed columns lie outside image_shape")
        pairs = np.column_stack((rows, columns))
        if len(np.unique(pairs, axis=0)) != len(rows):
            raise ValueError("selected seed coordinates must be unique")
        order = np.lexsort((columns, rows))
        if not np.array_equal(order, np.arange(len(rows))):
            raise ValueError("selected seed coordinates must be in row-major order")
        cell_count = self.cell_grid_shape[0] * self.cell_grid_shape[1]
        if np.any(cell_ids < 0) or np.any(cell_ids >= cell_count):
            raise ValueError("selected seed cell IDs lie outside cell_grid_shape")
        expected_cell_ids = _cell_ids_for_coordinates(
            rows,
            columns,
            image_shape=image_shape,
            cell_grid_shape=self.cell_grid_shape,
        )
        if not np.array_equal(cell_ids, expected_cell_ids):
            raise ValueError("selected seed cell IDs disagree with their coordinates")
        if occupied.size == 0:
            raise ValueError("occupied_cell_ids must not be empty")
        if np.any(occupied < 0) or np.any(occupied >= cell_count):
            raise ValueError("occupied seed cell IDs lie outside cell_grid_shape")
        if not np.array_equal(occupied, np.unique(occupied)):
            raise ValueError("occupied_cell_ids must be sorted and unique")
        if not set(np.unique(cell_ids)).issubset(set(occupied)):
            raise ValueError("selected seed cells must be occupied")
        if policy == "spatial-stratified" and not np.array_equal(
            np.unique(cell_ids), occupied
        ):
            raise ValueError("spatial-stratified selection must represent every occupied cell")
        if maximum is not None:
            occupancy = np.bincount(cell_ids, minlength=cell_count)
            if np.any(occupancy > maximum):
                raise ValueError("selected seed cell occupancy exceeds maximum_seeds_per_cell")

        object.__setattr__(self, "image_shape", image_shape)
        object.__setattr__(self, "seed_stride", stride)
        object.__setattr__(self, "seed_selection_policy", policy)
        object.__setattr__(self, "maximum_seeds_per_cell", maximum)
        object.__setattr__(self, "rows", _readonly(rows, dtype=np.int64))
        object.__setattr__(self, "columns", _readonly(columns, dtype=np.int64))
        object.__setattr__(self, "cell_ids", _readonly(cell_ids, dtype=np.int64))
        object.__setattr__(
            self,
            "occupied_cell_ids",
            _readonly(occupied, dtype=np.int64),
        )

    @property
    def seed_count(self) -> int:
        return int(len(self.rows))

    @property
    def occupied_cell_count(self) -> int:
        return int(len(self.occupied_cell_ids))

    @property
    def selected_cell_count(self) -> int:
        return int(len(np.unique(self.cell_ids)))

    @property
    def cell_occupancy(self) -> IntArray:
        count = self.cell_grid_shape[0] * self.cell_grid_shape[1]
        result = np.bincount(self.cell_ids, minlength=count).astype(np.int64)
        result.setflags(write=False)
        return result

    def summary(self) -> dict[str, object]:
        occupancy = self.cell_occupancy
        active = occupancy[occupancy > 0]
        return {
            "seed_selection_policy": self.seed_selection_policy,
            "image_shape": list(self.image_shape),
            "requested_cell_grid_shape": list(self.requested_cell_grid_shape),
            "cell_grid_shape": list(self.cell_grid_shape),
            "seed_stride": self.seed_stride,
            "maximum_seeds_per_cell": self.maximum_seeds_per_cell,
            "seed_count": self.seed_count,
            "occupied_cell_count": self.occupied_cell_count,
            "selected_cell_count": self.selected_cell_count,
            "minimum_selected_cell_occupancy": int(np.min(active)),
            "maximum_selected_cell_occupancy": int(np.max(active)),
        }


def select_spatial_tracklet_seeds(
    seed_mask: BoolArray,
    *,
    seed_stride: int = 8,
    seed_selection_policy: SeedSelectionPolicy = "spatial-stratified",
    cell_grid_rows: int = 4,
    cell_grid_columns: int = 4,
    maximum_seeds_per_cell: int | None = None,
) -> SpatialSeedSelection:
    """Select deterministic seeds while retaining target-free image-cell support."""

    raw_mask = np.asarray(seed_mask)
    if raw_mask.dtype != np.dtype(bool) or raw_mask.ndim != 2:
        raise ValueError("seed_mask must be a two-dimensional boolean array")
    mask = np.asarray(raw_mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("seed_mask contains no admissible seed")
    stride = _strict_integer(seed_stride, name="seed_stride", minimum=1)
    policy = _selection_policy(seed_selection_policy)
    requested_grid = (
        _strict_integer(cell_grid_rows, name="cell_grid_rows", minimum=1),
        _strict_integer(cell_grid_columns, name="cell_grid_columns", minimum=1),
    )
    maximum = _optional_positive_integer(
        maximum_seeds_per_cell,
        name="maximum_seeds_per_cell",
    )
    image_shape = cast(tuple[int, int], mask.shape)
    cell_grid = _effective_cell_grid(image_shape, requested_grid)
    all_rows, all_columns = np.nonzero(mask)
    all_cell_ids = _cell_ids_for_coordinates(
        all_rows,
        all_columns,
        image_shape=image_shape,
        cell_grid_shape=cell_grid,
    )
    occupied = np.unique(all_cell_ids)

    if policy == "regular-grid":
        grid = np.zeros(image_shape, dtype=bool)
        grid[::stride, ::stride] = True
        selected_rows, selected_columns = np.nonzero(mask & grid)
        if not len(selected_rows):
            raise ValueError("regular-grid selection produced no admissible seed")
        if maximum is not None:
            regular_cell_ids = _cell_ids_for_coordinates(
                selected_rows,
                selected_columns,
                image_shape=image_shape,
                cell_grid_shape=cell_grid,
            )
            keep = np.zeros(len(selected_rows), dtype=bool)
            for cell_id in np.unique(regular_cell_ids):
                selected = np.flatnonzero(regular_cell_ids == cell_id)
                keep[selected[:maximum]] = True
            selected_rows = selected_rows[keep]
            selected_columns = selected_columns[keep]
    else:
        row_boundaries = _cell_boundaries(image_shape[0], cell_grid[0])
        column_boundaries = _cell_boundaries(image_shape[1], cell_grid[1])
        selected_coordinates: list[tuple[int, int]] = []
        for row_cell in range(cell_grid[0]):
            row_start = int(row_boundaries[row_cell])
            row_stop = int(row_boundaries[row_cell + 1])
            for column_cell in range(cell_grid[1]):
                column_start = int(column_boundaries[column_cell])
                column_stop = int(column_boundaries[column_cell + 1])
                local_rows, local_columns = np.nonzero(
                    mask[row_start:row_stop, column_start:column_stop]
                )
                if not len(local_rows):
                    continue
                candidates = np.column_stack(
                    (local_rows + row_start, local_columns + column_start)
                ).astype(np.int64)
                anchors = tuple(
                    (row, column)
                    for row in _cell_anchors(row_start, row_stop, stride)
                    for column in _cell_anchors(column_start, column_stop, stride)
                )
                chosen = _select_nearest_unique(
                    candidates,
                    anchors,
                    maximum_count=maximum,
                )
                selected_coordinates.extend(
                    (int(candidates[index, 0]), int(candidates[index, 1]))
                    for index in chosen
                )
        selected_coordinates.sort()
        selected_rows = np.asarray(
            [row for row, _ in selected_coordinates],
            dtype=np.int64,
        )
        selected_columns = np.asarray(
            [column for _, column in selected_coordinates],
            dtype=np.int64,
        )

    selected_cell_ids = _cell_ids_for_coordinates(
        selected_rows,
        selected_columns,
        image_shape=image_shape,
        cell_grid_shape=cell_grid,
    )
    return SpatialSeedSelection(
        image_shape=image_shape,
        requested_cell_grid_shape=requested_grid,
        cell_grid_shape=cell_grid,
        seed_stride=stride,
        seed_selection_policy=policy,
        maximum_seeds_per_cell=maximum,
        rows=selected_rows,
        columns=selected_columns,
        cell_ids=selected_cell_ids,
        occupied_cell_ids=occupied,
    )


__all__ = [
    "BoolArray",
    "SPATIAL_TRACKLET_CLAIM_BOUNDARY",
    "SeedSelectionPolicy",
    "SpatialSeedSelection",
    "select_spatial_tracklet_seeds",
]
