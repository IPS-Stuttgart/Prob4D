"""Deterministic geometry-aware sampling for dense overlap alignment.

The legacy overlap path uses an unstratified random subset when more than the
requested number of correspondences are available.  This module instead keeps
broad frame and spatial-tile coverage and samples each tile across its depth
range.  The resulting cluster IDs remain suitable for the existing
frame-by-spatial-tile sandwich covariance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .alignment import (
    DENSE_ALIGNMENT_COVARIANCE_METHOD,
    IID_COVARIANCE_FALLBACK,
    POINTWISE_COVARIANCE_FALLBACK,
    AlignmentCovarianceCalibration,
    CovarianceFallbackPolicy,
    WindowAlignment,
    estimate_sim3_robust,
)
from .data import PredictionWindow

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _readonly(values: np.ndarray, *, dtype: type) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StratifiedCorrespondenceSample:
    """One deterministic overlap sample and its covariance-cluster metadata."""

    source_points: FloatArray
    target_points: FloatArray
    common_frames: IntArray
    frame_ids: IntArray
    rows: IntArray
    columns: IntArray
    cluster_ids: IntArray
    available_count: int

    def __post_init__(self) -> None:
        source = np.asarray(self.source_points, dtype=np.float64)
        target = np.asarray(self.target_points, dtype=np.float64)
        frames = np.asarray(self.common_frames, dtype=np.int64)
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        clusters = np.asarray(self.cluster_ids, dtype=np.int64)
        if source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape:
            raise ValueError("sampled source and target points must have shape (N, 3)")
        count = len(source)
        if count < 4:
            raise ValueError("stratified overlap sample requires at least four points")
        for name, values in (
            ("frame_ids", frame_ids),
            ("rows", rows),
            ("columns", columns),
            ("cluster_ids", clusters),
        ):
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
        if (
            frames.ndim != 1
            or frames.size == 0
            or np.any(frames < 0)
            or np.any(np.diff(frames) <= 0)
        ):
            raise ValueError(
                "common_frames must be nonempty, nonnegative, and strictly increasing"
            )
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("sampled overlap points must be finite")
        if np.any(frame_ids < 0) or not np.all(np.isin(frame_ids, frames)):
            raise ValueError("sampled frame IDs must be contained in common_frames")
        if np.any(rows < 0) or np.any(columns < 0) or np.any(clusters < 0):
            raise ValueError("sample indices and cluster IDs must be nonnegative")
        available_count = int(self.available_count)
        if available_count < count:
            raise ValueError("available_count cannot be smaller than the sample")
        object.__setattr__(self, "source_points", _readonly(source, dtype=np.float64))
        object.__setattr__(self, "target_points", _readonly(target, dtype=np.float64))
        object.__setattr__(self, "common_frames", _readonly(frames, dtype=np.int64))
        object.__setattr__(self, "frame_ids", _readonly(frame_ids, dtype=np.int64))
        object.__setattr__(self, "rows", _readonly(rows, dtype=np.int64))
        object.__setattr__(self, "columns", _readonly(columns, dtype=np.int64))
        object.__setattr__(self, "cluster_ids", _readonly(clusters, dtype=np.int64))
        object.__setattr__(self, "available_count", available_count)

    @property
    def sample_count(self) -> int:
        return len(self.source_points)

    @property
    def cluster_count(self) -> int:
        return int(np.unique(self.cluster_ids).size)


def _evenly_spaced_positions(count: int, requested: int) -> np.ndarray:
    if count < 1 or requested < 1 or requested > count:
        raise ValueError("evenly spaced selection has invalid size")
    if requested == count:
        return np.arange(count, dtype=np.int64)
    if requested == 1:
        return np.asarray([(count - 1) // 2], dtype=np.int64)
    positions = np.rint(np.linspace(0, count - 1, requested)).astype(np.int64)
    # Rounding can duplicate positions for very small groups. Fill any gaps with
    # the earliest unused indices so the requested count remains exact.
    positions = np.unique(positions)
    if len(positions) < requested:
        unused = np.setdiff1d(
            np.arange(count, dtype=np.int64),
            positions,
            assume_unique=True,
        )
        positions = np.sort(
            np.concatenate((positions, unused[: requested - len(positions)]))
        )
    return positions


def _cluster_quotas(counts: np.ndarray, budget: int) -> np.ndarray:
    """Allocate a deterministic exact budget across nonempty clusters."""

    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("cluster counts must be a nonempty positive vector")
    if budget < 1 or budget > int(np.sum(counts)):
        raise ValueError("sampling budget lies outside available correspondences")
    cluster_count = len(counts)
    quotas = np.zeros(cluster_count, dtype=np.int64)
    if budget < cluster_count:
        selected = _evenly_spaced_positions(cluster_count, budget)
        quotas[selected] = 1
        return quotas

    quotas[:] = 1
    remaining = budget - cluster_count
    capacity = counts - 1
    while remaining > 0 and np.any(capacity > 0):
        active = capacity > 0
        weights = np.sqrt(counts.astype(np.float64)) * active
        desired = remaining * weights / np.sum(weights)
        additions = np.minimum(np.floor(desired).astype(np.int64), capacity)
        if not np.any(additions):
            fractional = desired - np.floor(desired)
            order = np.lexsort((np.arange(cluster_count), -fractional))
            for index in order:
                if remaining == 0:
                    break
                if capacity[index] > 0:
                    additions[index] += 1
                    remaining -= 1
            quotas += additions
            capacity -= additions
            continue
        quotas += additions
        capacity -= additions
        remaining -= int(np.sum(additions))
    return quotas


def stratified_overlapping_correspondences(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int = 100_000,
    spatial_tile_size: int = 32,
) -> StratifiedCorrespondenceSample:
    """Collect a deterministic frame/tile/depth-stratified overlap sample."""

    if reference.shape[1:] != moving.shape[1:]:
        raise ValueError("overlapping windows must use the same spatial resolution")
    if max_correspondences < 4:
        raise ValueError("max_correspondences must be at least four")
    if spatial_tile_size < 1:
        raise ValueError("spatial_tile_size must be positive")
    common_frames = reference.common_frames(moving)
    if common_frames.size == 0:
        raise ValueError("windows do not overlap")

    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    raw_cluster_parts: list[np.ndarray] = []
    tile_columns = int(np.ceil(reference.shape[2] / spatial_tile_size))
    tiles_per_frame = int(np.ceil(reference.shape[1] / spatial_tile_size)) * tile_columns

    for frame_position, frame in enumerate(common_frames):
        reference_index = reference.local_index(int(frame))
        moving_index = moving.local_index(int(frame))
        mask = reference.valid_mask[reference_index] & moving.valid_mask[moving_index]
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            continue
        source_parts.append(moving.point_map[moving_index][mask])
        target_parts.append(reference.point_map[reference_index][mask])
        frame_parts.append(np.full(rows.size, int(frame), dtype=np.int64))
        row_parts.append(rows.astype(np.int64))
        column_parts.append(columns.astype(np.int64))
        tile_ids = rows // spatial_tile_size * tile_columns + columns // spatial_tile_size
        raw_cluster_parts.append(
            frame_position * tiles_per_frame + tile_ids.astype(np.int64)
        )

    if not source_parts:
        raise ValueError("overlap has no valid point correspondences")
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    frame_ids = np.concatenate(frame_parts)
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    raw_clusters = np.concatenate(raw_cluster_parts)
    available_count = len(source)
    if available_count < 4:
        raise ValueError("overlap has fewer than four valid point correspondences")

    unique_clusters, inverse = np.unique(raw_clusters, return_inverse=True)
    if available_count <= max_correspondences:
        selected = np.arange(available_count, dtype=np.int64)
    else:
        counts = np.bincount(inverse, minlength=len(unique_clusters))
        quotas = _cluster_quotas(counts, max_correspondences)
        selected_parts: list[np.ndarray] = []
        depth = 0.5 * (
            np.linalg.norm(source, axis=1) + np.linalg.norm(target, axis=1)
        )
        for cluster_index, quota in enumerate(quotas):
            if quota == 0:
                continue
            members = np.flatnonzero(inverse == cluster_index)
            order = np.lexsort(
                (
                    members,
                    columns[members],
                    rows[members],
                    depth[members],
                )
            )
            ordered_members = members[order]
            positions = _evenly_spaced_positions(len(ordered_members), int(quota))
            selected_parts.append(ordered_members[positions])
        selected = np.concatenate(selected_parts)
        selected = selected[
            np.lexsort(
                (
                    columns[selected],
                    rows[selected],
                    frame_ids[selected],
                )
            )
        ]
        if len(selected) != max_correspondences:
            raise RuntimeError("stratified sampler did not fill the requested budget")

    _, compact_clusters = np.unique(raw_clusters[selected], return_inverse=True)
    return StratifiedCorrespondenceSample(
        source_points=source[selected],
        target_points=target[selected],
        common_frames=common_frames,
        frame_ids=frame_ids[selected],
        rows=rows[selected],
        columns=columns[selected],
        cluster_ids=compact_clusters.astype(np.int64),
        available_count=available_count,
    )


def align_windows_stratified(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int = 100_000,
    spatial_tile_size: int = 32,
    covariance_calibration: AlignmentCovarianceCalibration | None = None,
    fallback_policy: CovarianceFallbackPolicy = "error",
) -> WindowAlignment:
    """Estimate a gauge using deterministic stratified correspondences.

    Fewer than eight frame/tile clusters fail closed by default.  The explicit
    ``pointwise`` fallback is intended only for small reconstruction controls and
    is recorded in the returned :class:`~prob4d.alignment.AlignmentResult`.
    """

    if fallback_policy not in {"error", "pointwise"}:
        raise ValueError("fallback_policy must be 'error' or 'pointwise'")
    sample = stratified_overlapping_correspondences(
        reference,
        moving,
        max_correspondences=max_correspondences,
        spatial_tile_size=spatial_tile_size,
    )
    clusters: np.ndarray | None = np.asarray(sample.cluster_ids, dtype=np.int64)
    covariance_fallback: str | None = None
    if sample.cluster_count <= 7:
        if fallback_policy == "error":
            raise ValueError(
                "stratified alignment produced fewer than eight independent spatial "
                "clusters; explicitly allow the pointwise approximation for a "
                "reconstruction control"
            )
        if sample.sample_count > 7:
            clusters = np.arange(sample.sample_count, dtype=np.int64)
            covariance_fallback = POINTWISE_COVARIANCE_FALLBACK
        else:
            clusters = None
            covariance_fallback = IID_COVARIANCE_FALLBACK

    result = estimate_sim3_robust(
        sample.source_points,
        sample.target_points,
        covariance_cluster_ids=clusters,
    )
    result = replace(result, covariance_fallback=covariance_fallback)
    if covariance_calibration is not None:
        calibration_method = getattr(covariance_calibration, "covariance_method", None)
        if calibration_method not in {None, DENSE_ALIGNMENT_COVARIANCE_METHOD}:
            raise ValueError(
                "gauge covariance calibration was fitted for a different "
                "covariance method"
            )
        calibration_cluster_size = getattr(
            covariance_calibration,
            "covariance_cluster_size",
            None,
        )
        if (
            calibration_cluster_size is not None
            and int(calibration_cluster_size) != spatial_tile_size
        ):
            raise ValueError(
                "gauge covariance calibration spatial cluster size does not match "
                "the stratified alignment"
            )
        result = replace(
            result,
            covariance=covariance_calibration.apply(result.covariance),
            covariance_calibration_id=covariance_calibration.artifact_id,
        )
    return WindowAlignment(
        reference_id=reference.window_id,
        moving_id=moving.window_id,
        common_frames=sample.common_frames,
        result=result,
    )


__all__ = [
    "StratifiedCorrespondenceSample",
    "align_windows_stratified",
    "stratified_overlapping_correspondences",
]
