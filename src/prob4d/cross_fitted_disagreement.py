"""Cluster-cross-fitted overlap disagreement for uncertainty diagnostics.

The ordinary overlap disagreement is evaluated with the same correspondences that
fit each relative ``Sim(3)`` gauge.  This module provides an additive diagnostic
that holds out whole frame-by-spatial-tile clusters, refits the gauge on the
remaining clusters, and scores only the held-out rows.

No in-sample transform is substituted when a fold cannot be fitted.  Rows from a
failed fold therefore retain zero evidence count, making missing cross-fitted
support auditable instead of optimistic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._scientific_scalars import require_genuine_integer
from .alignment import WindowAlignment, estimate_sim3_robust
from .data import PredictionWindow
from .sim3 import Sim3
from .uncertainty import DisagreementEvidence

IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class CrossFittedDisagreementReport:
    """Audit information for one cross-fitted disagreement calculation."""

    alignment_count: int
    requested_folds: int
    candidate_folds: int
    fitted_folds: int
    skipped_folds: int
    skipped_alignments: int
    overlap_points: int
    evaluated_points: int
    cluster_size: int
    maximum_training_correspondences: int
    seed: int

    def __post_init__(self) -> None:
        integer_fields = {
            "alignment_count": self.alignment_count,
            "requested_folds": self.requested_folds,
            "candidate_folds": self.candidate_folds,
            "fitted_folds": self.fitted_folds,
            "skipped_folds": self.skipped_folds,
            "skipped_alignments": self.skipped_alignments,
            "overlap_points": self.overlap_points,
            "evaluated_points": self.evaluated_points,
            "cluster_size": self.cluster_size,
            "maximum_training_correspondences": (self.maximum_training_correspondences),
            "seed": self.seed,
        }
        minimums = {
            "requested_folds": 2,
            "cluster_size": 1,
            "maximum_training_correspondences": 4,
        }
        for name, value in integer_fields.items():
            normalized = require_genuine_integer(
                value,
                name=name,
                minimum=minimums.get(name, 0),
            )
            object.__setattr__(self, name, normalized)
        if self.fitted_folds + self.skipped_folds != self.candidate_folds:
            raise ValueError("fitted and skipped folds must sum to candidate_folds")
        if self.evaluated_points > self.overlap_points:
            raise ValueError("evaluated_points cannot exceed overlap_points")

    @property
    def evaluated_fraction(self) -> float:
        """Fraction of overlap rows scored strictly out of fold."""

        if self.overlap_points == 0:
            return 0.0
        return self.evaluated_points / self.overlap_points

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible audit summary."""

        return {
            "alignment_count": self.alignment_count,
            "requested_folds": self.requested_folds,
            "candidate_folds": self.candidate_folds,
            "fitted_folds": self.fitted_folds,
            "skipped_folds": self.skipped_folds,
            "skipped_alignments": self.skipped_alignments,
            "overlap_points": self.overlap_points,
            "evaluated_points": self.evaluated_points,
            "evaluated_fraction": self.evaluated_fraction,
            "cluster_size": self.cluster_size,
            "maximum_training_correspondences": (self.maximum_training_correspondences),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class _OverlapRows:
    reference_index: int
    moving_index: int
    rows: IntArray
    columns: IntArray
    cluster_ids: IntArray
    reference_rays: NDArray[np.floating]
    moving_rays: NDArray[np.floating]


def _clustered_overlap_rows(
    reference: PredictionWindow,
    moving: PredictionWindow,
    alignment: WindowAlignment,
    *,
    cluster_size: int,
) -> tuple[list[_OverlapRows], int, int]:
    if reference.shape[1:] != moving.shape[1:]:
        raise ValueError("overlapping windows must use the same spatial resolution")
    records: list[_OverlapRows] = []
    cluster_offset = 0
    overlap_points = 0
    width = reference.shape[2]
    tile_columns = int(np.ceil(width / cluster_size))
    for frame in alignment.common_frames:
        reference_index = reference.local_index(int(frame))
        moving_index = moving.local_index(int(frame))
        mask = reference.valid_mask[reference_index] & moving.valid_mask[moving_index]
        rows, columns = np.nonzero(mask)
        if not rows.size:
            continue
        tile_ids = rows // cluster_size * tile_columns + columns // cluster_size
        _, compact = np.unique(tile_ids, return_inverse=True)
        cluster_ids = compact.astype(np.int64) + cluster_offset
        cluster_offset += int(np.max(compact) + 1)
        overlap_points += int(rows.size)
        records.append(
            _OverlapRows(
                reference_index=reference_index,
                moving_index=moving_index,
                rows=rows.astype(np.int64),
                columns=columns.astype(np.int64),
                cluster_ids=cluster_ids,
                reference_rays=reference.rays_at(
                    reference_index,
                    dtype=np.float64,
                )[rows, columns],
                moving_rays=moving.rays_at(
                    moving_index,
                    dtype=np.float64,
                )[rows, columns],
            )
        )
    return records, cluster_offset, overlap_points


def _stable_alignment_seed(
    seed: int,
    alignment: WindowAlignment,
) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(alignment.reference_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(alignment.moving_id.encode("utf-8"))
    digest.update(b"\0")
    frames = np.asarray(alignment.common_frames, dtype="<i8")
    digest.update(frames.tobytes())
    return int.from_bytes(digest.digest()[:8], "big")


def _fold_assignment(
    cluster_count: int,
    fold_count: int,
    *,
    seed: int,
) -> IntArray:
    generator = np.random.default_rng(seed)
    order = generator.permutation(cluster_count)
    result = np.empty(cluster_count, dtype=np.int64)
    result[order] = np.arange(cluster_count, dtype=np.int64) % fold_count
    return result


def _training_correspondences(
    reference: PredictionWindow,
    moving: PredictionWindow,
    records: list[_OverlapRows],
    cluster_folds: IntArray,
    *,
    held_out_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for record in records:
        selected = cluster_folds[record.cluster_ids] != held_out_fold
        if not np.any(selected):
            continue
        rows = record.rows[selected]
        columns = record.columns[selected]
        source_parts.append(moving.point_map[record.moving_index][rows, columns])
        target_parts.append(reference.point_map[record.reference_index][rows, columns])
    if not source_parts:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.concatenate(source_parts), np.concatenate(target_parts)


def _residual_energy(
    reference_points: np.ndarray,
    moving_points: np.ndarray,
    reference_rays: np.ndarray,
    moving_rays: np.ndarray,
    transform: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    source_aligned = transform.transform_points(moving_points)
    residual = reference_points - source_aligned
    rays = reference_rays + transform.rotate_directions(moving_rays)
    ray_norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = np.divide(
        rays,
        ray_norm,
        out=reference_rays.copy(),
        where=ray_norm > 1e-12,
    )
    parallel = np.sum(residual * rays, axis=-1) ** 2
    total = np.sum(residual**2, axis=-1)
    lateral = 0.5 * np.maximum(total - parallel, 0.0)
    return parallel, lateral


def _accumulate_rows(
    evidence: dict[str, DisagreementEvidence],
    reference: PredictionWindow,
    moving: PredictionWindow,
    record: _OverlapRows,
    selected: np.ndarray,
    transform: Sim3,
) -> int:
    rows = record.rows[selected]
    columns = record.columns[selected]
    if not rows.size:
        return 0
    parallel, lateral = _residual_energy(
        reference.point_map[record.reference_index][rows, columns],
        moving.point_map[record.moving_index][rows, columns],
        record.reference_rays[selected],
        record.moving_rays[selected],
        transform,
    )
    for window_id, local_index in (
        (reference.window_id, record.reference_index),
        (moving.window_id, record.moving_index),
    ):
        item = evidence[window_id]
        item.parallel_sum[local_index][rows, columns] += parallel
        item.lateral_sum[local_index][rows, columns] += lateral
        item.count[local_index][rows, columns] += 1.0
    return int(rows.size)


def accumulate_cross_fitted_disagreement(
    windows: dict[str, PredictionWindow],
    alignments: list[WindowAlignment],
    *,
    folds: int = 4,
    cluster_size: int = 32,
    maximum_training_correspondences: int = 100_000,
    seed: int = 0,
) -> tuple[
    dict[str, DisagreementEvidence],
    CrossFittedDisagreementReport,
]:
    """Estimate overlap disagreement without scoring gauge-fit correspondences.

    Each alignment is partitioned into deterministic frame-by-spatial-tile
    clusters.  For every fold, the relative gauge is fitted on all other
    clusters and residual energy is accumulated only for the held-out clusters.

    A fold that has too few or rank-deficient training correspondences is skipped.
    The original in-sample alignment is never substituted.  Consequently, rows
    from skipped folds retain zero evidence count and are visible in the returned
    report.

    The resulting evidence changes the uncertainty model's semantics.  Existing
    point-uncertainty calibrations must therefore be regenerated before this
    diagnostic is promoted into a claim-bearing provider export.
    """

    fold_count = require_genuine_integer(folds, name="folds", minimum=2)
    cluster_width = require_genuine_integer(
        cluster_size,
        name="cluster_size",
        minimum=1,
    )
    maximum = require_genuine_integer(
        maximum_training_correspondences,
        name="maximum_training_correspondences",
        minimum=4,
    )
    normalized_seed = require_genuine_integer(seed, name="seed", minimum=0)
    required_window_ids = {
        window_id
        for alignment in alignments
        for window_id in (alignment.reference_id, alignment.moving_id)
    }
    missing = required_window_ids.difference(windows)
    if missing:
        raise KeyError(f"alignment windows are missing: {sorted(missing)}")

    evidence = {
        window_id: DisagreementEvidence.empty(window.shape) for window_id, window in windows.items()
    }
    candidate_folds = 0
    fitted_folds = 0
    skipped_folds = 0
    skipped_alignments = 0
    overlap_points = 0
    evaluated_points = 0

    for alignment in alignments:
        reference = windows[alignment.reference_id]
        moving = windows[alignment.moving_id]
        records, cluster_count, alignment_points = _clustered_overlap_rows(
            reference,
            moving,
            alignment,
            cluster_size=cluster_width,
        )
        overlap_points += alignment_points
        effective_folds = min(fold_count, cluster_count)
        if effective_folds < 2 or not records:
            skipped_alignments += 1
            continue
        candidate_folds += effective_folds
        alignment_seed = _stable_alignment_seed(normalized_seed, alignment)
        cluster_folds = _fold_assignment(
            cluster_count,
            effective_folds,
            seed=alignment_seed,
        )
        alignment_fitted_folds = 0

        for held_out_fold in range(effective_folds):
            source, target = _training_correspondences(
                reference,
                moving,
                records,
                cluster_folds,
                held_out_fold=held_out_fold,
            )
            if source.shape[0] > maximum:
                generator = np.random.default_rng(alignment_seed + 97_409 * held_out_fold)
                selected = np.sort(
                    generator.choice(
                        source.shape[0],
                        size=maximum,
                        replace=False,
                    )
                )
                source = source[selected]
                target = target[selected]
            if source.shape[0] < 4:
                skipped_folds += 1
                continue
            try:
                transform = estimate_sim3_robust(source, target).transform
            except (ValueError, np.linalg.LinAlgError):
                skipped_folds += 1
                continue

            fold_points = 0
            for record in records:
                held_out = cluster_folds[record.cluster_ids] == held_out_fold
                fold_points += _accumulate_rows(
                    evidence,
                    reference,
                    moving,
                    record,
                    held_out,
                    transform,
                )
            fitted_folds += 1
            alignment_fitted_folds += 1
            evaluated_points += fold_points

        if alignment_fitted_folds == 0:
            skipped_alignments += 1

    report = CrossFittedDisagreementReport(
        alignment_count=len(alignments),
        requested_folds=fold_count,
        candidate_folds=candidate_folds,
        fitted_folds=fitted_folds,
        skipped_folds=skipped_folds,
        skipped_alignments=skipped_alignments,
        overlap_points=overlap_points,
        evaluated_points=evaluated_points,
        cluster_size=cluster_width,
        maximum_training_correspondences=maximum,
        seed=normalized_seed,
    )
    return evidence, report


__all__ = [
    "CrossFittedDisagreementReport",
    "accumulate_cross_fitted_disagreement",
]
