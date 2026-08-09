"""Observation-factor conversion with spatial seed-cell likelihood groups."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .causal_tracklets import CausalTrackletSet, tracklets_to_observation_factors
from .observation_factors import ObservationFactor
from .spatial_tracklet_builder import seed_cell_ids_by_track
from .uncertainty import StructuredCovariance

FloatArray: TypeAlias = NDArray[np.floating[Any]]
CorrelationGroupMode = Literal["frame", "frame-seed-cell"]


def _strict_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_real(
    value: object,
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


def _real_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _correlation_group_mode(value: object) -> CorrelationGroupMode:
    mode = _strict_string(value, name="correlation_group_mode")
    if mode not in {"frame", "frame-seed-cell"}:
        raise ValueError(
            "correlation_group_mode must be 'frame' or 'frame-seed-cell'"
        )
    return cast(CorrelationGroupMode, mode)


def spatial_tracklets_to_observation_factors(
    tracklets: CausalTrackletSet,
    covariance: StructuredCovariance | FloatArray,
    *,
    view_id: str,
    prior_reliability: FloatArray | None = None,
    prior_nominal_probability: float = 1.0,
    effective_samples_per_group: float = 64.0,
    correlation_group_prefix: str = "scene-flow-tracklets",
    factor_id_prefix: str | None = None,
    correlation_group_mode: CorrelationGroupMode = "frame-seed-cell",
) -> tuple[ObservationFactor, ...]:
    """Convert tracklets while optionally preserving spatial cells as groups."""

    mode = _correlation_group_mode(correlation_group_mode)
    if mode == "frame":
        return tracklets_to_observation_factors(
            tracklets,
            covariance,
            view_id=view_id,
            prior_reliability=prior_reliability,
            prior_nominal_probability=prior_nominal_probability,
            effective_samples_per_group=effective_samples_per_group,
            correlation_group_prefix=correlation_group_prefix,
            factor_id_prefix=factor_id_prefix,
        )
    if not isinstance(tracklets, CausalTrackletSet):
        raise TypeError("tracklets must be a CausalTrackletSet")
    view = _strict_string(view_id, name="view_id")
    group_prefix = _strict_string(
        correlation_group_prefix,
        name="correlation_group_prefix",
    )
    nominal = _strict_real(
        prior_nominal_probability,
        name="prior_nominal_probability",
        maximum=1.0,
    )
    effective = _strict_real(
        effective_samples_per_group,
        name="effective_samples_per_group",
        strictly_positive=True,
    )
    cell_by_track = seed_cell_ids_by_track(tracklets)
    row_cells = cell_by_track[np.asarray(tracklets.track_ids, dtype=np.int64)]

    rays: np.ndarray | None
    if isinstance(covariance, StructuredCovariance):
        if covariance.parallel_variance.shape != tracklets.source_shape:
            raise ValueError("structured covariance shape differs from tracklet source")
        covariance_grid = covariance.matrices()
        rays = covariance.ray_directions
    else:
        covariance_grid = _real_array(covariance, name="covariance")
        expected = tracklets.source_shape + (3, 3)
        if covariance_grid.shape != expected:
            raise ValueError(f"covariance must have shape {expected}")
        rays = None

    if prior_reliability is None:
        reliability_grid = np.ones(tracklets.source_shape, dtype=np.float64)
    else:
        reliability_grid = _real_array(
            prior_reliability,
            name="prior_reliability",
        )
        if reliability_grid.shape != tracklets.source_shape:
            raise ValueError("prior_reliability must match tracklet source_shape")
        if np.any((reliability_grid < 0.0) | (reliability_grid > 1.0)):
            raise ValueError("prior_reliability must lie in [0, 1]")

    prefix = (
        f"{tracklets.window_id}:{view}:spatial-tracklets"
        if factor_id_prefix is None
        else _strict_string(factor_id_prefix, name="factor_id_prefix")
    )
    factors: list[ObservationFactor] = []
    for frame in np.unique(tracklets.frame_indices):
        frame_rows = np.flatnonzero(tracklets.frame_indices == frame)
        for cell_id in np.unique(row_cells[frame_rows]):
            selected = frame_rows[row_cells[frame_rows] == cell_id]
            local_indices = np.unique(tracklets.local_frame_indices[selected])
            if len(local_indices) != 1:
                raise ValueError("one absolute frame maps to multiple local frame indices")
            local_index = int(local_indices[0])
            rows = tracklets.rows[selected]
            columns = tracklets.columns[selected]
            count = len(selected)
            composite_weight = min(1.0, effective / count)
            factors.append(
                ObservationFactor(
                    factor_id=(
                        f"{prefix}:frame-{int(frame)}:seed-cell-{int(cell_id)}"
                    ),
                    frame_index=int(frame),
                    view_id=view,
                    window_id=tracklets.window_id,
                    gauge_id=tracklets.window_id,
                    point_ids=tracklets.track_ids[selected],
                    points_local_m=tracklets.points_local[selected],
                    valid_mask=np.ones(count, dtype=bool),
                    local_covariance_m2=covariance_grid[local_index, rows, columns],
                    association_probability=tracklets.association_probability[selected],
                    prior_reliability=reliability_grid[local_index, rows, columns],
                    prior_nominal_probability=nominal,
                    composite_weight=composite_weight,
                    correlation_group_id=(
                        f"{group_prefix}:frame-{int(frame)}:seed-cell-{int(cell_id)}"
                    ),
                    causal_frame_stop=tracklets.causal_frame_stop,
                    ray_directions_local=(
                        None if rays is None else rays[local_index, rows, columns]
                    ),
                )
            )
    return tuple(factors)


__all__ = [
    "CorrelationGroupMode",
    "spatial_tracklets_to_observation_factors",
]
