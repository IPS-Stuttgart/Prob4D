"""Frozen mechanics for the source-only recurrent CUT3R comparison.

This module contains no CUT3R or Torch import.  It owns the temporal schedule,
the newest-eligible control, and the common gauge/uncertainty path used by the
two restarted arms.  Keeping those operations here makes the scientific
contrast testable without a GPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .alignment import DEFAULT_COVARIANCE_CLUSTER_SIZE, WindowAlignment, align_windows
from .cut3r_camera_geometry import CameraRelativeDepthDisagreementModel
from .data import PredictionWindow
from .fusion import FusedSequence, fuse_windows
from .gauge import GaugeEstimate, RelativeGaugeConstraint, SequentialGaugeEstimator
from .sim3 import Sim3
from .uncertainty import accumulate_disagreement

SOURCE_COMPARISON_METHOD_ID: Final = (
    "cut3r-direct-sequential-sim3-decoded-uniform-source-v1"
)
ALIGNMENT_MAX_CORRESPONDENCES: Final = 100_000
ALIGNMENT_MAX_ITERATIONS: Final = 64
ALIGNMENT_HUBER_MULTIPLIER: Final = 2.5
ALIGNMENT_TOLERANCE: Final = 1e-8
GAUGE_COVARIANCE_INTERSECTION_GRID_SIZE: Final = 21


@dataclass(frozen=True, slots=True)
class WindowSpan:
    """One end-exclusive causal window in absolute frame coordinates."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int:
            raise TypeError("window bounds must be genuine integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("window bounds must define a nonempty nonnegative interval")

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def window_id(self) -> str:
        return f"window-{self.start:06d}-{self.stop:06d}"


@dataclass(frozen=True, slots=True)
class RestartedComparisonProducts:
    """The two restarted arms and their shared gauge diagnostics."""

    newest: FusedSequence
    fused: FusedSequence
    alignments: tuple[WindowAlignment, ...]
    gauges: tuple[GaugeEstimate, ...]


def causal_window_schedule(
    frame_start: int,
    frame_stop_exclusive: int,
    *,
    window_size: int,
    overlap: int,
) -> tuple[WindowSpan, ...]:
    """Return regular causal windows with one deterministic end-anchored tail."""

    values = (frame_start, frame_stop_exclusive, window_size, overlap)
    if any(type(value) is not int for value in values):
        raise TypeError("window schedule inputs must be genuine integers")
    if frame_start < 0 or frame_stop_exclusive <= frame_start:
        raise ValueError("frame interval must be nonempty and nonnegative")
    if window_size < 2 or not 0 < overlap < window_size:
        raise ValueError("overlap must lie strictly between zero and window_size")
    frame_count = frame_stop_exclusive - frame_start
    if frame_count < window_size:
        raise ValueError("source interval is shorter than the frozen window size")

    stride = window_size - overlap
    final_start = frame_stop_exclusive - window_size
    starts = list(range(frame_start, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    spans = tuple(WindowSpan(start, start + window_size) for start in starts)
    if spans[0].start != frame_start or spans[-1].stop != frame_stop_exclusive:
        raise AssertionError("window schedule failed to cover the frozen interval")
    return spans


def newest_eligible_windows(
    windows: list[PredictionWindow],
) -> list[PredictionWindow]:
    """Assign each valid pixel to the latest-starting window that supports it.

    This creates the control arm without changing geometry, gauges, uncertainty,
    or fallback support.  An older window contributes only where every newer
    overlapping window is invalid for that exact frame and pixel.
    """

    if not windows:
        raise ValueError("newest selection requires at least one window")
    ordered = sorted(
        windows,
        key=lambda window: (window.start_frame, window.stop_frame, window.window_id),
    )
    if len({window.window_id for window in ordered}) != len(ordered):
        raise ValueError("newest selection requires unique window IDs")
    spatial_shape = ordered[0].shape[1:]
    if any(window.shape[1:] != spatial_shape for window in ordered):
        raise ValueError("newest selection requires one common spatial grid")

    claimed = {
        int(frame): np.zeros(spatial_shape, dtype=bool)
        for frame in np.unique(np.concatenate([window.frame_indices for window in ordered]))
    }
    owned_masks = {
        window.window_id: np.zeros(window.shape, dtype=bool) for window in ordered
    }
    for window in reversed(ordered):
        owned = owned_masks[window.window_id]
        for local_index, frame in enumerate(window.frame_indices):
            frame_id = int(frame)
            eligible = window.valid_mask[local_index] & ~claimed[frame_id]
            owned[local_index] = eligible
            claimed[frame_id] |= eligible

    return [
        PredictionWindow(
            window_id=window.window_id,
            frame_indices=window.frame_indices,
            point_map=window.point_map,
            valid_mask=owned_masks[window.window_id],
            scene_flow=window.scene_flow,
            deform_mask=(
                None
                if window.deform_mask is None
                else window.deform_mask & owned_masks[window.window_id]
            ),
            ray_directions=window.ray_directions,
            dense_storage_dtype=window.dense_storage_dtype,
        )
        for window in ordered
    ]


def _ordered_windows(windows: list[PredictionWindow]) -> list[PredictionWindow]:
    if len(windows) < 2:
        raise ValueError("restarted comparison requires at least two windows")
    ordered = sorted(
        windows,
        key=lambda window: (window.start_frame, window.stop_frame, window.window_id),
    )
    if len({window.window_id for window in ordered}) != len(ordered):
        raise ValueError("restarted windows must have unique IDs")
    for reference, moving in zip(ordered, ordered[1:], strict=False):
        if reference.common_frames(moving).size == 0:
            raise ValueError("adjacent restarted windows must overlap")
    return ordered


def build_restarted_comparison_products(
    windows: list[PredictionWindow],
    *,
    random_seed: int,
) -> RestartedComparisonProducts:
    """Build information-matched newest and decoded-uniform restarted arms."""

    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a genuine nonnegative integer")
    ordered = _ordered_windows(windows)
    alignments = tuple(
        align_windows(
            reference,
            moving,
            max_correspondences=ALIGNMENT_MAX_CORRESPONDENCES,
            seed=random_seed,
            covariance_cluster_size=DEFAULT_COVARIANCE_CLUSTER_SIZE,
            fallback_policy="pointwise",
        )
        for reference, moving in zip(ordered, ordered[1:], strict=False)
    )
    constraints = [
        RelativeGaugeConstraint.from_window_alignment(alignment)
        for alignment in alignments
    ]
    estimates_by_id = SequentialGaugeEstimator(
        covariance_intersection_grid_size=GAUGE_COVARIANCE_INTERSECTION_GRID_SIZE
    ).estimate([window.window_id for window in ordered], constraints)
    gauges = {
        window_id: estimate.global_from_local
        for window_id, estimate in estimates_by_id.items()
    }
    gauge_covariances = {
        window_id: estimate.covariance for window_id, estimate in estimates_by_id.items()
    }

    windows_by_id = {window.window_id: window for window in ordered}
    disagreement = accumulate_disagreement(windows_by_id, list(alignments))
    uncertainty_model = CameraRelativeDepthDisagreementModel()
    point_uncertainties = {
        window.window_id: uncertainty_model.predict(
            window,
            disagreement[window.window_id],
        )
        for window in ordered
    }
    fused = fuse_windows(
        ordered,
        gauges,
        point_uncertainties,
        method="uniform",
        gauge_covariances=gauge_covariances,
    )
    newest = fuse_windows(
        newest_eligible_windows(ordered),
        gauges,
        point_uncertainties,
        method="uniform",
        gauge_covariances=gauge_covariances,
    )
    ordered_estimates = tuple(estimates_by_id[window.window_id] for window in ordered)
    return RestartedComparisonProducts(
        newest=newest,
        fused=fused,
        alignments=alignments,
        gauges=ordered_estimates,
    )


def build_native_product(window: PredictionWindow) -> FusedSequence:
    """Represent one continuous CUT3R pass through the common output contract."""

    uncertainty = CameraRelativeDepthDisagreementModel().predict(window)
    return fuse_windows(
        [window],
        {window.window_id: Sim3.identity()},
        {window.window_id: uncertainty},
        method="uniform",
    )


__all__ = [
    "ALIGNMENT_HUBER_MULTIPLIER",
    "ALIGNMENT_MAX_CORRESPONDENCES",
    "ALIGNMENT_MAX_ITERATIONS",
    "ALIGNMENT_TOLERANCE",
    "GAUGE_COVARIANCE_INTERSECTION_GRID_SIZE",
    "SOURCE_COMPARISON_METHOD_ID",
    "RestartedComparisonProducts",
    "WindowSpan",
    "build_native_product",
    "build_restarted_comparison_products",
    "causal_window_schedule",
    "newest_eligible_windows",
]
