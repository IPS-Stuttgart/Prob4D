"""Small correlated long-sequence benchmark for estimator development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import PredictionWindow
from .gauge import ScaleAnchor
from .metrics import TruthSequence
from .sim3 import Sim3


@dataclass(frozen=True)
class SyntheticProblem:
    truth: TruthSequence
    overlap_windows: list[PredictionWindow]
    disjoint_windows: list[PredictionWindow]
    true_overlap_gauges: dict[str, Sim3]
    true_disjoint_gauges: dict[str, Sim3]
    scale_anchors: list[ScaleAnchor]
    boundary_frames: list[int]


def _anisotropic_noise(
    generator: np.random.Generator,
    rays: np.ndarray,
    parallel_standard_deviation: np.ndarray,
    lateral_standard_deviation: np.ndarray,
) -> np.ndarray:
    parallel = generator.normal(size=rays.shape[:-1]) * parallel_standard_deviation
    raw_lateral = generator.normal(size=rays.shape)
    raw_lateral -= np.sum(raw_lateral * rays, axis=-1, keepdims=True) * rays
    return parallel[..., None] * rays + lateral_standard_deviation[..., None] * raw_lateral


def _make_truth(num_frames: int, height: int, width: int) -> TruthSequence:
    rows, columns = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.4, 1.4, width),
        indexing="ij",
    )
    points = np.empty((num_frames, height, width, 3), dtype=np.float64)
    flow = np.zeros_like(points)
    for frame in range(num_frames):
        depth = 5.0 + 0.4 * np.sin(1.7 * columns + 0.08 * frame) + 0.2 * rows
        points[frame, ..., 0] = columns * depth / 3.2
        points[frame, ..., 1] = rows * depth / 3.2
        points[frame, ..., 2] = depth
        object_mask = (np.abs(rows) < 0.45) & (np.abs(columns - 0.25) < 0.5)
        displacement = 0.025 * frame
        points[frame, object_mask, 0] += displacement
        flow[frame, object_mask, 0] = 0.025
    deform_mask = np.ones((num_frames, height, width), dtype=bool)
    deform_mask[-1] = False
    return TruthSequence(
        frame_indices=np.arange(num_frames),
        point_map=points,
        valid_mask=np.ones((num_frames, height, width), dtype=bool),
        scene_flow=flow,
        deform_mask=deform_mask,
    )


def _gauge_for_start(start: int, stride: int, generator: np.random.Generator) -> Sim3:
    index = start / max(stride, 1)
    vector = np.array(
        [
            0.018 * index,
            0.004 * index,
            -0.006 * index,
            0.003 * index,
            0.12 * index,
            -0.025 * index,
            0.04 * index,
        ]
    )
    vector += generator.normal(scale=np.array([0.002, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01]))
    if start == 0:
        vector[:] = 0.0
    # The common unknown monocular scale is deliberately not granted to the
    # unanchored estimators. Relative overlap constraints cannot observe it.
    return Sim3(scale=1.2).compose(Sim3.from_vector(vector))


def _build_windows(
    truth: TruthSequence,
    starts: list[int],
    window_size: int,
    stride: int,
    generator: np.random.Generator,
    shared_noise: np.ndarray,
    *,
    prefix: str,
    correlation: float,
) -> tuple[list[PredictionWindow], dict[str, Sim3]]:
    windows: list[PredictionWindow] = []
    gauges: dict[str, Sim3] = {}
    truth_rays = truth.point_map / np.linalg.norm(truth.point_map, axis=-1, keepdims=True)
    for window_number, start in enumerate(starts):
        stop = min(start + window_size, truth.point_map.shape[0])
        frames = np.arange(start, stop)
        window_id = f"{prefix}{window_number:03d}"
        gauge = _gauge_for_start(start, stride, generator)
        gauges[window_id] = gauge
        depth = np.linalg.norm(truth.point_map[frames], axis=-1)
        parallel_std = 0.006 + 0.0025 * depth
        lateral_std = 0.002 + 0.0006 * depth
        independent = _anisotropic_noise(
            generator,
            truth_rays[frames],
            parallel_std,
            lateral_std,
        )
        global_error = (
            np.sqrt(correlation) * shared_noise[frames] + np.sqrt(1.0 - correlation) * independent
        )
        prediction_global = truth.point_map[frames] + global_error
        flow_noise = 0.15 * _anisotropic_noise(
            generator,
            truth_rays[frames],
            parallel_std,
            lateral_std,
        )
        inverse = gauge.inverse()
        point_map = inverse.transform_points(prediction_global)
        scene_flow = inverse.transform_vectors(truth.scene_flow[frames] + flow_noise)
        local_rays = inverse.rotate_directions(truth_rays[frames])
        valid_mask = truth.valid_mask[frames].copy()
        # Sparse gross failures exercise robust alignment without dominating fusion.
        outliers = generator.random(valid_mask.shape) < 0.005
        point_map[outliers] += generator.normal(scale=0.4, size=(np.count_nonzero(outliers), 3))
        windows.append(
            PredictionWindow(
                window_id=window_id,
                frame_indices=frames,
                point_map=point_map,
                valid_mask=valid_mask,
                scene_flow=scene_flow,
                deform_mask=truth.deform_mask[frames],
                ray_directions=local_rays,
            )
        )
    return windows, gauges


def make_synthetic_problem(
    *,
    seed: int = 7,
    num_frames: int = 70,
    height: int = 10,
    width: int = 14,
    window_size: int = 25,
    overlap: int = 8,
    correlation: float = 0.75,
) -> SyntheticProblem:
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0, 1)")
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than window size")
    generator = np.random.default_rng(seed)
    truth = _make_truth(num_frames, height, width)
    rays = truth.point_map / np.linalg.norm(truth.point_map, axis=-1, keepdims=True)
    depth = np.linalg.norm(truth.point_map, axis=-1)
    shared_noise = _anisotropic_noise(
        generator,
        rays,
        0.006 + 0.0025 * depth,
        0.002 + 0.0006 * depth,
    )
    overlap_starts = list(range(0, num_frames, stride))
    if overlap_starts[-1] + 1 >= num_frames:
        overlap_starts.pop()
    disjoint_starts = list(range(0, num_frames, window_size))
    overlap_windows, overlap_gauges = _build_windows(
        truth,
        overlap_starts,
        window_size,
        stride,
        generator,
        shared_noise,
        prefix="overlap_",
        correlation=correlation,
    )
    disjoint_windows, disjoint_gauges = _build_windows(
        truth,
        disjoint_starts,
        window_size,
        window_size,
        generator,
        shared_noise,
        prefix="disjoint_",
        correlation=correlation,
    )

    scale_anchors: list[ScaleAnchor] = []
    for window in overlap_windows[::2]:
        local_first = window.point_map[0, 0, 0]
        local_second = window.point_map[0, -1, -1]
        frame = window.start_frame
        metric_distance = float(
            np.linalg.norm(truth.point_map[frame, 0, 0] - truth.point_map[frame, -1, -1])
        )
        scale_anchors.append(
            ScaleAnchor.from_metric_pair(
                window.window_id,
                local_first,
                local_second,
                metric_distance,
                standard_deviation=0.02,
            )
        )

    return SyntheticProblem(
        truth=truth,
        overlap_windows=overlap_windows,
        disjoint_windows=disjoint_windows,
        true_overlap_gauges=overlap_gauges,
        true_disjoint_gauges=disjoint_gauges,
        scale_anchors=scale_anchors,
        boundary_frames=[window.start_frame for window in overlap_windows[1:]],
    )
