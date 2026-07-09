"""Decoded-space fusion of aligned MotionCrafter predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .data import PredictionWindow
from .sim3 import Sim3
from .uncertainty import StructuredCovariance

FloatArray = NDArray[np.floating]
FusionMethod = Literal["uniform", "precision", "covariance_intersection"]


@dataclass(frozen=True)
class FusedSequence:
    """A global-gauge dense sequence and its marginal covariance."""

    frame_indices: NDArray[np.integer]
    point_map: FloatArray
    valid_mask: NDArray[np.bool_]
    point_covariance: FloatArray
    contributors: NDArray[np.integer]
    scene_flow: FloatArray | None = None
    deform_mask: NDArray[np.bool_] | None = None
    flow_covariance: FloatArray | None = None

    def __post_init__(self) -> None:
        frames = np.asarray(self.frame_indices, dtype=np.int64)
        points = np.asarray(self.point_map, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=bool)
        covariance = np.asarray(self.point_covariance, dtype=np.float64)
        contributors = np.asarray(self.contributors)
        if points.shape[:-1] != mask.shape or points.shape[-1] != 3:
            raise ValueError("point_map and valid_mask shapes are inconsistent")
        if covariance.shape != points.shape + (3,):
            raise ValueError("point_covariance must have shape (T, H, W, 3, 3)")
        if contributors.shape != mask.shape:
            raise ValueError("contributors must have shape (T, H, W)")
        if frames.shape != (points.shape[0],):
            raise ValueError("frame_indices must match the sequence time dimension")
        if (self.scene_flow is None) != (self.deform_mask is None):
            raise ValueError("scene_flow and deform_mask must both be present or absent")
        if self.scene_flow is not None:
            flow = np.asarray(self.scene_flow, dtype=np.float64)
            flow_mask = np.asarray(self.deform_mask, dtype=bool)
            flow_covariance = np.asarray(self.flow_covariance, dtype=np.float64)
            if flow.shape != points.shape or flow_mask.shape != mask.shape:
                raise ValueError("scene-flow arrays must match point-map shape")
            if flow_covariance.shape != covariance.shape:
                raise ValueError("flow_covariance must match point_covariance shape")


def _regularized_inverse(covariance: FloatArray, floor: float = 1e-12) -> FloatArray:
    covariance = np.asarray(covariance, dtype=np.float64)
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    identity = np.eye(symmetric.shape[-1])
    try:
        return np.linalg.inv(symmetric + floor * identity)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, floor)
        return np.einsum("...ij,...j,...kj->...ik", eigenvectors, 1.0 / eigenvalues, eigenvectors)


def fuse_gaussians_independent(
    first_mean: FloatArray,
    first_covariance: FloatArray,
    second_mean: FloatArray,
    second_covariance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fuse Gaussian estimates while deliberately assuming independence."""

    first_information = _regularized_inverse(first_covariance)
    second_information = _regularized_inverse(second_covariance)
    covariance = _regularized_inverse(first_information + second_information)
    information_vector = np.einsum("...ij,...j->...i", first_information, first_mean) + np.einsum(
        "...ij,...j->...i", second_information, second_mean
    )
    mean = np.einsum("...ij,...j->...i", covariance, information_vector)
    return mean, covariance


def _ci_weight_grid(grid_size: int, minimum_weight: float) -> FloatArray:
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    if not 0.0 <= minimum_weight < 0.5:
        raise ValueError("minimum_weight must be in [0, 0.5)")
    grid = np.linspace(minimum_weight, 1.0 - minimum_weight, grid_size)
    # Evaluate the midpoint first so equal-objective ties produce equal fusion.
    return grid[np.argsort(np.abs(grid - 0.5), kind="stable")]


def fuse_gaussians_covariance_intersection(
    first_mean: FloatArray,
    first_covariance: FloatArray,
    second_mean: FloatArray,
    second_covariance: FloatArray,
    *,
    grid_size: int = 21,
    minimum_weight: float = 0.0,
    chunk_size: int = 16_384,
    weight_mode: Literal["global", "pointwise"] = "global",
    weight_sample_size: int = 4_096,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Conservatively fuse estimates with unknown cross-correlation.

    By default, one CI weight minimizes mean log determinant over a representative
    sample. In :func:`fuse_windows` this means one weight per overlapping frame,
    which is fast and gives the shared-backbone correlation a coherent treatment.
    ``pointwise`` mode retains the more expensive diagnostic alternative.
    """

    first_mean = np.asarray(first_mean, dtype=np.float64)
    second_mean = np.asarray(second_mean, dtype=np.float64)
    first_covariance = np.asarray(first_covariance, dtype=np.float64)
    second_covariance = np.asarray(second_covariance, dtype=np.float64)
    if first_mean.shape != second_mean.shape:
        raise ValueError("CI means must have matching shapes")
    dimension = first_mean.shape[-1]
    if first_covariance.shape != first_mean.shape + (dimension,):
        raise ValueError("first covariance shape does not match mean")
    if second_covariance.shape != first_covariance.shape:
        raise ValueError("CI covariance shapes must match")

    leading_shape = first_mean.shape[:-1]
    mean_one = first_mean.reshape(-1, dimension)
    mean_two = second_mean.reshape(-1, dimension)
    covariance_one = first_covariance.reshape(-1, dimension, dimension)
    covariance_two = second_covariance.reshape(-1, dimension, dimension)
    output_mean = np.empty_like(mean_one)
    output_covariance = np.empty_like(covariance_one)
    output_weight = np.empty(mean_one.shape[0], dtype=np.float64)
    weight_grid = _ci_weight_grid(grid_size, minimum_weight)

    if weight_mode == "global":
        if mean_one.shape[0] <= weight_sample_size:
            sample = np.arange(mean_one.shape[0])
        else:
            sample = np.linspace(0, mean_one.shape[0] - 1, weight_sample_size, dtype=np.int64)
        sampled_information_one = _regularized_inverse(covariance_one[sample])
        sampled_information_two = _regularized_inverse(covariance_two[sample])
        best_score = np.inf
        best_weight = 0.5
        for weight in weight_grid:
            information = (
                weight * sampled_information_one + (1.0 - weight) * sampled_information_two
            )
            covariance = _regularized_inverse(information)
            _, log_determinant = np.linalg.slogdet(covariance)
            score = float(np.mean(log_determinant))
            if score < best_score - 1e-12:
                best_score = score
                best_weight = float(weight)

        for start in range(0, mean_one.shape[0], chunk_size):
            stop = min(start + chunk_size, mean_one.shape[0])
            information_one = _regularized_inverse(covariance_one[start:stop])
            information_two = _regularized_inverse(covariance_two[start:stop])
            information = best_weight * information_one + (1.0 - best_weight) * information_two
            covariance = _regularized_inverse(information)
            information_vector = best_weight * np.einsum(
                "...ij,...j->...i", information_one, mean_one[start:stop]
            ) + (1.0 - best_weight) * np.einsum(
                "...ij,...j->...i", information_two, mean_two[start:stop]
            )
            output_mean[start:stop] = np.einsum("...ij,...j->...i", covariance, information_vector)
            output_covariance[start:stop] = covariance
            output_weight[start:stop] = best_weight
        return (
            output_mean.reshape(first_mean.shape),
            output_covariance.reshape(first_covariance.shape),
            output_weight.reshape(leading_shape),
        )

    if weight_mode != "pointwise":
        raise ValueError("weight_mode must be 'global' or 'pointwise'")

    for start in range(0, mean_one.shape[0], chunk_size):
        stop = min(start + chunk_size, mean_one.shape[0])
        information_one = _regularized_inverse(covariance_one[start:stop])
        information_two = _regularized_inverse(covariance_two[start:stop])
        best_score = np.full(stop - start, np.inf)
        best_weight = np.full(stop - start, 0.5)
        best_covariance = np.empty_like(covariance_one[start:stop])
        for weight in weight_grid:
            information = weight * information_one + (1.0 - weight) * information_two
            covariance = _regularized_inverse(information)
            _, log_determinant = np.linalg.slogdet(covariance)
            improved = log_determinant < best_score - 1e-12
            best_score[improved] = log_determinant[improved]
            best_weight[improved] = weight
            best_covariance[improved] = covariance[improved]

        information_vector = best_weight[:, None] * np.einsum(
            "...ij,...j->...i", information_one, mean_one[start:stop]
        ) + (1.0 - best_weight)[:, None] * np.einsum(
            "...ij,...j->...i", information_two, mean_two[start:stop]
        )
        output_mean[start:stop] = np.einsum("...ij,...j->...i", best_covariance, information_vector)
        output_covariance[start:stop] = best_covariance
        output_weight[start:stop] = best_weight

    return (
        output_mean.reshape(first_mean.shape),
        output_covariance.reshape(first_covariance.shape),
        output_weight.reshape(leading_shape),
    )


def _uniform_update(
    mean: FloatArray,
    covariance: FloatArray,
    count: NDArray[np.integer],
    incoming_mean: FloatArray,
    incoming_covariance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    next_count = count + 1
    next_mean = mean + (incoming_mean - mean) / next_count[..., None]
    old_offset = mean - next_mean
    new_offset = incoming_mean - next_mean
    next_covariance = (
        count[..., None, None]
        * (covariance + np.einsum("...i,...j->...ij", old_offset, old_offset))
        + incoming_covariance
        + np.einsum("...i,...j->...ij", new_offset, new_offset)
    ) / next_count[..., None, None]
    return next_mean, next_covariance


def _fuse_update(
    method: FusionMethod,
    mean: FloatArray,
    covariance: FloatArray,
    count: NDArray[np.integer],
    incoming_mean: FloatArray,
    incoming_covariance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if method == "uniform":
        return _uniform_update(mean, covariance, count, incoming_mean, incoming_covariance)
    if method == "precision":
        return fuse_gaussians_independent(mean, covariance, incoming_mean, incoming_covariance)
    if method == "covariance_intersection":
        fused_mean, fused_covariance, _ = fuse_gaussians_covariance_intersection(
            mean, covariance, incoming_mean, incoming_covariance
        )
        return fused_mean, fused_covariance
    raise ValueError(f"unknown fusion method {method!r}")


def fuse_windows(
    windows: list[PredictionWindow],
    gauges: dict[str, Sim3],
    point_uncertainties: dict[str, StructuredCovariance],
    *,
    method: FusionMethod,
    flow_uncertainties: dict[str, StructuredCovariance] | None = None,
) -> FusedSequence:
    """Transform decoded windows to a common gauge and fuse duplicate pixels."""

    if not windows:
        raise ValueError("at least one prediction window is required")
    height, width = windows[0].shape[1:]
    if any(window.shape[1:] != (height, width) for window in windows):
        raise ValueError("all windows must use the same spatial resolution")
    all_frames = np.unique(np.concatenate([window.frame_indices for window in windows]))
    frame_positions = {int(frame): index for index, frame in enumerate(all_frames)}
    shape = (all_frames.size, height, width)
    point_map = np.zeros(shape + (3,), dtype=np.float64)
    valid_mask = np.zeros(shape, dtype=bool)
    point_covariance = np.zeros(shape + (3, 3), dtype=np.float64)
    contributors = np.zeros(shape, dtype=np.uint16)

    has_flow = any(window.scene_flow is not None for window in windows)
    scene_flow = np.zeros_like(point_map) if has_flow else None
    deform_mask = np.zeros(shape, dtype=bool) if has_flow else None
    flow_covariance = np.zeros_like(point_covariance) if has_flow else None
    flow_contributors = np.zeros(shape, dtype=np.uint16) if has_flow else None

    for window in windows:
        if window.window_id not in gauges or window.window_id not in point_uncertainties:
            raise KeyError(f"missing gauge or uncertainty for window {window.window_id!r}")
        gauge = gauges[window.window_id]
        transformed_points = gauge.transform_points(window.point_map)
        transformed_point_covariance = (
            point_uncertainties[window.window_id].transformed(gauge).matrices()
        )
        if window.scene_flow is not None:
            uncertainty = (
                flow_uncertainties[window.window_id]
                if flow_uncertainties is not None
                else point_uncertainties[window.window_id]
            )
            transformed_flow = gauge.transform_vectors(window.scene_flow)
            transformed_flow_covariance = uncertainty.transformed(gauge).matrices()

        for local_index, frame in enumerate(window.frame_indices):
            output_index = frame_positions[int(frame)]
            incoming = window.valid_mask[local_index]
            new = incoming & ~valid_mask[output_index]
            overlap = incoming & valid_mask[output_index]
            point_map[output_index][new] = transformed_points[local_index][new]
            point_covariance[output_index][new] = transformed_point_covariance[local_index][new]
            valid_mask[output_index][new] = True
            contributors[output_index][new] = 1
            if np.any(overlap):
                updated_mean, updated_covariance = _fuse_update(
                    method,
                    point_map[output_index][overlap],
                    point_covariance[output_index][overlap],
                    contributors[output_index][overlap],
                    transformed_points[local_index][overlap],
                    transformed_point_covariance[local_index][overlap],
                )
                point_map[output_index][overlap] = updated_mean
                point_covariance[output_index][overlap] = updated_covariance
                contributors[output_index][overlap] += 1

            if window.scene_flow is None:
                continue
            flow_incoming = window.deform_mask[local_index]
            flow_new = flow_incoming & ~deform_mask[output_index]
            flow_overlap = flow_incoming & deform_mask[output_index]
            scene_flow[output_index][flow_new] = transformed_flow[local_index][flow_new]
            flow_covariance[output_index][flow_new] = transformed_flow_covariance[local_index][
                flow_new
            ]
            deform_mask[output_index][flow_new] = True
            flow_contributors[output_index][flow_new] = 1
            if np.any(flow_overlap):
                updated_mean, updated_covariance = _fuse_update(
                    method,
                    scene_flow[output_index][flow_overlap],
                    flow_covariance[output_index][flow_overlap],
                    flow_contributors[output_index][flow_overlap],
                    transformed_flow[local_index][flow_overlap],
                    transformed_flow_covariance[local_index][flow_overlap],
                )
                scene_flow[output_index][flow_overlap] = updated_mean
                flow_covariance[output_index][flow_overlap] = updated_covariance
                flow_contributors[output_index][flow_overlap] += 1

    return FusedSequence(
        frame_indices=all_frames,
        point_map=point_map,
        valid_mask=valid_mask,
        point_covariance=point_covariance,
        contributors=contributors,
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        flow_covariance=flow_covariance,
    )
