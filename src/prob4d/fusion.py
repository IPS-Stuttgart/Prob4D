"""Decoded-space fusion of aligned MotionCrafter predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .covariance import regularized_inverse_psd
from .data import PredictionWindow
from .sim3 import Sim3, so3_log, so3_right_jacobian
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
    return regularized_inverse_psd(
        covariance,
        name="fusion covariance",
        eigenvalue_floor=floor,
    )


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
    """Conservatively fuse two estimates with unknown cross-correlation.

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
            sign, log_determinant = np.linalg.slogdet(covariance)
            if np.any(sign <= 0.0) or not np.all(np.isfinite(log_determinant)):
                raise ValueError("covariance intersection produced a non-positive covariance")
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
            output_mean[start:stop] = np.einsum(
                "...ij,...j->...i", covariance, information_vector
            )
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
            sign, log_determinant = np.linalg.slogdet(covariance)
            if np.any(sign <= 0.0) or not np.all(np.isfinite(log_determinant)):
                raise ValueError("covariance intersection produced a non-positive covariance")
            improved = log_determinant < best_score - 1e-12
            best_score[improved] = log_determinant[improved]
            best_weight[improved] = weight
            best_covariance[improved] = covariance[improved]

        information_vector = best_weight[:, None] * np.einsum(
            "...ij,...j->...i", information_one, mean_one[start:stop]
        ) + (1.0 - best_weight)[:, None] * np.einsum(
            "...ij,...j->...i", information_two, mean_two[start:stop]
        )
        output_mean[start:stop] = np.einsum(
            "...ij,...j->...i", best_covariance, information_vector
        )
        output_covariance[start:stop] = best_covariance
        output_weight[start:stop] = best_weight

    return (
        output_mean.reshape(first_mean.shape),
        output_covariance.reshape(first_covariance.shape),
        output_weight.reshape(leading_shape),
    )



def fuse_gaussians_generalized_covariance_intersection(
    means: FloatArray,
    covariances: FloatArray,
    *,
    grid_size: int = 21,
    minimum_weight: float = 0.0,
    chunk_size: int = 16_384,
    weight_sample_size: int = 4_096,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fuse one or more estimates in one generalized-CI problem.

    The first axis indexes contributors. One global simplex weight vector is
    optimized jointly, avoiding the non-associativity of repeated pairwise CI.
    One contributor is returned unchanged, while two contributors delegate to
    :func:`fuse_gaussians_covariance_intersection` for exact numerical parity.
    """

    values = np.asarray(means, dtype=np.float64)
    matrices = np.asarray(covariances, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 1:
        raise ValueError("generalized CI means must have shape (K, ..., D)")
    if not np.all(np.isfinite(values)):
        raise ValueError("generalized CI means must be finite")
    contributor_count = values.shape[0]
    dimension = values.shape[-1]
    if matrices.shape != values.shape + (dimension,):
        raise ValueError("generalized CI covariance shape does not match means")
    if not 0.0 <= minimum_weight < 1.0 / contributor_count:
        raise ValueError("minimum_weight must lie in [0, 1 / contributor_count)")
    if chunk_size < 1 or weight_sample_size < 1 or maximum_iterations < 1:
        raise ValueError("generalized CI sizes and iteration count must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("generalized CI tolerance must be finite and positive")
    if contributor_count == 1:
        _regularized_inverse(matrices)
        return values[0].copy(), matrices[0].copy(), np.ones(1, dtype=np.float64)
    if contributor_count == 2:
        mean, covariance, first_weight = fuse_gaussians_covariance_intersection(
            values[0],
            matrices[0],
            values[1],
            matrices[1],
            grid_size=grid_size,
            minimum_weight=minimum_weight,
            chunk_size=chunk_size,
            weight_mode="global",
            weight_sample_size=weight_sample_size,
        )
        value = float(np.ravel(first_weight)[0])
        return mean, covariance, np.asarray([value, 1.0 - value])

    from ._generalized_ci import fuse_nway_covariance_intersection

    return fuse_nway_covariance_intersection(
        values,
        matrices,
        minimum_weight=minimum_weight,
        weight_sample_size=weight_sample_size,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        chunk_size=chunk_size,
    )


def _fuse_contributor_stack(
    means: FloatArray,
    covariances: FloatArray,
    *,
    method: FusionMethod,
) -> tuple[FloatArray, FloatArray]:
    contributor_count = means.shape[0]
    if contributor_count == 1:
        return means[0], covariances[0]
    if method == "uniform":
        fused_mean = np.mean(means, axis=0)
        offsets = means - fused_mean[None]
        fused_covariance = np.mean(
            covariances + np.einsum("mni,mnj->mnij", offsets, offsets),
            axis=0,
        )
        return fused_mean, fused_covariance
    if method == "precision":
        information = _regularized_inverse(covariances)
        fused_covariance = _regularized_inverse(np.sum(information, axis=0))
        information_vector = np.sum(
            np.einsum("mnij,mnj->mni", information, means),
            axis=0,
        )
        fused_mean = np.einsum("nij,nj->ni", fused_covariance, information_vector)
        return fused_mean, fused_covariance
    if method == "covariance_intersection":
        if contributor_count == 2:
            fused_mean, fused_covariance, _ = fuse_gaussians_covariance_intersection(
                means[0],
                covariances[0],
                means[1],
                covariances[1],
            )
            return fused_mean, fused_covariance
        from ._generalized_ci import fuse_nway_covariance_intersection

        fused_mean, fused_covariance, _ = fuse_nway_covariance_intersection(
            means,
            covariances,
            canonicalize=False,
        )
        return fused_mean, fused_covariance
    raise ValueError(f"unknown fusion method {method!r}")


def _fuse_frame_fields(
    means: list[FloatArray],
    covariances: list[FloatArray],
    masks: list[NDArray[np.bool_]],
    *,
    method: FusionMethod,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], NDArray[np.uint16]]:
    """Fuse all contributors to one frame without pairwise ordering effects."""

    stacked_means = np.stack(means)
    stacked_covariances = np.stack(covariances)
    stacked_masks = np.stack(masks)
    contributor_count, height, width = stacked_masks.shape
    active_rows = stacked_masks.reshape(contributor_count, -1).T
    patterns, inverse = np.unique(active_rows, axis=0, return_inverse=True)
    output_mean = np.zeros((height * width, 3), dtype=np.float64)
    output_covariance = np.zeros((height * width, 3, 3), dtype=np.float64)
    output_count = np.sum(active_rows, axis=1, dtype=np.uint16)
    flat_means = stacked_means.reshape(contributor_count, -1, 3)
    flat_covariances = stacked_covariances.reshape(contributor_count, -1, 3, 3)
    for pattern_index, pattern in enumerate(patterns):
        active = np.flatnonzero(pattern)
        if len(active) == 0:
            continue
        selected = inverse == pattern_index
        fused_mean, fused_covariance = _fuse_contributor_stack(
            flat_means[active][:, selected],
            flat_covariances[active][:, selected],
            method=method,
        )
        output_mean[selected] = fused_mean
        output_covariance[selected] = fused_covariance
    valid = output_count > 0
    return (
        output_mean.reshape(height, width, 3),
        output_covariance.reshape(height, width, 3, 3),
        valid.reshape(height, width),
        output_count.reshape(height, width),
    )


def _structured_covariance_frame(
    uncertainty: StructuredCovariance,
    transform: Sim3,
    local_index: int,
) -> FloatArray:
    rays = transform.rotate_directions(uncertainty.ray_directions[local_index])
    parallel = transform.scale**2 * uncertainty.parallel_variance[local_index]
    lateral = transform.scale**2 * uncertainty.lateral_variance[local_index]
    outer = np.einsum("...i,...j->...ij", rays, rays)
    return lateral[..., None, None] * np.eye(3) + (
        parallel - lateral
    )[..., None, None] * outer

def _gauge_induced_covariance(
    values: FloatArray,
    transform: Sim3,
    gauge_covariance: FloatArray,
    *,
    include_translation: bool,
    chunk_size: int = 16_384,
) -> FloatArray:
    """Propagate a seven-dimensional Sim(3) covariance into vectors or points."""

    values = np.asarray(values, dtype=np.float64)
    covariance = np.asarray(gauge_covariance, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("gauge covariance propagation requires three-dimensional values")
    if covariance.shape != (7, 7):
        raise ValueError("gauge covariance must have shape (7, 7)")
    covariance = 0.5 * (covariance + covariance.T)
    flattened = values.reshape(-1, 3)
    propagated = np.empty((flattened.shape[0], 3, 3), dtype=np.float64)
    identity = np.eye(3)
    right_jacobian = so3_right_jacobian(so3_log(transform.rotation))
    for start in range(0, flattened.shape[0], chunk_size):
        stop = min(start + chunk_size, flattened.shape[0])
        chunk = flattened[start:stop]
        scaled_rotated = transform.scale * np.einsum("ij,nj->ni", transform.rotation, chunk)
        skew_matrices = np.zeros((chunk.shape[0], 3, 3), dtype=np.float64)
        skew_matrices[:, 0, 1] = -chunk[:, 2]
        skew_matrices[:, 0, 2] = chunk[:, 1]
        skew_matrices[:, 1, 0] = chunk[:, 2]
        skew_matrices[:, 1, 2] = -chunk[:, 0]
        skew_matrices[:, 2, 0] = -chunk[:, 1]
        skew_matrices[:, 2, 1] = chunk[:, 0]
        jacobian = np.zeros((chunk.shape[0], 3, 7), dtype=np.float64)
        jacobian[:, :, 0] = scaled_rotated
        jacobian[:, :, 1:4] = -transform.scale * np.einsum(
            "ij,njk,kl->nil", transform.rotation, skew_matrices, right_jacobian
        )
        if include_translation:
            jacobian[:, :, 4:7] = identity
        propagated[start:stop] = np.einsum(
            "nij,jk,nlk->nil", jacobian, covariance, jacobian
        )
    return propagated.reshape(values.shape + (3,))



def fuse_windows(
    windows: list[PredictionWindow],
    gauges: dict[str, Sim3],
    point_uncertainties: dict[str, StructuredCovariance],
    *,
    method: FusionMethod,
    flow_uncertainties: dict[str, StructuredCovariance] | None = None,
    gauge_covariances: dict[str, FloatArray] | None = None,
) -> FusedSequence:
    """Transform and jointly fuse duplicate pixels in canonical window order.

    Every frame/mask pattern is fused in one batch. Uniform and independent
    precision fusion use their exact multi-input formulas; covariance intersection
    solves one generalized simplex problem for all contributors.
    """

    if not windows:
        raise ValueError("at least one prediction window is required")
    if len({window.window_id for window in windows}) != len(windows):
        raise ValueError("prediction window IDs must be unique")
    ordered_windows = sorted(
        windows,
        key=lambda window: (
            window.start_frame,
            window.stop_frame,
            window.window_id,
        ),
    )
    height, width = ordered_windows[0].shape[1:]
    if any(window.shape[1:] != (height, width) for window in ordered_windows):
        raise ValueError("all windows must use the same spatial resolution")
    if method not in {"uniform", "precision", "covariance_intersection"}:
        raise ValueError(f"unknown fusion method {method!r}")
    for window in ordered_windows:
        if window.window_id not in gauges or window.window_id not in point_uncertainties:
            raise KeyError(f"missing gauge or uncertainty for window {window.window_id!r}")
        if gauge_covariances is not None and window.window_id not in gauge_covariances:
            raise KeyError(f"missing gauge covariance for window {window.window_id!r}")
        if (
            window.scene_flow is not None
            and flow_uncertainties is not None
            and window.window_id not in flow_uncertainties
        ):
            raise KeyError(f"missing flow uncertainty for window {window.window_id!r}")

    all_frames = np.unique(
        np.concatenate([window.frame_indices for window in ordered_windows])
    )
    shape = (all_frames.size, height, width)
    point_map = np.zeros(shape + (3,), dtype=np.float64)
    valid_mask = np.zeros(shape, dtype=bool)
    point_covariance = np.zeros(shape + (3, 3), dtype=np.float64)
    contributors = np.zeros(shape, dtype=np.uint16)
    has_flow = any(window.scene_flow is not None for window in ordered_windows)
    scene_flow = np.zeros_like(point_map) if has_flow else None
    deform_mask = np.zeros(shape, dtype=bool) if has_flow else None
    flow_covariance = np.zeros_like(point_covariance) if has_flow else None

    for output_index, frame in enumerate(all_frames):
        point_means: list[FloatArray] = []
        point_covariances: list[FloatArray] = []
        point_masks: list[NDArray[np.bool_]] = []
        flow_means: list[FloatArray] = []
        flow_covariances: list[FloatArray] = []
        flow_masks: list[NDArray[np.bool_]] = []
        for window in ordered_windows:
            try:
                local_index = window.local_index(int(frame))
            except KeyError:
                continue
            gauge = gauges[window.window_id]
            local_points = window.point_map[local_index]
            transformed_points = gauge.transform_points(local_points)
            transformed_point_covariance = _structured_covariance_frame(
                point_uncertainties[window.window_id],
                gauge,
                local_index,
            )
            if gauge_covariances is not None:
                transformed_point_covariance += _gauge_induced_covariance(
                    local_points,
                    gauge,
                    gauge_covariances[window.window_id],
                    include_translation=True,
                )
            point_means.append(transformed_points)
            point_covariances.append(transformed_point_covariance)
            point_masks.append(window.valid_mask[local_index])

            if window.scene_flow is None:
                continue
            uncertainty = (
                flow_uncertainties[window.window_id]
                if flow_uncertainties is not None
                else point_uncertainties[window.window_id]
            )
            local_flow = window.scene_flow[local_index]
            transformed_flow = gauge.transform_vectors(local_flow)
            transformed_flow_covariance = _structured_covariance_frame(
                uncertainty,
                gauge,
                local_index,
            )
            if gauge_covariances is not None:
                transformed_flow_covariance += _gauge_induced_covariance(
                    local_flow,
                    gauge,
                    gauge_covariances[window.window_id],
                    include_translation=False,
                )
            flow_means.append(transformed_flow)
            flow_covariances.append(transformed_flow_covariance)
            flow_masks.append(window.deform_mask[local_index])

        (
            point_map[output_index],
            point_covariance[output_index],
            valid_mask[output_index],
            contributors[output_index],
        ) = _fuse_frame_fields(
            point_means,
            point_covariances,
            point_masks,
            method=method,
        )
        if flow_means:
            (
                scene_flow[output_index],
                flow_covariance[output_index],
                deform_mask[output_index],
                _,
            ) = _fuse_frame_fields(
                flow_means,
                flow_covariances,
                flow_masks,
                method=method,
            )

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
