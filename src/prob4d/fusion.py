"""Decoded-space fusion of aligned MotionCrafter predictions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .covariance import regularized_inverse_psd, validated_covariance_psd
from .data import PredictionWindow
from .sim3 import Sim3, so3_log, so3_right_jacobian
from .uncertainty import StructuredCovariance

FloatArray = NDArray[np.floating]
FusionMethod = Literal["uniform", "precision", "covariance_intersection"]
DEFAULT_FUSION_TILE_SIZE = 16_384
DEFAULT_CI_WEIGHT_SAMPLE_SIZE = 4_096
FrameFieldLoader = Callable[
    [int, NDArray[np.int64]],
    tuple[FloatArray, FloatArray],
]


@dataclass(frozen=True)
class FusedSequence:
    """A validated immutable global-gauge dense sequence.

    The public constructor defensively copies every NumPy field, normalizes it to
    the canonical dtype, and makes it read-only. Point and flow covariance
    matrices are validated as symmetric positive semidefinite on their active
    geometry. Inactive payload entries are preserved because historical artifacts
    may use arbitrary sentinels outside their masks.
    """

    frame_indices: NDArray[np.integer]
    point_map: FloatArray
    valid_mask: NDArray[np.bool_]
    point_covariance: FloatArray
    contributors: NDArray[np.integer]
    scene_flow: FloatArray | None = None
    deform_mask: NDArray[np.bool_] | None = None
    flow_covariance: FloatArray | None = None

    def __post_init__(self) -> None:
        self._validate_and_store(copy_arrays=True)

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        frame_indices: NDArray[np.integer],
        point_map: FloatArray,
        valid_mask: NDArray[np.bool_],
        point_covariance: FloatArray,
        contributors: NDArray[np.integer],
        scene_flow: FloatArray | None = None,
        deform_mask: NDArray[np.bool_] | None = None,
        flow_covariance: FloatArray | None = None,
    ) -> FusedSequence:
        """Adopt private producer arrays without a second dense defensive copy.

        The caller transfers ownership: accepted arrays are validated in place and
        made read-only. This private path is used only where Prob4D allocated all
        payloads locally and no external mutable aliases can survive the return.
        """

        instance = object.__new__(cls)
        object.__setattr__(instance, "frame_indices", frame_indices)
        object.__setattr__(instance, "point_map", point_map)
        object.__setattr__(instance, "valid_mask", valid_mask)
        object.__setattr__(instance, "point_covariance", point_covariance)
        object.__setattr__(instance, "contributors", contributors)
        object.__setattr__(instance, "scene_flow", scene_flow)
        object.__setattr__(instance, "deform_mask", deform_mask)
        object.__setattr__(instance, "flow_covariance", flow_covariance)
        instance._validate_and_store(copy_arrays=False)
        return instance

    def _validate_and_store(self, *, copy_arrays: bool) -> None:
        frames = _canonical_array(self.frame_indices, dtype=np.int64, copy=copy_arrays)
        points = _canonical_array(self.point_map, dtype=np.float64, copy=copy_arrays)
        mask = _canonical_array(self.valid_mask, dtype=bool, copy=copy_arrays)
        raw_contributors = np.asarray(self.contributors)

        if frames.ndim != 1 or frames.size == 0:
            raise ValueError("frame_indices must be a non-empty one-dimensional array")
        if np.any(frames < 0):
            raise ValueError("frame_indices must be non-negative")
        if np.any(np.diff(frames) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("point_map must have shape (T, H, W, 3)")
        if mask.shape != points.shape[:-1]:
            raise ValueError("valid_mask must have shape (T, H, W)")
        if points.shape[0] != frames.size:
            raise ValueError("frame_indices must match the sequence time dimension")
        _validate_finite_active_vectors(
            points,
            active_mask=mask,
            name="valid point_map",
        )

        if raw_contributors.shape != mask.shape:
            raise ValueError("contributors must have shape (T, H, W)")
        if not np.issubdtype(raw_contributors.dtype, np.integer):
            raise ValueError("contributors must contain integers")
        contributor_values = np.asarray(raw_contributors, dtype=np.int64)
        if np.any(contributor_values < 0):
            raise ValueError("contributors must be non-negative")
        if np.any(contributor_values > np.iinfo(np.uint16).max):
            raise ValueError("contributors exceed the uint16 storage range")
        if np.any(mask & (contributor_values == 0)):
            raise ValueError("valid points must have at least one contributor")
        contributors = _canonical_array(
            raw_contributors,
            dtype=np.uint16,
            copy=copy_arrays,
        )

        covariance = _validated_covariance_field(
            self.point_covariance,
            expected_shape=points.shape + (3,),
            active_mask=mask,
            name="point_covariance",
            copy=copy_arrays,
        )

        flow_values = (self.scene_flow, self.deform_mask, self.flow_covariance)
        present = tuple(value is not None for value in flow_values)
        if any(present) and not all(present):
            raise ValueError(
                "scene_flow, deform_mask, and flow_covariance must all be present or absent"
            )

        flow: np.ndarray | None = None
        flow_mask: np.ndarray | None = None
        flow_covariance: np.ndarray | None = None
        if all(present):
            assert self.scene_flow is not None
            assert self.deform_mask is not None
            assert self.flow_covariance is not None
            flow = _canonical_array(
                self.scene_flow,
                dtype=np.float64,
                copy=copy_arrays,
            )
            flow_mask = _canonical_array(
                self.deform_mask,
                dtype=bool,
                copy=copy_arrays,
            )
            if flow.shape != points.shape or flow_mask.shape != mask.shape:
                raise ValueError("scene-flow arrays must match point-map shape")
            active_flow = flow_mask & mask
            _validate_finite_active_vectors(
                flow,
                active_mask=active_flow,
                name="active scene_flow",
            )
            flow_covariance = _validated_covariance_field(
                self.flow_covariance,
                expected_shape=covariance.shape,
                active_mask=active_flow,
                name="flow_covariance",
                copy=copy_arrays,
            )

        object.__setattr__(self, "frame_indices", _readonly_owned(frames))
        object.__setattr__(self, "point_map", _readonly_owned(points))
        object.__setattr__(self, "valid_mask", _readonly_owned(mask))
        object.__setattr__(self, "point_covariance", _readonly_owned(covariance))
        object.__setattr__(self, "contributors", _readonly_owned(contributors))
        object.__setattr__(
            self,
            "scene_flow",
            None if flow is None else _readonly_owned(flow),
        )
        object.__setattr__(
            self,
            "deform_mask",
            None if flow_mask is None else _readonly_owned(flow_mask),
        )
        object.__setattr__(
            self,
            "flow_covariance",
            None if flow_covariance is None else _readonly_owned(flow_covariance),
        )


def _canonical_array(
    value: ArrayLike,
    *,
    dtype: DTypeLike,
    copy: bool,
) -> np.ndarray:
    """Normalize an array and optionally make a defensive owned copy."""

    array = np.asarray(value, dtype=dtype)
    return array.copy() if copy else array


def _readonly_owned(value: np.ndarray) -> np.ndarray:
    """Freeze an array that is already an owned defensive copy."""

    value.setflags(write=False)
    return value


def _validate_finite_active_vectors(
    values: np.ndarray,
    *,
    active_mask: NDArray[np.bool_],
    name: str,
    chunk_size: int = 65_536,
) -> None:
    """Validate active three-vectors without materializing one full masked copy."""

    if chunk_size < 1:
        raise ValueError("vector validation chunk_size must be positive")
    flat_values = values.reshape(-1, values.shape[-1])
    flat_active = np.asarray(active_mask, dtype=bool).reshape(-1)
    for start in range(0, flat_values.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_values.shape[0])
        selected = flat_active[start:stop]
        if np.any(selected) and not np.all(np.isfinite(flat_values[start:stop][selected])):
            raise ValueError(f"{name} entries must be finite")


def _validated_covariance_field(
    value: FloatArray | None,
    *,
    expected_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_],
    name: str,
    copy: bool,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Validate active dense 3-D covariances without a full-field eigensolve.

    Positive semidefiniteness of a symmetric 3x3 matrix is equivalent to all
    principal minors being non-negative. The fast path checks those minors in
    bounded chunks. Only matrices with a negative principal minor enter the
    scale-aware eigendecomposition used by :func:`validated_covariance_psd`, so
    the common production path remains linear and memory bounded while numerical
    near-boundary cases retain the repository-wide tolerance semantics.
    """

    if value is None:
        raise ValueError(f"{name} is required")
    if chunk_size < 1:
        raise ValueError("covariance validation chunk_size must be positive")
    covariance = _canonical_array(value, dtype=np.float64, copy=copy)
    if covariance.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")

    flat_covariance = covariance.reshape(-1, 3, 3)
    flat_active = np.asarray(active_mask, dtype=bool).reshape(-1)
    for start in range(0, flat_covariance.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_covariance.shape[0])
        selected = flat_active[start:stop]
        if not np.any(selected):
            continue
        chunk = flat_covariance[start:stop]
        matrices = chunk[selected]
        if not np.all(np.isfinite(matrices)):
            raise ValueError(f"active {name} entries must be finite")

        transposed = np.swapaxes(matrices, -1, -2)
        symmetric = 0.5 * (matrices + transposed)
        if not np.allclose(matrices, symmetric, atol=1e-12, rtol=1e-10):
            raise ValueError(f"active {name} matrices must be symmetric")

        a = symmetric[:, 0, 0]
        b = symmetric[:, 0, 1]
        c = symmetric[:, 0, 2]
        d = symmetric[:, 1, 1]
        e = symmetric[:, 1, 2]
        f = symmetric[:, 2, 2]
        suspicious = (
            (a < 0.0)
            | (d < 0.0)
            | (f < 0.0)
            | (a * d - b * b < 0.0)
            | (a * f - c * c < 0.0)
            | (d * f - e * e < 0.0)
            | (
                a * d * f
                + 2.0 * b * c * e
                - a * e * e
                - d * c * c
                - f * b * b
                < 0.0
            )
        )
        if np.any(suspicious):
            symmetric[suspicious] = validated_covariance_psd(
                symmetric[suspicious],
                name=f"active {name}",
                readonly=False,
            )
        chunk[selected] = symmetric
    return covariance


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


def _covariance_intersection_weights(
    means: FloatArray,
    covariances: FloatArray,
) -> FloatArray:
    """Optimize one global CI weight vector for a bounded representative sample."""

    contributor_count = means.shape[0]
    if contributor_count < 2:
        return np.ones(1, dtype=np.float64)
    if contributor_count == 2:
        _, _, first_weight = fuse_gaussians_covariance_intersection(
            means[0],
            covariances[0],
            means[1],
            covariances[1],
            weight_sample_size=means.shape[1],
        )
        value = float(np.ravel(first_weight)[0])
        return np.asarray([value, 1.0 - value], dtype=np.float64)

    from ._generalized_ci import fuse_nway_covariance_intersection

    _, _, weights = fuse_nway_covariance_intersection(
        means,
        covariances,
        weight_sample_size=means.shape[1],
        canonicalize=False,
    )
    return weights


def _fuse_contributor_stack_fixed_ci(
    means: FloatArray,
    covariances: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply already optimized CI weights to one bounded spatial tile."""

    contributor_count = means.shape[0]
    normalized_weights = np.asarray(weights, dtype=np.float64)
    if normalized_weights.shape != (contributor_count,):
        raise ValueError("CI weights must match the contributor count")
    if contributor_count == 1:
        return means[0], covariances[0]

    information = _regularized_inverse(covariances)
    if contributor_count == 2:
        first_weight = float(normalized_weights[0])
        second_weight = float(normalized_weights[1])
        fused_covariance = _regularized_inverse(
            first_weight * information[0] + second_weight * information[1]
        )
        information_vector = first_weight * np.einsum(
            "nij,nj->ni", information[0], means[0]
        ) + second_weight * np.einsum(
            "nij,nj->ni", information[1], means[1]
        )
    else:
        fused_covariance = _regularized_inverse(
            np.einsum(
                "k,knij->nij",
                normalized_weights,
                information,
                optimize=True,
            )
        )
        information_vector = np.einsum(
            "k,knij,knj->ni",
            normalized_weights,
            information,
            means,
            optimize=True,
        )
    fused_mean = np.einsum(
        "nij,nj->ni",
        fused_covariance,
        information_vector,
        optimize=True,
    )
    return fused_mean, fused_covariance


def _representative_positions(sample_count: int) -> NDArray[np.int64]:
    """Match the existing deterministic CI sampling policy without dense copies."""

    if sample_count < 1:
        raise ValueError("CI weight optimization requires at least one sample")
    if sample_count <= DEFAULT_CI_WEIGHT_SAMPLE_SIZE:
        return np.arange(sample_count, dtype=np.int64)
    return np.linspace(
        0,
        sample_count - 1,
        DEFAULT_CI_WEIGHT_SAMPLE_SIZE,
        dtype=np.int64,
    )


def _validated_loaded_tile(
    loader: FrameFieldLoader,
    contributor_index: int,
    indices: NDArray[np.int64],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one contributor tile and validate its compact dense shapes."""

    means, covariances = loader(contributor_index, indices)
    mean_array = np.asarray(means, dtype=np.float64)
    covariance_array = np.asarray(covariances, dtype=np.float64)
    expected_mean_shape = (indices.size, 3)
    expected_covariance_shape = (indices.size, 3, 3)
    if mean_array.shape != expected_mean_shape:
        raise ValueError(f"frame tile means must have shape {expected_mean_shape}")
    if covariance_array.shape != expected_covariance_shape:
        raise ValueError(
            f"frame tile covariances must have shape {expected_covariance_shape}"
        )
    return mean_array, covariance_array


def _fuse_frame_field_loader(
    masks: list[NDArray[np.bool_]],
    *,
    method: FusionMethod,
    loader: FrameFieldLoader,
    tile_size: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], NDArray[np.uint16]]:
    """Fuse one frame from a tile loader while retaining frame-global CI weights."""

    if tile_size < 1:
        raise ValueError("fusion tile_size must be positive")
    if not masks:
        raise ValueError("frame fusion requires at least one contributor")
    if len(masks) > np.iinfo(np.uint16).max:
        raise ValueError("frame fusion contributor count exceeds uint16 storage")

    normalized_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    first_shape = normalized_masks[0].shape
    if len(first_shape) != 2:
        raise ValueError("frame masks must have shape (H, W)")
    if any(mask.shape != first_shape for mask in normalized_masks):
        raise ValueError(f"frame masks must have shape {first_shape}")

    stacked_masks = np.stack(normalized_masks)
    contributor_count, height, width = stacked_masks.shape
    active_rows = stacked_masks.reshape(contributor_count, -1).T
    patterns, inverse = np.unique(active_rows, axis=0, return_inverse=True)
    output_mean = np.zeros((height * width, 3), dtype=np.float64)
    output_covariance = np.zeros((height * width, 3, 3), dtype=np.float64)
    output_count = np.sum(active_rows, axis=1, dtype=np.uint16)

    for pattern_index, pattern in enumerate(patterns):
        active = np.flatnonzero(pattern)
        if active.size == 0:
            continue
        selected_indices = np.flatnonzero(inverse == pattern_index)

        ci_weights: FloatArray | None = None
        if method == "covariance_intersection" and active.size > 1:
            sample_positions = _representative_positions(selected_indices.size)
            sample_indices = selected_indices[sample_positions]
            loaded_sample = [
                _validated_loaded_tile(loader, int(index), sample_indices)
                for index in active
            ]
            sample_means = np.stack([values[0] for values in loaded_sample])
            sample_covariances = np.stack([values[1] for values in loaded_sample])
            ci_weights = _covariance_intersection_weights(
                sample_means,
                sample_covariances,
            )

        for start in range(0, selected_indices.size, tile_size):
            stop = min(start + tile_size, selected_indices.size)
            tile_indices = selected_indices[start:stop]
            loaded_tile = [
                _validated_loaded_tile(loader, int(index), tile_indices)
                for index in active
            ]
            tile_means = np.stack([values[0] for values in loaded_tile])
            tile_covariances = np.stack([values[1] for values in loaded_tile])
            if ci_weights is None:
                fused_mean, fused_covariance = _fuse_contributor_stack(
                    tile_means,
                    tile_covariances,
                    method=method,
                )
            else:
                fused_mean, fused_covariance = _fuse_contributor_stack_fixed_ci(
                    tile_means,
                    tile_covariances,
                    ci_weights,
                )
            output_mean[tile_indices] = fused_mean
            output_covariance[tile_indices] = fused_covariance

    valid = output_count > 0
    return (
        output_mean.reshape(height, width, 3),
        output_covariance.reshape(height, width, 3, 3),
        valid.reshape(height, width),
        output_count.reshape(height, width),
    )


def _fuse_frame_fields(
    means: list[FloatArray],
    covariances: list[FloatArray],
    masks: list[NDArray[np.bool_]],
    *,
    method: FusionMethod,
    tile_size: int = DEFAULT_FUSION_TILE_SIZE,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], NDArray[np.uint16]]:
    """Fuse already materialized frame fields in bounded spatial tiles."""

    if not means or len(means) != len(covariances) or len(means) != len(masks):
        raise ValueError("frame fusion requires matching nonempty contributor lists")

    first_mask = np.asarray(masks[0], dtype=bool)
    if first_mask.ndim != 2:
        raise ValueError("frame masks must have shape (H, W)")
    height, width = first_mask.shape
    expected_mean_shape = (height, width, 3)
    expected_covariance_shape = (height, width, 3, 3)
    normalized_means: list[np.ndarray] = []
    normalized_covariances: list[np.ndarray] = []
    for mean, covariance in zip(means, covariances, strict=True):
        mean_array = np.asarray(mean, dtype=np.float64)
        covariance_array = np.asarray(covariance, dtype=np.float64)
        if mean_array.shape != expected_mean_shape:
            raise ValueError(f"frame means must have shape {expected_mean_shape}")
        if covariance_array.shape != expected_covariance_shape:
            raise ValueError(
                f"frame covariances must have shape {expected_covariance_shape}"
            )
        normalized_means.append(mean_array.reshape(-1, 3))
        normalized_covariances.append(covariance_array.reshape(-1, 3, 3))

    def loader(
        contributor_index: int,
        indices: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray]:
        return (
            normalized_means[contributor_index][indices],
            normalized_covariances[contributor_index][indices],
        )

    return _fuse_frame_field_loader(
        masks,
        method=method,
        loader=loader,
        tile_size=tile_size,
    )


@dataclass(frozen=True)
class _WindowFrameField:
    """One local point or flow field with lazily materialized world covariance."""

    values: FloatArray
    uncertainty: StructuredCovariance
    transform: Sim3
    local_index: int
    gauge_covariance: FloatArray | None
    include_translation: bool


def _structured_covariance_rows(
    uncertainty: StructuredCovariance,
    transform: Sim3,
    local_index: int,
    flat_indices: NDArray[np.int64],
) -> FloatArray:
    """Expand structured covariance only for selected spatial rows."""

    rays = uncertainty.ray_directions[local_index].reshape(-1, 3)[flat_indices]
    rays = transform.rotate_directions(rays)
    parallel = (
        transform.scale**2
        * uncertainty.parallel_variance[local_index].reshape(-1)[flat_indices]
    )
    lateral = (
        transform.scale**2
        * uncertainty.lateral_variance[local_index].reshape(-1)[flat_indices]
    )
    outer = np.einsum("ni,nj->nij", rays, rays)
    return lateral[:, None, None] * np.eye(3) + (
        parallel - lateral
    )[:, None, None] * outer


def _load_window_frame_tile(
    field: _WindowFrameField,
    flat_indices: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray]:
    """Transform one compact tile and add local plus gauge-induced covariance."""

    local_values = np.asarray(field.values).reshape(-1, 3)[flat_indices]
    transformed = (
        field.transform.transform_points(local_values)
        if field.include_translation
        else field.transform.transform_vectors(local_values)
    )
    covariance = _structured_covariance_rows(
        field.uncertainty,
        field.transform,
        field.local_index,
        flat_indices,
    )
    if field.gauge_covariance is not None:
        covariance += _gauge_induced_covariance(
            local_values,
            field.transform,
            field.gauge_covariance,
            include_translation=field.include_translation,
        )
    return transformed, covariance


def _fuse_window_frame_fields(
    fields: list[_WindowFrameField],
    masks: list[NDArray[np.bool_]],
    *,
    method: FusionMethod,
    tile_size: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], NDArray[np.uint16]]:
    """Fuse local window fields without materializing contributor-sized matrices."""

    if len(fields) != len(masks) or not fields:
        raise ValueError("window-frame fusion requires matching nonempty fields and masks")

    def loader(
        contributor_index: int,
        indices: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray]:
        return _load_window_frame_tile(fields[contributor_index], indices)

    return _fuse_frame_field_loader(
        masks,
        method=method,
        loader=loader,
        tile_size=tile_size,
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
    fusion_tile_size: int = DEFAULT_FUSION_TILE_SIZE,
) -> FusedSequence:
    """Transform and jointly fuse duplicate pixels in canonical window order.

    Uniform and independent precision fusion use their exact multi-input formulas;
    covariance intersection solves one generalized simplex problem for every full
    frame/mask pattern. Dense application is then processed in bounded spatial
    tiles without changing those global weights.
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
    if fusion_tile_size < 1:
        raise ValueError("fusion_tile_size must be positive")
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
        point_fields: list[_WindowFrameField] = []
        point_masks: list[NDArray[np.bool_]] = []
        flow_fields: list[_WindowFrameField] = []
        flow_masks: list[NDArray[np.bool_]] = []
        for window in ordered_windows:
            try:
                local_index = window.local_index(int(frame))
            except KeyError:
                continue
            gauge = gauges[window.window_id]
            gauge_covariance = (
                None
                if gauge_covariances is None
                else gauge_covariances[window.window_id]
            )
            point_fields.append(
                _WindowFrameField(
                    values=window.point_map[local_index],
                    uncertainty=point_uncertainties[window.window_id],
                    transform=gauge,
                    local_index=local_index,
                    gauge_covariance=gauge_covariance,
                    include_translation=True,
                )
            )
            point_masks.append(window.valid_mask[local_index])

            if window.scene_flow is None:
                continue
            uncertainty = (
                flow_uncertainties[window.window_id]
                if flow_uncertainties is not None
                else point_uncertainties[window.window_id]
            )
            flow_fields.append(
                _WindowFrameField(
                    values=window.scene_flow[local_index],
                    uncertainty=uncertainty,
                    transform=gauge,
                    local_index=local_index,
                    gauge_covariance=gauge_covariance,
                    include_translation=False,
                )
            )
            assert window.deform_mask is not None
            flow_masks.append(window.deform_mask[local_index])

        (
            point_map[output_index],
            point_covariance[output_index],
            valid_mask[output_index],
            contributors[output_index],
        ) = _fuse_window_frame_fields(
            point_fields,
            point_masks,
            method=method,
            tile_size=fusion_tile_size,
        )
        if flow_fields:
            assert scene_flow is not None
            assert flow_covariance is not None
            assert deform_mask is not None
            (
                scene_flow[output_index],
                flow_covariance[output_index],
                deform_mask[output_index],
                _,
            ) = _fuse_window_frame_fields(
                flow_fields,
                flow_masks,
                method=method,
                tile_size=fusion_tile_size,
            )

    return FusedSequence._from_owned_arrays(
        frame_indices=all_frames,
        point_map=point_map,
        valid_mask=valid_mask,
        point_covariance=point_covariance,
        contributors=contributors,
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        flow_covariance=flow_covariance,
    )
