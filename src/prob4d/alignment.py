"""Robust relative-gauge estimation from overlapping prediction windows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from ._scientific_scalars import require_finite_real, require_genuine_integer
from .data import PredictionWindow
from .sim3 import Sim3, skew, so3_log, so3_right_jacobian

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
CovarianceFallbackPolicy = Literal["error", "pointwise"]

DEFAULT_COVARIANCE_CLUSTER_SIZE = 32
DENSE_ALIGNMENT_COVARIANCE_METHOD = "frame_spatial_cluster_robust_v1"
POINTWISE_COVARIANCE_FALLBACK = "insufficient_spatial_clusters_pointwise_v1"
IID_COVARIANCE_FALLBACK = "insufficient_spatial_clusters_iid_v1"


class AlignmentCovarianceCalibration(Protocol):
    """Structural contract used by the provider without importing artifact types."""

    @property
    def artifact_id(self) -> str: ...

    def apply(self, covariance: FloatArray) -> FloatArray: ...


@dataclass
class AlignmentCovarianceDiagnostics:
    """Task-local audit of covariance calibration and approximation use."""

    alignment_count: int = 0
    calibrated_alignment_count: int = 0
    covariance_fallbacks: list[str] = field(default_factory=list)

    def record(self, result: AlignmentResult) -> None:
        self.alignment_count += 1
        if result.covariance_calibration_id is not None:
            self.calibrated_alignment_count += 1
        if result.covariance_fallback is not None:
            self.covariance_fallbacks.append(result.covariance_fallback)

    @property
    def fallback_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.covariance_fallbacks).items()))


@dataclass(frozen=True)
class _AlignmentCovarianceContext:
    calibration: AlignmentCovarianceCalibration | None
    fallback_policy: CovarianceFallbackPolicy
    diagnostics: AlignmentCovarianceDiagnostics | None


_DEFAULT_ALIGNMENT_COVARIANCE_CONTEXT = _AlignmentCovarianceContext(
    calibration=None,
    fallback_policy="pointwise",
    diagnostics=None,
)
_ALIGNMENT_COVARIANCE_CONTEXT: ContextVar[_AlignmentCovarianceContext] = ContextVar(
    "prob4d_alignment_covariance_context",
    default=_DEFAULT_ALIGNMENT_COVARIANCE_CONTEXT,
)


@contextmanager
def alignment_covariance_context(
    *,
    calibration: AlignmentCovarianceCalibration | None = None,
    fallback_policy: CovarianceFallbackPolicy = "error",
) -> Iterator[AlignmentCovarianceDiagnostics]:
    """Apply one task-local calibration and covariance-fallback policy.

    The stable provider enters this context around an export. Direct low-level
    alignment calls retain the historical pointwise fallback unless they opt in.
    Context-local state avoids process-global mutation and is safe for concurrent
    tasks and threads with independently copied contexts.
    """

    if fallback_policy not in {"error", "pointwise"}:
        raise ValueError("fallback_policy must be 'error' or 'pointwise'")
    diagnostics = AlignmentCovarianceDiagnostics()
    token = _ALIGNMENT_COVARIANCE_CONTEXT.set(
        _AlignmentCovarianceContext(
            calibration=calibration,
            fallback_policy=fallback_policy,
            diagnostics=diagnostics,
        )
    )
    try:
        yield diagnostics
    finally:
        _ALIGNMENT_COVARIANCE_CONTEXT.reset(token)


@dataclass(frozen=True)
class AlignmentResult:
    """A robust estimate mapping source points into target coordinates."""

    transform: Sim3
    covariance: FloatArray
    residual_rms: float
    inlier_fraction: float
    num_correspondences: int
    covariance_method: str = "iid_gauss_newton"
    num_covariance_clusters: int = 0
    information_rank: int = 7
    information_condition: float = 1.0
    covariance_calibration_id: str | None = None
    covariance_fallback: str | None = None

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        if covariance.shape != (7, 7):
            raise ValueError("alignment covariance must have shape (7, 7)")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("alignment covariance must be finite")
        symmetric = 0.5 * (covariance + covariance.T)
        if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
            raise ValueError("alignment covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(symmetric)) < -1e-10:
            raise ValueError("alignment covariance must be positive semidefinite")
        residual_rms = require_finite_real(
            self.residual_rms,
            name="residual_rms",
            minimum=0.0,
        )
        inlier_fraction = require_finite_real(
            self.inlier_fraction,
            name="inlier_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        information_condition = require_finite_real(
            self.information_condition,
            name="information_condition",
            minimum=0.0,
            minimum_inclusive=False,
        )
        num_correspondences = require_genuine_integer(
            self.num_correspondences,
            name="num_correspondences",
            minimum=0,
        )
        if not self.covariance_method:
            raise ValueError("covariance_method must be nonempty")
        num_covariance_clusters = require_genuine_integer(
            self.num_covariance_clusters,
            name="num_covariance_clusters",
            minimum=0,
        )
        information_rank = require_genuine_integer(
            self.information_rank,
            name="information_rank",
            minimum=0,
            maximum=7,
        )
        calibration_id = self.covariance_calibration_id
        if calibration_id is not None and (
            len(calibration_id) != 64
            or any(character not in "0123456789abcdef" for character in calibration_id)
        ):
            raise ValueError("covariance_calibration_id must be a lowercase SHA-256 digest")
        fallback = self.covariance_fallback
        if fallback is not None and not fallback:
            raise ValueError("covariance_fallback must be nonempty when supplied")
        symmetric.setflags(write=False)
        object.__setattr__(self, "covariance", symmetric)
        object.__setattr__(self, "residual_rms", residual_rms)
        object.__setattr__(self, "inlier_fraction", inlier_fraction)
        object.__setattr__(self, "num_correspondences", num_correspondences)
        object.__setattr__(
            self,
            "num_covariance_clusters",
            num_covariance_clusters,
        )
        object.__setattr__(self, "information_rank", information_rank)
        object.__setattr__(self, "information_condition", information_condition)


@dataclass(frozen=True)
class WindowAlignment:
    """Relative transform from ``moving_id`` into ``reference_id`` coordinates."""

    reference_id: str
    moving_id: str
    common_frames: NDArray[np.integer]
    result: AlignmentResult

    def __post_init__(self) -> None:
        reference_id = str(self.reference_id)
        moving_id = str(self.moving_id)
        frames = np.asarray(self.common_frames, dtype=np.int64).copy()
        if not reference_id or not moving_id:
            raise ValueError("alignment window IDs must be nonempty")
        if reference_id == moving_id:
            raise ValueError("alignment window IDs must be distinct")
        if (
            frames.ndim != 1
            or frames.size == 0
            or np.any(frames < 0)
            or np.any(np.diff(frames) <= 0)
        ):
            raise ValueError(
                "common_frames must be nonempty, nonnegative, and strictly increasing"
            )
        frames.setflags(write=False)
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "moving_id", moving_id)
        object.__setattr__(self, "common_frames", frames)


@dataclass(frozen=True)
class _CovarianceEstimate:
    covariance: FloatArray
    method: str
    num_clusters: int
    information_rank: int
    information_condition: float


def _weighted_umeyama(source: FloatArray, target: FloatArray, weights: FloatArray) -> Sim3:
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= np.finfo(np.float64).eps:
        raise ValueError("alignment weights have zero total mass")
    normalized = weights / weight_sum
    source_mean = np.sum(normalized[:, None] * source, axis=0)
    target_mean = np.sum(normalized[:, None] * target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (normalized[:, None] * target_centered).T @ source_centered
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ correction @ right_transpose
    source_variance = float(np.sum(normalized * np.sum(source_centered**2, axis=1)))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("source correspondences have no spatial extent")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("estimated alignment scale is not positive")
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def _parameter_jacobian(
    point: FloatArray,
    transform: Sim3,
    rotation_jacobian: FloatArray,
) -> FloatArray:
    scaled_rotated = transform.scale * (transform.rotation @ point)
    jacobian = np.empty((3, 7), dtype=np.float64)
    jacobian[:, 0] = scaled_rotated
    jacobian[:, 1:4] = (
        -transform.scale * transform.rotation @ skew(point) @ rotation_jacobian
    )
    jacobian[:, 4:7] = np.eye(3)
    return jacobian


def _alignment_covariance_estimate(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    transform: Sim3,
    *,
    cluster_ids: IntArray | None = None,
) -> _CovarianceEstimate:
    transformed = transform.transform_points(source)
    residuals = target - transformed
    information = np.zeros((7, 7), dtype=np.float64)
    rotation_jacobian = so3_right_jacobian(so3_log(transform.rotation))
    active_mask = weights > 1e-3
    active = int(np.count_nonzero(active_mask))

    cluster_inverse: IntArray | None = None
    cluster_scores: FloatArray | None = None
    num_clusters = active
    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids)
        if clusters.shape != (source.shape[0],):
            raise ValueError("cluster_ids must have shape (N,)")
        _, compact = np.unique(clusters[active_mask], return_inverse=True)
        num_clusters = int(np.max(compact) + 1) if compact.size else 0
        if num_clusters <= 7:
            raise ValueError(
                "cluster-robust covariance requires at least eight active clusters"
            )
        cluster_inverse = np.full(source.shape[0], -1, dtype=np.int64)
        cluster_inverse[active_mask] = compact
        cluster_scores = np.zeros((num_clusters, 7), dtype=np.float64)

    for index, (point, residual, weight) in enumerate(
        zip(source, residuals, weights, strict=True)
    ):
        jacobian = _parameter_jacobian(point, transform, rotation_jacobian)
        information += float(weight) * jacobian.T @ jacobian
        if cluster_scores is not None and active_mask[index]:
            score = float(weight) * jacobian.T @ residual
            cluster_scores[cluster_inverse[index]] += score

    singular_values = np.linalg.svd(information, compute_uv=False)
    threshold = max(float(singular_values[0]) * 1e-10, np.finfo(np.float64).eps)
    information_rank = int(np.count_nonzero(singular_values > threshold))
    if information_rank < 7:
        raise ValueError(
            f"alignment geometry is rank-deficient ({information_rank}/7 observable parameters)"
        )
    information_condition = float(singular_values[0] / singular_values[-1])
    inverse_information = np.linalg.pinv(information, rcond=1e-10)

    if cluster_scores is None:
        degrees_of_freedom = max(1, 3 * active - 7)
        variance = float(np.sum(weights[:, None] * residuals**2) / degrees_of_freedom)
        covariance = variance * inverse_information
        method = "iid_gauss_newton"
    else:
        meat = cluster_scores.T @ cluster_scores
        finite_sample_correction = (num_clusters / (num_clusters - 1)) * (
            (active - 1) / max(active - 7, 1)
        )
        covariance = (
            finite_sample_correction
            * inverse_information
            @ meat
            @ inverse_information
        )
        method = DENSE_ALIGNMENT_COVARIANCE_METHOD

    floor = np.diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-12, 1e-12, 1e-12])
    covariance = 0.5 * (covariance + covariance.T) + floor
    return _CovarianceEstimate(
        covariance=covariance,
        method=method,
        num_clusters=num_clusters,
        information_rank=information_rank,
        information_condition=information_condition,
    )


def _alignment_covariance(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    transform: Sim3,
    *,
    cluster_ids: IntArray | None = None,
) -> FloatArray:
    return _alignment_covariance_estimate(
        source,
        target,
        weights,
        transform,
        cluster_ids=cluster_ids,
    ).covariance


def estimate_sim3_robust(
    source: FloatArray,
    target: FloatArray,
    *,
    weights: FloatArray | None = None,
    covariance_cluster_ids: IntArray | None = None,
    max_iterations: int = 20,
    huber_multiplier: float = 2.5,
    tolerance: float = 1e-8,
) -> AlignmentResult:
    """Estimate a similarity transform with Huber iteratively reweighted least squares.

    ``covariance_cluster_ids`` enables a cluster-robust sandwich covariance while
    leaving the fitted transform unchanged. Dense-window alignment supplies
    frame-by-spatial-tile clusters; sparse registrations retain the IID model by
    default.
    """

    iteration_count = require_genuine_integer(
        max_iterations,
        name="max_iterations",
        minimum=1,
    )
    robust_multiplier = require_finite_real(
        huber_multiplier,
        name="huber_multiplier",
        minimum=0.0,
        minimum_inclusive=False,
    )
    convergence_tolerance = require_finite_real(
        tolerance,
        name="tolerance",
        minimum=0.0,
        minimum_inclusive=False,
    )
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if source.shape[0] < 4:
        raise ValueError("at least four correspondences are required")
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 4:
        raise ValueError("at least four finite correspondences are required")

    if weights is None:
        base_weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != finite.shape:
            raise ValueError("weights must have shape (N,)")
        base_weights = supplied[finite]
        if np.any(base_weights < 0) or not np.all(np.isfinite(base_weights)):
            raise ValueError("weights must be finite and non-negative")

    clusters: IntArray | None = None
    if covariance_cluster_ids is not None:
        supplied_clusters = np.asarray(covariance_cluster_ids)
        if supplied_clusters.shape != finite.shape:
            raise ValueError("covariance_cluster_ids must have shape (N,)")
        clusters = supplied_clusters[finite]

    robust_weights = base_weights.copy()
    previous_vector: FloatArray | None = None
    cutoff = np.inf
    transform = Sim3.identity()
    for _ in range(iteration_count):
        transform = _weighted_umeyama(source, target, robust_weights)
        residual_norms = np.linalg.norm(target - transform.transform_points(source), axis=1)
        median = float(np.median(residual_norms))
        mad = float(np.median(np.abs(residual_norms - median)))
        robust_scale = max(1.4826 * mad, np.finfo(np.float64).eps)
        cutoff = max(
            median + robust_multiplier * robust_scale,
            np.finfo(np.float64).eps,
        )
        huber_weights = np.minimum(1.0, cutoff / np.maximum(residual_norms, cutoff))
        robust_weights = base_weights * huber_weights
        current_vector = transform.as_vector()
        if (
            previous_vector is not None
            and np.linalg.norm(current_vector - previous_vector) < convergence_tolerance
        ):
            break
        previous_vector = current_vector

    residuals = target - transform.transform_points(source)
    residual_norms = np.linalg.norm(residuals, axis=1)
    covariance = _alignment_covariance_estimate(
        source,
        target,
        robust_weights,
        transform,
        cluster_ids=clusters,
    )
    weighted_squared_error = float(np.sum(robust_weights * residual_norms**2))
    weight_sum = max(float(robust_weights.sum()), np.finfo(np.float64).eps)
    return AlignmentResult(
        transform=transform,
        covariance=covariance.covariance,
        residual_rms=float(np.sqrt(weighted_squared_error / weight_sum)),
        inlier_fraction=float(np.mean(residual_norms <= cutoff)),
        num_correspondences=int(source.shape[0]),
        covariance_method=covariance.method,
        num_covariance_clusters=covariance.num_clusters,
        information_rank=covariance.information_rank,
        information_condition=covariance.information_condition,
    )


def _fallback_clusters(
    source: FloatArray,
    *,
    fallback_policy: CovarianceFallbackPolicy,
    fallback: str,
) -> tuple[IntArray | None, str | None]:
    if fallback_policy == "error":
        raise ValueError(
            "spatial covariance clustering produced fewer than eight independent "
            "clusters; explicitly allow the pointwise approximation for a "
            "reconstruction control"
        )
    if source.shape[0] > 7:
        return np.arange(source.shape[0], dtype=np.int64), fallback
    return None, IID_COVARIANCE_FALLBACK


def _overlapping_correspondence_data(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int,
    seed: int,
    covariance_cluster_size: int | None,
    fallback_policy: CovarianceFallbackPolicy = "pointwise",
) -> tuple[
    FloatArray,
    FloatArray,
    NDArray[np.integer],
    IntArray | None,
    str | None,
]:
    if reference.shape[1:] != moving.shape[1:]:
        raise ValueError("overlapping windows must use the same spatial resolution")
    maximum = require_genuine_integer(
        max_correspondences,
        name="max_correspondences",
        minimum=4,
    )
    normalized_seed = require_genuine_integer(seed, name="seed", minimum=0)
    cluster_size = None
    if covariance_cluster_size is not None:
        cluster_size = require_genuine_integer(
            covariance_cluster_size,
            name="covariance_cluster_size",
            minimum=1,
        )
    if fallback_policy not in {"error", "pointwise"}:
        raise ValueError("fallback_policy must be 'error' or 'pointwise'")
    common_frames = reference.common_frames(moving)
    if common_frames.size == 0:
        raise ValueError("windows do not overlap")

    source_parts: list[FloatArray] = []
    target_parts: list[FloatArray] = []
    cluster_parts: list[IntArray] = []
    cluster_offset = 0
    width = reference.shape[2]
    for frame in common_frames:
        reference_index = reference.local_index(int(frame))
        moving_index = moving.local_index(int(frame))
        mask = reference.valid_mask[reference_index] & moving.valid_mask[moving_index]
        source_parts.append(moving.point_map[moving_index][mask])
        target_parts.append(reference.point_map[reference_index][mask])
        if cluster_size is not None:
            rows, columns = np.nonzero(mask)
            tile_columns = int(np.ceil(width / cluster_size))
            tile_ids = rows // cluster_size * tile_columns + columns // cluster_size
            _, compact_ids = np.unique(tile_ids, return_inverse=True)
            cluster_parts.append(compact_ids.astype(np.int64) + cluster_offset)
            cluster_offset += int(np.max(compact_ids) + 1) if compact_ids.size else 0

    source = np.concatenate(source_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    if source.shape[0] < 4:
        raise ValueError("overlap has fewer than four valid point correspondences")
    clusters: IntArray | None = None
    covariance_fallback: str | None = None
    if cluster_size is not None:
        clusters = np.concatenate(cluster_parts)
        if np.unique(clusters).size <= 7:
            clusters, covariance_fallback = _fallback_clusters(
                source,
                fallback_policy=fallback_policy,
                fallback=POINTWISE_COVARIANCE_FALLBACK,
            )

    if source.shape[0] > maximum:
        generator = np.random.default_rng(normalized_seed)
        selection = np.sort(
            generator.choice(source.shape[0], size=maximum, replace=False)
        )
        source = source[selection]
        target = target[selection]
        if clusters is not None:
            clusters = clusters[selection]
            if np.unique(clusters).size <= 7:
                clusters, covariance_fallback = _fallback_clusters(
                    source,
                    fallback_policy=fallback_policy,
                    fallback=POINTWISE_COVARIANCE_FALLBACK,
                )
    return source, target, common_frames, clusters, covariance_fallback


def overlapping_correspondences(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int = 100_000,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, NDArray[np.integer]]:
    """Collect same-frame, same-pixel points from two decoded windows."""

    source, target, common_frames, _, _ = _overlapping_correspondence_data(
        reference,
        moving,
        max_correspondences=max_correspondences,
        seed=seed,
        covariance_cluster_size=None,
    )
    return source, target, common_frames


def align_windows(
    reference: PredictionWindow,
    moving: PredictionWindow,
    *,
    max_correspondences: int = 100_000,
    seed: int = 0,
    covariance_cluster_size: int | None = DEFAULT_COVARIANCE_CLUSTER_SIZE,
    covariance_calibration: AlignmentCovarianceCalibration | None = None,
    fallback_policy: CovarianceFallbackPolicy | None = None,
) -> WindowAlignment:
    """Estimate a moving-to-reference transform and correlation-aware covariance."""

    context = _ALIGNMENT_COVARIANCE_CONTEXT.get()
    resolved_fallback_policy = (
        context.fallback_policy if fallback_policy is None else fallback_policy
    )
    source, target, common_frames, clusters, covariance_fallback = (
        _overlapping_correspondence_data(
            reference,
            moving,
            max_correspondences=max_correspondences,
            seed=seed,
            covariance_cluster_size=covariance_cluster_size,
            fallback_policy=resolved_fallback_policy,
        )
    )
    result = estimate_sim3_robust(
        source,
        target,
        covariance_cluster_ids=clusters,
    )
    result = replace(result, covariance_fallback=covariance_fallback)
    calibration = covariance_calibration or context.calibration
    if calibration is not None:
        result = replace(
            result,
            covariance=calibration.apply(result.covariance),
            covariance_calibration_id=calibration.artifact_id,
        )
    if context.diagnostics is not None:
        context.diagnostics.record(result)
    return WindowAlignment(
        reference_id=reference.window_id,
        moving_id=moving.window_id,
        common_frames=common_frames,
        result=result,
    )


__all__ = [
    "DENSE_ALIGNMENT_COVARIANCE_METHOD",
    "DEFAULT_COVARIANCE_CLUSTER_SIZE",
    "IID_COVARIANCE_FALLBACK",
    "POINTWISE_COVARIANCE_FALLBACK",
    "AlignmentCovarianceCalibration",
    "AlignmentCovarianceDiagnostics",
    "AlignmentResult",
    "CovarianceFallbackPolicy",
    "WindowAlignment",
    "align_windows",
    "alignment_covariance_context",
    "estimate_sim3_robust",
    "overlapping_correspondences",
]
