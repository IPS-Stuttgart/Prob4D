"""Pure numerical helpers for the DOT-rope CUT3R source experiment.

The module intentionally contains no dataset or provider imports.  It supplies
robust Sim(3) fitting, marker sampling, clustered bootstrap uncertainty, and the
fixed-mean covariance closures compared by the registered source-only study.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def canonical_json(value: object) -> bytes:
    """Encode a JSON-compatible value using the repository's canonical form."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: object) -> str:
    """Return the SHA-256 identity of a canonical JSON-compatible value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Sim3:
    """Proper similarity transform ``y = scale * rotation @ x + translation``."""

    scale: float
    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Sim3 scale must be finite and positive")
        if rotation.shape != (3, 3):
            raise ValueError("Sim3 rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("Sim3 translation must have shape (3,)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Sim3 values must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
            raise ValueError("Sim3 rotation is not orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
            raise ValueError("Sim3 rotation must be proper")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: NDArray[np.floating]) -> FloatArray:
        array = np.asarray(points, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError("points must end in dimension three")
        return self.scale * np.einsum("ij,...j->...i", self.rotation, array) + self.translation

    def inverse(self) -> Sim3:
        inverse_scale = 1.0 / self.scale
        inverse_rotation = self.rotation.T
        inverse_translation = -inverse_scale * (inverse_rotation @ self.translation)
        return Sim3(inverse_scale, inverse_rotation, inverse_translation)

    def compose(self, other: Sim3) -> Sim3:
        """Return ``self(other(x))``."""
        scale = self.scale * other.scale
        rotation = self.rotation @ other.rotation
        translation = self.scale * (self.rotation @ other.translation) + self.translation
        return Sim3(scale, rotation, translation)


def skew(vector: NDArray[np.floating]) -> FloatArray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(vector: NDArray[np.floating]) -> FloatArray:
    """Convert an axis-angle vector to a proper rotation matrix."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("rotation vector must have shape (3,)")
    theta = float(np.linalg.norm(value))
    if theta < 1.0e-10:
        cross = skew(value)
        return np.eye(3) + cross + 0.5 * (cross @ cross)
    axis = value / theta
    cross = skew(axis)
    return np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)


def matrix_to_rotvec(rotation: NDArray[np.floating]) -> FloatArray:
    """Convert a proper rotation matrix to its principal axis-angle vector."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3)")
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1.0e-9:
        return 0.5 * np.asarray(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
    if math.pi - theta < 1.0e-6:
        diagonal = np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1.0e-8:
            axis = np.asarray([1.0, 0.0, 0.0])
        else:
            if pivot == 0:
                axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
            elif pivot == 1:
                axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
                axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
            else:
                axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
                axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
            axis /= np.linalg.norm(axis)
        return theta * axis
    axis = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(theta))
    return theta * axis


def sim3_to_vector(transform: Sim3) -> FloatArray:
    """Return ``[translation, principal rotation vector, log scale]``."""
    return np.concatenate(
        (
            transform.translation,
            matrix_to_rotvec(transform.rotation),
            np.asarray([math.log(transform.scale)]),
        )
    )


def vector_to_sim3(vector: NDArray[np.floating]) -> Sim3:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError("Sim3 vector must have shape (7,)")
    return Sim3(
        math.exp(float(value[6])),
        rotvec_to_matrix(value[3:6]),
        value[:3].copy(),
    )


def fit_sim3(source: NDArray[np.floating], target: NDArray[np.floating]) -> Sim3:
    """Fit a proper least-squares Sim(3) with Umeyama's closed form."""
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("source and target must have matching shape (n, 3)")
    if x.shape[0] < 3:
        raise ValueError("at least three correspondences are required")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[finite]
    y = y[finite]
    if x.shape[0] < 3:
        raise ValueError("fewer than three finite correspondences remain")
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance = float(np.mean(np.sum(centered_x * centered_x, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("source correspondences have zero spatial variance")
    covariance = centered_y.T @ centered_x / x.shape[0]
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vh) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vh
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("fitted similarity scale is invalid")
    translation = mean_y - scale * (rotation @ mean_x)
    return Sim3(scale, rotation, translation)


def robust_fit_sim3(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    retain_fraction: float = 0.8,
    iterations: int = 3,
) -> tuple[Sim3, FloatArray]:
    """Iteratively trim large residuals and refit a proper Sim(3)."""
    if not 0.5 <= retain_fraction <= 1.0:
        raise ValueError("retain_fraction must lie in [0.5, 1]")
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    indices = np.flatnonzero(finite)
    if indices.size < 3:
        raise ValueError("insufficient finite correspondences")
    transform = fit_sim3(x[indices], y[indices])
    for _ in range(iterations):
        residual = np.linalg.norm(transform.apply(x[indices]) - y[indices], axis=1)
        keep_count = max(3, int(math.ceil(retain_fraction * indices.size)))
        order = np.argsort(residual, kind="stable")[:keep_count]
        next_indices = np.sort(indices[order])
        if np.array_equal(next_indices, indices):
            break
        indices = next_indices
        transform = fit_sim3(x[indices], y[indices])
    all_residuals = np.linalg.norm(transform.apply(x) - y, axis=1)
    return transform, all_residuals


def parse_coordinate_text(text: str, dimensions: int) -> FloatArray:
    """Parse comma/space separated DOT coordinate rows."""
    if dimensions not in (2, 3):
        raise ValueError("coordinate dimension must be two or three")
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.replace(";", ",").replace("\t", ",").replace(" ", ",").split(",")
        values: list[float] = []
        for field in fields:
            value = field.strip()
            if not value:
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        if len(values) >= dimensions:
            rows.append(values[-dimensions:])
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dimensions or result.shape[0] < 3:
        raise ValueError("coordinate payload has no valid point table")
    return result


def bilinear_sample(
    field: NDArray[np.floating],
    coordinates: NDArray[np.floating],
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Sample an ``H x W x C`` field at floating ``(u, v)`` coordinates."""
    image = np.asarray(field, dtype=np.float64)
    points = np.asarray(coordinates, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError("field must have shape (height, width, channels)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    height, width, channels = image.shape
    u = points[:, 0]
    v = points[:, 1]
    valid = (
        np.isfinite(points).all(axis=1)
        & (u >= 0.0)
        & (v >= 0.0)
        & (u <= width - 1.0)
        & (v <= height - 1.0)
    )
    clipped_u = np.clip(u, 0.0, width - 1.0)
    clipped_v = np.clip(v, 0.0, height - 1.0)
    u0 = np.floor(clipped_u).astype(np.int64)
    v0 = np.floor(clipped_v).astype(np.int64)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    du = clipped_u - u0
    dv = clipped_v - v0
    result = (
        (1.0 - du)[:, None] * (1.0 - dv)[:, None] * image[v0, u0]
        + du[:, None] * (1.0 - dv)[:, None] * image[v0, u1]
        + (1.0 - du)[:, None] * dv[:, None] * image[v1, u0]
        + du[:, None] * dv[:, None] * image[v1, u1]
    )
    if result.shape != (points.shape[0], channels):
        raise AssertionError("bilinear sampler returned an unexpected shape")
    valid &= np.isfinite(result).all(axis=1)
    return result, valid


def clustered_bootstrap_sim3(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    groups: NDArray[np.integer],
    *,
    replicates: int,
    seed: int,
    retain_fraction: float = 0.8,
) -> tuple[FloatArray, list[Sim3]]:
    """Bootstrap frame clusters and markers, returning covariance and transforms."""
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_array = np.asarray(groups, dtype=np.int64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("bootstrap correspondences must have shape (n, 3)")
    if group_array.shape != (x.shape[0],):
        raise ValueError("bootstrap groups must match correspondence rows")
    unique_groups = np.unique(group_array)
    if unique_groups.size < 2:
        raise ValueError("at least two frame clusters are required")
    if replicates < 16:
        raise ValueError("at least sixteen bootstrap replicates are required")
    group_indices = {group: np.flatnonzero(group_array == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    transforms: list[Sim3] = []
    attempts = 0
    max_attempts = replicates * 10
    while len(transforms) < replicates and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        sampled_indices: list[int] = []
        for group in sampled_groups:
            indices = group_indices[int(group)]
            chosen = rng.choice(indices, size=indices.size, replace=True)
            sampled_indices.extend(int(value) for value in chosen)
        try:
            transform, _ = robust_fit_sim3(
                x[sampled_indices],
                y[sampled_indices],
                retain_fraction=retain_fraction,
                iterations=2,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        transforms.append(transform)
    if len(transforms) < max(16, replicates // 2):
        raise ValueError("too few valid clustered bootstrap transforms")
    vectors = np.stack([sim3_to_vector(transform) for transform in transforms])
    covariance = np.cov(vectors, rowvar=False, ddof=1)
    covariance = nearest_psd(covariance, floor=0.0)
    return covariance, transforms


def nearest_psd(matrix: NDArray[np.floating], *, floor: float) -> FloatArray:
    """Symmetrize a matrix and clip its eigenvalues to a finite floor."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    return (eigenvectors * clipped) @ eigenvectors.T


def transform_probe_vector(
    parameter: NDArray[np.floating],
    probes: NDArray[np.floating],
) -> FloatArray:
    """Apply an absolute Sim(3) parameter and flatten all transformed probes."""
    return vector_to_sim3(parameter).apply(probes).reshape(-1)


def finite_difference_derivatives(
    function: Callable[[FloatArray], FloatArray],
    center: NDArray[np.floating],
    steps: NDArray[np.floating],
) -> tuple[FloatArray, FloatArray]:
    """Evaluate a centered Jacobian and full Hessian for a vector function."""
    x0 = np.asarray(center, dtype=np.float64)
    h = np.asarray(steps, dtype=np.float64)
    if x0.ndim != 1 or h.shape != x0.shape or np.any(h <= 0.0):
        raise ValueError("finite-difference center/steps are invalid")
    f0 = np.asarray(function(x0), dtype=np.float64)
    output_dimension = f0.size
    dimension = x0.size
    jacobian = np.empty((output_dimension, dimension), dtype=np.float64)
    hessian = np.empty((output_dimension, dimension, dimension), dtype=np.float64)
    plus: list[FloatArray] = []
    minus: list[FloatArray] = []
    for index in range(dimension):
        delta = np.zeros(dimension)
        delta[index] = h[index]
        positive = np.asarray(function(x0 + delta), dtype=np.float64)
        negative = np.asarray(function(x0 - delta), dtype=np.float64)
        plus.append(positive)
        minus.append(negative)
        jacobian[:, index] = (positive - negative) / (2.0 * h[index])
        hessian[:, index, index] = (positive - 2.0 * f0 + negative) / (h[index] ** 2)
    for first in range(dimension):
        for second in range(first + 1, dimension):
            delta_first = np.zeros(dimension)
            delta_second = np.zeros(dimension)
            delta_first[first] = h[first]
            delta_second[second] = h[second]
            mixed = (
                function(x0 + delta_first + delta_second)
                - function(x0 + delta_first - delta_second)
                - function(x0 - delta_first + delta_second)
                + function(x0 - delta_first - delta_second)
            ) / (4.0 * h[first] * h[second])
            hessian[:, first, second] = mixed
            hessian[:, second, first] = mixed
    return jacobian, hessian


def _weighted_covariance_about_fixed_mean(
    samples: FloatArray,
    weights: FloatArray,
    fixed_mean: FloatArray,
) -> FloatArray:
    centered = samples - fixed_mean[None, :]
    return np.einsum("n,ni,nj->ij", weights, centered, centered)


def covariance_closures(
    center: NDArray[np.floating],
    covariance: NDArray[np.floating],
    probes: NDArray[np.floating],
    bootstrap_transforms: list[Sim3],
    *,
    scalar_inflation: float,
    finite_difference_steps: NDArray[np.floating],
    orbit_nodes: int,
    tensor_gh_order: int,
) -> dict[str, FloatArray]:
    """Compute all registered fixed-mean query covariance closures."""
    x0 = np.asarray(center, dtype=np.float64)
    sigma = nearest_psd(covariance, floor=0.0)
    probe_array = np.asarray(probes, dtype=np.float64)

    def function(value):
        return transform_probe_vector(value, probe_array)

    fixed_mean = function(x0)
    jacobian, hessian = finite_difference_derivatives(
        function,
        x0,
        np.asarray(finite_difference_steps, dtype=np.float64),
    )
    first_order = jacobian @ sigma @ jacobian.T

    dimension = x0.size
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    square_root = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    radial_samples: list[FloatArray] = []
    radial_scale = math.sqrt(dimension)
    for index in range(dimension):
        delta = radial_scale * square_root[:, index]
        radial_samples.append(function(x0 + delta))
        radial_samples.append(function(x0 - delta))
    radial_array = np.stack(radial_samples)
    radial_weights = np.full(2 * dimension, 1.0 / (2.0 * dimension))
    spherical_radial = _weighted_covariance_about_fixed_mean(
        radial_array,
        radial_weights,
        fixed_mean,
    )

    quadratic = first_order.copy()
    for first in range(fixed_mean.size):
        for second in range(first, fixed_mean.size):
            curvature = 0.5 * np.trace(hessian[first] @ sigma @ hessian[second] @ sigma)
            quadratic[first, second] += curvature
            if first != second:
                quadratic[second, first] += curvature
    quadratic = nearest_psd(quadratic, floor=0.0)

    rotation_covariance = sigma[3:6, 3:6]
    rotation_values, rotation_vectors = np.linalg.eigh(rotation_covariance)
    principal_index = int(np.argmax(rotation_values))
    principal_variance = max(float(rotation_values[principal_index]), 0.0)
    orbit_direction = np.zeros(dimension)
    orbit_direction[3:6] = rotation_vectors[:, principal_index]
    orbit_nodes_array, orbit_weights_array = np.polynomial.hermite.hermgauss(orbit_nodes)
    orbit_weights_array = orbit_weights_array / math.sqrt(math.pi)
    if principal_variance > 0.0:
        orbit_parameters = np.stack(
            [
                x0 + math.sqrt(2.0 * principal_variance) * node * orbit_direction
                for node in orbit_nodes_array
            ]
        )
        orbit_samples = np.stack([function(value) for value in orbit_parameters])
        orbit_covariance = _weighted_covariance_about_fixed_mean(
            orbit_samples,
            orbit_weights_array,
            fixed_mean,
        )
        residual_sigma = nearest_psd(
            sigma - principal_variance * np.outer(orbit_direction, orbit_direction),
            floor=0.0,
        )
        orbit_covariance += jacobian @ residual_sigma @ jacobian.T
    else:
        orbit_covariance = first_order.copy()

    nodes, weights = np.polynomial.hermite.hermgauss(tensor_gh_order)
    weights = weights / math.sqrt(math.pi)
    tensor_samples: list[FloatArray] = []
    tensor_weights: list[float] = []
    for multi_index in np.ndindex(*(tensor_gh_order for _ in range(dimension))):
        standardized = np.asarray([nodes[index] for index in multi_index])
        weight = float(np.prod([weights[index] for index in multi_index]))
        parameter = x0 + square_root @ (math.sqrt(2.0) * standardized)
        tensor_samples.append(function(parameter))
        tensor_weights.append(weight)
    tensor_array = np.stack(tensor_samples)
    tensor_weight_array = np.asarray(tensor_weights)
    tensor_gh = _weighted_covariance_about_fixed_mean(
        tensor_array,
        tensor_weight_array,
        fixed_mean,
    )

    bootstrap_samples = np.stack(
        [transform.apply(probe_array).reshape(-1) for transform in bootstrap_transforms]
    )
    bootstrap_weights = np.full(bootstrap_samples.shape[0], 1.0 / bootstrap_samples.shape[0])
    bootstrap_fallback = _weighted_covariance_about_fixed_mean(
        bootstrap_samples,
        bootstrap_weights,
        fixed_mean,
    )

    return {
        "local_first_order": nearest_psd(first_order, floor=0.0),
        "axis_spherical_radial": nearest_psd(spherical_radial, floor=0.0),
        "scalar_inflation": nearest_psd(scalar_inflation * first_order, floor=0.0),
        "pointwise_quadratic": np.diag(np.maximum(np.diag(quadratic), 0.0)),
        "shared_quadratic_curvature": quadratic,
        "dominant_rotation_orbit": nearest_psd(orbit_covariance, floor=0.0),
        "tensor_gauss_hermite": nearest_psd(tensor_gh, floor=0.0),
        "cluster_bootstrap_fallback": nearest_psd(bootstrap_fallback, floor=0.0),
    }


def normalized_gaussian_score(
    truth: NDArray[np.floating],
    fixed_mean: NDArray[np.floating],
    covariance: NDArray[np.floating],
    *,
    span: float,
    observation_noise_fraction: float,
) -> dict[str, float | bool]:
    """Score one joint Gaussian after normalizing all coordinates by rope span."""
    target = np.asarray(truth, dtype=np.float64).reshape(-1) / span
    mean = np.asarray(fixed_mean, dtype=np.float64).reshape(-1) / span
    value = np.asarray(covariance, dtype=np.float64) / (span * span)
    dimension = target.size
    noise_variance = observation_noise_fraction**2
    value = nearest_psd(value + noise_variance * np.eye(dimension), floor=noise_variance)
    difference = target - mean
    sign, log_determinant = np.linalg.slogdet(value)
    if sign <= 0.0 or not math.isfinite(float(log_determinant)):
        raise ValueError("predictive covariance is not positive definite")
    mahalanobis = float(difference @ np.linalg.solve(value, difference))
    nll = 0.5 * (dimension * math.log(2.0 * math.pi) + float(log_determinant) + mahalanobis)
    threshold = (
        dimension
        * (1.0 - 2.0 / (9.0 * dimension) + 1.6448536269514722 * math.sqrt(2.0 / (9.0 * dimension)))
        ** 3
    )
    return {
        "normalized_nll_per_dimension": nll / dimension,
        "mahalanobis": mahalanobis,
        "chi2_95_threshold_approx": threshold,
        "covered_95": mahalanobis <= threshold,
        "mean_error_fraction_of_span": float(np.linalg.norm(difference) / math.sqrt(dimension)),
        "mean_predictive_sd_fraction_of_span": float(math.sqrt(float(np.trace(value)) / dimension)),
    }


def make_off_axis_probes(points: NDArray[np.floating], *, count: int) -> tuple[FloatArray, float]:
    """Construct a deterministic ring of probes around the rope principal axis."""
    cloud = np.asarray(points, dtype=np.float64)
    finite = cloud[np.isfinite(cloud).all(axis=1)]
    if finite.shape[0] < 6:
        raise ValueError("too few finite points for probe construction")
    center = np.mean(finite, axis=0)
    centered = finite - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = centered @ axis
    span = float(np.max(projection) - np.min(projection))
    if span <= 1.0e-8:
        raise ValueError("rope span is degenerate")
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    radius = 0.25 * span
    probes = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        axial_offset = 0.15 * span * (-1.0 if index % 2 == 0 else 1.0)
        probes.append(
            center
            + axial_offset * axis
            + radius * (math.cos(angle) * first + math.sin(angle) * second)
        )
    return np.asarray(probes), span
