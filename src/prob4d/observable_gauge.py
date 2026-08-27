"""Observability-aware Gaussian factors for partially constrained Sim(3) gauges.

The stable alignment path rejects rank-deficient overlap geometry.  This
experimental kernel retains only the observable subspace in an origin-invariant,
centroid-normalized local chart.  A downstream Gaussian prior supplies the
unobservable directions instead of a numerical ridge pretending that the visual
factor measured them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .alignment import AlignmentNonConvergenceError, _normalized_transform_step, _weighted_umeyama
from .sim3 import Sim3, skew, so3_exp, so3_log

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

IID_OBSERVABLE_INFORMATION = "iid_observable_information_v1"
CLUSTER_OBSERVABLE_INFORMATION = "cluster_observable_information_v1"


def _readonly(value: FloatArray, *, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _finite_positive(value: float, *, name: str, allow_zero: bool = False) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or (numeric < 0.0 if allow_zero else numeric <= 0.0):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return numeric


def _integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _symmetric_positive_definite(value: FloatArray, *, size: int, name: str) -> FloatArray:
    array = _readonly(value, shape=(size, size), name=name).copy()
    symmetric = 0.5 * (array + array.T)
    if not np.allclose(array, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    symmetric.setflags(write=False)
    return symmetric


def _canonicalize_basis_signs(basis: FloatArray) -> FloatArray:
    result = np.asarray(basis, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        nonzero = np.flatnonzero(np.abs(result[:, column]) > 1e-12)
        if nonzero.size and result[int(nonzero[0]), column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class CentroidGaugeChart:
    """Dimensionless local Sim(3) chart around a fitted transform.

    Coordinates are ``[log scale, left rotation(3), centroid translation / rho]``.
    Scale and rotation act around the weighted source centroid, while translation
    moves the transformed centroid.  ``rho`` is the fitted RMS cloud radius.
    """

    linearization: Sim3
    source_centroid: FloatArray
    cloud_scale: float

    def __post_init__(self) -> None:
        centroid = _readonly(
            self.source_centroid,
            shape=(3,),
            name="source_centroid",
        )
        cloud_scale = _finite_positive(self.cloud_scale, name="cloud_scale")
        object.__setattr__(self, "source_centroid", centroid)
        object.__setattr__(self, "cloud_scale", cloud_scale)

    @property
    def reference_centroid(self) -> FloatArray:
        center = np.asarray(
            self.linearization.transform_points(self.source_centroid),
            dtype=np.float64,
        )
        center.setflags(write=False)
        return center

    def to_local(self, transform: Sim3) -> FloatArray:
        """Map a nearby transform to the chart coordinates."""

        log_scale = np.log(transform.scale / self.linearization.scale)
        left_rotation = transform.rotation @ self.linearization.rotation.T
        rotation = so3_log(left_rotation)
        center_delta = (
            transform.transform_points(self.source_centroid) - self.reference_centroid
        ) / self.cloud_scale
        return np.concatenate(([log_scale], rotation, center_delta))

    def from_local(self, local: FloatArray) -> Sim3:
        """Map chart coordinates to a Sim(3) transform."""

        coordinates = _readonly(local, shape=(7,), name="local")
        scale = float(np.exp(coordinates[0]) * self.linearization.scale)
        rotation = so3_exp(coordinates[1:4]) @ self.linearization.rotation
        center = self.reference_centroid + self.cloud_scale * coordinates[4:7]
        translation = center - scale * (rotation @ self.source_centroid)
        return Sim3(scale=scale, rotation=rotation, translation=translation)

    def vector_to_local_jacobian(
        self,
        transform: Sim3,
        *,
        relative_step: float = 1e-6,
    ) -> FloatArray:
        """Linearize standard ``Sim3.as_vector`` coordinates into this chart."""

        step_scale = _finite_positive(relative_step, name="relative_step")
        vector = transform.as_vector()
        if float(np.linalg.norm(vector[1:4])) >= np.pi - 1e-5:
            raise ValueError("vector covariance transport reaches the SO(3) branch cut")
        baseline = self.to_local(transform)
        if float(np.linalg.norm(baseline[1:4])) >= np.pi - 1e-5:
            raise ValueError("local covariance transport reaches the SO(3) branch cut")
        jacobian = np.empty((7, 7), dtype=np.float64)
        for index in range(7):
            step = step_scale * max(1.0, abs(float(vector[index])))
            perturbed = vector.copy()
            perturbed[index] += step
            mapped = self.to_local(Sim3.from_vector(perturbed))
            jacobian[:, index] = (mapped - baseline) / step
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("vector-to-local covariance Jacobian must be finite")
        jacobian.setflags(write=False)
        return jacobian

    def transport_vector_gaussian(
        self,
        mean: Sim3,
        covariance_vector: FloatArray,
    ) -> GaugeGaussianPosterior:
        """Transport a standard seven-vector Gaussian into the local chart."""

        covariance = _symmetric_positive_definite(
            covariance_vector,
            size=7,
            name="covariance_vector",
        )
        jacobian = self.vector_to_local_jacobian(mean)
        local_covariance = jacobian @ covariance @ jacobian.T
        local_covariance = 0.5 * (local_covariance + local_covariance.T)
        return GaugeGaussianPosterior(
            chart=self,
            mean_local=self.to_local(mean),
            covariance_local=local_covariance,
        )


@dataclass(frozen=True)
class GaugeGaussianPosterior:
    """Gaussian posterior in one :class:`CentroidGaugeChart`."""

    chart: CentroidGaugeChart
    mean_local: FloatArray
    covariance_local: FloatArray

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_local, shape=(7,), name="mean_local")
        covariance = _symmetric_positive_definite(
            self.covariance_local,
            size=7,
            name="covariance_local",
        )
        object.__setattr__(self, "mean_local", mean)
        object.__setattr__(self, "covariance_local", covariance)

    @property
    def mean_transform(self) -> Sim3:
        return self.chart.from_local(self.mean_local)


@dataclass(frozen=True)
class ObservableGaugeFactor:
    """A Gaussian factor carrying information only in an observable subspace."""

    chart: CentroidGaugeChart
    observable_basis: FloatArray
    nullspace_basis: FloatArray
    observable_information: FloatArray
    normalized_geometry_spectrum: FloatArray
    rank_threshold: float
    residual_rms: float
    residual_variance: float
    inlier_fraction: float
    num_correspondences: int
    covariance_method: str
    num_covariance_clusters: int = 0

    def __post_init__(self) -> None:
        observable = np.asarray(self.observable_basis, dtype=np.float64).copy()
        nullspace = np.asarray(self.nullspace_basis, dtype=np.float64).copy()
        if observable.ndim != 2 or observable.shape[0] != 7 or observable.shape[1] < 1:
            raise ValueError("observable_basis must have shape (7, rank) with rank positive")
        rank = int(observable.shape[1])
        if rank > 7:
            raise ValueError("observable rank must not exceed seven")
        if nullspace.shape != (7, 7 - rank):
            raise ValueError("nullspace_basis must have shape (7, 7-rank)")
        combined = np.concatenate((observable, nullspace), axis=1)
        if not np.allclose(combined.T @ combined, np.eye(7), atol=1e-9, rtol=1e-9):
            raise ValueError("observable and nullspace bases must form an orthonormal basis")
        information = _symmetric_positive_definite(
            self.observable_information,
            size=rank,
            name="observable_information",
        )
        spectrum = _readonly(
            self.normalized_geometry_spectrum,
            shape=(7,),
            name="normalized_geometry_spectrum",
        )
        if np.any(spectrum < 0.0) or np.any(np.diff(spectrum) > 1e-12):
            raise ValueError("normalized_geometry_spectrum must be nonnegative and descending")
        if not np.isclose(spectrum[0], 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError("normalized_geometry_spectrum must start at one")
        rank_threshold = _finite_positive(self.rank_threshold, name="rank_threshold")
        residual_rms = _finite_positive(
            self.residual_rms,
            name="residual_rms",
            allow_zero=True,
        )
        residual_variance = _finite_positive(
            self.residual_variance,
            name="residual_variance",
        )
        inlier_fraction = float(self.inlier_fraction)
        if not np.isfinite(inlier_fraction) or not 0.0 <= inlier_fraction <= 1.0:
            raise ValueError("inlier_fraction must lie in [0, 1]")
        correspondences = _integer(
            self.num_correspondences,
            name="num_correspondences",
            minimum=4,
        )
        clusters = _integer(
            self.num_covariance_clusters,
            name="num_covariance_clusters",
            minimum=0,
        )
        if not self.covariance_method:
            raise ValueError("covariance_method must be nonempty")
        observable.setflags(write=False)
        nullspace.setflags(write=False)
        object.__setattr__(self, "observable_basis", observable)
        object.__setattr__(self, "nullspace_basis", nullspace)
        object.__setattr__(self, "observable_information", information)
        object.__setattr__(self, "normalized_geometry_spectrum", spectrum)
        object.__setattr__(self, "rank_threshold", rank_threshold)
        object.__setattr__(self, "residual_rms", residual_rms)
        object.__setattr__(self, "residual_variance", residual_variance)
        object.__setattr__(self, "inlier_fraction", inlier_fraction)
        object.__setattr__(self, "num_correspondences", correspondences)
        object.__setattr__(self, "num_covariance_clusters", clusters)

    @property
    def rank(self) -> int:
        return int(self.observable_basis.shape[1])

    @property
    def information_matrix(self) -> FloatArray:
        matrix = self.observable_basis @ self.observable_information @ self.observable_basis.T
        matrix = 0.5 * (matrix + matrix.T)
        matrix.setflags(write=False)
        return matrix

    @property
    def observable_covariance(self) -> FloatArray:
        covariance = np.linalg.solve(
            self.observable_information,
            np.eye(self.rank),
        )
        covariance = 0.5 * (covariance + covariance.T)
        covariance.setflags(write=False)
        return covariance

    def quadratic_cost_local(self, local: FloatArray) -> float:
        coordinates = _readonly(local, shape=(7,), name="local")
        projected = self.observable_basis.T @ coordinates
        return 0.5 * float(projected @ self.observable_information @ projected)

    def quadratic_cost(self, transform: Sim3) -> float:
        return self.quadratic_cost_local(self.chart.to_local(transform))

    def fuse_local_gaussian(
        self,
        prior_mean_local: FloatArray,
        prior_covariance_local: FloatArray,
    ) -> GaugeGaussianPosterior:
        """Fuse the factor with a full-rank Gaussian prior in the same chart."""

        prior_mean = _readonly(prior_mean_local, shape=(7,), name="prior_mean_local")
        prior_covariance = _symmetric_positive_definite(
            prior_covariance_local,
            size=7,
            name="prior_covariance_local",
        )
        prior_information = np.linalg.solve(prior_covariance, np.eye(7))
        posterior_information = prior_information + self.information_matrix
        posterior_covariance = np.linalg.solve(posterior_information, np.eye(7))
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
        posterior_mean = posterior_covariance @ (prior_information @ prior_mean)
        return GaugeGaussianPosterior(
            chart=self.chart,
            mean_local=posterior_mean,
            covariance_local=posterior_covariance,
        )

    def fuse_transform_gaussian(
        self,
        prior_mean: Sim3,
        prior_covariance_local: FloatArray,
    ) -> GaugeGaussianPosterior:
        return self.fuse_local_gaussian(
            self.chart.to_local(prior_mean),
            prior_covariance_local,
        )

    def fuse_vector_gaussian(
        self,
        prior_mean: Sim3,
        prior_covariance_vector: FloatArray,
    ) -> GaugeGaussianPosterior:
        """Transport and fuse an existing standard-coordinate gauge prior."""

        transported = self.chart.transport_vector_gaussian(
            prior_mean,
            prior_covariance_vector,
        )
        return self.fuse_local_gaussian(
            transported.mean_local,
            transported.covariance_local,
        )


def _prepare_inputs(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray | None,
    covariance_cluster_ids: IntArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray | None]:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        source_array.shape != target_array.shape
        or source_array.ndim != 2
        or source_array.shape[1] != 3
    ):
        raise ValueError("source and target must both have shape (N, 3)")
    if source_array.shape[0] < 4:
        raise ValueError("at least four correspondences are required")
    finite = np.all(np.isfinite(source_array), axis=1) & np.all(
        np.isfinite(target_array), axis=1
    )
    source_array = source_array[finite]
    target_array = target_array[finite]
    if source_array.shape[0] < 4:
        raise ValueError("at least four finite correspondences are required")
    if weights is None:
        base_weights = np.ones(source_array.shape[0], dtype=np.float64)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != finite.shape:
            raise ValueError("weights must have shape (N,)")
        base_weights = supplied[finite]
        if np.any(base_weights < 0.0) or not np.all(np.isfinite(base_weights)):
            raise ValueError("weights must be finite and nonnegative")
    if float(np.sum(base_weights)) <= np.finfo(np.float64).eps:
        raise ValueError("alignment weights have zero total mass")
    clusters: IntArray | None = None
    if covariance_cluster_ids is not None:
        supplied_clusters = np.asarray(covariance_cluster_ids)
        if supplied_clusters.shape != finite.shape:
            raise ValueError("covariance_cluster_ids must have shape (N,)")
        clusters = supplied_clusters[finite]
    return source_array, target_array, base_weights, clusters


def _robust_fit(
    source: FloatArray,
    target: FloatArray,
    base_weights: FloatArray,
    *,
    max_iterations: int,
    huber_multiplier: float,
    tolerance: float,
) -> tuple[Sim3, FloatArray, float]:
    iterations = _integer(max_iterations, name="max_iterations", minimum=2)
    multiplier = _finite_positive(huber_multiplier, name="huber_multiplier")
    convergence_tolerance = _finite_positive(tolerance, name="tolerance")
    robust_weights = base_weights.copy()
    previous_transform: Sim3 | None = None
    cutoff = np.inf
    transform = Sim3.identity()
    transform_delta = np.inf
    relative_weight_delta = np.inf
    converged = False
    for _ in range(iterations):
        fit_weights = robust_weights
        transform = _weighted_umeyama(source, target, fit_weights)
        residual_norms = np.linalg.norm(
            target - transform.transform_points(source),
            axis=1,
        )
        median = float(np.median(residual_norms))
        mad = float(np.median(np.abs(residual_norms - median)))
        robust_scale = max(1.4826 * mad, np.finfo(np.float64).eps)
        cutoff = max(median + multiplier * robust_scale, np.finfo(np.float64).eps)
        huber_weights = np.minimum(1.0, cutoff / np.maximum(residual_norms, cutoff))
        next_weights = base_weights * huber_weights
        transform_delta = (
            np.inf
            if previous_transform is None
            else _normalized_transform_step(
                source,
                base_weights,
                previous_transform,
                transform,
            )
        )
        weight_norm = max(float(np.linalg.norm(fit_weights)), np.finfo(np.float64).eps)
        relative_weight_delta = float(
            np.linalg.norm(next_weights - fit_weights) / weight_norm
        )
        if previous_transform is not None and transform_delta < convergence_tolerance:
            robust_weights = fit_weights
            converged = True
            break
        previous_transform = transform
        robust_weights = next_weights
    if not converged:
        raise AlignmentNonConvergenceError(
            max_iterations=iterations,
            transform_delta=transform_delta,
            relative_weight_delta=relative_weight_delta,
        )
    return transform, robust_weights, cutoff


def _intrinsic_information(
    source: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    transform: Sim3,
    *,
    cluster_ids: IntArray | None,
) -> tuple[
    CentroidGaugeChart,
    FloatArray,
    FloatArray,
    FloatArray | None,
    int,
]:
    active_mask = weights > 1e-3
    active = int(np.count_nonzero(active_mask))
    if active < 4:
        raise ValueError("fewer than four correspondences retain positive robust weight")
    weight_sum = float(np.sum(weights))
    source_centroid = np.sum(weights[:, None] * source, axis=0) / weight_sum
    transformed = transform.transform_points(source)
    reference_centroid = transform.transform_points(source_centroid)
    centered = transformed - reference_centroid
    cloud_scale = float(
        np.sqrt(np.sum(weights * np.sum(centered**2, axis=1)) / weight_sum)
    )
    if cloud_scale <= np.finfo(np.float64).eps:
        raise ValueError("source correspondences have no spatial extent")
    chart = CentroidGaugeChart(
        linearization=transform,
        source_centroid=source_centroid,
        cloud_scale=cloud_scale,
    )
    residuals = target - transformed
    information = np.zeros((7, 7), dtype=np.float64)
    cluster_inverse: IntArray | None = None
    cluster_scores: FloatArray | None = None
    num_clusters = 0
    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids)
        if clusters.shape != (source.shape[0],):
            raise ValueError("covariance_cluster_ids must have shape (N,)")
        _, compact = np.unique(clusters[active_mask], return_inverse=True)
        num_clusters = int(np.max(compact) + 1) if compact.size else 0
        cluster_inverse = np.full(source.shape[0], -1, dtype=np.int64)
        cluster_inverse[active_mask] = compact
        cluster_scores = np.zeros((num_clusters, 7), dtype=np.float64)
    identity = np.eye(3)
    for index, (point, residual, weight) in enumerate(
        zip(centered, residuals, weights, strict=True)
    ):
        jacobian = np.empty((3, 7), dtype=np.float64)
        jacobian[:, 0] = point
        jacobian[:, 1:4] = -skew(point)
        jacobian[:, 4:7] = cloud_scale * identity
        information += float(weight) * jacobian.T @ jacobian
        if cluster_scores is not None and active_mask[index]:
            cluster_scores[cluster_inverse[index]] += float(weight) * jacobian.T @ residual
    return chart, residuals, information, cluster_scores, num_clusters


def estimate_observable_sim3_factor(
    source: FloatArray,
    target: FloatArray,
    *,
    weights: FloatArray | None = None,
    covariance_cluster_ids: IntArray | None = None,
    max_iterations: int = 64,
    huber_multiplier: float = 2.5,
    tolerance: float = 1e-8,
    rank_threshold: float = 1e-8,
    residual_variance_floor: float = 1e-12,
) -> ObservableGaugeFactor:
    """Fit a robust Sim(3) representative and retain its observable information.

    ``rank_threshold`` applies to the descending geometry spectrum after the
    centroid-normalized chart has removed origin and mixed translation-scale
    effects.  It is a scientific observability threshold, not a ridge.  Directions
    below it remain in the factor nullspace and must be supplied by a prior or
    exact fallback downstream.
    """

    relative_threshold = _finite_positive(rank_threshold, name="rank_threshold")
    if relative_threshold >= 1.0:
        raise ValueError("rank_threshold must be smaller than one")
    variance_floor = _finite_positive(
        residual_variance_floor,
        name="residual_variance_floor",
    )
    source_array, target_array, base_weights, clusters = _prepare_inputs(
        source,
        target,
        weights,
        covariance_cluster_ids,
    )
    transform, robust_weights, cutoff = _robust_fit(
        source_array,
        target_array,
        base_weights,
        max_iterations=max_iterations,
        huber_multiplier=huber_multiplier,
        tolerance=tolerance,
    )
    chart, residuals, geometry_information, cluster_scores, num_clusters = (
        _intrinsic_information(
            source_array,
            target_array,
            robust_weights,
            transform,
            cluster_ids=clusters,
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (geometry_information + geometry_information.T)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0])
    if maximum <= np.finfo(np.float64).eps:
        raise ValueError("alignment geometry carries no observable gauge information")
    normalized_spectrum = eigenvalues / maximum
    rank = int(np.count_nonzero(normalized_spectrum >= relative_threshold))
    if rank < 1:
        raise ValueError("rank_threshold removes every gauge direction")
    observable_basis = _canonicalize_basis_signs(eigenvectors[:, :rank])
    nullspace_basis = _canonicalize_basis_signs(eigenvectors[:, rank:])
    active = int(np.count_nonzero(robust_weights > 1e-3))
    weighted_squared_error = float(np.sum(robust_weights[:, None] * residuals**2))
    degrees_of_freedom = max(1, 3 * active - rank)
    residual_variance = max(weighted_squared_error / degrees_of_freedom, variance_floor)
    if cluster_scores is None:
        observable_information = np.diag(eigenvalues[:rank] / residual_variance)
        method = IID_OBSERVABLE_INFORMATION
    else:
        if num_clusters <= rank:
            raise ValueError(
                "cluster-robust observable covariance requires more active clusters "
                "than observable gauge dimensions"
            )
        meat = cluster_scores.T @ cluster_scores
        reduced_meat = observable_basis.T @ meat @ observable_basis
        inverse_geometry = np.diag(1.0 / eigenvalues[:rank])
        correction = (num_clusters / (num_clusters - 1)) * (
            (active - 1) / max(active - rank, 1)
        )
        reduced_covariance = (
            correction * inverse_geometry @ reduced_meat @ inverse_geometry
        )
        reduced_covariance = 0.5 * (reduced_covariance + reduced_covariance.T)
        covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(
            reduced_covariance
        )
        covariance_floor = max(
            float(np.max(np.abs(covariance_eigenvalues), initial=0.0)) * 1e-12,
            np.finfo(np.float64).eps,
        )
        reduced_covariance = (
            covariance_eigenvectors
            * np.maximum(covariance_eigenvalues, covariance_floor)
        ) @ covariance_eigenvectors.T
        observable_information = np.linalg.solve(
            reduced_covariance,
            np.eye(rank),
        )
        method = CLUSTER_OBSERVABLE_INFORMATION
    residual_norms = np.linalg.norm(residuals, axis=1)
    weight_sum = max(float(np.sum(robust_weights)), np.finfo(np.float64).eps)
    return ObservableGaugeFactor(
        chart=chart,
        observable_basis=observable_basis,
        nullspace_basis=nullspace_basis,
        observable_information=observable_information,
        normalized_geometry_spectrum=normalized_spectrum,
        rank_threshold=relative_threshold,
        residual_rms=float(np.sqrt(np.sum(robust_weights * residual_norms**2) / weight_sum)),
        residual_variance=residual_variance,
        inlier_fraction=float(np.mean(residual_norms <= cutoff)),
        num_correspondences=int(source_array.shape[0]),
        covariance_method=method,
        num_covariance_clusters=num_clusters,
    )


__all__ = [
    "CLUSTER_OBSERVABLE_INFORMATION",
    "IID_OBSERVABLE_INFORMATION",
    "CentroidGaugeChart",
    "GaugeGaussianPosterior",
    "ObservableGaugeFactor",
    "estimate_observable_sim3_factor",
]
