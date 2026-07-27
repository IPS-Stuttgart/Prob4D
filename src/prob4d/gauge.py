"""Recursive and fixed-lag estimation of MotionCrafter window gauges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._gauge_ci import fuse_sim3_covariance_intersection
from .alignment import WindowAlignment
from .sim3 import Sim3

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class RelativeGaugeConstraint:
    """Measurement mapping ``moving_id`` coordinates into ``reference_id`` coordinates."""

    reference_id: str
    moving_id: str
    reference_from_moving: Sim3
    covariance: FloatArray
    residual_rms: float = 0.0
    num_correspondences: int = 0

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("relative gauge covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)

    @classmethod
    def from_window_alignment(cls, alignment: WindowAlignment) -> RelativeGaugeConstraint:
        return cls(
            reference_id=alignment.reference_id,
            moving_id=alignment.moving_id,
            reference_from_moving=alignment.result.transform,
            covariance=alignment.result.covariance,
            residual_rms=alignment.result.residual_rms,
            num_correspondences=alignment.result.num_correspondences,
        )


@dataclass(frozen=True)
class GaugeEstimate:
    window_id: str
    global_from_local: Sim3
    covariance: FloatArray

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class GaugeAnchor:
    window_id: str
    global_from_local: Sim3
    covariance: FloatArray

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge-anchor covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class GaugeCovarianceCalibration:
    """Blockwise inflation for correlated dense-overlap gauge measurements."""

    scale: float
    rotation: float
    translation: float
    trim_quantile: float = 0.99
    count: int = 0

    def __post_init__(self) -> None:
        if self.scale <= 0 or self.rotation <= 0 or self.translation <= 0:
            raise ValueError("gauge covariance inflation factors must be positive")
        if not 0.0 < self.trim_quantile <= 1.0:
            raise ValueError("trim_quantile must be in (0, 1]")
        if self.count < 0:
            raise ValueError("calibration count must be non-negative")

    @property
    def scaling_matrix(self) -> FloatArray:
        factors = np.asarray(
            [self.scale] + [self.rotation] * 3 + [self.translation] * 3,
            dtype=np.float64,
        )
        return np.diag(np.sqrt(factors))

    def apply(self, covariance: FloatArray) -> FloatArray:
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge covariance must have shape (7, 7)")
        scaling = self.scaling_matrix
        inflated = scaling @ covariance @ scaling
        return 0.5 * (inflated + inflated.T)

    @classmethod
    def fit(
        cls,
        errors: FloatArray,
        covariances: FloatArray,
        *,
        trim_quantile: float = 0.99,
    ) -> GaugeCovarianceCalibration:
        errors = np.asarray(errors, dtype=np.float64)
        covariances = np.asarray(covariances, dtype=np.float64)
        if errors.ndim != 2 or errors.shape[1] != 7:
            raise ValueError("gauge errors must have shape (N, 7)")
        if covariances.shape != (errors.shape[0], 7, 7):
            raise ValueError("gauge covariances must have shape (N, 7, 7)")
        if errors.shape[0] == 0:
            raise ValueError("at least one gauge error is required")
        if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(covariances)):
            raise ValueError("gauge calibration inputs must be finite")
        if not 0.0 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must be in (0, 1]")

        def normalized_quadratic(
            block_errors: FloatArray, block_covariances: FloatArray
        ) -> FloatArray:
            values = np.empty(block_errors.shape[0], dtype=np.float64)
            for index, (error, covariance) in enumerate(
                zip(block_errors, block_covariances, strict=True)
            ):
                symmetric = 0.5 * (covariance + covariance.T)
                values[index] = (
                    error @ np.linalg.pinv(symmetric, rcond=1e-10) @ error
                ) / error.size
            return values

        def trimmed_mean(values: FloatArray) -> float:
            upper = float(np.quantile(values, trim_quantile))
            return max(float(np.mean(np.minimum(values, upper))), 1e-6)

        scale_ratios = errors[:, 0] ** 2 / np.maximum(covariances[:, 0, 0], 1e-12)
        rotation_ratios = normalized_quadratic(errors[:, 1:4], covariances[:, 1:4, 1:4])
        translation_ratios = normalized_quadratic(errors[:, 4:7], covariances[:, 4:7, 4:7])
        return cls(
            scale=trimmed_mean(scale_ratios),
            rotation=trimmed_mean(rotation_ratios),
            translation=trimmed_mean(translation_ratios),
            trim_quantile=trim_quantile,
            count=errors.shape[0],
        )


@dataclass(frozen=True)
class ScaleAnchor:
    """A sparse metric observation of one window's global scale."""

    window_id: str
    scale: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if self.scale <= 0 or self.standard_deviation <= 0:
            raise ValueError("scale anchor values must be strictly positive")

    @classmethod
    def from_metric_pair(
        cls,
        window_id: str,
        first_local_point: FloatArray,
        second_local_point: FloatArray,
        metric_distance: float,
        standard_deviation: float,
    ) -> ScaleAnchor:
        local_distance = float(
            np.linalg.norm(
                np.asarray(first_local_point, dtype=np.float64)
                - np.asarray(second_local_point, dtype=np.float64)
            )
        )
        if local_distance <= np.finfo(np.float64).eps:
            raise ValueError("metric anchor points must be distinct")
        return cls(
            window_id=window_id,
            scale=metric_distance / local_distance,
            standard_deviation=standard_deviation,
        )


@dataclass(frozen=True)
class PointAnchor:
    """A sparse associated 3D observation in the global metric frame."""

    window_id: str
    local_point: FloatArray
    global_point: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        local = np.asarray(self.local_point, dtype=np.float64)
        global_point = np.asarray(self.global_point, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if local.shape != (3,) or global_point.shape != (3,):
            raise ValueError("point-anchor coordinates must have shape (3,)")
        if covariance.shape != (3, 3):
            raise ValueError("point-anchor covariance must have shape (3, 3)")
        object.__setattr__(self, "local_point", local)
        object.__setattr__(self, "global_point", global_point)
        object.__setattr__(self, "covariance", covariance)


def _numerical_jacobian(function, vector: FloatArray) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)
    baseline = np.asarray(function(vector), dtype=np.float64)
    jacobian = np.empty((baseline.size, vector.size), dtype=np.float64)
    for index in range(vector.size):
        step = 1e-6 * max(1.0, abs(float(vector[index])))
        perturbed = vector.copy()
        perturbed[index] += step
        jacobian[:, index] = (np.asarray(function(perturbed)) - baseline) / step
    return jacobian


def _compose_with_covariance(
    first: Sim3,
    first_covariance: FloatArray,
    second: Sim3,
    second_covariance: FloatArray,
) -> tuple[Sim3, FloatArray]:
    first_vector = first.as_vector()
    second_vector = second.as_vector()
    output = first.compose(second)
    first_jacobian = _numerical_jacobian(
        lambda vector: Sim3.from_vector(vector).compose(second).as_vector(), first_vector
    )
    second_jacobian = _numerical_jacobian(
        lambda vector: first.compose(Sim3.from_vector(vector)).as_vector(), second_vector
    )
    covariance = (
        first_jacobian @ first_covariance @ first_jacobian.T
        + second_jacobian @ second_covariance @ second_jacobian.T
    )
    return output, 0.5 * (covariance + covariance.T)


def _inverse_with_covariance(transform: Sim3, covariance: FloatArray) -> tuple[Sim3, FloatArray]:
    vector = transform.as_vector()
    inverse = transform.inverse()
    jacobian = _numerical_jacobian(
        lambda value: Sim3.from_vector(value).inverse().as_vector(), vector
    )
    inverse_covariance = jacobian @ covariance @ jacobian.T
    return inverse, 0.5 * (inverse_covariance + inverse_covariance.T)


def _whitener(covariance: FloatArray, floor: float = 1e-10) -> FloatArray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


class SequentialGaugeEstimator:
    """Initialize gauges with deterministic multi-estimate covariance intersection."""

    def __init__(self, *, covariance_intersection_grid_size: int = 21) -> None:
        if covariance_intersection_grid_size < 3:
            raise ValueError("covariance_intersection_grid_size must be at least three")
        self.covariance_intersection_grid_size = covariance_intersection_grid_size

    def estimate(
        self,
        ordered_window_ids: list[str],
        constraints: list[RelativeGaugeConstraint],
        *,
        initial_transform: Sim3 | None = None,
        initial_covariance: FloatArray | None = None,
    ) -> dict[str, GaugeEstimate]:
        if not ordered_window_ids:
            raise ValueError("ordered_window_ids must not be empty")
        if len(set(ordered_window_ids)) != len(ordered_window_ids):
            raise ValueError("window IDs must be unique")
        first_id = ordered_window_ids[0]
        first_transform = initial_transform or Sim3.identity()
        first_covariance = (
            np.diag([1e-10] * 7)
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=np.float64)
        )
        estimates = {first_id: GaugeEstimate(first_id, first_transform, first_covariance)}

        for window_id in ordered_window_ids[1:]:
            candidates: list[tuple[Sim3, FloatArray]] = []
            for constraint in constraints:
                if constraint.moving_id == window_id and constraint.reference_id in estimates:
                    reference = estimates[constraint.reference_id]
                    candidates.append(
                        _compose_with_covariance(
                            reference.global_from_local,
                            reference.covariance,
                            constraint.reference_from_moving,
                            constraint.covariance,
                        )
                    )
                elif constraint.reference_id == window_id and constraint.moving_id in estimates:
                    moving = estimates[constraint.moving_id]
                    inverse, inverse_covariance = _inverse_with_covariance(
                        constraint.reference_from_moving, constraint.covariance
                    )
                    candidates.append(
                        _compose_with_covariance(
                            moving.global_from_local,
                            moving.covariance,
                            inverse,
                            inverse_covariance,
                        )
                    )
            if not candidates:
                raise ValueError(f"window {window_id!r} has no constraint to an initialized gauge")

            minimum_weight = min(0.05, 0.5 / len(candidates))
            transform, covariance, _ = fuse_sim3_covariance_intersection(
                candidates,
                minimum_weight=minimum_weight,
                max_sweeps=max(16, self.covariance_intersection_grid_size),
                line_search_iterations=max(
                    32,
                    2 * self.covariance_intersection_grid_size,
                ),
            )
            estimates[window_id] = GaugeEstimate(window_id, transform, covariance)
        return estimates


def relative_constraint_residual(
    constraint: RelativeGaugeConstraint,
    gauges: dict[str, Sim3],
) -> FloatArray:
    predicted = gauges[constraint.reference_id].inverse().compose(gauges[constraint.moving_id])
    return constraint.reference_from_moving.inverse().compose(predicted).as_vector()


def constraint_cost(
    constraints: list[RelativeGaugeConstraint],
    gauges: dict[str, Sim3],
) -> float:
    cost = 0.0
    for constraint in constraints:
        residual = relative_constraint_residual(constraint, gauges)
        whitened = _whitener(constraint.covariance) @ residual
        cost += float(whitened @ whitened)
    return cost


class FixedLagGaugeSmoother:
    """Rolling nonlinear least-squares smoother over recent ``Sim(3)`` gauges."""

    def __init__(
        self,
        *,
        lag: int = 4,
        max_iterations: int = 10,
        damping: float = 1e-5,
        tolerance: float = 1e-7,
    ) -> None:
        if lag < 2:
            raise ValueError("fixed-lag smoother requires lag >= 2")
        self.lag = lag
        self.max_iterations = max_iterations
        self.damping = damping
        self.tolerance = tolerance

    def smooth(
        self,
        ordered_window_ids: list[str],
        initial_estimates: dict[str, GaugeEstimate],
        constraints: list[RelativeGaugeConstraint],
        *,
        gauge_anchors: list[GaugeAnchor] | None = None,
        scale_anchors: list[ScaleAnchor] | None = None,
        point_anchors: list[PointAnchor] | None = None,
    ) -> dict[str, GaugeEstimate]:
        gauge_anchors = gauge_anchors or []
        scale_anchors = scale_anchors or []
        point_anchors = point_anchors or []
        state = {
            window_id: initial_estimates[window_id].global_from_local
            for window_id in ordered_window_ids
        }
        covariances = {
            window_id: initial_estimates[window_id].covariance.copy()
            for window_id in ordered_window_ids
        }
        order = {window_id: index for index, window_id in enumerate(ordered_window_ids)}

        for end in range(1, len(ordered_window_ids)):
            start = max(1, end - self.lag + 1)
            active_ids = ordered_window_ids[start : end + 1]
            active_set = set(active_ids)
            usable_constraints = [
                constraint
                for constraint in constraints
                if order[constraint.reference_id] <= end
                and order[constraint.moving_id] <= end
                and (constraint.reference_id in active_set or constraint.moving_id in active_set)
            ]
            active_gauge_anchors = [a for a in gauge_anchors if a.window_id in active_set]
            active_scale_anchors = [a for a in scale_anchors if a.window_id in active_set]
            active_point_anchors = [a for a in point_anchors if a.window_id in active_set]
            if not usable_constraints and not (
                active_gauge_anchors or active_scale_anchors or active_point_anchors
            ):
                continue

            initial_vector = np.concatenate(
                [state[window_id].as_vector() for window_id in active_ids]
            )

            def unpack(
                vector: FloatArray,
                active_window_ids: tuple[str, ...] = tuple(active_ids),
            ) -> dict[str, Sim3]:
                result = state.copy()
                for index, window_id in enumerate(active_window_ids):
                    result[window_id] = Sim3.from_vector(vector[7 * index : 7 * (index + 1)])
                return result

            def residual_vector(
                vector: FloatArray,
                relative_factors: tuple[RelativeGaugeConstraint, ...] = tuple(usable_constraints),
                gauge_factors: tuple[GaugeAnchor, ...] = tuple(active_gauge_anchors),
                scale_factors: tuple[ScaleAnchor, ...] = tuple(active_scale_anchors),
                point_factors: tuple[PointAnchor, ...] = tuple(active_point_anchors),
            ) -> FloatArray:
                gauges = unpack(vector)
                residual_parts: list[FloatArray] = []
                for constraint in relative_factors:
                    residual = relative_constraint_residual(constraint, gauges)
                    residual_parts.append(_whitener(constraint.covariance) @ residual)
                for anchor in gauge_factors:
                    error = (
                        anchor.global_from_local.inverse()
                        .compose(gauges[anchor.window_id])
                        .as_vector()
                    )
                    residual_parts.append(_whitener(anchor.covariance) @ error)
                for anchor in scale_factors:
                    residual_parts.append(
                        np.array(
                            [
                                (np.log(gauges[anchor.window_id].scale) - np.log(anchor.scale))
                                / anchor.standard_deviation
                            ]
                        )
                    )
                for anchor in point_factors:
                    error = (
                        gauges[anchor.window_id].transform_points(anchor.local_point)
                        - anchor.global_point
                    )
                    residual_parts.append(_whitener(anchor.covariance) @ error)
                return np.concatenate(residual_parts)

            vector = initial_vector
            hessian = np.eye(vector.size)
            for _ in range(self.max_iterations):
                residual = residual_vector(vector)
                jacobian = _numerical_jacobian(residual_vector, vector)
                hessian = jacobian.T @ jacobian + self.damping * np.eye(vector.size)
                gradient = jacobian.T @ residual
                try:
                    increment = -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    increment = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
                if np.linalg.norm(increment) < self.tolerance:
                    break
                current_cost = float(residual @ residual)
                accepted = False
                for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    candidate = vector + fraction * increment
                    candidate_residual = residual_vector(candidate)
                    if float(candidate_residual @ candidate_residual) < current_cost:
                        vector = candidate
                        accepted = True
                        break
                if not accepted:
                    break

            optimized = unpack(vector)
            joint_covariance = np.linalg.pinv(hessian, rcond=1e-10)
            for index, window_id in enumerate(active_ids):
                state[window_id] = optimized[window_id]
                block = joint_covariance[7 * index : 7 * (index + 1), 7 * index : 7 * (index + 1)]
                covariances[window_id] = 0.5 * (block + block.T)

        return {
            window_id: GaugeEstimate(window_id, state[window_id], covariances[window_id])
            for window_id in ordered_window_ids
        }
