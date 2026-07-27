"""Fixed-lag gauge smoothing with an uncertainty-preserving boundary prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .covariance import regularized_inverse_psd
from .gauge import (
    GaugeAnchor,
    GaugeEstimate,
    PointAnchor,
    RelativeGaugeConstraint,
    ScaleAnchor,
)
from .sim3 import Sim3

FloatArray = NDArray[np.floating]


def _validated_psd(
    values: FloatArray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    readonly: bool = True,
) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    if not np.allclose(matrix, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), np.finfo(float).tiny)
    tolerance = 1e-14 + 1e-10 * scale
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if np.any(eigenvalues < 0.0):
        symmetric = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    else:
        symmetric = symmetric.copy()
    if readonly:
        symmetric.setflags(write=False)
    return symmetric


@dataclass(frozen=True)
class _QuadraticPrior:
    window_ids: tuple[str, ...]
    mean: FloatArray
    information: FloatArray

    def __post_init__(self) -> None:
        dimension = 7 * len(self.window_ids)
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("fixed-lag prior mean has changed shape or contains non-finite values")
        information = _validated_psd(
            self.information,
            name="fixed-lag boundary information",
            shape=(dimension, dimension),
        )
        mean.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "information", information)


def _central_jacobian(function, vector: FloatArray) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)
    baseline = np.asarray(function(vector), dtype=np.float64)
    jacobian = np.empty((baseline.size, vector.size), dtype=np.float64)
    for index in range(vector.size):
        step = 1e-6 * max(1.0, abs(float(vector[index])))
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.asarray(function(plus), dtype=np.float64)
            - np.asarray(function(minus), dtype=np.float64)
        ) / (2.0 * step)
    return jacobian


def _information_square_root(information: FloatArray) -> FloatArray:
    matrix = _validated_psd(
        information,
        name="fixed-lag boundary information",
        readonly=False,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    retained = np.where(eigenvalues > spectral_scale * 1e-12, eigenvalues, 0.0)
    return np.sqrt(retained)[:, None] * eigenvectors.T


def _whitener(covariance: FloatArray, *, name: str) -> FloatArray:
    information = regularized_inverse_psd(covariance, name=name, eigenvalue_floor=1e-12)
    return _information_square_root(information)


def _relative_constraint_residual(
    constraint: RelativeGaugeConstraint,
    gauges: dict[str, Sim3],
) -> FloatArray:
    predicted = gauges[constraint.reference_id].inverse().compose(gauges[constraint.moving_id])
    return constraint.reference_from_moving.inverse().compose(predicted).as_vector()


class MarginalizedFixedLagGaugeSmoother:
    """Rolling nonlinear smoother with a Schur-complement boundary prior.

    Unlike the legacy fixed-lag implementation, an expired gauge is not fixed at
    its posterior mean with zero uncertainty. Factors touching that gauge and the
    previous boundary prior are linearized once, marginalized with a Schur
    complement, and carried as a quadratic prior on the remaining active gauges.
    The returned objects still expose marginal ``7 x 7`` covariances only; a
    portable all-window export therefore remains unable to reconstruct historical
    cross-window covariance from these marginals alone.
    """

    def __init__(
        self,
        *,
        lag: int = 4,
        max_iterations: int = 10,
        damping: float = 1e-5,
        tolerance: float = 1e-7,
        covariance_floor: float = 1e-12,
    ) -> None:
        if lag < 2:
            raise ValueError("fixed-lag smoother requires lag >= 2")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be finite and positive")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(covariance_floor) or covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be finite and positive")
        self.lag = lag
        self.max_iterations = max_iterations
        self.damping = damping
        self.tolerance = tolerance
        self.covariance_floor = covariance_floor

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
        gauge_anchors = list(gauge_anchors or [])
        scale_anchors = list(scale_anchors or [])
        point_anchors = list(point_anchors or [])
        self._validate_inputs(
            ordered_window_ids,
            initial_estimates,
            constraints,
            gauge_anchors,
            scale_anchors,
            point_anchors,
        )
        order = {window_id: index for index, window_id in enumerate(ordered_window_ids)}
        state = {
            window_id: initial_estimates[window_id].global_from_local
            for window_id in ordered_window_ids
        }
        covariances = {
            window_id: np.asarray(initial_estimates[window_id].covariance).copy()
            for window_id in ordered_window_ids
        }

        constraints_by_arrival: list[list[RelativeGaugeConstraint]] = [
            [] for _ in ordered_window_ids
        ]
        for constraint in sorted(
            constraints,
            key=lambda item: (
                max(order[item.reference_id], order[item.moving_id]),
                item.reference_id,
                item.moving_id,
            ),
        ):
            arrival = max(order[constraint.reference_id], order[constraint.moving_id])
            constraints_by_arrival[arrival].append(constraint)
        gauge_anchors_by_arrival = self._anchors_by_arrival(
            ordered_window_ids,
            gauge_anchors,
        )
        scale_anchors_by_arrival = self._anchors_by_arrival(
            ordered_window_ids,
            scale_anchors,
        )
        point_anchors_by_arrival = self._anchors_by_arrival(
            ordered_window_ids,
            point_anchors,
        )

        first_id = ordered_window_ids[0]
        first_information = regularized_inverse_psd(
            initial_estimates[first_id].covariance,
            name="initial fixed-lag gauge covariance",
            eigenvalue_floor=self.covariance_floor,
        )
        prior = _QuadraticPrior(
            window_ids=(first_id,),
            mean=state[first_id].as_vector(),
            information=first_information,
        )
        active_constraints: list[RelativeGaugeConstraint] = []
        active_gauge_anchors: list[GaugeAnchor] = []
        active_scale_anchors: list[ScaleAnchor] = []
        active_point_anchors: list[PointAnchor] = []

        for end in range(len(ordered_window_ids)):
            start = max(0, end - self.lag + 1)
            active_ids = ordered_window_ids[start : end + 1]
            active_constraints.extend(constraints_by_arrival[end])
            active_gauge_anchors.extend(gauge_anchors_by_arrival[end])
            active_scale_anchors.extend(scale_anchors_by_arrival[end])
            active_point_anchors.extend(point_anchors_by_arrival[end])
            prior = self._expand_prior(prior, active_ids, state)
            vector, joint_covariance = self._optimize(
                active_ids,
                state,
                prior,
                active_constraints,
                active_gauge_anchors,
                active_scale_anchors,
                active_point_anchors,
            )
            for index, window_id in enumerate(active_ids):
                state[window_id] = Sim3.from_vector(vector[7 * index : 7 * (index + 1)])
                block = joint_covariance[
                    7 * index : 7 * (index + 1),
                    7 * index : 7 * (index + 1),
                ]
                covariances[window_id] = _validated_psd(
                    block,
                    name=f"fixed-lag gauge covariance for {window_id}",
                    shape=(7, 7),
                    readonly=False,
                )

            if end == len(ordered_window_ids) - 1 or len(active_ids) < self.lag:
                continue
            expired_id = active_ids[0]
            touching_constraints = [
                factor
                for factor in active_constraints
                if expired_id in (factor.reference_id, factor.moving_id)
            ]
            touching_gauge_anchors = [
                factor for factor in active_gauge_anchors if factor.window_id == expired_id
            ]
            touching_scale_anchors = [
                factor for factor in active_scale_anchors if factor.window_id == expired_id
            ]
            touching_point_anchors = [
                factor for factor in active_point_anchors if factor.window_id == expired_id
            ]
            prior = self._marginalize_oldest(
                active_ids,
                state,
                prior,
                touching_constraints,
                touching_gauge_anchors,
                touching_scale_anchors,
                touching_point_anchors,
            )
            active_constraints = self._remove_by_identity(
                active_constraints,
                touching_constraints,
            )
            active_gauge_anchors = self._remove_by_identity(
                active_gauge_anchors,
                touching_gauge_anchors,
            )
            active_scale_anchors = self._remove_by_identity(
                active_scale_anchors,
                touching_scale_anchors,
            )
            active_point_anchors = self._remove_by_identity(
                active_point_anchors,
                touching_point_anchors,
            )

        return {
            window_id: GaugeEstimate(window_id, state[window_id], covariances[window_id])
            for window_id in ordered_window_ids
        }

    def _validate_inputs(
        self,
        ordered_window_ids: list[str],
        initial_estimates: dict[str, GaugeEstimate],
        constraints: list[RelativeGaugeConstraint],
        gauge_anchors: list[GaugeAnchor],
        scale_anchors: list[ScaleAnchor],
        point_anchors: list[PointAnchor],
    ) -> None:
        if not ordered_window_ids:
            raise ValueError("ordered_window_ids must not be empty")
        if len(set(ordered_window_ids)) != len(ordered_window_ids):
            raise ValueError("window IDs must be unique")
        if set(initial_estimates) != set(ordered_window_ids):
            raise ValueError("initial estimates must match the ordered window IDs")
        order = {window_id: index for index, window_id in enumerate(ordered_window_ids)}
        for constraint in constraints:
            if constraint.reference_id not in order or constraint.moving_id not in order:
                raise ValueError("relative constraint references an unknown window")
            span = abs(order[constraint.reference_id] - order[constraint.moving_id])
            if span >= self.lag:
                raise ValueError(
                    "relative constraint span reaches beyond the fixed lag; increase lag "
                    "so every factor arrives before either endpoint is marginalized"
                )
        for anchor in [*gauge_anchors, *scale_anchors, *point_anchors]:
            if anchor.window_id not in order:
                raise ValueError("gauge anchor references an unknown window")

    @staticmethod
    def _anchors_by_arrival(
        ordered_window_ids: list[str],
        anchors: list,
    ) -> list[list]:
        order = {window_id: index for index, window_id in enumerate(ordered_window_ids)}
        grouped: list[list] = [[] for _ in ordered_window_ids]
        for anchor in sorted(anchors, key=lambda item: item.window_id):
            grouped[order[anchor.window_id]].append(anchor)
        return grouped

    @staticmethod
    def _remove_by_identity(values: list, removed: list) -> list:
        removed_ids = {id(value) for value in removed}
        return [value for value in values if id(value) not in removed_ids]

    @staticmethod
    def _expand_prior(
        prior: _QuadraticPrior,
        active_ids: list[str],
        state: dict[str, Sim3],
    ) -> _QuadraticPrior:
        position = {window_id: index for index, window_id in enumerate(active_ids)}
        if any(window_id not in position for window_id in prior.window_ids):
            raise RuntimeError("fixed-lag prior references a gauge outside the active window")
        mean = np.concatenate([state[window_id].as_vector() for window_id in active_ids])
        information = np.zeros((mean.size, mean.size), dtype=np.float64)
        for row, row_id in enumerate(prior.window_ids):
            row_target = slice(7 * position[row_id], 7 * (position[row_id] + 1))
            mean[row_target] = prior.mean[7 * row : 7 * (row + 1)]
            for column, column_id in enumerate(prior.window_ids):
                column_target = slice(
                    7 * position[column_id],
                    7 * (position[column_id] + 1),
                )
                information[row_target, column_target] = prior.information[
                    7 * row : 7 * (row + 1),
                    7 * column : 7 * (column + 1),
                ]
        return _QuadraticPrior(tuple(active_ids), mean, information)

    def _residual_function(
        self,
        active_ids: list[str],
        state: dict[str, Sim3],
        prior: _QuadraticPrior,
        constraints: list[RelativeGaugeConstraint],
        gauge_anchors: list[GaugeAnchor],
        scale_anchors: list[ScaleAnchor],
        point_anchors: list[PointAnchor],
    ):
        prior_root = _information_square_root(prior.information)
        active_tuple = tuple(active_ids)

        def unpack(vector: FloatArray) -> dict[str, Sim3]:
            gauges = state.copy()
            for index, window_id in enumerate(active_tuple):
                gauges[window_id] = Sim3.from_vector(vector[7 * index : 7 * (index + 1)])
            return gauges

        def residual(vector: FloatArray) -> FloatArray:
            gauges = unpack(vector)
            parts: list[FloatArray] = [prior_root @ (vector - prior.mean)]
            for constraint in constraints:
                error = _relative_constraint_residual(constraint, gauges)
                parts.append(
                    _whitener(
                        constraint.covariance,
                        name="relative gauge covariance",
                    )
                    @ error
                )
            for anchor in gauge_anchors:
                error = (
                    anchor.global_from_local.inverse()
                    .compose(gauges[anchor.window_id])
                    .as_vector()
                )
                parts.append(
                    _whitener(anchor.covariance, name="gauge-anchor covariance") @ error
                )
            for anchor in scale_anchors:
                parts.append(
                    np.asarray(
                        [
                            (
                                np.log(gauges[anchor.window_id].scale)
                                - np.log(anchor.scale)
                            )
                            / anchor.standard_deviation
                        ]
                    )
                )
            for anchor in point_anchors:
                error = (
                    gauges[anchor.window_id].transform_points(anchor.local_point)
                    - anchor.global_point
                )
                parts.append(
                    _whitener(anchor.covariance, name="point-anchor covariance") @ error
                )
            return np.concatenate(parts)

        return residual

    def _optimize(
        self,
        active_ids: list[str],
        state: dict[str, Sim3],
        prior: _QuadraticPrior,
        constraints: list[RelativeGaugeConstraint],
        gauge_anchors: list[GaugeAnchor],
        scale_anchors: list[ScaleAnchor],
        point_anchors: list[PointAnchor],
    ) -> tuple[FloatArray, FloatArray]:
        vector = np.concatenate([state[window_id].as_vector() for window_id in active_ids])
        residual_function = self._residual_function(
            active_ids,
            state,
            prior,
            constraints,
            gauge_anchors,
            scale_anchors,
            point_anchors,
        )
        for _ in range(self.max_iterations):
            residual = residual_function(vector)
            jacobian = _central_jacobian(residual_function, vector)
            hessian = 0.5 * (jacobian.T @ jacobian + (jacobian.T @ jacobian).T)
            gradient = jacobian.T @ residual
            system = hessian + self.damping * np.eye(vector.size)
            try:
                increment = -np.linalg.solve(system, gradient)
            except np.linalg.LinAlgError:
                increment = -np.linalg.lstsq(system, gradient, rcond=None)[0]
            if np.linalg.norm(increment) < self.tolerance:
                break
            current_cost = float(residual @ residual)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = vector + fraction * increment
                candidate_residual = residual_function(candidate)
                if float(candidate_residual @ candidate_residual) < current_cost:
                    vector = candidate
                    accepted = True
                    break
            if not accepted:
                break

        residual = residual_function(vector)
        jacobian = _central_jacobian(residual_function, vector)
        hessian = jacobian.T @ jacobian
        hessian = _validated_psd(
            0.5 * (hessian + hessian.T),
            name="fixed-lag normal information",
            readonly=False,
        )
        covariance = regularized_inverse_psd(
            hessian,
            name="fixed-lag normal information",
            eigenvalue_floor=self.covariance_floor,
        )
        return vector, covariance

    def _marginalize_oldest(
        self,
        active_ids: list[str],
        state: dict[str, Sim3],
        prior: _QuadraticPrior,
        constraints: list[RelativeGaugeConstraint],
        gauge_anchors: list[GaugeAnchor],
        scale_anchors: list[ScaleAnchor],
        point_anchors: list[PointAnchor],
    ) -> _QuadraticPrior:
        vector = np.concatenate([state[window_id].as_vector() for window_id in active_ids])
        residual_function = self._residual_function(
            active_ids,
            state,
            prior,
            constraints,
            gauge_anchors,
            scale_anchors,
            point_anchors,
        )
        residual = residual_function(vector)
        jacobian = _central_jacobian(residual_function, vector)
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        hessian = _validated_psd(
            0.5 * (hessian + hessian.T),
            name="fixed-lag marginalization information",
            readonly=False,
        )
        old_information = hessian[:7, :7]
        old_to_retained = hessian[:7, 7:]
        retained_to_old = hessian[7:, :7]
        retained_information = hessian[7:, 7:]
        old_gradient = gradient[:7]
        retained_gradient = gradient[7:]
        old_covariance = regularized_inverse_psd(
            old_information,
            name="expired fixed-lag gauge information",
            eigenvalue_floor=self.covariance_floor,
        )
        information = (
            retained_information
            - retained_to_old @ old_covariance @ old_to_retained
        )
        gradient = retained_gradient - retained_to_old @ old_covariance @ old_gradient
        information = _validated_psd(
            0.5 * (information + information.T),
            name="marginalized fixed-lag boundary information",
            readonly=False,
        )
        covariance = regularized_inverse_psd(
            information,
            name="marginalized fixed-lag boundary information",
            eigenvalue_floor=self.covariance_floor,
        )
        mean = vector[7:] - covariance @ gradient
        return _QuadraticPrior(tuple(active_ids[1:]), mean, information)


__all__ = ["MarginalizedFixedLagGaugeSmoother"]
