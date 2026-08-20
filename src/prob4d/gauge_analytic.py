"""Explicit analytic covariance propagation for recursive ``Sim(3)`` gauges.

The historical estimator in :mod:`prob4d.gauge` retains finite-difference
covariance propagation for compatibility. This additive module exposes an
experimental version-2 estimator whose composition and inversion derivatives
are analytic and fail closed at the SO(3) logarithm branch cut.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .composition_jacobian import analytic_sim3_compose_jacobians
from .covariance import validated_covariance_psd
from .gauge import (
    GaugeEstimate,
    RelativeGaugeConstraint,
    _estimate_sequential_gauges,
)
from .sim3 import Sim3, skew, so3_log, so3_right_jacobian

FloatArray = NDArray[np.floating]

ANALYTIC_GAUGE_PROPAGATION_METHOD = "analytic_sim3_compose_inverse_v1"
ANALYTIC_GAUGE_PROPAGATION_VERSION = 1


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def analytic_sim3_inverse_jacobian(
    transform: Sim3,
    *,
    branch_cut_tolerance: float = 1e-7,
) -> FloatArray:
    """Return the exact derivative of ``transform.inverse().as_vector()``.

    Coordinates are ``[log_scale, rotation_vector(3), translation(3)]``. The
    principal SO(3) logarithm is non-differentiable at rotations of angle pi, so
    those inputs fail closed instead of producing platform-dependent covariance.
    """

    tolerance = _finite_nonnegative_real(
        branch_cut_tolerance,
        name="branch_cut_tolerance",
    )
    rotation_vector = so3_log(transform.rotation)
    if np.pi - float(np.linalg.norm(rotation_vector)) <= tolerance:
        raise ValueError("Sim(3) inverse Jacobian is undefined at the SO(3) log branch cut")

    inverse = transform.inverse()
    jacobian = np.zeros((7, 7), dtype=np.float64)
    jacobian[0, 0] = -1.0
    jacobian[1:4, 1:4] = -np.eye(3)
    jacobian[4:7, 0] = -inverse.translation
    jacobian[4:7, 1:4] = skew(inverse.translation) @ so3_right_jacobian(rotation_vector)
    jacobian[4:7, 4:7] = -(1.0 / transform.scale) * transform.rotation.T
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("analytic Sim(3) inverse Jacobian is non-finite")
    return jacobian


def compose_sim3_with_covariance_analytic(
    first: Sim3,
    first_covariance: FloatArray,
    second: Sim3,
    second_covariance: FloatArray,
    *,
    branch_cut_tolerance: float = 1e-7,
) -> tuple[Sim3, FloatArray]:
    """Compose independent uncertain transforms with analytic first derivatives."""

    first_covariance = validated_covariance_psd(
        first_covariance,
        name="first gauge covariance",
        shape=(7, 7),
        readonly=False,
    )
    second_covariance = validated_covariance_psd(
        second_covariance,
        name="second gauge covariance",
        shape=(7, 7),
        readonly=False,
    )
    first_jacobian, second_jacobian = analytic_sim3_compose_jacobians(
        first,
        second,
        branch_cut_tolerance=branch_cut_tolerance,
    )
    output = first.compose(second)
    covariance = (
        first_jacobian @ first_covariance @ first_jacobian.T
        + second_jacobian @ second_covariance @ second_jacobian.T
    )
    return output, validated_covariance_psd(
        0.5 * (covariance + covariance.T),
        name="composed gauge covariance",
        shape=(7, 7),
        readonly=False,
    )


def invert_sim3_with_covariance_analytic(
    transform: Sim3,
    covariance: FloatArray,
    *,
    branch_cut_tolerance: float = 1e-7,
) -> tuple[Sim3, FloatArray]:
    """Invert one uncertain transform with an analytic first derivative."""

    covariance = validated_covariance_psd(
        covariance,
        name="gauge covariance",
        shape=(7, 7),
        readonly=False,
    )
    jacobian = analytic_sim3_inverse_jacobian(
        transform,
        branch_cut_tolerance=branch_cut_tolerance,
    )
    inverse = transform.inverse()
    inverse_covariance = jacobian @ covariance @ jacobian.T
    return inverse, validated_covariance_psd(
        0.5 * (inverse_covariance + inverse_covariance.T),
        name="inverse gauge covariance",
        shape=(7, 7),
        readonly=False,
    )


class AnalyticSequentialGaugeEstimatorV2:
    """Sequential gauge initialization with analytic covariance propagation.

    Transform estimation and covariance-intersection semantics match
    :class:`prob4d.gauge.SequentialGaugeEstimator`; only composition and inverse
    covariance derivatives change. The historical estimator remains untouched.
    """

    jacobian_method = ANALYTIC_GAUGE_PROPAGATION_METHOD

    def __init__(
        self,
        *,
        covariance_intersection_grid_size: int = 21,
        branch_cut_tolerance: float = 1e-7,
    ) -> None:
        if isinstance(covariance_intersection_grid_size, bool) or not isinstance(
            covariance_intersection_grid_size,
            (int, np.integer),
        ):
            raise TypeError("covariance_intersection_grid_size must be a genuine integer")
        if covariance_intersection_grid_size < 3:
            raise ValueError("covariance_intersection_grid_size must be at least three")
        self.covariance_intersection_grid_size = int(covariance_intersection_grid_size)
        self.branch_cut_tolerance = _finite_nonnegative_real(
            branch_cut_tolerance,
            name="branch_cut_tolerance",
        )

    def estimate(
        self,
        ordered_window_ids: list[str],
        constraints: list[RelativeGaugeConstraint],
        *,
        initial_transform: Sim3 | None = None,
        initial_covariance: FloatArray | None = None,
    ) -> dict[str, GaugeEstimate]:
        def prepare_initial_covariance(covariance: FloatArray) -> FloatArray:
            return validated_covariance_psd(
                covariance,
                name="initial gauge covariance",
                shape=(7, 7),
                readonly=False,
            )

        def compose_with_covariance(
            first: Sim3,
            first_covariance: FloatArray,
            second: Sim3,
            second_covariance: FloatArray,
        ) -> tuple[Sim3, FloatArray]:
            return compose_sim3_with_covariance_analytic(
                first,
                first_covariance,
                second,
                second_covariance,
                branch_cut_tolerance=self.branch_cut_tolerance,
            )

        def invert_with_covariance(
            transform: Sim3,
            covariance: FloatArray,
        ) -> tuple[Sim3, FloatArray]:
            return invert_sim3_with_covariance_analytic(
                transform,
                covariance,
                branch_cut_tolerance=self.branch_cut_tolerance,
            )

        return _estimate_sequential_gauges(
            ordered_window_ids,
            constraints,
            covariance_intersection_grid_size=self.covariance_intersection_grid_size,
            compose_with_covariance=compose_with_covariance,
            invert_with_covariance=invert_with_covariance,
            initial_transform=initial_transform,
            initial_covariance=initial_covariance,
            prepare_initial_covariance=prepare_initial_covariance,
        )


__all__ = [
    "ANALYTIC_GAUGE_PROPAGATION_METHOD",
    "ANALYTIC_GAUGE_PROPAGATION_VERSION",
    "AnalyticSequentialGaugeEstimatorV2",
    "analytic_sim3_inverse_jacobian",
    "compose_sim3_with_covariance_analytic",
    "invert_sim3_with_covariance_analytic",
]
