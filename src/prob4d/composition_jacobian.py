"""Versioned Jacobians for seven-coordinate ``Sim(3)`` composition.

The legacy central-finite-difference implementation and the provider-v2 analytic
implementation live behind explicit callables.  A task-local mode remains for
backward-compatible internal callers, but importing this module never replaces
functions in :mod:`prob4d.observation_export`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import numpy as np

from .sim3 import Sim3, skew, so3_log, so3_right_jacobian

CompositionJacobianMode = Literal["legacy_finite_difference", "analytic"]
CompositionJacobianFunction = Callable[
    [Sim3, Sim3],
    tuple[np.ndarray, np.ndarray],
]
COMPOSITION_JACOBIAN_MODES: tuple[CompositionJacobianMode, ...] = (
    "legacy_finite_difference",
    "analytic",
)
_MODE: ContextVar[CompositionJacobianMode] = ContextVar(
    "prob4d_composition_jacobian_mode",
    default="legacy_finite_difference",
)


def _validated_mode(mode: CompositionJacobianMode) -> CompositionJacobianMode:
    if mode not in COMPOSITION_JACOBIAN_MODES:
        raise ValueError(
            "composition Jacobian mode must be one of "
            f"{COMPOSITION_JACOBIAN_MODES}"
        )
    return mode


def _numerical_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    vector: np.ndarray,
) -> np.ndarray:
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


def legacy_sim3_compose_jacobians(
    parent: Sim3,
    relative: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen provider-v1 central-finite-difference Jacobians."""

    parent_vector = parent.as_vector()
    relative_vector = relative.as_vector()
    parent_jacobian = _numerical_jacobian(
        lambda value: Sim3.from_vector(value).compose(relative).as_vector(),
        parent_vector,
    )
    relative_jacobian = _numerical_jacobian(
        lambda value: parent.compose(Sim3.from_vector(value)).as_vector(),
        relative_vector,
    )
    return parent_jacobian, relative_jacobian


# Retained for source compatibility with focused tests and diagnostic modules.
_LEGACY_COMPOSE_JACOBIANS = legacy_sim3_compose_jacobians


def so3_right_jacobian_inverse(rotation_vector: np.ndarray) -> np.ndarray:
    """Return the inverse SO(3) right Jacobian in axis-angle coordinates."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    generator = skew(vector)
    if angle < 1e-4:
        coefficient = 1.0 / 12.0 + angle**2 / 720.0 + angle**4 / 30_240.0
    else:
        coefficient = 1.0 / angle**2 - 1.0 / (
            2.0 * angle * np.tan(0.5 * angle)
        )
    return np.eye(3) + 0.5 * generator + coefficient * generator @ generator


def analytic_sim3_compose_jacobians(
    parent: Sim3,
    relative: Sim3,
    *,
    branch_cut_tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact first derivatives of ``parent.compose(relative)``.

    Coordinates follow :meth:`prob4d.sim3.Sim3.as_vector`:
    ``[log_scale, rotation_vector(3), translation(3)]``. Axis-angle coordinates
    are non-differentiable at the SO(3) logarithm's pi branch cut, so ambiguous
    parent, relative, or composed coordinates fail closed rather than producing
    platform-dependent covariance.
    """

    tolerance = float(branch_cut_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("branch_cut_tolerance must be finite and non-negative")

    parent_rotation = so3_log(parent.rotation)
    relative_rotation = so3_log(relative.rotation)
    output_rotation = so3_log(parent.rotation @ relative.rotation)
    for label, vector in (
        ("parent", parent_rotation),
        ("relative", relative_rotation),
        ("composed", output_rotation),
    ):
        if np.pi - float(np.linalg.norm(vector)) <= tolerance:
            raise ValueError(
                f"Sim(3) composition Jacobian is undefined at the {label} "
                "SO(3) log branch cut"
            )

    output_right_inverse = so3_right_jacobian_inverse(output_rotation)
    parent_right = so3_right_jacobian(parent_rotation)
    relative_right = so3_right_jacobian(relative_rotation)

    parent_jacobian: np.ndarray = np.zeros((7, 7), dtype=np.float64)
    relative_jacobian: np.ndarray = np.zeros((7, 7), dtype=np.float64)
    parent_jacobian[0, 0] = 1.0
    relative_jacobian[0, 0] = 1.0
    parent_jacobian[1:4, 1:4] = (
        output_right_inverse @ relative.rotation.T @ parent_right
    )
    relative_jacobian[1:4, 1:4] = output_right_inverse @ relative_right

    transported_translation = parent.scale * (
        parent.rotation @ relative.translation
    )
    parent_jacobian[4:7, 0] = transported_translation
    parent_jacobian[4:7, 1:4] = (
        -parent.scale
        * parent.rotation
        @ skew(relative.translation)
        @ parent_right
    )
    parent_jacobian[4:7, 4:7] = np.eye(3)
    relative_jacobian[4:7, 4:7] = parent.scale * parent.rotation

    if not np.all(np.isfinite(parent_jacobian)) or not np.all(
        np.isfinite(relative_jacobian)
    ):
        raise ValueError("analytic Sim(3) composition Jacobian is non-finite")
    return parent_jacobian, relative_jacobian


def composition_jacobian_function(
    mode: CompositionJacobianMode,
) -> CompositionJacobianFunction:
    """Return the implementation associated with one declared mode."""

    selected = _validated_mode(mode)
    if selected == "analytic":
        return analytic_sim3_compose_jacobians
    return legacy_sim3_compose_jacobians


def compose_jacobians_for_mode(
    mode: CompositionJacobianMode,
    parent: Sim3,
    relative: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one explicitly declared composition-Jacobian mode."""

    return composition_jacobian_function(mode)(parent, relative)


def current_composition_jacobian_mode() -> CompositionJacobianMode:
    """Return the task-local compatibility mode."""

    return _MODE.get()


@contextmanager
def composition_jacobian_mode(
    mode: CompositionJacobianMode,
) -> Iterator[None]:
    """Select a compatibility mode without process-global function mutation."""

    token = _MODE.set(_validated_mode(mode))
    try:
        yield
    finally:
        _MODE.reset(token)


def _dispatch_compose_jacobians(
    parent: Sim3,
    relative: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility dispatcher used by the stable exporter wrapper."""

    return compose_jacobians_for_mode(
        current_composition_jacobian_mode(),
        parent,
        relative,
    )


__all__ = [
    "COMPOSITION_JACOBIAN_MODES",
    "CompositionJacobianFunction",
    "CompositionJacobianMode",
    "analytic_sim3_compose_jacobians",
    "compose_jacobians_for_mode",
    "composition_jacobian_function",
    "composition_jacobian_mode",
    "current_composition_jacobian_mode",
    "legacy_sim3_compose_jacobians",
    "so3_right_jacobian_inverse",
]
