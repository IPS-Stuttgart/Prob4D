"""Context-local analytic Jacobians for seven-coordinate ``Sim(3)`` composition.

Provider v1 retains the historical central-finite-difference implementation.
Provider v2 selects the analytic implementation through a task-local context,
so importing the new provider cannot silently reinterpret frozen v1 artifacts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import numpy as np

from . import observation_export as _observation_export
from .sim3 import Sim3, skew, so3_log, so3_right_jacobian

CompositionJacobianMode = Literal["legacy_finite_difference", "analytic"]
COMPOSITION_JACOBIAN_MODES: tuple[CompositionJacobianMode, ...] = (
    "legacy_finite_difference",
    "analytic",
)
_MODE: ContextVar[CompositionJacobianMode] = ContextVar(
    "prob4d_composition_jacobian_mode",
    default="legacy_finite_difference",
)
_LEGACY_COMPOSE_JACOBIANS = _observation_export._compose_jacobians


def so3_right_jacobian_inverse(rotation_vector: np.ndarray) -> np.ndarray:
    """Return the inverse SO(3) right Jacobian in axis-angle coordinates."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    generator = skew(vector)
    if angle < 1e-4:
        coefficient = (
            1.0 / 12.0
            + angle**2 / 720.0
            + angle**4 / 30_240.0
        )
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

    parent_jacobian = np.zeros((7, 7), dtype=np.float64)
    relative_jacobian = np.zeros((7, 7), dtype=np.float64)
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


def current_composition_jacobian_mode() -> CompositionJacobianMode:
    """Return the task-local composition-Jacobian mode."""

    return _MODE.get()


@contextmanager
def composition_jacobian_mode(
    mode: CompositionJacobianMode,
) -> Iterator[None]:
    """Select composition derivatives without process-global mode leakage."""

    if mode not in COMPOSITION_JACOBIAN_MODES:
        raise ValueError(
            f"composition Jacobian mode must be one of {COMPOSITION_JACOBIAN_MODES}"
        )
    token = _MODE.set(mode)
    try:
        yield
    finally:
        _MODE.reset(token)


def _dispatch_compose_jacobians(
    parent: Sim3,
    relative: Sim3,
) -> tuple[np.ndarray, np.ndarray]:
    if current_composition_jacobian_mode() == "analytic":
        return analytic_sim3_compose_jacobians(parent, relative)
    return _LEGACY_COMPOSE_JACOBIANS(parent, relative)


def _install_dispatcher() -> None:
    current = _observation_export._compose_jacobians
    if current is _dispatch_compose_jacobians:
        return
    if current is not _LEGACY_COMPOSE_JACOBIANS:
        raise RuntimeError("Prob4D composition-Jacobian implementation changed unexpectedly")
    _observation_export._compose_jacobians = _dispatch_compose_jacobians


_install_dispatcher()


__all__ = [
    "COMPOSITION_JACOBIAN_MODES",
    "CompositionJacobianMode",
    "analytic_sim3_compose_jacobians",
    "composition_jacobian_mode",
    "current_composition_jacobian_mode",
    "so3_right_jacobian_inverse",
]
