from __future__ import annotations

from contextvars import copy_context

import numpy as np
import pytest

import prob4d.observation_export as observation_export
from prob4d.composition_jacobian import (
    _LEGACY_COMPOSE_JACOBIANS,
    analytic_sim3_compose_jacobians,
    composition_jacobian_mode,
    current_composition_jacobian_mode,
    so3_right_jacobian_inverse,
)
from prob4d.sim3 import Sim3, so3_exp, so3_right_jacobian


def _random_sim3(rng: np.random.Generator) -> Sim3:
    return Sim3(
        scale=float(np.exp(rng.normal(scale=0.35))),
        rotation=so3_exp(rng.normal(size=3) * 0.55),
        translation=rng.normal(size=3),
    )


def test_right_jacobian_inverse_is_an_inverse() -> None:
    for vector in (
        np.zeros(3),
        np.asarray([1e-8, -2e-8, 3e-8]),
        np.asarray([0.2, -0.3, 0.4]),
        np.asarray([2.5, 0.1, -0.2]),
    ):
        inverse = so3_right_jacobian_inverse(vector)
        np.testing.assert_allclose(
            inverse @ so3_right_jacobian(vector),
            np.eye(3),
            atol=2e-12,
            rtol=2e-12,
        )


def test_analytic_composition_jacobians_match_finite_differences() -> None:
    rng = np.random.default_rng(2718)
    checked = 0
    while checked < 64:
        parent = _random_sim3(rng)
        relative = _random_sim3(rng)
        output_angle = np.linalg.norm(parent.compose(relative).as_vector()[1:4])
        if np.pi - output_angle < 1e-3:
            continue
        analytic_parent, analytic_relative = analytic_sim3_compose_jacobians(
            parent,
            relative,
        )
        numeric_parent, numeric_relative = _LEGACY_COMPOSE_JACOBIANS(
            parent,
            relative,
        )
        np.testing.assert_allclose(
            analytic_parent,
            numeric_parent,
            atol=4e-8,
            rtol=4e-8,
        )
        np.testing.assert_allclose(
            analytic_relative,
            numeric_relative,
            atol=4e-8,
            rtol=4e-8,
        )
        checked += 1


def test_analytic_composition_jacobians_handle_near_identity_rotation() -> None:
    parent = Sim3(
        scale=1.2,
        rotation=so3_exp(np.asarray([1e-10, -2e-10, 3e-10])),
        translation=np.asarray([0.1, -0.2, 0.3]),
    )
    relative = Sim3(
        scale=0.9,
        rotation=so3_exp(np.asarray([-4e-10, 1e-10, 2e-10])),
        translation=np.asarray([-0.4, 0.5, 0.6]),
    )
    analytic_parent, analytic_relative = analytic_sim3_compose_jacobians(
        parent,
        relative,
    )
    numeric_parent, numeric_relative = _LEGACY_COMPOSE_JACOBIANS(parent, relative)
    np.testing.assert_allclose(analytic_parent, numeric_parent, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(
        analytic_relative,
        numeric_relative,
        atol=2e-9,
        rtol=2e-9,
    )


def test_analytic_jacobian_fails_closed_at_so3_log_branch() -> None:
    parent = Sim3(rotation=so3_exp(np.asarray([np.pi, 0.0, 0.0])))
    with pytest.raises(ValueError, match="log branch cut"):
        analytic_sim3_compose_jacobians(parent, Sim3.identity())


def test_dispatcher_preserves_provider_v1_default_and_context_locality() -> None:
    parent = Sim3(
        scale=1.1,
        rotation=so3_exp(np.asarray([0.2, -0.1, 0.3])),
        translation=np.asarray([0.5, -0.2, 0.1]),
    )
    relative = Sim3(
        scale=0.8,
        rotation=so3_exp(np.asarray([-0.3, 0.1, 0.2])),
        translation=np.asarray([-0.4, 0.7, 0.2]),
    )
    legacy = _LEGACY_COMPOSE_JACOBIANS(parent, relative)
    assert current_composition_jacobian_mode() == "legacy_finite_difference"
    dispatched = observation_export._compose_jacobians(parent, relative)
    np.testing.assert_array_equal(dispatched[0], legacy[0])
    np.testing.assert_array_equal(dispatched[1], legacy[1])

    with composition_jacobian_mode("analytic"):
        assert current_composition_jacobian_mode() == "analytic"
        dispatched = observation_export._compose_jacobians(parent, relative)
        analytic = analytic_sim3_compose_jacobians(parent, relative)
        np.testing.assert_array_equal(dispatched[0], analytic[0])
        np.testing.assert_array_equal(dispatched[1], analytic[1])

        isolated = copy_context()
        assert isolated.run(current_composition_jacobian_mode) == "analytic"

    assert current_composition_jacobian_mode() == "legacy_finite_difference"


def test_invalid_mode_and_branch_tolerance_fail_closed() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        with composition_jacobian_mode("unsupported"):  # type: ignore[arg-type]
            pass
    with pytest.raises(ValueError, match="branch_cut_tolerance"):
        analytic_sim3_compose_jacobians(
            Sim3.identity(),
            Sim3.identity(),
            branch_cut_tolerance=-1.0,
        )
