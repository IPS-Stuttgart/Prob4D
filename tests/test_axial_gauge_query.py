"""Analytic and adversarial controls, with no provider or data access."""

from __future__ import annotations

import numpy as np
import pytest

from prob4d.axial_gauge_query import AxialGaugeOrbit, AxialQueryFamily, affine_axial_queries
from prob4d.axial_gauge_query_study import run_axial_gauge_query_study


def _orbit() -> AxialGaugeOrbit:
    return AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def _family(points: np.ndarray, weights: np.ndarray) -> AxialQueryFamily:
    return affine_axial_queries(
        points, weights, point_group_ids=("g",) * len(points), orbits={"g": _orbit()}
    )


def _quaternion_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Independent finite-rotation reference, not the production decomposition."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = np.sin(angle / 2.0) * axis
    w = np.cos(angle / 2.0)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_zero_first_derivative_does_not_mean_finite_invariance() -> None:
    family = _family(np.array([[0.0, 0.1, 0.0]]), np.array([[[0.0, 1.0, 0.0]]]))
    assert family.sine[0, 0] == 0.0
    assert family.cosine[0, 0] == 0.1
    np.testing.assert_allclose(family.bounds(), [[-0.1], [0.1]])
    np.testing.assert_allclose(family.evaluate(np.array([np.pi])), [-0.1])
    support = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _orbit().maximum_support_motion(support) == 0.0
    np.testing.assert_array_equal(_orbit().transform(support, np.pi), support)


@pytest.mark.parametrize("point", [[0.04, 0.1, 0.0], [0.04, 0.0, 0.0]])
def test_axial_query_is_globally_invariant(point: list[float]) -> None:
    family = _family(np.array([point]), np.array([[[1.0, 0.0, 0.0]]]))
    np.testing.assert_allclose(family.bounds(), [[0.04], [0.04]])


def test_shared_gauge_cancellation_and_invalid_merging_control() -> None:
    points = np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]])
    weights = np.array([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
    shared = _family(points, weights)
    separate = affine_axial_queries(
        points,
        weights,
        point_group_ids=("a", "b"),
        orbits={"a": _orbit(), "b": _orbit()},
    )
    np.testing.assert_array_equal(shared.bounds(), [[0.0], [0.0]])
    np.testing.assert_allclose(separate.bounds(), [[-0.2], [0.2]])
    # Coincident axes do not justify merging two separately variable gauges.
    np.testing.assert_allclose(separate.evaluate(np.array([0.0, np.pi])), [0.2])


def test_action_regret_has_an_attaining_witness() -> None:
    family = affine_axial_queries(
        np.array([[0.0, 0.1, 0.0]]),
        np.array([[[0.0, -1.0, 0.0]], [[0.0, 1.0, 0.0]]]),
        offsets=np.array([0.1, 0.1]),
        point_group_ids=("g",),
        orbits={"g": _orbit()},
    )
    np.testing.assert_allclose(family.evaluate(np.zeros(1)), [0.0, 0.2])
    np.testing.assert_allclose(family.worst_case_regrets(), [0.2, 0.2])
    for action in range(2):
        competitor, angles = family.regret_witness(action)
        losses = family.evaluate(angles)
        assert losses[action] - np.min(losses) == pytest.approx(0.2)
        assert losses[action] - losses[competitor] == pytest.approx(0.2)
    assert not family.within_regret_budget(0, maximum_regret=0.05)
    assert family.within_regret_budget(0, maximum_regret=0.2)
    assert not family.within_regret_budget(0, maximum_regret=0.2, numerical_margin=1e-9)


def test_shared_action_nuisance_cancels_before_bounding() -> None:
    family = AxialQueryFamily(
        ("g",), np.array([0.05, 0.10]), np.array([[0.1], [0.1]]), np.zeros((2, 1))
    )
    lower, upper = family.bounds()
    assert upper[0] > lower[1]  # Marginal intervals overlap.
    np.testing.assert_allclose(family.contrast_bounds()[1][0, 1], -0.05)
    np.testing.assert_allclose(family.worst_case_regrets(), [0.0, 0.05])
    assert family.within_regret_budget(0, maximum_regret=0.0)


def test_multigroup_affine_bounds_and_regrets_against_finite_rotations() -> None:
    rng = np.random.default_rng(8302026)
    for _ in range(20):
        points = rng.normal(size=(6, 3))
        weights = rng.normal(size=(4, 6, 3))
        offsets = rng.normal(size=4)
        labels = ("a", "b", "a", "b", "a", "b")
        orbits = {
            "a": AxialGaugeOrbit(rng.normal(size=3), rng.normal(size=3)),
            "b": AxialGaugeOrbit(rng.normal(size=3), rng.normal(size=3)),
        }
        family = affine_axial_queries(
            points, weights, offsets=offsets, point_group_ids=labels, orbits=orbits
        )
        angles = rng.uniform(-np.pi, np.pi, size=2)
        transformed = np.empty_like(points)
        for index, group in enumerate(labels):
            orbit = orbits[group]
            rotation = _quaternion_rotation(orbit.axis, angles[family.group_ids.index(group)])
            transformed[index] = orbit.pivot + rotation @ (points[index] - orbit.pivot)
        reference = offsets + np.einsum("qnc,nc->q", weights, transformed)
        np.testing.assert_allclose(family.evaluate(angles), reference, atol=1e-12)
        lower, upper = family.bounds()
        assert np.all(reference >= lower - 1e-12)
        assert np.all(reference <= upper + 1e-12)
        for query in range(4):
            maximum_angles = np.arctan2(family.sine[query], family.cosine[query])
            assert family.evaluate(maximum_angles)[query] == pytest.approx(upper[query])
            assert family.evaluate(maximum_angles + np.pi)[query] == pytest.approx(lower[query])
            competitor, witness = family.regret_witness(query)
            losses = family.evaluate(witness)
            regret = family.worst_case_regrets()[query]
            assert losses[query] - losses[competitor] == pytest.approx(regret, abs=1e-12)
            assert losses[query] - min(losses) == pytest.approx(regret, abs=1e-12)


def test_rigid_frame_and_axis_sign_invariance() -> None:
    points = np.array([[0.5, 0.2, -0.1], [-0.4, 0.8, 0.3]])
    weights = np.array([[[0.1, 1.0, -0.3], [0.2, -0.2, 0.7]]])
    original = _family(points, weights)
    rotation = _quaternion_rotation(np.array([1.0, 2.0, -1.0]), 1.2)
    shift = np.array([2.0, -0.4, 0.7])
    moved_points = points @ rotation.T + shift
    moved_weights = weights @ rotation.T
    offsets = -np.einsum("qnc,c->q", moved_weights, shift)
    moved = affine_axial_queries(
        moved_points,
        moved_weights,
        offsets=offsets,
        point_group_ids=("g", "g"),
        orbits={"g": AxialGaugeOrbit(shift, rotation @ np.array([1.0, 0.0, 0.0]))},
    )
    np.testing.assert_allclose(original.bounds(), moved.bounds(), atol=1e-12)
    flipped = affine_axial_queries(
        points,
        weights,
        point_group_ids=("g", "g"),
        orbits={"g": AxialGaugeOrbit(np.zeros(3), np.array([-1.0, 0.0, 0.0]))},
    )
    np.testing.assert_allclose(original.bounds(), flipped.bounds())
    np.testing.assert_allclose(
        original.evaluate(np.array([0.7])), flipped.evaluate(np.array([-0.7]))
    )


def test_near_line_is_not_silently_called_an_exact_symmetry() -> None:
    support = np.array([[-1.0, 0.0, 0.0], [1.0, 0.002, 0.0]])
    assert _orbit().maximum_support_motion(support) == pytest.approx(0.004)
    assert _orbit().maximum_support_motion(support) > 0.001


def test_constant_family_without_free_angles() -> None:
    family = AxialQueryFamily((), np.array([1.0, 2.0]), np.empty((2, 0)), np.empty((2, 0)))
    np.testing.assert_allclose(family.bounds(), [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(family.worst_case_regrets(), [0.0, 1.0])
    assert family.regret_witness(1)[1].size == 0


def test_readonly_copies_and_determinism() -> None:
    axis = np.array([10.0, 0.0, 0.0])
    orbit = AxialGaugeOrbit(np.zeros(3), axis)
    axis[0] = 0.0
    np.testing.assert_array_equal(orbit.axis, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        orbit.axis[0] = 2.0
    result = run_axial_gauge_query_study()
    assert result == run_axial_gauge_query_study()
    assert result["stationary_derivative_counterexample"]["finite_orbit_interval_m"] == [-0.1, 0.1]
    assert result["action_control"]["within_budget"] is False
    assert result["shared_action_nuisance_control"]["action_zero_within_zero_budget"] is True
    assert all(value == 0 for value in result["information_boundary"].values())


@pytest.mark.parametrize("axis", [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 2.0]])
def test_invalid_axis_rejected(axis: list[float]) -> None:
    with pytest.raises(ValueError):
        AxialGaugeOrbit(np.zeros(3), np.array(axis))


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_regret_budget_rejected(value: float) -> None:
    family = AxialQueryFamily((), np.ones(1), np.empty((1, 0)), np.empty((1, 0)))
    with pytest.raises(ValueError):
        family.within_regret_budget(0, maximum_regret=value)
    with pytest.raises(ValueError):
        family.within_regret_budget(0, maximum_regret=0.0, numerical_margin=value)


@pytest.mark.parametrize("action", [True, 0.5, -1, 2])
def test_invalid_action_rejected(action: object) -> None:
    family = AxialQueryFamily((), np.ones(1), np.empty((1, 0)), np.empty((1, 0)))
    with pytest.raises((ValueError, TypeError)):
        family.regret_witness(action)


@pytest.mark.parametrize("labels", [("missing",), ("g", "g"), ("",), (" g",), "g"])
def test_invalid_group_lineage_rejected(labels: object) -> None:
    with pytest.raises(ValueError):
        affine_axial_queries(
            np.zeros((1, 3)), np.ones((1, 1, 3)), point_group_ids=labels, orbits={"g": _orbit()}
        )


def test_bad_shapes_and_nonfinite_values_rejected() -> None:
    with pytest.raises(ValueError):
        _family(np.zeros((0, 3)), np.ones((1, 0, 3)))
    with pytest.raises(ValueError):
        _family(np.zeros((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError):
        _family(np.full((1, 3), np.inf), np.ones((1, 1, 3)))
    with pytest.raises(ValueError):
        AxialQueryFamily("g", np.ones(1), np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError):
        AxialQueryFamily(("g", "g"), np.ones(1), np.ones((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        AxialQueryFamily(("g",), np.ones(1), np.ones((2, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError):
        affine_axial_queries(
            np.zeros((1, 3)),
            np.ones((1, 1, 3)),
            offsets=np.ones(2),
            point_group_ids=("g",),
            orbits={"g": _orbit()},
        )
    with pytest.raises(ValueError):
        _orbit().transform(np.zeros((1, 3)), np.nan)


def test_existing_local_gate_parity() -> None:
    # Imports must succeed in the complete repository: no silent integration skip.
    from prob4d.query_observability import (
        QueryObservabilityGate,
        evaluate_query_observability,
        point_position_query_jacobian,
    )
    from prob4d.query_observability_study import _controlled_factor

    factor = _controlled_factor(complete_nullspace=False)
    jacobian = point_position_query_jacobian(factor, np.array([0.0, 0.1, 0.0]))[1:2]
    report = evaluate_query_observability(
        factor, prior_covariance_local=np.eye(7), query_jacobian_local=jacobian
    )
    gate = QueryObservabilityGate(0.8, 0.8, 0.5)
    reference = run_axial_gauge_query_study()["stationary_derivative_counterexample"]
    assert report.direct_observability_fraction == pytest.approx(1.0)
    assert report.nullspace_sensitivity_fraction == pytest.approx(0.0)
    assert report.metric_variance_reduction_fraction == pytest.approx(10.0 / 11.0)
    assert report.worst_supported_variance_ratio == pytest.approx(1.0 / 11.0)
    expected = reference["local_linear_reference"]["local_gate_admits"]
    assert gate.evaluate(report).admitted is expected
