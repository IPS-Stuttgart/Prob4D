from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from prob4d.axial_query_certificate import (
    AngleArc,
    AxialRotationOrbit,
    HarmonicQuery,
    certify_shared_orbit_advantage,
)
from prob4d.axial_query_study import run_axial_query_study


def orbit() -> AxialRotationOrbit:
    return AxialRotationOrbit(np.zeros(3), np.array([0.0, 0.0, 1.0]), "shared-test-gauge")


def query(c: float, a: float, b: float) -> HarmonicQuery:
    return HarmonicQuery(c, a, b, orbit().key)


def test_stationary_query_is_not_constant_on_finite_orbit() -> None:
    model = orbit()
    support = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    q = model.affine_query([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    assert model.maximum_support_displacement(support) == 0.0
    assert q.derivative_at_zero == 0.0
    assert q.amplitude == 1.0
    assert q.bounds().lower == -1.0
    assert q.bounds().upper == 1.0
    for angle in np.linspace(-math.pi, math.pi, 31):
        np.testing.assert_array_equal(model.transform(support, float(angle)), support)
    assert q.evaluate(0.0) == 1.0
    assert q.evaluate(math.pi) == -1.0


def test_shared_gauge_cancels_before_interval_optimization() -> None:
    fallback = query(4.0, 2.0, 3.0)
    candidate = query(3.75, 2.0, 3.0)
    assert fallback.bounds().lower - candidate.bounds().upper < 0.0
    result = certify_shared_orbit_advantage(
        fallback_loss=fallback, candidate_loss=candidate, scope_admitted=True
    )
    assert result.admitted
    assert result.lower_advantage == 0.25
    assert result.upper_advantage == 0.25


def test_ambiguous_query_can_still_have_a_certified_decision() -> None:
    result = certify_shared_orbit_advantage(
        fallback_loss=query(4.0, 0.0, 0.0),
        candidate_loss=query(2.75, -1.0, 0.0),
        scope_admitted=True,
    )
    assert result.admitted
    assert result.lower_advantage == 0.25


def test_omitted_effect_envelope_can_reverse_admission() -> None:
    kwargs = dict(
        fallback_loss=query(4.0, 0.0, 0.0),
        candidate_loss=query(2.75, -1.0, 0.0),
        scope_admitted=True,
    )
    assert certify_shared_orbit_advantage(**kwargs).admitted
    result = certify_shared_orbit_advantage(**kwargs, advantage_error_bound=0.5)
    assert not result.admitted
    assert result.lower_advantage == -0.25
    assert result.reason_codes == ("nonpositive-robust-advantage",)


def test_unadmitted_scope_and_empty_support_never_accept_vacuously() -> None:
    result = certify_shared_orbit_advantage(
        fallback_loss=query(10.0, 0.0, 0.0),
        candidate_loss=query(0.0, 0.0, 0.0),
        scope_admitted=False,
        arc=None,
    )
    assert not result.admitted
    assert result.lower_advantage is None
    assert result.upper_advantage is None
    assert result.reason_codes == (
        "orbit-model-scope-not-admitted",
        "infeasible-anchor-support",
    )


def test_different_orbits_cannot_be_treated_as_shared_uncertainty() -> None:
    first = query(2.0, 1.0, 0.0)
    other_orbit = AxialRotationOrbit(np.ones(3), np.array([0.0, 1.0, 0.0]), "shared-test-gauge")
    other = HarmonicQuery(1.0, 1.0, 0.0, other_orbit.key)
    with pytest.raises(ValueError, match="same shared orbit"):
        first.minus(other)


def test_arc_crossing_pi_and_interior_extrema_are_handled() -> None:
    q = query(0.0, 1.0, 0.0)
    wrapped = AngleArc(math.pi - 0.05, 0.20)
    bounds = q.bounds(wrapped)
    assert bounds.lower == pytest.approx(-1.0)
    assert bounds.upper == pytest.approx(math.cos(math.pi - 0.25))
    assert wrapped.contains(-math.pi)
    assert wrapped.contains(bounds.lower_angle)
    assert wrapped.contains(bounds.upper_angle, atol=1e-14)
    maximum = q.bounds(AngleArc(0.1, 0.4))
    assert maximum.upper == 1.0


def test_zero_width_and_constant_queries() -> None:
    q = query(0.3, 1.0, 2.0)
    result = q.bounds(AngleArc(0.45, 0.0))
    assert result.lower == result.upper == q.evaluate(0.45)
    constant = query(3.0, 0.0, 0.0).bounds(AngleArc(2.0, 0.1))
    assert constant.lower == constant.upper == 3.0


def test_random_affine_queries_match_direct_transformed_points() -> None:
    rng = np.random.default_rng(11)
    for _ in range(100):
        model = AxialRotationOrbit(rng.normal(size=3), rng.normal(size=3), "shared-test-gauge")
        points = rng.normal(size=(12, 3))
        weights = rng.normal(size=(12, 3))
        offset = float(rng.normal())
        q = model.affine_query(points, weights, offset=offset)
        for angle in rng.uniform(-8.0, 8.0, 8):
            direct = offset + np.sum(weights * model.transform(points, float(angle)))
            assert q.evaluate(float(angle)) == pytest.approx(direct, abs=1e-12)


def test_random_arc_bounds_contain_dense_grid_and_attain_their_extrema() -> None:
    rng = np.random.default_rng(22)
    for _ in range(400):
        q = query(*(float(v) for v in rng.normal(size=3)))
        arc = AngleArc(float(rng.uniform(-5, 5)), float(rng.uniform(0, math.pi)))
        bounds = q.bounds(arc)
        grid = np.linspace(arc.center - arc.half_width, arc.center + arc.half_width, 2049)
        values = q.constant + q.cosine * np.cos(grid) + q.sine * np.sin(grid)
        assert np.min(values) >= bounds.lower - 2e-12
        assert np.max(values) <= bounds.upper + 2e-12
        assert q.evaluate(bounds.lower_angle) == pytest.approx(bounds.lower, abs=2e-12)
        assert q.evaluate(bounds.upper_angle) == pytest.approx(bounds.upper, abs=2e-12)
        assert arc.contains(bounds.lower_angle, atol=2e-12)
        assert arc.contains(bounds.upper_angle, atol=2e-12)


def test_rigid_coordinate_change_preserves_scalar_query_ranges() -> None:
    rng = np.random.default_rng(33)
    for _ in range(50):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix[:, -1] *= np.linalg.det(matrix)
        shift = rng.normal(size=3)
        source = AxialRotationOrbit(rng.normal(size=3), rng.normal(size=3), "shared-test-gauge")
        target = AxialRotationOrbit(
            matrix @ source.origin + shift, matrix @ source.axis, "shared-test-gauge"
        )
        points = rng.normal(size=(8, 3))
        weights = rng.normal(size=(8, 3))
        moved_weights = weights @ matrix.T
        first = source.affine_query(points, weights, offset=0.7)
        second = target.affine_query(
            points @ matrix.T + shift,
            moved_weights,
            offset=0.7 - float(np.sum(moved_weights @ shift)),
        )
        np.testing.assert_allclose(
            [first.constant, first.cosine, first.sine],
            [second.constant, second.cosine, second.sine],
            atol=2e-12,
        )


def test_near_line_support_is_not_silently_promoted_to_exact_symmetry() -> None:
    points = np.array([[0.003, 0.0, 0.2], [-0.001, 0.0, -0.2]])
    model = orbit()
    assert model.maximum_support_displacement(points) == pytest.approx(0.006)
    motion = np.linalg.norm(model.transform(points, math.pi) - points, axis=1)
    assert np.max(motion) == pytest.approx(model.maximum_support_displacement(points))


def test_off_axis_bounded_anchor_has_exact_feasible_arc() -> None:
    model = orbit()
    arc = model.bounded_anchor_arc([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], error_radius=1.0)
    assert arc is not None
    assert arc.center == 0.0
    assert arc.half_width == pytest.approx(math.pi / 3)
    bounds = query(0.25, 1.0, 0.0).bounds(arc)
    assert bounds.lower == pytest.approx(0.75)
    assert bounds.upper == pytest.approx(1.25)


def test_on_axis_anchor_is_either_uninformative_or_inconsistent() -> None:
    model = orbit()
    arc = model.bounded_anchor_arc([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], error_radius=0.0)
    assert arc == AngleArc()
    assert model.bounded_anchor_arc(
        [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], error_radius=0.5
    ) is None


def test_anchor_full_circle_singleton_and_infeasible_boundaries() -> None:
    model = orbit()
    assert model.bounded_anchor_arc(
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], error_radius=2.0
    ) == AngleArc()
    singleton = model.bounded_anchor_arc(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], error_radius=0.0
    )
    assert singleton is not None
    assert singleton.half_width == 0.0
    assert singleton.center == pytest.approx(math.pi / 2)
    assert model.bounded_anchor_arc(
        [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], error_radius=0.5
    ) is None


def test_anchor_arc_matches_direct_residual_and_contains_bounded_sensor_truth() -> None:
    rng = np.random.default_rng(44)
    for _ in range(200):
        model = AxialRotationOrbit(rng.normal(size=3), rng.normal(size=3), "shared-test-gauge")
        point = rng.normal(size=3)
        theta = float(rng.uniform(-math.pi, math.pi))
        radius = float(rng.uniform(0.001, 0.3))
        noise = rng.normal(size=3)
        noise *= radius * float(rng.uniform(0.0, 0.9)) / np.linalg.norm(noise)
        observed = model.transform(point[None, :], theta)[0] + noise
        arc = model.bounded_anchor_arc(point, observed, error_radius=radius)
        assert arc is not None and arc.contains(theta, atol=5e-11)
        for candidate in rng.uniform(-math.pi, math.pi, 30):
            residual = np.linalg.norm(
                model.transform(point[None, :], float(candidate))[0] - observed
            )
            if abs(residual - radius) > 1e-10:
                assert arc.contains(float(candidate)) == (residual <= radius)


def test_axes_are_normalized_and_arrays_irreversibly_readonly() -> None:
    original = np.array([0.0, 0.0, 8.0])
    model = AxialRotationOrbit(np.zeros(3), original, "shared-test-gauge")
    original[:] = 1.0
    np.testing.assert_array_equal(model.axis, [0.0, 0.0, 1.0])
    for array in (model.origin, model.axis):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coefficients_and_angles_fail_closed(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        query(value, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        AngleArc(value, 0.1)
    with pytest.raises(ValueError, match="finite"):
        query(0.0, 1.0, 1.0).evaluate(value)


@pytest.mark.parametrize("field", ["advantage_error_bound", "required_margin", "numerical_slack"])
def test_negative_gate_controls_fail_closed(field: str) -> None:
    kwargs = {
        "fallback_loss": query(1.0, 0.0, 0.0),
        "candidate_loss": query(0.0, 0.0, 0.0),
        "scope_admitted": True,
        field: -0.1,
    }
    with pytest.raises(ValueError, match="nonnegative"):
        certify_shared_orbit_advantage(**kwargs)


def test_invalid_geometry_scope_arc_and_orbit_key_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive norm"):
        AxialRotationOrbit(np.zeros(3), np.zeros(3), "shared-test-gauge")
    with pytest.raises(ValueError, match="three-vector"):
        AxialRotationOrbit(np.ones(2), np.ones(3), "shared-test-gauge")
    with pytest.raises(ValueError, match="same shape"):
        orbit().affine_query(np.ones((2, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        orbit().affine_query([], [])
    with pytest.raises(TypeError, match="scope_admitted"):
        certify_shared_orbit_advantage(
            fallback_loss=query(1.0, 0.0, 0.0),
            candidate_loss=query(0.0, 0.0, 0.0),
            scope_admitted=1,
        )
    with pytest.raises(ValueError, match="exceed pi"):
        AngleArc(0.0, 4.0)
    with pytest.raises(TypeError, match="real scalar"):
        AngleArc(True, 0.5)
    with pytest.raises(ValueError, match="unit axis"):
        replace(query(0.0, 0.0, 0.0), orbit_key=("shared-test-gauge", (0.0,) * 6))


def test_exact_ties_and_numerically_indistinguishable_improvements_reject() -> None:
    for gain in (0.0, 1e-13, 1e-12):
        result = certify_shared_orbit_advantage(
            fallback_loss=query(gain, 0.0, 0.0),
            candidate_loss=query(0.0, 0.0, 0.0),
            scope_admitted=True,
        )
        assert not result.admitted


def test_controlled_study_is_reproducible_and_preserves_claim_boundaries() -> None:
    first = run_axial_query_study(seed=7, cases_per_family=16)
    assert first == run_axial_query_study(seed=7, cases_per_family=16)
    assert first["total_cases"] == 64
    assert first["evidence_class"] == "constructed-controlled-mechanism-not-real-provider"
    assert first["anchor_truth_exclusion_count"] == 0
    totals = first["totals"]
    assert totals["local-query-gate-then-plugin"]["accepted"] == 64
    assert totals["local-query-gate-then-plugin"]["admitted_with_possible_harm"] == 32
    assert totals["independent-query-intervals"]["accepted"] == 16
    assert totals["shared-orbit-certificate"]["accepted"] == 32
    for arm in ("shared-orbit-certificate", "shared-orbit-plus-one-bounded-anchor"):
        assert totals[arm]["sampled_harmful_accepts"] == 0
        assert totals[arm]["admitted_with_possible_harm"] == 0
        assert totals[arm]["exact_fallback_identity_failures"] == 0


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_study_size_rejected(size: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        run_axial_query_study(cases_per_family=size)


def test_matching_axes_do_not_authorize_cancellation_of_independent_gauges() -> None:
    first = orbit()
    independent = AxialRotationOrbit(first.origin, first.axis, "independent-test-gauge")
    fallback = first.affine_query([[1, 0, 0]], [[1, 0, 0]], offset=4.0)
    candidate = independent.affine_query([[1, 0, 0]], [[1, 0, 0]], offset=3.75)
    with pytest.raises(ValueError, match="same shared orbit"):
        certify_shared_orbit_advantage(
            fallback_loss=fallback, candidate_loss=candidate, scope_admitted=True
        )


@pytest.mark.parametrize("identity", ["", "   ", None])
def test_gauge_identity_is_required(identity: object) -> None:
    with pytest.raises(ValueError, match="shared_gauge_id"):
        AxialRotationOrbit(np.zeros(3), np.array([0.0, 0.0, 1.0]), identity)


def test_certificate_record_rejects_contradictory_acceptance() -> None:
    accepted = certify_shared_orbit_advantage(
        fallback_loss=query(1.0, 0.0, 0.0),
        candidate_loss=query(0.0, 0.0, 0.0),
        scope_admitted=True,
    )
    with pytest.raises(ValueError, match="no positive robust advantage"):
        replace(accepted, lower_advantage=-1.0)
    with pytest.raises(ValueError, match="reasons disagree"):
        replace(accepted, admitted=False)
    with pytest.raises(ValueError, match="scope changed"):
        replace(accepted, scope="deployment-safety")
