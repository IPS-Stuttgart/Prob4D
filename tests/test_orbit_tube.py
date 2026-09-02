from __future__ import annotations

import math

import pytest

from prob4d.orbit_tube import (
    certify_orbit_tube,
    fit_groupwise_orbit_tube,
    fit_split_conformal_orbit_tube,
    group_max_nonconformity,
    query_diameter_tube_bound,
    robust_advantage_tube_bound,
)


def test_group_max_nonconformity_uses_complete_groups() -> None:
    scores = group_max_nonconformity(
        [
            [0.1, 0.3, 0.2],
            [0.05, 0.04],
            [0.7],
        ]
    )
    assert scores == (0.3, 0.05, 0.7)


def test_group_max_nonconformity_fails_closed_on_bad_groups() -> None:
    with pytest.raises(ValueError, match="at least one calibration group"):
        group_max_nonconformity([])
    with pytest.raises(ValueError, match="must not be empty"):
        group_max_nonconformity([[0.1], []])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        group_max_nonconformity([[0.1, math.nan]])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        group_max_nonconformity([[0.1, -0.2]])


def test_split_conformal_uses_finite_sample_higher_rank() -> None:
    calibration = fit_split_conformal_orbit_tube(
        range(1, 10),
        miscoverage=0.2,
    )
    # n=9 and ceil((n+1)*(1-alpha))=ceil(8)=8.
    assert calibration.quantile_rank == 8
    assert calibration.radius == 8.0
    assert calibration.coverage_lower_bound == pytest.approx(0.8)
    assert calibration.minimum_miscoverage_for_finite_radius == pytest.approx(0.1)
    assert calibration.finite_radius
    assert calibration.covers(8.0)
    assert not calibration.covers(8.1)


def test_calibration_identity_is_order_invariant() -> None:
    forward = fit_split_conformal_orbit_tube(
        [0.3, 0.1, 0.2, 0.4],
        miscoverage=0.25,
    )
    reverse = fit_split_conformal_orbit_tube(
        [0.4, 0.2, 0.1, 0.3],
        miscoverage=0.25,
    )
    assert forward == reverse
    assert len(forward.calibration_id) == 64


def test_impossible_finite_sample_confidence_returns_infinite_radius() -> None:
    calibration = fit_split_conformal_orbit_tube(
        [0.1, 0.2, 0.3, 0.4],
        miscoverage=0.05,
    )
    # Four groups cannot support 95% finite split-conformal coverage:
    # ceil(5*0.95)=5, but only four calibration order statistics exist.
    assert calibration.quantile_rank == 5
    assert calibration.coverage_lower_bound == 1.0
    assert math.isinf(calibration.radius)
    assert not calibration.finite_radius

    decision = certify_orbit_tube(
        calibration,
        exact_orbit_query_diameter=0.0,
        query_lipschitz=0.0,
        query_tolerance=0.0,
        exact_orbit_advantage_lower_bound=1.0,
        advantage_lipschitz=0.0,
    )
    assert not decision.accepted
    assert decision.reasons == ("finite-sample-confidence-not-supportable",)


def test_zero_radius_recovers_exact_orbit_certificate() -> None:
    calibration = fit_split_conformal_orbit_tube(
        [0.0] * 9,
        miscoverage=0.2,
    )
    decision = certify_orbit_tube(
        calibration,
        exact_orbit_query_diameter=0.02,
        query_lipschitz=10.0,
        query_tolerance=0.02,
        exact_orbit_advantage_lower_bound=0.5,
        advantage_lipschitz=20.0,
        required_advantage_margin=0.1,
        omitted_effect_bound=0.05,
        numerical_slack=0.01,
    )
    assert decision.accepted
    assert decision.reasons == ("certified",)
    assert decision.query_diameter_upper_bound == pytest.approx(0.02)
    assert decision.robust_advantage_lower_bound == pytest.approx(0.44)


def test_lipschitz_tube_penalties_are_exactly_reported() -> None:
    calibration = fit_split_conformal_orbit_tube(
        [0.2] * 9,
        miscoverage=0.2,
    )
    decision = certify_orbit_tube(
        calibration,
        exact_orbit_query_diameter=0.1,
        query_lipschitz=2.0,
        query_tolerance=0.9,
        exact_orbit_advantage_lower_bound=1.0,
        advantage_lipschitz=1.5,
        required_advantage_margin=0.5,
        omitted_effect_bound=0.1,
        numerical_slack=0.02,
    )
    assert decision.query_tube_penalty == pytest.approx(0.8)
    assert decision.query_diameter_upper_bound == pytest.approx(0.9)
    assert decision.advantage_tube_penalty == pytest.approx(0.3)
    assert decision.robust_advantage_lower_bound == pytest.approx(0.58)
    assert decision.accepted


def test_query_and_advantage_failures_are_distinguished() -> None:
    calibration = fit_groupwise_orbit_tube(
        [[0.1, 0.2], [0.2, 0.2], [0.15], [0.2]],
        miscoverage=0.25,
    )
    decision = certify_orbit_tube(
        calibration,
        exact_orbit_query_diameter=0.3,
        query_lipschitz=1.0,
        query_tolerance=0.5,
        exact_orbit_advantage_lower_bound=0.2,
        advantage_lipschitz=1.0,
        required_advantage_margin=0.1,
    )
    assert not decision.accepted
    assert decision.reasons == (
        "query-diameter-exceeds-tolerance",
        "robust-advantage-not-above-margin",
    )


def test_standalone_bounds_handle_zero_lipschitz_and_infinite_radius() -> None:
    assert query_diameter_tube_bound(
        exact_orbit_diameter=0.3,
        query_lipschitz=0.0,
        radius=math.inf,
    ) == pytest.approx(0.3)
    assert robust_advantage_tube_bound(
        exact_orbit_advantage_lower_bound=0.4,
        advantage_lipschitz=0.0,
        radius=math.inf,
        omitted_effect_bound=0.1,
        numerical_slack=0.02,
    ) == pytest.approx(0.28)


def test_invalid_calibration_and_certificate_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        fit_split_conformal_orbit_tube([0.1], miscoverage=0.0)
    with pytest.raises(ValueError, match="strictly between"):
        fit_split_conformal_orbit_tube([0.1], miscoverage=1.0)
    with pytest.raises(ValueError, match="at least one"):
        fit_split_conformal_orbit_tube([], miscoverage=0.2)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fit_split_conformal_orbit_tube([math.inf], miscoverage=0.2)

    calibration = fit_split_conformal_orbit_tube(
        [0.1] * 9,
        miscoverage=0.2,
    )
    with pytest.raises(ValueError, match="query_tolerance"):
        certify_orbit_tube(
            calibration,
            exact_orbit_query_diameter=0.0,
            query_lipschitz=1.0,
            query_tolerance=-1.0,
            exact_orbit_advantage_lower_bound=1.0,
            advantage_lipschitz=1.0,
        )
