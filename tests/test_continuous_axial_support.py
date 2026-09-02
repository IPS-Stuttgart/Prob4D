from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.axial_query_certificate import AxialRotationOrbit
from prob4d.continuous_axial_support import (
    axial_tube_residual,
    calibrate_group_conformal_upper_bound,
    certify_full_circle_vector_query,
    empirical_upper_quantile,
    point_position_query,
    squared_distance_query,
    support_from_conformal_threshold,
)


def orbit() -> AxialRotationOrbit:
    return AxialRotationOrbit(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        "continuous-test-gauge",
    )


def test_group_conformal_order_statistic_and_infeasible_small_sample() -> None:
    values = np.array([0.4, 0.1, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
    finite = calibrate_group_conformal_upper_bound(values, miscoverage=0.1)
    assert finite.finite
    assert finite.order_statistic == 9
    assert finite.threshold == 0.9
    assert finite.coverage_level == 0.9

    unavailable = calibrate_group_conformal_upper_bound(
        values[:4],
        miscoverage=0.1,
    )
    assert not unavailable.finite
    assert unavailable.threshold is None
    assert unavailable.order_statistic == 5


def test_empirical_upper_quantile_uses_conservative_order_statistic() -> None:
    values = np.arange(1.0, 11.0)
    assert empirical_upper_quantile(values, probability=0.8) == 8.0
    assert empirical_upper_quantile(values, probability=1.0) == 10.0


def test_continuous_residual_recovers_angle_and_off_orbit_error() -> None:
    model = orbit()
    representative = np.array([2.0, 0.0, 0.5])
    theta = 0.7
    noise = np.array([0.0, 0.0, 0.2])
    observed = model.transform(representative[None, :], theta)[0] + noise
    result = axial_tube_residual(
        model,
        representative,
        observed,
        angle_normalizer=math.pi / 3.0,
        radial_scale=2.0,
    )
    assert result.angle_radians == pytest.approx(theta)
    assert result.euclidean_residual == pytest.approx(0.2)
    assert result.normalized_score == pytest.approx(
        max(theta / (math.pi / 3.0), 0.1)
    )


def test_calibrated_tube_contains_every_point_below_its_score_threshold() -> None:
    rng = np.random.default_rng(9)
    model = orbit()
    representative = np.array([1.4, 0.0, -0.2])
    threshold = 0.35
    support = support_from_conformal_threshold(
        model,
        normalized_score_threshold=threshold,
        radial_scale=1.4,
        angle_normalizer=math.pi / 3.0,
    )
    for _ in range(100):
        angle = float(rng.uniform(-support.arc.half_width, support.arc.half_width))
        noise = np.array(
            [
                0.0,
                0.0,
                support.euclidean_radius * float(rng.uniform(-0.99, 0.99)),
            ]
        )
        point = model.transform(representative[None, :], angle)[0] + noise
        assert support.contains(model, representative, point, atol=1e-11)


def test_scalar_query_bounds_expand_by_exact_euclidean_lipschitz_radius() -> None:
    model = orbit()
    representative = np.array([1.0, 0.0, 0.0])
    support = support_from_conformal_threshold(
        model,
        normalized_score_threshold=0.2,
        radial_scale=1.0,
        angle_normalizer=math.pi / 3.0,
    )
    query = point_position_query(model, representative).scalar_projection(
        np.array([1.0, 0.0, 0.0])
    )
    bounds = support.expand_scalar_bounds(query, euclidean_lipschitz=1.0)
    assert bounds.upper == pytest.approx(1.2)
    assert bounds.lower == pytest.approx(
        math.cos(math.pi / 15.0) - 0.2
    )


def test_vector_full_circle_diameter_is_exact_singular_value_formula() -> None:
    model = orbit()
    query = point_position_query(model, [2.0, 0.0, 1.0])
    assert query.full_circle_weighted_diameter() == pytest.approx(4.0)
    weight = np.array([[1.0, 0.0, 0.0]])
    assert query.full_circle_weighted_diameter(weight=weight) == pytest.approx(4.0)
    assert query.full_circle_weighted_diameter(
        weight=weight,
        additive_query_radius=0.3,
    ) == pytest.approx(4.6)

    dense = np.stack(
        [query.evaluate(float(angle)) for angle in np.linspace(-math.pi, math.pi, 721)]
    )
    differences = dense[:, None, :] - dense[None, :, :]
    assert np.max(np.linalg.norm(differences, axis=-1)) == pytest.approx(4.0)


def test_vector_query_certificate_fails_closed_on_scope_and_diameter() -> None:
    query = point_position_query(orbit(), [1.0, 0.0, 0.0])
    accepted = certify_full_circle_vector_query(
        query,
        tolerance=2.0,
        scope_admitted=True,
    )
    assert accepted.admitted
    rejected = certify_full_circle_vector_query(
        query,
        tolerance=1.9,
        scope_admitted=False,
    )
    assert not rejected.admitted
    assert rejected.reason_codes == (
        "orbit-model-scope-not-admitted",
        "query-diameter-exceeds-tolerance",
    )


def test_squared_distance_query_matches_direct_continuous_geometry() -> None:
    rng = np.random.default_rng(12)
    model = AxialRotationOrbit(
        rng.normal(size=3),
        rng.normal(size=3),
        "continuous-test-gauge",
    )
    representative = rng.normal(size=3)
    target = rng.normal(size=3)
    query = squared_distance_query(model, representative, target)
    for angle in rng.uniform(-math.pi, math.pi, 100):
        transformed = model.transform(representative[None, :], float(angle))[0]
        expected = float(np.sum((transformed - target) ** 2))
        assert query.evaluate(float(angle)) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.1, float("nan")],
        [-0.1, 0.2],
        [[0.1, 0.2]],
    ],
)
def test_invalid_calibration_scores_fail_closed(scores: object) -> None:
    with pytest.raises(ValueError):
        calibrate_group_conformal_upper_bound(scores, miscoverage=0.1)


def test_orbit_identity_cannot_be_swapped_after_calibration() -> None:
    first = orbit()
    second = AxialRotationOrbit(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        "other-gauge",
    )
    support = support_from_conformal_threshold(
        first,
        normalized_score_threshold=0.2,
        radial_scale=1.0,
        angle_normalizer=math.pi / 3.0,
    )
    with pytest.raises(ValueError, match="identities"):
        support.contains(second, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
