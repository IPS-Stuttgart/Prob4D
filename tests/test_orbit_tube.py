from __future__ import annotations

import math

import numpy as np
import pytest

from prob4d.orbit_tube import (
    AxialCircleOrbit,
    ScalarBounds,
    calibrate_group_maximum_radius,
    minimal_rotation_transport,
)


def test_point_distance_is_exact_for_axial_and_radial_departures() -> None:
    orbit = AxialCircleOrbit(
        center=np.array([1.0, 2.0, 3.0]),
        axis=np.array([0.0, 0.0, 2.0]),
        radius=4.0,
    )
    assert orbit.point_distance([5.0, 2.0, 3.0]) == pytest.approx(0.0)
    assert orbit.point_distance([6.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert orbit.point_distance([5.0, 2.0, 5.0]) == pytest.approx(2.0)
    assert orbit.point_distance([6.0, 2.0, 5.0]) == pytest.approx(math.sqrt(5.0))


def test_affine_query_bounds_match_dense_circle_and_tube_formula() -> None:
    orbit = AxialCircleOrbit(
        center=np.array([1.0, -2.0, 0.5]),
        axis=np.array([1.0, 1.0, 1.0]),
        radius=2.5,
    )
    direction = np.array([0.3, -0.5, 0.8])
    bounds = orbit.affine_query_bounds(direction, offset=1.2, tube_radius=0.4)

    axis = orbit.axis
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, basis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0.0, 2.0 * math.pi, 100_001)
    circle = (
        orbit.center[None]
        + orbit.radius * np.cos(angles)[:, None] * first[None]
        + orbit.radius * np.sin(angles)[:, None] * second[None]
    )
    values = 1.2 + circle @ direction
    expansion = 0.4 * np.linalg.norm(direction)
    assert bounds.lower == pytest.approx(float(values.min()) - expansion, abs=1e-9)
    assert bounds.upper == pytest.approx(float(values.max()) + expansion, abs=1e-9)


def test_group_maximum_calibration_uses_complete_groups() -> None:
    group_ids = [f"group-{index:02d}" for index in range(18) for _ in range(2)]
    scores = np.asarray(
        [value for index in range(18) for value in (float(index), index + 0.25)],
        dtype=np.float64,
    )
    calibration = calibrate_group_maximum_radius(
        scores,
        group_ids,
        miscoverage=0.10,
    )
    assert calibration.group_count == 18
    assert calibration.order_statistic_rank == 18
    assert calibration.finite_sample_coverage_level == pytest.approx(18.0 / 19.0)
    assert calibration.radius == pytest.approx(17.25)
    assert calibration.covers(17.25)
    assert not calibration.covers(17.26)


def test_group_calibration_requires_enough_independent_groups() -> None:
    with pytest.raises(ValueError, match="too few independent groups"):
        calibrate_group_maximum_radius(
            np.arange(8, dtype=np.float64),
            [f"group-{index}" for index in range(8)],
            miscoverage=0.10,
        )


def test_minimal_rotation_transport_maps_axes_and_handles_antipodes() -> None:
    mapped = minimal_rotation_transport([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    assert np.linalg.norm(mapped) == pytest.approx(1.0)
    assert mapped @ np.array([0.0, 1.0, 0.0]) == pytest.approx(0.0, abs=1e-12)

    source = np.array([0.2, -0.3, 0.9])
    source /= np.linalg.norm(source)
    target = -source
    radial = np.cross(source, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(radial) < 1e-8:
        radial = np.cross(source, np.array([0.0, 1.0, 0.0]))
    radial /= np.linalg.norm(radial)
    transported = minimal_rotation_transport(radial, source, target)
    assert np.linalg.norm(transported) == pytest.approx(1.0)
    assert transported @ target == pytest.approx(0.0, abs=1e-12)


def test_scalar_bounds_threshold_decision_is_fail_closed() -> None:
    assert ScalarBounds(1.0, 2.0).threshold_sign() == 1
    assert ScalarBounds(-2.0, -1.0).threshold_sign() == -1
    assert ScalarBounds(-1.0, 2.0).threshold_sign() is None
    assert ScalarBounds(0.1, 0.2).threshold_sign(margin=0.15) is None


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: AxialCircleOrbit([0, 0, 0], [0, 0, 0], 1.0), "positive norm"),
        (lambda: AxialCircleOrbit([0, 0, 0], [0, 0, 1], -1.0), "nonnegative"),
        (lambda: ScalarBounds(2.0, 1.0), "lower must not exceed"),
    ],
)
def test_invalid_inputs_fail_closed(factory: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]
