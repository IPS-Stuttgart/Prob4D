from __future__ import annotations

import numpy as np
import pytest

from prob4d.orbit_identifiability import (
    QueryOrbitCertificate,
    paired_orbit_error_bound,
    query_set_diameter,
)


def _hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    distance = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(max(np.max(np.min(distance, axis=1)), np.max(np.min(distance, axis=0))))


def test_certificate_bounds_random_linear_query_orbits() -> None:
    rng = np.random.default_rng(260902)
    for _ in range(50):
        estimated = rng.normal(size=(17, 4))
        perturbation = rng.normal(size=estimated.shape)
        perturbation /= np.maximum(np.linalg.norm(perturbation, axis=1, keepdims=True), 1e-15)
        radii = rng.uniform(0.0, 0.2, size=(estimated.shape[0], 1))
        true = estimated + radii * perturbation
        query_matrix = rng.normal(size=(3, 4))
        lipschitz = float(np.linalg.norm(query_matrix, ord=2))
        estimated_query = estimated @ query_matrix.T
        true_query = true @ query_matrix.T
        delta = _hausdorff(true, estimated)
        certificate = QueryOrbitCertificate(
            query_set_diameter(estimated_query),
            lipschitz,
            delta,
        )
        assert query_set_diameter(true_query) <= (
            certificate.true_query_diameter_upper_bound + 1e-12
        )


def test_factor_two_is_tight_for_translated_two_point_sets() -> None:
    estimated = np.array([[0.0], [0.0]])
    true = np.array([[-0.3], [0.3]])
    certificate = QueryOrbitCertificate(
        estimated_query_diameter=query_set_diameter(estimated),
        query_lipschitz_constant=1.0,
        orbit_hausdorff_radius=0.3,
    )
    assert certificate.true_query_diameter_upper_bound == pytest.approx(0.6)
    assert query_set_diameter(true) == pytest.approx(0.6)


def test_paired_parameterization_upper_bounds_sampled_hausdorff_distance() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 129, endpoint=False)
    reference = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    estimate = np.column_stack(
        (1.03 * np.cos(angles), 0.97 * np.sin(angles), 0.02 * np.ones_like(angles))
    )
    paired = paired_orbit_error_bound(reference, estimate)
    assert paired + 1e-15 >= _hausdorff(reference, estimate)


def test_admission_is_monotone_in_orbit_error_and_lipschitz_constant() -> None:
    base = QueryOrbitCertificate(0.1, 2.0, 0.05)
    assert base.true_query_diameter_upper_bound == pytest.approx(0.3)
    assert base.admitted(0.3)
    assert not QueryOrbitCertificate(0.1, 2.0, 0.051).admitted(0.3)
    assert not QueryOrbitCertificate(0.1, 2.01, 0.05).admitted(0.3)


@pytest.mark.parametrize(
    "arguments",
    [
        (-1.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, -1.0),
        (np.nan, 1.0, 0.0),
        (0.0, np.inf, 0.0),
    ],
)
def test_invalid_certificate_inputs_are_rejected(arguments) -> None:
    with pytest.raises(ValueError):
        QueryOrbitCertificate(*arguments)


@pytest.mark.parametrize(
    "values",
    [[], [[np.nan]], np.ones((1, 1, 1))],
)
def test_invalid_query_sets_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        query_set_diameter(values)


def test_invalid_paired_orbits_are_rejected() -> None:
    with pytest.raises(ValueError):
        paired_orbit_error_bound(np.zeros((3, 2)), np.zeros((2, 2)))
