from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from prob4d.robust_orbit_query import (
    batch_axial_orbit_diameters,
    certify_axial_linear_query,
    certify_uniformly_sampled_periodic_query,
    sampled_pairwise_diameter,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "robust-orbit-query-stress-v1.json"
STUDY = ROOT / "scripts" / "science" / "run_robust_orbit_query_stress.py"


def _load_study():
    name = "robust_orbit_query_stress_test"
    spec = importlib.util.spec_from_file_location(name, STUDY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_local_stationarity_does_not_imply_finite_orbit_invariance() -> None:
    certificate = certify_axial_linear_query(
        [[1.0, 0.0]],
        coefficient_error_bound=0.0,
        invariance_tolerance=0.5,
        nominal_angle_radians=0.0,
    )
    assert certificate.nominal_local_derivative_norm == 0.0
    assert certificate.estimated_diameter == 2.0
    assert certificate.lower_diameter == 2.0
    assert certificate.upper_diameter == 2.0
    assert certificate.decision == "certified-variant"
    assert not certificate.update_admitted


def test_bounded_error_certificate_contains_every_random_true_diameter() -> None:
    rng = np.random.default_rng(8675309)
    for dimension in (1, 2, 3, 7):
        for _ in range(100):
            truth = rng.normal(size=(dimension, 2))
            raw_error = rng.normal(size=(dimension, 2))
            raw_norm = float(batch_axial_orbit_diameters(raw_error) / 2.0)
            error_bound = float(rng.uniform(0.0, 0.5))
            error = raw_error * (error_bound / raw_norm)
            estimate = truth + error
            certificate = certify_axial_linear_query(
                estimate,
                coefficient_error_bound=error_bound,
                invariance_tolerance=1.0,
                nominal_angle_radians=float(rng.uniform(-np.pi, np.pi)),
            )
            true_diameter = float(batch_axial_orbit_diameters(truth))
            assert certificate.lower_diameter <= true_diameter + 1e-13
            assert true_diameter <= certificate.upper_diameter + 1e-13
            if certificate.update_admitted:
                assert true_diameter <= certificate.invariance_tolerance + 1e-13


def test_three_way_decision_is_fail_closed() -> None:
    invariant = certify_axial_linear_query(
        np.zeros((3, 2)),
        coefficient_error_bound=0.1,
        invariance_tolerance=0.21,
    )
    assert invariant.decision == "certified-invariant"
    assert invariant.update_admitted

    uncertain = certify_axial_linear_query(
        np.zeros((3, 2)),
        coefficient_error_bound=0.1,
        invariance_tolerance=0.19,
    )
    assert uncertain.decision == "undetermined"
    assert not uncertain.update_admitted

    variant = certify_axial_linear_query(
        [[0.5, 0.0]],
        coefficient_error_bound=0.1,
        invariance_tolerance=0.7,
    )
    assert variant.lower_diameter == pytest.approx(0.8)
    assert variant.decision == "certified-variant"
    assert not variant.update_admitted


def test_vectorized_axial_diameter_matches_dense_singular_values() -> None:
    rng = np.random.default_rng(314159)
    coefficients = rng.normal(size=(257, 5, 2))
    actual = batch_axial_orbit_diameters(coefficients)
    expected = 2.0 * np.linalg.svd(coefficients, compute_uv=False)[:, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    assert not actual.flags.writeable


def test_lipschitz_sample_certificate_contains_complete_orbit_diameter() -> None:
    coefficient = np.array(
        [
            [0.7, -0.2],
            [0.3, 0.9],
            [-0.4, 0.1],
        ]
    )
    true_diameter = float(batch_axial_orbit_diameters(coefficient))
    lipschitz = true_diameter / 2.0
    for count in (7, 15, 31, 63):
        angles = np.arange(count) * (2.0 * np.pi / count)
        harmonic = np.column_stack((np.cos(angles), np.sin(angles)))
        values = harmonic @ coefficient.T
        certificate = certify_uniformly_sampled_periodic_query(
            values,
            angular_lipschitz_bound=lipschitz,
            invariance_tolerance=0.5,
        )
        assert certificate.lower_diameter <= true_diameter + 1e-14
        assert true_diameter <= certificate.upper_diameter + 1e-14
        assert certificate.decision == "certified-variant"


def test_sampled_pairwise_diameter_handles_vector_queries() -> None:
    values = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-3.0, -4.0],
        ]
    )
    assert sampled_pairwise_diameter(values) == 10.0


@pytest.mark.parametrize(
    "coefficients",
    [
        [],
        [1.0, 2.0],
        np.ones((2, 3)),
        np.ones((0, 2)),
        [[np.nan, 0.0]],
        [[1j, 0.0]],
    ],
)
def test_invalid_axial_coefficients_are_rejected(coefficients) -> None:
    with pytest.raises(ValueError):
        certify_axial_linear_query(
            coefficients,
            coefficient_error_bound=0.0,
            invariance_tolerance=1.0,
        )


@pytest.mark.parametrize(
    "error,tolerance,angle",
    [
        (-1.0, 1.0, 0.0),
        (np.nan, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, np.inf),
        (True, 1.0, 0.0),
    ],
)
def test_invalid_certificate_scalars_are_rejected(error, tolerance, angle) -> None:
    with pytest.raises(ValueError):
        certify_axial_linear_query(
            [[1.0, 0.0]],
            coefficient_error_bound=error,
            invariance_tolerance=tolerance,
            nominal_angle_radians=angle,
        )


def test_invalid_sampled_periodic_contracts_are_rejected() -> None:
    for values in ([], [[1.0]], [[1.0], [2.0]], [[1.0], [2.0], [np.nan]]):
        with pytest.raises(ValueError):
            certify_uniformly_sampled_periodic_query(
                values,
                angular_lipschitz_bound=1.0,
                invariance_tolerance=1.0,
            )
    with pytest.raises(ValueError):
        certify_uniformly_sampled_periodic_query(
            [[0.0], [1.0], [0.0]],
            angular_lipschitz_bound=-1.0,
            invariance_tolerance=1.0,
        )


def test_registered_study_protocol_and_reduced_deterministic_run() -> None:
    module = _load_study()
    protocol = module._load_protocol(PROTOCOL)
    assert protocol["schema"] == "prob4d.robust-orbit-query-stress.v1"
    assert protocol["case_count"] == 20000
    assert protocol["coefficient_error_ratios"] == [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    assert protocol["sampled_grid_counts"] == [7, 15, 31, 63]

    reduced = json.loads(json.dumps(protocol))
    reduced["case_count"] = 2000
    reduced["near_boundary_case_count"] = 2000
    reduced["sampled_grid_counts"] = [7, 15]
    reduced["benchmark"] = {
        "query_count": 256,
        "query_dimension": 3,
        "angular_sample_count": 16,
        "repetitions": 2,
    }
    first = module.build_report(reduced)
    second = module.build_report(reduced)
    assert first["decision"] == "passed"
    assert first["registered_checks"] == second["registered_checks"]
    assert first["population"] == second["population"]
    assert first["bounded_error_rows"] == second["bounded_error_rows"]
    assert first["sampled_orbit_rows"] == second["sampled_orbit_rows"]
    assert first["diameter_parity"] == second["diameter_parity"]
