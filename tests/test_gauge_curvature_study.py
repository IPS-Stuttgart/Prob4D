"""Verify the independent cubature comparators and exact-risk arithmetic."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge_curvature import finite_difference_gauge_moments

SPEC = importlib.util.spec_from_file_location(
    "gauge_curvature_study",
    Path(__file__).resolve().parents[1] / "examples/gauge_curvature_study.py",
)
assert SPEC is not None and SPEC.loader is not None
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


@pytest.mark.parametrize("rank", [1, 2, 4, 7])
def test_fifth_degree_rule_matches_gaussian_moments(rank):
    for degree, exact in ((0, 1), (1, 0), (2, 1), (3, 0), (4, 3), (5, 0)):
        mean, _ = STUDY.fifth_degree_cubature(lambda z, degree=degree: z[0]**degree, rank)
        np.testing.assert_allclose(mean, exact, atol=2e-13)
    if rank >= 2:
        mean, _ = STUDY.fifth_degree_cubature(lambda z: z[0]**2 * z[1]**2, rank)
        np.testing.assert_allclose(mean, 1.0, atol=2e-13)


@pytest.mark.parametrize("sigma", [0.025, 0.1, 0.35])
def test_independent_hermite_reference_matches_analytic_oracle(sigma):
    _, variance = STUDY.gauss_hermite_product(
        lambda z: 0.5 * math.sin(sigma * z[0]) * math.sin(sigma * z[1]), 2, 15,
    )
    np.testing.assert_allclose(
        variance, STUDY.exact_sine_product_variance(sigma, 0.5), rtol=1e-13,
    )


def test_oracle_second_moments_have_nees_one_without_gaussian_noise_assumption():
    result = STUDY.gaussian_update_metrics(0.002**2, 0.005**2, 0.005**2)
    np.testing.assert_allclose(result["expected_nees"], 1.0, atol=1e-15)
    np.testing.assert_allclose(result["expected_mse_m2"], result["posterior_variance_m2"])


def test_exact_physical_fallback_ignores_observation_noise():
    result = STUDY.gaussian_update_metrics(0.002**2, 100.0, None)
    assert result["gain"] == 0
    assert result["expected_rmse_mm"] == 2
    assert result["expected_nees"] == 1


def test_complex_gauge_inputs_fail_instead_of_discarding_imaginary_parts():
    with pytest.raises(ValueError, match="real-valued"):
        finite_difference_gauge_moments(lambda x: x, np.array([1 + 2j]), [[1.0]])


def test_fourth_moment_complete_reference_is_not_axis_rule():
    _, axis = STUDY.axis_cubature(lambda z: z[0] * z[1], 7)
    _, fifth = STUDY.fifth_degree_cubature(lambda z: z[0] * z[1], 7)
    assert axis == 0
    np.testing.assert_allclose(fifth, 1.0, atol=1e-15)


def test_fifth_degree_signed_weights_are_not_a_psd_guarantee():
    # Rank seven has negative axial weights. An output concentrated at one
    # such node gives a negative quadrature variance. This is an adversarial
    # mathematical test, not a representative Sim3 performance claim.
    def spike(z):
        return float(abs(z[0] - 3.0) < 1e-12 and np.count_nonzero(z[1:]) == 0)

    _, variance = STUDY.fifth_degree_cubature(spike, 7)
    assert variance < 0
