"""Algebraic/adversarial tests; these are not real-provider evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.query_information_audit import (
    QueryInformationPolicy,
    assess_query_update,
    audit_gaussian_update,
)


def problem() -> dict[str, np.ndarray]:
    prior = np.diag([0.1, 0.8, 0.3, 0.4, 0.5, 0.2, 0.6])
    prior[0, 1] = prior[1, 0] = 0.08
    mean = np.array([0.2, 0.4, -0.1, 0.2, 0.0, 0.3, -0.2])
    information = np.diag([10.0, 0.0, 5.0, 8.0, 4.0, 9.0, 7.0])
    natural = information @ np.array([0.1, 0.0, 0.2, 0.3, 0.1, -0.1, 0.2])
    inverse = np.linalg.solve(prior, np.eye(7))
    posterior = np.linalg.solve(inverse + information, np.eye(7))
    return {
        "prior_mean": mean,
        "prior_covariance": prior,
        "candidate_mean": posterior @ (inverse @ mean + natural),
        "candidate_covariance": posterior,
        "likelihood_information": information,
        "likelihood_natural_parameter": natural,
    }


def test_exact_posterior_matches_independent_measurement_space_update() -> None:
    values = problem()
    information = values["likelihood_information"]
    indices = np.flatnonzero(np.diag(information))
    design = np.eye(7)[indices]
    noise = np.diag(1.0 / np.diag(information)[indices])
    observation = noise @ values["likelihood_natural_parameter"][indices]
    prior = values["prior_covariance"]
    gain = np.linalg.solve(design @ prior @ design.T + noise, design @ prior).T
    mean = values["prior_mean"] + gain @ (observation - design @ values["prior_mean"])
    covariance = prior - gain @ design @ prior
    np.testing.assert_allclose(mean, values["candidate_mean"], atol=2e-15)
    np.testing.assert_allclose(covariance, values["candidate_covariance"], atol=2e-15)
    report = audit_gaussian_update(**values)
    assert report.valid
    assert report.information_relative_error < 1e-14
    assert report.natural_parameter_relative_error < 1e-14


def test_fabricated_nullspace_precision_is_rejected() -> None:
    values = problem()
    inverse = np.linalg.solve(values["prior_covariance"], np.eye(7))
    extra = np.zeros((7, 7))
    extra[1, 1] = 20.0
    covariance = np.linalg.solve(inverse + values["likelihood_information"] + extra, np.eye(7))
    values["candidate_covariance"] = covariance
    values["candidate_mean"] = covariance @ (
        inverse @ values["prior_mean"] + values["likelihood_natural_parameter"]
    )
    report = audit_gaussian_update(**values)
    assert not report.valid
    assert "information-update-mismatch" in report.reason_codes
    assert report.unsupported_information_relative_norm > 0.0


def test_covariance_only_audit_misses_unjustified_mean() -> None:
    values = problem()
    values["candidate_mean"][1] += 0.7
    report = audit_gaussian_update(**values)
    assert report.information_relative_error < 1e-14
    assert not report.valid
    assert report.reason_codes == ("natural-parameter-update-mismatch",)


def test_repeated_information_is_detected_even_in_observable_subspace() -> None:
    values = problem()
    inverse = np.linalg.solve(values["prior_covariance"], np.eye(7))
    covariance = np.linalg.solve(inverse + 4.0 * values["likelihood_information"], np.eye(7))
    values["candidate_covariance"] = covariance
    values["candidate_mean"] = covariance @ (
        inverse @ values["prior_mean"] + 4.0 * values["likelihood_natural_parameter"]
    )
    report = audit_gaussian_update(**values)
    assert not report.valid
    assert "information-update-mismatch" in report.reason_codes


def test_linear_reparameterization_and_origin_invariance() -> None:
    values = problem()
    values["candidate_mean"][1] += 0.1
    reference = audit_gaussian_update(**values)
    rng = np.random.default_rng(12)
    orthogonal = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    transform = orthogonal @ np.diag([0.1, 0.3, 1.0, 2.0, 4.0, 7.0, 10.0])
    inverse = np.linalg.solve(transform, np.eye(7))
    offset = np.array([2.0, 1.0, 3.0, 4.0, -1.0, 0.0, 0.5])
    info = inverse.T @ values["likelihood_information"] @ inverse
    mapped = {
        "prior_mean": transform @ values["prior_mean"] + offset,
        "prior_covariance": transform @ values["prior_covariance"] @ transform.T,
        "candidate_mean": transform @ values["candidate_mean"] + offset,
        "candidate_covariance": transform @ values["candidate_covariance"] @ transform.T,
        "likelihood_information": info,
        "likelihood_natural_parameter": inverse.T @ values["likelihood_natural_parameter"]
        + info @ offset,
    }
    result = audit_gaussian_update(**mapped)
    assert result.reason_codes == reference.reason_codes
    assert result.natural_parameter_relative_error == pytest.approx(
        reference.natural_parameter_relative_error, rel=1e-10, abs=1e-11
    )
    assert result.information_relative_error < 1e-10


def test_factor_supported_query() -> None:
    values = problem()
    jacobian = np.eye(7)[[2, 3, 4]]
    result = assess_query_update(**values, query_jacobian=jacobian, query_tolerances=np.ones(3))
    assert result.admitted
    assert result.route == "factor-supported"


def test_prior_bounded_query_is_distinct_from_direct_support() -> None:
    values = problem()
    jacobian = np.eye(7)[[0, 1, 2]]
    result = assess_query_update(**values, query_jacobian=jacobian, query_tolerances=np.ones(3))
    assert result.admitted
    assert result.route == "prior-bounded"
    assert result.nullspace_sensitivity_fraction > 0.0
    policy = replace(QueryInformationPolicy(), allow_prior_bounded=False)
    denied = assess_query_update(
        **values, query_jacobian=jacobian, query_tolerances=np.ones(3), policy=policy
    )
    assert not denied.admitted
    assert denied.route == "fallback-unresolved"


def test_absolute_tolerance_catches_unresolved_query() -> None:
    values = problem()
    result = assess_query_update(
        **values, query_jacobian=np.eye(7)[[0, 1, 2]], query_tolerances=np.full(3, 0.01)
    )
    assert not result.admitted
    assert result.route == "fallback-unresolved"
    assert "query-uncertainty-exceeds-tolerance" in result.reason_codes


def test_invalid_mean_cannot_be_rescued_by_tiny_reported_uncertainty() -> None:
    values = problem()
    values["candidate_mean"][1] += 1.0
    result = assess_query_update(
        **values, query_jacobian=np.eye(7)[[0, 1, 2]], query_tolerances=np.full(3, 100.0)
    )
    assert not result.admitted
    assert result.maximum_standardized_variance < 1.0
    assert result.route == "fallback-invalid-update"


def test_joint_direction_not_just_marginal_variance() -> None:
    values = {
        "prior_mean": np.zeros(2), "prior_covariance": 2.0 * np.eye(2),
        "candidate_mean": np.zeros(2),
        "candidate_covariance": np.array([[0.8, 0.7], [0.7, 0.8]]),
    }
    values["likelihood_information"] = np.linalg.solve(
        values["candidate_covariance"], np.eye(2)
    ) - 0.5 * np.eye(2)
    values["likelihood_natural_parameter"] = np.zeros(2)
    result = assess_query_update(
        **values, query_jacobian=np.eye(2), query_tolerances=np.ones(2)
    )
    assert result.audit.valid
    assert result.maximum_standardized_variance == pytest.approx(1.5)
    assert not result.admitted


@pytest.mark.parametrize("field", ["prior_covariance", "candidate_covariance"])
def test_singular_covariance_raises_without_numerical_repair(field: str) -> None:
    values = problem()
    values[field] = np.zeros((7, 7))
    with pytest.raises(ValueError, match="positive definite"):
        audit_gaussian_update(**values)


@pytest.mark.parametrize("bad", [np.nan, -1.0, float("inf")])
def test_invalid_tolerance_raises(bad: float) -> None:
    values = problem()
    with pytest.raises(ValueError):
        assess_query_update(
            **values, query_jacobian=np.eye(7)[[2, 3, 4]],
            query_tolerances=np.array([1.0, bad, 1.0]),
        )


def test_zero_information_does_not_authorize_useful_update() -> None:
    values = problem()
    values["likelihood_information"] = np.zeros((7, 7))
    values["likelihood_natural_parameter"] = np.zeros(7)
    values["candidate_mean"] = values["prior_mean"].copy()
    values["candidate_covariance"] = values["prior_covariance"].copy()
    result = assess_query_update(
        **values, query_jacobian=np.eye(7), query_tolerances=np.full(7, 100.0)
    )
    assert result.audit.valid
    assert not result.admitted
    assert "insufficient-query-variance-reduction" in result.reason_codes


def test_audit_does_not_establish_calibration() -> None:
    values = problem()
    # A mutually consistent but wrong factor is deliberately indistinguishable
    # algebraically. This boundary prevents overclaiming the new audit.
    inverse = np.linalg.solve(values["prior_covariance"], np.eye(7))
    values["likelihood_information"] *= 100.0
    values["likelihood_natural_parameter"] *= 100.0
    covariance = np.linalg.solve(inverse + values["likelihood_information"], np.eye(7))
    values["candidate_covariance"] = covariance
    values["candidate_mean"] = covariance @ (
        inverse @ values["prior_mean"] + values["likelihood_natural_parameter"]
    )
    assert audit_gaussian_update(**values).valid


def test_existing_observable_gauge_factor_integration() -> None:
    # Available in a full Prob4D checkout; a partial offline file bundle skips.
    module = pytest.importorskip("prob4d.observable_gauge")
    sim3 = pytest.importorskip("prob4d.sim3")
    chart = module.CentroidGaugeChart(
        sim3.Sim3(scale=1.0, rotation=np.eye(3), translation=np.zeros(3)), np.zeros(3), 1.0
    )
    indices = [0, 2, 3, 4, 5, 6]
    factor = module.ObservableGaugeFactor(
        chart=chart, observable_basis=np.eye(7)[:, indices],
        nullspace_basis=np.eye(7)[:, [1]], observable_information=10.0 * np.eye(6),
        normalized_geometry_spectrum=np.array([1.0] * 6 + [0.0]), rank_threshold=1e-8,
        residual_rms=0.1, residual_variance=0.01, inlier_fraction=1.0,
        num_correspondences=8, covariance_method="analytic-test",
    )
    mean = np.arange(7, dtype=float) / 10.0
    prior = np.eye(7)
    posterior = factor.fuse_local_gaussian(mean, prior)
    report = audit_gaussian_update(
        prior_mean=mean, prior_covariance=prior,
        candidate_mean=posterior.mean_local, candidate_covariance=posterior.covariance_local,
        likelihood_information=factor.information_matrix,
        likelihood_natural_parameter=np.zeros(7),
    )
    assert report.valid
