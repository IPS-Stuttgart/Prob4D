"""Audit the information actually added by a Gaussian query-update proposal.

Experimental, array-only companion to ``query_observability``. The expected
likelihood must come from independently retained factors, not from the proposal
being checked. Agreement establishes Gaussian algebra, NOT factor calibration,
correct linearization, or physical truth. The caller owns complete-belief routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _array(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _symmetric(value: Any, size: int, name: str, *, definite: bool) -> FloatArray:
    array = _array(value, (size, size), name)
    scale = max(float(np.linalg.norm(array, ord=2)), np.finfo(float).tiny)
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-10 * scale):
        raise ValueError(f"{name} must be symmetric")
    array = (array + array.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(array)
    if definite:
        if eigenvalues[0] <= 0.0:
            raise ValueError(f"{name} must be positive definite")
    elif eigenvalues[0] < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return array


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number, not a bool")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class QueryInformationPolicy:
    """Outcome-independent tolerances in a declared linear-Gaussian chart."""

    information_rtol: float = 1e-8
    natural_parameter_rtol: float = 1e-8
    rank_rtol: float = 1e-10
    direct_nullspace_fraction_max: float = 1e-8
    minimum_variance_reduction: float = 0.05
    allow_prior_bounded: bool = True

    def __post_init__(self) -> None:
        for name in (
            "information_rtol", "natural_parameter_rtol", "rank_rtol",
            "direct_nullspace_fraction_max", "minimum_variance_reduction",
        ):
            number = _nonnegative(getattr(self, name), name)
            if name in (
                "rank_rtol", "direct_nullspace_fraction_max", "minimum_variance_reduction"
            ) and number >= 1.0:
                raise ValueError(f"{name} must be less than one")
            object.__setattr__(self, name, number)
        if self.rank_rtol == 0.0:
            raise ValueError("rank_rtol must be positive")
        if type(self.allow_prior_bounded) is not bool:
            raise TypeError("allow_prior_bounded must be a bool")


@dataclass(frozen=True)
class GaussianInformationAudit:
    """Unit-scaled residuals of both Gaussian natural-parameter equations."""

    valid: bool
    information_relative_error: float
    natural_parameter_relative_error: float
    unsupported_information_relative_norm: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class QueryInformationDecision:
    """Admission metadata only; no belief, covariance, or fallback is changed."""

    admitted: bool
    route: str
    audit: GaussianInformationAudit
    nullspace_sensitivity_fraction: float
    maximum_standardized_variance: float
    variance_reduction_fraction: float
    reason_codes: tuple[str, ...]


def audit_gaussian_update(
    *,
    prior_mean: FloatArray,
    prior_covariance: FloatArray,
    candidate_mean: FloatArray,
    candidate_covariance: FloatArray,
    likelihood_information: FloatArray,
    likelihood_natural_parameter: FloatArray,
    policy: QueryInformationPolicy | None = None,
) -> GaussianInformationAudit:
    """Check a proposal against a separately supplied likelihood (Lambda, eta).

    With ``P0=L L.T``, compare ``L.T (Pc^-1-P0^-1) L`` with ``L.T Lambda L``.
    Also compare ``L.T Pc^-1 (mc-m0)`` with ``L.T (eta-Lambda m0)``. Centering
    at the prior mean avoids dependence on the coordinate origin. Norms are
    invariant to invertible linear reparameterizations, up to roundoff. The
    reported nullspace norm uses the prior-whitened expected information.

    A covariance-only check misses an unjustified mean change. Conversely,
    perfect agreement cannot detect wrong but mutually consistent input factors.
    Invalid array shapes/covariances raise rather than authorizing an update.
    """
    policy = QueryInformationPolicy() if policy is None else policy
    mean = np.asarray(prior_mean, dtype=np.float64)
    if mean.ndim != 1 or mean.size == 0 or not np.all(np.isfinite(mean)):
        raise ValueError("prior_mean must be a finite nonempty vector")
    size = int(mean.size)
    proposed = _array(candidate_mean, (size,), "candidate_mean")
    prior = _symmetric(prior_covariance, size, "prior_covariance", definite=True)
    candidate = _symmetric(candidate_covariance, size, "candidate_covariance", definite=True)
    information = _symmetric(
        likelihood_information, size, "likelihood_information", definite=False
    )
    natural = _array(likelihood_natural_parameter, (size,), "likelihood_natural_parameter")
    chol = np.linalg.cholesky(prior)
    expected = chol.T @ information @ chol
    observed = chol.T @ np.linalg.solve(candidate, chol) - np.eye(size)
    matrix_scale = max(1.0, float(np.linalg.norm(expected, ord="fro")))
    matrix_error = float(np.linalg.norm(observed - expected, ord="fro") / matrix_scale)
    expected_vector = chol.T @ (natural - information @ mean)
    observed_vector = chol.T @ np.linalg.solve(candidate, proposed - mean)
    vector_error = float(
        np.linalg.norm(observed_vector - expected_vector)
        / max(1.0, np.linalg.norm(expected_vector))
    )
    eigenvalues, eigenvectors = np.linalg.eigh((expected + expected.T) * 0.5)
    threshold = policy.rank_rtol * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    nullspace = eigenvectors[:, eigenvalues <= threshold]
    unsupported = float(np.linalg.norm(nullspace.T @ observed, ord="fro") / matrix_scale)
    reasons = []
    if matrix_error > policy.information_rtol:
        reasons.append("information-update-mismatch")
    if vector_error > policy.natural_parameter_rtol:
        reasons.append("natural-parameter-update-mismatch")
    return GaussianInformationAudit(
        not reasons, matrix_error, vector_error, unsupported, tuple(reasons)
    )


def assess_query_update(
    *,
    prior_mean: FloatArray,
    prior_covariance: FloatArray,
    candidate_mean: FloatArray,
    candidate_covariance: FloatArray,
    likelihood_information: FloatArray,
    likelihood_natural_parameter: FloatArray,
    query_jacobian: FloatArray,
    query_tolerances: FloatArray,
    policy: QueryInformationPolicy | None = None,
) -> QueryInformationDecision:
    """Require justified information, absolute query precision, and useful gain.

    Tolerances are positive one-standard-deviation scales, NOT coverage claims.
    The largest eigenvalue of ``diag(1/tol) J Pc J.T diag(1/tol)`` must not
    exceed one. This controls every standardized linear query direction, not
    merely its diagonal. Query support is measured in the caller-declared chart;
    prior-bounded does not mean directly measured. A gate is not a harm guarantee.
    """
    policy = QueryInformationPolicy() if policy is None else policy
    audit = audit_gaussian_update(
        prior_mean=prior_mean, prior_covariance=prior_covariance,
        candidate_mean=candidate_mean, candidate_covariance=candidate_covariance,
        likelihood_information=likelihood_information,
        likelihood_natural_parameter=likelihood_natural_parameter, policy=policy,
    )
    size = int(np.asarray(prior_mean).size)
    jacobian = np.asarray(query_jacobian, dtype=np.float64)
    if (
        jacobian.ndim != 2 or jacobian.shape[0] < 1 or jacobian.shape[1] != size
        or not np.all(np.isfinite(jacobian))
    ):
        raise ValueError("query_jacobian must be a finite nonempty Q-by-state matrix")
    tolerances = _array(query_tolerances, (jacobian.shape[0],), "query_tolerances")
    if np.any(tolerances <= 0.0):
        raise ValueError("query_tolerances must be positive")
    standardized = jacobian / tolerances[:, None]
    prior_query = standardized @ np.asarray(prior_covariance) @ standardized.T
    candidate_query = standardized @ np.asarray(candidate_covariance) @ standardized.T
    candidate_query = (candidate_query + candidate_query.T) * 0.5
    maximum = max(0.0, float(np.linalg.eigvalsh(candidate_query)[-1]))
    prior_trace = float(np.trace(prior_query))
    reduction = (
        float(1.0 - np.trace(candidate_query) / prior_trace) if prior_trace > 0.0 else 0.0
    )
    information = np.asarray(likelihood_information, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh((information + information.T) * 0.5)
    threshold = policy.rank_rtol * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    nullspace = eigenvectors[:, eigenvalues <= threshold]
    sensitivity = float(np.sum(standardized**2))
    null_fraction = (
        float(np.sum((standardized @ nullspace) ** 2) / sensitivity) if sensitivity > 0 else 0.0
    )
    reasons = list(audit.reason_codes)
    direct = null_fraction <= policy.direct_nullspace_fraction_max
    if maximum > 1.0:
        reasons.append("query-uncertainty-exceeds-tolerance")
    if reduction < policy.minimum_variance_reduction:
        reasons.append("insufficient-query-variance-reduction")
    if not direct and not policy.allow_prior_bounded:
        reasons.append("prior-bounded-query-not-permitted")
    if not audit.valid:
        route = "fallback-invalid-update"
    elif reasons:
        route = "fallback-unresolved"
    else:
        route = "factor-supported" if direct else "prior-bounded"
    return QueryInformationDecision(
        not reasons, route, audit, null_fraction, maximum, reduction, tuple(reasons)
    )
