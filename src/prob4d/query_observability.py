"""Decision-relevant observability diagnostics for partial Sim(3) gauge factors.

A rank-deficient visual factor can be useful for one physical query and
uninformative for another.  This module projects an
:class:`~prob4d.observable_gauge.ObservableGaugeFactor` through a user-supplied
query Jacobian in the factor's centroid-normalized local chart.  It reports
direct geometric support separately from prior-mediated posterior variance
reduction and provides a threshold gate whose values must be frozen on source
or calibration groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .observable_gauge import ObservableGaugeFactor
from .sim3 import skew

FloatArray: TypeAlias = NDArray[np.floating[Any]]

_DIRECT_SUPPORT_REASON = "insufficient-direct-query-observability"
_VARIANCE_REDUCTION_REASON = "insufficient-query-variance-reduction"
_WORST_DIRECTION_REASON = "excessive-worst-direction-variance-ratio"


def _readonly_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
    columns: int | None = None,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if columns is not None and (array.shape[0] < 1 or array.shape[1] != columns):
        raise ValueError(f"{name} must have shape (Q, {columns}) with Q positive")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _symmetric_positive_definite_square(
    value: object,
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (array + array.T)
    if not np.allclose(array, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(symmetric))) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    symmetric.setflags(write=False)
    return symmetric


def _symmetric_positive_definite(value: object, *, name: str) -> FloatArray:
    matrix = _readonly_matrix(value, name=name, shape=(7, 7))
    return _symmetric_positive_definite_square(matrix, name=name)


def _symmetric_positive_semidefinite(
    value: FloatArray,
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    symmetric = 0.5 * (array + array.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if float(np.min(np.linalg.eigvalsh(symmetric))) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    symmetric = symmetric.copy()
    symmetric.setflags(write=False)
    return symmetric


def _unit_interval(value: object, *, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return numeric


def _supported_variance_ratio(
    prior_covariance: FloatArray,
    posterior_covariance: FloatArray,
) -> float:
    # These covariances are expressed in the declared output metric. A shared
    # normalization makes support independent of absolute units; an epsilon
    # floor on the unscaled eigenvalues can erase an unresolved query.
    scale = float(np.max(np.abs(prior_covariance), initial=0.0))
    if not np.isfinite(scale):
        raise ValueError("diagnostic query covariance must be finite")
    if scale == 0.0:
        return 0.0
    normalized_prior = prior_covariance / scale
    normalized_posterior = posterior_covariance / scale
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_prior)
    maximum = float(np.max(eigenvalues, initial=0.0))
    supported = eigenvalues > maximum * 1e-10
    if not np.any(supported):
        return 0.0
    whitener = (
        eigenvectors[:, supported] / np.sqrt(eigenvalues[supported])
    ).T
    relative = whitener @ normalized_posterior @ whitener.T
    relative = 0.5 * (relative + relative.T)
    maximum_ratio = float(np.max(np.linalg.eigvalsh(relative), initial=0.0))
    if maximum_ratio > 1.0 + 1e-8:
        raise ValueError("posterior query covariance exceeds the prior covariance")
    return float(np.clip(maximum_ratio, 0.0, 1.0))


@dataclass(frozen=True)
class QueryObservabilityReport:
    """Projection of one partial gauge factor into a downstream query."""

    factor_rank: int
    query_dimension: int
    direct_observability_fraction: float
    nullspace_sensitivity_fraction: float
    query_metric: FloatArray
    prior_query_covariance: FloatArray
    posterior_query_covariance: FloatArray
    metric_variance_reduction_fraction: float
    worst_supported_variance_ratio: float
    gauge_invariant_query: bool

    def __post_init__(self) -> None:
        if isinstance(self.factor_rank, bool) or not isinstance(
            self.factor_rank, (int, np.integer)
        ):
            raise TypeError("factor_rank must be an integer")
        factor_rank = int(self.factor_rank)
        if not 1 <= factor_rank <= 7:
            raise ValueError("factor_rank must lie in [1, 7]")
        if isinstance(self.query_dimension, bool) or not isinstance(
            self.query_dimension, (int, np.integer)
        ):
            raise TypeError("query_dimension must be an integer")
        query_dimension = int(self.query_dimension)
        if query_dimension < 1:
            raise ValueError("query_dimension must be positive")
        direct = _unit_interval(
            self.direct_observability_fraction,
            name="direct_observability_fraction",
        )
        nullspace = _unit_interval(
            self.nullspace_sensitivity_fraction,
            name="nullspace_sensitivity_fraction",
        )
        if not np.isclose(direct + nullspace, 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError(
                "direct and nullspace sensitivity fractions must sum to one"
            )
        query_metric = _readonly_matrix(
            self.query_metric,
            name="query_metric",
            shape=(query_dimension, query_dimension),
        )
        query_metric = _symmetric_positive_definite_square(
            query_metric,
            name="query_metric",
        )
        prior = _readonly_matrix(
            self.prior_query_covariance,
            name="prior_query_covariance",
            shape=(query_dimension, query_dimension),
        )
        prior = _symmetric_positive_semidefinite(
            prior,
            name="prior_query_covariance",
        )
        posterior = _readonly_matrix(
            self.posterior_query_covariance,
            name="posterior_query_covariance",
            shape=(query_dimension, query_dimension),
        )
        posterior = _symmetric_positive_semidefinite(
            posterior,
            name="posterior_query_covariance",
        )
        reduction = _unit_interval(
            self.metric_variance_reduction_fraction,
            name="metric_variance_reduction_fraction",
        )
        worst_ratio = _unit_interval(
            self.worst_supported_variance_ratio,
            name="worst_supported_variance_ratio",
        )
        if type(self.gauge_invariant_query) is not bool:
            raise TypeError("gauge_invariant_query must be a bool")
        object.__setattr__(self, "factor_rank", factor_rank)
        object.__setattr__(self, "query_dimension", query_dimension)
        object.__setattr__(self, "direct_observability_fraction", direct)
        object.__setattr__(self, "nullspace_sensitivity_fraction", nullspace)
        object.__setattr__(self, "query_metric", query_metric)
        object.__setattr__(self, "prior_query_covariance", prior)
        object.__setattr__(self, "posterior_query_covariance", posterior)
        object.__setattr__(
            self,
            "metric_variance_reduction_fraction",
            reduction,
        )
        object.__setattr__(
            self,
            "worst_supported_variance_ratio",
            worst_ratio,
        )

    @property
    def prior_metric_variance(self) -> float:
        return float(np.trace(self.query_metric @ self.prior_query_covariance))

    @property
    def posterior_metric_variance(self) -> float:
        return float(np.trace(self.query_metric @ self.posterior_query_covariance))


@dataclass(frozen=True)
class QueryObservabilityGate:
    """Source-frozen thresholds for admitting a query-specific factor."""

    minimum_direct_observability_fraction: float = 0.0
    minimum_metric_variance_reduction_fraction: float = 0.0
    maximum_worst_supported_variance_ratio: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_direct_observability_fraction",
            _unit_interval(
                self.minimum_direct_observability_fraction,
                name="minimum_direct_observability_fraction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_metric_variance_reduction_fraction",
            _unit_interval(
                self.minimum_metric_variance_reduction_fraction,
                name="minimum_metric_variance_reduction_fraction",
            ),
        )
        object.__setattr__(
            self,
            "maximum_worst_supported_variance_ratio",
            _unit_interval(
                self.maximum_worst_supported_variance_ratio,
                name="maximum_worst_supported_variance_ratio",
            ),
        )

    def evaluate(
        self,
        report: QueryObservabilityReport,
    ) -> QueryObservabilityDecision:
        """Return a deterministic admission decision for one report."""

        reasons: list[str] = []
        if (
            report.direct_observability_fraction
            < self.minimum_direct_observability_fraction
        ):
            reasons.append(_DIRECT_SUPPORT_REASON)
        if (
            report.metric_variance_reduction_fraction
            < self.minimum_metric_variance_reduction_fraction
        ):
            reasons.append(_VARIANCE_REDUCTION_REASON)
        if (
            report.worst_supported_variance_ratio
            > self.maximum_worst_supported_variance_ratio
        ):
            reasons.append(_WORST_DIRECTION_REASON)
        return QueryObservabilityDecision(
            admitted=not reasons,
            reason_codes=tuple(reasons),
        )


@dataclass(frozen=True)
class QueryObservabilityDecision:
    """Admission metadata; the caller owns exact fallback semantics."""

    admitted: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise TypeError("admitted must be a bool")
        reason_codes = tuple(str(reason) for reason in self.reason_codes)
        if any(not reason for reason in reason_codes):
            raise ValueError("reason_codes must contain only nonempty strings")
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("reason_codes must not contain duplicates")
        if self.admitted and reason_codes:
            raise ValueError("an admitted decision cannot contain rejection reasons")
        if not self.admitted and not reason_codes:
            raise ValueError("a rejected decision must contain at least one reason")
        object.__setattr__(self, "reason_codes", reason_codes)


def point_position_query_jacobian(
    factor: ObservableGaugeFactor,
    source_point: FloatArray,
) -> FloatArray:
    """Linearize one gauge-transformed source-point position.

    The returned ``3 x 7`` Jacobian uses the factor's intrinsic coordinates.
    It is suitable for endpoint, marker, or probe-position queries whose point
    is expressed in the moving/source window coordinates.
    """

    point = np.asarray(source_point, dtype=np.float64).copy()
    if point.shape != (3,):
        raise ValueError("source_point must have shape (3,)")
    if not np.all(np.isfinite(point)):
        raise ValueError("source_point must be finite")
    transformed = factor.chart.linearization.transform_points(point)
    centered = transformed - factor.chart.reference_centroid
    jacobian = np.empty((3, 7), dtype=np.float64)
    jacobian[:, 0] = centered
    jacobian[:, 1:4] = -skew(centered)
    jacobian[:, 4:7] = factor.chart.cloud_scale * np.eye(3)
    jacobian.setflags(write=False)
    return jacobian


def evaluate_query_observability(
    factor: ObservableGaugeFactor,
    *,
    prior_covariance_local: FloatArray,
    query_jacobian_local: FloatArray,
    query_metric: FloatArray | None = None,
) -> QueryObservabilityReport:
    """Project factor information and a complete prior into one local query.

    ``query_jacobian_local`` linearizes a downstream query with respect to the
    factor chart coordinates
    ``[log scale, left rotation(3), centroid translation / cloud scale]``.
    Direct observability depends only on the query, its declared output metric,
    and the factor subspaces. Variance reduction additionally depends on the
    supplied complete prior and therefore remains explicitly separate. The
    identity metric is used only when ``query_metric`` is omitted.
    """

    prior = _symmetric_positive_definite(
        prior_covariance_local,
        name="prior_covariance_local",
    )
    query_jacobian = _readonly_matrix(
        query_jacobian_local,
        name="query_jacobian_local",
        columns=7,
    )
    metric = (
        np.eye(query_jacobian.shape[0], dtype=np.float64)
        if query_metric is None
        else _symmetric_positive_definite_square(
            query_metric,
            name="query_metric",
        )
    )
    if metric.shape != (query_jacobian.shape[0], query_jacobian.shape[0]):
        raise ValueError("query_metric must match the query dimension")
    prior_information = np.linalg.solve(prior, np.eye(7))
    posterior_information = prior_information + factor.information_matrix
    posterior = np.linalg.solve(posterior_information, np.eye(7))
    posterior = 0.5 * (posterior + posterior.T)

    prior_query = _symmetric_positive_semidefinite(
        query_jacobian @ prior @ query_jacobian.T,
        name="prior_query_covariance",
    )
    posterior_query = _symmetric_positive_semidefinite(
        query_jacobian @ posterior @ query_jacobian.T,
        name="posterior_query_covariance",
    )

    metric_sqrt = np.linalg.cholesky(metric).T
    metric_jacobian = metric_sqrt @ query_jacobian
    jacobian_scale = float(np.max(np.abs(metric_jacobian), initial=0.0))
    if not np.isfinite(jacobian_scale):
        raise ValueError("metric-weighted query Jacobian must be finite")
    # This means local first-order insensitivity, not global invariance of a
    # nonlinear query. Small nonzero sensitivities must not become invariant
    # solely because the output units or metric scale changed.
    gauge_invariant = not bool(np.any(query_jacobian))
    if jacobian_scale == 0.0:
        if not gauge_invariant:
            raise ValueError("nonzero metric-weighted query Jacobian underflowed")
        normalized_jacobian = metric_jacobian
    else:
        normalized_jacobian = metric_jacobian / jacobian_scale
    observable_energy = float(
        np.sum((normalized_jacobian @ factor.observable_basis) ** 2)
    )
    nullspace_energy = float(
        np.sum((normalized_jacobian @ factor.nullspace_basis) ** 2)
    )
    total_energy = observable_energy + nullspace_energy
    if gauge_invariant:
        direct_fraction = 1.0
        nullspace_fraction = 0.0
    else:
        direct_fraction = observable_energy / total_energy
        nullspace_fraction = nullspace_energy / total_energy

    # Rank/support decisions must use the same metric as direct observability.
    # Under q_new = A q and M_new = A^{-T} M A^{-1}, these metric-space
    # covariances differ only by orthogonal congruence and a common scale.
    metric_prior_query = _symmetric_positive_semidefinite(
        normalized_jacobian @ prior @ normalized_jacobian.T,
        name="metric_prior_query_covariance",
    )
    metric_posterior_query = _symmetric_positive_semidefinite(
        normalized_jacobian @ posterior @ normalized_jacobian.T,
        name="metric_posterior_query_covariance",
    )
    covariance_scale = float(np.max(np.abs(metric_prior_query), initial=0.0))
    if not np.isfinite(covariance_scale):
        raise ValueError("metric-weighted prior query covariance must be finite")
    if covariance_scale == 0.0:
        if not gauge_invariant:
            raise ValueError("nonzero prior query covariance underflowed")
        trace_reduction = 0.0
    else:
        prior_trace = float(np.trace(metric_prior_query / covariance_scale))
        posterior_trace = float(
            np.trace(metric_posterior_query / covariance_scale)
        )
        trace_reduction = float(
            np.clip((prior_trace - posterior_trace) / prior_trace, 0.0, 1.0)
        )

    return QueryObservabilityReport(
        factor_rank=factor.rank,
        query_dimension=int(query_jacobian.shape[0]),
        direct_observability_fraction=direct_fraction,
        nullspace_sensitivity_fraction=nullspace_fraction,
        query_metric=metric,
        prior_query_covariance=prior_query,
        posterior_query_covariance=posterior_query,
        metric_variance_reduction_fraction=trace_reduction,
        worst_supported_variance_ratio=_supported_variance_ratio(
            metric_prior_query,
            metric_posterior_query,
        ),
        gauge_invariant_query=gauge_invariant,
    )


__all__ = [
    "QueryObservabilityDecision",
    "QueryObservabilityGate",
    "QueryObservabilityReport",
    "evaluate_query_observability",
    "point_position_query_jacobian",
]
