"""Conditional query information for correlated, partial 4-D gauge windows.

Experimental, linear-Gaussian research API. All rows must refer to one declared
seven-dimensional gauge chart. A full, externally calibrated joint *noise*
covariance is required: absence of cross-covariance is not independence.

The implementation conditions candidate noise on the actually assimilated
history. It never substitutes the candidate's marginal information, inserts a
ridge into a geometric nullspace, or uses candidate outcomes to select a window.
A utility is an expected squared-query-loss reduction, not a safety certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]


class UnsupportedDeterministicConstraint(ValueError):
    """A zero-noise direction constrains the state; use a constrained solver."""


def _array(value: object, *, name: str, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _symmetric(value: object, *, name: str, size: int) -> FloatArray:
    result = _array(value, name=name, ndim=2)
    if result.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    scale = max(float(np.max(np.abs(result), initial=0.0)), float(np.finfo(float).tiny))
    if float(np.max(np.abs(result - result.T), initial=0.0)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return _array((result + result.T) / 2, name=name, ndim=2)


def _positive_definite(value: object, *, name: str, size: int) -> FloatArray:
    result = _symmetric(value, name=name, size=size)
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _noise_support(
    covariance: FloatArray,
    design: FloatArray,
    *,
    rtol: float,
    reference_noise_scale: float = 0.0,
    reference_design_scale: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return a supported whitener and zero-noise basis, without a ridge."""
    if covariance.shape[0] == 0:
        return np.empty((0, 0)), np.empty((0, 0))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), reference_noise_scale)
    threshold = rtol * scale
    if float(eigenvalues[0]) < -threshold:
        raise ValueError("joint or conditional noise covariance is not positive semidefinite")
    supported = eigenvalues > threshold
    zero_basis = eigenvectors[:, ~supported]
    design_scale = max(float(np.linalg.norm(design)), reference_design_scale)
    if float(np.linalg.norm(zero_basis.T @ design)) > rtol * design_scale:
        raise UnsupportedDeterministicConstraint(
            "zero-noise direction contains state information; no pseudoinverse rescue"
        )
    whitener = (eigenvectors[:, supported] / np.sqrt(eigenvalues[supported])).T
    return whitener, zero_basis


def _covariance_update(covariance: FloatArray, design: FloatArray) -> FloatArray:
    """Joseph-form update for unit-noise supported measurements."""
    if not np.any(design):
        return covariance
    innovation = np.eye(design.shape[0]) + design @ covariance @ design.T
    gain = np.linalg.solve(innovation, design @ covariance).T
    residual = np.eye(7) - gain @ design
    updated = residual @ covariance @ residual.T + gain @ gain.T
    return _positive_definite(updated, name="posterior covariance", size=7)


@dataclass(frozen=True)
class GaussianGaugeBelief:
    """Complete Gaussian belief in the caller's declared local gauge chart."""

    chart_id: str
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        _text(self.chart_id, "chart_id")
        mean = _array(self.mean, name="mean", ndim=1)
        if mean.shape != (7,):
            raise ValueError("mean must have shape (7,)")
        covariance = _positive_definite(self.covariance, name="covariance", size=7)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class ConditionalWindowFactor:
    """An outcome-free conditional likelihood, including singular-noise support."""

    candidate_id: str
    history_ids: tuple[str, ...]
    regression: FloatArray
    design: FloatArray
    noise_covariance: FloatArray
    whitener: FloatArray
    zero_noise_basis: FloatArray
    history_zero_noise_basis: FloatArray
    noise_scale: float
    rtol: float

    @property
    def whitened_design(self) -> FloatArray:
        return self.whitener @ self.design

    @property
    def information_matrix(self) -> FloatArray:
        matrix = self.whitened_design
        return matrix.T @ matrix

    @property
    def information_rank(self) -> int:
        singular = np.linalg.svd(self.whitened_design, compute_uv=False)
        if singular.size == 0 or singular[0] == 0:
            return 0
        return int(np.count_nonzero(singular > self.rtol * singular[0]))

    def whitened_value(self, history: FloatArray, candidate: FloatArray) -> FloatArray:
        """Validate zero-noise identities before using supported residuals."""
        residual = candidate - self.regression @ history
        scale = max(
            float(np.linalg.norm(history)),
            float(np.linalg.norm(candidate)),
            float(np.linalg.norm(self.regression @ history)),
            float(np.sqrt(self.noise_scale)),
        )
        if (
            float(np.linalg.norm(self.history_zero_noise_basis.T @ history)) > self.rtol * scale
            or float(np.linalg.norm(self.zero_noise_basis.T @ residual)) > self.rtol * scale
        ):
            raise ValueError("observations violate a deterministic noise-support identity")
        return self.whitener @ residual


@dataclass(frozen=True)
class CorrelatedGaugeDesign:
    """Frozen window designs and their full, known joint observation noise.

    ``design_matrix`` maps one common local gauge vector to stacked measurements.
    ``noise_covariance`` is conditional on that vector, NOT a predictive covariance
    containing state uncertainty. Window IDs and sizes partition its row axis.
    ``dependence_id`` names the externally fitted covariance model; it is metadata,
    not independent evidence that the covariance is calibrated.

    Singular covariance is supported only when its zero-noise directions contain
    no state information (e.g. exact repeated source evidence). Informative exact
    constraints fail closed rather than being dropped by a pseudoinverse.
    """

    chart_id: str
    dependence_id: str
    window_ids: tuple[str, ...]
    window_sizes: tuple[int, ...]
    design_matrix: FloatArray
    noise_covariance: FloatArray
    rtol: float = 1e-10

    def __post_init__(self) -> None:
        _text(self.chart_id, "chart_id")
        _text(self.dependence_id, "dependence_id")
        ids = tuple(self.window_ids)
        sizes = tuple(self.window_sizes)
        if not ids or len(ids) != len(sizes) or len(set(ids)) != len(ids):
            raise ValueError("window IDs must be nonempty, unique, and match window sizes")
        for window_id in ids:
            _text(window_id, "window ID")
        if any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
            for size in sizes
        ):
            raise ValueError("window sizes must be positive integers")
        if isinstance(self.rtol, bool) or not np.isfinite(self.rtol) or not 0 < self.rtol < 1:
            raise ValueError("rtol must lie strictly between zero and one")
        design = _array(self.design_matrix, name="design_matrix", ndim=2)
        if design.shape != (sum(sizes), 7):
            raise ValueError("design_matrix must have shape (sum(window_sizes), 7)")
        covariance = _symmetric(self.noise_covariance, name="noise_covariance", size=sum(sizes))
        _noise_support(covariance, design, rtol=self.rtol)
        object.__setattr__(self, "window_ids", ids)
        object.__setattr__(self, "window_sizes", sizes)
        object.__setattr__(self, "design_matrix", design)
        object.__setattr__(self, "noise_covariance", covariance)

    def rows(self, window_ids: tuple[str, ...]) -> tuple[int, ...]:
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("a window cannot appear twice in a history")
        result: list[int] = []
        for window_id in window_ids:
            if window_id not in self.window_ids:
                raise ValueError(f"unknown window: {window_id}")
            index = self.window_ids.index(window_id)
            start = sum(self.window_sizes[:index])
            result.extend(range(start, start + self.window_sizes[index]))
        return tuple(result)

    def conditional_factor(
        self, history_ids: tuple[str, ...], candidate_id: str
    ) -> ConditionalWindowFactor:
        """Construct p(y_candidate | x, y_history) without reading any y."""
        if candidate_id in history_ids:
            raise ValueError("candidate has already been assimilated")
        old_rows = self.rows(history_ids)
        new_rows = self.rows((candidate_id,))
        old_design = self.design_matrix[list(old_rows)]
        new_design = self.design_matrix[list(new_rows)]
        old_noise = self.noise_covariance[np.ix_(old_rows, old_rows)]
        cross_noise = self.noise_covariance[np.ix_(new_rows, old_rows)]
        new_noise = self.noise_covariance[np.ix_(new_rows, new_rows)]
        old_whitener, old_zero = _noise_support(old_noise, old_design, rtol=self.rtol)
        old_inverse = old_whitener.T @ old_whitener
        regression = cross_noise @ old_inverse
        design = new_design - regression @ old_design
        design_scale = max(
            float(np.linalg.norm(new_design)),
            float(np.linalg.norm(regression @ old_design)),
        )
        if float(np.linalg.norm(design)) <= self.rtol * design_scale:
            design = np.zeros_like(design)
        noise = new_noise - regression @ cross_noise.T
        noise = (noise + noise.T) / 2
        noise_scale = float(np.max(np.linalg.eigvalsh(new_noise), initial=0.0))
        whitener, zero = _noise_support(
            noise,
            design,
            rtol=self.rtol,
            reference_noise_scale=noise_scale,
            reference_design_scale=design_scale,
        )
        return ConditionalWindowFactor(
            candidate_id=candidate_id,
            history_ids=tuple(history_ids),
            regression=_array(regression, name="regression", ndim=2),
            design=_array(design, name="conditional design", ndim=2),
            noise_covariance=_array(noise, name="conditional noise", ndim=2),
            whitener=_array(whitener, name="whitener", ndim=2),
            zero_noise_basis=_array(zero, name="zero_noise_basis", ndim=2),
            history_zero_noise_basis=_array(old_zero, name="history_zero_noise_basis", ndim=2),
            noise_scale=noise_scale,
            rtol=self.rtol,
        )


@dataclass(frozen=True)
class QueryWindowUtility:
    """Expected metric-squared query-loss reduction, not realized benefit."""

    candidate_id: str
    conditional_information_rank: int
    prior_metric_variance: float
    posterior_metric_variance: float
    cost: float

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        rank = self.conditional_information_rank
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 0 <= rank <= 7:
            raise ValueError("conditional_information_rank must be an integer in [0, 7]")
        for name in ("prior_metric_variance", "posterior_metric_variance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.cost, bool) or not np.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("cost must be finite and positive")
        scale = max(self.prior_metric_variance, float(np.finfo(float).tiny))
        if self.posterior_metric_variance > self.prior_metric_variance + 1e-10 * scale:
            raise ValueError("posterior query variance must not exceed prior variance")

    @property
    def variance_reduction(self) -> float:
        return max(0.0, self.prior_metric_variance - self.posterior_metric_variance)

    @property
    def reduction_per_cost(self) -> float:
        return self.variance_reduction / self.cost


class ConditionalGaugeSession:
    """Track actual evidence consumption so conditional updates use the right prior.

    The initial prior must be independent of this model's noise and must not have
    consumed any model window. Start a new session from that prior and replay the
    admitted history; do not wrap an already-updated belief as an untouched prior.
    """

    def __init__(self, model: CorrelatedGaugeDesign, prior: GaussianGaugeBelief) -> None:
        if model.chart_id != prior.chart_id:
            raise ValueError("model and prior must use the same declared gauge chart")
        self.model = model
        self._belief = prior
        self._history: dict[str, FloatArray] = {}

    @property
    def belief(self) -> GaussianGaugeBelief:
        return self._belief

    @property
    def history_ids(self) -> tuple[str, ...]:
        return tuple(self._history)

    def preview_query(
        self,
        candidate_id: str,
        query_jacobian: FloatArray,
        *,
        query_metric: FloatArray | None = None,
        cost: float = 1.0,
    ) -> QueryWindowUtility:
        query = _array(query_jacobian, name="query_jacobian", ndim=2)
        if query.shape[0] < 1 or query.shape[1] != 7:
            raise ValueError("query_jacobian must have shape (Q, 7), Q positive")
        metric = (
            np.eye(query.shape[0])
            if query_metric is None
            else _positive_definite(query_metric, name="query_metric", size=query.shape[0])
        )
        if isinstance(cost, bool) or not np.isfinite(cost) or cost <= 0:
            raise ValueError("cost must be finite and positive")
        factor = self.model.conditional_factor(self.history_ids, candidate_id)
        before = self.belief.covariance
        after = _covariance_update(before, factor.whitened_design)
        prior_variance = float(np.trace(metric @ query @ before @ query.T))
        posterior_variance = float(np.trace(metric @ query @ after @ query.T))
        variance_scale = max(prior_variance, float(np.finfo(float).tiny))
        if posterior_variance > prior_variance + 1e-10 * variance_scale:
            raise ValueError("conditional update increased query covariance")
        return QueryWindowUtility(
            candidate_id, factor.information_rank, prior_variance, posterior_variance, float(cost)
        )

    def assimilate(self, candidate_id: str, value: FloatArray) -> GaussianGaugeBelief:
        factor = self.model.conditional_factor(self.history_ids, candidate_id)
        candidate = _array(value, name="candidate value", ndim=1)
        if candidate.shape != (factor.design.shape[0],):
            raise ValueError("candidate value does not match the window size")
        history = np.concatenate(tuple(self._history.values())) if self._history else np.empty(0)
        whitened_value = factor.whitened_value(history, candidate)
        design = factor.whitened_design
        prior = self.belief
        posterior = prior
        if np.any(design):
            innovation = np.eye(design.shape[0]) + design @ prior.covariance @ design.T
            gain = np.linalg.solve(innovation, design @ prior.covariance).T
            mean = prior.mean + gain @ (whitened_value - design @ prior.mean)
            posterior = GaussianGaugeBelief(
                prior.chart_id, mean, _covariance_update(prior.covariance, design)
            )
        self._history[candidate_id] = candidate
        self._belief = posterior
        return posterior


def select_query_window(
    utilities: tuple[QueryWindowUtility, ...], *, minimum_gain_per_cost: float = 0.0
) -> str | None:
    """One-step, outcome-free selection; lexical ties; None means no new update.

    This is not a globally optimal multi-window schedule. Correlated observation
    utilities need not be submodular; a multi-step approximation guarantee is not
    asserted. Thresholds and costs must be frozen before candidate outcomes.
    """
    if not np.isfinite(minimum_gain_per_cost) or minimum_gain_per_cost < 0:
        raise ValueError("minimum_gain_per_cost must be finite and nonnegative")
    if len({item.candidate_id for item in utilities}) != len(utilities):
        raise ValueError("candidate IDs must be unique")
    if not utilities:
        return None
    best = min(utilities, key=lambda item: (-item.reduction_per_cost, item.candidate_id))
    return best.candidate_id if best.reduction_per_cost > minimum_gain_per_cost else None
