"""Trajectory-level uncertainty diagnostics and source-only maximum calibration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from .covariance import covariance_statistics

_CHI_SQUARED_3 = {
    0.50: 2.3659738843753377,
    0.80: 4.64162767608745,
    0.90: 6.251388631170325,
    0.95: 7.814727903251179,
}


@dataclass(frozen=True, slots=True)
class PathwiseMaximumCalibrationV1:
    """Frozen split-conformal threshold for a trajectory maximum statistic."""

    requested_miscoverage: float
    calibration_trajectory_count: int
    order_statistic_rank: int
    finite_sample_coverage_level: float
    maximum_mahalanobis_squared_threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.requested_miscoverage, (bool, np.bool_)) or not isinstance(
            self.requested_miscoverage,
            Real,
        ):
            raise TypeError("requested_miscoverage must be a real scalar")
        if isinstance(self.calibration_trajectory_count, (bool, np.bool_)) or not isinstance(
            self.calibration_trajectory_count,
            Integral,
        ):
            raise TypeError("calibration_trajectory_count must be an integer")
        if isinstance(self.order_statistic_rank, (bool, np.bool_)) or not isinstance(
            self.order_statistic_rank,
            Integral,
        ):
            raise TypeError("order_statistic_rank must be an integer")
        if isinstance(self.finite_sample_coverage_level, (bool, np.bool_)) or not isinstance(
            self.finite_sample_coverage_level,
            Real,
        ):
            raise TypeError("finite_sample_coverage_level must be a real scalar")
        if isinstance(
            self.maximum_mahalanobis_squared_threshold,
            (bool, np.bool_),
        ) or not isinstance(self.maximum_mahalanobis_squared_threshold, Real):
            raise TypeError("maximum Mahalanobis threshold must be a real scalar")
        miscoverage = float(self.requested_miscoverage)
        count = int(self.calibration_trajectory_count)
        rank = int(self.order_statistic_rank)
        finite_level = float(self.finite_sample_coverage_level)
        threshold = float(self.maximum_mahalanobis_squared_threshold)
        if not math.isfinite(miscoverage) or not 0.0 < miscoverage < 1.0:
            raise ValueError("requested_miscoverage must lie in (0, 1)")
        if count < 1 or not 1 <= rank <= count:
            raise ValueError("calibration count and order-statistic rank are inconsistent")
        if not math.isfinite(finite_level) or not 0.0 < finite_level < 1.0:
            raise ValueError("finite_sample_coverage_level must lie in (0, 1)")
        expected_level = rank / (count + 1)
        if not math.isclose(finite_level, expected_level, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("finite_sample_coverage_level does not match rank/(n+1)")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("maximum Mahalanobis threshold must be finite and non-negative")
        object.__setattr__(self, "requested_miscoverage", miscoverage)
        object.__setattr__(self, "calibration_trajectory_count", count)
        object.__setattr__(self, "order_statistic_rank", rank)
        object.__setattr__(self, "finite_sample_coverage_level", finite_level)
        object.__setattr__(self, "maximum_mahalanobis_squared_threshold", threshold)

    @property
    def requested_coverage(self) -> float:
        return 1.0 - self.requested_miscoverage

    def to_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        result["requested_coverage"] = self.requested_coverage
        return result


@dataclass(frozen=True, slots=True)
class PathwiseUncertaintyDiagnostics:
    """Equal-trajectory diagnostics that expose clustered calibration failures."""

    trajectory_count: int
    evaluated_step_count: int
    mean_valid_step_fraction: float
    minimum_valid_step_fraction: float
    marginal_coverage_50: float
    marginal_coverage_80: float
    marginal_coverage_90: float
    marginal_coverage_95: float
    all_steps_inside_marginal_50_fraction: float
    all_steps_inside_marginal_80_fraction: float
    all_steps_inside_marginal_90_fraction: float
    all_steps_inside_marginal_95_fraction: float
    independence_reference_all_steps_50_fraction: float
    independence_reference_all_steps_80_fraction: float
    independence_reference_all_steps_90_fraction: float
    independence_reference_all_steps_95_fraction: float
    mean_max_mahalanobis_squared: float
    median_max_mahalanobis_squared: float
    p90_max_mahalanobis_squared: float
    p95_max_mahalanobis_squared: float
    mean_longest_marginal_95_failure_run: float
    p90_longest_marginal_95_failure_run: float
    maximum_longest_marginal_95_failure_run: int
    mean_longest_unsupported_run: float
    maximum_longest_unsupported_run: int
    equal_trajectory_gaussian_nll_per_dimension: float
    simultaneous_nominal_coverage: float | None
    simultaneous_maximum_threshold: float | None
    simultaneous_coverage: float | None
    simultaneous_coverage_shortfall: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _validated_inputs(
    residual_xyz_m: object,
    covariance_m2: object,
    valid_mask: object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(residual_xyz_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    single_trajectory = residual.ndim == 2
    if single_trajectory:
        if residual.shape[1:] != (3,):
            raise ValueError("residual_xyz_m must have shape (P, T, 3) or (T, 3)")
        residual = residual[None]
        if covariance.shape == (residual.shape[1], 3, 3):
            covariance = covariance[None]
    if residual.ndim != 3 or residual.shape[0] < 1 or residual.shape[1] < 1:
        raise ValueError("residual_xyz_m must have nonempty shape (P, T, 3)")
    if residual.shape[2] != 3:
        raise ValueError("residual_xyz_m must have shape (P, T, 3)")
    if covariance.shape != (*residual.shape[:2], 3, 3):
        raise ValueError("covariance_m2 must have shape (P, T, 3, 3)")
    if valid_mask is None:
        valid = np.ones(residual.shape[:2], dtype=bool)
    else:
        raw_valid = np.asarray(valid_mask)
        if raw_valid.dtype != np.dtype(bool):
            raise TypeError("valid_mask must contain genuine booleans")
        valid = np.asarray(raw_valid, dtype=bool)
        if single_trajectory and valid.shape == (residual.shape[1],):
            valid = valid[None]
        if valid.shape != residual.shape[:2]:
            raise ValueError("valid_mask must have shape (P, T)")
    if np.any(np.count_nonzero(valid, axis=1) == 0):
        raise ValueError("every trajectory must contain at least one valid step")
    if not np.all(np.isfinite(residual[valid])):
        raise ValueError("valid residual_xyz_m entries must be finite")
    if not np.all(np.isfinite(covariance[valid])):
        raise ValueError("valid covariance_m2 entries must be finite")
    return residual, covariance, valid


def _mahalanobis_and_nll(
    residual: np.ndarray,
    covariance: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_covariance = covariance[valid]
    symmetric, inverse, log_determinant = covariance_statistics(
        selected_covariance,
        name="pathwise covariance",
    )
    selected_residual = residual[valid]
    mahalanobis = np.einsum(
        "ni,nij,nj->n",
        selected_residual,
        inverse,
        selected_residual,
        optimize=True,
    )
    gaussian_nll = 0.5 * (
        3.0 * math.log(2.0 * math.pi) + log_determinant + mahalanobis
    )
    if not np.all(np.isfinite(symmetric)):
        raise ValueError("pathwise covariance normalization failed")
    mahalanobis_matrix = np.full(valid.shape, np.nan, dtype=np.float64)
    nll_matrix = np.full(valid.shape, np.nan, dtype=np.float64)
    mahalanobis_matrix[valid] = mahalanobis
    nll_matrix[valid] = gaussian_nll
    return mahalanobis_matrix, nll_matrix


def _at_most(value: np.ndarray, threshold: float) -> np.ndarray:
    tolerance = 1e-12 * max(abs(float(threshold)), 1.0)
    return np.asarray(value, dtype=np.float64) <= float(threshold) + tolerance


def _longest_true_run(value: np.ndarray) -> int:
    longest = 0
    current = 0
    for selected in np.asarray(value, dtype=bool):
        if selected:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _trajectory_maxima(mahalanobis: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.max(mahalanobis[index][valid[index]])) for index in range(len(valid))],
        dtype=np.float64,
    )


def fit_pathwise_maximum_calibration(
    residual_xyz_m: object,
    covariance_m2: object,
    *,
    valid_mask: object | None = None,
    miscoverage: float = 0.05,
) -> PathwiseMaximumCalibrationV1:
    """Fit a finite-sample trajectory-maximum threshold on calibration units only.

    Each first-axis trajectory is one exchangeability unit.  The returned object
    must be frozen before target residuals or outcomes are opened.
    """

    if isinstance(miscoverage, (bool, np.bool_)) or not isinstance(miscoverage, Real):
        raise TypeError("miscoverage must be a real scalar")
    alpha = float(miscoverage)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie in (0, 1)")
    residual, covariance, valid = _validated_inputs(
        residual_xyz_m,
        covariance_m2,
        valid_mask,
    )
    mahalanobis, _ = _mahalanobis_and_nll(residual, covariance, valid)
    maxima = _trajectory_maxima(mahalanobis, valid)
    count = int(maxima.size)
    rank = int(math.ceil((count + 1) * (1.0 - alpha)))
    if rank > count:
        minimum_count = int(math.ceil((1.0 - alpha) / alpha))
        raise ValueError(
            "finite pathwise calibration is unavailable at the requested "
            f"miscoverage with {count} trajectories; at least {minimum_count} are required"
        )
    threshold = float(np.partition(maxima, rank - 1)[rank - 1])
    return PathwiseMaximumCalibrationV1(
        requested_miscoverage=alpha,
        calibration_trajectory_count=count,
        order_statistic_rank=rank,
        finite_sample_coverage_level=rank / (count + 1),
        maximum_mahalanobis_squared_threshold=threshold,
    )


def pathwise_uncertainty_diagnostics(
    residual_xyz_m: object,
    covariance_m2: object,
    *,
    valid_mask: object | None = None,
    calibration: PathwiseMaximumCalibrationV1 | None = None,
) -> PathwiseUncertaintyDiagnostics:
    """Evaluate marginal and trajectory-level calibration without pooling tracks.

    Fractions named ``all_steps_inside_marginal_*`` use pointwise chi-squared
    ellipsoids and are diagnostic, not simultaneous confidence statements.  A
    simultaneous coverage estimate is emitted only when an independently fitted
    :class:`PathwiseMaximumCalibrationV1` is supplied.
    """

    if calibration is not None and not isinstance(calibration, PathwiseMaximumCalibrationV1):
        raise TypeError("calibration must be a PathwiseMaximumCalibrationV1")
    residual, covariance, valid = _validated_inputs(
        residual_xyz_m,
        covariance_m2,
        valid_mask,
    )
    mahalanobis, gaussian_nll = _mahalanobis_and_nll(residual, covariance, valid)
    trajectory_count, step_count = valid.shape
    valid_counts = np.count_nonzero(valid, axis=1)
    evaluated_step_count = int(np.sum(valid_counts))
    maxima = _trajectory_maxima(mahalanobis, valid)

    marginal_coverages: dict[float, float] = {}
    all_step_coverages: dict[float, float] = {}
    independence_references: dict[float, float] = {}
    for level, threshold in _CHI_SQUARED_3.items():
        inside = valid & _at_most(mahalanobis, threshold)
        marginal_coverages[level] = float(np.count_nonzero(inside) / evaluated_step_count)
        all_step_coverages[level] = float(
            np.mean(
                [
                    bool(np.all(_at_most(mahalanobis[index][valid[index]], threshold)))
                    for index in range(trajectory_count)
                ]
            )
        )
        independence_references[level] = float(np.mean(level**valid_counts))

    marginal_95_failures = valid & ~_at_most(mahalanobis, _CHI_SQUARED_3[0.95])
    failure_runs = np.asarray(
        [_longest_true_run(row) for row in marginal_95_failures],
        dtype=np.int64,
    )
    unsupported_runs = np.asarray(
        [_longest_true_run(~row) for row in valid],
        dtype=np.int64,
    )
    trajectory_nll_per_dimension = np.asarray(
        [
            float(np.sum(gaussian_nll[index][valid[index]]) / (3 * valid_counts[index]))
            for index in range(trajectory_count)
        ],
        dtype=np.float64,
    )

    simultaneous_nominal: float | None = None
    simultaneous_threshold: float | None = None
    simultaneous_coverage: float | None = None
    simultaneous_shortfall: float | None = None
    if calibration is not None:
        simultaneous_nominal = calibration.requested_coverage
        simultaneous_threshold = calibration.maximum_mahalanobis_squared_threshold
        simultaneous_coverage = float(np.mean(_at_most(maxima, simultaneous_threshold)))
        simultaneous_shortfall = max(0.0, simultaneous_nominal - simultaneous_coverage)

    return PathwiseUncertaintyDiagnostics(
        trajectory_count=trajectory_count,
        evaluated_step_count=evaluated_step_count,
        mean_valid_step_fraction=float(np.mean(valid_counts / step_count)),
        minimum_valid_step_fraction=float(np.min(valid_counts / step_count)),
        marginal_coverage_50=marginal_coverages[0.50],
        marginal_coverage_80=marginal_coverages[0.80],
        marginal_coverage_90=marginal_coverages[0.90],
        marginal_coverage_95=marginal_coverages[0.95],
        all_steps_inside_marginal_50_fraction=all_step_coverages[0.50],
        all_steps_inside_marginal_80_fraction=all_step_coverages[0.80],
        all_steps_inside_marginal_90_fraction=all_step_coverages[0.90],
        all_steps_inside_marginal_95_fraction=all_step_coverages[0.95],
        independence_reference_all_steps_50_fraction=independence_references[0.50],
        independence_reference_all_steps_80_fraction=independence_references[0.80],
        independence_reference_all_steps_90_fraction=independence_references[0.90],
        independence_reference_all_steps_95_fraction=independence_references[0.95],
        mean_max_mahalanobis_squared=float(np.mean(maxima)),
        median_max_mahalanobis_squared=float(np.median(maxima)),
        p90_max_mahalanobis_squared=float(np.quantile(maxima, 0.90)),
        p95_max_mahalanobis_squared=float(np.quantile(maxima, 0.95)),
        mean_longest_marginal_95_failure_run=float(np.mean(failure_runs)),
        p90_longest_marginal_95_failure_run=float(np.quantile(failure_runs, 0.90)),
        maximum_longest_marginal_95_failure_run=int(np.max(failure_runs)),
        mean_longest_unsupported_run=float(np.mean(unsupported_runs)),
        maximum_longest_unsupported_run=int(np.max(unsupported_runs)),
        equal_trajectory_gaussian_nll_per_dimension=float(
            np.mean(trajectory_nll_per_dimension)
        ),
        simultaneous_nominal_coverage=simultaneous_nominal,
        simultaneous_maximum_threshold=simultaneous_threshold,
        simultaneous_coverage=simultaneous_coverage,
        simultaneous_coverage_shortfall=simultaneous_shortfall,
    )


__all__ = [
    "PathwiseMaximumCalibrationV1",
    "PathwiseUncertaintyDiagnostics",
    "fit_pathwise_maximum_calibration",
    "pathwise_uncertainty_diagnostics",
]
