"""Group-aware pathwise uncertainty diagnostics and source-only calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias, cast

import numpy as np

from .covariance import covariance_statistics

_CHI_SQUARED_3 = {
    0.50: 2.3659738843753377,
    0.80: 4.64162767608745,
    0.90: 6.251388631170325,
    0.95: 7.814727903251179,
}
IndependentUnit: TypeAlias = Literal["physical-object", "acquisition-session"]
_INDEPENDENT_UNITS = frozenset({"physical-object", "acquisition-session"})


def _validated_independent_unit(value: object) -> IndependentUnit:
    if type(value) is not str:
        raise TypeError("independent_unit must be a string")
    if value not in _INDEPENDENT_UNITS:
        raise ValueError(
            "independent_unit must be 'physical-object' or 'acquisition-session'"
        )
    return cast(IndependentUnit, value)


def _validated_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_group_id(value: object, *, index: int) -> str:
    if type(value) is not str:
        raise TypeError(f"group_ids[{index}] must be a string")
    if not value or value != value.strip():
        raise ValueError(f"group_ids[{index}] must be a nonempty trimmed string")
    if len(value.encode("utf-8")) > 128:
        raise ValueError(f"group_ids[{index}] must contain at most 128 UTF-8 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"group_ids[{index}] must not contain control characters")
    return cast(str, value)


def _group_assignment_sha256(group_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        group_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _validated_grouping(
    group_ids: Sequence[str],
    *,
    trajectory_count: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    str,
]:
    if isinstance(group_ids, (str, bytes)):
        raise TypeError("group_ids must be a sequence with one entry per trajectory")
    supplied = tuple(group_ids)
    if len(supplied) != trajectory_count:
        raise ValueError(
            "group_ids must contain exactly one entry per first-axis trajectory"
        )
    validated = tuple(
        _validated_group_id(value, index=index)
        for index, value in enumerate(supplied)
    )
    unique = tuple(sorted(set(validated)))
    members = tuple(
        tuple(index for index, group_id in enumerate(validated) if group_id == selected)
        for selected in unique
    )
    return validated, unique, members, _group_assignment_sha256(validated)


@dataclass(frozen=True, slots=True)
class PathwiseMaximumCalibrationV1:
    """Frozen split-conformal threshold over independent object/session groups."""

    requested_miscoverage: float
    independent_unit: IndependentUnit
    calibration_trajectory_group_ids: tuple[str, ...]
    calibration_group_assignment_sha256: str
    calibration_trajectory_count: int
    calibration_group_count: int
    order_statistic_rank: int
    finite_sample_coverage_level: float
    maximum_mahalanobis_squared_threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.requested_miscoverage, (bool, np.bool_)) or not isinstance(
            self.requested_miscoverage,
            Real,
        ):
            raise TypeError("requested_miscoverage must be a real scalar")
        if not isinstance(self.calibration_trajectory_group_ids, tuple):
            raise TypeError("calibration_trajectory_group_ids must be a tuple")
        miscoverage = float(self.requested_miscoverage)
        independent_unit = _validated_independent_unit(self.independent_unit)
        trajectory_count = _validated_integer(
            self.calibration_trajectory_count,
            name="calibration_trajectory_count",
        )
        group_count = _validated_integer(
            self.calibration_group_count,
            name="calibration_group_count",
        )
        rank = _validated_integer(
            self.order_statistic_rank,
            name="order_statistic_rank",
        )
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
        finite_level = float(self.finite_sample_coverage_level)
        threshold = float(self.maximum_mahalanobis_squared_threshold)
        if not math.isfinite(miscoverage) or not 0.0 < miscoverage < 1.0:
            raise ValueError("requested_miscoverage must lie in (0, 1)")
        grouping, unique, _, digest = _validated_grouping(
            self.calibration_trajectory_group_ids,
            trajectory_count=trajectory_count,
        )
        supplied_digest = _validated_sha256(
            self.calibration_group_assignment_sha256,
            name="calibration_group_assignment_sha256",
        )
        if supplied_digest != digest:
            raise ValueError("calibration group-assignment digest does not match group_ids")
        if group_count != len(unique):
            raise ValueError("calibration_group_count does not match distinct group_ids")
        if not 1 <= rank <= group_count:
            raise ValueError("calibration group count and order-statistic rank are inconsistent")
        if not math.isfinite(finite_level) or not 0.0 < finite_level < 1.0:
            raise ValueError("finite_sample_coverage_level must lie in (0, 1)")
        expected_level = rank / (group_count + 1)
        if not math.isclose(finite_level, expected_level, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                "finite_sample_coverage_level does not match rank/(group_count+1)"
            )
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("maximum Mahalanobis threshold must be finite and non-negative")
        object.__setattr__(self, "requested_miscoverage", miscoverage)
        object.__setattr__(self, "independent_unit", independent_unit)
        object.__setattr__(self, "calibration_trajectory_group_ids", grouping)
        object.__setattr__(self, "calibration_group_assignment_sha256", digest)
        object.__setattr__(self, "calibration_trajectory_count", trajectory_count)
        object.__setattr__(self, "calibration_group_count", group_count)
        object.__setattr__(self, "order_statistic_rank", rank)
        object.__setattr__(self, "finite_sample_coverage_level", finite_level)
        object.__setattr__(self, "maximum_mahalanobis_squared_threshold", threshold)

    @property
    def requested_coverage(self) -> float:
        return 1.0 - self.requested_miscoverage

    @property
    def calibration_group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.calibration_trajectory_group_ids)))

    @property
    def calibration_group_trajectory_counts(self) -> tuple[int, ...]:
        return tuple(
            self.calibration_trajectory_group_ids.count(group_id)
            for group_id in self.calibration_group_ids
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["requested_coverage"] = self.requested_coverage
        result["calibration_group_ids"] = self.calibration_group_ids
        result["calibration_group_trajectory_counts"] = (
            self.calibration_group_trajectory_counts
        )
        return result


@dataclass(frozen=True, slots=True)
class PathwiseUncertaintyDiagnostics:
    """Equal-group pathwise diagnostics with per-trajectory run descriptions."""

    independent_unit: IndependentUnit
    target_trajectory_group_ids: tuple[str, ...]
    target_group_assignment_sha256: str
    group_count: int
    trajectory_count: int
    evaluated_step_count: int
    mean_group_valid_step_fraction: float
    minimum_group_valid_step_fraction: float
    equal_group_marginal_coverage_50: float
    equal_group_marginal_coverage_80: float
    equal_group_marginal_coverage_90: float
    equal_group_marginal_coverage_95: float
    all_groups_inside_marginal_50_fraction: float
    all_groups_inside_marginal_80_fraction: float
    all_groups_inside_marginal_90_fraction: float
    all_groups_inside_marginal_95_fraction: float
    equal_group_independence_reference_all_steps_50_fraction: float
    equal_group_independence_reference_all_steps_80_fraction: float
    equal_group_independence_reference_all_steps_90_fraction: float
    equal_group_independence_reference_all_steps_95_fraction: float
    mean_group_max_mahalanobis_squared: float
    median_group_max_mahalanobis_squared: float
    p90_group_max_mahalanobis_squared: float
    p95_group_max_mahalanobis_squared: float
    mean_trajectory_longest_marginal_95_failure_run: float
    p90_trajectory_longest_marginal_95_failure_run: float
    maximum_trajectory_longest_marginal_95_failure_run: int
    mean_trajectory_longest_unsupported_run: float
    maximum_trajectory_longest_unsupported_run: int
    equal_group_gaussian_nll_per_dimension: float
    simultaneous_nominal_group_coverage: float | None
    simultaneous_group_maximum_threshold: float | None
    simultaneous_group_coverage: float | None
    simultaneous_group_coverage_shortfall: float | None

    def __post_init__(self) -> None:
        independent_unit = _validated_independent_unit(self.independent_unit)
        trajectory_count = _validated_integer(
            self.trajectory_count,
            name="trajectory_count",
        )
        group_count = _validated_integer(self.group_count, name="group_count")
        grouping, unique, _, digest = _validated_grouping(
            self.target_trajectory_group_ids,
            trajectory_count=trajectory_count,
        )
        supplied_digest = _validated_sha256(
            self.target_group_assignment_sha256,
            name="target_group_assignment_sha256",
        )
        if supplied_digest != digest:
            raise ValueError("target group-assignment digest does not match group_ids")
        if group_count != len(unique):
            raise ValueError("group_count does not match distinct target group_ids")
        simultaneous = (
            self.simultaneous_nominal_group_coverage,
            self.simultaneous_group_maximum_threshold,
            self.simultaneous_group_coverage,
            self.simultaneous_group_coverage_shortfall,
        )
        if any(value is None for value in simultaneous) and not all(
            value is None for value in simultaneous
        ):
            raise ValueError("simultaneous group diagnostics must be all present or all absent")
        object.__setattr__(self, "independent_unit", independent_unit)
        object.__setattr__(self, "target_trajectory_group_ids", grouping)
        object.__setattr__(self, "target_group_assignment_sha256", digest)
        object.__setattr__(self, "group_count", group_count)
        object.__setattr__(self, "trajectory_count", trajectory_count)

    @property
    def target_group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.target_trajectory_group_ids)))

    @property
    def target_group_trajectory_counts(self) -> tuple[int, ...]:
        return tuple(
            self.target_trajectory_group_ids.count(group_id)
            for group_id in self.target_group_ids
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["target_group_ids"] = self.target_group_ids
        result["target_group_trajectory_counts"] = self.target_group_trajectory_counts
        return result


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


def _group_maxima(
    trajectory_maxima: np.ndarray,
    members: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    return np.asarray(
        [float(np.max(trajectory_maxima[list(indices)])) for indices in members],
        dtype=np.float64,
    )


def fit_pathwise_maximum_calibration(
    residual_xyz_m: object,
    covariance_m2: object,
    *,
    group_ids: Sequence[str],
    independent_unit: IndependentUnit,
    valid_mask: object | None = None,
    miscoverage: float = 0.05,
) -> PathwiseMaximumCalibrationV1:
    """Fit a finite-sample maximum threshold over independent object/session groups.

    The first axis still indexes trajectories, but ``group_ids`` assigns every
    trajectory to one complete physical object or acquisition session. The
    conformal statistic is the maximum over all admitted trajectories and valid
    steps in one group. The number of groups, never the number of tracks, sets the
    finite-sample rank.
    """

    if isinstance(miscoverage, (bool, np.bool_)) or not isinstance(miscoverage, Real):
        raise TypeError("miscoverage must be a real scalar")
    alpha = float(miscoverage)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie in (0, 1)")
    unit = _validated_independent_unit(independent_unit)
    residual, covariance, valid = _validated_inputs(
        residual_xyz_m,
        covariance_m2,
        valid_mask,
    )
    grouping, unique, members, digest = _validated_grouping(
        group_ids,
        trajectory_count=len(residual),
    )
    mahalanobis, _ = _mahalanobis_and_nll(residual, covariance, valid)
    maxima = _group_maxima(_trajectory_maxima(mahalanobis, valid), members)
    count = int(maxima.size)
    rank = int(math.ceil((count + 1) * (1.0 - alpha)))
    if rank > count:
        minimum_count = int(math.ceil((1.0 - alpha) / alpha))
        raise ValueError(
            "finite pathwise calibration is unavailable at the requested "
            f"miscoverage with {count} independent groups; at least "
            f"{minimum_count} independent groups are required"
        )
    threshold = float(np.partition(maxima, rank - 1)[rank - 1])
    return PathwiseMaximumCalibrationV1(
        requested_miscoverage=alpha,
        independent_unit=unit,
        calibration_trajectory_group_ids=grouping,
        calibration_group_assignment_sha256=digest,
        calibration_trajectory_count=len(grouping),
        calibration_group_count=len(unique),
        order_statistic_rank=rank,
        finite_sample_coverage_level=rank / (count + 1),
        maximum_mahalanobis_squared_threshold=threshold,
    )


def pathwise_uncertainty_diagnostics(
    residual_xyz_m: object,
    covariance_m2: object,
    *,
    group_ids: Sequence[str],
    independent_unit: IndependentUnit,
    valid_mask: object | None = None,
    calibration: PathwiseMaximumCalibrationV1 | None = None,
) -> PathwiseUncertaintyDiagnostics:
    """Evaluate pathwise uncertainty with equal object/session-group weighting.

    Per-trajectory failure and support runs remain descriptive. Coverage, maximum
    statistics, Gaussian score, and any calibrated simultaneous claim are
    aggregated over complete independent groups.
    """

    if calibration is not None and not isinstance(calibration, PathwiseMaximumCalibrationV1):
        raise TypeError("calibration must be a PathwiseMaximumCalibrationV1")
    unit = _validated_independent_unit(independent_unit)
    residual, covariance, valid = _validated_inputs(
        residual_xyz_m,
        covariance_m2,
        valid_mask,
    )
    grouping, unique, members, digest = _validated_grouping(
        group_ids,
        trajectory_count=len(residual),
    )
    if calibration is not None:
        if calibration.independent_unit != unit:
            raise ValueError("calibration and target independent_unit must match")
        overlap = sorted(set(calibration.calibration_group_ids) & set(unique))
        if overlap:
            raise ValueError(
                "calibration and target group_ids must be disjoint; "
                f"overlap={overlap}"
            )

    mahalanobis, gaussian_nll = _mahalanobis_and_nll(residual, covariance, valid)
    trajectory_count, step_count = valid.shape
    trajectory_valid_counts = np.count_nonzero(valid, axis=1)
    evaluated_step_count = int(np.sum(trajectory_valid_counts))
    group_valid_counts = np.asarray(
        [int(np.count_nonzero(valid[list(indices)])) for indices in members],
        dtype=np.int64,
    )
    group_possible_counts = np.asarray(
        [len(indices) * step_count for indices in members],
        dtype=np.int64,
    )
    trajectory_maxima = _trajectory_maxima(mahalanobis, valid)
    group_maxima = _group_maxima(trajectory_maxima, members)

    marginal_coverages: dict[float, float] = {}
    all_group_coverages: dict[float, float] = {}
    independence_references: dict[float, float] = {}
    for level, threshold in _CHI_SQUARED_3.items():
        inside = valid & _at_most(mahalanobis, threshold)
        per_group = np.asarray(
            [
                np.count_nonzero(inside[list(indices)]) / group_valid_counts[index]
                for index, indices in enumerate(members)
            ],
            dtype=np.float64,
        )
        marginal_coverages[level] = float(np.mean(per_group))
        all_group_coverages[level] = float(
            np.mean(
                [
                    bool(
                        np.all(
                            _at_most(
                                mahalanobis[list(indices)][valid[list(indices)]],
                                threshold,
                            )
                        )
                    )
                    for indices in members
                ]
            )
        )
        independence_references[level] = float(
            np.mean(level**group_valid_counts)
        )

    marginal_95_failures = valid & ~_at_most(mahalanobis, _CHI_SQUARED_3[0.95])
    failure_runs = np.asarray(
        [_longest_true_run(row) for row in marginal_95_failures],
        dtype=np.int64,
    )
    unsupported_runs = np.asarray(
        [_longest_true_run(~row) for row in valid],
        dtype=np.int64,
    )
    group_nll_per_dimension = np.asarray(
        [
            float(
                np.sum(gaussian_nll[list(indices)][valid[list(indices)]])
                / (3 * group_valid_counts[index])
            )
            for index, indices in enumerate(members)
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
        simultaneous_coverage = float(
            np.mean(_at_most(group_maxima, simultaneous_threshold))
        )
        simultaneous_shortfall = max(
            0.0,
            simultaneous_nominal - simultaneous_coverage,
        )

    return PathwiseUncertaintyDiagnostics(
        independent_unit=unit,
        target_trajectory_group_ids=grouping,
        target_group_assignment_sha256=digest,
        group_count=len(unique),
        trajectory_count=trajectory_count,
        evaluated_step_count=evaluated_step_count,
        mean_group_valid_step_fraction=float(
            np.mean(group_valid_counts / group_possible_counts)
        ),
        minimum_group_valid_step_fraction=float(
            np.min(group_valid_counts / group_possible_counts)
        ),
        equal_group_marginal_coverage_50=marginal_coverages[0.50],
        equal_group_marginal_coverage_80=marginal_coverages[0.80],
        equal_group_marginal_coverage_90=marginal_coverages[0.90],
        equal_group_marginal_coverage_95=marginal_coverages[0.95],
        all_groups_inside_marginal_50_fraction=all_group_coverages[0.50],
        all_groups_inside_marginal_80_fraction=all_group_coverages[0.80],
        all_groups_inside_marginal_90_fraction=all_group_coverages[0.90],
        all_groups_inside_marginal_95_fraction=all_group_coverages[0.95],
        equal_group_independence_reference_all_steps_50_fraction=(
            independence_references[0.50]
        ),
        equal_group_independence_reference_all_steps_80_fraction=(
            independence_references[0.80]
        ),
        equal_group_independence_reference_all_steps_90_fraction=(
            independence_references[0.90]
        ),
        equal_group_independence_reference_all_steps_95_fraction=(
            independence_references[0.95]
        ),
        mean_group_max_mahalanobis_squared=float(np.mean(group_maxima)),
        median_group_max_mahalanobis_squared=float(np.median(group_maxima)),
        p90_group_max_mahalanobis_squared=float(np.quantile(group_maxima, 0.90)),
        p95_group_max_mahalanobis_squared=float(np.quantile(group_maxima, 0.95)),
        mean_trajectory_longest_marginal_95_failure_run=float(np.mean(failure_runs)),
        p90_trajectory_longest_marginal_95_failure_run=float(
            np.quantile(failure_runs, 0.90)
        ),
        maximum_trajectory_longest_marginal_95_failure_run=int(
            np.max(failure_runs)
        ),
        mean_trajectory_longest_unsupported_run=float(np.mean(unsupported_runs)),
        maximum_trajectory_longest_unsupported_run=int(np.max(unsupported_runs)),
        equal_group_gaussian_nll_per_dimension=float(
            np.mean(group_nll_per_dimension)
        ),
        simultaneous_nominal_group_coverage=simultaneous_nominal,
        simultaneous_group_maximum_threshold=simultaneous_threshold,
        simultaneous_group_coverage=simultaneous_coverage,
        simultaneous_group_coverage_shortfall=simultaneous_shortfall,
    )


__all__ = [
    "PathwiseMaximumCalibrationV1",
    "PathwiseUncertaintyDiagnostics",
    "fit_pathwise_maximum_calibration",
    "pathwise_uncertainty_diagnostics",
]
