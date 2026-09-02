"""Multi-axis benchmark for probabilistic 4-D information contracts.

The benchmark evaluates sealed provider submissions along separate axes:
forecast accuracy, Gaussian calibration, shared dependence, query/gauge
admissibility, exact fallback, finite-action regret, and covariance payload
efficiency.  It intentionally does not collapse these properties into one
leaderboard scalar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text
from .joint_covariance_metrics import evaluate_joint_gaussian_group

SUITE_SCHEMA: Final = "prob4d.information-contract-suite"
SUITE_VERSION: Final = 1
RESULT_SCHEMA: Final = "prob4d.information-contract-benchmark-result"
RESULT_VERSION: Final = 1
CLAIM_BOUNDARY: Final = (
    "This benchmark scores the supplied sealed predictions, uncertainty, "
    "admission decisions, and finite-action records on the declared cases. "
    "It does not establish provider competence outside the suite, physical "
    "correctness of a declared gauge or quotient, target safety, causal "
    "identification, or state of the art."
)
LEADERBOARD_POLICY: Final = (
    "No single aggregate rank is defined. Compare methods on the declared "
    "task axes and statistical-unit aggregation; accuracy cannot compensate "
    "for a failed information contract."
)
_ALLOWED_TASKS: Final = frozenset(
    {
        "forecast",
        "calibration",
        "dependence",
        "query",
        "gauge",
        "fallback",
        "decision",
        "communication",
    }
)
_ALLOWED_ARRAYS: Final = frozenset(
    {
        "truth_xyz_m",
        "prediction_mean_xyz_m",
        "conditional_covariance_m2",
        "shared_factor_m",
        "fallback_mean_xyz_m",
        "fallback_conditional_covariance_m2",
        "fallback_shared_factor_m",
        "query_matrix",
        "nullspace_basis",
        "query_admitted",
        "reported_query_mean",
        "reported_query_variance",
        "decision_loss_by_hypothesis",
        "hypothesis_prior",
        "quotient_class",
        "quotient_mass",
        "reported_worst_case_regret",
        "selected_action",
        "fallback_action",
        "decision_admitted",
        "regret_tolerance",
        "realized_action_loss",
    }
)
FloatArray = NDArray[np.float64]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read benchmark payload {path}") from error
    return digest.hexdigest()


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            relation = "at most" if maximum_inclusive else "less than"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _relative_payload(root: Path, value: object, *, name: str) -> Path:
    raw = _string(value, name=name)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a relative path without '..'")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes the suite directory") from error
    if not resolved.is_file():
        raise ValueError(f"{name} does not identify a file: {relative}")
    return resolved


def _hex_sha256(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _float_array(value: object, *, name: str, ndim: int | None = None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar_integer(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer scalar")
    return int(array)


def _scalar_boolean(value: object, *, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a Boolean scalar")
    return bool(array)


def _validate_covariance(
    local: object,
    factor: object,
    *,
    sample_count: int,
    prefix: str,
) -> tuple[FloatArray, FloatArray]:
    covariance = _float_array(local, name=f"{prefix}_conditional_covariance_m2", ndim=3)
    shared = _float_array(factor, name=f"{prefix}_shared_factor_m", ndim=3)
    if covariance.shape != (sample_count, 3, 3):
        raise ValueError(
            f"{prefix}_conditional_covariance_m2 must have shape "
            f"({sample_count}, 3, 3)"
        )
    if shared.shape[:2] != (sample_count, 3):
        raise ValueError(
            f"{prefix}_shared_factor_m must have shape ({sample_count}, 3, R)"
        )
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(-2, -1)), 1.0)
    asymmetry = np.max(
        np.abs(covariance - np.swapaxes(covariance, -1, -2)),
        axis=(-2, -1),
    )
    if np.any(asymmetry > 1e-12 + 1e-10 * scale):
        raise ValueError(f"{prefix}_conditional_covariance_m2 must be symmetric")
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"{prefix}_conditional_covariance_m2 must be positive definite"
        ) from error
    return symmetric, shared


def _required_arrays(tasks: frozenset[str]) -> set[str]:
    required: set[str] = set()
    if tasks.intersection(
        {"forecast", "calibration", "dependence", "query", "gauge", "fallback", "communication"}
    ):
        required.update({"truth_xyz_m", "prediction_mean_xyz_m"})
    if tasks.intersection(
        {"calibration", "dependence", "query", "gauge", "fallback", "communication"}
    ):
        required.update({"conditional_covariance_m2", "shared_factor_m"})
    if tasks.intersection({"query", "gauge", "fallback"}):
        required.add("query_matrix")
    if "gauge" in tasks or "fallback" in tasks:
        required.update({"nullspace_basis", "query_admitted"})
    if "fallback" in tasks:
        required.update(
            {
                "fallback_mean_xyz_m",
                "fallback_conditional_covariance_m2",
                "fallback_shared_factor_m",
                "reported_query_mean",
                "reported_query_variance",
            }
        )
    if "decision" in tasks:
        required.update(
            {
                "decision_loss_by_hypothesis",
                "hypothesis_prior",
                "quotient_class",
                "quotient_mass",
                "reported_worst_case_regret",
                "selected_action",
                "fallback_action",
                "decision_admitted",
                "regret_tolerance",
                "realized_action_loss",
            }
        )
    return required


def _load_payload(path: Path, *, tasks: frozenset[str]) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            unknown = names.difference(_ALLOWED_ARRAYS)
            if unknown:
                raise ValueError(f"payload contains unregistered arrays: {sorted(unknown)}")
            missing = _required_arrays(tasks).difference(names)
            if missing:
                raise ValueError(f"payload omits required arrays: {sorted(missing)}")
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("payload "):
            raise
        raise ValueError(f"cannot load benchmark payload {path}") from error


def _marginal_variance(local: FloatArray, factor: FloatArray) -> FloatArray:
    return np.diagonal(local, axis1=-2, axis2=-1) + np.sum(np.square(factor), axis=2)


def _coverage(
    residual: FloatArray,
    variance: FloatArray,
    probability: float,
) -> tuple[float, float]:
    z_value = NormalDist().inv_cdf(0.5 + 0.5 * probability)
    standard_deviation = np.sqrt(variance)
    covered = np.abs(residual) <= z_value * standard_deviation
    width = 2.0 * z_value * standard_deviation
    return float(np.mean(covered)), float(np.mean(width))


def _query_moments(
    mean_xyz_m: FloatArray,
    local_covariance_m2: FloatArray,
    shared_factor_m: FloatArray,
    query_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    sample_count = mean_xyz_m.shape[0]
    query_count = query_matrix.shape[0]
    query_blocks = query_matrix.reshape(query_count, sample_count, 3)
    mean = query_matrix @ mean_xyz_m.reshape(-1)
    local_variance = np.einsum(
        "qni,nij,qnj->q",
        query_blocks,
        local_covariance_m2,
        query_blocks,
    )
    shared_flat = shared_factor_m.reshape(sample_count * 3, shared_factor_m.shape[2])
    projected = query_matrix @ shared_flat
    variance = local_variance + np.sum(np.square(projected), axis=1)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("query variances must be finite and positive")
    return mean, variance


def _forecast_metrics(
    truth: FloatArray,
    prediction: FloatArray,
) -> dict[str, Any]:
    residual = truth - prediction
    distances = np.linalg.norm(residual, axis=1)
    return {
        "point_count": int(truth.shape[0]),
        "rmse_m": float(np.sqrt(np.mean(np.square(residual)))),
        "point_rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "mean_point_error_m": float(np.mean(distances)),
    }


def _calibration_metrics(
    truth: FloatArray,
    prediction: FloatArray,
    local: FloatArray,
    factor: FloatArray,
    *,
    coverage_probability: float,
    relative_rank_tolerance: float,
) -> dict[str, Any]:
    residual = truth - prediction
    joint = evaluate_joint_gaussian_group(
        residual,
        local,
        factor,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    marginal_coverage, mean_width = _coverage(
        residual,
        _marginal_variance(local, factor),
        coverage_probability,
    )
    return {
        **joint,
        "marginal_coverage_probability": coverage_probability,
        "marginal_coverage": marginal_coverage,
        "mean_marginal_interval_width_m": mean_width,
        "absolute_normalized_nees_error": abs(float(joint["normalized_nees"]) - 1.0),
    }


def _dependence_metrics(
    truth: FloatArray,
    prediction: FloatArray,
    local: FloatArray,
    factor: FloatArray,
    *,
    relative_rank_tolerance: float,
) -> dict[str, Any]:
    residual = truth - prediction
    full = evaluate_joint_gaussian_group(
        residual,
        local,
        factor,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    total_variance = _marginal_variance(local, factor)
    diagonal = np.zeros_like(local)
    diagonal[:, np.arange(3), np.arange(3)] = total_variance
    empty_factor = np.empty((truth.shape[0], 3, 0), dtype=np.float64)
    matched = evaluate_joint_gaussian_group(
        residual,
        diagonal,
        empty_factor,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return {
        "comparison": "full conditional-plus-low-rank versus marginal-matched diagonal",
        "full_gaussian_nll_per_dimension": full["gaussian_nll_per_dimension"],
        "marginal_matched_diagonal_gaussian_nll_per_dimension": matched[
            "gaussian_nll_per_dimension"
        ],
        "full_nll_gain_per_dimension": float(
            matched["gaussian_nll_per_dimension"] - full["gaussian_nll_per_dimension"]
        ),
        "full_normalized_nees": full["normalized_nees"],
        "marginal_matched_diagonal_normalized_nees": matched["normalized_nees"],
        "full_nees_error_gain": float(
            abs(float(matched["normalized_nees"]) - 1.0)
            - abs(float(full["normalized_nees"]) - 1.0)
        ),
        "effective_shared_rank": full["effective_shared_rank"],
    }


def _query_metrics(
    truth: FloatArray,
    prediction: FloatArray,
    local: FloatArray,
    factor: FloatArray,
    query_matrix: FloatArray,
    *,
    coverage_probability: float,
) -> tuple[dict[str, Any], FloatArray, FloatArray, FloatArray]:
    truth_query = query_matrix @ truth.reshape(-1)
    predicted_query, variance = _query_moments(
        prediction, local, factor, query_matrix
    )
    residual = truth_query - predicted_query
    nll = 0.5 * (
        np.log(2.0 * math.pi * variance) + np.square(residual) / variance
    )
    coverage, mean_width = _coverage(
        residual,
        variance,
        coverage_probability,
    )
    return (
        {
            "query_count": int(query_matrix.shape[0]),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "gaussian_nll_mean": float(np.mean(nll)),
            "normalized_nees_mean": float(np.mean(np.square(residual) / variance)),
            "marginal_coverage_probability": coverage_probability,
            "marginal_coverage": coverage,
            "mean_interval_width": mean_width,
        },
        truth_query,
        predicted_query,
        variance,
    )


def _gauge_metrics(
    query_matrix: FloatArray,
    nullspace_basis: FloatArray,
    submitted_admission: NDArray[np.bool_],
    *,
    tolerance: float,
) -> tuple[dict[str, Any], NDArray[np.bool_]]:
    dimension = query_matrix.shape[1]
    basis = _float_array(nullspace_basis, name="nullspace_basis", ndim=2)
    if basis.shape[0] != dimension:
        raise ValueError(
            f"nullspace_basis must have shape ({dimension}, G)"
        )
    if basis.shape[1]:
        left, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
        threshold = (
            np.finfo(np.float64).eps
            * max(basis.shape)
            * float(singular_values[0])
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        if rank != basis.shape[1]:
            raise ValueError("nullspace_basis must have full column rank")
        orthonormal = left[:, :rank]
        numerator = np.linalg.norm(query_matrix @ orthonormal, axis=1)
    else:
        rank = 0
        numerator = np.zeros(query_matrix.shape[0], dtype=np.float64)
    denominator = np.linalg.norm(query_matrix, axis=1)
    sensitivity = numerator / denominator
    expected = sensitivity <= tolerance
    admitted = np.asarray(submitted_admission)
    if admitted.shape != (query_matrix.shape[0],) or admitted.dtype.kind != "b":
        raise ValueError("query_admitted must be a Boolean vector with one entry per query")
    false_accept = admitted & ~expected
    false_reject = ~admitted & expected
    return (
        {
            "nullspace_dimension": rank,
            "gauge_sensitivity_tolerance": tolerance,
            "sensitivity_fraction": sensitivity.tolist(),
            "expected_admitted": expected.tolist(),
            "submitted_admitted": admitted.tolist(),
            "correct_accept_count": int(np.count_nonzero(admitted & expected)),
            "correct_reject_count": int(np.count_nonzero(~admitted & ~expected)),
            "false_accept_count": int(np.count_nonzero(false_accept)),
            "false_reject_count": int(np.count_nonzero(false_reject)),
            "false_accept_fraction": float(np.mean(false_accept)),
            "false_reject_fraction": float(np.mean(false_reject)),
        },
        expected,
    )


def _fallback_metrics(
    prediction: FloatArray,
    local: FloatArray,
    factor: FloatArray,
    fallback: FloatArray,
    fallback_local: FloatArray,
    fallback_factor: FloatArray,
    query_matrix: FloatArray,
    admitted: NDArray[np.bool_],
    reported_mean: object,
    reported_variance: object,
    *,
    moment_atol: float,
) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate_mean, candidate_variance = _query_moments(
        prediction, local, factor, query_matrix
    )
    fallback_mean, fallback_variance = _query_moments(
        fallback, fallback_local, fallback_factor, query_matrix
    )
    reported_q_mean = _float_array(
        reported_mean, name="reported_query_mean", ndim=1
    )
    reported_q_variance = _float_array(
        reported_variance, name="reported_query_variance", ndim=1
    )
    expected_shape = (query_matrix.shape[0],)
    if reported_q_mean.shape != expected_shape or reported_q_variance.shape != expected_shape:
        raise ValueError("reported query moments must have one entry per query")
    if np.any(reported_q_variance <= 0.0):
        raise ValueError("reported_query_variance must be positive")

    branch_mean = np.where(admitted, candidate_mean, fallback_mean)
    branch_variance = np.where(admitted, candidate_variance, fallback_variance)
    output_consistent = bool(
        np.allclose(reported_q_mean, branch_mean, rtol=0.0, atol=moment_atol)
        and np.allclose(
            reported_q_variance,
            branch_variance,
            rtol=0.0,
            atol=moment_atol,
        )
    )
    rejected = ~admitted
    exact_fallback = bool(
        np.array_equal(reported_q_mean[rejected], fallback_mean[rejected])
        and np.array_equal(
            reported_q_variance[rejected],
            fallback_variance[rejected],
        )
    )
    rejected_count = int(np.count_nonzero(rejected))
    return (
        {
            "accepted_query_count": int(np.count_nonzero(admitted)),
            "rejected_query_count": rejected_count,
            "reported_output_consistent": output_consistent,
            "exact_fallback_on_rejection": exact_fallback,
            "rejected_exact_fraction": 1.0 if rejected_count == 0 else float(exact_fallback),
            "maximum_mean_branch_error": float(
                np.max(np.abs(reported_q_mean - branch_mean), initial=0.0)
            ),
            "maximum_variance_branch_error": float(
                np.max(np.abs(reported_q_variance - branch_variance), initial=0.0)
            ),
        },
        {
            "query_output_consistent": output_consistent,
            "exact_fallback": exact_fallback,
        },
    )


def _decision_metrics(
    arrays: Mapping[str, NDArray[Any]],
    *,
    moment_atol: float,
) -> tuple[dict[str, Any], dict[str, bool]]:
    losses = _float_array(
        arrays["decision_loss_by_hypothesis"],
        name="decision_loss_by_hypothesis",
        ndim=2,
    )
    hypothesis_count, action_count = losses.shape
    if hypothesis_count < 1 or action_count < 2:
        raise ValueError("decision loss matrix must have shape (H, A), H >= 1, A >= 2")
    prior = _float_array(arrays["hypothesis_prior"], name="hypothesis_prior", ndim=1)
    if prior.shape != (hypothesis_count,) or np.any(prior < 0.0) or float(prior.sum()) <= 0.0:
        raise ValueError("hypothesis_prior must be nonnegative with positive total mass")
    classes = np.asarray(arrays["quotient_class"])
    if classes.shape != (hypothesis_count,) or classes.dtype.kind not in {"i", "u"}:
        raise ValueError("quotient_class must be an integer vector with one entry per hypothesis")
    classes = classes.astype(np.int64, copy=False)
    class_count = int(classes.max(initial=-1)) + 1
    if class_count < 1 or set(classes.tolist()) != set(range(class_count)):
        raise ValueError("quotient_class labels must be contiguous integers starting at zero")
    mass = _float_array(arrays["quotient_mass"], name="quotient_mass", ndim=1)
    if mass.shape != (class_count,) or np.any(mass < 0.0):
        raise ValueError("quotient_mass must have one nonnegative entry per class")
    if not math.isclose(float(mass.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("quotient_mass must sum to one")
    support = prior > 0.0
    for class_index in range(class_count):
        if not np.any(support & (classes == class_index)):
            raise ValueError("every quotient class must contain positive prior support")

    pairwise = np.zeros((action_count, action_count), dtype=np.float64)
    for class_index in range(class_count):
        selected = support & (classes == class_index)
        differences = (
            losses[selected, :, None] - losses[selected, None, :]
        )
        pairwise += mass[class_index] * np.max(differences, axis=0)
    worst_regret = np.max(pairwise, axis=1)
    reported = _float_array(
        arrays["reported_worst_case_regret"],
        name="reported_worst_case_regret",
        ndim=1,
    )
    if reported.shape != (action_count,):
        raise ValueError("reported_worst_case_regret must have one entry per action")
    regret_consistent = bool(
        np.allclose(reported, worst_regret, rtol=0.0, atol=moment_atol)
    )

    selected_action = _scalar_integer(arrays["selected_action"], name="selected_action")
    fallback_action = _scalar_integer(arrays["fallback_action"], name="fallback_action")
    if not 0 <= selected_action < action_count or not 0 <= fallback_action < action_count:
        raise ValueError("selected_action and fallback_action must index the action set")
    submitted_admitted = _scalar_boolean(
        arrays["decision_admitted"], name="decision_admitted"
    )
    tolerance_value = np.asarray(arrays["regret_tolerance"])
    if tolerance_value.shape != ():
        raise ValueError("regret_tolerance must be a scalar")
    tolerance = _real(
        tolerance_value.item(),
        name="regret_tolerance",
        minimum=0.0,
    )
    minimax_action = int(np.argmin(worst_regret))
    expected_admitted = bool(float(worst_regret[minimax_action]) <= tolerance)
    admission_consistent = submitted_admitted is expected_admitted
    expected_selected = minimax_action if submitted_admitted else fallback_action
    policy_consistent = selected_action == expected_selected

    realized = _float_array(
        arrays["realized_action_loss"],
        name="realized_action_loss",
        ndim=1,
    )
    if realized.shape != (action_count,):
        raise ValueError("realized_action_loss must have one entry per action")
    realized_regret = float(realized[selected_action] - np.min(realized))
    return (
        {
            "hypothesis_count": hypothesis_count,
            "quotient_class_count": class_count,
            "action_count": action_count,
            "worst_case_regret": worst_regret.tolist(),
            "reported_worst_case_regret": reported.tolist(),
            "regret_tolerance": tolerance,
            "minimax_action": minimax_action,
            "expected_admitted": expected_admitted,
            "submitted_admitted": submitted_admitted,
            "selected_action": selected_action,
            "fallback_action": fallback_action,
            "registered_selected_regret": float(worst_regret[selected_action]),
            "realized_action_loss": realized.tolist(),
            "realized_regret": realized_regret,
        },
        {
            "decision_regret_consistent": regret_consistent,
            "decision_admission_consistent": admission_consistent,
            "decision_policy_consistent": policy_consistent,
        },
    )


def _communication_metrics(
    local: FloatArray,
    factor: FloatArray,
    *,
    payload_bytes: int,
) -> dict[str, Any]:
    sample_count = int(local.shape[0])
    dimension = 3 * sample_count
    rank = int(factor.shape[2])
    dense_symmetric_bytes = 8 * dimension * (dimension + 1) // 2
    submitted_covariance_bytes = int(local.nbytes + factor.nbytes)
    theoretical_structured_bytes = 8 * (6 * sample_count + dimension * rank)
    return {
        "state_dimension": dimension,
        "shared_rank": rank,
        "payload_file_bytes": int(payload_bytes),
        "dense_symmetric_covariance_bytes": dense_symmetric_bytes,
        "submitted_covariance_array_bytes": submitted_covariance_bytes,
        "theoretical_symmetric_block_plus_factor_bytes": theoretical_structured_bytes,
        "dense_to_submitted_covariance_ratio": (
            None
            if submitted_covariance_bytes == 0
            else dense_symmetric_bytes / submitted_covariance_bytes
        ),
        "dense_to_theoretical_structured_ratio": (
            None
            if theoretical_structured_bytes == 0
            else dense_symmetric_bytes / theoretical_structured_bytes
        ),
    }


def _case_tasks(value: object, *, name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty task list")
    tasks = [_string(item, name=f"{name} entry") for item in value]
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"{name} must not contain duplicates")
    unknown = set(tasks).difference(_ALLOWED_TASKS)
    if unknown:
        raise ValueError(f"{name} contains unknown tasks: {sorted(unknown)}")
    dependencies = {
        "calibration": {"forecast"},
        "dependence": {"calibration"},
        "query": {"calibration"},
        "gauge": {"query"},
        "fallback": {"gauge"},
        "communication": {"calibration"},
    }
    task_set = set(tasks)
    for task, required in dependencies.items():
        if task in task_set and not required.issubset(task_set):
            raise ValueError(f"task {task!r} requires {sorted(required)}")
    return frozenset(task_set)


def _evaluate_case(
    case: Mapping[str, Any],
    *,
    suite_root: Path,
    coverage_probability: float,
    gauge_tolerance: float,
    moment_atol: float,
    relative_rank_tolerance: float,
) -> dict[str, Any]:
    case_id = _string(case.get("case_id"), name="case_id")
    group_id = _string(case.get("group_id"), name=f"{case_id}.group_id")
    tasks = _case_tasks(case.get("tasks"), name=f"{case_id}.tasks")
    payload = _relative_payload(
        suite_root, case.get("payload"), name=f"{case_id}.payload"
    )
    expected_sha256 = _hex_sha256(
        case.get("payload_sha256"), name=f"{case_id}.payload_sha256"
    )
    actual_sha256 = _sha256_file(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{case_id} payload SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    arrays = _load_payload(payload, tasks=tasks)
    unexpected_case_fields = set(case).difference(
        {"case_id", "group_id", "payload", "payload_sha256", "tasks", "metadata"}
    )
    if unexpected_case_fields:
        raise ValueError(
            f"{case_id} contains unregistered manifest fields: "
            f"{sorted(unexpected_case_fields)}"
        )
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{case_id}.metadata must be an object")

    result: dict[str, Any] = {
        "case_id": case_id,
        "group_id": group_id,
        "tasks": sorted(tasks),
        "payload": payload.relative_to(suite_root).as_posix(),
        "payload_sha256": actual_sha256,
        "metadata": metadata,
        "metrics": {},
        "contract_checks": {},
    }

    truth: FloatArray | None = None
    prediction: FloatArray | None = None
    local: FloatArray | None = None
    factor: FloatArray | None = None
    query_matrix: FloatArray | None = None
    submitted_admission: NDArray[np.bool_] | None = None

    if tasks.intersection(
        {"forecast", "calibration", "dependence", "query", "gauge", "fallback", "communication"}
    ):
        truth = _float_array(arrays["truth_xyz_m"], name="truth_xyz_m", ndim=2)
        prediction = _float_array(
            arrays["prediction_mean_xyz_m"],
            name="prediction_mean_xyz_m",
            ndim=2,
        )
        if truth.ndim != 2 or truth.shape[0] < 1 or truth.shape[1] != 3:
            raise ValueError("truth_xyz_m must have nonempty shape (N, 3)")
        if prediction.shape != truth.shape:
            raise ValueError("prediction_mean_xyz_m must match truth_xyz_m")

    if tasks.intersection(
        {"calibration", "dependence", "query", "gauge", "fallback", "communication"}
    ):
        assert truth is not None
        local, factor = _validate_covariance(
            arrays["conditional_covariance_m2"],
            arrays["shared_factor_m"],
            sample_count=truth.shape[0],
            prefix="prediction",
        )

    if "forecast" in tasks:
        assert truth is not None and prediction is not None
        result["metrics"]["forecast"] = _forecast_metrics(truth, prediction)

    if "calibration" in tasks:
        assert (
            truth is not None
            and prediction is not None
            and local is not None
            and factor is not None
        )
        result["metrics"]["calibration"] = _calibration_metrics(
            truth,
            prediction,
            local,
            factor,
            coverage_probability=coverage_probability,
            relative_rank_tolerance=relative_rank_tolerance,
        )

    if "dependence" in tasks:
        assert (
            truth is not None
            and prediction is not None
            and local is not None
            and factor is not None
        )
        result["metrics"]["dependence"] = _dependence_metrics(
            truth,
            prediction,
            local,
            factor,
            relative_rank_tolerance=relative_rank_tolerance,
        )

    if tasks.intersection({"query", "gauge", "fallback"}):
        assert truth is not None
        query_matrix = _float_array(
            arrays["query_matrix"], name="query_matrix", ndim=2
        )
        dimension = truth.size
        if query_matrix.shape[0] < 1 or query_matrix.shape[1] != dimension:
            raise ValueError(
                f"query_matrix must have shape (Q, {dimension}), Q >= 1"
            )
        if np.any(np.linalg.norm(query_matrix, axis=1) == 0.0):
            raise ValueError("query_matrix must not contain a zero query")

    if "query" in tasks:
        assert (
            truth is not None
            and prediction is not None
            and local is not None
            and factor is not None
            and query_matrix is not None
        )
        query_metrics, _, _, _ = _query_metrics(
            truth,
            prediction,
            local,
            factor,
            query_matrix,
            coverage_probability=coverage_probability,
        )
        result["metrics"]["query"] = query_metrics

    if "gauge" in tasks:
        assert query_matrix is not None
        submitted_admission = np.asarray(arrays["query_admitted"])
        gauge_metrics, _ = _gauge_metrics(
            query_matrix,
            arrays["nullspace_basis"],
            submitted_admission,
            tolerance=gauge_tolerance,
        )
        result["metrics"]["gauge"] = gauge_metrics
        result["contract_checks"]["gauge_admission_consistent"] = (
            gauge_metrics["false_accept_count"] == 0
            and gauge_metrics["false_reject_count"] == 0
        )

    if "fallback" in tasks:
        assert (
            truth is not None
            and prediction is not None
            and local is not None
            and factor is not None
            and query_matrix is not None
            and submitted_admission is not None
        )
        fallback = _float_array(
            arrays["fallback_mean_xyz_m"],
            name="fallback_mean_xyz_m",
            ndim=2,
        )
        if fallback.shape != truth.shape:
            raise ValueError("fallback_mean_xyz_m must match truth_xyz_m")
        fallback_local, fallback_factor = _validate_covariance(
            arrays["fallback_conditional_covariance_m2"],
            arrays["fallback_shared_factor_m"],
            sample_count=truth.shape[0],
            prefix="fallback",
        )
        fallback_metrics, checks = _fallback_metrics(
            prediction,
            local,
            factor,
            fallback,
            fallback_local,
            fallback_factor,
            query_matrix,
            submitted_admission,
            arrays["reported_query_mean"],
            arrays["reported_query_variance"],
            moment_atol=moment_atol,
        )
        result["metrics"]["fallback"] = fallback_metrics
        result["contract_checks"].update(checks)

    if "decision" in tasks:
        decision_metrics, checks = _decision_metrics(
            arrays,
            moment_atol=moment_atol,
        )
        result["metrics"]["decision"] = decision_metrics
        result["contract_checks"].update(checks)

    if "communication" in tasks:
        assert local is not None and factor is not None
        result["metrics"]["communication"] = _communication_metrics(
            local,
            factor,
            payload_bytes=payload.stat().st_size,
        )

    result["contract_pass"] = all(result["contract_checks"].values())
    return result


def _summary_values(case: Mapping[str, Any]) -> dict[str, float]:
    metrics = case["metrics"]
    values: dict[str, float] = {}
    mapping = {
        ("forecast", "rmse_m"): "forecast_rmse_m",
        ("forecast", "point_rmse_m"): "forecast_point_rmse_m",
        ("calibration", "gaussian_nll_per_dimension"): "gaussian_nll_per_dimension",
        ("calibration", "normalized_nees"): "normalized_nees",
        ("calibration", "marginal_coverage"): "marginal_coverage",
        ("dependence", "full_nll_gain_per_dimension"): "dependence_nll_gain_per_dimension",
        ("dependence", "full_nees_error_gain"): "dependence_nees_error_gain",
        ("query", "rmse"): "query_rmse",
        ("query", "gaussian_nll_mean"): "query_gaussian_nll_mean",
        ("query", "normalized_nees_mean"): "query_normalized_nees_mean",
        ("query", "marginal_coverage"): "query_marginal_coverage",
        ("gauge", "false_accept_fraction"): "gauge_false_accept_fraction",
        ("gauge", "false_reject_fraction"): "gauge_false_reject_fraction",
        ("fallback", "rejected_exact_fraction"): "exact_fallback_fraction",
        ("decision", "realized_regret"): "realized_decision_regret",
        ("communication", "dense_to_submitted_covariance_ratio"): (
            "dense_to_submitted_covariance_ratio"
        ),
    }
    for (section, field), name in mapping.items():
        section_value = metrics.get(section)
        if not isinstance(section_value, dict):
            continue
        value = section_value.get(field)
        if value is not None:
            values[name] = float(value)
    return values


def _mean_summaries(summaries: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for summary in summaries for key in summary})
    return {
        key: float(np.mean([summary[key] for summary in summaries if key in summary]))
        for key in keys
    }


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_summaries = [_summary_values(case) for case in cases]
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case, summary in zip(cases, case_summaries, strict=True):
        grouped[str(case["group_id"])].append(summary)
    group_summaries = {
        group_id: _mean_summaries(values)
        for group_id, values in sorted(grouped.items())
    }
    task_counts: dict[str, int] = defaultdict(int)
    check_failures: dict[str, int] = defaultdict(int)
    for case in cases:
        for task in case["tasks"]:
            task_counts[str(task)] += 1
        for name, passed in case["contract_checks"].items():
            if not passed:
                check_failures[str(name)] += 1
    return {
        "case_count": len(cases),
        "independent_group_count": len(grouped),
        "declared_task_case_counts": dict(sorted(task_counts.items())),
        "contract": {
            "passed_case_count": sum(bool(case["contract_pass"]) for case in cases),
            "failed_case_count": sum(not bool(case["contract_pass"]) for case in cases),
            "all_cases_pass": all(bool(case["contract_pass"]) for case in cases),
            "check_failure_counts": dict(sorted(check_failures.items())),
        },
        "equal_case_mean": _mean_summaries(case_summaries),
        "per_group_mean": group_summaries,
        "equal_group_mean": _mean_summaries(list(group_summaries.values())),
    }


def evaluate_information_contract_suite(suite_path: str | Path) -> dict[str, Any]:
    """Validate and evaluate one suite without altering its inputs."""

    path = Path(suite_path)
    suite = _load_json_object(path)
    if set(suite).difference(
        {
            "schema_name",
            "schema_version",
            "suite_id",
            "aggregation_unit",
            "thresholds",
            "cases",
            "claim_boundary",
        }
    ):
        unknown = sorted(
            set(suite).difference(
                {
                    "schema_name",
                    "schema_version",
                    "suite_id",
                    "aggregation_unit",
                    "thresholds",
                    "cases",
                    "claim_boundary",
                }
            )
        )
        raise ValueError(f"suite contains unregistered fields: {unknown}")
    if suite.get("schema_name") != SUITE_SCHEMA or suite.get("schema_version") != SUITE_VERSION:
        raise ValueError("unsupported information-contract suite schema")
    suite_id = _string(suite.get("suite_id"), name="suite_id")
    if suite.get("aggregation_unit") != "group_id":
        raise ValueError("aggregation_unit must be 'group_id'")
    thresholds = suite.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    allowed_thresholds = {
        "coverage_probability",
        "gauge_sensitivity_tolerance",
        "moment_atol",
        "relative_rank_tolerance",
    }
    if set(thresholds).difference(allowed_thresholds):
        raise ValueError(
            "thresholds contain unregistered fields: "
            f"{sorted(set(thresholds).difference(allowed_thresholds))}"
        )
    coverage_probability = _real(
        thresholds.get("coverage_probability"),
        name="coverage_probability",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if coverage_probability == 0.0:
        raise ValueError("coverage_probability must be greater than zero")
    gauge_tolerance = _real(
        thresholds.get("gauge_sensitivity_tolerance"),
        name="gauge_sensitivity_tolerance",
        minimum=0.0,
        maximum=1.0,
    )
    moment_atol = _real(
        thresholds.get("moment_atol"),
        name="moment_atol",
        minimum=0.0,
    )
    relative_rank_tolerance = _real(
        thresholds.get("relative_rank_tolerance"),
        name="relative_rank_tolerance",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    cases_value = suite.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("cases must be a nonempty list")
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, case_value in enumerate(cases_value):
        if not isinstance(case_value, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case = _evaluate_case(
            case_value,
            suite_root=path.parent.resolve(),
            coverage_probability=coverage_probability,
            gauge_tolerance=gauge_tolerance,
            moment_atol=moment_atol,
            relative_rank_tolerance=relative_rank_tolerance,
        )
        if case["case_id"] in case_ids:
            raise ValueError(f"duplicate case_id {case['case_id']!r}")
        case_ids.add(case["case_id"])
        cases.append(case)
    cases.sort(key=lambda value: value["case_id"])
    declared_boundary = suite.get("claim_boundary")
    if declared_boundary is not None:
        _string(declared_boundary, name="claim_boundary")
    return {
        "schema_name": RESULT_SCHEMA,
        "schema_version": RESULT_VERSION,
        "suite_id": suite_id,
        "suite_sha256": _sha256_file(path),
        "claim_boundary": CLAIM_BOUNDARY,
        "suite_claim_boundary": declared_boundary,
        "leaderboard_policy": LEADERBOARD_POLICY,
        "thresholds": {
            "coverage_probability": coverage_probability,
            "gauge_sensitivity_tolerance": gauge_tolerance,
            "moment_atol": moment_atol,
            "relative_rank_tolerance": relative_rank_tolerance,
        },
        "cases": cases,
        "aggregate": _aggregate(cases),
    }



def generate_smoke_suite(directory: str | Path, *, overwrite: bool = False) -> Path:
    """Create the deterministic development fixture without importing it at startup."""

    from .information_contract_benchmark_smoke import generate_smoke_suite as generate

    return generate(directory, overwrite=overwrite)


def _write_result(
    path: Path,
    result: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    atomic_write_text(
        path,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        overwrite=overwrite,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one sealed suite")
    evaluate.add_argument("suite", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    smoke = subparsers.add_parser(
        "smoke", help="generate and evaluate a deterministic smoke suite"
    )
    smoke.add_argument("directory", type=Path)
    smoke.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_information_contract_suite(args.suite)
        _write_result(args.output, result, overwrite=args.overwrite)
        return 0
    suite_path = generate_smoke_suite(args.directory, overwrite=args.overwrite)
    result = evaluate_information_contract_suite(suite_path)
    _write_result(
        args.directory / "result.json",
        result,
        overwrite=args.overwrite,
    )
    return 0


__all__ = [
    "CLAIM_BOUNDARY",
    "LEADERBOARD_POLICY",
    "RESULT_SCHEMA",
    "RESULT_VERSION",
    "SUITE_SCHEMA",
    "SUITE_VERSION",
    "evaluate_information_contract_suite",
    "generate_smoke_suite",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
