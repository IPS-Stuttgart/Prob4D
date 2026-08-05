"""Joint Gaussian calibration diagnostics for conditional plus low-rank covariance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA = "prob4d.joint-covariance-diagnostics"
JOINT_COVARIANCE_DIAGNOSTIC_VERSION = 1
JOINT_COVARIANCE_CLAIM_BOUNDARY = (
    "This diagnostic evaluates matched residuals under the supplied conditional "
    "and shared low-rank covariance. It does not establish observation-provider "
    "competence, guarded BayesianPhysTwin benefit, or Causal4D intervention benefit."
)


def _validated_relative_rank_tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("relative_rank_tolerance must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("relative_rank_tolerance must lie in [0, 1)")
    return result


def _validated_inputs(
    residual_xyz_m: np.ndarray,
    local_covariance_m2: np.ndarray,
    low_rank_factor_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(residual_xyz_m, dtype=np.float64)
    local = np.asarray(local_covariance_m2, dtype=np.float64)
    factor = np.asarray(low_rank_factor_m, dtype=np.float64)
    if residual.ndim != 2 or residual.shape[1:] != (3,) or residual.shape[0] == 0:
        raise ValueError("residual_xyz_m must have nonempty shape (N, 3)")
    if local.shape != (residual.shape[0], 3, 3):
        raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
    if factor.ndim != 3 or factor.shape[:2] != residual.shape:
        raise ValueError("low_rank_factor_m must have shape (N, 3, R)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_xyz_m must be finite")
    if not np.all(np.isfinite(local)):
        raise ValueError("local_covariance_m2 must be finite")
    if not np.all(np.isfinite(factor)):
        raise ValueError("low_rank_factor_m must be finite")
    symmetric = 0.5 * (local + np.swapaxes(local, -1, -2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(-2, -1)), 1.0)
    asymmetry = np.max(np.abs(local - np.swapaxes(local, -1, -2)), axis=(-2, -1))
    if np.any(asymmetry > 1e-12 + 1e-10 * scale):
        raise ValueError("local_covariance_m2 must be symmetric")
    try:
        cholesky = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("local_covariance_m2 must be positive definite") from error
    return residual, symmetric, factor, cholesky


def _small_gram_singular_structure(
    whitened_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``W.T @ W`` plus descending singular values/right vectors.

    The low-rank dimension is normally much smaller than the number of residual
    coordinates. Working in the rank-by-rank Gram matrix therefore avoids a tall
    SVD and never materializes left singular vectors with shape ``(3N, R)``.
    """

    rank_columns = int(whitened_factor.shape[1])
    cross_gram = whitened_factor.T @ whitened_factor
    if rank_columns == 0:
        return (
            cross_gram,
            np.empty(0, dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )
    eigenvalues, right_vectors = np.linalg.eigh(cross_gram)
    order = np.arange(rank_columns - 1, -1, -1)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    right_vectors = right_vectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    return cross_gram, singular_values, right_vectors


def evaluate_joint_gaussian_group(
    residual_xyz_m: np.ndarray,
    local_covariance_m2: np.ndarray,
    low_rank_factor_m: np.ndarray,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Evaluate one dependence group using Woodbury and determinant identities.

    The covariance is ``blockdiag(local_covariance_m2) + U U.T``, where ``U`` is
    the row-stacked ``low_rank_factor_m``. The implementation never materializes
    the dense ``3N x 3N`` covariance.
    """

    tolerance = _validated_relative_rank_tolerance(relative_rank_tolerance)
    residual, _, factor, local_cholesky = _validated_inputs(
        residual_xyz_m,
        local_covariance_m2,
        low_rank_factor_m,
    )
    whitened_residual = np.linalg.solve(
        local_cholesky,
        residual[..., None],
    )[..., 0].reshape(-1)
    rank_columns = factor.shape[-1]
    if rank_columns:
        whitened_factor = np.linalg.solve(local_cholesky, factor).reshape(
            -1,
            rank_columns,
        )
    else:
        whitened_factor = np.empty((whitened_residual.size, 0), dtype=np.float64)

    cross_gram, singular_values, right_vectors = _small_gram_singular_structure(whitened_factor)
    gram = np.eye(rank_columns, dtype=np.float64) + cross_gram
    gram_cholesky = np.linalg.cholesky(gram)
    projected = whitened_factor.T @ whitened_residual
    correction_solution = np.linalg.solve(gram_cholesky, projected)
    correction = float(correction_solution @ correction_solution)
    local_energy = float(whitened_residual @ whitened_residual)
    mahalanobis = max(local_energy - correction, 0.0)

    local_log_determinant = float(
        2.0 * np.sum(np.log(np.diagonal(local_cholesky, axis1=-2, axis2=-1)))
    )
    shared_log_determinant_increment = float(
        2.0 * np.sum(np.log(np.diag(gram_cholesky)))
    )
    log_determinant = local_log_determinant + shared_log_determinant_increment
    dimension = int(whitened_residual.size)
    gaussian_nll = 0.5 * (
        dimension * math.log(2.0 * math.pi) + log_determinant + mahalanobis
    )

    leading = float(singular_values[0]) if singular_values.size else 0.0
    tiny = float(np.finfo(np.float64).tiny)
    threshold = tolerance * max(leading, tiny)
    effective_rank = int(np.count_nonzero(singular_values > threshold))

    if effective_rank:
        active_singular_values = singular_values[:effective_rank]
        active_right_vectors = right_vectors[:, :effective_rank]
        coefficients = (active_right_vectors.T @ projected) / active_singular_values
        shared_energy = float(
            np.sum(np.square(coefficients) / (1.0 + np.square(active_singular_values)))
        )
        projected_local_energy = float(coefficients @ coefficients)
    else:
        shared_energy = 0.0
        projected_local_energy = 0.0
    conditional_dimension = dimension - effective_rank
    conditional_energy = max(local_energy - projected_local_energy, 0.0)

    return {
        "sample_count": int(residual.shape[0]),
        "dimension": dimension,
        "low_rank_column_count": int(rank_columns),
        "effective_shared_rank": effective_rank,
        "conditional_subspace_dimension": conditional_dimension,
        "mahalanobis_squared": mahalanobis,
        "normalized_nees": mahalanobis / dimension,
        "gaussian_nll": gaussian_nll,
        "gaussian_nll_per_dimension": gaussian_nll / dimension,
        "local_log_determinant": local_log_determinant,
        "shared_log_determinant_increment": shared_log_determinant_increment,
        "joint_log_determinant": log_determinant,
        "shared_subspace_normalized_energy": (
            None if effective_rank == 0 else shared_energy / effective_rank
        ),
        "conditional_subspace_normalized_energy": (
            None
            if conditional_dimension == 0
            else conditional_energy / conditional_dimension
        ),
        "relative_rank_tolerance": tolerance,
    }


def _validated_group_ids(value: np.ndarray | None, *, sample_count: int) -> np.ndarray:
    if value is None:
        return np.zeros(sample_count, dtype=np.int64)
    groups = np.asarray(value)
    if groups.ndim != 1 or groups.shape[0] != sample_count:
        raise ValueError("factor_group_ids must have shape (N,)")
    if groups.dtype.kind not in {"i", "u", "U", "S"}:
        raise ValueError("factor_group_ids must contain integer or string identifiers")
    if groups.dtype.kind in {"U", "S"}:
        normalized = groups.astype(str)
        if np.any(np.char.str_len(normalized) == 0):
            raise ValueError("factor_group_ids must not contain empty strings")
        return normalized
    return groups.astype(np.int64, copy=False)


def _json_group_id(value: object) -> int | str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _mean_present(groups: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(group[field]) for group in groups if group[field] is not None]
    return None if not values else float(np.mean(values))


def evaluate_joint_gaussian_groups(
    residual_xyz_m: np.ndarray,
    local_covariance_m2: np.ndarray,
    low_rank_factor_m: np.ndarray,
    *,
    factor_group_ids: np.ndarray | None = None,
    relative_rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Evaluate independent factor groups and aggregate them with equal group weight."""

    residual, local, factor, _ = _validated_inputs(
        residual_xyz_m,
        local_covariance_m2,
        low_rank_factor_m,
    )
    groups = _validated_group_ids(factor_group_ids, sample_count=residual.shape[0])
    group_records: list[dict[str, Any]] = []
    for group_id in np.unique(groups):
        selected = groups == group_id
        metrics = evaluate_joint_gaussian_group(
            residual[selected],
            local[selected],
            factor[selected],
            relative_rank_tolerance=relative_rank_tolerance,
        )
        group_records.append({"factor_group_id": _json_group_id(group_id), **metrics})

    return {
        "sample_count": int(residual.shape[0]),
        "group_count": len(group_records),
        "groups": group_records,
        "equal_group_mean": {
            "normalized_nees": _mean_present(group_records, "normalized_nees"),
            "gaussian_nll_per_dimension": _mean_present(
                group_records,
                "gaussian_nll_per_dimension",
            ),
            "shared_subspace_normalized_energy": _mean_present(
                group_records,
                "shared_subspace_normalized_energy",
            ),
            "conditional_subspace_normalized_energy": _mean_present(
                group_records,
                "conditional_subspace_normalized_energy",
            ),
        },
    }


def _write_text_exclusive(path: Path, content: str) -> None:
    """Publish complete text without replacing a concurrently created path."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_joint_covariance_diagnostic(
    input_path: str | Path,
    output_path: str | Path,
    *,
    relative_rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Evaluate one portable matched-residual NPZ and write an exclusive report."""

    source = Path(input_path)
    payload = source.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        required = {"residual_xyz_m", "local_covariance_m2", "low_rank_factor_m"}
        allowed = required | {"factor_group_ids"}
        missing = sorted(required - set(data.files))
        extra = sorted(set(data.files) - allowed)
        if missing or extra:
            raise ValueError(
                "joint-covariance input fields changed; "
                f"missing={missing}, extra={extra}"
            )
        evaluation = evaluate_joint_gaussian_groups(
            data["residual_xyz_m"],
            data["local_covariance_m2"],
            data["low_rank_factor_m"],
            factor_group_ids=(
                data["factor_group_ids"] if "factor_group_ids" in data else None
            ),
            relative_rank_tolerance=relative_rank_tolerance,
        )
    report = {
        "schema_name": JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA,
        "schema_version": JOINT_COVARIANCE_DIAGNOSTIC_VERSION,
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "evaluation": evaluation,
        "claim_boundary": JOINT_COVARIANCE_CLAIM_BOUNDARY,
    }
    destination = Path(output_path)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _write_text_exclusive(destination, encoded)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1e-10)
    arguments = parser.parse_args(argv)
    run_joint_covariance_diagnostic(
        arguments.input,
        arguments.output,
        relative_rank_tolerance=arguments.relative_rank_tolerance,
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOINT_COVARIANCE_CLAIM_BOUNDARY",
    "JOINT_COVARIANCE_DIAGNOSTIC_SCHEMA",
    "JOINT_COVARIANCE_DIAGNOSTIC_VERSION",
    "evaluate_joint_gaussian_group",
    "evaluate_joint_gaussian_groups",
    "run_joint_covariance_diagnostic",
]
