"""Paired dependence ablations for Prob4D conditional plus low-rank covariance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .joint_covariance_metrics import (
    _write_text_exclusive,
    evaluate_joint_gaussian_groups,
)

JOINT_COVARIANCE_ABLATION_SCHEMA = "prob4d.joint-covariance-ablation"
JOINT_COVARIANCE_ABLATION_VERSION = 1
JOINT_COVARIANCE_ABLATION_CLAIM_BOUNDARY = (
    "This target-free diagnostic compares the supplied full joint covariance with "
    "marginal-preserving and conditional-only dependence ablations on the same "
    "matched residual groups. It does not establish real provider competence, "
    "BayesianPhysTwin benefit, Causal4D intervention benefit, or deployment safety."
)
MAX_BOOTSTRAP_REPLICATES = 100_000
_BOOTSTRAP_INDEX_BUDGET = 1_000_000
_POSITIVE_FAVORS_JOINT = (
    "Positive paired advantages favor the full joint covariance over the named ablation."
)
_ARM_DEFINITIONS = {
    "joint": "blockdiag(D_i) + U U^T",
    "marginal_preserving_independence": "blockdiag(D_i + U_i U_i^T)",
    "conditional_only": "blockdiag(D_i)",
}
_COMPARISON_METRICS = (
    "gaussian_nll_per_dimension_advantage",
    "normalized_nees_absolute_error_advantage",
)


def _validated_bootstrap_replicates(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    result = int(value)
    if not 0 <= result <= MAX_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "bootstrap_replicates must lie in "
            f"[0, {MAX_BOOTSTRAP_REPLICATES}]"
        )
    return result


def _validated_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("bootstrap_seed must be an integer")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError("bootstrap_seed must lie in [0, 2**32 - 1]")
    return result


def _validated_confidence_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("confidence_level must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    return result


def _marginal_preserving_local_covariance(
    local_covariance_m2: np.ndarray,
    low_rank_factor_m: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_covariance_m2, dtype=np.float64)
    factor = np.asarray(low_rank_factor_m, dtype=np.float64)
    if local.ndim != 3 or local.shape[1:] != (3, 3):
        raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
    if factor.ndim != 3 or factor.shape[:2] != local.shape[:2]:
        raise ValueError("low_rank_factor_m must have shape (N, 3, R)")
    return local + np.einsum("nir,njr->nij", factor, factor, optimize=True)


def _zero_shared_factor(sample_count: int) -> np.ndarray:
    return np.empty((sample_count, 3, 0), dtype=np.float64)


def _group_key(value: object) -> tuple[str, int | str]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return ("int", int(value))
    return ("str", str(value))


def _paired_comparison(
    joint_groups: Sequence[Mapping[str, Any]],
    ablation_groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ablation_by_id = {
        _group_key(group["factor_group_id"]): group for group in ablation_groups
    }
    if len(ablation_by_id) != len(ablation_groups):
        raise ValueError("ablation group identifiers must be unique")

    records: list[dict[str, Any]] = []
    for joint in joint_groups:
        group_id = joint["factor_group_id"]
        key = _group_key(group_id)
        if key not in ablation_by_id:
            raise ValueError("joint and ablation group identifiers differ")
        ablation = ablation_by_id.pop(key)
        if int(joint["dimension"]) != int(ablation["dimension"]):
            raise ValueError("joint and ablation group dimensions differ")
        joint_nll = float(joint["gaussian_nll_per_dimension"])
        ablation_nll = float(ablation["gaussian_nll_per_dimension"])
        joint_nees = float(joint["normalized_nees"])
        ablation_nees = float(ablation["normalized_nees"])
        records.append(
            {
                "factor_group_id": group_id,
                "dimension": int(joint["dimension"]),
                "gaussian_nll_per_dimension_advantage": ablation_nll - joint_nll,
                "normalized_nees_absolute_error_advantage": (
                    abs(ablation_nees - 1.0) - abs(joint_nees - 1.0)
                ),
                "joint_gaussian_nll_per_dimension": joint_nll,
                "ablation_gaussian_nll_per_dimension": ablation_nll,
                "joint_normalized_nees": joint_nees,
                "ablation_normalized_nees": ablation_nees,
            }
        )
    if ablation_by_id:
        raise ValueError("joint and ablation group identifiers differ")
    return records


def _metric_values(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if not records:
        raise ValueError("at least one independent factor group is required")
    return np.asarray(
        [
            [float(record[metric]) for metric in _COMPARISON_METRICS]
            for record in records
        ],
        dtype=np.float64,
    )


def _bootstrap_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    values = _metric_values(records)
    group_count = int(values.shape[0])
    means = np.mean(values, axis=0)
    mean_payload = {
        metric: {"mean": float(means[index])}
        for index, metric in enumerate(_COMPARISON_METRICS)
    }
    if bootstrap_replicates == 0:
        return {
            "available": False,
            "reason": "bootstrap-disabled",
            "unit": "factor_group",
            "group_count": group_count,
            "replicates": 0,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "metrics": mean_payload,
        }
    if group_count < 2:
        return {
            "available": False,
            "reason": "fewer-than-two-independent-groups",
            "unit": "factor_group",
            "group_count": group_count,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "metrics": mean_payload,
        }

    generator = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(
        (bootstrap_replicates, len(_COMPARISON_METRICS)),
        dtype=np.float64,
    )
    chunk_size = max(
        1,
        min(
            bootstrap_replicates,
            _BOOTSTRAP_INDEX_BUDGET // group_count,
        ),
    )
    for start in range(0, bootstrap_replicates, chunk_size):
        stop = min(start + chunk_size, bootstrap_replicates)
        indices = generator.integers(
            0,
            group_count,
            size=(stop - start, group_count),
            endpoint=False,
        )
        bootstrap_means[start:stop] = np.mean(values[indices], axis=1)

    tail = 0.5 * (1.0 - confidence_level)
    quantiles = np.quantile(bootstrap_means, [tail, 1.0 - tail], axis=0)
    result_metrics = {
        metric: {
            "mean": float(means[index]),
            "lower": float(quantiles[0, index]),
            "upper": float(quantiles[1, index]),
        }
        for index, metric in enumerate(_COMPARISON_METRICS)
    }
    return {
        "available": True,
        "reason": None,
        "unit": "factor_group",
        "group_count": group_count,
        "replicates": bootstrap_replicates,
        "seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "metrics": result_metrics,
    }


def _descriptive_comparison_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = _metric_values(records)
    return {
        "equal_group_mean": {
            metric: float(np.mean(values[:, index]))
            for index, metric in enumerate(_COMPARISON_METRICS)
        },
        "joint_better_group_fraction": {
            metric: float(np.mean(values[:, index] > 0.0))
            for index, metric in enumerate(_COMPARISON_METRICS)
        },
    }


def compare_joint_covariance_ablations(
    residual_xyz_m: np.ndarray,
    local_covariance_m2: np.ndarray,
    low_rank_factor_m: np.ndarray,
    *,
    factor_group_ids: np.ndarray | None = None,
    relative_rank_tolerance: float = 1e-10,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 7,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compare full dependence with marginal-preserving and conditional-only arms."""

    replicates = _validated_bootstrap_replicates(bootstrap_replicates)
    seed = _validated_seed(bootstrap_seed)
    level = _validated_confidence_level(confidence_level)

    residual = np.asarray(residual_xyz_m, dtype=np.float64)
    local = np.asarray(local_covariance_m2, dtype=np.float64)
    factor = np.asarray(low_rank_factor_m, dtype=np.float64)
    sample_count = int(residual.shape[0]) if residual.ndim else 0
    zero_factor = _zero_shared_factor(sample_count)
    marginal_local = _marginal_preserving_local_covariance(local, factor)

    arms = {
        "joint": evaluate_joint_gaussian_groups(
            residual,
            local,
            factor,
            factor_group_ids=factor_group_ids,
            relative_rank_tolerance=relative_rank_tolerance,
        ),
        "marginal_preserving_independence": evaluate_joint_gaussian_groups(
            residual,
            marginal_local,
            zero_factor,
            factor_group_ids=factor_group_ids,
            relative_rank_tolerance=relative_rank_tolerance,
        ),
        "conditional_only": evaluate_joint_gaussian_groups(
            residual,
            local,
            zero_factor,
            factor_group_ids=factor_group_ids,
            relative_rank_tolerance=relative_rank_tolerance,
        ),
    }

    comparisons: dict[str, Any] = {}
    for ablation_name in ("marginal_preserving_independence", "conditional_only"):
        records = _paired_comparison(
            arms["joint"]["groups"],
            arms[ablation_name]["groups"],
        )
        comparisons[ablation_name] = {
            "interpretation": _POSITIVE_FAVORS_JOINT,
            "groups": records,
            **_descriptive_comparison_summary(records),
            "paired_group_bootstrap": _bootstrap_summary(
                records,
                bootstrap_replicates=replicates,
                bootstrap_seed=seed,
                confidence_level=level,
            ),
        }

    return {
        "sample_count": arms["joint"]["sample_count"],
        "group_count": arms["joint"]["group_count"],
        "arm_definitions": dict(_ARM_DEFINITIONS),
        "arms": arms,
        "comparisons": comparisons,
        "bootstrap": {
            "unit": "factor_group",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": level,
        },
    }


def run_joint_covariance_ablation(
    input_path: str | Path,
    output_path: str | Path,
    *,
    relative_rank_tolerance: float = 1e-10,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 7,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Evaluate exact NPZ bytes and publish one no-clobber ablation report."""

    source = Path(input_path)
    payload = source.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        required = {"residual_xyz_m", "local_covariance_m2", "low_rank_factor_m"}
        allowed = required | {"factor_group_ids"}
        missing = sorted(required - set(data.files))
        extra = sorted(set(data.files) - allowed)
        if missing or extra:
            raise ValueError(
                "joint-covariance ablation input fields changed; "
                f"missing={missing}, extra={extra}"
            )
        evaluation = compare_joint_covariance_ablations(
            data["residual_xyz_m"],
            data["local_covariance_m2"],
            data["low_rank_factor_m"],
            factor_group_ids=(
                data["factor_group_ids"] if "factor_group_ids" in data else None
            ),
            relative_rank_tolerance=relative_rank_tolerance,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )

    report = {
        "schema_name": JOINT_COVARIANCE_ABLATION_SCHEMA,
        "schema_version": JOINT_COVARIANCE_ABLATION_VERSION,
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "evaluation": evaluation,
        "claim_boundary": JOINT_COVARIANCE_ABLATION_CLAIM_BOUNDARY,
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
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    arguments = parser.parse_args(argv)
    run_joint_covariance_ablation(
        arguments.input,
        arguments.output,
        relative_rank_tolerance=arguments.relative_rank_tolerance,
        bootstrap_replicates=arguments.bootstrap_replicates,
        bootstrap_seed=arguments.bootstrap_seed,
        confidence_level=arguments.confidence_level,
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOINT_COVARIANCE_ABLATION_CLAIM_BOUNDARY",
    "JOINT_COVARIANCE_ABLATION_SCHEMA",
    "JOINT_COVARIANCE_ABLATION_VERSION",
    "MAX_BOOTSTRAP_REPLICATES",
    "compare_joint_covariance_ablations",
    "run_joint_covariance_ablation",
]
