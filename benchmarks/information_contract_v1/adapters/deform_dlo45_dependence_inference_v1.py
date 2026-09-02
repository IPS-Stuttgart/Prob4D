#!/usr/bin/env python3
"""Trajectory-level secondary inference for the DEFORM dependence witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _vector(value: object, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with at least two groups")
    return result


def paired_group_inference(
    full_nll: object,
    diagonal_nll: object,
    full_nees: object,
    diagonal_nees: object,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap complete groups and compute an exact paired sign test."""

    full_nll_array = _vector(full_nll, name="full_nll")
    diagonal_nll_array = _vector(diagonal_nll, name="diagonal_nll")
    full_nees_array = _vector(full_nees, name="full_nees")
    diagonal_nees_array = _vector(diagonal_nees, name="diagonal_nees")
    length = len(full_nll_array)
    if any(
        len(value) != length
        for value in (diagonal_nll_array, full_nees_array, diagonal_nees_array)
    ):
        raise ValueError("all paired group vectors must have identical length")
    if np.any(full_nees_array <= 0.0) or np.any(diagonal_nees_array <= 0.0):
        raise ValueError("group normalized error ratios must be positive")
    if isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    nll_gain = diagonal_nll_array - full_nll_array
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, length, size=(replicates, length))
    bootstrap_nll_gain = np.mean(nll_gain[selected], axis=1)
    bootstrap_calibration_gain = np.abs(
        np.log(np.mean(diagonal_nees_array[selected], axis=1))
    ) - np.abs(np.log(np.mean(full_nees_array[selected], axis=1)))

    wins = int(np.count_nonzero(nll_gain > 0.0))
    losses = int(np.count_nonzero(nll_gain < 0.0))
    ties = length - wins - losses
    non_ties = wins + losses
    if non_ties:
        greater = max(wins, losses)
        one_sided = sum(
            math.comb(non_ties, value) for value in range(greater, non_ties + 1)
        ) / (2**non_ties)
        two_sided = min(1.0, 2.0 * one_sided)
    else:
        two_sided = 1.0

    calibration_gain = abs(math.log(float(np.mean(diagonal_nees_array)))) - abs(
        math.log(float(np.mean(full_nees_array)))
    )
    return {
        "independent_group_count": length,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "paired_nll_gain_diagonal_minus_full": float(np.mean(nll_gain)),
        "paired_nll_gain_ci95": np.quantile(
            bootstrap_nll_gain, (0.025, 0.975)
        ).tolist(),
        "paired_calibration_error_gain_diagonal_minus_full": calibration_gain,
        "paired_calibration_error_gain_ci95": np.quantile(
            bootstrap_calibration_gain, (0.025, 0.975)
        ).tolist(),
        "full_dependence_nll_win_count": wins,
        "diagonal_nll_win_count": losses,
        "nll_tie_count": ties,
        "paired_sign_test_two_sided_p": two_sided,
    }


def summarize(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_name") != "prob4d.information-contract-witness-result":
        raise ValueError("unexpected witness result schema")
    if result.get("target_query_reselection") is not False:
        raise ValueError("held query was reselected")
    submissions = result.get("submissions")
    if not isinstance(submissions, dict):
        raise ValueError("witness result omits submissions")
    full = submissions.get("full-source-fitted-dependence")
    diagonal = submissions.get("marginal-matched-diagonal")
    if not isinstance(full, dict) or not isinstance(diagonal, dict):
        raise ValueError("required dependence submissions are absent")
    inference = paired_group_inference(
        full["per_group_query_gaussian_nll"],
        diagonal["per_group_query_gaussian_nll"],
        full["per_group_query_normalized_error_ratio"],
        diagonal["per_group_query_normalized_error_ratio"],
        replicates=20_000,
        seed=20_260_902,
    )
    record: dict[str, object] = {
        "schema_name": "prob4d.deform-dlo45-dependence-witness-inference",
        "schema_version": 1,
        "classification": (
            "post-hoc trajectory-level uncertainty for a retrospective "
            "public-data diagnostic"
        ),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "witness_id": result["witness_id"],
        "target_query_reselection": False,
        **inference,
        "claim_boundary": (
            "The paired bootstrap and sign test use complete held trajectories as "
            "the statistical units. They were added after inspection of the "
            "retrospective result and are secondary, not preregistered confirmatory "
            "inference."
        ),
    }
    record["inference_id"] = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = summarize(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
