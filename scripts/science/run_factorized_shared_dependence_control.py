#!/usr/bin/env python3
"""Reproduce dense parity, moment, and storage controls for shared dependence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.factorized_dependence import BlockSharedGaussianCovariance

SCHEMA = "prob4d.factorized-shared-dependence-control.v1"
RESULT_SCHEMA = "prob4d.factorized-shared-dependence-result.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    expected = {
        "schema",
        "evidence_kind",
        "seed",
        "group_count",
        "block_dimension",
        "latent_rank",
        "strength",
        "factor_scale",
        "factor_diagonal_offset",
        "sample_count",
        "paper_scale_group_count",
        "information_boundary",
        "claim_boundary",
    }
    if type(protocol) is not dict or set(protocol) != expected:
        raise ValueError("protocol field set changed")
    if protocol["schema"] != SCHEMA:
        raise ValueError("unsupported protocol schema")
    if protocol["evidence_kind"] != "designed-linear-algebra-control":
        raise ValueError("unsupported evidence kind")
    for name in (
        "seed",
        "group_count",
        "block_dimension",
        "latent_rank",
        "sample_count",
        "paper_scale_group_count",
    ):
        if type(protocol[name]) is not int or protocol[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if protocol["latent_rank"] < protocol["block_dimension"]:
        raise ValueError("latent_rank must be at least block_dimension")
    for name in ("strength", "factor_scale", "factor_diagonal_offset"):
        value = protocol[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= float(protocol["strength"]) < 1.0:
        raise ValueError("strength must lie in [0, 1)")
    if protocol["factor_scale"] <= 0.0 or protocol["factor_diagonal_offset"] <= 0.0:
        raise ValueError("factor scale and offset must be positive")
    expected_boundary = {
        "provider_predictions_opened": False,
        "source_outcomes_opened": False,
        "heldout_outcomes_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }
    if protocol["information_boundary"] != expected_boundary:
        raise ValueError("information boundary changed")
    if not isinstance(protocol["claim_boundary"], str) or not protocol["claim_boundary"].strip():
        raise ValueError("claim_boundary must be nonempty")
    return protocol


def _build_model(protocol: dict[str, Any]) -> BlockSharedGaussianCovariance:
    generator = np.random.default_rng(protocol["seed"])
    groups = protocol["group_count"]
    dimension = protocol["block_dimension"]
    rank = protocol["latent_rank"]
    factors = generator.normal(
        scale=float(protocol["factor_scale"]),
        size=(groups, dimension, rank),
    )
    factors[:, :, :dimension] += float(protocol["factor_diagonal_offset"]) * np.eye(
        dimension
    )[None, :, :]
    blocks = np.einsum("idr,ier->ide", factors, factors)
    return BlockSharedGaussianCovariance(
        marginal_blocks=blocks,
        shared_factors=factors,
        strength=float(protocol["strength"]),
    )


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    model = _build_model(protocol)
    generator = np.random.default_rng(protocol["seed"] + 1)
    residual = generator.normal(size=model.dimension)
    right = generator.normal(size=(model.dimension, 5))
    dense = model.dense_covariance()
    dense_solve = np.linalg.solve(dense, right)
    dense_residual_solve = np.linalg.solve(dense, residual)
    sign, dense_logdet = np.linalg.slogdet(dense)
    if sign <= 0.0:
        raise RuntimeError("designed dense covariance is not positive definite")
    dense_quadratic = float(residual @ dense_residual_solve)
    dense_nll = 0.5 * (
        model.dimension * math.log(2.0 * math.pi)
        + dense_logdet
        + dense_quadratic
    )

    samples = model.sample(generator, protocol["sample_count"]).reshape(
        protocol["sample_count"], model.dimension
    )
    empirical_mean = np.mean(samples, axis=0)
    empirical_covariance = np.cov(samples, rowvar=False, bias=True)
    sample_relative_frobenius_error = float(
        np.linalg.norm(empirical_covariance - dense) / np.linalg.norm(dense)
    )

    paper_groups = protocol["paper_scale_group_count"]
    dimension = protocol["block_dimension"]
    rank = protocol["latent_rank"]
    factorized_values = paper_groups * dimension * (dimension + rank)
    dense_values = (paper_groups * dimension) ** 2
    item_size = np.dtype(np.float64).itemsize

    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "evidence_kind": protocol["evidence_kind"],
        "protocol": protocol,
        "protocol_id": _content_id(protocol),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "model": {
            "dimension": model.dimension,
            "group_count": model.group_count,
            "block_dimension": model.block_dimension,
            "latent_rank": model.latent_rank,
            "strength": model.strength,
        },
        "dense_parity": {
            "maximum_solve_absolute_error": float(
                np.max(np.abs(model.solve(right) - dense_solve))
            ),
            "log_determinant_absolute_error": abs(
                model.log_determinant() - float(dense_logdet)
            ),
            "quadratic_form_absolute_error": abs(
                model.quadratic_form(residual) - dense_quadratic
            ),
            "gaussian_nll_absolute_error": abs(
                model.gaussian_nll(residual) - dense_nll
            ),
        },
        "sampling_control": {
            "sample_count": protocol["sample_count"],
            "maximum_empirical_mean_absolute_value": float(
                np.max(np.abs(empirical_mean))
            ),
            "relative_covariance_frobenius_error": sample_relative_frobenius_error,
        },
        "paper_scale_storage": {
            "group_count": paper_groups,
            "dimension": paper_groups * dimension,
            "factorized_float64_bytes": factorized_values * item_size,
            "dense_float64_bytes": dense_values * item_size,
            "dense_to_factorized_storage_ratio": dense_values / factorized_values,
        },
        "complexity": {
            "storage": "O(N*D*(D+R))",
            "solve": "O(N*D^3 + N*D^2*R + N*D*R^2 + R^3)",
            "sample": "O(N*D*(D+R)) per draw",
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    source = Path(__file__)
    report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["artifact_id"] = _content_id(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    report = build_report(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "artifact_id": report["artifact_id"],
                "storage_ratio": report["paper_scale_storage"][
                    "dense_to_factorized_storage_ratio"
                ],
                "maximum_solve_absolute_error": report["dense_parity"][
                    "maximum_solve_absolute_error"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
