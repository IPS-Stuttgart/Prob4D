#!/usr/bin/env python3
"""Validate and quantify the conservative axial geometric error budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.axial_orbit_geometry_bound import (
    bound_axial_query_coefficient_error,
    project_axial_query_coefficients,
)

SCHEMA = "prob4d.axial-orbit-geometric-error-control.v1"
RESULT_SCHEMA = "prob4d.axial-orbit-geometric-error-control-result.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _interval(
    value: object,
    *,
    name: str,
    integer: bool,
    minimum: float,
) -> tuple[float, float] | tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    if integer:
        if any(type(item) is not int for item in value):
            raise ValueError(f"{name} must contain integers")
        lower, upper = int(value[0]), int(value[1])
    else:
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError(f"{name} must contain real values")
        lower, upper = float(value[0]), float(value[1])
    if lower < minimum or upper < lower:
        raise ValueError(f"{name} is invalid")
    return lower, upper


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "evidence_kind",
        "seed",
        "case_count",
        "point_count_range",
        "query_dimension_range",
        "maximum_point_error",
        "maximum_pivot_error",
        "maximum_axis_tangent_perturbation",
        "registered_checks",
        "claim_boundary",
    }
    if type(protocol) is not dict or set(protocol) != expected:
        raise ValueError("protocol fields changed")
    if protocol["schema"] != SCHEMA:
        raise ValueError("protocol schema changed")
    if protocol["evidence_kind"] != "designed-geometric-error-bound-control":
        raise ValueError("evidence kind changed")
    if type(protocol["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if type(protocol["case_count"]) is not int or protocol["case_count"] < 1000:
        raise ValueError("case_count must be an integer of at least 1000")
    _interval(
        protocol["point_count_range"],
        name="point_count_range",
        integer=True,
        minimum=1,
    )
    _interval(
        protocol["query_dimension_range"],
        name="query_dimension_range",
        integer=True,
        minimum=1,
    )
    for name in (
        "maximum_point_error",
        "maximum_pivot_error",
        "maximum_axis_tangent_perturbation",
    ):
        value = protocol[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if float(protocol["maximum_axis_tangent_perturbation"]) > 1.0:
        raise ValueError("axis tangent perturbation is unexpectedly broad")
    checks = protocol["registered_checks"]
    if set(checks) != {
        "maximum_operator_bound_violation",
        "minimum_nonzero_ratio_count",
        "maximum_actual_to_bound_ratio",
        "minimum_maximum_actual_to_bound_ratio",
    }:
        raise ValueError("registered checks changed")
    if not isinstance(protocol["claim_boundary"], str) or not protocol["claim_boundary"].strip():
        raise ValueError("claim_boundary is invalid")
    return protocol


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(probability * 100):02d}": float(np.quantile(values, probability))
        for probability in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    }


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(protocol["seed"]))
    point_range = _interval(
        protocol["point_count_range"],
        name="point_count_range",
        integer=True,
        minimum=1,
    )
    query_range = _interval(
        protocol["query_dimension_range"],
        name="query_dimension_range",
        integer=True,
        minimum=1,
    )
    assert isinstance(point_range[0], int)
    assert isinstance(query_range[0], int)

    ratios: list[float] = []
    slacks: list[float] = []
    actual_errors: list[float] = []
    bounds: list[float] = []
    axis_errors: list[float] = []
    point_error_maxima: list[float] = []
    pivot_errors: list[float] = []
    maximum_violation = 0.0

    for _ in range(int(protocol["case_count"])):
        point_count = int(rng.integers(point_range[0], point_range[1] + 1))
        query_dimension = int(rng.integers(query_range[0], query_range[1] + 1))
        estimated_points = rng.normal(size=(point_count, 3))
        estimated_axis = _unit(rng.normal(size=3))
        estimated_pivot = rng.normal(size=3)
        query_weights = rng.normal(size=(query_dimension, point_count, 3))

        point_limits = rng.uniform(
            0.0,
            float(protocol["maximum_point_error"]),
            size=point_count,
        )
        point_directions = rng.normal(size=(point_count, 3))
        point_directions /= np.linalg.norm(point_directions, axis=1)[:, None]
        point_magnitudes = rng.uniform(0.0, 1.0, size=point_count) * point_limits
        true_points = estimated_points + point_directions * point_magnitudes[:, None]

        pivot_limit = float(
            rng.uniform(0.0, float(protocol["maximum_pivot_error"]))
        )
        pivot_direction = _unit(rng.normal(size=3))
        pivot_magnitude = float(rng.uniform(0.0, pivot_limit))
        true_pivot = estimated_pivot + pivot_direction * pivot_magnitude

        tangent = rng.normal(size=3)
        tangent -= estimated_axis * float(tangent @ estimated_axis)
        if float(np.linalg.norm(tangent)) < 1e-14:
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(seed @ estimated_axis)) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
            tangent = seed - estimated_axis * float(seed @ estimated_axis)
        tangent = _unit(tangent)
        tangent_scale = float(
            rng.uniform(
                0.0,
                float(protocol["maximum_axis_tangent_perturbation"]),
            )
        )
        true_axis = _unit(estimated_axis + tangent_scale * tangent)
        axis_error = float(np.linalg.norm(true_axis - estimated_axis))

        estimate = project_axial_query_coefficients(
            estimated_points,
            axis=estimated_axis,
            pivot=estimated_pivot,
            query_weights=query_weights,
        )
        truth = project_axial_query_coefficients(
            true_points,
            axis=true_axis,
            pivot=true_pivot,
            query_weights=query_weights,
        )
        actual = float(np.linalg.svd(truth - estimate, compute_uv=False)[0])
        budget = bound_axial_query_coefficient_error(
            estimated_points,
            estimated_axis=estimated_axis,
            estimated_pivot=estimated_pivot,
            query_weights=query_weights,
            point_position_error_bounds=point_limits,
            axis_vector_error_bound=axis_error,
            pivot_position_error_bound=pivot_limit,
        )
        bound = budget.coefficient_operator_error_bound
        violation = max(0.0, actual - bound)
        maximum_violation = max(maximum_violation, violation)
        actual_errors.append(actual)
        bounds.append(bound)
        slacks.append(bound - actual)
        axis_errors.append(axis_error)
        point_error_maxima.append(float(np.max(point_limits)))
        pivot_errors.append(pivot_limit)
        if bound > 0.0:
            ratios.append(actual / bound)

    ratio_array = np.asarray(ratios, dtype=np.float64)
    bound_array = np.asarray(bounds, dtype=np.float64)
    actual_array = np.asarray(actual_errors, dtype=np.float64)
    slack_array = np.asarray(slacks, dtype=np.float64)
    checks = protocol["registered_checks"]
    registered = {
        "no_operator_bound_violation": maximum_violation
        <= float(checks["maximum_operator_bound_violation"]),
        "enough_nonzero_ratio_cases": ratio_array.size
        >= int(checks["minimum_nonzero_ratio_count"]),
        "all_actual_to_bound_ratios_at_most_one": float(np.max(ratio_array))
        <= float(checks["maximum_actual_to_bound_ratio"]),
        "control_exercises_nontrivial_bound_tightness": float(np.max(ratio_array))
        >= float(checks["minimum_maximum_actual_to_bound_ratio"]),
    }
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "evidence_kind": protocol["evidence_kind"],
        "protocol": protocol,
        "protocol_sha256": _content_id(protocol),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "case_count": int(protocol["case_count"]),
        "nonzero_ratio_count": int(ratio_array.size),
        "maximum_operator_bound_violation": maximum_violation,
        "actual_to_bound_ratio_quantiles": _quantiles(ratio_array),
        "actual_error_quantiles": _quantiles(actual_array),
        "coefficient_bound_quantiles": _quantiles(bound_array),
        "bound_slack_quantiles": _quantiles(slack_array),
        "axis_error_quantiles": _quantiles(np.asarray(axis_errors)),
        "maximum_point_error_quantiles": _quantiles(
            np.asarray(point_error_maxima)
        ),
        "pivot_error_bound_quantiles": _quantiles(np.asarray(pivot_errors)),
        "registered_checks": registered,
        "decision": "passed" if all(registered.values()) else "failed",
        "claim_boundary": protocol["claim_boundary"],
    }
    report["artifact_id"] = _content_id(report)
    return report


def _summary(report: dict[str, Any]) -> str:
    ratio = report["actual_to_bound_ratio_quantiles"]
    return "\n".join(
        (
            "# Axial geometric error-bound control",
            "",
            f"Artifact ID: `{report['artifact_id']}`",
            "",
            f"Decision: **{report['decision']}**",
            "",
            f"Cases: {report['case_count']}",
            "",
            f"Maximum operator-bound violation: {report['maximum_operator_bound_violation']:.6g}",
            "",
            "Actual coefficient error divided by the conservative returned bound:",
            "",
            f"- median: {ratio['q50']:.6f}",
            f"- 95th percentile: {ratio['q95']:.6f}",
            f"- 99th percentile: {ratio['q99']:.6f}",
            f"- maximum: {ratio['q100']:.6f}",
            "",
            report["claim_boundary"],
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    report = build_report(protocol)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        _summary(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_id": report["artifact_id"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
