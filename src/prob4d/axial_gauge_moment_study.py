"""Reproduce a target-free analytic control for nonlinear axial gauge moments.

Results are generated evidence and belong in BayesianPhysTwin-Paper, not in the
code repository. Gaussian expected scores assess moment approximations only;
the non-Gaussian orbit law is not called a calibrated Gaussian posterior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from . import axial_gauge_moments
from .axial_gauge_moments import AxialGaugeOrbit, CircularMoments2

SCHEMA = "prob4d.axial-gauge-moment-control.v1"
METHODS = (
    "first-order",
    "second-order",
    "spherical-radial-2",
    "gauss-hermite-5",
    "gauss-hermite-32",
    "exact-circular-moments",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hermite_radial(radius: float, sigma: float, nodes: int) -> tuple[float, float]:
    locations, mass = np.polynomial.hermite.hermgauss(nodes)
    probability = mass / np.sqrt(np.pi)
    query = radius * np.cos(np.sqrt(2.0) * sigma * locations)
    mean = float(probability @ query)
    return mean, float(probability @ (query - mean) ** 2)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "schema",
        "evidence_kind",
        "radius_m",
        "independent_readout_std_m",
        "wrapped_normal_std_radians",
        "primary_std_radians",
        "illustrative_std_screen_m",
        "shared_point_copies",
        "reference_hermite_nodes",
        "methods",
        "information_boundary",
        "claim_boundary",
    }
    if set(protocol) != required or protocol["schema"] != SCHEMA:
        raise ValueError("unknown protocol schema or fields")
    if protocol["evidence_kind"] != "designed-analytic-mechanism-control":
        raise ValueError("this runner is only an analytic mechanism control")
    boundary = protocol["information_boundary"]
    expected_boundary = {
        "real_provider_predictions",
        "source_outcomes",
        "target_outcomes",
        "bayesian_phystwin_execution",
        "causal4d_execution",
        "protected_cohort_access_authorized",
    }
    if not isinstance(boundary, dict) or set(boundary) != expected_boundary:
        raise ValueError("information boundary must be complete")
    if any(value is not False for value in boundary.values()):
        raise ValueError("this control cannot authorize real-data or downstream access")
    if protocol["methods"] != list(METHODS):
        raise ValueError("the complete declared method set is required")
    for key in ("radius_m", "independent_readout_std_m", "illustrative_std_screen_m"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    sigmas = protocol["wrapped_normal_std_radians"]
    if not isinstance(sigmas, list) or not sigmas:
        raise ValueError("a nonempty angular sweep is required")
    if any(
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(sigma)
        or sigma <= 0.0
        for sigma in sigmas
    ):
        raise ValueError("angular standard deviations must be finite and positive")
    if len(set(sigmas)) != len(sigmas) or protocol["primary_std_radians"] not in sigmas:
        raise ValueError("primary case must occur once in the declared sweep")
    for key, lower, upper in (
        ("shared_point_copies", 2, 10000), ("reference_hermite_nodes", 64, 256)
    ):
        value = protocol[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    if not isinstance(protocol["claim_boundary"], str) or not protocol["claim_boundary"].strip():
        raise ValueError("claim_boundary must be nonempty")


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    _validate_protocol(protocol)
    radius = float(protocol["radius_m"])
    noise_variance = float(protocol["independent_readout_std_m"]) ** 2
    orbit = AxialGaugeOrbit(axis=np.array([0.0, 0.0, 1.0]), pivot=np.zeros(3))
    point = np.array([[radius, 0.0, 0.0]])
    weights = np.array([[1.0, 0.0, 0.0]])
    cases: list[dict[str, Any]] = []
    for sigma_value in protocol["wrapped_normal_std_radians"]:
        sigma = float(sigma_value)
        angular = CircularMoments2.wrapped_normal(0.0, sigma * sigma)
        query = orbit.point_moments(point, angular).project(weights)
        truth_mean = float(query.mean[0])
        truth_variance = float(query.covariance[0, 0]) + noise_variance
        reference_mean, reference_variance = _hermite_radial(
            radius, sigma, protocol["reference_hermite_nodes"]
        )
        approximations = {
            "first-order": (radius, 0.0),
            "second-order": (radius * (1.0 - 0.5 * sigma**2), 0.5 * radius**2 * sigma**4),
            "spherical-radial-2": (radius * float(np.cos(sigma)), 0.0),
            "gauss-hermite-5": _hermite_radial(radius, sigma, 5),
            "gauss-hermite-32": _hermite_radial(radius, sigma, 32),
            "exact-circular-moments": (truth_mean, truth_variance - noise_variance),
        }
        rows = []
        for name in METHODS:
            mean, axial_variance = approximations[name]
            variance = axial_variance + noise_variance
            expected_squared_error = truth_variance + (truth_mean - mean) ** 2
            expected_score = 0.5 * (
                math.log(2.0 * math.pi * variance) + expected_squared_error / variance
            )
            rows.append(
                {
                    "method": name,
                    "query_mean_m": mean,
                    "query_std_m": math.sqrt(variance),
                    "axial_variance_m2": axial_variance,
                    "expected_gaussian_nll_nats_metre_coordinates": expected_score,
                    "expected_squared_standardized_error": expected_squared_error / variance,
                    "illustrative_std_screen_passes": math.sqrt(variance)
                    <= float(protocol["illustrative_std_screen_m"]),
                }
            )
        cases.append(
            {
                "angular_std_radians": sigma,
                "true_query_mean_m": truth_mean,
                "true_query_variance_m2": truth_variance,
                "local_angular_derivative_m_per_radian": 0.0,
                "full_orbit_bounds_m": query.full_orbit_bounds[0].tolist(),
                "reference_mean_absolute_error_m": abs(reference_mean - truth_mean),
                "reference_variance_absolute_error_m2": abs(
                    reference_variance + noise_variance - truth_variance
                ),
                "methods": rows,
            }
        )
    copies = protocol["shared_point_copies"]
    shared = orbit.point_moments(
        np.repeat(point, copies, axis=0),
        CircularMoments2.wrapped_normal(0.0, float(protocol["primary_std_radians"]) ** 2),
    )
    average_weights = np.zeros((copies, 3))
    average_weights[:, 0] = 1.0 / copies
    exact_average = float(shared.project(average_weights).covariance[0, 0])
    independent_average = float(np.sum(shared.marginal_covariance[:, 0, 0]) / copies**2)
    contrast = np.zeros((copies, 3))
    contrast[0, 0], contrast[1, 0] = 1.0, -1.0
    source_hashes = {
        "src/prob4d/axial_gauge_moments.py": hashlib.sha256(
            Path(axial_gauge_moments.__file__).read_bytes()
        ).hexdigest(),
        "src/prob4d/axial_gauge_moment_study.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    report: dict[str, Any] = {
        "schema": "prob4d.axial-gauge-moment-result.v1",
        "evidence_kind": protocol["evidence_kind"],
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "source_file_sha256": source_hashes,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "cases": cases,
        "shared_covariance_control": {
            "point_copies_not_independent_units": copies,
            "shared_average_axial_variance_m2": exact_average,
            "invalid_independent_average_axial_variance_m2": independent_average,
            "variance_understatement_factor": exact_average / independent_average,
            "identical_point_contrast_axial_variance_m2": float(
                shared.project(contrast).covariance[0, 0]
            ),
            "shared_factor_shape": list(shared.shared_factors.shape),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    report["artifact_id"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    protocol = json.loads(
        args.protocol.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    report = build_report(protocol)
    content = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a previous retained result is never overwritten.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print(report["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
