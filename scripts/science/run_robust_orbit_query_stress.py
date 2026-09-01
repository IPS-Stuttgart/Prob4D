#!/usr/bin/env python3
"""Run the registered approximate-orbit query-identifiability control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.robust_orbit_query import batch_axial_orbit_diameters

SCHEMA = "prob4d.robust-orbit-query-stress.v1"
RESULT_SCHEMA = "prob4d.robust-orbit-query-stress-result.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "evidence_kind",
        "seed",
        "case_count",
        "query_dimension",
        "stationary_case_fraction",
        "invariance_tolerance",
        "invariant_diameter_range",
        "variant_diameter_range",
        "coefficient_error_ratios",
        "sampled_grid_counts",
        "near_boundary_case_count",
        "near_boundary_variant_diameter_range",
        "benchmark",
        "registered_checks",
        "claim_boundary",
    }
    if type(protocol) is not dict or set(protocol) != expected:
        raise ValueError("protocol fields changed")
    if protocol["schema"] != SCHEMA:
        raise ValueError("protocol schema changed")
    if protocol["evidence_kind"] != "designed-bounded-error-robustness-control":
        raise ValueError("evidence kind changed")
    for name, lower in (
        ("case_count", 1000),
        ("query_dimension", 1),
        ("near_boundary_case_count", 1000),
    ):
        if type(protocol[name]) is not int or protocol[name] < lower:
            raise ValueError(f"{name} is invalid")
    if type(protocol["seed"]) is not int:
        raise ValueError("seed must be an integer")
    fraction = float(protocol["stationary_case_fraction"])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("stationary_case_fraction is invalid")
    tolerance = float(protocol["invariance_tolerance"])
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("invariance_tolerance is invalid")
    invariant = protocol["invariant_diameter_range"]
    variant = protocol["variant_diameter_range"]
    near = protocol["near_boundary_variant_diameter_range"]
    for name, interval in (
        ("invariant_diameter_range", invariant),
        ("variant_diameter_range", variant),
        ("near_boundary_variant_diameter_range", near),
    ):
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) for value in interval)
            or not 0.0 <= float(interval[0]) < float(interval[1])
        ):
            raise ValueError(f"{name} is invalid")
    if float(invariant[1]) >= tolerance or float(variant[0]) <= tolerance:
        raise ValueError("class intervals must be separated by the tolerance")
    if float(near[0]) <= tolerance:
        raise ValueError("near-boundary cases must be truly variant")
    ratios = protocol["coefficient_error_ratios"]
    if (
        not isinstance(ratios, list)
        or not ratios
        or float(ratios[0]) != 0.0
        or any(float(value) < 0.0 for value in ratios)
        or sorted(float(value) for value in ratios) != [float(value) for value in ratios]
        or len(set(float(value) for value in ratios)) != len(ratios)
    ):
        raise ValueError("coefficient_error_ratios is invalid")
    sample_counts = protocol["sampled_grid_counts"]
    if (
        not isinstance(sample_counts, list)
        or any(type(value) is not int or value < 3 or value % 2 == 0 for value in sample_counts)
        or sorted(sample_counts) != sample_counts
    ):
        raise ValueError("sampled_grid_counts must be increasing odd integers")
    benchmark = protocol["benchmark"]
    if set(benchmark) != {
        "query_count",
        "query_dimension",
        "angular_sample_count",
        "repetitions",
    }:
        raise ValueError("benchmark fields changed")
    if any(type(benchmark[name]) is not int or benchmark[name] <= 0 for name in benchmark):
        raise ValueError("benchmark values must be positive integers")
    checks = protocol["registered_checks"]
    if set(checks) != {
        "maximum_certified_harmful_acceptance_rate",
        "minimum_local_harmful_acceptance_rate_at_zero_error",
        "minimum_positive_error_naive_harmful_acceptance_rate",
        "maximum_sampled_certified_harmful_acceptance_rate",
        "minimum_coarse_naive_sampled_harmful_acceptance_rate",
        "maximum_factorized_dense_diameter_error",
    }:
        raise ValueError("registered checks changed")
    if not isinstance(protocol["claim_boundary"], str) or not protocol["claim_boundary"].strip():
        raise ValueError("claim_boundary is invalid")
    return protocol


def _unit_vectors(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    values = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(values, axis=1)
    while np.any(norms == 0.0):
        mask = norms == 0.0
        values[mask] = rng.normal(size=(int(np.count_nonzero(mask)), dimension))
        norms = np.linalg.norm(values, axis=1)
    return values / norms[:, None]


def _coefficient_bank(
    rng: np.random.Generator,
    diameters: np.ndarray,
    query_dimension: int,
    stationary_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = diameters.size
    stationary = rng.random(count) < stationary_fraction
    phase = rng.uniform(-np.pi, np.pi, size=count)
    coefficients = np.empty((count, query_dimension, 2), dtype=np.float64)

    stationary_count = int(np.count_nonzero(stationary))
    directions = _unit_vectors(rng, stationary_count, query_dimension)
    phase_vectors = np.column_stack((np.cos(phase[stationary]), np.sin(phase[stationary])))
    coefficients[stationary] = (
        0.5 * diameters[stationary, None, None] * directions[:, :, None] * phase_vectors[:, None, :]
    )

    general = ~stationary
    general_count = int(np.count_nonzero(general))
    raw = rng.normal(size=(general_count, query_dimension, 2))
    raw_sigma = batch_axial_orbit_diameters(raw) / 2.0
    while np.any(raw_sigma == 0.0):
        mask = raw_sigma == 0.0
        raw[mask] = rng.normal(size=(int(np.count_nonzero(mask)), query_dimension, 2))
        raw_sigma = batch_axial_orbit_diameters(raw) / 2.0
    coefficients[general] = raw * (0.5 * diameters[general] / raw_sigma)[:, None, None]
    nominal_angles = rng.uniform(-np.pi, np.pi, size=count)
    nominal_angles[stationary] = phase[stationary]
    return coefficients, nominal_angles, stationary


def _bounded_perturbations(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    error_bound: float,
) -> np.ndarray:
    if error_bound == 0.0:
        return np.zeros(shape, dtype=np.float64)
    raw = rng.normal(size=shape)
    sigma = batch_axial_orbit_diameters(raw) / 2.0
    magnitudes = rng.uniform(0.0, error_bound, size=shape[0])
    return raw * (magnitudes / sigma)[:, None, None]


def _gate_metrics(
    admitted: np.ndarray,
    true_invariant: np.ndarray,
    *,
    undetermined: np.ndarray | None = None,
) -> dict[str, Any]:
    true_variant = ~true_invariant
    if undetermined is None:
        undetermined = np.zeros(admitted.shape, dtype=bool)
    return {
        "admitted_count": int(np.count_nonzero(admitted)),
        "harmful_acceptance_count": int(np.count_nonzero(admitted & true_variant)),
        "harmful_acceptance_rate_among_variant": float(np.mean(admitted[true_variant])),
        "useful_acceptance_count": int(np.count_nonzero(admitted & true_invariant)),
        "useful_acceptance_rate_among_invariant": float(np.mean(admitted[true_invariant])),
        "fallback_count": int(np.count_nonzero(~admitted)),
        "fallback_rate": float(np.mean(~admitted)),
        "undetermined_count": int(np.count_nonzero(undetermined)),
        "undetermined_rate": float(np.mean(undetermined)),
    }


def _bounded_error_study(
    protocol: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = int(protocol["case_count"])
    tolerance = float(protocol["invariance_tolerance"])
    half = count // 2
    invariant_interval = protocol["invariant_diameter_range"]
    variant_interval = protocol["variant_diameter_range"]
    diameters = np.concatenate(
        (
            rng.uniform(*invariant_interval, size=half),
            rng.uniform(*variant_interval, size=count - half),
        )
    )
    permutation = rng.permutation(count)
    diameters = diameters[permutation]
    coefficients, angles, stationary = _coefficient_bank(
        rng,
        diameters,
        int(protocol["query_dimension"]),
        float(protocol["stationary_case_fraction"]),
    )
    true_invariant = diameters <= tolerance
    rows: list[dict[str, Any]] = []
    maximum_bound_violation = 0.0
    for ratio_value in protocol["coefficient_error_ratios"]:
        ratio = float(ratio_value)
        error_bound = ratio * tolerance
        perturbation = _bounded_perturbations(
            rng,
            coefficients.shape,
            error_bound,
        )
        estimated_coefficients = coefficients + perturbation
        estimated_diameter = batch_axial_orbit_diameters(estimated_coefficients)
        lower = np.maximum(0.0, estimated_diameter - 2.0 * error_bound)
        upper = estimated_diameter + 2.0 * error_bound
        maximum_bound_violation = max(
            maximum_bound_violation,
            float(np.max(np.maximum(lower - diameters, 0.0))),
            float(np.max(np.maximum(diameters - upper, 0.0))),
        )
        robust_admitted = upper <= tolerance
        robust_variant = lower > tolerance
        robust_undetermined = ~(robust_admitted | robust_variant)
        naive_admitted = estimated_diameter <= tolerance
        tangent = np.column_stack((-np.sin(angles), np.cos(angles)))
        derivative = np.einsum(
            "ndk,nk->nd",
            estimated_coefficients,
            tangent,
        )
        local_diameter_proxy = 2.0 * np.linalg.norm(derivative, axis=1)
        local_admitted = local_diameter_proxy <= tolerance
        for method, admitted, undetermined in (
            ("certified_bounded_error", robust_admitted, robust_undetermined),
            ("naive_estimated_diameter", naive_admitted, None),
            ("nominal_local_derivative", local_admitted, None),
        ):
            rows.append(
                {
                    "coefficient_error_ratio": ratio,
                    "coefficient_error_bound": error_bound,
                    "method": method,
                    **_gate_metrics(
                        admitted,
                        true_invariant,
                        undetermined=undetermined,
                    ),
                }
            )
    return rows, {
        "case_count": count,
        "invariant_case_count": int(np.count_nonzero(true_invariant)),
        "variant_case_count": int(np.count_nonzero(~true_invariant)),
        "stationary_case_count": int(np.count_nonzero(stationary)),
        "stationary_variant_case_count": int(np.count_nonzero(stationary & ~true_invariant)),
        "maximum_diameter_interval_violation": maximum_bound_violation,
    }


def _sampled_diameter(coefficients: np.ndarray, sample_count: int) -> np.ndarray:
    angles = np.arange(sample_count, dtype=np.float64) * (2.0 * np.pi / sample_count)
    harmonics = np.column_stack((np.cos(angles), np.sin(angles)))
    maximum_squared = np.zeros(coefficients.shape[0], dtype=np.float64)
    for first in range(sample_count):
        for second in range(first + 1, sample_count):
            difference = harmonics[first] - harmonics[second]
            query_difference = np.einsum("ndk,k->nd", coefficients, difference)
            maximum_squared = np.maximum(
                maximum_squared,
                np.sum(query_difference * query_difference, axis=1),
            )
    return np.sqrt(maximum_squared)


def _sampling_study(
    protocol: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    count = int(protocol["near_boundary_case_count"])
    tolerance = float(protocol["invariance_tolerance"])
    diameters = rng.uniform(
        *protocol["near_boundary_variant_diameter_range"],
        size=count,
    )
    coefficients, _, _ = _coefficient_bank(
        rng,
        diameters,
        int(protocol["query_dimension"]),
        float(protocol["stationary_case_fraction"]),
    )
    rows: list[dict[str, Any]] = []
    for sample_count in protocol["sampled_grid_counts"]:
        sampled = _sampled_diameter(coefficients, int(sample_count))
        lipschitz = diameters / 2.0
        cover_radius = np.pi / float(sample_count)
        certified_upper = sampled + 2.0 * lipschitz * cover_radius
        certified_admitted = certified_upper <= tolerance
        naive_admitted = sampled <= tolerance
        rows.extend(
            (
                {
                    "sample_count": int(sample_count),
                    "method": "lipschitz_certified_sampling",
                    "case_count": count,
                    "harmful_acceptance_count": int(np.count_nonzero(certified_admitted)),
                    "harmful_acceptance_rate": float(np.mean(certified_admitted)),
                    "fallback_rate": float(np.mean(~certified_admitted)),
                    "maximum_sampled_understatement": float(np.max(diameters - sampled)),
                    "maximum_certified_upper_shortfall": float(
                        np.max(np.maximum(diameters - certified_upper, 0.0))
                    ),
                },
                {
                    "sample_count": int(sample_count),
                    "method": "naive_sampled_diameter",
                    "case_count": count,
                    "harmful_acceptance_count": int(np.count_nonzero(naive_admitted)),
                    "harmful_acceptance_rate": float(np.mean(naive_admitted)),
                    "fallback_rate": float(np.mean(~naive_admitted)),
                    "maximum_sampled_understatement": float(np.max(diameters - sampled)),
                    "maximum_certified_upper_shortfall": None,
                },
            )
        )
    return rows


def _diameter_parity(rng: np.random.Generator) -> dict[str, Any]:
    coefficients = rng.normal(size=(4096, 5, 2))
    factorized = batch_axial_orbit_diameters(coefficients)
    dense = 2.0 * np.linalg.svd(coefficients, compute_uv=False)[:, 0]
    return {
        "case_count": int(coefficients.shape[0]),
        "maximum_absolute_error": float(np.max(np.abs(factorized - dense))),
        "mean_absolute_error": float(np.mean(np.abs(factorized - dense))),
    }


def _median_runtime(callable_object: Any, repetitions: int) -> tuple[float, Any]:
    timings = []
    result = None
    for _ in range(repetitions):
        gc.collect()
        start = time.perf_counter_ns()
        result = callable_object()
        timings.append((time.perf_counter_ns() - start) * 1e-6)
    return float(np.median(timings)), result


def _benchmark(
    protocol: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    settings = protocol["benchmark"]
    query_count = int(settings["query_count"])
    dimension = int(settings["query_dimension"])
    sample_count = int(settings["angular_sample_count"])
    repetitions = int(settings["repetitions"])
    coefficients = rng.normal(size=(query_count, dimension, 2))
    angles = np.arange(sample_count, dtype=np.float64) * (2.0 * np.pi / sample_count)
    harmonics = np.column_stack((np.cos(angles), np.sin(angles)))

    def exact() -> np.ndarray:
        return batch_axial_orbit_diameters(coefficients)

    def sampled() -> np.ndarray:
        values = np.einsum("ndk,sk->nsd", coefficients, harmonics)
        return 2.0 * np.max(np.linalg.norm(values, axis=-1), axis=1)

    exact_ms, exact_value = _median_runtime(exact, repetitions)
    sampled_ms, sampled_value = _median_runtime(sampled, repetitions)
    return {
        "query_count": query_count,
        "query_dimension": dimension,
        "angular_sample_count": sample_count,
        "repetitions": repetitions,
        "exact_vectorized_median_milliseconds": exact_ms,
        "sampled_orbit_median_milliseconds": sampled_ms,
        "sampled_to_exact_runtime_ratio": sampled_ms / exact_ms,
        "coefficient_storage_float64_bytes": int(coefficients.nbytes),
        "sampled_query_storage_float64_bytes": int(
            query_count * sample_count * dimension * np.dtype(np.float64).itemsize
        ),
        "sampled_to_coefficient_storage_ratio": float(sample_count / 2.0),
        "maximum_sampled_diameter_absolute_error": float(
            np.max(np.abs(sampled_value - exact_value))
        ),
    }


def _registered_checks(
    protocol: dict[str, Any],
    bounded_rows: list[dict[str, Any]],
    sampling_rows: list[dict[str, Any]],
    parity: dict[str, Any],
) -> dict[str, bool]:
    registered = protocol["registered_checks"]
    certified = [row for row in bounded_rows if row["method"] == "certified_bounded_error"]
    local_zero = next(
        row
        for row in bounded_rows
        if row["method"] == "nominal_local_derivative" and row["coefficient_error_ratio"] == 0.0
    )
    naive_positive = [
        row
        for row in bounded_rows
        if row["method"] == "naive_estimated_diameter" and row["coefficient_error_ratio"] > 0.0
    ]
    sampled_certified = [
        row for row in sampling_rows if row["method"] == "lipschitz_certified_sampling"
    ]
    sampled_naive = [row for row in sampling_rows if row["method"] == "naive_sampled_diameter"]
    return {
        "bounded_error_certificate_has_no_harmful_acceptance": max(
            row["harmful_acceptance_rate_among_variant"] for row in certified
        )
        <= float(registered["maximum_certified_harmful_acceptance_rate"]),
        "local_gate_fails_at_stationary_representatives": local_zero[
            "harmful_acceptance_rate_among_variant"
        ]
        > float(registered["minimum_local_harmful_acceptance_rate_at_zero_error"]),
        "naive_estimated_orbit_fails_under_nonzero_error": max(
            row["harmful_acceptance_rate_among_variant"] for row in naive_positive
        )
        > float(registered["minimum_positive_error_naive_harmful_acceptance_rate"]),
        "lipschitz_sampling_certificate_has_no_harmful_acceptance": max(
            row["harmful_acceptance_rate"] for row in sampled_certified
        )
        <= float(registered["maximum_sampled_certified_harmful_acceptance_rate"]),
        "uncertified_coarse_sampling_misses_variant_queries": sampled_naive[0][
            "harmful_acceptance_rate"
        ]
        > float(registered["minimum_coarse_naive_sampled_harmful_acceptance_rate"]),
        "closed_form_diameter_matches_dense_svd": parity["maximum_absolute_error"]
        <= float(registered["maximum_factorized_dense_diameter_error"]),
    }


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(protocol["seed"]))
    bounded_rows, population = _bounded_error_study(protocol, rng)
    sampling_rows = _sampling_study(protocol, rng)
    parity = _diameter_parity(rng)
    benchmark = _benchmark(protocol, rng)
    checks = _registered_checks(protocol, bounded_rows, sampling_rows, parity)
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
        "population": population,
        "bounded_error_rows": bounded_rows,
        "sampled_orbit_rows": sampling_rows,
        "diameter_parity": parity,
        "benchmark": benchmark,
        "registered_checks": checks,
        "decision": "passed" if all(checks.values()) else "failed",
        "claim_boundary": protocol["claim_boundary"],
    }
    report["artifact_id"] = _content_id(report)
    return report


def _summary(report: dict[str, Any]) -> str:
    bounded = report["bounded_error_rows"]
    zero_local = next(
        row
        for row in bounded
        if row["method"] == "nominal_local_derivative" and row["coefficient_error_ratio"] == 0.0
    )
    certified = [row for row in bounded if row["method"] == "certified_bounded_error"]
    lines = [
        "# Robust finite-orbit query-identifiability control",
        "",
        f"Artifact ID: `{report['artifact_id']}`",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "## Bounded coefficient error",
        "",
        "| Error/tolerance | Method | Harmful acceptance | Useful acceptance | Undetermined |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in bounded:
        lines.append(
            f"| {row['coefficient_error_ratio']:.2f} | {row['method']} | "
            f"{row['harmful_acceptance_rate_among_variant']:.4f} | "
            f"{row['useful_acceptance_rate_among_invariant']:.4f} | "
            f"{row['undetermined_rate']:.4f} |"
        )
    lines += [
        "",
        (
            "The certified gate had maximum harmful acceptance "
            f"{max(row['harmful_acceptance_rate_among_variant'] for row in certified):.4f}. "
            "At zero coefficient error, the nominal local-derivative gate accepted "
            f"{zero_local['harmful_acceptance_rate_among_variant']:.4f} of truly variant queries."
        ),
        "",
        "## Uniform orbit sampling",
        "",
        "| Samples | Method | Harmful acceptance | Maximum diameter understatement |",
        "|---:|---|---:|---:|",
    ]
    for row in report["sampled_orbit_rows"]:
        lines.append(
            f"| {row['sample_count']} | {row['method']} | "
            f"{row['harmful_acceptance_rate']:.4f} | "
            f"{row['maximum_sampled_understatement']:.6g} |"
        )
    benchmark = report["benchmark"]
    lines += [
        "",
        "## Computational control",
        "",
        (
            f"For {benchmark['query_count']} {benchmark['query_dimension']}-D queries, "
            f"the exact two-column certificate took a median "
            f"{benchmark['exact_vectorized_median_milliseconds']:.3f} ms versus "
            f"{benchmark['sampled_orbit_median_milliseconds']:.3f} ms for "
            f"{benchmark['angular_sample_count']} orbit samples "
            f"({benchmark['sampled_to_exact_runtime_ratio']:.1f}x runtime ratio)."
        ),
        (
            "Materializing the sampled query bank requires "
            f"{benchmark['sampled_to_coefficient_storage_ratio']:.1f}x the storage "
            "of the exact two-column coefficients."
        ),
        "",
        report["claim_boundary"],
    ]
    return "\n".join(lines) + "\n"


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
    args.output.with_suffix(".md").write_text(_summary(report), encoding="utf-8")
    print(json.dumps({"artifact_id": report["artifact_id"], "decision": report["decision"]}))
    return 0 if report["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
