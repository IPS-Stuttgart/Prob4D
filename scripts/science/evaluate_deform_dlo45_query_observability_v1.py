"""Evaluate query-aware partial-gauge use on held-out real DEFORM geometry.

The registered experiment opens the 28 official DLO4/DLO5 evaluation
trajectories exactly once after freezing the rank threshold, local support,
query definitions, prior, noise model, and query gate on the training split.
Real trajectories supply geometry and motion states; known Sim(3) gauges and
correspondence noise provide auditable ground truth.  This is controlled-gauge
real-geometry evidence, not learned-provider competence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import audit_deform_dlo45_observability_v1 as base
from prob4d.observable_gauge import (
    GaugeGaussianPosterior,
    estimate_observable_sim3_factor,
)
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
    point_position_query_jacobian,
)
from prob4d.sim3 import Sim3, so3_exp

SCHEMA = "prob4d.deform-dlo45-query-observability-heldout-evaluation"
SCHEMA_VERSION = 1
REQUEST_SCHEMA = "prob4d.deform-dlo45-query-observability-evaluation-request"
CHI2_3_90 = 6.251388631170325
METHODS = (
    "physical_fallback",
    "full_rank_only",
    "observable_subspace_unconditional",
    "query_aware",
    "invalid_full_rank_completion",
)
QUERIES = ("segment_centroid", "off_axis_probe")


@dataclass
class MetricAccumulator:
    count: int = 0
    squared_coordinate_error_sum: float = 0.0
    nll_sum: float = 0.0
    nees_sum: float = 0.0
    covered_90_count: int = 0
    width_sum_m: float = 0.0
    harmful_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    exact_fallback_count: int = 0

    def add(
        self,
        *,
        error: np.ndarray,
        covariance: np.ndarray,
        harmful: bool,
        accepted: bool,
        exact_fallback: bool,
    ) -> None:
        covariance = 0.5 * (covariance + covariance.T)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("query covariance must be positive definite")
        precision_error = np.linalg.solve(covariance, error)
        nees = float(error @ precision_error)
        if not np.isfinite(nees):
            raise ValueError("query NEES must be finite")
        nll = 0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + nees)
        self.count += 1
        self.squared_coordinate_error_sum += float(np.sum(error**2))
        self.nll_sum += nll
        self.nees_sum += nees
        self.covered_90_count += int(nees <= CHI2_3_90)
        self.width_sum_m += float(np.sqrt(np.trace(covariance) / 3.0))
        self.harmful_count += int(harmful)
        self.accepted_count += int(accepted)
        self.rejected_count += int(not accepted)
        self.exact_fallback_count += int(exact_fallback)

    def finalize(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("cannot finalize an empty metric accumulator")
        return {
            "count": self.count,
            "rmse_mm": float(
                1000.0
                * np.sqrt(self.squared_coordinate_error_sum / (3.0 * self.count))
            ),
            "mean_gaussian_nll": self.nll_sum / self.count,
            "normalized_nees": self.nees_sum / (3.0 * self.count),
            "empirical_90pct_coverage": self.covered_90_count / self.count,
            "mean_marginal_standard_deviation_mm": (
                1000.0 * self.width_sum_m / self.count
            ),
            "harmful_fraction_vs_fallback": self.harmful_count / self.count,
            "accepted_fraction": self.accepted_count / self.count,
            "rejected_fraction": self.rejected_count / self.count,
            "exact_fallback_fraction": self.exact_fallback_count / self.count,
        }


def load_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != 1:
        raise ValueError("unsupported request schema")
    supplied = request.get("request_id")
    unhashed = dict(request)
    unhashed.pop("request_id", None)
    if supplied != base.canonical_sha256(unhashed):
        raise ValueError("request_id does not match canonical request contents")
    if request.get("stage") != "heldout-evaluation":
        raise ValueError("unexpected stage")
    if Path(str(request.get("dataset_root"))) != base.EXPECTED_ROOT:
        raise ValueError(f"dataset_root must be exactly {base.EXPECTED_ROOT}")
    if request.get("dlo_types") != ["DLO4", "DLO5"]:
        raise ValueError("dlo_types must be exactly DLO4 and DLO5")
    if request.get("segment_length") != 4:
        raise ValueError("segment_length must be exactly four")
    if float(request.get("rank_threshold")) != 0.01:
        raise ValueError("rank_threshold must equal the frozen value 0.01")
    gate = request.get("query_gate")
    if gate != {
        "minimum_direct_observability_fraction": 0.9,
        "minimum_metric_variance_reduction_fraction": 0.0,
        "maximum_worst_supported_variance_ratio": 1.0,
    }:
        raise ValueError("query gate differs from the source-frozen gate")
    expected_boundary = {
        "opened_split": "eval",
        "evaluation_outcomes_opened": True,
        "source_gate_frozen_before_opening": True,
        "provider_predictions_opened": False,
        "bayesian_phystwin_outcomes_opened": False,
        "causal4d_outcomes_opened": False,
        "post_open_retuning_permitted": False,
    }
    if request.get("information_boundary") != expected_boundary:
        raise ValueError("information boundary changed")
    return request


def deterministic_normal(segment: np.ndarray) -> np.ndarray:
    tangent = segment[-1] - segment[0]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= np.finfo(np.float64).eps:
        raise ValueError("segment endpoints coincide")
    tangent /= tangent_norm
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ tangent)))]
    normal = np.cross(tangent, reference)
    normal /= np.linalg.norm(normal)
    return normal


def truth_transform(
    segment: np.ndarray,
    generator: np.random.Generator,
    request: dict[str, Any],
) -> tuple[Sim3, float]:
    centroid = np.mean(segment, axis=0)
    tangent = segment[-1] - segment[0]
    tangent /= np.linalg.norm(tangent)
    twist_min, twist_max = (
        float(value) for value in request["absolute_twist_range_rad"]
    )
    sign = -1.0 if generator.random() < 0.5 else 1.0
    twist = sign * generator.uniform(twist_min, twist_max)
    base_rotation = so3_exp(
        generator.normal(
            scale=float(request["base_rotation_std_rad"]),
            size=3,
        )
    )
    rotation = base_rotation @ so3_exp(tangent * twist)
    scale = float(
        np.exp(generator.normal(scale=float(request["log_scale_std"])))
    )
    target_centroid = centroid + generator.normal(
        scale=float(request["translation_std_m"]),
        size=3,
    )
    translation = target_centroid - scale * (rotation @ centroid)
    return Sim3(scale=scale, rotation=rotation, translation=translation), twist


def fuse_information(
    factor,
    prior_mean_local: np.ndarray,
    prior_covariance_local: np.ndarray,
    information: np.ndarray,
) -> GaugeGaussianPosterior:
    prior_information = np.linalg.solve(prior_covariance_local, np.eye(7))
    posterior_information = prior_information + information
    posterior_covariance = np.linalg.solve(posterior_information, np.eye(7))
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    posterior_mean = posterior_covariance @ (prior_information @ prior_mean_local)
    return GaugeGaussianPosterior(
        chart=factor.chart,
        mean_local=posterior_mean,
        covariance_local=posterior_covariance,
    )


def query_prediction(
    factor,
    query_source: np.ndarray,
    posterior: GaugeGaussianPosterior,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = point_position_query_jacobian(factor, query_source)
    mean = np.asarray(
        posterior.mean_transform.transform_points(query_source),
        dtype=np.float64,
    )
    covariance = jacobian @ posterior.covariance_local @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("projected query covariance is not positive definite")
    return mean, covariance


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def mean_ci(
    values: list[float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
    }


def aggregate_groups(
    group_rows: dict[str, dict[str, dict[str, dict[str, float | int]]]],
    request: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    bootstrap_replicates = int(request["bootstrap_replicates"])
    bootstrap_seed = int(request["bootstrap_seed"])
    for query in QUERIES:
        result[query] = {}
        for method in METHODS:
            rows = [
                methods[method]
                for queries in group_rows.values()
                for query_name, methods in queries.items()
                if query_name == query
            ]
            if len(rows) != len(group_rows):
                raise ValueError("every evaluation group must contribute every method/query")
            scalar_names = (
                "rmse_mm",
                "mean_gaussian_nll",
                "normalized_nees",
                "empirical_90pct_coverage",
                "mean_marginal_standard_deviation_mm",
                "harmful_fraction_vs_fallback",
                "accepted_fraction",
                "exact_fallback_fraction",
            )
            aggregate: dict[str, Any] = {
                "independent_groups": len(rows),
                "nested_cases": int(sum(int(row["count"]) for row in rows)),
            }
            for scalar_name in scalar_names:
                values = [float(row[scalar_name]) for row in rows]
                aggregate[scalar_name] = mean_ci(
                    values,
                    seed=stable_seed(
                        bootstrap_seed,
                        f"{query}/{method}/{scalar_name}",
                    ),
                    replicates=bootstrap_replicates,
                )
            fallback_rows = [
                queries[query]["physical_fallback"]
                for queries in group_rows.values()
            ]
            rmse_differences = [
                float(fallback["rmse_mm"]) - float(row["rmse_mm"])
                for fallback, row in zip(fallback_rows, rows, strict=True)
            ]
            nll_differences = [
                float(fallback["mean_gaussian_nll"])
                - float(row["mean_gaussian_nll"])
                for fallback, row in zip(fallback_rows, rows, strict=True)
            ]
            aggregate["paired_rmse_improvement_mm"] = mean_ci(
                rmse_differences,
                seed=stable_seed(bootstrap_seed, f"{query}/{method}/paired-rmse"),
                replicates=bootstrap_replicates,
            )
            aggregate["paired_nll_improvement"] = mean_ci(
                nll_differences,
                seed=stable_seed(bootstrap_seed, f"{query}/{method}/paired-nll"),
                replicates=bootstrap_replicates,
            )
            result[query][method] = aggregate
    return result


def run(request: dict[str, Any]) -> dict[str, Any]:
    root = base.EXPECTED_ROOT
    segment_length = int(request["segment_length"])
    frame_stride = int(request["frame_stride"])
    rank_threshold = float(request["rank_threshold"])
    noise_sigma = float(request["correspondence_noise_sigma_m"])
    prior_std = np.asarray(request["prior_standard_deviations_local"], dtype=np.float64)
    prior_covariance = np.diag(prior_std**2)
    invalid_ratio = float(request["invalid_nullspace_precision_ratio"])
    probe_factor = float(request["probe_radius_cloud_scale_factor"])
    seed = int(request["experiment_seed"])
    generator = np.random.default_rng(seed)
    gate = QueryObservabilityGate(**request["query_gate"])

    manifest_rows: list[dict[str, Any]] = []
    group_accumulators: dict[
        str, dict[str, dict[str, MetricAccumulator]]
    ] = defaultdict(
        lambda: {
            query: {method: MetricAccumulator() for method in METHODS}
            for query in QUERIES
        }
    )
    direct_fractions: dict[str, list[float]] = {query: [] for query in QUERIES}
    factor_rank_counts: Counter[int] = Counter()
    fit_failures: Counter[str] = Counter()
    geometry_rank_six_preselected = 0
    successful_cases = 0
    twist_magnitudes: list[float] = []
    cases_by_object: Counter[str] = Counter()

    for dlo_type in request["dlo_types"]:
        directory = root / dlo_type / "eval"
        files = sorted(directory.glob("*.pkl"), key=lambda item: int(item.stem))
        if len(files) != 14:
            raise ValueError(
                f"expected 14 official evaluation files for {dlo_type}, found {len(files)}"
            )
        for path in files:
            group_id = f"{dlo_type}/{path.name}"
            manifest_rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": base.sha256_file(path),
                    "independent_group": group_id,
                }
            )
            frames = base.load_trajectory(path)
            for frame_index in range(0, frames.shape[0], frame_stride):
                frame = frames[frame_index]
                for start in range(frame.shape[0] - segment_length + 1):
                    segment = frame[start : start + segment_length]
                    spectrum, _, _ = base.geometry_spectrum(segment)
                    if int(np.count_nonzero(spectrum >= rank_threshold)) != 6:
                        continue
                    geometry_rank_six_preselected += 1
                    truth, twist = truth_transform(segment, generator, request)
                    target = truth.transform_points(segment) + generator.normal(
                        scale=noise_sigma,
                        size=segment.shape,
                    )
                    try:
                        factor = estimate_observable_sim3_factor(
                            segment,
                            target,
                            rank_threshold=rank_threshold,
                        )
                    except (ValueError, np.linalg.LinAlgError) as error:
                        fit_failures[type(error).__name__] += 1
                        continue
                    factor_rank_counts[factor.rank] += 1
                    if factor.rank != 6:
                        continue
                    truth_local = factor.chart.to_local(truth)
                    prior_mean_local = truth_local + generator.multivariate_normal(
                        np.zeros(7),
                        prior_covariance,
                    )
                    fallback = GaugeGaussianPosterior(
                        chart=factor.chart,
                        mean_local=prior_mean_local,
                        covariance_local=prior_covariance,
                    )
                    observable = factor.fuse_local_gaussian(
                        prior_mean_local,
                        prior_covariance,
                    )
                    weakest_observable_precision = float(
                        np.min(np.linalg.eigvalsh(factor.observable_information))
                    )
                    invalid_information = factor.information_matrix.copy()
                    invalid_information += (
                        invalid_ratio
                        * weakest_observable_precision
                        * factor.nullspace_basis
                        @ factor.nullspace_basis.T
                    )
                    invalid = fuse_information(
                        factor,
                        prior_mean_local,
                        prior_covariance,
                        invalid_information,
                    )
                    centroid = np.mean(segment, axis=0)
                    probe = (
                        centroid
                        + probe_factor
                        * factor.chart.cloud_scale
                        * deterministic_normal(segment)
                    )
                    query_points = {
                        "segment_centroid": centroid,
                        "off_axis_probe": probe,
                    }
                    successful_cases += 1
                    cases_by_object[dlo_type] += 1
                    twist_magnitudes.append(abs(twist))

                    for query_name, query_source in query_points.items():
                        jacobian = point_position_query_jacobian(factor, query_source)
                        report = evaluate_query_observability(
                            factor,
                            prior_covariance_local=prior_covariance,
                            query_jacobian_local=jacobian,
                        )
                        decision = gate.evaluate(report)
                        direct_fractions[query_name].append(
                            report.direct_observability_fraction
                        )
                        true_query = np.asarray(
                            truth.transform_points(query_source),
                            dtype=np.float64,
                        )
                        fallback_mean, fallback_covariance = query_prediction(
                            factor,
                            query_source,
                            fallback,
                        )
                        fallback_error = true_query - fallback_mean
                        fallback_error_norm = float(np.linalg.norm(fallback_error))

                        method_posteriors = {
                            "physical_fallback": fallback,
                            "full_rank_only": fallback,
                            "observable_subspace_unconditional": observable,
                            "query_aware": observable if decision.admitted else fallback,
                            "invalid_full_rank_completion": invalid,
                        }
                        for method, posterior in method_posteriors.items():
                            mean, covariance = query_prediction(
                                factor,
                                query_source,
                                posterior,
                            )
                            error = true_query - mean
                            harmful = (
                                float(np.linalg.norm(error))
                                > fallback_error_norm + 1e-12
                            )
                            if method == "physical_fallback":
                                harmful = False
                            accepted = method not in {
                                "physical_fallback",
                                "full_rank_only",
                            }
                            if method == "query_aware":
                                accepted = decision.admitted
                            exact_fallback = False
                            if method in {"physical_fallback", "full_rank_only"}:
                                exact_fallback = True
                            elif method == "query_aware" and not decision.admitted:
                                exact_fallback = bool(
                                    np.array_equal(mean, fallback_mean)
                                    and np.array_equal(covariance, fallback_covariance)
                                )
                                if not exact_fallback:
                                    raise AssertionError(
                                        "rejected query-aware update did not reproduce fallback"
                                    )
                            group_accumulators[group_id][query_name][method].add(
                                error=error,
                                covariance=covariance,
                                harmful=harmful,
                                accepted=accepted,
                                exact_fallback=exact_fallback,
                            )

    expected_groups = 28
    nonempty_groups = {
        group_id
        for group_id, queries in group_accumulators.items()
        if all(
            queries[query][method].count > 0
            for query in QUERIES
            for method in METHODS
        )
    }
    if len(nonempty_groups) != expected_groups:
        missing = sorted(
            set(row["independent_group"] for row in manifest_rows) - nonempty_groups
        )
        raise ValueError(f"not every official evaluation file contributed cases: {missing}")

    group_rows: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for group_id in sorted(nonempty_groups):
        group_rows[group_id] = {
            query: {
                method: group_accumulators[group_id][query][method].finalize()
                for method in METHODS
            }
            for query in QUERIES
        }
    aggregate = aggregate_groups(group_rows, request)

    manifest: dict[str, Any] = {
        "files": sorted(manifest_rows, key=lambda row: row["path"]),
        "file_count": len(manifest_rows),
        "total_bytes": int(sum(row["bytes"] for row in manifest_rows)),
    }
    manifest["manifest_sha256"] = base.canonical_sha256(manifest)

    def metric(query: str, method: str, name: str) -> float:
        return float(aggregate[query][method][name]["mean"])

    probe_rejections = 1.0 - metric(
        "off_axis_probe", "query_aware", "accepted_fraction"
    )
    criteria = {
        "all_28_official_evaluation_groups_contributed": len(group_rows) == 28,
        "at_least_1000_rank_six_cases": successful_cases >= 1000,
        "centroid_acceptance_at_least_99pct": metric(
            "segment_centroid", "query_aware", "accepted_fraction"
        )
        >= 0.99,
        "off_axis_rejection_at_least_90pct": probe_rejections >= 0.90,
        "rejected_query_aware_updates_are_exact_fallback": metric(
            "off_axis_probe", "query_aware", "exact_fallback_fraction"
        )
        >= probe_rejections - 1e-12,
        "centroid_query_aware_rmse_beats_fallback": metric(
            "segment_centroid", "query_aware", "rmse_mm"
        )
        < metric("segment_centroid", "physical_fallback", "rmse_mm"),
        "centroid_query_aware_nll_beats_fallback": metric(
            "segment_centroid", "query_aware", "mean_gaussian_nll"
        )
        < metric("segment_centroid", "physical_fallback", "mean_gaussian_nll"),
        "query_aware_off_axis_harmful_acceptance_at_most_1pct": metric(
            "off_axis_probe", "query_aware", "harmful_fraction_vs_fallback"
        )
        <= 0.01,
    }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": request["stage"],
        "request_id": request["request_id"],
        "evidence_class": "heldout-public-real-geometry-controlled-gauge-evaluation",
        "source_freeze": {
            "source_geometry_result_id": request["source_geometry_result_id"],
            "source_gate_result_id": request["source_gate_result_id"],
            "source_manifest_sha256": request["source_manifest_sha256"],
            "rank_threshold": rank_threshold,
            "query_gate": request["query_gate"],
        },
        "dataset": {
            "name": "DEFORM",
            "objects": request["dlo_types"],
            "opened_split": "eval",
            "manifest": manifest,
            "independent_group_definition": "one official DLO evaluation trajectory file",
        },
        "design": {
            "segment_length": segment_length,
            "frame_stride": frame_stride,
            "correspondence_noise_sigma_m": noise_sigma,
            "absolute_twist_range_rad": request["absolute_twist_range_rad"],
            "base_rotation_std_rad": request["base_rotation_std_rad"],
            "log_scale_std": request["log_scale_std"],
            "translation_std_m": request["translation_std_m"],
            "probe_radius_cloud_scale_factor": probe_factor,
            "prior_standard_deviations_local": prior_std.tolist(),
            "invalid_nullspace_precision_ratio": invalid_ratio,
            "experiment_seed": seed,
            "bootstrap_replicates": request["bootstrap_replicates"],
            "bootstrap_seed": request["bootstrap_seed"],
        },
        "accounting": {
            "geometry_rank_six_preselected": geometry_rank_six_preselected,
            "successful_rank_six_cases": successful_cases,
            "factor_rank_counts": {
                str(rank): count for rank, count in sorted(factor_rank_counts.items())
            },
            "fit_failures": dict(sorted(fit_failures.items())),
            "cases_by_object": dict(sorted(cases_by_object.items())),
            "independent_groups": len(group_rows),
            "mean_absolute_controlled_twist_rad": float(
                np.mean(twist_magnitudes)
            ),
        },
        "direct_observability": {
            query: base.quantiles(values)
            for query, values in direct_fractions.items()
        },
        "aggregate_equal_group_results": aggregate,
        "per_group_results": group_rows,
        "criteria": criteria,
        "decision": "pass" if all(criteria.values()) else "bounded-or-negative",
        "information_boundary": request["information_boundary"],
        "claim_boundary": [
            "The 28 official DLO4/DLO5 evaluation trajectories are held-out real geometry and independent file-level groups.",
            "Known controlled Sim(3) gauges and correspondence noise provide exact ground truth on those real geometries.",
            "The experiment does not establish learned visual-provider competence or end-to-end BayesianPhysTwin/Causal4D benefit.",
            "No threshold, support, query, prior, comparator, or seed may be retuned on this opened evaluation result.",
        ],
    }
    result["result_id"] = base.canonical_sha256(result)
    return result


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Held-out DEFORM DLO4/DLO5 query-observability evaluation",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Decision: **{result['decision']}**",
        f"- Independent evaluation files: `{result['accounting']['independent_groups']}`",
        f"- Rank-six cases: `{result['accounting']['successful_rank_six_cases']}`",
        "",
        "## Equal-file aggregate results",
        "",
    ]
    for query in QUERIES:
        lines.append(f"### {query}")
        lines.append("")
        lines.append(
            "| Method | RMSE [mm] | Gaussian NLL | 90% coverage | norm. NEES | "
            "accepted | harmful vs fallback |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            row = result["aggregate_equal_group_results"][query][method]
            lines.append(
                f"| {method} | {row['rmse_mm']['mean']:.3f} | "
                f"{row['mean_gaussian_nll']['mean']:.3f} | "
                f"{row['empirical_90pct_coverage']['mean']:.3f} | "
                f"{row['normalized_nees']['mean']:.3f} | "
                f"{row['accepted_fraction']['mean']:.3f} | "
                f"{row['harmful_fraction_vs_fallback']['mean']:.3f} |"
            )
        lines.append("")
    lines.extend(["## Registered criteria", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in result["criteria"].items()
    )
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    request = load_request(args.request)
    result = run(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_summary(result, args.output_dir / "summary.md")
    print(json.dumps({"result_id": result["result_id"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
