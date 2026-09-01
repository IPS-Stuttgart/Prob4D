#!/usr/bin/env python3
"""Calibrate DLO4/DLO5 query covariance using existing source files only.

The source stage opens the 112 official training trajectories and freezes one
scalar inflation for the accepted segment-centroid covariance. The target stage
reproduces the earlier 28-file evaluation and applies only that frozen scalar.
Means, factors, admission decisions, and rejected-query fallback do not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import audit_deform_dlo45_observability_v1 as base
import evaluate_deform_dlo45_query_observability_v1 as original
import numpy as np

PROTOCOL_SCHEMA = "prob4d.deform-dlo45-source-calibrated-query-covariance.v1"
REQUEST_SCHEMA = "prob4d.deform-dlo45-source-calibrated-query-covariance-request.v1"
CALIBRATION_SCHEMA = "prob4d.deform-dlo45-query-covariance-calibration.v1"
RESULT_SCHEMA = "prob4d.deform-dlo45-query-covariance-evaluation.v1"
CHI2_3_90 = 6.251388631170325
QUERIES = ("segment_centroid", "off_axis_probe")
METHODS = ("physical_fallback", "raw_query_aware", "source_calibrated_query_aware")


class CapturingAccumulator(original.MetricAccumulator):
    """Retain per-case NEES while preserving the original finalized metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.nees_values: list[float] = []

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
        self.nees_values.append(float(error @ np.linalg.solve(covariance, error)))
        super().add(
            error=error,
            covariance=covariance,
            harmful=harmful,
            accepted=accepted,
            exact_fallback=exact_fallback,
        )

    def finalize(self) -> dict[str, float | int | list[float]]:
        result = super().finalize()
        result["_nees_values"] = self.nees_values
        return result


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: dict[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unsupported protocol")
    if protocol.get("protocol_id") != content_id(protocol, "protocol_id"):
        raise ValueError("protocol_id mismatch")
    dataset = protocol["dataset"]
    if dataset["dlo_types"] != ["DLO4", "DLO5"]:
        raise ValueError("DLO roster changed")
    if dataset["source_split"] != "train" or dataset["target_split"] != "eval":
        raise ValueError("train/eval split changed")
    if dataset["expected_source_files_per_type"] != 56:
        raise ValueError("expected 56 training files per family")
    if dataset["expected_target_files_per_type"] != 14:
        raise ValueError("expected 14 evaluation files per family")
    calibration = protocol["calibration"]
    expected = {
        "chi_square_3d_90": CHI2_3_90,
        "minimum_inflation": 1.0,
        "query": "segment_centroid",
        "rule": "max(moment_nees_factor,equal_group_weighted_coverage_factor,1)",
        "target_coverage": 0.9,
        "weighting": "equal trajectory weight; equal case weight within trajectory",
    }
    if calibration != expected:
        raise ValueError("calibration rule changed")
    if protocol["information_boundary"]["new_data_collection_authorized"] is not False:
        raise ValueError("new data collection must remain forbidden")
    if protocol["information_boundary"]["target_split_previously_opened"] is not True:
        raise ValueError("post-hoc target status must remain explicit")


def validate_request(request: dict[str, Any], protocol: dict[str, Any]) -> None:
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != 1:
        raise ValueError("unsupported request")
    if request.get("request_id") != content_id(request, "request_id"):
        raise ValueError("request_id mismatch")
    if request.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("request/protocol mismatch")
    required = {
        "execution_authorized": True,
        "new_data_collection_authorized": False,
        "source_calibration_authorized": True,
        "existing_eval_reanalysis_authorized": True,
    }
    for key, expected in required.items():
        if request.get(key) is not expected:
            raise ValueError(f"request field {key} changed")


def original_request(protocol: dict[str, Any], *, seed: int) -> dict[str, Any]:
    design = protocol["design"]
    return {
        "dlo_types": ["DLO4", "DLO5"],
        "segment_length": design["segment_length"],
        "frame_stride": design["frame_stride"],
        "rank_threshold": design["rank_threshold"],
        "correspondence_noise_sigma_m": design["correspondence_noise_sigma_m"],
        "absolute_twist_range_rad": design["absolute_twist_range_rad"],
        "base_rotation_std_rad": design["base_rotation_std_rad"],
        "log_scale_std": design["log_scale_std"],
        "translation_std_m": design["translation_std_m"],
        "probe_radius_cloud_scale_factor": design["probe_radius_cloud_scale_factor"],
        "prior_standard_deviations_local": design["prior_standard_deviations_local"],
        "invalid_nullspace_precision_ratio": 1.0,
        "experiment_seed": seed,
        "bootstrap_replicates": 10,
        "bootstrap_seed": seed,
        "query_gate": protocol["query_gate"],
        "source_geometry_result_id": "source-calibration",
        "source_gate_result_id": "source-calibration",
        "source_manifest_sha256": "source-calibration",
        "stage": "heldout-evaluation",
        "information_boundary": {
            "opened_split": "eval",
            "evaluation_outcomes_opened": True,
            "source_gate_frozen_before_opening": True,
            "provider_predictions_opened": False,
            "bayesian_phystwin_outcomes_opened": False,
            "causal4d_outcomes_opened": False,
            "post_open_retuning_permitted": False,
        },
        "request_id": "internal-existing-data-reanalysis",
    }


@contextmanager
def patched_original_root(root: Path) -> Iterator[None]:
    old_root = base.EXPECTED_ROOT
    old_accumulator = original.MetricAccumulator
    base.EXPECTED_ROOT = root
    original.MetricAccumulator = CapturingAccumulator
    try:
        yield
    finally:
        base.EXPECTED_ROOT = old_root
        original.MetricAccumulator = old_accumulator


def run_original(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    with patched_original_root(root):
        return original.run(request)


def family_files(root: Path, family: str, split: str, count: int) -> list[Path]:
    paths = sorted((root / family / split).glob("*.pkl"), key=lambda path: int(path.stem))
    if len(paths) != count:
        raise ValueError(f"expected {count} {family}/{split} files, found {len(paths)}")
    return paths


def source_fold_root(dataset_root: Path, fold: int) -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory(prefix=f"dlo45-source-fold-{fold}-")
    root = Path(temporary.name)
    for family in ("DLO4", "DLO5"):
        destination = root / family / "eval"
        destination.mkdir(parents=True)
        selected = family_files(dataset_root, family, "train", 56)[14 * fold : 14 * (fold + 1)]
        for source in selected:
            (destination / source.name).symlink_to(source.resolve(strict=True))
    return temporary


def manifest(root: Path, split: str, expected: int) -> dict[str, Any]:
    rows = []
    for family in ("DLO4", "DLO5"):
        for path in family_files(root, family, split, expected):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": base.sha256_file(path),
                }
            )
    result: dict[str, Any] = {
        "split": split,
        "file_count": len(rows),
        "total_bytes": int(sum(row["bytes"] for row in rows)),
        "files": rows,
    }
    result["manifest_sha256"] = base.canonical_sha256(result)
    return result


def equal_group_weighted_quantile(
    values_by_group: dict[str, list[float]], quantile: float
) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    if not values_by_group or any(not values for values in values_by_group.values()):
        raise ValueError("all groups must contain scores")
    group_weight = 1.0 / len(values_by_group)
    weighted = []
    for values in values_by_group.values():
        case_weight = group_weight / len(values)
        weighted.extend((float(value), case_weight) for value in values)
    cumulative = 0.0
    for value, weight in sorted(weighted):
        cumulative += weight
        if cumulative + 1e-15 >= quantile:
            return value
    return max(value for value, _ in weighted)


def source_calibrate(
    dataset_root: Path,
    protocol: dict[str, Any],
    request: dict[str, Any],
    revision: str,
) -> dict[str, Any]:
    scores: dict[str, list[float]] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}
    source_seed = int(protocol["design"]["source_seed"])
    for fold in range(4):
        temporary = source_fold_root(dataset_root, fold)
        try:
            result = run_original(
                Path(temporary.name),
                original_request(protocol, seed=source_seed),
            )
        finally:
            temporary.cleanup()
        for group, queries in result["per_group_results"].items():
            row = queries["segment_centroid"]["query_aware"]
            if float(row["accepted_fraction"]) != 1.0:
                raise ValueError(f"source centroid was not admitted for {group}")
            values = [float(value) for value in row["_nees_values"]]
            scores[group] = values
            raw_metrics[group] = row
    if len(scores) != 112:
        raise ValueError(f"expected 112 source groups, found {len(scores)}")

    group_normalized_nees = [float(np.mean(values)) / 3.0 for values in scores.values()]
    moment = float(np.mean(group_normalized_nees))
    coverage = equal_group_weighted_quantile(scores, 0.9) / CHI2_3_90
    inflation = max(1.0, moment, coverage)

    calibrated_coverage = []
    calibrated_nees = []
    calibrated_nll = []
    raw_coverage = []
    raw_nll = []
    for group, row in raw_metrics.items():
        values = np.asarray(scores[group], dtype=np.float64)
        raw_coverage.append(float(row["empirical_90pct_coverage"]))
        raw_nll.append(float(row["mean_gaussian_nll"]))
        calibrated_coverage.append(float(np.mean(values <= CHI2_3_90 * inflation)))
        calibrated_nees.append(float(np.mean(values)) / (3.0 * inflation))
        calibrated_nll.append(
            float(row["mean_gaussian_nll"])
            + 0.5
            * (
                3.0 * math.log(inflation)
                + 3.0 * float(row["normalized_nees"]) * (1.0 / inflation - 1.0)
            )
        )

    result: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "repository_revision": revision,
        "stage": "source-calibration",
        "dataset": {
            "opened_split": "train",
            "independent_group_definition": "one official training trajectory file",
            "manifest": manifest(dataset_root, "train", 56),
        },
        "accounting": {
            "independent_groups": len(scores),
            "accepted_centroid_cases": int(sum(len(values) for values in scores.values())),
        },
        "calibration": {
            "inflation_factor": inflation,
            "moment_nees_factor": moment,
            "equal_group_weighted_coverage_factor": coverage,
            "selection_rule": protocol["calibration"]["rule"],
        },
        "source_metrics_equal_group": {
            "raw_normalized_nees": float(np.mean(group_normalized_nees)),
            "calibrated_normalized_nees": float(np.mean(calibrated_nees)),
            "raw_90pct_coverage": float(np.mean(raw_coverage)),
            "calibrated_90pct_coverage": float(np.mean(calibrated_coverage)),
            "raw_gaussian_nll": float(np.mean(raw_nll)),
            "calibrated_gaussian_nll": float(np.mean(calibrated_nll)),
        },
        "information_boundary": {
            "source_split_opened": True,
            "evaluation_file_contents_opened": False,
            "evaluation_outcomes_opened": False,
            "new_data_collected": False,
            "target_side_retuning_permitted": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["calibration_id"] = content_id(result, "calibration_id")
    return result


def stripped(row: dict[str, Any]) -> dict[str, float | int]:
    return {key: value for key, value in row.items() if key != "_nees_values"}


def inflate_metrics(row: dict[str, Any], inflation: float) -> dict[str, float | int]:
    if float(row["accepted_fraction"]) == 0.0:
        return stripped(row)
    values = np.asarray(row["_nees_values"], dtype=np.float64)
    result = stripped(row)
    result["mean_gaussian_nll"] = float(row["mean_gaussian_nll"]) + 0.5 * (
        3.0 * math.log(inflation) + 3.0 * float(row["normalized_nees"]) * (1.0 / inflation - 1.0)
    )
    result["normalized_nees"] = float(row["normalized_nees"]) / inflation
    result["empirical_90pct_coverage"] = float(np.mean(values <= CHI2_3_90 * inflation))
    result["mean_marginal_standard_deviation_mm"] = float(
        row["mean_marginal_standard_deviation_mm"]
    ) * math.sqrt(inflation)
    return result


def aggregate(groups: dict[str, dict[str, dict[str, dict[str, Any]]]], protocol: dict[str, Any]):
    result: dict[str, Any] = {}
    seed = int(protocol["inference"]["bootstrap_seed"])
    replicates = int(protocol["inference"]["bootstrap_replicates"])
    scalars = (
        "rmse_mm",
        "mean_gaussian_nll",
        "normalized_nees",
        "empirical_90pct_coverage",
        "mean_marginal_standard_deviation_mm",
        "harmful_fraction_vs_fallback",
        "accepted_fraction",
        "exact_fallback_fraction",
    )
    for query in QUERIES:
        result[query] = {}
        for method in METHODS:
            rows = [value[query][method] for value in groups.values()]
            method_result: dict[str, Any] = {
                "independent_groups": len(rows),
                "nested_cases": int(sum(int(row["count"]) for row in rows)),
            }
            for scalar in scalars:
                method_result[scalar] = original.mean_ci(
                    [float(row[scalar]) for row in rows],
                    seed=original.stable_seed(seed, f"{query}/{method}/{scalar}"),
                    replicates=replicates,
                )
            result[query][method] = method_result
        raw = [value[query]["raw_query_aware"] for value in groups.values()]
        calibrated = [value[query]["source_calibrated_query_aware"] for value in groups.values()]
        result[query]["paired_calibrated_vs_raw"] = {
            "nll_improvement": original.mean_ci(
                [
                    float(first["mean_gaussian_nll"]) - float(second["mean_gaussian_nll"])
                    for first, second in zip(raw, calibrated, strict=True)
                ],
                seed=original.stable_seed(seed, f"{query}/calibrated-vs-raw/nll"),
                replicates=replicates,
            )
        }
    return result


def expected_raw_reproduction(result: dict[str, Any], protocol: dict[str, Any]):
    expected = protocol["original_result_binding"]["expected_raw_equal_group_metrics"]
    differences = {}
    for query, methods in expected.items():
        for method, metrics in methods.items():
            actual_method = (
                "physical_fallback" if method == "physical_fallback" else "raw_query_aware"
            )
            for metric, expected_value in metrics.items():
                actual = float(result[query][actual_method][metric]["mean"])
                differences[f"{query}/{method}/{metric}"] = actual - float(expected_value)
    maximum = max(abs(value) for value in differences.values())
    tolerance = float(protocol["original_result_binding"]["absolute_tolerance"])
    return {
        "maximum_absolute_difference": maximum,
        "absolute_tolerance": tolerance,
        "passed": maximum <= tolerance,
        "differences": differences,
    }


def target_evaluate(
    dataset_root: Path,
    protocol: dict[str, Any],
    request: dict[str, Any],
    calibration: dict[str, Any],
    revision: str,
) -> dict[str, Any]:
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected calibration schema")
    if calibration.get("calibration_id") != content_id(calibration, "calibration_id"):
        raise ValueError("calibration_id mismatch")
    if calibration.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("calibration/protocol mismatch")
    if calibration["information_boundary"]["evaluation_file_contents_opened"] is not False:
        raise ValueError("source calibration is not target-closed")
    inflation = float(calibration["calibration"]["inflation_factor"])
    if not 1.0 <= inflation <= 10.0:
        raise ValueError("inflation is outside the registered bound")

    raw = run_original(
        dataset_root,
        original_request(protocol, seed=int(protocol["design"]["target_seed"])),
    )
    groups = {}
    for group, queries in raw["per_group_results"].items():
        groups[group] = {}
        for query in QUERIES:
            fallback = queries[query]["physical_fallback"]
            query_aware = queries[query]["query_aware"]
            groups[group][query] = {
                "physical_fallback": stripped(fallback),
                "raw_query_aware": stripped(query_aware),
                "source_calibrated_query_aware": inflate_metrics(query_aware, inflation),
            }
    summary = aggregate(groups, protocol)
    reproduction = expected_raw_reproduction(summary, protocol)

    def metric(method: str, name: str) -> float:
        return float(summary["segment_centroid"][method][name]["mean"])

    raw_coverage = metric("raw_query_aware", "empirical_90pct_coverage")
    calibrated_coverage = metric("source_calibrated_query_aware", "empirical_90pct_coverage")
    raw_nees = metric("raw_query_aware", "normalized_nees")
    calibrated_nees = metric("source_calibrated_query_aware", "normalized_nees")
    paired = summary["segment_centroid"]["paired_calibrated_vs_raw"]["nll_improvement"]
    coverage_band = protocol["inference"]["target_coverage_acceptance_band"]
    nees_band = protocol["inference"]["target_normalized_nees_acceptance_band"]
    criteria = {
        "raw_target_reproduces_immutable_result": bool(reproduction["passed"]),
        "source_calibration_is_target_closed": (
            calibration["information_boundary"]["evaluation_file_contents_opened"] is False
        ),
        "centroid_rmse_is_unchanged": math.isclose(
            metric("raw_query_aware", "rmse_mm"),
            metric("source_calibrated_query_aware", "rmse_mm"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "off_axis_rejections_remain_exact_fallback": float(
            summary["off_axis_probe"]["source_calibrated_query_aware"]["exact_fallback_fraction"][
                "mean"
            ]
        )
        == 1.0,
        "centroid_nll_improves": metric("source_calibrated_query_aware", "mean_gaussian_nll")
        < metric("raw_query_aware", "mean_gaussian_nll"),
        "paired_centroid_nll_lower_95_is_positive": float(paired["ci95_lower"]) > 0.0,
        "centroid_nll_still_beats_fallback": metric(
            "source_calibrated_query_aware", "mean_gaussian_nll"
        )
        < metric("physical_fallback", "mean_gaussian_nll"),
        "centroid_coverage_moves_closer_to_90pct": abs(calibrated_coverage - 0.9)
        < abs(raw_coverage - 0.9),
        "centroid_nees_moves_closer_to_one": abs(calibrated_nees - 1.0) < abs(raw_nees - 1.0),
        "centroid_coverage_in_registered_band": float(coverage_band[0])
        <= calibrated_coverage
        <= float(coverage_band[1]),
        "centroid_nees_in_registered_band": float(nees_band[0])
        <= calibrated_nees
        <= float(nees_band[1]),
    }
    invariants = all(
        criteria[name]
        for name in (
            "raw_target_reproduces_immutable_result",
            "source_calibration_is_target_closed",
            "centroid_rmse_is_unchanged",
            "off_axis_rejections_remain_exact_fallback",
        )
    )
    directional = all(
        criteria[name]
        for name in (
            "centroid_nll_improves",
            "centroid_nll_still_beats_fallback",
            "centroid_coverage_moves_closer_to_90pct",
            "centroid_nees_moves_closer_to_one",
        )
    )
    if invariants and directional and all(criteria.values()):
        decision = "source-calibrated-strong-positive"
    elif invariants and directional:
        decision = "source-calibrated-directional-positive"
    elif invariants:
        decision = "source-calibrated-mixed-or-negative"
    else:
        decision = "technical-integrity-failure"

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "repository_revision": revision,
        "stage": "existing-evaluation-reanalysis",
        "decision": decision,
        "source_calibration": calibration,
        "dataset": {
            "opened_split": "eval",
            "independent_group_definition": "one official evaluation trajectory file",
            "manifest": manifest(dataset_root, "eval", 14),
        },
        "accounting": {
            "independent_groups": len(groups),
            "nested_cases": int(
                sum(
                    int(value["segment_centroid"]["raw_query_aware"]["count"])
                    for value in groups.values()
                )
            ),
        },
        "aggregate_equal_group_results": summary,
        "per_group_results": groups,
        "raw_result_reproduction": reproduction,
        "invariance": {
            "means_unchanged_by_construction": True,
            "admission_unchanged_by_construction": True,
            "rejected_covariance_unchanged_by_construction": True,
        },
        "criteria": criteria,
        "information_boundary": {
            "source_calibration_opened_train_split_only": True,
            "target_split_previously_opened": True,
            "target_split_opened_for_this_reanalysis": True,
            "new_data_collected": False,
            "target_side_retuning_permitted": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result, "result_id")
    return result


def make_summary(result: dict[str, Any]) -> str:
    centroid = result["aggregate_equal_group_results"]["segment_centroid"]
    paired = centroid["paired_calibrated_vs_raw"]["nll_improvement"]
    rows = [
        "# Source-calibrated DLO4/DLO5 query covariance",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"- Calibration ID: `{result['source_calibration']['calibration_id']}`",
        f"- Result ID: `{result['result_id']}`",
        "- No new data were collected.",
        "- Means, factors, admission, and exact rejected-query fallback are unchanged.",
        "",
        "| Equal-trajectory result | Fallback | Raw query-aware | Source-calibrated |",
        "|---|---:|---:|---:|",
    ]
    for label, metric in (
        ("Centroid RMSE [mm]", "rmse_mm"),
        ("Centroid Gaussian NLL", "mean_gaussian_nll"),
        ("Centroid 90% coverage", "empirical_90pct_coverage"),
        ("Centroid normalized NEES", "normalized_nees"),
        ("Centroid marginal SD [mm]", "mean_marginal_standard_deviation_mm"),
    ):
        values = [float(centroid[method][metric]["mean"]) for method in METHODS]
        rows.append(f"| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    rows.extend(
        [
            "",
            "Paired calibrated-versus-raw NLL improvement: "
            f"`{paired['mean']:.6f}` "
            f"[`{paired['ci95_lower']:.6f}`, `{paired['ci95_upper']:.6f}`].",
            "",
            "## Registered checks",
            "",
        ]
    )
    rows.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in result["criteria"].items()
    )
    rows.extend(
        [
            "",
            "This is a post-hoc source-only calibration repair on the already-opened "
            "evaluation split, not a fresh confirmation cohort.",
            "",
        ]
    )
    return "\n".join(rows)


def command_validate(args: argparse.Namespace) -> int:
    protocol = load_json(Path(args.protocol))
    request = load_json(Path(args.request))
    validate_protocol(protocol)
    validate_request(request, protocol)
    print(json.dumps({"protocol_id": protocol["protocol_id"], "request_id": request["request_id"]}))
    return 0


def command_calibrate(args: argparse.Namespace) -> int:
    protocol = load_json(Path(args.protocol))
    request = load_json(Path(args.request))
    validate_protocol(protocol)
    validate_request(request, protocol)
    root = Path(args.dataset_root).resolve(strict=True)
    if root != Path(request["dataset_root"]):
        raise ValueError("dataset root changed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    calibration = source_calibrate(root, protocol, request, args.repository_revision)
    write_json(output / "calibration.json", calibration)
    print(
        json.dumps(
            {
                "calibration_id": calibration["calibration_id"],
                "inflation_factor": calibration["calibration"]["inflation_factor"],
            }
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    protocol = load_json(Path(args.protocol))
    request = load_json(Path(args.request))
    calibration = load_json(Path(args.calibration))
    validate_protocol(protocol)
    validate_request(request, protocol)
    root = Path(args.dataset_root).resolve(strict=True)
    if root != Path(request["dataset_root"]):
        raise ValueError("dataset root changed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = target_evaluate(root, protocol, request, calibration, args.repository_revision)
    write_json(output / "result.json", result)
    (output / "SUMMARY.md").write_text(make_summary(result), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "result_id": result["result_id"]}))
    return 3 if result["decision"] == "technical-integrity-failure" else 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--protocol", required=True)
    validate.add_argument("--request", required=True)
    validate.set_defaults(function=command_validate)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--dataset-root", required=True)
    calibrate.add_argument("--protocol", required=True)
    calibrate.add_argument("--request", required=True)
    calibrate.add_argument("--repository-revision", required=True)
    calibrate.add_argument("--output-dir", required=True)
    calibrate.set_defaults(function=command_calibrate)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--request", required=True)
    evaluate.add_argument("--calibration", required=True)
    evaluate.add_argument("--repository-revision", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(function=command_evaluate)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
