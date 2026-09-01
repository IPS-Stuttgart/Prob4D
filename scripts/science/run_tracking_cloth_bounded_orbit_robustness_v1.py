#!/usr/bin/env python3
"""Secondary robustness study for uncertain axial-orbit generators.

The primary Tracking Cloth result used exact motion-capture anchor and probe
positions to instantiate a controlled hidden SO(2) ambiguity.  This study keeps
the same public real trajectories but perturbs every anchor and probe inside a
declared Euclidean error ball.  It compares a plug-in orbit width with a
fail-closed outer bound whose containment follows analytically from the error
balls.

This is a post-primary robustness analysis.  It is not an independent held-out
confirmation and does not introduce a learned visual provider.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.bounded_axial_orbit import (
    bounded_axial_radius,
    point_to_line_coordinates,
)
from prob4d.motive_csv import read_motive_markers

PROTOCOL_ID = "6b372b78329a809aee508a68ffd51abeb945fc941c0db29ca996459ddf52f711"
PROTOCOL_SCHEMA = "prob4d.tracking-cloth-bounded-orbit-robustness.v1"
RESULT_SCHEMA = "prob4d.tracking-cloth-bounded-orbit-robustness-result.v1"
SOURCE_SEAL_SCHEMA = "prob4d.tracking-cloth-bounded-orbit-source-seal.v1"
PARENT_PROTOCOL_ID = "9c0fb1a4191743a5038a2f26e521db1640fd5abfc3cac389e851485b7836a472"
PARENT_PROTOCOL_BLOB = "2b82734df4292d2c47328828e2307942347ca2a7"
PARENT_RESULT_ID = "1441a141a8eccc1ae3a503c701c72e8d702a0bcb17226238f3d32effa3e58111"
MARKER_TRIPLET = ("1", "20", "5")
SOURCE_PATTERN = re.compile(
    r"^(cotton|denim|wool)_A2_(shake|twist)_(fast|slow)_(hands|hanger)[.]csv$"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(value)}\0".encode() + value,
        usedforsecurity=False,
    ).hexdigest()


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("robustness protocol must be one JSON object")
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id", None)
    if supplied != PROTOCOL_ID or _content_id(unsigned) != supplied:
        raise ValueError("robustness protocol identity changed")
    if value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("robustness protocol schema changed")
    parent = value.get("parent_result", {})
    if (
        parent.get("protocol_id") != PARENT_PROTOCOL_ID
        or parent.get("protocol_git_blob_sha1") != PARENT_PROTOCOL_BLOB
        or parent.get("result_id") != PARENT_RESULT_ID
        or parent.get("status") != "evaluated-real-geometry-passed"
    ):
        raise ValueError("parent result binding changed")
    dataset = value.get("dataset", {})
    if dataset.get("source_recording_count") != 24:
        raise ValueError("source recording count changed")
    if dataset.get("target_recording_count") != 15:
        raise ValueError("target recording count changed")
    if dataset.get("selected_marker_triplet") != list(MARKER_TRIPLET):
        raise ValueError("marker triplet changed")
    error = value.get("bounded_error_model", {})
    if error.get("point_error_bound_fractions_of_true_anchor_distance") != [
        0.0,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
    ]:
        raise ValueError("point-error sweep changed")
    if error.get("replicates_per_frame") != 16 or error.get("seed") != 20260902:
        raise ValueError("perturbation replication changed")
    source_thresholds = value.get("source_only_width_thresholds", {})
    if source_thresholds.get("quantiles") != [0.25, 0.5, 0.75]:
        raise ValueError("source threshold quantiles changed")
    sampling = value.get("sampling", {})
    if sampling != {
        "source_frames_per_recording": 12,
        "target_frames_per_recording": 128,
        "minimum_anchor_distance_mm": 20.0,
        "minimum_probe_radius_mm": 5.0,
    }:
        raise ValueError("sampling contract changed")
    order = value.get("information_order", {})
    if order.get("threshold_values_are_computed_from_source_trajectories_only") is not True:
        raise ValueError("source-only threshold rule changed")
    if order.get("target_side_retuning_allowed") is not False:
        raise ValueError("target-side retuning was enabled")
    if order.get("self_collision_target_trajectories_opened") is not False:
        raise ValueError("unsupported self-collision outcomes were authorized")
    return value


def _load_parent_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    path = Path(protocol["parent_result"]["protocol_path"])
    content = path.read_bytes()
    if _git_blob_sha1(content) != PARENT_PROTOCOL_BLOB:
        raise ValueError("parent v3 protocol Git blob changed")
    value = json.loads(content)
    if value.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError("parent v3 protocol identity changed")
    paths = value.get("dataset", {}).get("target_relative_paths")
    if not isinstance(paths, list) or len(paths) != 15:
        raise ValueError("parent target roster changed")
    if any("/Self-collisions/" in path for path in paths):
        raise ValueError("unsupported self-collision path entered the target roster")
    return value


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=min(length, maximum), dtype=np.int64))


def _source_paths(dataset_root: Path) -> list[Path]:
    free_hanging = dataset_root / "tracking_dataset" / "Free-hanging"
    paths = sorted(
        path for path in free_hanging.glob("*.csv") if SOURCE_PATTERN.fullmatch(path.name)
    )
    if len(paths) != 24:
        raise ValueError(f"expected 24 source recordings, found {len(paths)}")
    return paths


def _target_paths(dataset_root: Path, parent: dict[str, Any]) -> list[Path]:
    paths = [dataset_root / relative for relative in parent["dataset"]["target_relative_paths"]]
    if len(paths) != 15 or not all(path.is_file() for path in paths):
        raise ValueError("the exact 15-recording target roster is unavailable")
    return paths


def _recording_cases(
    path: Path,
    dataset_root: Path,
    maximum_frames: int,
    minimum_anchor: float,
    minimum_radius: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coordinates, scale, details = read_motive_markers(path, MARKER_TRIPLET)
    cases: list[dict[str, Any]] = []
    for frame_index in _sample_indices(coordinates.shape[0], maximum_frames):
        frame = coordinates[int(frame_index)]
        if not np.all(np.isfinite(frame)):
            continue
        anchor_distance = float(np.linalg.norm(frame[1] - frame[0]))
        axial, radius = point_to_line_coordinates(frame[0], frame[1], frame[2])
        if anchor_distance < minimum_anchor or radius < minimum_radius:
            continue
        cases.append(
            {
                "frame_index": int(frame_index),
                "anchor_a": frame[0],
                "anchor_b": frame[1],
                "probe": frame[2],
                "anchor_distance": anchor_distance,
                "axial_coordinate": axial,
                "radius": radius,
                "true_width": 2.0 * radius,
            }
        )
    relative = path.relative_to(dataset_root).as_posix()
    return cases, {
        "group_id": _stable_id(relative),
        "relative_path": relative,
        "available_rows": int(details["rows"]),
        "sampled_case_count": len(cases),
        "unit_scale_to_mm": scale,
    }


def _hash_direction(*parts: object) -> np.ndarray:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    maximum = float((1 << 64) - 1)
    vector = np.array(
        [
            2.0 * int.from_bytes(digest[8 * index : 8 * (index + 1)], "big") / maximum - 1.0
            for index in range(3)
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).tiny:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _bootstrap(values: Iterable[float], replicates: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap received empty or nonfinite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    draws = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "groups": int(array.size),
    }


def _source_seal(
    dataset_root: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    sampling = protocol["sampling"]
    widths: list[float] = []
    groups: list[dict[str, Any]] = []
    for path in _source_paths(dataset_root):
        cases, metadata = _recording_cases(
            path,
            dataset_root,
            int(sampling["source_frames_per_recording"]),
            float(sampling["minimum_anchor_distance_mm"]),
            float(sampling["minimum_probe_radius_mm"]),
        )
        if not cases:
            raise ValueError(f"source recording produced no valid cases: {path.name}")
        widths.extend(float(case["true_width"]) for case in cases)
        groups.append(metadata)
    quantiles = protocol["source_only_width_thresholds"]["quantiles"]
    thresholds = {
        f"q{int(round(100.0 * quantile)):02d}": float(np.quantile(widths, quantile))
        for quantile in quantiles
    }
    value: dict[str, Any] = {
        "schema": SOURCE_SEAL_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "source_group_count": len(groups),
        "source_case_count": len(widths),
        "source_group_ids": [row["group_id"] for row in groups],
        "thresholds_mm": thresholds,
        "width_summary_mm": {
            "minimum": float(np.min(widths)),
            "median": float(np.median(widths)),
            "maximum": float(np.max(widths)),
        },
        "groups": groups,
        "target_trajectory_values_opened_before_source_seal": False,
    }
    value["source_seal_id"] = _content_id(value)
    return value, thresholds


def _evaluate_group(
    cases: list[dict[str, Any]],
    metadata: dict[str, Any],
    thresholds: dict[str, float],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    error_model = protocol["bounded_error_model"]
    ratios = error_model["point_error_bound_fractions_of_true_anchor_distance"]
    replicates = int(error_model["replicates_per_frame"])
    seed = int(error_model["seed"])
    noise_rows: list[dict[str, Any]] = []

    for ratio in ratios:
        total = 0
        informative = 0
        plugin_contains = 0
        outer_contains = 0
        plugin_width_ratios: list[float] = []
        outer_width_ratios: list[float] = []
        threshold_counts = {
            name: {
                "oracle_accept": 0,
                "plugin_accept": 0,
                "outer_accept": 0,
                "plugin_false_accept": 0,
                "outer_false_accept": 0,
                "plugin_false_reject": 0,
                "outer_false_reject": 0,
            }
            for name in thresholds
        }
        maximum_outer_violation = -math.inf

        for case in cases:
            true_width = float(case["true_width"])
            epsilon = float(ratio) * float(case["anchor_distance"])
            frame_index = int(case["frame_index"])
            for replicate in range(replicates):
                observed = []
                for label, point in (
                    ("anchor-a", case["anchor_a"]),
                    ("anchor-b", case["anchor_b"]),
                    ("probe", case["probe"]),
                ):
                    direction = _hash_direction(
                        seed,
                        metadata["group_id"],
                        frame_index,
                        ratio,
                        replicate,
                        label,
                    )
                    observed.append(np.asarray(point) + epsilon * direction)
                bound = bounded_axial_radius(*observed, epsilon)
                plugin_width = bound.observed_full_orbit_width
                outer_width = bound.outer_full_orbit_width
                total += 1
                informative += int(bound.informative)
                plugin_contains += int(plugin_width + 1e-10 >= true_width)
                outer_contains += int(outer_width + 1e-10 >= true_width)
                maximum_outer_violation = max(
                    maximum_outer_violation,
                    true_width - outer_width,
                )
                plugin_width_ratios.append(plugin_width / true_width)
                if math.isfinite(outer_width):
                    outer_width_ratios.append(outer_width / true_width)

                for name, threshold in thresholds.items():
                    oracle_accept = true_width <= threshold
                    plugin_accept = plugin_width <= threshold
                    outer_accept = bound.accepts_width(threshold)
                    counts = threshold_counts[name]
                    counts["oracle_accept"] += int(oracle_accept)
                    counts["plugin_accept"] += int(plugin_accept)
                    counts["outer_accept"] += int(outer_accept)
                    counts["plugin_false_accept"] += int(plugin_accept and not oracle_accept)
                    counts["outer_false_accept"] += int(outer_accept and not oracle_accept)
                    counts["plugin_false_reject"] += int(not plugin_accept and oracle_accept)
                    counts["outer_false_reject"] += int(not outer_accept and oracle_accept)

        if total == 0:
            raise ValueError(
                f"target group produced no perturbation cases: {metadata['relative_path']}"
            )
        threshold_rows = []
        for name, counts in threshold_counts.items():
            threshold_rows.append(
                {
                    "threshold_name": name,
                    "threshold_mm": thresholds[name],
                    **{key: value / total for key, value in counts.items()},
                }
            )
        noise_rows.append(
            {
                "point_error_fraction": float(ratio),
                "perturbation_case_count": total,
                "informative_fraction": informative / total,
                "plugin_orbit_containment": plugin_contains / total,
                "outer_orbit_containment": outer_contains / total,
                "plugin_width_ratio_mean": float(np.mean(plugin_width_ratios)),
                "plugin_width_ratio_median": float(np.median(plugin_width_ratios)),
                "outer_width_ratio_mean": float(np.mean(outer_width_ratios)),
                "outer_width_ratio_median": float(np.median(outer_width_ratios)),
                "maximum_outer_width_violation_mm": maximum_outer_violation,
                "thresholds": threshold_rows,
            }
        )
    return {**metadata, "noise_levels": noise_rows}


def _aggregate(
    groups: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    error_ratios = protocol["bounded_error_model"][
        "point_error_bound_fractions_of_true_anchor_distance"
    ]
    threshold_names = [
        f"q{int(round(100.0 * quantile)):02d}"
        for quantile in protocol["source_only_width_thresholds"]["quantiles"]
    ]
    replicates = int(protocol["inference"]["bootstrap_replicates"])
    seed = int(protocol["inference"]["bootstrap_seed"])
    result: list[dict[str, Any]] = []
    scalar_metrics = (
        "informative_fraction",
        "plugin_orbit_containment",
        "outer_orbit_containment",
        "plugin_width_ratio_mean",
        "outer_width_ratio_mean",
    )
    threshold_metrics = (
        "oracle_accept",
        "plugin_accept",
        "outer_accept",
        "plugin_false_accept",
        "outer_false_accept",
        "plugin_false_reject",
        "outer_false_reject",
    )

    for ratio_index, ratio in enumerate(error_ratios):
        rows = [group["noise_levels"][ratio_index] for group in groups]
        aggregate = {
            "point_error_fraction": float(ratio),
            "recording_count": len(groups),
            "total_perturbation_cases": sum(row["perturbation_case_count"] for row in rows),
            "metrics": {
                name: _bootstrap(
                    [float(row[name]) for row in rows],
                    replicates,
                    seed + 1000 * ratio_index + index,
                )
                for index, name in enumerate(scalar_metrics)
            },
            "maximum_outer_width_violation_mm": max(
                float(row["maximum_outer_width_violation_mm"]) for row in rows
            ),
            "thresholds": [],
        }
        for threshold_index, threshold_name in enumerate(threshold_names):
            threshold_rows = [
                next(item for item in row["thresholds"] if item["threshold_name"] == threshold_name)
                for row in rows
            ]
            aggregate["thresholds"].append(
                {
                    "threshold_name": threshold_name,
                    "threshold_mm": threshold_rows[0]["threshold_mm"],
                    "metrics": {
                        name: _bootstrap(
                            [float(row[name]) for row in threshold_rows],
                            replicates,
                            seed + 1000 * ratio_index + 100 * threshold_index + index,
                        )
                        for index, name in enumerate(threshold_metrics)
                    },
                }
            )
        result.append(aggregate)
    return result


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# Tracking Cloth bounded-orbit robustness",
        "",
        f"Result ID: `{result['result_id']}`",
        "",
        (
            "This secondary study perturbs both anchors and the probe within "
            "exact Euclidean error balls on the 15 already evaluated public "
            "real-trajectory recordings."
        ),
        "Thresholds are source-only quantiles of the true full radial-orbit width.",
        "",
        (
            "| point error / anchor span | plug-in containment | outer "
            "containment | outer width / true width | q50 plug-in false "
            "accept | q50 outer false accept | q50 outer false reject |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregate"]:
        metrics = row["metrics"]
        q50 = next(item for item in row["thresholds"] if item["threshold_name"] == "q50")
        threshold = q50["metrics"]
        lines.append(
            "| {ratio:.1%} | {plugin:.4f} | {outer:.4f} | {inflation:.3f} | "
            "{plugin_false:.4f} | {outer_false:.4f} | {outer_reject:.4f} |".format(
                ratio=row["point_error_fraction"],
                plugin=metrics["plugin_orbit_containment"]["mean"],
                outer=metrics["outer_orbit_containment"]["mean"],
                inflation=metrics["outer_width_ratio_mean"]["mean"],
                plugin_false=threshold["plugin_false_accept"]["mean"],
                outer_false=threshold["outer_false_accept"]["mean"],
                outer_reject=threshold["outer_false_reject"]["mean"],
            )
        )
    lines += [
        "",
        (
            "The outer certificate is conditionally exact only when the "
            "declared point-error balls contain the true points. Width "
            "inflation quantifies its conservatism; it is not a learned-"
            "provider accuracy result."
        ),
    ]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = [
        "point_error_fraction",
        "threshold_name",
        "threshold_mm",
        "metric",
        "mean",
        "ci95_low",
        "ci95_high",
        "groups",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for aggregate in result["aggregate"]:
            for name, value in aggregate["metrics"].items():
                writer.writerow(
                    {
                        "point_error_fraction": aggregate["point_error_fraction"],
                        "threshold_name": "",
                        "threshold_mm": "",
                        "metric": name,
                        **value,
                    }
                )
            for threshold in aggregate["thresholds"]:
                for name, value in threshold["metrics"].items():
                    writer.writerow(
                        {
                            "point_error_fraction": aggregate["point_error_fraction"],
                            "threshold_name": threshold["threshold_name"],
                            "threshold_mm": threshold["threshold_mm"],
                            "metric": name,
                            **value,
                        }
                    )


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    protocol = _load_protocol(protocol_path)
    parent = _load_parent_protocol(protocol)
    dataset_root = Path(args.dataset_root).resolve(strict=True)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)

    source_seal, thresholds = _source_seal(dataset_root, protocol)
    _write_json(output / "source_seal.json", source_seal)

    sampling = protocol["sampling"]
    groups = []
    for path in _target_paths(dataset_root, parent):
        cases, metadata = _recording_cases(
            path,
            dataset_root,
            int(sampling["target_frames_per_recording"]),
            float(sampling["minimum_anchor_distance_mm"]),
            float(sampling["minimum_probe_radius_mm"]),
        )
        if not cases:
            raise ValueError(f"target recording produced no valid cases: {path.name}")
        groups.append(_evaluate_group(cases, metadata, thresholds, protocol))

    aggregate = _aggregate(groups, protocol)
    maximum_violation = max(float(row["maximum_outer_width_violation_mm"]) for row in aggregate)
    outer_false_accept = max(
        float(threshold["metrics"]["outer_false_accept"]["mean"])
        for row in aggregate
        for threshold in row["thresholds"]
    )
    zero_error = aggregate[0]
    zero_plugin_error = max(
        abs(float(zero_error["metrics"][name]["mean"]) - expected)
        for name, expected in (
            ("plugin_orbit_containment", 1.0),
            ("outer_orbit_containment", 1.0),
            ("plugin_width_ratio_mean", 1.0),
            ("outer_width_ratio_mean", 1.0),
        )
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "source_revision": str(args.source_revision),
        "parent_result": protocol["parent_result"],
        "source_seal_id": source_seal["source_seal_id"],
        "source_thresholds_mm": thresholds,
        "target_group_count": len(groups),
        "target_case_count": sum(group["sampled_case_count"] for group in groups),
        "groups": groups,
        "aggregate": aggregate,
        "mathematical_checks": {
            "maximum_outer_width_violation_mm": maximum_violation,
            "maximum_outer_false_acceptance": outer_false_accept,
            "maximum_zero_error_discrepancy": zero_plugin_error,
            "all_outer_bounds_contained_true_orbits": maximum_violation <= 1e-9,
            "all_outer_gates_had_zero_false_acceptance": outer_false_accept == 0.0,
            "zero_error_matched_oracle": zero_plugin_error <= 1e-12,
        },
        "self_collision_header_audit": {
            "audit_id": "8e062f5ac4a8521b51cf616986331614f1bc91b5ca1d4971ed7647a5c4bc2ed9",
            "target_trajectory_values_parsed": False,
            "compatible_source_target_namespace_pairs": 0,
            "disposition": "not forced into the outcome study",
        },
        "information_order": protocol["information_order"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _content_id(result)
    _write_json(output / "result.json", result)
    _write_csv(output / "aggregate.csv", result)
    (output / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "source_seal_id": source_seal["source_seal_id"],
                "target_groups": result["target_group_count"],
                "target_cases": result["target_case_count"],
                "checks": result["mathematical_checks"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
