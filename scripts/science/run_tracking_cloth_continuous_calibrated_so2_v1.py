#!/usr/bin/env python3
"""Cross-fitted continuous-SO(2) support calibration on real cloth trajectories.

Each recording supplies a causal prefix that selects two axis markers and one
off-axis probe. At evaluation time the axis markers are observed, the probe is
treated as hidden, and a transported constant-angular-velocity predictor defines
one representative on a continuous axial orbit. Recording-level nonconformity
scores calibrate a continuous angle-arc plus Euclidean-tube support without
using held-fold trajectories.

The study is a real-trajectory, simulated-occlusion mechanism experiment. It
does not validate a learned visual provider, physical-state recovery, conditional
coverage, deployment safety, or state of the art.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.axial_query_certificate import (
    AngleArc,
    AxialRotationOrbit,
    certify_shared_orbit_advantage,
)
from prob4d.continuous_axial_support import (
    axial_tube_residual,
    calibrate_group_conformal_upper_bound,
    empirical_upper_quantile,
    point_position_query,
    squared_distance_query,
    support_from_conformal_threshold,
)
from prob4d.motive_csv import read_motive_layout, read_motive_markers

SCHEMA = "prob4d.tracking-cloth-continuous-calibrated-so2.v1"
TERMINAL_STATUSES = {
    "evaluated-continuous-calibrated-so2-positive",
    "evaluated-continuous-calibrated-so2-negative",
    "dataset-support-negative",
}


@dataclass(frozen=True, slots=True)
class Recording:
    relative_path: str
    size: str
    material: str
    marker_labels: tuple[str, ...]
    positions_mm: np.ndarray
    selected_indices: tuple[int, int, int]
    selected_labels: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class Case:
    angle_radians: float
    orbit_residual_mm: float
    radial_scale_mm: float
    normalized_score: float
    representative_mm: np.ndarray
    truth_mm: np.ndarray
    axis_center_mm: np.ndarray
    origin_mm: np.ndarray
    axis: np.ndarray
    gauge_id: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _tokens(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", text.casefold()) if part}


def _metadata(relative_path: str) -> tuple[str, str]:
    tokens = _tokens(relative_path)
    materials = [value for value in ("cotton", "denim", "wool", "polyester") if value in tokens]
    sizes = [value.upper() for value in ("a2", "a3") if value in tokens]
    if len(materials) != 1 or len(sizes) != 1:
        raise ValueError(f"ambiguous material/size metadata: {relative_path}")
    return materials[0], sizes[0]


def _group_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()[:16]


def _fold(relative_path: str, size: str, fold_count: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{size}|{relative_path}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def _sample_indices(start: int, stop: int, maximum: int) -> np.ndarray:
    if stop <= start or maximum < 1:
        return np.empty(0, dtype=np.int64)
    count = min(stop - start, maximum)
    return np.unique(np.linspace(start, stop - 1, num=count, dtype=np.int64))


def _line_geometry(
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    probe: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    delta = anchor_b - anchor_a
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("degenerate anchor line")
    axis = delta / distance
    centered = probe - anchor_a
    axial = float(centered @ axis)
    radial = centered - axial * axis
    return distance, float(np.linalg.norm(radial)), axis


def _select_triplet(
    positions: np.ndarray,
    labels: tuple[str, ...],
    *,
    prefix_frames: int,
    minimum_anchor_distance_mm: float,
    minimum_probe_radius_mm: float,
) -> tuple[tuple[int, int, int], dict[str, float]]:
    prefix = positions[:prefix_frames]
    marker_count = prefix.shape[1]
    minimum_valid = max(6, prefix_frames // 4)
    best_key: tuple[float, str, str, str] | None = None
    best_indices: tuple[int, int, int] | None = None
    best_details: dict[str, float] | None = None
    for first in range(marker_count):
        for second in range(first + 1, marker_count):
            for probe in range(marker_count):
                if probe in {first, second}:
                    continue
                selected = prefix[:, [first, second, probe]]
                valid = np.all(np.isfinite(selected), axis=(1, 2))
                if int(np.sum(valid)) < minimum_valid:
                    continue
                geometry = selected[valid]
                deltas = geometry[:, 1] - geometry[:, 0]
                distances = np.linalg.norm(deltas, axis=1)
                nondegenerate = distances > 1e-9
                if int(np.sum(nondegenerate)) < minimum_valid:
                    continue
                geometry = geometry[nondegenerate]
                distances = distances[nondegenerate]
                axes = (geometry[:, 1] - geometry[:, 0]) / distances[:, None]
                probe_delta = geometry[:, 2] - geometry[:, 0]
                axial = np.sum(probe_delta * axes, axis=1)
                radius = np.linalg.norm(probe_delta - axial[:, None] * axes, axis=1)
                anchor_q10 = float(np.quantile(distances, 0.10))
                radius_q10 = float(np.quantile(radius, 0.10))
                radius_median = float(np.median(radius))
                if (
                    anchor_q10 < minimum_anchor_distance_mm
                    or radius_q10 < minimum_probe_radius_mm
                ):
                    continue
                score = min(anchor_q10, 2.0 * radius_q10) + 0.01 * radius_median
                names = (labels[first], labels[second], labels[probe])
                key = (score, *names)
                if best_key is None or key > best_key:
                    best_key = key
                    best_indices = (first, second, probe)
                    best_details = {
                        "score": score,
                        "anchor_distance_q10_mm": anchor_q10,
                        "probe_radius_q10_mm": radius_q10,
                        "probe_radius_median_mm": radius_median,
                        "valid_prefix_frames": int(len(geometry)),
                    }
    if best_indices is None or best_details is None:
        raise ValueError("causal prefix did not provide a nondegenerate marker triplet")
    return best_indices, best_details


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_align(source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    source = np.asarray(source_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine > 1e-12:
        matrix = _skew(cross)
        return np.eye(3) + matrix + matrix @ matrix * ((1.0 - cosine) / (sine * sine))
    if cosine > 0.0:
        return np.eye(3)
    basis = np.eye(3)[int(np.argmin(np.abs(source)))]
    axis = np.cross(source, basis)
    axis /= np.linalg.norm(axis)
    return 2.0 * np.outer(axis, axis) - np.eye(3)


def _transport_point(
    point: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> np.ndarray:
    source_delta = source_b - source_a
    target_delta = target_b - target_a
    source_length = float(np.linalg.norm(source_delta))
    target_length = float(np.linalg.norm(target_delta))
    if min(source_length, target_length) <= 1e-9:
        raise ValueError("cannot transport across a degenerate anchor line")
    rotation = _rotation_align(source_delta / source_length, target_delta / target_length)
    return target_a + (target_length / source_length) * rotation @ (point - source_a)


def _predict_probe(
    earlier: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    gauge_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0, b0, p0 = earlier
    a1, b1, p1 = previous
    a2, b2, _ = current
    transported_0 = _transport_point(p0, a0, b0, a1, b1)
    _, radius_1, axis_1 = _line_geometry(a1, b1, transported_0)
    if radius_1 <= 1e-9:
        raise ValueError("transported earlier probe lies on the intermediate axis")
    orbit_1 = AxialRotationOrbit(a1, axis_1, f"{gauge_id}:velocity")
    increment = axial_tube_residual(
        orbit_1,
        transported_0,
        p1,
        angle_normalizer=1.0,
        radial_scale=radius_1,
    ).angle_radians

    transported_1 = _transport_point(p1, a1, b1, a2, b2)
    _, radius_2, axis_2 = _line_geometry(a2, b2, transported_1)
    if radius_2 <= 1e-9:
        raise ValueError("transported previous probe lies on the current axis")
    orbit_2 = AxialRotationOrbit(a2, axis_2, gauge_id)
    representative = orbit_2.transform(transported_1[None, :], increment)[0]
    centered = representative - a2
    axial = float(centered @ axis_2)
    axis_center = a2 + axial * axis_2
    return representative, axis_center, axis_2


def _make_cases(
    recording: Recording,
    *,
    horizon_frames: int,
    angle_normalizer_rad: float,
    selection_prefix_frames: int,
    maximum_cases: int,
    minimum_anchor_distance_mm: float,
    minimum_probe_radius_mm: float,
) -> list[Case]:
    positions = recording.positions_mm[:, recording.selected_indices]
    start = max(selection_prefix_frames, 2 * horizon_frames)
    indices = _sample_indices(start, len(positions), maximum_cases)
    cases: list[Case] = []
    for target_index in indices:
        earlier = positions[int(target_index - 2 * horizon_frames)]
        previous = positions[int(target_index - horizon_frames)]
        current = positions[int(target_index)]
        if not np.all(np.isfinite(np.stack((earlier, previous, current)))):
            continue
        try:
            anchor_distance, _, _ = _line_geometry(current[0], current[1], current[2])
            if anchor_distance < minimum_anchor_distance_mm:
                continue
            gauge_id = (
                f"tracking-cloth:{_group_id(recording.relative_path)}:"
                f"h{horizon_frames}:t{int(target_index)}"
            )
            representative, axis_center, axis = _predict_probe(
                earlier,
                previous,
                current,
                gauge_id,
            )
            radial_scale = float(np.linalg.norm(representative - axis_center))
            if radial_scale < minimum_probe_radius_mm:
                continue
            orbit = AxialRotationOrbit(current[0], axis, gauge_id)
            residual = axial_tube_residual(
                orbit,
                representative,
                current[2],
                angle_normalizer=angle_normalizer_rad,
                radial_scale=radial_scale,
            )
        except ValueError:
            continue
        cases.append(
            Case(
                angle_radians=residual.angle_radians,
                orbit_residual_mm=residual.euclidean_residual,
                radial_scale_mm=radial_scale,
                normalized_score=residual.normalized_score,
                representative_mm=np.array(representative, copy=True),
                truth_mm=np.array(current[2], copy=True),
                axis_center_mm=np.array(axis_center, copy=True),
                origin_mm=np.array(current[0], copy=True),
                axis=np.array(axis, copy=True),
                gauge_id=gauge_id,
            )
        )
    return cases


def _read_recording(
    path: Path,
    dataset_root: Path,
    protocol: dict[str, Any],
) -> tuple[Recording | None, dict[str, Any]]:
    relative = path.relative_to(dataset_root).as_posix()
    material, size = _metadata(relative)
    layout = read_motive_layout(path)
    expected_markers = protocol["dataset"]["expected_cloth_marker_counts"].get(size)
    metadata = {
        "group_id": _group_id(relative),
        "relative_path_sha256": hashlib.sha256(relative.encode()).hexdigest(),
        "material": material,
        "size": size,
        "available_marker_count": len(layout.markers),
        "length_units": layout.length_units,
    }
    if expected_markers is None or len(layout.markers) != int(expected_markers):
        metadata["exclusion_reason"] = "non-cloth-only-marker-layout"
        return None, metadata
    coordinates, scale, details = read_motive_markers(path, layout.marker_labels)
    selection, selection_details = _select_triplet(
        coordinates,
        layout.marker_labels,
        prefix_frames=int(protocol["geometry"]["selection_prefix_frames"]),
        minimum_anchor_distance_mm=float(
            protocol["geometry"]["minimum_anchor_distance_mm"]
        ),
        minimum_probe_radius_mm=float(protocol["geometry"]["minimum_probe_radius_mm"]),
    )
    recording = Recording(
        relative_path=relative,
        size=size,
        material=material,
        marker_labels=layout.marker_labels,
        positions_mm=coordinates,
        selected_indices=selection,
        selected_labels=tuple(layout.marker_labels[index] for index in selection),
    )
    metadata.update(
        {
            "unit_scale_to_mm": scale,
            "rows": details["rows"],
            "selected_marker_labels": list(recording.selected_labels),
            "selection": selection_details,
        }
    )
    return recording, metadata


def _squared_error(first: np.ndarray, second: np.ndarray) -> float:
    delta = first - second
    return float(delta @ delta)


def _sampled_query_bounds(
    query: Any,
    arc: AngleArc,
    *,
    count: int,
    padding: float,
) -> tuple[float, float]:
    if count < 2:
        raise ValueError("sample count must be at least two")
    angles = np.linspace(arc.center - arc.half_width, arc.center + arc.half_width, count)
    values = np.array([query.evaluate(float(angle)) for angle in angles])
    return float(np.min(values) - padding), float(np.max(values) + padding)


def _evaluate_cases(
    cases: list[Case],
    *,
    threshold: float,
    angle_normalizer_rad: float,
    grid_sizes: list[int],
) -> dict[str, Any]:
    counts = {
        "case_count": 0,
        "support_covered": 0,
        "continuous_query_covered": 0,
        "local_harmful": 0,
        "calibrated_accepted": 0,
        "calibrated_harmful_accepted": 0,
        "full_circle_accepted": 0,
        "exact_fallback": 0,
    }
    grid_covered = {count: 0 for count in grid_sizes}
    grid_width_deficit: dict[int, list[float]] = {count: [] for count in grid_sizes}
    losses = {"fallback": [], "local": [], "calibrated": []}
    interval_widths: list[float] = []
    arc_width_degrees: list[float] = []
    tube_radii: list[float] = []

    for case in cases:
        orbit = AxialRotationOrbit(case.origin_mm, case.axis, case.gauge_id)
        support = support_from_conformal_threshold(
            orbit,
            normalized_score_threshold=threshold,
            radial_scale=case.radial_scale_mm,
            angle_normalizer=angle_normalizer_rad,
        )
        support_covered = support.contains(
            orbit,
            case.representative_mm,
            case.truth_mm,
            atol=1e-9,
        )
        counts["support_covered"] += int(support_covered)

        candidate_loss = squared_distance_query(
            orbit,
            case.representative_mm,
            case.representative_mm,
        )
        fallback_loss = squared_distance_query(
            orbit,
            case.representative_mm,
            case.axis_center_mm,
        )
        error_bound = (
            2.0
            * float(np.linalg.norm(case.representative_mm - case.axis_center_mm))
            * support.euclidean_radius
        )
        calibrated = certify_shared_orbit_advantage(
            fallback_loss=fallback_loss,
            candidate_loss=candidate_loss,
            scope_admitted=True,
            arc=support.arc,
            advantage_error_bound=error_bound,
            required_margin=0.0,
            numerical_slack=1e-9,
        )
        full_circle = certify_shared_orbit_advantage(
            fallback_loss=fallback_loss,
            candidate_loss=candidate_loss,
            scope_admitted=True,
            arc=AngleArc(),
            advantage_error_bound=error_bound,
            required_margin=0.0,
            numerical_slack=1e-9,
        )

        fallback_error = _squared_error(case.truth_mm, case.axis_center_mm)
        candidate_error = _squared_error(case.truth_mm, case.representative_mm)
        selected = case.representative_mm if calibrated.admitted else case.axis_center_mm
        selected_error = _squared_error(case.truth_mm, selected)
        losses["fallback"].append(fallback_error)
        losses["local"].append(candidate_error)
        losses["calibrated"].append(selected_error)
        counts["local_harmful"] += int(candidate_error > fallback_error)
        counts["calibrated_accepted"] += int(calibrated.admitted)
        counts["calibrated_harmful_accepted"] += int(
            calibrated.admitted and candidate_error > fallback_error
        )
        counts["full_circle_accepted"] += int(full_circle.admitted)
        counts["exact_fallback"] += int(
            calibrated.admitted
            or np.array_equal(selected, case.axis_center_mm)
        )

        point_query = point_position_query(orbit, case.representative_mm)
        world_x = point_query.scalar_projection([1.0, 0.0, 0.0])
        exact = support.expand_scalar_bounds(world_x, euclidean_lipschitz=1.0)
        truth_x = float(case.truth_mm[0])
        query_covered = exact.lower - 1e-9 <= truth_x <= exact.upper + 1e-9
        counts["continuous_query_covered"] += int(query_covered)
        exact_width = exact.upper - exact.lower
        interval_widths.append(exact_width)
        for grid_size in grid_sizes:
            lower, upper = _sampled_query_bounds(
                world_x,
                support.arc,
                count=grid_size,
                padding=support.euclidean_radius,
            )
            grid_covered[grid_size] += int(lower - 1e-9 <= truth_x <= upper + 1e-9)
            grid_width_deficit[grid_size].append(max(0.0, exact_width - (upper - lower)))

        counts["case_count"] += 1
        arc_width_degrees.append(2.0 * math.degrees(support.arc.half_width))
        tube_radii.append(support.euclidean_radius)

    if counts["case_count"] == 0:
        raise ValueError("test recording has no valid cases")
    total = counts["case_count"]
    accepted = counts["calibrated_accepted"]

    def fraction(name: str) -> float:
        return counts[name] / total

    def rmse(name: str) -> float:
        return math.sqrt(float(np.mean(losses[name])))

    return {
        "case_count": total,
        "support_coverage": fraction("support_covered"),
        "continuous_query_interval_coverage": fraction("continuous_query_covered"),
        "local_harmful_fraction": fraction("local_harmful"),
        "calibrated_acceptance": fraction("calibrated_accepted"),
        "calibrated_harmful_accepted_fraction_all_cases": fraction(
            "calibrated_harmful_accepted"
        ),
        "calibrated_harmful_fraction_among_accepted": (
            counts["calibrated_harmful_accepted"] / accepted if accepted else None
        ),
        "full_circle_acceptance": fraction("full_circle_accepted"),
        "exact_fallback_fraction": fraction("exact_fallback"),
        "fallback_rmse_mm": rmse("fallback"),
        "local_rmse_mm": rmse("local"),
        "calibrated_rmse_mm": rmse("calibrated"),
        "continuous_interval_mean_width_mm": float(np.mean(interval_widths)),
        "continuous_arc_mean_width_degrees": float(np.mean(arc_width_degrees)),
        "tube_mean_radius_mm": float(np.mean(tube_radii)),
        "finite_grid_query_interval_coverage": {
            str(count): grid_covered[count] / total for count in grid_sizes
        },
        "finite_grid_mean_width_deficit_mm": {
            str(count): float(np.mean(grid_width_deficit[count])) for count in grid_sizes
        },
    }


def _bootstrap_mean(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[samples], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "group_count": int(array.size),
    }


def _aggregate(
    evaluations: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    replicates = int(protocol["inference"]["bootstrap_replicates"])
    base_seed = int(protocol["inference"]["bootstrap_seed"])
    by_horizon: dict[str, Any] = {}
    for horizon in protocol["geometry"]["horizons_frames"]:
        selected = [row for row in evaluations if row["horizon_frames"] == horizon]
        if not selected:
            continue
        metrics = (
            "support_coverage",
            "continuous_query_interval_coverage",
            "local_harmful_fraction",
            "calibrated_acceptance",
            "calibrated_harmful_accepted_fraction_all_cases",
            "full_circle_acceptance",
            "exact_fallback_fraction",
            "fallback_rmse_mm",
            "local_rmse_mm",
            "calibrated_rmse_mm",
            "continuous_interval_mean_width_mm",
            "continuous_arc_mean_width_degrees",
            "tube_mean_radius_mm",
        )
        summary = {
            metric: _bootstrap_mean(
                [float(row["metrics"][metric]) for row in selected],
                replicates=replicates,
                seed=base_seed + int(horizon) * 100 + index,
            )
            for index, metric in enumerate(metrics)
        }
        accepted_cases = sum(
            int(round(row["metrics"]["calibrated_acceptance"] * row["metrics"]["case_count"]))
            for row in selected
        )
        harmful_accepted = sum(
            int(
                round(
                    row["metrics"]["calibrated_harmful_accepted_fraction_all_cases"]
                    * row["metrics"]["case_count"]
                )
            )
            for row in selected
        )
        summary["pooled_harmful_fraction_among_accepted"] = (
            harmful_accepted / accepted_cases if accepted_cases else None
        )
        summary["recording_count"] = len(selected)
        summary["case_count"] = sum(row["metrics"]["case_count"] for row in selected)
        summary["threshold_range"] = [
            min(row["calibration"]["threshold"] for row in selected),
            max(row["calibration"]["threshold"] for row in selected),
        ]
        summary["grid_interval_coverage"] = {}
        for grid_size in protocol["diagnostics"]["finite_grid_sizes"]:
            key = str(grid_size)
            summary["grid_interval_coverage"][key] = _bootstrap_mean(
                [
                    float(row["metrics"]["finite_grid_query_interval_coverage"][key])
                    for row in selected
                ],
                replicates=replicates,
                seed=base_seed + int(horizon) * 1000 + int(grid_size),
            )
        by_horizon[str(horizon)] = summary

    criteria_settings = protocol["registered_criteria"]
    positive_horizons: list[int] = []
    criteria_by_horizon: dict[str, dict[str, bool]] = {}
    for horizon_text, row in by_horizon.items():
        criteria = {
            "minimum_recording_coverage": row["support_coverage"]["mean"]
            >= float(criteria_settings["minimum_recording_mean_support_coverage"]),
            "minimum_query_interval_coverage": row[
                "continuous_query_interval_coverage"
            ]["mean"]
            >= float(criteria_settings["minimum_recording_mean_query_interval_coverage"]),
            "nontrivial_acceptance": row["calibrated_acceptance"]["mean"]
            >= float(criteria_settings["minimum_calibrated_acceptance"]),
            "reduces_harmful_acceptance": row[
                "calibrated_harmful_accepted_fraction_all_cases"
            ]["mean"]
            < row["local_harmful_fraction"]["mean"],
            "exact_fallback": row["exact_fallback_fraction"]["mean"] == 1.0,
            "full_circle_is_conservative": row["full_circle_acceptance"]["mean"] == 0.0,
        }
        criteria_by_horizon[horizon_text] = criteria
        if all(criteria.values()):
            positive_horizons.append(int(horizon_text))
    return {
        "by_horizon": by_horizon,
        "criteria_by_horizon": criteria_by_horizon,
        "positive_horizons": positive_horizons,
        "overall_positive": bool(positive_horizons),
    }


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# Continuous SO(2) support calibration on Tracking Cloth",
        "",
        f"Status: **{result['status']}**",
        "",
    ]
    if result["status"] == "dataset-support-negative":
        lines.append(result["reason"])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            (
                "The support is a continuous angle arc plus Euclidean remainder tube. "
                "One recording contributes one calibration score; complete recordings "
                "are the empirical units."
            ),
            "",
            "| Horizon [frames] | Groups | Cases | Support cov. | Accept | Local harm | "
            "Accepted harm | Fallback/local/calibrated RMSE [mm] |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, row in result["aggregate"]["by_horizon"].items():
        accepted_harm = row["pooled_harmful_fraction_among_accepted"]
        accepted_text = "n/a" if accepted_harm is None else f"{100 * accepted_harm:.2f}%"
        lines.append(
            "| {h} | {groups} | {cases} | {coverage:.2f}% | {accept:.2f}% | "
            "{local:.2f}% | {accepted} | {fallback:.3f}/{local_rmse:.3f}/"
            "{calibrated:.3f} |".format(
                h=horizon,
                groups=row["recording_count"],
                cases=row["case_count"],
                coverage=100 * row["support_coverage"]["mean"],
                accept=100 * row["calibrated_acceptance"]["mean"],
                local=100 * row["local_harmful_fraction"]["mean"],
                accepted=accepted_text,
                fallback=row["fallback_rmse_mm"]["mean"],
                local_rmse=row["local_rmse_mm"]["mean"],
                calibrated=row["calibrated_rmse_mm"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            f"Positive registered horizons: {result['aggregate']['positive_horizons']}",
            "",
            (
                "Claim boundary: real motion-capture trajectories with simulated probe "
                "occlusion and a kinematic representative; no learned-provider, "
                "conditional-coverage, deployment-safety, or state-of-the-art claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    dataset_root: Path,
    protocol_path: Path,
    output_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported protocol schema")
    unsigned_protocol = dict(protocol)
    supplied_protocol_id = unsigned_protocol.pop("protocol_id", None)
    expected_protocol_id = _sha256_bytes(_canonical_bytes(unsigned_protocol))
    if supplied_protocol_id != expected_protocol_id:
        raise ValueError("protocol identity changed")
    csv_paths = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    if len(csv_paths) != int(protocol["dataset"]["expected_csv_files"]):
        raise ValueError("official CSV roster changed")

    recordings: list[Recording] = []
    inventory_rows: list[dict[str, Any]] = []
    for path in csv_paths:
        try:
            recording, metadata = _read_recording(path, dataset_root, protocol)
        except (OSError, UnicodeError, ValueError) as error:
            relative = path.relative_to(dataset_root).as_posix()
            metadata = {
                "group_id": _group_id(relative),
                "relative_path_sha256": hashlib.sha256(relative.encode()).hexdigest(),
                "exclusion_reason": f"parser-or-geometry-support:{error}",
            }
            recording = None
        inventory_rows.append(metadata)
        if recording is not None:
            recordings.append(recording)

    inventory = {
        "schema": "prob4d.tracking-cloth-continuous-calibrated-inventory.v1",
        "dataset_root_name": dataset_root.name,
        "csv_file_count": len(csv_paths),
        "accepted_recording_count": len(recordings),
        "accepted_by_size": {
            size: sum(recording.size == size for recording in recordings)
            for size in ("A2", "A3")
        },
        "excluded_recording_count": len(csv_paths) - len(recordings),
        "file_manifest_sha256": _sha256_bytes(_canonical_bytes(inventory_rows)),
        "files": inventory_rows,
    }
    _write_json(output_dir / "inventory.json", inventory)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "claim_boundary": protocol["claim_boundary"],
    }
    expected = int(protocol["dataset"]["expected_cloth_only_recordings"])
    if len(recordings) != expected:
        result.update(
            {
                "status": "dataset-support-negative",
                "reason": f"expected {expected} cloth-only recordings, found {len(recordings)}",
                "inventory": {
                    key: value for key, value in inventory.items() if key != "files"
                },
            }
        )
    else:
        horizons = [int(value) for value in protocol["geometry"]["horizons_frames"]]
        angle_normalizer = float(protocol["calibration"]["angle_normalizer_rad"])
        case_map: dict[tuple[str, int], list[Case]] = {}
        recording_scores: dict[tuple[str, int], float] = {}
        case_support_rows: list[dict[str, Any]] = []
        for recording in recordings:
            for horizon in horizons:
                cases = _make_cases(
                    recording,
                    horizon_frames=horizon,
                    angle_normalizer_rad=angle_normalizer,
                    selection_prefix_frames=int(
                        protocol["geometry"]["selection_prefix_frames"]
                    ),
                    maximum_cases=int(protocol["geometry"]["maximum_cases_per_recording"]),
                    minimum_anchor_distance_mm=float(
                        protocol["geometry"]["minimum_anchor_distance_mm"]
                    ),
                    minimum_probe_radius_mm=float(
                        protocol["geometry"]["minimum_probe_radius_mm"]
                    ),
                )
                if len(cases) < int(protocol["geometry"]["minimum_cases_per_recording"]):
                    raise ValueError(
                        f"{recording.relative_path} horizon {horizon} has only "
                        f"{len(cases)} valid cases"
                    )
                key = (recording.relative_path, horizon)
                case_map[key] = cases
                scores = [case.normalized_score for case in cases]
                recording_scores[key] = empirical_upper_quantile(
                    scores,
                    probability=float(protocol["calibration"]["within_recording_quantile"]),
                )
                case_support_rows.append(
                    {
                        "group_id": _group_id(recording.relative_path),
                        "size": recording.size,
                        "material": recording.material,
                        "horizon_frames": horizon,
                        "case_count": len(cases),
                        "recording_score": recording_scores[key],
                        "selected_marker_labels": list(recording.selected_labels),
                    }
                )
        _write_json(output_dir / "recording_scores.json", case_support_rows)

        fold_count = int(protocol["inference"]["fold_count"])
        salt = str(protocol["inference"]["fold_salt"])
        evaluations: list[dict[str, Any]] = []
        for size in ("A2", "A3"):
            size_records = [recording for recording in recordings if recording.size == size]
            for horizon in horizons:
                for fold_index in range(fold_count):
                    calibration_records = [
                        recording
                        for recording in size_records
                        if _fold(recording.relative_path, size, fold_count, salt)
                        != fold_index
                    ]
                    test_records = [
                        recording
                        for recording in size_records
                        if _fold(recording.relative_path, size, fold_count, salt)
                        == fold_index
                    ]
                    if not calibration_records or not test_records:
                        raise ValueError(
                            f"empty calibration/test split for {size} fold {fold_index}"
                        )
                    calibration = calibrate_group_conformal_upper_bound(
                        [
                            recording_scores[(recording.relative_path, horizon)]
                            for recording in calibration_records
                        ],
                        miscoverage=float(protocol["calibration"]["group_miscoverage"]),
                    )
                    if not calibration.finite or calibration.threshold is None:
                        raise ValueError(
                            f"finite conformal threshold unavailable for {size} fold "
                            f"{fold_index}"
                        )
                    for recording in test_records:
                        metrics = _evaluate_cases(
                            case_map[(recording.relative_path, horizon)],
                            threshold=calibration.threshold,
                            angle_normalizer_rad=angle_normalizer,
                            grid_sizes=[
                                int(value)
                                for value in protocol["diagnostics"]["finite_grid_sizes"]
                            ],
                        )
                        evaluations.append(
                            {
                                "group_id": _group_id(recording.relative_path),
                                "size": size,
                                "material": recording.material,
                                "fold": fold_index,
                                "horizon_frames": horizon,
                                "calibration": calibration.summary(),
                                "recording_score": recording_scores[
                                    (recording.relative_path, horizon)
                                ],
                                "recording_score_covered": (
                                    recording_scores[(recording.relative_path, horizon)]
                                    <= calibration.threshold
                                ),
                                "metrics": metrics,
                            }
                        )
        aggregate = _aggregate(evaluations, protocol)
        result.update(
            {
                "status": (
                    "evaluated-continuous-calibrated-so2-positive"
                    if aggregate["overall_positive"]
                    else "evaluated-continuous-calibrated-so2-negative"
                ),
                "inventory": {
                    key: value for key, value in inventory.items() if key != "files"
                },
                "evaluation_count": len(evaluations),
                "evaluations": evaluations,
                "aggregate": aggregate,
            }
        )

    result_bytes = _canonical_bytes(result)
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    manifest = {
        "schema": "prob4d.tracking-cloth-continuous-calibrated-manifest.v1",
        "source_revision": source_revision,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "result_sha256": _sha256_bytes(result_bytes),
        "inventory_sha256": _sha256_file(output_dir / "inventory.json"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "raw_trajectory_payload_copied": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root is unavailable: {args.dataset_root}")
    result = run(
        args.dataset_root.resolve(),
        args.protocol.resolve(),
        args.output_dir.resolve(),
        args.source_revision,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0 if result["status"] in TERMINAL_STATUSES else 3


if __name__ == "__main__":
    raise SystemExit(main())
