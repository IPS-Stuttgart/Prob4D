#!/usr/bin/env python3
"""Continuous-SO(2) decision-risk calibration on real cloth trajectories.

This stacked experiment addresses the principal negative result of the support-
only study. A transported constant-angular-velocity probe estimate is the
candidate and transported zero-angular-velocity persistence is the complete
fallback. A source-only continuous axial support gives a geometric base lower
bound on fallback-minus-candidate squared-error advantage.

A disjoint source calibration set then conformalizes the *signed* recording-wise
maximum deficit between that base bound and realized advantage. If the next
exchangeable recording's score is covered, every accepted case in that recording
is simultaneously non-harmful relative to the fallback.

The collision-family trajectories are a registered distribution-shift diagnostic,
not an independent confirmation: they were previously opened by a related
finite-orbit study. No threshold is selected from their outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
from collections.abc import Iterable
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
from prob4d.group_advantage_calibration import (
    SignedGroupConformalUpperBound,
    calibrate_signed_group_upper_bound,
    group_max_advantage_deficit,
)
from prob4d.motive_csv import (
    common_marker_labels,
    read_motive_layout,
    read_motive_markers,
)

SCHEMA = "prob4d.tracking-cloth-continuous-risk-calibrated-so2.v2"
TERMINAL_STATUSES = {
    "evaluated-continuous-risk-calibrated-so2-positive",
    "evaluated-continuous-risk-calibrated-so2-negative",
    "dataset-support-negative",
}


@dataclass(frozen=True, slots=True)
class RecordingRef:
    path: Path
    relative_path: str
    label: str
    scenario: str | None
    material: str
    size: str

    @property
    def group_id(self) -> str:
        return hashlib.sha256(self.relative_path.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Trajectory:
    recording: RecordingRef
    positions_mm: np.ndarray
    unit_scale_to_mm: float
    row_count: int


@dataclass(frozen=True, slots=True)
class Case:
    normalized_support_score: float
    representative_mm: np.ndarray
    fallback_mm: np.ndarray
    truth_mm: np.ndarray
    origin_mm: np.ndarray
    axis: np.ndarray
    radial_scale_mm: float
    angular_increment_rad: float
    gauge_id: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
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
    return {
        part
        for part in re.split(r"[^a-z0-9]+", text.casefold())
        if part
    }


def _metadata(relative_path: str) -> tuple[str, str]:
    tokens = _tokens(relative_path)
    materials = [
        value
        for value in ("cotton", "denim", "wool", "polyester")
        if value in tokens
    ]
    sizes = [
        value.upper()
        for value in ("a2", "a3")
        if value in tokens
    ]
    if len(materials) != 1 or len(sizes) != 1:
        raise ValueError(f"ambiguous material/size metadata: {relative_path}")
    return materials[0], sizes[0]


def _classify_recordings(
    dataset_root: Path,
    paths: list[Path],
    protocol: dict[str, Any],
) -> tuple[list[RecordingRef], dict[str, Any]]:
    dataset = protocol["dataset"]
    source_aliases = dataset["source_aliases"]
    target_aliases = dataset["target_aliases"]
    assigned: dict[Path, str] = {}

    for path in paths:
        relative = path.relative_to(dataset_root).as_posix()
        lower = relative.casefold()
        hits: list[str] = []
        for label, aliases in source_aliases.items():
            if any(alias.casefold() in lower for alias in aliases):
                hits.append(label)
        if any(alias.casefold() in lower for alias in target_aliases):
            hits.append("collision")
        if len(set(hits)) == 1:
            assigned[path] = hits[0]

    expected_total = int(dataset["expected_csv_files"])
    expected_source = int(dataset["expected_source_files"])
    expected_target = int(dataset["expected_target_files"])

    def valid_assignment(candidate: dict[Path, str]) -> bool:
        counts: dict[str, int] = {}
        for label in candidate.values():
            counts[label] = counts.get(label, 0) + 1
        return (
            len(candidate) == expected_total
            and counts.get("collision", 0) == expected_target
            and sum(counts.get(label, 0) for label in source_aliases)
            == expected_source
            and all(counts.get(label, 0) > 0 for label in source_aliases)
        )

    classification_mode = "declared-aliases"
    if not valid_assignment(assigned):
        ignored = {
            "csv",
            "data",
            "dataset",
            "recording",
            "recordings",
            "tracking",
            "cloth",
            "deformation",
            "trial",
            "trials",
        }
        token_files: dict[str, set[Path]] = {}
        for path in paths:
            relative = path.relative_to(dataset_root).as_posix()
            for token in _tokens(relative):
                if token in ignored or token.isdigit() or len(token) < 3:
                    continue
                token_files.setdefault(token, set()).add(path)
        source_tokens = sorted(
            token
            for token, members in token_files.items()
            if len(members) * 2 == expected_source
        )
        target_tokens = sorted(
            token
            for token, members in token_files.items()
            if len(members) == expected_target
        )
        inferred: dict[Path, str] | None = None
        inferred_tokens: tuple[str, str, str] | None = None
        for target_token in target_tokens:
            target_set = token_files[target_token]
            for first_token, second_token in itertools.combinations(source_tokens, 2):
                first_set = token_files[first_token]
                second_set = token_files[second_token]
                if (
                    first_set & second_set
                    or first_set & target_set
                    or second_set & target_set
                ):
                    continue
                if first_set | second_set | target_set != set(paths):
                    continue
                inferred = {path: "shake" for path in first_set}
                inferred.update({path: "twist" for path in second_set})
                inferred.update({path: "collision" for path in target_set})
                inferred_tokens = (first_token, second_token, target_token)
                break
            if inferred is not None:
                break
        if inferred is None or not valid_assignment(inferred):
            raise ValueError("could not identify the registered source/target split")
        assigned = inferred
        classification_mode = "exact-count-name-partition:" + ",".join(
            inferred_tokens or ()
        )

    records: list[RecordingRef] = []
    scenario_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    size_counts: dict[str, int] = {}
    source_a2_material_counts: dict[str, int] = {}
    for path in sorted(paths):
        relative = path.relative_to(dataset_root).as_posix()
        material, size = _metadata(relative)
        label = assigned[path]
        scenario: str | None = None
        if label == "collision":
            lower_tokens = _tokens(relative)
            if "table" in lower_tokens:
                scenario = "table_collision"
            elif {"stick", "hitting"} & lower_tokens:
                scenario = "stick_hitting"
            else:
                scenario = "self_collision"
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        elif size == "A2":
            source_a2_material_counts[material] = (
                source_a2_material_counts.get(material, 0) + 1
            )
        label_counts[label] = label_counts.get(label, 0) + 1
        size_counts[size] = size_counts.get(size, 0) + 1
        records.append(
            RecordingRef(
                path=path,
                relative_path=relative,
                label=label,
                scenario=scenario,
                material=material,
                size=size,
            )
        )

    expected_scenarios = {
        str(key): int(value)
        for key, value in dataset["target_scenario_counts"].items()
    }
    if scenario_counts != expected_scenarios:
        raise ValueError(f"collision-scenario counts changed: {scenario_counts}")
    expected_materials = {
        str(key): int(value)
        for key, value in dataset["expected_source_a2_by_material"].items()
    }
    if source_a2_material_counts != expected_materials:
        raise ValueError(
            "source A2 material counts changed: "
            f"{source_a2_material_counts}"
        )
    return records, {
        "classification_mode": classification_mode,
        "label_counts": label_counts,
        "size_counts": size_counts,
        "target_scenario_counts": scenario_counts,
        "source_a2_by_material": source_a2_material_counts,
    }


def _ordered(
    records: Iterable[RecordingRef],
    *,
    salt: str,
) -> list[RecordingRef]:
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha256(f"{salt}|{record.relative_path}".encode()).hexdigest(),
            record.relative_path,
        ),
    )


def _source_partitions(
    source_a2: list[RecordingRef],
    protocol: dict[str, Any],
) -> dict[str, list[RecordingRef]]:
    split = protocol["source_partition"]
    selection_per_material = int(split["selection_per_material"])
    support_per_material = int(split["support_calibration_per_material"])
    risk_per_material = int(split["risk_calibration_per_material"])
    salt = str(split["salt"])
    result = {
        "selection": [],
        "support_calibration": [],
        "risk_calibration": [],
    }
    for material in ("cotton", "denim", "wool", "polyester"):
        group = _ordered(
            [
                recording
                for recording in source_a2
                if recording.material == material
            ],
            salt=f"{salt}|{material}",
        )
        required = selection_per_material + support_per_material + risk_per_material
        if len(group) != required:
            raise ValueError(
                f"{material} has {len(group)} source A2 recordings, "
                f"expected {required}"
            )
        first = selection_per_material
        second = first + support_per_material
        result["selection"].extend(group[:first])
        result["support_calibration"].extend(group[first:second])
        result["risk_calibration"].extend(group[second:])
    return result


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


def _select_marker_triplet(
    samples: np.ndarray,
    marker_labels: list[str],
    *,
    minimum_anchor_distance_mm: float,
    minimum_probe_radius_mm: float,
) -> tuple[tuple[str, str, str], dict[str, float]]:
    if samples.ndim != 3 or samples.shape[1:] != (len(marker_labels), 3):
        raise ValueError("selection samples have an invalid shape")
    best_key: tuple[float, str, str, str] | None = None
    best_details: dict[str, float] | None = None
    marker_count = len(marker_labels)
    for first in range(marker_count):
        for second in range(first + 1, marker_count):
            a = samples[:, first]
            b = samples[:, second]
            delta = b - a
            distances = np.linalg.norm(delta, axis=1)
            valid_distance = distances > 1e-12
            if not np.any(valid_distance):
                continue
            axes = np.zeros_like(delta)
            axes[valid_distance] = (
                delta[valid_distance] / distances[valid_distance, None]
            )
            for probe in range(marker_count):
                if probe in {first, second}:
                    continue
                probe_delta = samples[:, probe] - a
                axial = np.sum(probe_delta * axes, axis=1)
                radii = np.linalg.norm(
                    probe_delta - axial[:, None] * axes,
                    axis=1,
                )
                valid = valid_distance & np.isfinite(radii)
                if int(np.sum(valid)) < max(16, samples.shape[0] // 4):
                    continue
                distance_q10 = float(np.quantile(distances[valid], 0.10))
                radius_q10 = float(np.quantile(radii[valid], 0.10))
                radius_median = float(np.median(radii[valid]))
                if (
                    distance_q10 < minimum_anchor_distance_mm
                    or radius_q10 < minimum_probe_radius_mm
                ):
                    continue
                score = min(distance_q10, 2.0 * radius_q10) + 0.05 * radius_median
                names = (
                    marker_labels[first],
                    marker_labels[second],
                    marker_labels[probe],
                )
                key = (score, *names)
                if best_key is None or key > best_key:
                    best_key = key
                    best_details = {
                        "score": score,
                        "anchor_distance_q10_mm": distance_q10,
                        "probe_radius_q10_mm": radius_q10,
                        "probe_radius_median_mm": radius_median,
                        "sample_frame_count": int(np.sum(valid)),
                    }
    if best_key is None or best_details is None:
        raise ValueError("source prefix did not yield a nondegenerate triplet")
    return (best_key[1], best_key[2], best_key[3]), best_details


def _selection_samples(
    records: list[RecordingRef],
    marker_labels: list[str],
    *,
    prefix_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for recording in records:
        coordinates, scale, details = read_motive_markers(
            recording.path,
            marker_labels,
        )
        valid = coordinates[np.all(np.isfinite(coordinates), axis=(1, 2))][
            :prefix_frames
        ]
        if len(valid) < max(6, prefix_frames // 2):
            raise ValueError(
                f"selection prefix is incomplete for {recording.group_id}"
            )
        arrays.append(valid)
        metadata.append(
            {
                "group_id": recording.group_id,
                "material": recording.material,
                "unit_scale_to_mm": scale,
                "used_prefix_frames": int(len(valid)),
                "available_rows": details["rows"],
            }
        )
    return np.concatenate(arrays, axis=0), metadata


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def _rotation_align(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine > 1e-12:
        matrix = _skew(cross)
        return (
            np.eye(3)
            + matrix
            + matrix @ matrix * ((1.0 - cosine) / (sine * sine))
        )
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
    rotation = _rotation_align(
        source_delta / source_length,
        target_delta / target_length,
    )
    return (
        target_a
        + (target_length / source_length) * rotation @ (point - source_a)
    )


def _predict_probe(
    earlier: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    gauge_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a0, b0, p0 = earlier
    a1, b1, p1 = previous
    a2, b2, _ = current
    transported_0 = _transport_point(p0, a0, b0, a1, b1)
    _, radius_1, axis_1 = _line_geometry(a1, b1, transported_0)
    if radius_1 <= 1e-9:
        raise ValueError("transported earlier probe lies on intermediate axis")
    orbit_1 = AxialRotationOrbit(a1, axis_1, f"{gauge_id}:velocity")
    increment = axial_tube_residual(
        orbit_1,
        transported_0,
        p1,
        angle_normalizer=1.0,
        radial_scale=radius_1,
    ).angle_radians

    fallback = _transport_point(p1, a1, b1, a2, b2)
    _, radius_2, axis_2 = _line_geometry(a2, b2, fallback)
    if radius_2 <= 1e-9:
        raise ValueError("transported previous probe lies on current axis")
    orbit_2 = AxialRotationOrbit(a2, axis_2, gauge_id)
    candidate = orbit_2.transform(fallback[None, :], increment)[0]
    return candidate, fallback, axis_2, increment


def _read_trajectory(
    recording: RecordingRef,
    marker_triplet: tuple[str, str, str],
) -> Trajectory:
    coordinates, scale, details = read_motive_markers(
        recording.path,
        marker_triplet,
    )
    return Trajectory(
        recording=recording,
        positions_mm=coordinates,
        unit_scale_to_mm=scale,
        row_count=int(details["rows"]),
    )


def _make_cases(
    trajectory: Trajectory,
    *,
    horizon_frames: int,
    angle_normalizer_rad: float,
    selection_prefix_frames: int,
    maximum_cases: int,
    minimum_anchor_distance_mm: float,
    minimum_probe_radius_mm: float,
) -> list[Case]:
    positions = trajectory.positions_mm
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
            anchor_distance, _, _ = _line_geometry(
                current[0],
                current[1],
                current[2],
            )
            if anchor_distance < minimum_anchor_distance_mm:
                continue
            gauge_id = (
                f"tracking-cloth-risk:{trajectory.recording.group_id}:"
                f"h{horizon_frames}:t{int(target_index)}"
            )
            representative, fallback, axis, increment = _predict_probe(
                earlier,
                previous,
                current,
                gauge_id,
            )
            orbit = AxialRotationOrbit(current[0], axis, gauge_id)
            centered = fallback - current[0]
            axial = float(centered @ axis)
            axis_center = current[0] + axial * axis
            radial_scale = float(np.linalg.norm(fallback - axis_center))
            if radial_scale < minimum_probe_radius_mm:
                continue
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
                normalized_support_score=residual.normalized_score,
                representative_mm=np.array(representative, copy=True),
                fallback_mm=np.array(fallback, copy=True),
                truth_mm=np.array(current[2], copy=True),
                origin_mm=np.array(current[0], copy=True),
                axis=np.array(axis, copy=True),
                radial_scale_mm=radial_scale,
                angular_increment_rad=increment,
                gauge_id=gauge_id,
            )
        )
    return cases


def _squared_error(first: np.ndarray, second: np.ndarray) -> float:
    delta = first - second
    return float(delta @ delta)


def _case_values(
    case: Case,
    *,
    support_threshold: float,
    angle_normalizer_rad: float,
    numerical_slack: float,
) -> dict[str, Any]:
    orbit = AxialRotationOrbit(case.origin_mm, case.axis, case.gauge_id)
    support = support_from_conformal_threshold(
        orbit,
        normalized_score_threshold=support_threshold,
        radial_scale=case.radial_scale_mm,
        angle_normalizer=angle_normalizer_rad,
    )
    candidate_loss = squared_distance_query(
        orbit,
        case.representative_mm,
        case.representative_mm,
    )
    fallback_loss = squared_distance_query(
        orbit,
        case.representative_mm,
        case.fallback_mm,
    )
    advantage_error_bound = (
        2.0
        * float(np.linalg.norm(case.representative_mm - case.fallback_mm))
        * support.euclidean_radius
    )
    geometric = certify_shared_orbit_advantage(
        fallback_loss=fallback_loss,
        candidate_loss=candidate_loss,
        scope_admitted=True,
        arc=support.arc,
        advantage_error_bound=advantage_error_bound,
        required_margin=0.0,
        numerical_slack=numerical_slack,
    )
    full_circle = certify_shared_orbit_advantage(
        fallback_loss=fallback_loss,
        candidate_loss=candidate_loss,
        scope_admitted=True,
        arc=AngleArc(),
        advantage_error_bound=advantage_error_bound,
        required_margin=0.0,
        numerical_slack=numerical_slack,
    )
    if geometric.lower_advantage is None:
        raise RuntimeError("finite calibrated arc produced no lower advantage")
    fallback_error = _squared_error(case.truth_mm, case.fallback_mm)
    candidate_error = _squared_error(case.truth_mm, case.representative_mm)
    actual_advantage = fallback_error - candidate_error
    point_query = point_position_query(orbit, case.representative_mm)
    world_x = point_query.scalar_projection([1.0, 0.0, 0.0])
    interval = support.expand_scalar_bounds(
        world_x,
        euclidean_lipschitz=1.0,
    )
    truth_x = float(case.truth_mm[0])
    return {
        "base_lower_advantage": float(geometric.lower_advantage),
        "actual_advantage": actual_advantage,
        "candidate_error": candidate_error,
        "fallback_error": fallback_error,
        "support_admitted": geometric.admitted,
        "full_circle_admitted": full_circle.admitted,
        "support_covered": support.contains(
            orbit,
            case.representative_mm,
            case.truth_mm,
            atol=numerical_slack,
        ),
        "query_interval_covered": (
            interval.lower - numerical_slack
            <= truth_x
            <= interval.upper + numerical_slack
        ),
        "arc_width_degrees": 2.0 * math.degrees(support.arc.half_width),
        "tube_radius_mm": support.euclidean_radius,
        "advantage_error_bound_mm2": advantage_error_bound,
    }


def _support_recording_score(
    cases: list[Case],
    *,
    within_recording_quantile: float,
) -> float:
    return empirical_upper_quantile(
        [case.normalized_support_score for case in cases],
        probability=within_recording_quantile,
    )


def _risk_recording_score(
    cases: list[Case],
    *,
    support_threshold: float,
    angle_normalizer_rad: float,
    numerical_slack: float,
) -> tuple[float, list[dict[str, Any]]]:
    values = [
        _case_values(
            case,
            support_threshold=support_threshold,
            angle_normalizer_rad=angle_normalizer_rad,
            numerical_slack=numerical_slack,
        )
        for case in cases
    ]
    score = group_max_advantage_deficit(
        [value["base_lower_advantage"] for value in values],
        [value["actual_advantage"] for value in values],
    )
    return score, values


def _evaluate_cases(
    cases: list[Case],
    *,
    support_threshold: float,
    risk_calibration: SignedGroupConformalUpperBound,
    angle_normalizer_rad: float,
    required_margin_mm2: float,
    numerical_slack: float,
    within_recording_quantile: float,
) -> dict[str, Any]:
    counts = {
        "case_count": 0,
        "support_covered": 0,
        "query_interval_covered": 0,
        "candidate_harmful": 0,
        "support_accepted": 0,
        "support_harmful_accepted": 0,
        "risk_accepted": 0,
        "risk_harmful_accepted": 0,
        "risk_exact_fallback": 0,
        "full_circle_accepted": 0,
    }
    squared_errors = {
        "fallback": [],
        "candidate": [],
        "support_policy": [],
        "risk_policy": [],
    }
    base_bounds: list[float] = []
    calibrated_bounds: list[float] = []
    actual_advantages: list[float] = []
    arc_widths: list[float] = []
    tube_radii: list[float] = []
    increments: list[float] = []

    for case in cases:
        value = _case_values(
            case,
            support_threshold=support_threshold,
            angle_normalizer_rad=angle_normalizer_rad,
            numerical_slack=numerical_slack,
        )
        base = float(value["base_lower_advantage"])
        actual = float(value["actual_advantage"])
        calibrated_lower = risk_calibration.lower_bound(base)
        risk_admitted = risk_calibration.admits(
            base,
            required_margin=required_margin_mm2,
            numerical_slack=numerical_slack,
        )
        support_admitted = bool(value["support_admitted"])
        harmful = actual < -numerical_slack
        support_selected = (
            case.representative_mm if support_admitted else case.fallback_mm
        )
        risk_selected = case.representative_mm if risk_admitted else case.fallback_mm

        fallback_error = float(value["fallback_error"])
        candidate_error = float(value["candidate_error"])
        squared_errors["fallback"].append(fallback_error)
        squared_errors["candidate"].append(candidate_error)
        squared_errors["support_policy"].append(
            _squared_error(case.truth_mm, support_selected)
        )
        squared_errors["risk_policy"].append(
            _squared_error(case.truth_mm, risk_selected)
        )
        counts["case_count"] += 1
        counts["support_covered"] += int(value["support_covered"])
        counts["query_interval_covered"] += int(value["query_interval_covered"])
        counts["candidate_harmful"] += int(harmful)
        counts["support_accepted"] += int(support_admitted)
        counts["support_harmful_accepted"] += int(support_admitted and harmful)
        counts["risk_accepted"] += int(risk_admitted)
        counts["risk_harmful_accepted"] += int(risk_admitted and harmful)
        counts["risk_exact_fallback"] += int(
            risk_admitted or np.array_equal(risk_selected, case.fallback_mm)
        )
        counts["full_circle_accepted"] += int(value["full_circle_admitted"])
        base_bounds.append(base)
        calibrated_bounds.append(calibrated_lower)
        actual_advantages.append(actual)
        arc_widths.append(float(value["arc_width_degrees"]))
        tube_radii.append(float(value["tube_radius_mm"]))
        increments.append(abs(case.angular_increment_rad))

    total = counts["case_count"]
    if total == 0:
        raise ValueError("recording has no evaluable cases")
    support_accepted = counts["support_accepted"]
    risk_accepted = counts["risk_accepted"]
    risk_score = group_max_advantage_deficit(base_bounds, actual_advantages)
    if risk_calibration.threshold is None:
        raise RuntimeError("risk threshold became unavailable")
    risk_score_covered = risk_score <= risk_calibration.threshold + numerical_slack
    if risk_score_covered and counts["risk_harmful_accepted"] > 0:
        raise RuntimeError(
            "covered recording contains a harmful risk-calibrated accepted case"
        )

    def fraction(name: str) -> float:
        return counts[name] / total

    def rmse(name: str) -> float:
        return math.sqrt(float(np.mean(squared_errors[name])))

    return {
        "case_count": total,
        "recording_support_score": _support_recording_score(
            cases,
            within_recording_quantile=within_recording_quantile,
        ),
        "recording_risk_score": risk_score,
        "risk_score_covered": risk_score_covered,
        "support_coverage": fraction("support_covered"),
        "query_interval_coverage": fraction("query_interval_covered"),
        "candidate_harmful_fraction": fraction("candidate_harmful"),
        "support_acceptance": fraction("support_accepted"),
        "support_harmful_accepted_fraction_all_cases": fraction(
            "support_harmful_accepted"
        ),
        "support_harmful_fraction_among_accepted": (
            counts["support_harmful_accepted"] / support_accepted
            if support_accepted
            else None
        ),
        "risk_acceptance": fraction("risk_accepted"),
        "risk_harmful_accepted_fraction_all_cases": fraction(
            "risk_harmful_accepted"
        ),
        "risk_harmful_fraction_among_accepted": (
            counts["risk_harmful_accepted"] / risk_accepted
            if risk_accepted
            else None
        ),
        "risk_exact_fallback_fraction": fraction("risk_exact_fallback"),
        "full_circle_acceptance": fraction("full_circle_accepted"),
        "fallback_rmse_mm": rmse("fallback"),
        "candidate_rmse_mm": rmse("candidate"),
        "support_policy_rmse_mm": rmse("support_policy"),
        "risk_policy_rmse_mm": rmse("risk_policy"),
        "fallback_sse_mm2": float(np.sum(squared_errors["fallback"])),
        "candidate_sse_mm2": float(np.sum(squared_errors["candidate"])),
        "support_policy_sse_mm2": float(np.sum(squared_errors["support_policy"])),
        "risk_policy_sse_mm2": float(np.sum(squared_errors["risk_policy"])),
        "mean_base_lower_advantage_mm2": float(np.mean(base_bounds)),
        "mean_calibrated_lower_advantage_mm2": float(np.mean(calibrated_bounds)),
        "mean_actual_advantage_mm2": float(np.mean(actual_advantages)),
        "mean_arc_width_degrees": float(np.mean(arc_widths)),
        "mean_tube_radius_mm": float(np.mean(tube_radii)),
        "mean_absolute_angular_increment_degrees": math.degrees(
            float(np.mean(increments))
        ),
    }


def _bootstrap_mean(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "group_count": int(array.size),
    }


def _aggregate_group(
    evaluations: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    metric_names = (
        "support_coverage",
        "query_interval_coverage",
        "candidate_harmful_fraction",
        "support_acceptance",
        "support_harmful_accepted_fraction_all_cases",
        "risk_acceptance",
        "risk_harmful_accepted_fraction_all_cases",
        "risk_exact_fallback_fraction",
        "full_circle_acceptance",
        "fallback_rmse_mm",
        "candidate_rmse_mm",
        "support_policy_rmse_mm",
        "risk_policy_rmse_mm",
        "mean_base_lower_advantage_mm2",
        "mean_calibrated_lower_advantage_mm2",
        "mean_actual_advantage_mm2",
        "mean_arc_width_degrees",
        "mean_tube_radius_mm",
        "mean_absolute_angular_increment_degrees",
    )
    result = {
        name: _bootstrap_mean(
            [float(row["metrics"][name]) for row in evaluations],
            replicates=replicates,
            seed=seed + index,
        )
        for index, name in enumerate(metric_names)
    }
    total_cases = sum(int(row["metrics"]["case_count"]) for row in evaluations)
    result.update(
        {
            "recording_count": len(evaluations),
            "case_count": total_cases,
            "recording_support_score_coverage": (
                sum(bool(row["recording_support_score_covered"]) for row in evaluations)
                / len(evaluations)
            ),
            "recording_risk_bound_coverage": (
                sum(bool(row["metrics"]["risk_score_covered"]) for row in evaluations)
                / len(evaluations)
            ),
        }
    )
    for method in ("fallback", "candidate", "support_policy", "risk_policy"):
        sse = sum(
            float(row["metrics"][f"{method}_sse_mm2"])
            for row in evaluations
        )
        result[f"pooled_{method}_rmse_mm"] = math.sqrt(sse / total_cases)
    risk_accepted = sum(
        round(
            float(row["metrics"]["risk_acceptance"])
            * int(row["metrics"]["case_count"])
        )
        for row in evaluations
    )
    risk_harmful = sum(
        round(
            float(row["metrics"]["risk_harmful_accepted_fraction_all_cases"])
            * int(row["metrics"]["case_count"])
        )
        for row in evaluations
    )
    support_accepted = sum(
        round(
            float(row["metrics"]["support_acceptance"])
            * int(row["metrics"]["case_count"])
        )
        for row in evaluations
    )
    support_harmful = sum(
        round(
            float(row["metrics"]["support_harmful_accepted_fraction_all_cases"])
            * int(row["metrics"]["case_count"])
        )
        for row in evaluations
    )
    result["pooled_risk_harmful_fraction_among_accepted"] = (
        risk_harmful / risk_accepted if risk_accepted else None
    )
    result["pooled_support_harmful_fraction_among_accepted"] = (
        support_harmful / support_accepted if support_accepted else None
    )
    return result


def _aggregate(
    evaluations: list[dict[str, Any]],
    calibration: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    replicates = int(protocol["inference"]["bootstrap_replicates"])
    base_seed = int(protocol["inference"]["bootstrap_seed"])
    by_horizon: dict[str, Any] = {}
    by_scenario: dict[str, Any] = {}
    for horizon in protocol["geometry"]["horizons_frames"]:
        selected = [
            row for row in evaluations if row["horizon_frames"] == horizon
        ]
        by_horizon[str(horizon)] = _aggregate_group(
            selected,
            replicates=replicates,
            seed=base_seed + int(horizon) * 100,
        )
        for scenario in (
            "table_collision",
            "stick_hitting",
            "self_collision",
        ):
            scenario_rows = [
                row for row in selected if row["scenario"] == scenario
            ]
            by_scenario[f"{horizon}:{scenario}"] = _aggregate_group(
                scenario_rows,
                replicates=replicates,
                seed=base_seed + int(horizon) * 1000 + len(by_scenario),
            )

    criteria_settings = protocol["registered_criteria"]
    criteria_by_horizon: dict[str, dict[str, bool]] = {}
    positive_horizons: list[int] = []
    for horizon_text, row in by_horizon.items():
        harmful_among = row["pooled_risk_harmful_fraction_among_accepted"]
        criteria = {
            "minimum_recording_risk_bound_coverage": (
                row["recording_risk_bound_coverage"]
                >= float(
                    criteria_settings[
                        "minimum_target_recording_risk_bound_coverage"
                    ]
                )
            ),
            "nontrivial_risk_acceptance": (
                row["risk_acceptance"]["mean"]
                >= float(criteria_settings["minimum_risk_calibrated_acceptance"])
            ),
            "reduces_harmful_acceptance": (
                row["risk_harmful_accepted_fraction_all_cases"]["mean"]
                < row["candidate_harmful_fraction"]["mean"]
            ),
            "accepted_harm_below_limit": (
                harmful_among is not None
                and harmful_among
                <= float(
                    criteria_settings[
                        "maximum_risk_harmful_fraction_among_accepted"
                    ]
                )
            ),
            "beats_zero_velocity_fallback": (
                row["pooled_risk_policy_rmse_mm"]
                <= row["pooled_fallback_rmse_mm"]
            ),
            "no_worse_than_support_only": (
                row["pooled_risk_policy_rmse_mm"]
                <= row["pooled_support_policy_rmse_mm"]
            ),
            "exact_fallback": (
                row["risk_exact_fallback_fraction"]["mean"] == 1.0
            ),
            "full_circle_rejects": (
                row["full_circle_acceptance"]["mean"] == 0.0
            ),
        }
        criteria_by_horizon[horizon_text] = criteria
        if all(criteria.values()):
            positive_horizons.append(int(horizon_text))
    return {
        "by_horizon": by_horizon,
        "by_horizon_and_scenario": by_scenario,
        "calibration": calibration,
        "criteria_by_horizon": criteria_by_horizon,
        "positive_horizons": positive_horizons,
        "overall_positive": (
            len(positive_horizons)
            >= int(criteria_settings["minimum_positive_horizons"])
        ),
    }


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# Continuous SO(2) decision-risk calibration",
        "",
        f"Status: **{result['status']}**",
        "",
    ]
    if result["status"] == "dataset-support-negative":
        lines.append(str(result["reason"]))
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            (
                "A source-only continuous support is followed by a disjoint "
                "recording-group conformal calibration of signed "
                "fallback-minus-candidate advantage deficit."
            ),
            "",
            "| Horizon | Groups | Cases | Risk-bound cov. | Candidate harm | "
            "Risk accept | Accepted harm | "
            "Fallback/candidate/support/risk RMSE [mm] |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, row in result["aggregate"]["by_horizon"].items():
        accepted_harm = row["pooled_risk_harmful_fraction_among_accepted"]
        accepted_text = (
            "n/a" if accepted_harm is None else f"{100 * accepted_harm:.3f}%"
        )
        lines.append(
            "| {h} | {groups} | {cases} | {coverage:.2f}% | "
            "{candidate_harm:.3f}% | {accept:.2f}% | {accepted_harm} | "
            "{fallback:.3f}/{candidate:.3f}/{support:.3f}/{risk:.3f} |".format(
                h=horizon,
                groups=row["recording_count"],
                cases=row["case_count"],
                coverage=100 * row["recording_risk_bound_coverage"],
                candidate_harm=(
                    100 * row["candidate_harmful_fraction"]["mean"]
                ),
                accept=100 * row["risk_acceptance"]["mean"],
                accepted_harm=accepted_text,
                fallback=row["pooled_fallback_rmse_mm"],
                candidate=row["pooled_candidate_rmse_mm"],
                support=row["pooled_support_policy_rmse_mm"],
                risk=row["pooled_risk_policy_rmse_mm"],
            )
        )
    lines.extend(
        [
            "",
            "Positive registered horizons: "
            f"{result['aggregate']['positive_horizons']}",
            "",
            (
                "The 56 collision-family recordings are a distribution-shift "
                "diagnostic previously opened by a related finite-orbit study; "
                "this is not an independent prospective confirmation."
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
    unsigned = dict(protocol)
    supplied_id = unsigned.pop("protocol_id", None)
    expected_id = _sha256_bytes(_canonical_bytes(unsigned))
    if supplied_id != expected_id:
        raise ValueError("protocol identity changed")

    csv_paths = sorted(
        path for path in dataset_root.rglob("*.csv") if path.is_file()
    )
    if len(csv_paths) != int(protocol["dataset"]["expected_csv_files"]):
        raise ValueError("official CSV roster changed")

    try:
        records, classification = _classify_recordings(
            dataset_root,
            csv_paths,
            protocol,
        )
        source_a2 = [
            recording
            for recording in records
            if recording.label != "collision" and recording.size == "A2"
        ]
        targets = [
            recording for recording in records if recording.label == "collision"
        ]
        partitions = _source_partitions(source_a2, protocol)
        common_labels = common_marker_labels(
            [recording.path for recording in source_a2 + targets],
            maximum=int(protocol["geometry"]["maximum_common_marker_count"]),
        )
        selection_samples, selection_metadata = _selection_samples(
            partitions["selection"],
            common_labels,
            prefix_frames=int(protocol["geometry"]["selection_prefix_frames"]),
        )
        marker_triplet, selection_details = _select_marker_triplet(
            selection_samples,
            common_labels,
            minimum_anchor_distance_mm=float(
                protocol["geometry"]["minimum_anchor_distance_mm"]
            ),
            minimum_probe_radius_mm=float(
                protocol["geometry"]["minimum_probe_radius_mm"]
            ),
        )
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        inventory = {
            "schema": (
                "prob4d.tracking-cloth-continuous-risk-calibrated-inventory.v2"
            ),
            "dataset_root_name": dataset_root.name,
            "csv_file_count": len(csv_paths),
            "support_error": str(error),
        }
        _write_json(output_dir / "inventory.json", inventory)
        result = {
            "schema": SCHEMA,
            "status": "dataset-support-negative",
            "reason": str(error),
            "source_revision": source_revision,
            "claim_boundary": protocol["claim_boundary"],
            "inventory": inventory,
        }
    else:
        split_receipt = {
            role: [
                {
                    "group_id": recording.group_id,
                    "material": recording.material,
                    "label": recording.label,
                }
                for recording in group
            ]
            for role, group in partitions.items()
        }
        inventory_rows: list[dict[str, Any]] = []
        for recording in records:
            layout = read_motive_layout(recording.path)
            inventory_rows.append(
                {
                    "group_id": recording.group_id,
                    "label": recording.label,
                    "scenario": recording.scenario,
                    "material": recording.material,
                    "size": recording.size,
                    "available_marker_count": len(layout.markers),
                    "header_row_count": layout.header_row_count,
                }
            )
        inventory = {
            "schema": (
                "prob4d.tracking-cloth-continuous-risk-calibrated-inventory.v2"
            ),
            "dataset_root_name": dataset_root.name,
            "csv_file_count": len(csv_paths),
            "classification": classification,
            "common_marker_count": len(common_labels),
            "selected_marker_labels": list(marker_triplet),
            "selection_details": selection_details,
            "selection_metadata": selection_metadata,
            "source_partition": split_receipt,
            "file_manifest_sha256": _sha256_bytes(
                _canonical_bytes(inventory_rows)
            ),
            "files": inventory_rows,
        }
        _write_json(output_dir / "inventory.json", inventory)
        _write_json(output_dir / "source_partition.json", split_receipt)

        horizons = [
            int(value) for value in protocol["geometry"]["horizons_frames"]
        ]
        angle_normalizer = float(
            protocol["calibration"]["angle_normalizer_rad"]
        )
        numerical_slack = float(
            protocol["numerics"]["absolute_loss_slack_mm2"]
        )

        source_trajectories: dict[str, Trajectory] = {}
        for role in ("support_calibration", "risk_calibration"):
            for recording in partitions[role]:
                source_trajectories[recording.group_id] = _read_trajectory(
                    recording,
                    marker_triplet,
                )

        support_cases: dict[tuple[str, int], list[Case]] = {}
        risk_cases: dict[tuple[str, int], list[Case]] = {}
        minimum_cases = int(
            protocol["geometry"]["minimum_cases_per_recording"]
        )
        for role, destination in (
            ("support_calibration", support_cases),
            ("risk_calibration", risk_cases),
        ):
            for recording in partitions[role]:
                trajectory = source_trajectories[recording.group_id]
                for horizon in horizons:
                    cases = _make_cases(
                        trajectory,
                        horizon_frames=horizon,
                        angle_normalizer_rad=angle_normalizer,
                        selection_prefix_frames=int(
                            protocol["geometry"]["selection_prefix_frames"]
                        ),
                        maximum_cases=int(
                            protocol["geometry"]["maximum_cases_per_recording"]
                        ),
                        minimum_anchor_distance_mm=float(
                            protocol["geometry"]["minimum_anchor_distance_mm"]
                        ),
                        minimum_probe_radius_mm=float(
                            protocol["geometry"]["minimum_probe_radius_mm"]
                        ),
                    )
                    if len(cases) < minimum_cases:
                        raise ValueError(
                            f"{recording.group_id} horizon {horizon} "
                            f"has {len(cases)} cases"
                        )
                    destination[(recording.group_id, horizon)] = cases

        calibration_bundle: dict[str, Any] = {
            "schema": (
                "prob4d.tracking-cloth-continuous-risk-calibration.v2"
            ),
            "selected_marker_labels": list(marker_triplet),
            "horizons": {},
        }
        risk_calibrations: dict[int, SignedGroupConformalUpperBound] = {}
        support_thresholds: dict[int, float] = {}
        for horizon in horizons:
            support_group_scores = [
                _support_recording_score(
                    support_cases[(recording.group_id, horizon)],
                    within_recording_quantile=float(
                        protocol["calibration"][
                            "within_recording_support_quantile"
                        ]
                    ),
                )
                for recording in partitions["support_calibration"]
            ]
            support_calibration = calibrate_group_conformal_upper_bound(
                support_group_scores,
                miscoverage=float(
                    protocol["calibration"]["support_group_miscoverage"]
                ),
            )
            if not support_calibration.finite or support_calibration.threshold is None:
                raise ValueError("finite support threshold unavailable")
            support_threshold = float(support_calibration.threshold)
            support_thresholds[horizon] = support_threshold

            risk_group_scores: list[float] = []
            risk_group_rows: list[dict[str, Any]] = []
            for recording in partitions["risk_calibration"]:
                score, values = _risk_recording_score(
                    risk_cases[(recording.group_id, horizon)],
                    support_threshold=support_threshold,
                    angle_normalizer_rad=angle_normalizer,
                    numerical_slack=numerical_slack,
                )
                risk_group_scores.append(score)
                risk_group_rows.append(
                    {
                        "group_id": recording.group_id,
                        "material": recording.material,
                        "case_count": len(values),
                        "signed_max_advantage_deficit_mm2": score,
                    }
                )
            risk_calibration = calibrate_signed_group_upper_bound(
                risk_group_scores,
                miscoverage=float(
                    protocol["calibration"]["risk_group_miscoverage"]
                ),
            )
            if not risk_calibration.finite or risk_calibration.threshold is None:
                raise ValueError("finite signed-risk threshold unavailable")
            risk_calibrations[horizon] = risk_calibration
            calibration_bundle["horizons"][str(horizon)] = {
                "support_group_scores": support_group_scores,
                "support": support_calibration.summary(),
                "risk_group_scores": risk_group_scores,
                "risk_group_rows": risk_group_rows,
                "risk": risk_calibration.summary(),
            }

        calibration_bytes = _canonical_bytes(calibration_bundle)
        (output_dir / "calibration.json").write_bytes(calibration_bytes)
        calibration_seal = _sha256_bytes(calibration_bytes)

        evaluations: list[dict[str, Any]] = []
        # Target trajectories are opened only after the calibration seal.
        for recording in targets:
            trajectory = _read_trajectory(recording, marker_triplet)
            for horizon in horizons:
                cases = _make_cases(
                    trajectory,
                    horizon_frames=horizon,
                    angle_normalizer_rad=angle_normalizer,
                    selection_prefix_frames=int(
                        protocol["geometry"]["selection_prefix_frames"]
                    ),
                    maximum_cases=int(
                        protocol["geometry"]["maximum_cases_per_recording"]
                    ),
                    minimum_anchor_distance_mm=float(
                        protocol["geometry"]["minimum_anchor_distance_mm"]
                    ),
                    minimum_probe_radius_mm=float(
                        protocol["geometry"]["minimum_probe_radius_mm"]
                    ),
                )
                if len(cases) < minimum_cases:
                    raise ValueError(
                        f"target {recording.group_id} horizon {horizon} "
                        f"has only {len(cases)} cases"
                    )
                metrics = _evaluate_cases(
                    cases,
                    support_threshold=support_thresholds[horizon],
                    risk_calibration=risk_calibrations[horizon],
                    angle_normalizer_rad=angle_normalizer,
                    required_margin_mm2=float(
                        protocol["decision"]["required_advantage_margin_mm2"]
                    ),
                    numerical_slack=numerical_slack,
                    within_recording_quantile=float(
                        protocol["calibration"][
                            "within_recording_support_quantile"
                        ]
                    ),
                )
                evaluations.append(
                    {
                        "group_id": recording.group_id,
                        "scenario": recording.scenario,
                        "material": recording.material,
                        "horizon_frames": horizon,
                        "unit_scale_to_mm": trajectory.unit_scale_to_mm,
                        "row_count": trajectory.row_count,
                        "calibration_seal_sha256": calibration_seal,
                        "recording_support_score_covered": (
                            metrics["recording_support_score"]
                            <= support_thresholds[horizon] + numerical_slack
                        ),
                        "metrics": metrics,
                    }
                )

        aggregate = _aggregate(evaluations, calibration_bundle, protocol)
        result = {
            "schema": SCHEMA,
            "status": (
                "evaluated-continuous-risk-calibrated-so2-positive"
                if aggregate["overall_positive"]
                else "evaluated-continuous-risk-calibrated-so2-negative"
            ),
            "source_revision": source_revision,
            "claim_boundary": protocol["claim_boundary"],
            "information_order": {
                "target_headers_used_before_seal": True,
                "target_trajectories_opened_after_calibration_seal": True,
                "target_side_threshold_tuning_allowed": False,
                "prior_related_target_access": True,
                "independent_confirmation": False,
            },
            "inventory": {
                key: value for key, value in inventory.items() if key != "files"
            },
            "calibration_seal_sha256": calibration_seal,
            "evaluation_count": len(evaluations),
            "evaluations": evaluations,
            "aggregate": aggregate,
        }

    result_bytes = _canonical_bytes(result)
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    manifest = {
        "schema": (
            "prob4d.tracking-cloth-continuous-risk-calibrated-manifest.v2"
        ),
        "source_revision": source_revision,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "result_sha256": _sha256_bytes(result_bytes),
        "inventory_sha256": _sha256_file(output_dir / "inventory.json"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "raw_trajectory_payload_copied": False,
    }
    if (output_dir / "calibration.json").is_file():
        manifest["calibration_sha256"] = _sha256_file(
            output_dir / "calibration.json"
        )
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "summary.md").write_text(
        _summary(result),
        encoding="utf-8",
    )
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0 if result["status"] in TERMINAL_STATUSES else 3


if __name__ == "__main__":
    raise SystemExit(main())
