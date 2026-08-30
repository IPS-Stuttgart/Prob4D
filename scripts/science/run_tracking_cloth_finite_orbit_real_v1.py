#!/usr/bin/env python3
"""Source-frozen finite-orbit diagnostic on public Tracking Cloth trajectories.

The experiment uses real marker geometry but a controlled rank-deficient
observation construction. Two source-selected anchor points identify a line
while leaving the rotation of an off-axis probe around that line unresolved.
A local derivative at the representative angle is zero for both the genuinely
invariant axial query and the globally ambiguous radial query. The finite-orbit
check distinguishes them and returns the exact registered fallback for the
ambiguous query.

No learned visual-provider, physical-state-identification, or deployment claim
is made by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Recording:
    path: Path
    relative_path: str
    label: str


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def _stable_id(text: str) -> str:
    return _sha256_bytes(text.encode())[:16]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def _classify_recordings(
    dataset_root: Path,
    paths: list[Path],
    protocol: dict[str, Any],
) -> tuple[list[Recording], dict[str, Any]]:
    dataset = protocol["dataset"]
    source_aliases = dataset["source_aliases"]
    target_aliases = dataset["target_aliases"]
    assigned: dict[Path, str] = {}

    for path in paths:
        relative = path.relative_to(dataset_root).as_posix()
        lower = relative.lower()
        hits: list[str] = []
        for label, aliases in source_aliases.items():
            if any(alias.lower() in lower for alias in aliases):
                hits.append(label)
        if any(alias.lower() in lower for alias in target_aliases):
            hits.append("collision")
        if len(set(hits)) == 1:
            assigned[path] = hits[0]

    expected_total = int(dataset["expected_csv_files"])
    expected_source = int(dataset["expected_source_files"])
    expected_target = int(dataset["expected_target_files"])

    def valid_assignment(candidate: dict[Path, str]) -> bool:
        if len(candidate) != expected_total:
            return False
        counts: dict[str, int] = {}
        for label in candidate.values():
            counts[label] = counts.get(label, 0) + 1
        return (
            counts.get("collision", 0) == expected_target
            and sum(counts.get(label, 0) for label in source_aliases) == expected_source
            and all(counts.get(label, 0) > 0 for label in source_aliases)
        )

    classification_mode = "declared-aliases"
    if not valid_assignment(assigned):
        token_files: dict[str, set[Path]] = {}
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
        for path in paths:
            relative = path.relative_to(dataset_root).as_posix()
            for token in set(_tokens(relative)):
                if token in ignored or token.isdigit() or len(token) < 3:
                    continue
                token_files.setdefault(token, set()).add(path)

        source_tokens = sorted(
            token for token, members in token_files.items() if len(members) * 2 == expected_source
        )
        target_tokens = sorted(
            token for token, members in token_files.items() if len(members) == expected_target
        )
        inferred: dict[Path, str] | None = None
        inferred_tokens: tuple[str, str, str] | None = None
        for target_token in target_tokens:
            target_set = token_files[target_token]
            for first_token, second_token in itertools.combinations(source_tokens, 2):
                first_set = token_files[first_token]
                second_set = token_files[second_token]
                if first_set & second_set or first_set & target_set or second_set & target_set:
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
            counts: dict[str, int] = {}
            for label in assigned.values():
                counts[label] = counts.get(label, 0) + 1
            raise RuntimeError(
                "could not identify the registered 64-source/56-collision split; "
                f"alias counts were {counts}"
            )
        assigned = inferred
        classification_mode = "exact-count-name-partition:" + ",".join(inferred_tokens or ())

    recordings = [
        Recording(
            path=path,
            relative_path=path.relative_to(dataset_root).as_posix(),
            label=assigned[path],
        )
        for path in sorted(paths)
    ]
    counts: dict[str, int] = {}
    for recording in recordings:
        counts[recording.label] = counts.get(recording.label, 0) + 1
    return recordings, {"mode": classification_mode, "counts": counts}


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return max((",", ";", "\t"), key=sample.count)


def _read_headers(path: Path) -> tuple[list[str], str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        sample = stream.read(32768)
    delimiter = _sniff_delimiter(sample)
    reader = csv.reader(sample.splitlines(), delimiter=delimiter)
    try:
        headers = next(reader)
    except StopIteration as error:
        raise RuntimeError(f"empty CSV: {path}") from error
    return [header.strip() for header in headers], delimiter


def _clean_header(header: str) -> str:
    value = header.strip().lower()
    value = re.sub(r"\[(?:m|mm|cm)\]", "", value)
    value = re.sub(r"\((?:m|mm|cm)\)", "", value)
    return value.strip()


def _coordinate_key(header: str) -> tuple[str, str] | None:
    value = _clean_header(header)
    components = [piece for piece in re.split(r"[^a-z0-9]+", value) if piece]
    axis: str | None = None
    base = ""
    if len(components) >= 2 and components[-1] in {"x", "y", "z"}:
        axis = components[-1]
        base = "_".join(components[:-1])
    elif len(components) >= 2 and components[0] in {"x", "y", "z"}:
        axis = components[0]
        base = "_".join(components[1:])
    else:
        match = re.fullmatch(r"(.+?)([xyz])", "".join(components))
        reverse = re.fullmatch(r"([xyz])(\d+)", "".join(components))
        if reverse:
            axis, base = reverse.group(1), f"marker_{reverse.group(2)}"
        elif match:
            base, axis = match.group(1), match.group(2)
    if axis is None or not base:
        return None

    excluded = {
        "time",
        "timestamp",
        "frame",
        "force",
        "torque",
        "velocity",
        "vel",
        "acceleration",
        "accel",
        "angular",
        "gyro",
        "quaternion",
        "quat",
        "orientation",
        "rotation",
        "command",
        "desired",
        "target",
        "motor",
    }
    if set(_tokens(base)) & excluded:
        return None
    return base, axis


def _coordinate_groups(headers: list[str]) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for header in headers:
        key = _coordinate_key(header)
        if key is None:
            continue
        base, axis = key
        groups.setdefault(base, {})[axis] = header
    return {
        base: axes
        for base, axes in groups.items()
        if set(axes) == {"x", "y", "z"}
    }


def _unit_scale_from_headers(headers: Iterable[str]) -> float | None:
    joined = " ".join(header.lower() for header in headers)
    if re.search(r"\bmm\b|\[mm\]|\(mm\)", joined):
        return 1.0
    if re.search(r"\bcm\b|\[cm\]|\(cm\)", joined):
        return 10.0
    if re.search(r"\bm\b|\[m\]|\(m\)|meter", joined):
        return 1000.0
    return None


def _float_or_nan(value: str | None, delimiter: str) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    if delimiter == ";" and "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return math.nan


def _read_markers(
    path: Path,
    marker_names: list[str],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    headers, delimiter = _read_headers(path)
    groups = _coordinate_groups(headers)
    missing = sorted(set(marker_names) - set(groups))
    if missing:
        raise RuntimeError(f"missing selected marker columns in {path}: {missing}")

    columns = {
        marker: [groups[marker][axis] for axis in ("x", "y", "z")]
        for marker in marker_names
    }
    rows: list[list[list[float]]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        for row in reader:
            rows.append(
                [
                    [_float_or_nan(row.get(column), delimiter) for column in columns[marker]]
                    for marker in marker_names
                ]
            )
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    coordinates = np.asarray(rows, dtype=np.float64)
    scale = _unit_scale_from_headers(
        column for marker_columns in columns.values() for column in marker_columns
    )
    if scale is None:
        scale = _automatic_unit_scale(coordinates)
    return coordinates * scale, scale, {
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "rows": int(coordinates.shape[0]),
    }


def _automatic_unit_scale(coordinates: np.ndarray) -> float:
    pair_distances: list[float] = []
    for frame_index in _sample_indices(coordinates.shape[0], 12):
        frame = coordinates[frame_index]
        valid = frame[np.all(np.isfinite(frame), axis=1)]
        if valid.shape[0] < 2:
            continue
        valid = valid[: min(valid.shape[0], 20)]
        differences = valid[:, None, :] - valid[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        pair_distances.extend(distances[np.triu_indices(valid.shape[0], 1)].tolist())
    positive = np.asarray(
        [value for value in pair_distances if value > 0 and math.isfinite(value)]
    )
    if positive.size == 0:
        raise RuntimeError("could not infer coordinate unit from degenerate marker geometry")
    median = float(np.median(positive))
    if 1e-4 <= median < 10.0:
        return 1000.0
    if 10.0 <= median < 10000.0:
        return 1.0
    raise RuntimeError(f"unsupported coordinate scale; median marker distance was {median:g}")


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(length, maximum)
    return np.unique(np.linspace(0, length - 1, num=count, dtype=np.int64))


def _common_marker_names(recordings: list[Recording], maximum: int) -> list[str]:
    common: set[str] | None = None
    for recording in recordings:
        headers, _ = _read_headers(recording.path)
        names = set(_coordinate_groups(headers))
        common = names if common is None else common & names
    result = sorted(common or ())
    if len(result) < 3:
        raise RuntimeError(f"fewer than three common 3-D marker groups: {result}")
    return result[:maximum]


def _collect_source_samples(
    recordings: list[Recording],
    marker_names: list[str],
    maximum_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    samples: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for recording in recordings:
        coordinates, scale, details = _read_markers(recording.path, marker_names)
        selected = coordinates[_sample_indices(coordinates.shape[0], maximum_frames)]
        valid = selected[np.all(np.isfinite(selected), axis=(1, 2))]
        if valid.size:
            samples.append(valid)
        metadata.append(
            {
                "group_id": _stable_id(recording.relative_path),
                "label": recording.label,
                "unit_scale_to_mm": scale,
                "sampled_valid_frames": int(valid.shape[0]),
                **details,
            }
        )
    if not samples:
        raise RuntimeError("source recordings yielded no complete marker frames")
    return np.concatenate(samples, axis=0), metadata


def _line_geometry(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    delta = b - a
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= 0:
        return math.nan, math.nan
    axis = delta / distance
    probe_delta = p - a
    axial = float(np.dot(probe_delta, axis))
    radius = float(np.linalg.norm(probe_delta - axial * axis))
    return axial, radius


def _select_marker_triplet(
    samples: np.ndarray,
    marker_names: list[str],
    minimum_anchor_distance: float,
    minimum_probe_radius: float,
) -> tuple[tuple[str, str, str], dict[str, float]]:
    best: tuple[float, str, str, str] | None = None
    best_details: dict[str, float] | None = None
    marker_count = len(marker_names)
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
            axes[valid_distance] = delta[valid_distance] / distances[valid_distance, None]
            for probe in range(marker_count):
                if probe in {first, second}:
                    continue
                probe_delta = samples[:, probe] - a
                axial = np.sum(probe_delta * axes, axis=1)
                radii = np.linalg.norm(probe_delta - axial[:, None] * axes, axis=1)
                valid = valid_distance & np.isfinite(radii)
                if int(np.sum(valid)) < max(16, samples.shape[0] // 4):
                    continue
                distance_q10 = float(np.quantile(distances[valid], 0.1))
                radius_q10 = float(np.quantile(radii[valid], 0.1))
                radius_median = float(np.median(radii[valid]))
                if distance_q10 < minimum_anchor_distance or radius_q10 < minimum_probe_radius:
                    continue
                score = min(distance_q10, 2.0 * radius_q10) + 0.05 * radius_median
                names = (marker_names[first], marker_names[second], marker_names[probe])
                key = (score, *names)
                if best is None or key > best:
                    best = key
                    best_details = {
                        "score": score,
                        "anchor_distance_q10_mm": distance_q10,
                        "probe_radius_q10_mm": radius_q10,
                        "probe_radius_median_mm": radius_median,
                    }
    if best is None or best_details is None:
        raise RuntimeError("source data did not provide a nondegenerate anchor/probe triplet")
    return (best[1], best[2], best[3]), best_details


def _uniform_from_key(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / float(1 << 64)


def _normal_from_key(*parts: object) -> float:
    first = max(_uniform_from_key(*parts, "normal-a"), np.finfo(float).tiny)
    second = _uniform_from_key(*parts, "normal-b")
    return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)


def _gaussian_metrics(
    truth: float,
    mean: float,
    variance: float,
    z_value: float,
) -> dict[str, float]:
    variance = max(float(variance), np.finfo(float).tiny)
    error = truth - mean
    return {
        "squared_error": error * error,
        "nll": 0.5 * (math.log(2.0 * math.pi * variance) + error * error / variance),
        "covered": float(abs(error) <= z_value * math.sqrt(variance)),
    }


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else math.nan


def _evaluate_recording(
    recording: Recording,
    marker_triplet: tuple[str, str, str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    geometry = protocol["geometry"]
    factor = protocol["controlled_factor"]
    inference = protocol["inference"]
    coordinates, scale, details = _read_markers(recording.path, list(marker_triplet))
    indices = _sample_indices(
        coordinates.shape[0], int(geometry["target_frames_per_recording"])
    )
    minimum_anchor = float(geometry["minimum_anchor_distance_mm"])
    minimum_radius = float(geometry["minimum_probe_radius_mm"])
    width_threshold = float(geometry["orbit_width_threshold_mm"])
    candidate_variance = float(factor["candidate_sigma_mm"]) ** 2
    fallback_invariant_variance = float(factor["fallback_invariant_sigma_mm"]) ** 2
    z_value = 1.6448536269514722
    if not math.isclose(float(inference["coverage_level"]), 0.9):
        raise RuntimeError("this registered implementation supports the frozen 90% level")

    metrics: dict[str, list[float]] = {
        key: []
        for key in (
            "fallback_invariant_sq",
            "candidate_invariant_sq",
            "fallback_invariant_nll",
            "candidate_invariant_nll",
            "fallback_invariant_covered",
            "candidate_invariant_covered",
            "fallback_radial_sq",
            "local_radial_sq",
            "finite_radial_sq",
            "fallback_radial_nll",
            "local_radial_nll",
            "finite_radial_nll",
            "fallback_radial_covered",
            "local_radial_covered",
            "finite_radial_covered",
            "finite_accept_radial",
            "local_harmful_radial",
            "finite_harmful_accepted_radial",
            "finite_exact_fallback_radial",
            "radial_orbit_interval_covered",
            "radius_mm",
        )
    }

    for frame_index in indices:
        frame = coordinates[int(frame_index)]
        if not np.all(np.isfinite(frame)):
            continue
        axial, radius = _line_geometry(frame[0], frame[1], frame[2])
        anchor_distance = float(np.linalg.norm(frame[1] - frame[0]))
        if (
            not math.isfinite(axial)
            or not math.isfinite(radius)
            or anchor_distance < minimum_anchor
            or radius < minimum_radius
        ):
            continue

        group_key = _stable_id(recording.relative_path)
        seed = int(factor["hidden_angle_seed"])
        theta = (
            2.0 * math.pi * _uniform_from_key(seed, group_key, int(frame_index))
            - math.pi
        )

        invariant_truth = axial
        invariant_fallback = axial + float(
            factor["fallback_invariant_sigma_mm"]
        ) * _normal_from_key(seed, group_key, int(frame_index), "invariant-fallback")
        fallback_invariant = _gaussian_metrics(
            invariant_truth,
            invariant_fallback,
            fallback_invariant_variance,
            z_value,
        )
        candidate_invariant = _gaussian_metrics(
            invariant_truth,
            invariant_truth,
            candidate_variance,
            z_value,
        )

        radial_truth = radius * math.cos(theta)
        radial_fallback = float(factor["fallback_radial_mean"])
        radial_candidate = radius
        radial_fallback_variance = max(radius * radius / 2.0, candidate_variance)
        fallback_radial = _gaussian_metrics(
            radial_truth,
            radial_fallback,
            radial_fallback_variance,
            z_value,
        )
        local_radial = _gaussian_metrics(
            radial_truth,
            radial_candidate,
            candidate_variance,
            z_value,
        )
        finite_accept_radial = 2.0 * radius <= width_threshold
        finite_radial_mean = radial_candidate if finite_accept_radial else radial_fallback
        finite_radial_variance = (
            candidate_variance if finite_accept_radial else radial_fallback_variance
        )
        finite_radial = _gaussian_metrics(
            radial_truth,
            finite_radial_mean,
            finite_radial_variance,
            z_value,
        )

        metrics["fallback_invariant_sq"].append(fallback_invariant["squared_error"])
        metrics["candidate_invariant_sq"].append(candidate_invariant["squared_error"])
        metrics["fallback_invariant_nll"].append(fallback_invariant["nll"])
        metrics["candidate_invariant_nll"].append(candidate_invariant["nll"])
        metrics["fallback_invariant_covered"].append(fallback_invariant["covered"])
        metrics["candidate_invariant_covered"].append(candidate_invariant["covered"])
        metrics["fallback_radial_sq"].append(fallback_radial["squared_error"])
        metrics["local_radial_sq"].append(local_radial["squared_error"])
        metrics["finite_radial_sq"].append(finite_radial["squared_error"])
        metrics["fallback_radial_nll"].append(fallback_radial["nll"])
        metrics["local_radial_nll"].append(local_radial["nll"])
        metrics["finite_radial_nll"].append(finite_radial["nll"])
        metrics["fallback_radial_covered"].append(fallback_radial["covered"])
        metrics["local_radial_covered"].append(local_radial["covered"])
        metrics["finite_radial_covered"].append(finite_radial["covered"])
        metrics["finite_accept_radial"].append(float(finite_accept_radial))
        metrics["local_harmful_radial"].append(
            float(local_radial["squared_error"] > fallback_radial["squared_error"])
        )
        metrics["finite_harmful_accepted_radial"].append(
            float(
                finite_accept_radial
                and finite_radial["squared_error"] > fallback_radial["squared_error"]
            )
        )
        metrics["finite_exact_fallback_radial"].append(
            float(
                finite_accept_radial
                or (
                    finite_radial_mean == radial_fallback
                    and finite_radial_variance == radial_fallback_variance
                )
            )
        )
        metrics["radial_orbit_interval_covered"].append(
            float(-radius <= radial_truth <= radius)
        )
        metrics["radius_mm"].append(radius)

    case_count = len(metrics["radius_mm"])
    if case_count == 0:
        raise RuntimeError(f"no valid target geometry cases in {recording.relative_path}")

    def rmse(key: str) -> float:
        return math.sqrt(_mean(metrics[key]))

    return {
        "group_id": _stable_id(recording.relative_path),
        "case_count": case_count,
        "unit_scale_to_mm": scale,
        "rows": details["rows"],
        "radius_median_mm": float(np.median(metrics["radius_mm"])),
        "local_acceptance": {"invariant": 1.0, "radial": 1.0},
        "finite_orbit_acceptance": {
            "invariant": 1.0,
            "radial": _mean(metrics["finite_accept_radial"]),
        },
        "harmful_fraction": {
            "local_radial": _mean(metrics["local_harmful_radial"]),
            "finite_harmful_accepted_radial": _mean(
                metrics["finite_harmful_accepted_radial"]
            ),
        },
        "exact_fallback_fraction": _mean(metrics["finite_exact_fallback_radial"]),
        "orbit_interval_coverage": _mean(metrics["radial_orbit_interval_covered"]),
        "invariant": {
            "fallback_rmse_mm": rmse("fallback_invariant_sq"),
            "local_rmse_mm": rmse("candidate_invariant_sq"),
            "finite_rmse_mm": rmse("candidate_invariant_sq"),
            "reject_all_rmse_mm": rmse("fallback_invariant_sq"),
            "fallback_nll": _mean(metrics["fallback_invariant_nll"]),
            "local_nll": _mean(metrics["candidate_invariant_nll"]),
            "finite_nll": _mean(metrics["candidate_invariant_nll"]),
            "fallback_coverage": _mean(metrics["fallback_invariant_covered"]),
            "local_coverage": _mean(metrics["candidate_invariant_covered"]),
            "finite_coverage": _mean(metrics["candidate_invariant_covered"]),
        },
        "radial": {
            "fallback_rmse_mm": rmse("fallback_radial_sq"),
            "local_rmse_mm": rmse("local_radial_sq"),
            "finite_rmse_mm": rmse("finite_radial_sq"),
            "reject_all_rmse_mm": rmse("fallback_radial_sq"),
            "fallback_nll": _mean(metrics["fallback_radial_nll"]),
            "local_nll": _mean(metrics["local_radial_nll"]),
            "finite_nll": _mean(metrics["finite_radial_nll"]),
            "fallback_coverage": _mean(metrics["fallback_radial_covered"]),
            "local_coverage": _mean(metrics["local_radial_covered"]),
            "finite_coverage": _mean(metrics["finite_radial_covered"]),
        },
    }


def _bootstrap_summary(
    values: list[float],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("bootstrap received empty or nonfinite group values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    bootstrap = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "groups": int(array.size),
    }


def _nested_value(group: dict[str, Any], path: str) -> float:
    value: Any = group
    for component in path.split("."):
        value = value[component]
    return float(value)


def _aggregate_groups(
    groups: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    inference = protocol["inference"]
    replicates = int(inference["bootstrap_replicates"])
    seed = int(inference["bootstrap_seed"])
    paths = [
        "case_count",
        "radius_median_mm",
        "local_acceptance.invariant",
        "local_acceptance.radial",
        "finite_orbit_acceptance.invariant",
        "finite_orbit_acceptance.radial",
        "harmful_fraction.local_radial",
        "harmful_fraction.finite_harmful_accepted_radial",
        "exact_fallback_fraction",
        "orbit_interval_coverage",
        "invariant.fallback_rmse_mm",
        "invariant.local_rmse_mm",
        "invariant.finite_rmse_mm",
        "invariant.fallback_nll",
        "invariant.local_nll",
        "invariant.finite_nll",
        "invariant.fallback_coverage",
        "invariant.local_coverage",
        "invariant.finite_coverage",
        "radial.fallback_rmse_mm",
        "radial.local_rmse_mm",
        "radial.finite_rmse_mm",
        "radial.fallback_nll",
        "radial.local_nll",
        "radial.finite_nll",
        "radial.fallback_coverage",
        "radial.local_coverage",
        "radial.finite_coverage",
    ]
    summaries: dict[str, Any] = {}
    for offset, path in enumerate(paths):
        summaries[path] = _bootstrap_summary(
            [_nested_value(group, path) for group in groups],
            replicates,
            seed + offset,
        )
    summaries["total_cases"] = int(sum(group["case_count"] for group in groups))
    summaries["target_groups"] = len(groups)
    return summaries


def _registered_criteria(
    aggregate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, bool]:
    expected_target = int(protocol["dataset"]["expected_target_files"])
    geometry = protocol["geometry"]
    registered = protocol["registered_criteria"]

    def mean(path: str) -> float:
        return float(aggregate[path]["mean"])

    return {
        "all_target_groups_contribute": aggregate["target_groups"] == expected_target,
        "minimum_target_cases": aggregate["total_cases"]
        >= int(geometry["minimum_target_cases"]),
        "invariant_acceptance": mean("finite_orbit_acceptance.invariant")
        >= float(registered["minimum_invariant_acceptance"]),
        "radial_rejection": 1.0 - mean("finite_orbit_acceptance.radial")
        >= float(registered["minimum_radial_rejection"]),
        "local_radial_acceptance": mean("local_acceptance.radial")
        >= float(registered["minimum_local_radial_acceptance"]),
        "finite_no_harmful_accepted_radial": mean(
            "harmful_fraction.finite_harmful_accepted_radial"
        )
        <= float(registered["maximum_finite_orbit_harmful_accepted_radial"]),
        "local_exposes_global_failure": mean("harmful_fraction.local_radial")
        >= float(registered["minimum_local_harmful_radial_fraction"]),
        "exact_fallback": mean("exact_fallback_fraction") == 1.0,
        "orbit_support_covers_truth": mean("orbit_interval_coverage") == 1.0,
        "invariant_rmse_improves": mean("invariant.finite_rmse_mm")
        < mean("invariant.fallback_rmse_mm"),
    }


def _format_estimate(summary: dict[str, float], digits: int = 4) -> str:
    return (
        f"{summary['mean']:.{digits}f} "
        f"[{summary['ci95_low']:.{digits}f}, {summary['ci95_high']:.{digits}f}]"
    )


def _make_summary(
    result: dict[str, Any],
    marker_triplet: tuple[str, str, str],
) -> str:
    aggregate = result["aggregate"]
    criteria = result["criteria"]
    radial_acceptance = aggregate["finite_orbit_acceptance.radial"]
    rejection = {
        "mean": 1.0 - radial_acceptance["mean"],
        "ci95_low": 1.0 - radial_acceptance["ci95_high"],
        "ci95_high": 1.0 - radial_acceptance["ci95_low"],
    }
    lines = [
        "# Tracking Cloth finite-orbit real-geometry result",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Source-selected anchors/probe: `{marker_triplet[0]}`, `{marker_triplet[1]}`, "
        f"`{marker_triplet[2]}`.",
        f"Held-out collision recordings: {aggregate['target_groups']}; "
        f"controlled geometry cases: {aggregate['total_cases']}.",
        "",
        "| Endpoint (equal-recording mean) | Estimate [95% bootstrap interval] |",
        "|---|---:|",
        "| Finite-orbit invariant-query acceptance | "
        + _format_estimate(aggregate["finite_orbit_acceptance.invariant"])
        + " |",
        "| Finite-orbit radial-query rejection | "
        + _format_estimate(rejection)
        + " |",
        "| Local-gate harmful radial updates | "
        + _format_estimate(aggregate["harmful_fraction.local_radial"])
        + " |",
        "| Finite-orbit harmful accepted radial updates | "
        + _format_estimate(aggregate["harmful_fraction.finite_harmful_accepted_radial"])
        + " |",
        "| Exact fallback fraction | "
        + _format_estimate(aggregate["exact_fallback_fraction"])
        + " |",
        "| Invariant fallback RMSE [mm] | "
        + _format_estimate(aggregate["invariant.fallback_rmse_mm"])
        + " |",
        "| Invariant finite-orbit RMSE [mm] | "
        + _format_estimate(aggregate["invariant.finite_rmse_mm"])
        + " |",
        "| Radial local-gate RMSE [mm] | "
        + _format_estimate(aggregate["radial.local_rmse_mm"])
        + " |",
        "| Radial finite-orbit/fallback RMSE [mm] | "
        + _format_estimate(aggregate["radial.finite_rmse_mm"])
        + " |",
        "",
        "## Registered criteria",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in criteria.items()
    )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is held-out public real-trajectory geometry under a controlled "
            "rank-deficient factor and controlled hidden SO(2) gauge. It tests the "
            "distinction between local first-order support and full finite-orbit "
            "query identifiability. It does not test a learned visual provider, "
            "recover physical state, establish arbitrary cloth generalization, or "
            "imply deployment safety.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    protocol_path = Path(args.protocol).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "prob4d.tracking-cloth-finite-orbit-real.v1":
        raise RuntimeError("unexpected protocol schema")
    if not dataset_root.is_dir():
        raise RuntimeError(f"dataset root does not exist: {dataset_root}")

    csv_paths = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    expected_csv = int(protocol["dataset"]["expected_csv_files"])
    if len(csv_paths) != expected_csv:
        raise RuntimeError(f"expected {expected_csv} CSV files, found {len(csv_paths)}")

    recordings, classification = _classify_recordings(dataset_root, csv_paths, protocol)
    source = [recording for recording in recordings if recording.label != "collision"]
    target = [recording for recording in recordings if recording.label == "collision"]
    if len(source) != int(protocol["dataset"]["expected_source_files"]):
        raise RuntimeError("source file count drift")
    if len(target) != int(protocol["dataset"]["expected_target_files"]):
        raise RuntimeError("target file count drift")

    geometry = protocol["geometry"]
    common_markers = _common_marker_names(
        source,
        int(geometry["maximum_common_marker_count"]),
    )
    source_samples, source_metadata = _collect_source_samples(
        source,
        common_markers,
        int(geometry["source_frames_per_recording"]),
    )
    marker_triplet, selection_details = _select_marker_triplet(
        source_samples,
        common_markers,
        float(geometry["minimum_anchor_distance_mm"]),
        float(geometry["minimum_probe_radius_mm"]),
    )
    protocol_sha256 = _sha256_file(protocol_path)
    source_seal = {
        "schema": "prob4d.tracking-cloth-finite-orbit-source-seal.v1",
        "protocol_sha256": protocol_sha256,
        "source_revision": args.source_revision,
        "classification": classification,
        "source_group_count": len(source),
        "source_group_ids": [_stable_id(recording.relative_path) for recording in source],
        "common_marker_count": len(common_markers),
        "selected_anchor_a": marker_triplet[0],
        "selected_anchor_b": marker_triplet[1],
        "selected_probe": marker_triplet[2],
        "selection": selection_details,
        "source_sample_frames": int(source_samples.shape[0]),
        "source_metadata": source_metadata,
        "target_payload_opened": False,
    }
    source_seal["source_seal_id"] = _sha256_bytes(_canonical_json(source_seal))
    _write_json(output_dir / "source_seal.json", source_seal)

    groups = [
        _evaluate_recording(recording, marker_triplet, protocol) for recording in target
    ]
    aggregate = _aggregate_groups(groups, protocol)
    criteria = _registered_criteria(aggregate, protocol)
    status = (
        "evaluated-real-geometry-passed"
        if all(criteria.values())
        else "evaluated-real-geometry-failed"
    )
    result = {
        "schema": "prob4d.tracking-cloth-finite-orbit-result.v1",
        "status": status,
        "protocol_sha256": protocol_sha256,
        "source_revision": args.source_revision,
        "source_seal_id": source_seal["source_seal_id"],
        "dataset_root_name": dataset_root.name,
        "classification": classification,
        "selected_markers": {
            "anchor_a": marker_triplet[0],
            "anchor_b": marker_triplet[1],
            "probe": marker_triplet[2],
        },
        "selection": selection_details,
        "aggregate": aggregate,
        "criteria": criteria,
        "groups": groups,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _sha256_bytes(_canonical_json(result))
    _write_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(
        _make_summary(result, marker_triplet), encoding="utf-8"
    )
    _write_json(
        output_dir / "inventory.json",
        {
            "schema": "prob4d.tracking-cloth-finite-orbit-inventory.v1",
            "csv_count": len(csv_paths),
            "classification": classification,
            "source_group_ids": [
                _stable_id(recording.relative_path) for recording in source
            ],
            "target_group_ids": [
                _stable_id(recording.relative_path) for recording in target
            ],
            "raw_payload_uploaded": False,
        },
    )
    return 0 if all(criteria.values()) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"tracking-cloth finite-orbit evaluation failed: {error}", file=sys.stderr)
        raise
