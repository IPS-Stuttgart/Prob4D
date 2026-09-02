"""Frozen Tracking Cloth cohort, split, and marker-selection helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prob4d.motive_csv import common_marker_labels, read_motive_layout, read_motive_markers

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Recording:
    path: Path
    relative_path: str
    group_id: str
    material: str


def _stable_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()[:16]


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token}


def _material(relative_path: str, allowed: tuple[str, ...]) -> str:
    tokens = _tokens(relative_path)
    found = [material for material in allowed if material in tokens]
    if len(found) != 1:
        raise ValueError(f"material metadata is ambiguous: {relative_path}")
    return found[0]


def _discover(
    dataset_root: Path,
    protocol: dict[str, Any],
) -> tuple[list[Recording], list[str], list[dict[str, object]]]:
    dataset = protocol["dataset"]
    all_paths = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    if len(all_paths) != dataset["expected_csv_files"]:
        raise ValueError(
            f"official CSV roster changed: {len(all_paths)} != {dataset['expected_csv_files']}"
        )
    allowed = tuple(dataset["allowed_materials"])
    cohort: list[Recording] = []
    headers: list[dict[str, object]] = []
    for path in all_paths:
        relative = path.relative_to(dataset_root).as_posix()
        tokens = _tokens(relative)
        is_self_collision = "self" in tokens and any(
            token.startswith("collision") for token in tokens
        )
        if (
            not is_self_collision
            or "a2" not in tokens
            or not any(item in tokens for item in allowed)
        ):
            continue
        material = _material(relative, allowed)
        group_id = _stable_id(relative)
        layout = read_motive_layout(path)
        cohort.append(Recording(path, relative, group_id, material))
        headers.append(
            {
                "group_id": group_id,
                "relative_path": relative,
                "material": material,
                "available_marker_count": len(layout.markers),
                "marker_labels": list(layout.marker_labels),
                "length_units": layout.length_units,
                "parser": "strict-motive-multirow-marker-v1",
            }
        )
    if len(cohort) != dataset["expected_cohort_files"]:
        raise ValueError(f"expected 27 Self-collisions recordings, found {len(cohort)}")
    actual_ids = sorted(recording.group_id for recording in cohort)
    expected_ids = sorted(dataset["prior_header_only_group_ids"])
    if actual_ids != expected_ids:
        raise ValueError("Self-collisions cohort differs from prior header-only support audit")
    common = common_marker_labels(
        [recording.path for recording in cohort],
        int(protocol["marker_selection"]["maximum_common_marker_count"]),
    )
    if len(common) < 3:
        raise ValueError("Self-collisions cohort has fewer than three common markers")
    return cohort, common, headers


def _split(
    cohort: list[Recording],
    protocol: dict[str, Any],
) -> tuple[list[Recording], list[Recording]]:
    split = protocol["split"]
    calibration: list[Recording] = []
    target: list[Recording] = []
    for material in protocol["dataset"]["allowed_materials"]:
        members = [recording for recording in cohort if recording.material == material]
        members.sort(
            key=lambda recording: hashlib.sha256(
                f"{split['salt']}\0{recording.relative_path}".encode()
            ).hexdigest()
        )
        expected = split["calibration_per_material"] + split["target_per_material"]
        if len(members) != expected:
            raise ValueError(
                f"expected {expected} {material} Self-collisions recordings, found {len(members)}"
            )
        calibration.extend(members[: split["calibration_per_material"]])
        target.extend(members[split["calibration_per_material"] :])
    calibration.sort(key=lambda recording: recording.group_id)
    target.sort(key=lambda recording: recording.group_id)
    if len(calibration) != split["expected_calibration_groups"]:
        raise ValueError("calibration group count changed")
    if len(target) != split["expected_target_groups"]:
        raise ValueError("target group count changed")
    if set(recording.group_id for recording in calibration) & set(
        recording.group_id for recording in target
    ):
        raise ValueError("calibration and target groups overlap")
    return calibration, target


def _sample_indices(length: int, maximum: int) -> NDArray[np.int64]:
    if length <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, min(length, maximum), dtype=np.int64))


def _pair_indices(length: int, maximum: int) -> NDArray[np.int64]:
    if length < 2 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 2, min(length - 1, maximum), dtype=np.int64))


def _collect_selection_samples(
    recordings: list[Recording],
    marker_labels: list[str],
    maximum_frames: int,
) -> tuple[FloatArray, list[dict[str, object]]]:
    samples: list[FloatArray] = []
    metadata: list[dict[str, object]] = []
    for recording in recordings:
        coordinates, scale, details = read_motive_markers(recording.path, marker_labels)
        selected = coordinates[_sample_indices(coordinates.shape[0], maximum_frames)]
        usable = np.count_nonzero(
            np.sum(np.all(np.isfinite(selected), axis=2), axis=1) >= 3
        )
        if selected.size:
            samples.append(selected)
        metadata.append(
            {
                "group_id": recording.group_id,
                "relative_path": recording.relative_path,
                "material": recording.material,
                "sampled_frames_with_at_least_three_markers": int(usable),
                "unit_scale_to_mm": scale,
                **details,
            }
        )
    if not samples:
        raise ValueError("calibration recordings yielded no selection frames")
    return np.concatenate(samples, axis=0), metadata


def _select_triplet(
    samples: FloatArray,
    marker_labels: list[str],
    protocol: dict[str, Any],
) -> tuple[tuple[str, str, str], dict[str, float]]:
    settings = protocol["marker_selection"]
    minimum_anchor = float(settings["minimum_anchor_distance_mm"])
    minimum_probe = float(settings["minimum_probe_radius_mm"])
    best: tuple[float, str, str, str] | None = None
    details: dict[str, float] | None = None
    for first in range(len(marker_labels)):
        for second in range(first + 1, len(marker_labels)):
            a = samples[:, first]
            b = samples[:, second]
            delta = b - a
            distances = np.linalg.norm(delta, axis=1)
            valid_distance = np.isfinite(distances) & (distances > 1e-9)
            if not np.any(valid_distance):
                continue
            for probe in range(len(marker_labels)):
                if probe in {first, second}:
                    continue
                p = samples[:, probe]
                valid = valid_distance & np.all(np.isfinite(p), axis=1)
                if np.count_nonzero(valid) < max(8, samples.shape[0] // 4):
                    continue
                direction = delta[valid] / distances[valid, None]
                relative = p[valid] - a[valid]
                axial = np.einsum("ij,ij->i", relative, direction)
                radial = relative - axial[:, None] * direction
                radii = np.linalg.norm(radial, axis=1)
                anchor_q10 = float(np.quantile(distances[valid], 0.10))
                radius_q10 = float(np.quantile(radii, 0.10))
                radius_median = float(np.median(radii))
                if anchor_q10 < minimum_anchor or radius_q10 < minimum_probe:
                    continue
                score = min(anchor_q10, 2.0 * radius_q10)
                candidate = (
                    score,
                    marker_labels[first],
                    marker_labels[second],
                    marker_labels[probe],
                )
                if best is None or candidate > best:
                    best = candidate
                    details = {
                        "score": score,
                        "anchor_distance_q10_mm": anchor_q10,
                        "probe_radius_q10_mm": radius_q10,
                        "probe_radius_median_mm": radius_median,
                    }
    if best is None or details is None:
        raise ValueError("calibration geometry contains no support-qualified marker triplet")
    return (best[1], best[2], best[3]), details
