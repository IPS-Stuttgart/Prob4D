"""Geometry scores and decision metrics for approximate axial-orbit evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prob4d._tracking_cloth_approximate_orbit_data import Recording, _pair_indices
from prob4d.motive_csv import read_motive_markers
from prob4d.orbit_tube import AxialCircleOrbit, minimal_rotation_transport

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PairEvidence:
    orbit_score_mm: float
    point_score_mm: float
    query_true_mm: float
    query_center_mm: float
    anchor_length_mm: float


def _pair_evidence(previous: FloatArray, current: FloatArray) -> PairEvidence | None:
    a0, b0, p0 = previous
    a1, b1, p1 = current
    line0 = b0 - a0
    line1 = b1 - a1
    length0 = float(np.linalg.norm(line0))
    length1 = float(np.linalg.norm(line1))
    if not all(math.isfinite(value) and value > 1e-9 for value in (length0, length1)):
        return None
    axis0 = line0 / length0
    axis1 = line1 / length1
    relative0 = p0 - a0
    axial0 = float(relative0 @ axis0)
    radial0 = relative0 - axial0 * axis0
    radius0 = float(np.linalg.norm(radial0))
    if not math.isfinite(radius0) or radius0 <= 1e-9:
        return None
    axial_fraction = axial0 / length0
    radius_fraction = radius0 / length0
    center = a1 + axial_fraction * length1 * axis1
    radius = radius_fraction * length1
    orbit = AxialCircleOrbit(center=center, axis=axis1, radius=radius)

    radial_direction = minimal_rotation_transport(radial0 / radius0, axis0, axis1)
    radial_direction = radial_direction - float(radial_direction @ axis1) * axis1
    radial_norm = float(np.linalg.norm(radial_direction))
    if not math.isfinite(radial_norm) or radial_norm <= 1e-9:
        return None
    canonical = center + radius * radial_direction / radial_norm
    orbit_score = orbit.point_distance(p1)
    point_score = float(np.linalg.norm(p1 - canonical))
    query_true = float((p1 - a1) @ axis1 - 0.5 * length1)
    query_center = float((center - a1) @ axis1 - 0.5 * length1)
    values = (orbit_score, point_score, query_true, query_center, length1)
    if not all(math.isfinite(value) for value in values):
        return None
    return PairEvidence(*values)


def _recording_pairs(
    recording: Recording,
    marker_triplet: tuple[str, str, str],
    protocol: dict[str, Any],
) -> tuple[list[PairEvidence], dict[str, object]]:
    coordinates, scale, details = read_motive_markers(recording.path, marker_triplet)
    maximum = int(protocol["prediction"]["pairs_per_recording"])
    pairs: list[PairEvidence] = []
    for index in _pair_indices(coordinates.shape[0], maximum):
        previous = coordinates[int(index)]
        current = coordinates[int(index) + 1]
        if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
            continue
        evidence = _pair_evidence(previous, current)
        if evidence is not None:
            pairs.append(evidence)
    metadata = {
        "group_id": recording.group_id,
        "relative_path": recording.relative_path,
        "material": recording.material,
        "valid_pair_count": len(pairs),
        "unit_scale_to_mm": scale,
        **details,
    }
    return pairs, metadata


def _method_metrics(
    query_true: FloatArray,
    query_center: FloatArray,
    radius: float,
) -> dict[str, object]:
    lower = query_center - radius
    upper = query_center + radius
    covered = (query_true >= lower) & (query_true <= upper)
    positive = lower > 0.0
    negative = upper < 0.0
    admitted = positive | negative
    predicted = np.where(positive, 1, np.where(negative, -1, 0))
    truth = np.sign(query_true).astype(np.int64)
    harmful = admitted & (predicted != truth)
    return {
        "radius_mm": float(radius),
        "interval_width_mm": float(2.0 * radius),
        "marginal_coverage": float(np.mean(covered)),
        "simultaneous_coverage": bool(np.all(covered)),
        "admission_fraction": float(np.mean(admitted)),
        "harmful_accepted_fraction_all_cases": float(np.mean(harmful)),
        "harmful_fraction_among_accepted": (
            float(np.count_nonzero(harmful) / np.count_nonzero(admitted))
            if np.any(admitted)
            else 0.0
        ),
        "accepted_count": int(np.count_nonzero(admitted)),
        "harmful_accepted_count": int(np.count_nonzero(harmful)),
    }


def _bootstrap_difference(
    values_a: FloatArray,
    values_b: FloatArray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("bootstrap inputs must be equal one-dimensional arrays")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values_a.size, size=(replicates, values_a.size))
    differences = np.mean(values_a[indices] - values_b[indices], axis=1)
    return {
        "mean": float(np.mean(values_a - values_b)),
        "lower": float(np.quantile(differences, 0.025)),
        "upper": float(np.quantile(differences, 0.975)),
        "replicates": replicates,
        "seed": seed,
    }
