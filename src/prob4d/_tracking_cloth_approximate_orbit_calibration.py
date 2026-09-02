"""Calibration stage for the prospective Tracking Cloth approximate-orbit study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from prob4d._tracking_cloth_approximate_orbit_data import (
    _collect_selection_samples,
    _discover,
    _select_triplet,
    _split,
)
from prob4d._tracking_cloth_approximate_orbit_evaluation import PairEvidence, _recording_pairs
from prob4d._tracking_cloth_approximate_orbit_io import (
    CALIBRATION_SCHEMA,
    _sha256,
    _write_json,
)
from prob4d.orbit_tube import calibrate_group_maximum_radius


def _calibrate(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    cohort, common_labels, header_metadata = _discover(dataset_root, protocol)
    calibration_groups, target_groups = _split(cohort, protocol)

    selection_samples, selection_metadata = _collect_selection_samples(
        calibration_groups,
        common_labels,
        int(protocol["marker_selection"]["selection_frames_per_recording"]),
    )
    triplet, selection = _select_triplet(selection_samples, common_labels, protocol)

    orbit_scores: list[float] = []
    point_scores: list[float] = []
    group_ids: list[str] = []
    calibration_pairs: list[PairEvidence] = []
    pair_metadata: list[dict[str, object]] = []
    minimum_pairs = int(protocol["prediction"]["minimum_valid_pairs"])
    for recording in calibration_groups:
        pairs, metadata = _recording_pairs(recording, triplet, protocol)
        pair_metadata.append(metadata)
        if len(pairs) < minimum_pairs:
            result = {
                "schema": CALIBRATION_SCHEMA,
                "status": "calibration-support-negative",
                "protocol_id": protocol["protocol_id"],
                "source_revision": args.source_revision,
                "failed_group_id": recording.group_id,
                "valid_pair_count": len(pairs),
                "minimum_valid_pairs": minimum_pairs,
                "target_trajectory_values_parsed": False,
            }
            result["calibration_id"] = _sha256(result)
            _write_json(output / "calibration.json", result)
            return 2
        calibration_pairs.extend(pairs)
        for pair in pairs:
            orbit_scores.append(pair.orbit_score_mm)
            point_scores.append(pair.point_score_mm)
            group_ids.append(recording.group_id)

    miscoverage = float(protocol["calibration"]["requested_miscoverage"])
    orbit_calibration = calibrate_group_maximum_radius(
        np.asarray(orbit_scores),
        group_ids,
        miscoverage=miscoverage,
    )
    point_calibration = calibrate_group_maximum_radius(
        np.asarray(point_scores),
        group_ids,
        miscoverage=miscoverage,
    )
    if point_calibration.radius <= 0.0:
        raise ValueError("generic point calibration radius must be positive")
    normalized_query_centers = np.asarray(
        [
            pair.query_center_mm / pair.anchor_length_mm + 0.5
            for pair in calibration_pairs
        ],
        dtype=np.float64,
    )
    if normalized_query_centers.size == 0 or not np.all(
        np.isfinite(normalized_query_centers)
    ):
        raise ValueError("calibration decision threshold could not be estimated")
    decision_threshold_fraction = float(np.median(normalized_query_centers))
    value: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "status": "calibrated",
        "protocol_id": protocol["protocol_id"],
        "source_revision": args.source_revision,
        "dataset": {
            "cohort_group_ids": sorted(recording.group_id for recording in cohort),
            "cohort_relative_paths": sorted(recording.relative_path for recording in cohort),
            "common_marker_labels": common_labels,
            "header_metadata": header_metadata,
        },
        "split": {
            "calibration_group_ids": [recording.group_id for recording in calibration_groups],
            "calibration_relative_paths": [
                recording.relative_path for recording in calibration_groups
            ],
            "target_group_ids": [recording.group_id for recording in target_groups],
            "target_relative_paths": [recording.relative_path for recording in target_groups],
        },
        "marker_selection": {
            "triplet": list(triplet),
            "statistics": selection,
            "selection_metadata": selection_metadata,
        },
        "pair_metadata": pair_metadata,
        "orbit_tube": orbit_calibration.to_dict(),
        "point_ball": point_calibration.to_dict(),
        "decision_threshold": {
            "normalized_axial_fraction": decision_threshold_fraction,
            "statistic": "median calibration query-center fraction",
            "target_values_used": False,
        },
        "information_order": {
            "all_cohort_headers_parsed": True,
            "calibration_trajectory_values_parsed": True,
            "target_trajectory_values_parsed": False,
            "calibration_sealed_before_target_access": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    value["calibration_id"] = _sha256(value)
    _write_json(output / "calibration.json", value)
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# Tracking Cloth approximate-orbit calibration",
                "",
                "Status: **calibrated**",
                "",
                f"Protocol ID: `{protocol['protocol_id']}`",
                f"Calibration ID: `{value['calibration_id']}`",
                f"Selected marker triplet: `{triplet[0]}/{triplet[1]}/{triplet[2]}`",
                f"Orbit-tube radius: {orbit_calibration.radius:.6f} mm",
                f"Point-ball radius: {point_calibration.radius:.6f} mm",
                f"Radius ratio: {orbit_calibration.radius / point_calibration.radius:.6f}",
                f"Decision threshold fraction: {decision_threshold_fraction:.6f}",
                "",
                "No target trajectory value was parsed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0
