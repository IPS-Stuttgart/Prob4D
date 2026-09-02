"""Sealed target stage for the prospective Tracking Cloth approximate-orbit study."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prob4d._tracking_cloth_approximate_orbit_data import _discover, _split
from prob4d._tracking_cloth_approximate_orbit_evaluation import (
    _bootstrap_difference,
    _method_metrics,
    _recording_pairs,
)
from prob4d._tracking_cloth_approximate_orbit_io import (
    RESULT_SCHEMA,
    _load_calibration,
    _sha256,
    _write_json,
)

FloatArray = NDArray[np.float64]


def _evaluate(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    calibration = _load_calibration(Path(args.calibration).resolve(), protocol)
    if calibration.get("status") != "calibrated":
        raise ValueError("target evaluation requires a successful calibration")

    cohort, common_labels, _ = _discover(dataset_root, protocol)
    calibration_groups, target_groups = _split(cohort, protocol)
    split = calibration["split"]
    if [recording.group_id for recording in calibration_groups] != split[
        "calibration_group_ids"
    ]:
        raise ValueError("calibration split changed")
    if [recording.group_id for recording in target_groups] != split["target_group_ids"]:
        raise ValueError("target split changed")
    if common_labels != calibration["dataset"]["common_marker_labels"]:
        raise ValueError("common marker namespace changed")
    triplet = tuple(calibration["marker_selection"]["triplet"])
    if len(triplet) != 3:
        raise ValueError("calibration marker triplet changed")

    orbit_radius = float(calibration["orbit_tube"]["radius"])
    point_radius = float(calibration["point_ball"]["radius"])
    minimum_pairs = int(protocol["prediction"]["minimum_valid_pairs"])
    groups: list[dict[str, object]] = []
    for recording in target_groups:
        pairs, metadata = _recording_pairs(recording, triplet, protocol)
        if len(pairs) < minimum_pairs:
            value = {
                "schema": RESULT_SCHEMA,
                "status": "target-support-negative",
                "protocol_id": protocol["protocol_id"],
                "calibration_id": calibration["calibration_id"],
                "source_revision": args.source_revision,
                "failed_group_id": recording.group_id,
                "valid_pair_count": len(pairs),
                "minimum_valid_pairs": minimum_pairs,
                "target_groups_replaced": False,
                "information_order": {
                    "calibration_artifact_verified_before_target_access": True,
                    "target_trajectory_values_parsed_during_calibration": False,
                    "target_side_retuning": False,
                    "target_groups_replaced": False,
                },
                "claim_boundary": protocol["claim_boundary"],
            }
            value["result_id"] = _sha256(value)
            _write_json(output / "result.json", value)
            return 2
        threshold_fraction = float(
            calibration["decision_threshold"]["normalized_axial_fraction"]
        )
        query_true = np.asarray(
            [
                pair.query_true_mm
                + (0.5 - threshold_fraction) * pair.anchor_length_mm
                for pair in pairs
            ],
            dtype=np.float64,
        )
        query_center = np.asarray(
            [
                pair.query_center_mm
                + (0.5 - threshold_fraction) * pair.anchor_length_mm
                for pair in pairs
            ],
            dtype=np.float64,
        )
        exact = _method_metrics(query_true, query_center, 0.0)
        orbit = _method_metrics(query_true, query_center, orbit_radius)
        point = _method_metrics(query_true, query_center, point_radius)
        groups.append(
            {
                **metadata,
                "query_center_rmse_mm": float(
                    math.sqrt(float(np.mean((query_true - query_center) ** 2)))
                ),
                "maximum_orbit_score_mm": max(pair.orbit_score_mm for pair in pairs),
                "maximum_point_score_mm": max(pair.point_score_mm for pair in pairs),
                "exact_estimated_orbit": exact,
                "conformal_orbit_tube": orbit,
                "generic_conformal_point_ball": point,
            }
        )

    def array(method: str, metric: str) -> FloatArray:
        return np.asarray([float(group[method][metric]) for group in groups], dtype=np.float64)

    methods = (
        "exact_estimated_orbit",
        "conformal_orbit_tube",
        "generic_conformal_point_ball",
    )
    aggregate: dict[str, object] = {}
    for method in methods:
        aggregate[method] = {
            "equal_recording_marginal_coverage": float(
                np.mean(array(method, "marginal_coverage"))
            ),
            "simultaneously_covered_recordings": int(
                np.sum(array(method, "simultaneous_coverage"))
            ),
            "simultaneous_recording_coverage_fraction": float(
                np.mean(array(method, "simultaneous_coverage"))
            ),
            "equal_recording_admission_fraction": float(
                np.mean(array(method, "admission_fraction"))
            ),
            "equal_recording_harmful_accepted_fraction_all_cases": float(
                np.mean(array(method, "harmful_accepted_fraction_all_cases"))
            ),
            "total_accepted_count": int(
                sum(int(group[method]["accepted_count"]) for group in groups)
            ),
            "total_harmful_accepted_count": int(
                sum(int(group[method]["harmful_accepted_count"]) for group in groups)
            ),
            "interval_width_mm": float(groups[0][method]["interval_width_mm"]),
        }

    width_ratio = orbit_radius / point_radius
    orbit_admission = array("conformal_orbit_tube", "admission_fraction")
    point_admission = array("generic_conformal_point_ball", "admission_fraction")
    exact_coverage = array("exact_estimated_orbit", "simultaneous_coverage")
    orbit_coverage = array("conformal_orbit_tube", "simultaneous_coverage")
    criteria = {
        "all_target_groups_supported": len(groups)
        == int(protocol["split"]["expected_target_groups"]),
        "orbit_simultaneous_coverage_at_least_registered_minimum": int(
            np.sum(orbit_coverage)
        )
        >= int(protocol["criteria"]["minimum_orbit_simultaneous_covered_groups"]),
        "orbit_tube_sharper_than_registered_ratio": width_ratio
        <= float(protocol["criteria"]["maximum_orbit_to_ball_width_ratio"]),
        "orbit_tube_admission_gain_at_least_registered_minimum": float(
            np.mean(orbit_admission - point_admission)
        )
        >= float(protocol["criteria"]["minimum_orbit_tube_admission_gain"]),
        "orbit_tube_no_more_harmful_than_point_ball": float(
            aggregate["conformal_orbit_tube"][
                "equal_recording_harmful_accepted_fraction_all_cases"
            ]
        )
        <= float(
            aggregate["generic_conformal_point_ball"][
                "equal_recording_harmful_accepted_fraction_all_cases"
            ]
        )
        + 1e-15,
        "approximate_tube_improves_over_exact_orbit_coverage": float(
            np.mean(orbit_coverage)
        )
        > float(np.mean(exact_coverage)),
    }
    status = (
        "approximate-orbit-tube-positive"
        if all(criteria.values())
        else "approximate-orbit-tube-mixed"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "calibration_id": calibration["calibration_id"],
        "source_revision": args.source_revision,
        "cohort": {
            "target_group_count": len(groups),
            "target_group_ids": [recording.group_id for recording in target_groups],
            "target_relative_paths": [recording.relative_path for recording in target_groups],
            "materials": sorted(set(recording.material for recording in target_groups)),
            "independent_unit": "complete-recording",
        },
        "marker_triplet": list(triplet),
        "calibration": {
            "requested_miscoverage": calibration["orbit_tube"]["requested_miscoverage"],
            "finite_sample_coverage_level": calibration["orbit_tube"][
                "finite_sample_coverage_level"
            ],
            "order_statistic_rank": calibration["orbit_tube"]["order_statistic_rank"],
            "calibration_group_count": calibration["orbit_tube"]["group_count"],
            "orbit_tube_radius_mm": orbit_radius,
            "generic_point_ball_radius_mm": point_radius,
            "orbit_to_point_radius_ratio": width_ratio,
        },
        "groups": groups,
        "aggregate": aggregate,
        "paired_recording_bootstrap": {
            "orbit_minus_point_admission_fraction": _bootstrap_difference(
                orbit_admission,
                point_admission,
                seed=int(protocol["analysis"]["bootstrap_seed"]),
                replicates=int(protocol["analysis"]["bootstrap_replicates"]),
            ),
            "orbit_minus_point_harmful_fraction": _bootstrap_difference(
                array(
                    "conformal_orbit_tube",
                    "harmful_accepted_fraction_all_cases",
                ),
                array(
                    "generic_conformal_point_ball",
                    "harmful_accepted_fraction_all_cases",
                ),
                seed=int(protocol["analysis"]["bootstrap_seed"]) + 1,
                replicates=int(protocol["analysis"]["bootstrap_replicates"]),
            ),
        },
        "criteria": criteria,
        "information_order": {
            "calibration_artifact_verified_before_target_access": True,
            "target_trajectory_values_parsed_during_calibration": False,
            "target_side_retuning": False,
            "target_groups_replaced": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _sha256(result)
    _write_json(output / "result.json", result)

    orbit_summary = aggregate["conformal_orbit_tube"]
    ball_summary = aggregate["generic_conformal_point_ball"]
    exact_summary = aggregate["exact_estimated_orbit"]
    (output / "summary.md").write_text(
        "\n".join(
            [
                "# Tracking Cloth approximate-orbit tube result",
                "",
                f"Status: **{status}**",
                "",
                f"Protocol ID: `{protocol['protocol_id']}`",
                f"Calibration ID: `{calibration['calibration_id']}`",
                f"Result ID: `{result['result_id']}`",
                "",
                f"- target recordings: {len(groups)}",
                f"- orbit-tube radius: {orbit_radius:.6f} mm",
                f"- generic point-ball radius: {point_radius:.6f} mm",
                f"- width ratio: {width_ratio:.6f}",
                (
                    "- simultaneous coverage (exact/orbit tube/point ball): "
                    f"{exact_summary['simultaneously_covered_recordings']}/"
                    f"{orbit_summary['simultaneously_covered_recordings']}/"
                    f"{ball_summary['simultaneously_covered_recordings']} of {len(groups)}"
                ),
                (
                    "- equal-recording decision admission (orbit tube / point ball): "
                    f"{orbit_summary['equal_recording_admission_fraction']:.6f} / "
                    f"{ball_summary['equal_recording_admission_fraction']:.6f}"
                ),
                (
                    "- harmful accepted fraction (orbit tube / point ball): "
                    f"{orbit_summary['equal_recording_harmful_accepted_fraction_all_cases']:.6f} / "
                    f"{ball_summary['equal_recording_harmful_accepted_fraction_all_cases']:.6f}"
                ),
                "",
                "This is a trajectory-level calibration test on an estimated orbit. "
                "It does not establish learned visual-provider competence or robot safety.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0 if status == "approximate-orbit-tube-positive" else 3
