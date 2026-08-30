"""Calibrate a query-observability gate on real DEFORM training geometry.

The source stage uses only DLO4/DLO5 training trajectories and only sliding
four-vertex segments whose source-frozen effective Sim(3) rank is six.  It
selects the smallest direct-observability threshold that retains at least 99%
of segment-centroid queries while admitting at most 10% of off-axis probes.
Official evaluation trajectories remain unopened.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import audit_deform_dlo45_observability_v1 as base
import numpy as np

from prob4d.observable_gauge import estimate_observable_sim3_factor
from prob4d.query_observability import (
    evaluate_query_observability,
    point_position_query_jacobian,
)

SCHEMA = "prob4d.deform-dlo45-query-gate-source-calibration"
SCHEMA_VERSION = 1
REQUEST_SCHEMA = "prob4d.deform-dlo45-query-gate-source-request"


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
    if request.get("stage") != "source-query-gate-calibration":
        raise ValueError("unexpected stage")
    if Path(str(request.get("dataset_root"))) != base.EXPECTED_ROOT:
        raise ValueError(f"dataset_root must be exactly {base.EXPECTED_ROOT}")
    if request.get("dlo_types") != ["DLO4", "DLO5"]:
        raise ValueError("dlo_types must be exactly DLO4 and DLO5")
    if request.get("segment_length") != 4:
        raise ValueError("segment_length must be exactly four")
    rank_threshold = float(request.get("rank_threshold"))
    if rank_threshold != 0.01:
        raise ValueError("rank_threshold must be the source-frozen value 0.01")
    candidates = [float(value) for value in request["direct_threshold_candidates"]]
    if candidates != sorted(candidates) or any(not 0.0 <= value <= 1.0 for value in candidates):
        raise ValueError("direct threshold candidates must be sorted in [0,1]")
    boundary = request.get("information_boundary")
    expected_boundary = {
        "opened_split": "train",
        "evaluation_file_contents_opened": False,
        "provider_predictions_opened": False,
        "bayesian_phystwin_outcomes_opened": False,
        "causal4d_outcomes_opened": False,
    }
    if boundary != expected_boundary:
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


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    root = base.EXPECTED_ROOT
    rank_threshold = float(request["rank_threshold"])
    frame_stride = int(request["frame_stride"])
    segment_length = int(request["segment_length"])
    noise_sigma = float(request["correspondence_noise_sigma_m"])
    probe_radius_factor = float(request["probe_radius_cloud_scale_factor"])
    seed = int(request["seed"])
    if frame_stride < 1 or noise_sigma <= 0.0 or probe_radius_factor <= 0.0:
        raise ValueError("invalid source calibration configuration")
    generator = np.random.default_rng(seed)
    prior_std = np.asarray(request["prior_standard_deviations_local"], dtype=np.float64)
    if prior_std.shape != (7,) or np.any(prior_std <= 0.0):
        raise ValueError("prior_standard_deviations_local must contain seven positives")
    prior_covariance = np.diag(prior_std**2)

    centroid_direct: list[float] = []
    probe_direct: list[float] = []
    centroid_reduction: list[float] = []
    probe_reduction: list[float] = []
    factor_rank_counts: Counter[int] = Counter()
    preselected = 0
    fit_failures: Counter[str] = Counter()
    cases_by_object: Counter[str] = Counter()
    file_manifest: list[dict[str, Any]] = []

    for dlo_type in request["dlo_types"]:
        directory = root / dlo_type / "train"
        files = sorted(directory.glob("*.pkl"), key=lambda item: int(item.stem))
        if len(files) != 56:
            raise ValueError(f"expected 56 training files for {dlo_type}")
        for path in files:
            file_manifest.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": base.sha256_file(path),
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
                    preselected += 1
                    target = segment + generator.normal(
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
                    centroid = np.mean(segment, axis=0)
                    normal = deterministic_normal(segment)
                    probe = centroid + probe_radius_factor * factor.chart.cloud_scale * normal
                    centroid_report = evaluate_query_observability(
                        factor,
                        prior_covariance_local=prior_covariance,
                        query_jacobian_local=point_position_query_jacobian(
                            factor,
                            centroid,
                        ),
                    )
                    probe_report = evaluate_query_observability(
                        factor,
                        prior_covariance_local=prior_covariance,
                        query_jacobian_local=point_position_query_jacobian(
                            factor,
                            probe,
                        ),
                    )
                    centroid_direct.append(centroid_report.direct_observability_fraction)
                    probe_direct.append(probe_report.direct_observability_fraction)
                    centroid_reduction.append(centroid_report.metric_variance_reduction_fraction)
                    probe_reduction.append(probe_report.metric_variance_reduction_fraction)
                    cases_by_object[dlo_type] += 1

    if len(centroid_direct) < int(request["minimum_rank_six_cases"]):
        raise ValueError("insufficient successfully fitted rank-six source cases")

    candidates: list[dict[str, Any]] = []
    selected: float | None = None
    for threshold in request["direct_threshold_candidates"]:
        threshold = float(threshold)
        centroid_acceptance = float(np.mean(np.asarray(centroid_direct) >= threshold))
        probe_acceptance = float(np.mean(np.asarray(probe_direct) >= threshold))
        qualifies = centroid_acceptance >= float(
            request["minimum_centroid_acceptance_fraction"]
        ) and probe_acceptance <= float(request["maximum_probe_acceptance_fraction"])
        candidates.append(
            {
                "minimum_direct_observability_fraction": threshold,
                "centroid_acceptance_fraction": centroid_acceptance,
                "off_axis_probe_acceptance_fraction": probe_acceptance,
                "qualifies": qualifies,
            }
        )
        if qualifies and selected is None:
            selected = threshold
    if selected is None:
        raise ValueError("no source gate candidate satisfies the frozen selection rule")

    manifest: dict[str, Any] = {
        "files": sorted(file_manifest, key=lambda row: row["path"]),
        "file_count": len(file_manifest),
        "total_bytes": int(sum(row["bytes"] for row in file_manifest)),
    }
    manifest["manifest_sha256"] = base.canonical_sha256(manifest)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": request["stage"],
        "request_id": request["request_id"],
        "source_geometry_result_id": request["source_geometry_result_id"],
        "source_geometry_manifest_sha256": request["source_geometry_manifest_sha256"],
        "dataset": {
            "name": "DEFORM",
            "objects": request["dlo_types"],
            "opened_split": "train",
            "manifest": manifest,
        },
        "design": {
            "rank_threshold": rank_threshold,
            "segment_length": segment_length,
            "frame_stride": frame_stride,
            "noise_sigma_m": noise_sigma,
            "probe_radius_cloud_scale_factor": probe_radius_factor,
            "prior_standard_deviations_local": prior_std.tolist(),
            "seed": seed,
        },
        "accounting": {
            "geometry_rank_six_preselected": preselected,
            "successful_rank_six_cases": len(centroid_direct),
            "factor_rank_counts": {
                str(rank): count for rank, count in sorted(factor_rank_counts.items())
            },
            "fit_failures": dict(sorted(fit_failures.items())),
            "cases_by_object": dict(sorted(cases_by_object.items())),
        },
        "source_distributions": {
            "centroid_direct_observability_fraction": quantiles(centroid_direct),
            "off_axis_probe_direct_observability_fraction": quantiles(probe_direct),
            "centroid_metric_variance_reduction_fraction": quantiles(centroid_reduction),
            "off_axis_probe_metric_variance_reduction_fraction": quantiles(probe_reduction),
        },
        "candidate_gates": candidates,
        "selection_rule": {
            "minimum_centroid_acceptance_fraction": request["minimum_centroid_acceptance_fraction"],
            "maximum_off_axis_probe_acceptance_fraction": request[
                "maximum_probe_acceptance_fraction"
            ],
            "tie_break": "smallest qualifying direct-observability threshold",
        },
        "selected_gate": {
            "minimum_direct_observability_fraction": selected,
            "minimum_metric_variance_reduction_fraction": 0.0,
            "maximum_worst_supported_variance_ratio": 1.0,
        },
        "information_boundary": request["information_boundary"],
        "claim_boundary": [
            "The gate was selected from official DLO4/DLO5 training trajectories only.",
            "Known synthetic correspondence noise is applied to held-out real source geometries.",
            "No official evaluation trajectory or downstream BayesianPhysTwin/Causal4D outcome was opened.",  # noqa: E501
            "The selected gate is frozen before any evaluation access.",
        ],
    }
    result["result_id"] = base.canonical_sha256(result)
    return result


def write_summary(result: dict[str, Any], path: Path) -> None:
    selected = result["selected_gate"]["minimum_direct_observability_fraction"]
    lines = [
        "# Source-only DEFORM query-gate calibration",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Rank-six cases: `{result['accounting']['successful_rank_six_cases']}`",
        f"- Selected minimum direct observability: `{selected}`",
        "",
        "## Candidate gates",
        "",
    ]
    for row in result["candidate_gates"]:
        lines.append(
            "- threshold={minimum_direct_observability_fraction}: centroid={centroid_acceptance_fraction:.3f}, "  # noqa: E501
            "off-axis={off_axis_probe_acceptance_fraction:.3f}, qualifies={qualifies}".format(**row)
        )
    lines.extend(["", "## Source distributions", ""])
    for name, values in result["source_distributions"].items():
        lines.append(
            f"- **{name}:** median={values['median']:.6f}, q05={values['q05']:.6f}, "
            f"q95={values['q95']:.6f}"
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
    print(json.dumps({"result_id": result["result_id"]}))


if __name__ == "__main__":
    main()
