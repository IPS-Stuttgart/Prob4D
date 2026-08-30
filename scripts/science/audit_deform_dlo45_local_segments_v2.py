"""Audit effective Sim(3) observability of real sliding DLO segments.

The study opens only official DLO4/DLO5 training trajectories.  Consecutive
four-, five-, and six-vertex segments represent outcome-blind partial visibility
of a deformable linear object.  No evaluation trajectory or learned-provider
prediction is opened.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import audit_deform_dlo45_observability_v1 as base
import numpy as np

SCHEMA = "prob4d.deform-dlo45-local-segment-source-audit"
SCHEMA_VERSION = 2
REQUEST_SCHEMA = "prob4d.deform-dlo45-local-segment-source-request"


def load_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != 2:
        raise ValueError("unsupported request schema")
    supplied = request.get("request_id")
    unhashed = dict(request)
    unhashed.pop("request_id", None)
    if supplied != base.canonical_sha256(unhashed):
        raise ValueError("request_id does not match canonical request contents")
    if request.get("stage") != "source-local-segment-audit":
        raise ValueError("unexpected stage")
    if Path(str(request.get("dataset_root"))) != base.EXPECTED_ROOT:
        raise ValueError(f"dataset_root must be exactly {base.EXPECTED_ROOT}")
    if request.get("dlo_types") != ["DLO4", "DLO5"]:
        raise ValueError("dlo_types must be exactly DLO4 and DLO5")
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
    lengths = request.get("segment_lengths")
    if lengths != [4, 5, 6]:
        raise ValueError("segment_lengths must be exactly [4, 5, 6]")
    return request


def _aggregate_spectra(
    spectra: dict[str, list[np.ndarray]],
    radii: dict[str, list[float]],
    line_ratios: dict[str, list[float]],
    thresholds: list[float],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for key, values in sorted(spectra.items()):
        matrix = np.asarray(values, dtype=np.float64)
        ranks_by_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            ranks = np.sum(matrix >= threshold, axis=1)
            ranks_by_threshold[f"{threshold:.12g}"] = {
                "rank_deficient_fraction": float(np.mean(ranks < 7)),
                "rank_six_fraction": float(np.mean(ranks == 6)),
                "rank_at_most_five_fraction": float(np.mean(ranks <= 5)),
                "full_rank_fraction": float(np.mean(ranks == 7)),
                "rank_counts": {
                    str(rank): int(np.count_nonzero(ranks == rank))
                    for rank in sorted(set(int(value) for value in ranks))
                },
            }
        groups[key] = {
            "cases": int(matrix.shape[0]),
            "lambda7_over_lambda1": base.quantiles(matrix[:, -1].tolist()),
            "lambda6_over_lambda1": base.quantiles(matrix[:, -2].tolist()),
            "cloud_radius_m": base.quantiles(radii[key]),
            "smallest_to_largest_centered_singular_value": base.quantiles(line_ratios[key]),
            "rank_by_threshold": ranks_by_threshold,
        }
    return groups


def run(request: dict[str, Any]) -> dict[str, Any]:
    root = base.EXPECTED_ROOT
    frame_stride = int(request["frame_stride"])
    segment_stride = int(request["segment_stride"])
    if frame_stride < 1 or segment_stride < 1:
        raise ValueError("strides must be positive")
    thresholds = [float(value) for value in request["rank_threshold_candidates"]]
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("rank thresholds must lie in (0,1)")
    if thresholds != sorted(thresholds):
        raise ValueError("rank thresholds must be sorted")

    manifest: list[dict[str, Any]] = []
    spectra: dict[str, list[np.ndarray]] = defaultdict(list)
    radii: dict[str, list[float]] = defaultdict(list)
    line_ratios: dict[str, list[float]] = defaultdict(list)
    case_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    shapes: dict[str, list[list[int]]] = defaultdict(list)

    for dlo_type in request["dlo_types"]:
        directory = root / dlo_type / "train"
        files = sorted(directory.glob("*.pkl"), key=lambda item: int(item.stem))
        if len(files) != 56:
            raise ValueError(
                f"expected 56 official training files for {dlo_type}, found {len(files)}"
            )
        for path in files:
            manifest.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": base.sha256_file(path),
                }
            )
            frames = base.load_trajectory(path)
            shapes[dlo_type].append([int(value) for value in frames.shape])
            for frame_index in range(0, frames.shape[0], frame_stride):
                frame = frames[frame_index]
                for length in request["segment_lengths"]:
                    name = f"local{length}-sliding-span1"
                    key = f"{dlo_type}/{name}"
                    for start in range(0, frame.shape[0] - length + 1, segment_stride):
                        support = frame[start : start + length]
                        normalized, radius, line_ratio = base.geometry_spectrum(support)
                        spectra[key].append(normalized)
                        radii[key].append(radius)
                        line_ratios[key].append(line_ratio)
                        case_counts[dlo_type][name] += 1

    manifest_record: dict[str, Any] = {
        "files": sorted(manifest, key=lambda row: row["path"]),
        "file_count": len(manifest),
        "total_bytes": int(sum(row["bytes"] for row in manifest)),
    }
    manifest_record["manifest_sha256"] = base.canonical_sha256(manifest_record)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": request["stage"],
        "request_id": request["request_id"],
        "evidence_class": "public-real-trajectory-source-only-local-geometry-audit",
        "dataset": {
            "name": "DEFORM",
            "objects": request["dlo_types"],
            "opened_split": "train",
            "manifest": manifest_record,
            "trajectory_shapes": shapes,
        },
        "design": {
            "frame_stride": frame_stride,
            "segment_stride": segment_stride,
            "segment_lengths": request["segment_lengths"],
            "rank_threshold_candidates": thresholds,
            "cases_by_object_and_support": case_counts,
        },
        "groups": _aggregate_spectra(spectra, radii, line_ratios, thresholds),
        "information_boundary": request["information_boundary"],
        "claim_boundary": [
            "Only official DLO4/DLO5 training trajectories were opened.",
            "Sliding consecutive vertices emulate an outcome-blind local visibility mask.",
            "The result characterizes geometry only; it is not learned-provider competence.",
            "No official evaluation trajectory, BayesianPhysTwin outcome, or Causal4D outcome was opened.",  # noqa: E501
        ],
    }
    result["result_id"] = base.canonical_sha256(result)
    return result


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# DEFORM DLO4/DLO5 sliding-segment source audit",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Files opened: `{result['dataset']['manifest']['file_count']}` training files",
        f"- Manifest SHA-256: `{result['dataset']['manifest']['manifest_sha256']}`",
        "",
        "## Local geometry",
        "",
    ]
    for name, group in result["groups"].items():
        lines.append(
            f"- **{name}:** cases={group['cases']}, median lambda7/lambda1="
            f"{group['lambda7_over_lambda1']['median']:.6g}, q05="
            f"{group['lambda7_over_lambda1']['q05']:.6g}"
        )
        for threshold, row in group["rank_by_threshold"].items():
            lines.append(
                f"  - `{threshold}`: rank<7={row['rank_deficient_fraction']:.3f}, "
                f"rank6={row['rank_six_fraction']:.3f}, rank<=5="
                f"{row['rank_at_most_five_fraction']:.3f}"
            )
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {entry}" for entry in result["claim_boundary"])
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
