#!/usr/bin/env python3
"""Develop an identity-free rod-pair rule on four labelled source recordings.

Only the exact Hitting paths frozen in the protocol are parsed. The true rod
labels 21/22 are used solely to score three geometry-only ranking rules. No
Self-collision trajectory is opened, so a later protocol can freeze one rule
before accessing the 36 unlabeled-marker recordings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.motive_csv import read_motive_layout, read_motive_markers

SCHEMA = "prob4d.tracking-cloth-augmented-rod-source-audit.v1"
RESULT_SCHEMA = "prob4d.tracking-cloth-augmented-rod-source-audit-result.v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported source-audit protocol")
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id", None)
    if supplied != _sha256(_canonical(unsigned)):
        raise ValueError("source-audit protocol identity changed")
    if value["dataset"]["self_collision_trajectory_access_allowed"] is not False:
        raise ValueError("Self-collision trajectory access was authorized")
    if value["information_order"]["target_side_tuning_allowed"] is not False:
        raise ValueError("target-side tuning was authorized")
    return value


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    return np.unique(
        np.linspace(0, length - 1, num=min(length, maximum), dtype=np.int64)
    )


def _pair_rows(
    coordinates: np.ndarray,
    labels: tuple[str, ...],
    *,
    minimum_valid_fraction: float,
    minimum_distance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_count = coordinates.shape[0]
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            a = coordinates[:, first]
            b = coordinates[:, second]
            valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
            valid_count = int(np.sum(valid))
            valid_fraction = valid_count / frame_count
            if valid_fraction < minimum_valid_fraction:
                continue
            distance = np.linalg.norm(a[valid] - b[valid], axis=1)
            median = float(np.median(distance))
            if not math.isfinite(median) or median < minimum_distance:
                continue
            q05, q95 = np.quantile(distance, [0.05, 0.95])
            mad = float(np.median(np.abs(distance - median)))
            relative_90 = float((q95 - q05) / median)
            relative_mad = float(1.4826 * mad / median)
            combined = float(relative_90 + relative_mad + 4.0 * (1.0 - valid_fraction))
            rows.append(
                {
                    "pair": sorted((labels[first], labels[second])),
                    "valid_fraction": valid_fraction,
                    "median_distance_mm": median,
                    "q05_distance_mm": float(q05),
                    "q95_distance_mm": float(q95),
                    "relative_90_spread": relative_90,
                    "relative_mad": relative_mad,
                    "combined_stability": combined,
                }
            )
    if not rows:
        raise RuntimeError("no candidate marker pairs passed the source support rules")
    return rows


def _rank(
    rows: list[dict[str, Any]],
    metric: str,
    true_pair: list[str],
    retain: int,
) -> tuple[int, list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[metric]),
            -float(row["valid_fraction"]),
            -float(row["median_distance_mm"]),
            tuple(row["pair"]),
        ),
    )
    target = sorted(true_pair)
    rank = next(
        index for index, row in enumerate(ordered, start=1) if row["pair"] == target
    )
    return rank, ordered[:retain]


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# Tracking Cloth labelled rod-pair source audit",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Recommended source metric: `{result['recommendation']['metric']}`",
        "",
        "| Recording | Markers | Candidate pairs | rel90 rank | relMAD rank | combined rank | Rod length [mm] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["recordings"]:
        ranks = row["true_pair_ranks"]
        lines.append(
            "| {path} | {markers} | {pairs} | {r90} | {rmad} | {combined} | {length:.3f} |".format(
                path=row["relative_path"],
                markers=row["marker_count"],
                pairs=row["candidate_pair_count"],
                r90=ranks["relative_90_spread"],
                rmad=ranks["relative_mad"],
                combined=ranks["combined_stability"],
                length=row["true_pair_median_distance_mm"],
            )
        )
    lines.extend(
        [
            "",
            "This is source-only method development. No Self-collision trajectory was parsed, "
            "and the recommendation is not yet a frozen target method or a utility result.",
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
    protocol = _load_protocol(protocol_path)
    all_csv = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    if len(all_csv) != int(protocol["dataset"]["expected_csv_files"]):
        raise ValueError("official CSV roster changed")

    settings = protocol["analysis"]
    true_pair = list(protocol["dataset"]["true_rod_marker_labels"])
    metrics = list(settings["ranking_metrics"])
    recordings: list[dict[str, Any]] = []
    opened: list[str] = []
    for relative_path in protocol["dataset"]["source_relative_paths"]:
        path = dataset_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        layout = read_motive_layout(path)
        labels = tuple(layout.marker_labels)
        if not set(true_pair).issubset(labels):
            raise RuntimeError(f"true source rod labels are absent from {relative_path}")
        coordinates, scale, details = read_motive_markers(path, labels)
        opened.append(relative_path)
        coordinates = coordinates[
            _sample_indices(coordinates.shape[0], int(settings["maximum_frames_per_recording"]))
        ]
        pairs = _pair_rows(
            coordinates,
            labels,
            minimum_valid_fraction=float(settings["minimum_valid_fraction"]),
            minimum_distance=float(settings["minimum_median_pair_distance_mm"]),
        )
        ranks: dict[str, int] = {}
        top: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            rank, retained = _rank(
                pairs,
                metric,
                true_pair,
                int(settings["top_pairs_retained_per_metric"]),
            )
            ranks[metric] = rank
            top[metric] = retained
        true_row = next(row for row in pairs if row["pair"] == sorted(true_pair))
        recordings.append(
            {
                "relative_path": relative_path,
                "path_sha256": _sha256(relative_path.encode()),
                "marker_count": len(labels),
                "candidate_pair_count": len(pairs),
                "unit_scale_to_mm": scale,
                "sampled_frame_count": int(coordinates.shape[0]),
                "source_rows": details["rows"],
                "true_pair_ranks": ranks,
                "true_pair_median_distance_mm": true_row["median_distance_mm"],
                "true_pair_metrics": {
                    metric: true_row[metric] for metric in metrics
                },
                "top_pairs": top,
            }
        )

    aggregates: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = [int(row["true_pair_ranks"][metric]) for row in recordings]
        aggregates[metric] = {
            "median_true_pair_rank": float(np.median(values)),
            "worst_true_pair_rank": max(values),
            "top1_recordings": sum(value == 1 for value in values),
            "top5_recordings": sum(value <= 5 for value in values),
            "ranks": values,
        }
    recommended = min(
        metrics,
        key=lambda metric: (
            aggregates[metric]["worst_true_pair_rank"],
            aggregates[metric]["median_true_pair_rank"],
            -aggregates[metric]["top1_recordings"],
            metric,
        ),
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "source-audit-complete",
        "protocol_id": protocol["protocol_id"],
        "source_revision": source_revision,
        "opened_relative_paths": opened,
        "self_collision_trajectory_paths_opened": [],
        "recordings": recordings,
        "aggregate": aggregates,
        "recommendation": {
            "metric": recommended,
            "selection_rule": "minimum worst rank, then median rank, then maximum top-1 count",
            "not_frozen_for_target": True,
        },
        "claim_boundary": protocol["decision"],
    }
    result_bytes = _canonical(result)
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": "prob4d.tracking-cloth-augmented-rod-source-audit-manifest.v1",
            "source_revision": source_revision,
            "protocol_sha256": _sha256(protocol_path.read_bytes()),
            "result_sha256": _sha256(result_bytes),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "self_collision_trajectory_paths_opened": [],
            "raw_trajectory_payload_copied": False,
        },
    )
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    result = run(
        args.dataset_root.resolve(),
        args.protocol.resolve(),
        args.output_dir.resolve(),
        args.source_revision,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
