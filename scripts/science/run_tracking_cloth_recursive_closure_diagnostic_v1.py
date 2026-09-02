#!/usr/bin/env python3
"""Diagnose strict recursive task-closure size on public Tracking Cloth data.

This is a retrospective structural diagnostic on an already-open public release.
It does not score held-out predictions and does not select a model by target
performance.  Its purpose is to test whether the strict LTI task-state closure
used by the controlled recursive-compression mechanism remains smaller than the
complete real-cloth displacement state after fitting source geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.recursive_task_sufficiency import recursive_linear_task_closure
from scripts.science.run_tracking_cloth_query_portfolio_v1 import (
    Samples,
    make_samples,
    parse_recording,
    select_query_indices,
)

SCHEMA = "prob4d.tracking-cloth-recursive-closure-diagnostic.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_rank(matrix: np.ndarray, relative_tolerance: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not singular.size or singular[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular > relative_tolerance * singular[0]))


def _fit_centered_map(
    input_values: np.ndarray,
    output_values: np.ndarray,
    ridge_relative: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if input_values.ndim != 2 or output_values.ndim != 2:
        raise ValueError("linear-map inputs must be matrices")
    if len(input_values) != len(output_values) or len(input_values) < 2:
        raise ValueError("linear-map fit requires paired rows")
    centered_input = input_values - input_values.mean(axis=0)
    centered_output = output_values - output_values.mean(axis=0)
    normalization = float(max(len(centered_input) - 1, 1))
    gram = centered_input.T @ centered_input / normalization
    diagonal = np.diag(gram)
    positive = diagonal[diagonal > 0.0]
    if not len(positive):
        raise ValueError("linear-map input has no positive variance")
    ridge = ridge_relative * float(np.median(positive))
    cross = centered_input.T @ centered_output / normalization
    coefficients = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), cross)
    fitted = centered_input @ coefficients
    residual = centered_output - fitted
    denominator = max(float(np.linalg.norm(centered_output, ord="fro")), 1e-30)
    return coefficients.T, {
        "paired_rows": int(len(centered_input)),
        "ridge": ridge,
        "relative_training_residual": float(np.linalg.norm(residual, ord="fro")) / denominator,
    }


def _task_matrices(marker_count: int) -> dict[str, np.ndarray]:
    dimension = 3 * marker_count
    centroid = np.zeros((3, dimension), dtype=np.float64)
    for marker in range(marker_count):
        centroid[:, 3 * marker : 3 * marker + 3] = np.eye(3) / marker_count

    central_index = marker_count // 2
    central = np.zeros((3, dimension), dtype=np.float64)
    central[:, 3 * central_index : 3 * central_index + 3] = np.eye(3)

    marker_indices = select_query_indices(marker_count, min(4, marker_count))
    four = np.zeros((3 * len(marker_indices), dimension), dtype=np.float64)
    for row, marker in enumerate(marker_indices):
        four[3 * row : 3 * row + 3, 3 * marker : 3 * marker + 3] = np.eye(3)

    return {
        "centroid": centroid,
        "one_central_marker": central,
        "four_evenly_spaced_markers": four,
    }


def _collect_samples(
    dataset_root: Path, protocol: dict[str, Any]
) -> tuple[list[Samples], dict[str, Any]]:
    csv_files = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    samples: list[Samples] = []
    rejected: list[dict[str, str]] = []
    compatible_sources: list[dict[str, object]] = []
    for path in csv_files:
        try:
            recording = parse_recording(path, dataset_root)
            sample = make_samples(recording, protocol)
        except ValueError as exc:
            rejected.append(
                {
                    "relative_path": path.relative_to(dataset_root).as_posix(),
                    "reason": str(exc),
                }
            )
            continue
        samples.append(sample)
        compatible_sources.append(
            {
                "relative_path": recording.relative_path,
                "size": recording.size,
                "marker_count": recording.marker_count,
                "source_sha256": recording.source_sha256,
                "source_bytes": recording.source_bytes,
            }
        )
    inventory_digest = hashlib.sha256(_canonical_bytes(compatible_sources)).hexdigest()
    return samples, {
        "csv_file_count": len(csv_files),
        "accepted_recording_count": len(samples),
        "rejected_recording_count": len(rejected),
        "compatible_source_manifest_sha256": inventory_digest,
        "accepted_by_size": {
            size: sum(sample.size == size for sample in samples) for size in ("A2", "A3")
        },
        "rejection_reasons": sorted({row["reason"] for row in rejected}),
    }


def _size_diagnostic(
    records: list[Samples],
    protocol: dict[str, Any],
    size: str,
) -> dict[str, Any]:
    selected = [record for record in records if record.size == size]
    if not selected:
        raise ValueError(f"no compatible {size} recordings")
    marker_count = selected[0].observations_m.shape[1]
    if any(record.observations_m.shape[1] != marker_count for record in selected):
        raise ValueError(f"{size} marker count is inconsistent")

    state_rows = np.concatenate(
        [
            record.future_displacements_m.reshape(len(record.future_displacements_m), -1)
            for record in selected
        ]
    )
    observation_rows = np.concatenate(
        [record.observations_m.reshape(len(record.observations_m), -1) for record in selected]
    )
    transition_input = np.concatenate(
        [
            record.future_displacements_m[:-1].reshape(len(record.future_displacements_m) - 1, -1)
            for record in selected
            if len(record.future_displacements_m) > 1
        ]
    )
    transition_output = np.concatenate(
        [
            record.future_displacements_m[1:].reshape(len(record.future_displacements_m) - 1, -1)
            for record in selected
            if len(record.future_displacements_m) > 1
        ]
    )
    ridge_relative = float(protocol["model"]["ridge_relative_to_median_state_variance"])
    observation_map, observation_fit = _fit_centered_map(
        state_rows,
        observation_rows,
        ridge_relative,
    )
    transition_map, transition_fit = _fit_centered_map(
        transition_input,
        transition_output,
        ridge_relative,
    )
    tasks = _task_matrices(marker_count)
    rows: list[dict[str, Any]] = []
    for tolerance_raw in protocol["rank_relative_tolerances"]:
        tolerance = float(tolerance_raw)
        observation_rank = _effective_rank(observation_map, tolerance)
        for task_name in protocol["tasks"]:
            task = tasks[str(task_name)]
            task_only = recursive_linear_task_closure(
                transition_map,
                task_matrix=task,
                observation_matrix=np.zeros((0, transition_map.shape[0])),
                rank_relative_tolerance=tolerance,
            )
            strict = recursive_linear_task_closure(
                transition_map,
                task_matrix=task,
                observation_matrix=observation_map,
                rank_relative_tolerance=tolerance,
            )
            rows.append(
                {
                    "rank_relative_tolerance": tolerance,
                    "task": task_name,
                    "task_dimension": int(task.shape[0]),
                    "observation_map_rank": observation_rank,
                    "task_only_closure_dimension": task_only.closure_dimension,
                    "strict_closure_dimension": strict.closure_dimension,
                    "state_dimension": int(transition_map.shape[0]),
                    "strict_closure_fraction": strict.closure_dimension / transition_map.shape[0],
                }
            )
    return {
        "size": size,
        "recording_count": len(selected),
        "marker_count": marker_count,
        "state_dimension": int(3 * marker_count),
        "window_count": int(len(state_rows)),
        "transition_pair_count": int(len(transition_input)),
        "observation_fit": observation_fit,
        "transition_fit": transition_fit,
        "rows": rows,
    }


def run(dataset_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    samples, inventory = _collect_samples(dataset_root, protocol)
    expected_csv = int(protocol["dataset"]["expected_csv_files"])
    expected_compatible = int(protocol["dataset"]["expected_compatible_recordings"])
    if inventory["csv_file_count"] != expected_csv:
        raise RuntimeError("official dataset CSV inventory changed")
    if inventory["accepted_recording_count"] != expected_compatible:
        raise RuntimeError("compatible recording inventory changed")
    sizes = [_size_diagnostic(samples, protocol, size) for size in ("A2", "A3")]
    all_rows = [row for item in sizes for row in item["rows"]]
    return {
        "schema": SCHEMA,
        "status": "evaluated-retrospective-structural-closure",
        "inventory": inventory,
        "sizes": sizes,
        "aggregate": {
            "all_strict_closures_full_state": all(
                row["strict_closure_dimension"] == row["state_dimension"] for row in all_rows
            ),
            "minimum_strict_closure_fraction": min(
                row["strict_closure_fraction"] for row in all_rows
            ),
            "maximum_strict_closure_fraction": max(
                row["strict_closure_fraction"] for row in all_rows
            ),
            "minimum_task_only_closure_fraction": min(
                row["task_only_closure_dimension"] / row["state_dimension"] for row in all_rows
            ),
            "maximum_task_only_closure_fraction": max(
                row["task_only_closure_dimension"] / row["state_dimension"] for row in all_rows
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/tracking-cloth-recursive-closure-diagnostic-v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["schema"] != SCHEMA:
        raise ValueError("unsupported protocol")
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists; never overwrite a diagnostic")
    args.output_dir.mkdir(parents=True)

    result = run(args.dataset_root, protocol)
    result_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    (args.output_dir / "protocol.json").write_bytes(protocol_bytes)
    (args.output_dir / "result.json").write_bytes(result_bytes)
    manifest = {
        "source_revision": args.source_revision,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dataset_doi": protocol["dataset"]["doi"],
        "raw_data_retained": False,
        "independent_confirmation": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
