#!/usr/bin/env python3
"""Fit gauge-covariance inflation on whole-family-held-out Sintel data."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from prob4d.benchmark import _build_alignments
from prob4d.experiments import _window_truth_gauge
from prob4d.gauge import GaugeCovarianceCalibration
from prob4d.io import load_prediction_bundle
from prob4d.sintel_uncertainty import (
    discover_inputs,
    held_out_split,
    load_sintel_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trim-quantile", type=float, default=0.99)
    return parser.parse_args()


def normalized_quadratic(error: np.ndarray, covariance: np.ndarray) -> float:
    covariance = 0.5 * (covariance + covariance.T)
    return float(error @ np.linalg.pinv(covariance, rcond=1e-10) @ error) / error.size


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    calibration_inputs, test_inputs = held_out_split(
        discover_inputs(args.dataset_dir, args.results_dir)
    )
    errors: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for item in calibration_inputs:
        bundle = load_prediction_bundle(item.prediction_manifest)
        truth = load_sintel_truth(item.ground_truth_hdf5, max_depth=70.0)
        alignments = _build_alignments(bundle)
        true_gauges = {
            window.window_id: _window_truth_gauge(window, truth)
            for window in bundle.overlap_windows
        }
        for alignment in alignments:
            true_transform = (
                true_gauges[alignment.reference_id]
                .inverse()
                .compose(true_gauges[alignment.moving_id])
            )
            error = true_transform.inverse().compose(alignment.result.transform).as_vector()
            covariance = alignment.result.covariance
            errors.append(error)
            covariances.append(covariance)
            records.append(
                {
                    "sequence": item.sequence,
                    "reference_id": alignment.reference_id,
                    "moving_id": alignment.moving_id,
                    "error": error.tolist(),
                    "scale_ratio": float(error[0] ** 2 / max(covariance[0, 0], 1e-12)),
                    "rotation_ratio": normalized_quadratic(error[1:4], covariance[1:4, 1:4]),
                    "translation_ratio": normalized_quadratic(error[4:7], covariance[4:7, 4:7]),
                }
            )
        print(item.sequence, flush=True)
        del bundle, truth, alignments, true_gauges
        gc.collect()

    error_array = np.asarray(errors, dtype=np.float64)
    covariance_array = np.asarray(covariances, dtype=np.float64)
    calibration = GaugeCovarianceCalibration.fit(
        error_array,
        covariance_array,
        trim_quantile=args.trim_quantile,
    )
    ratio_summary: dict[str, dict[str, float]] = {}
    for name in ("scale", "rotation", "translation"):
        values = np.asarray([record[f"{name}_ratio"] for record in records], dtype=np.float64)
        ratio_summary[name] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        }

    repository = Path(__file__).resolve().parents[1]
    payload = {
        "format_version": 1,
        "prob4d_commit": git_commit(repository),
        "dataset_directory": str(args.dataset_dir.resolve()),
        "results_directory": str(args.results_dir.resolve()),
        "calibration_sequences": [item.sequence for item in calibration_inputs],
        "test_sequences": [item.sequence for item in test_inputs],
        "calibration": asdict(calibration),
        "inflation": {
            "scale": calibration.scale,
            "rotation": calibration.rotation,
            "translation": calibration.translation,
        },
        "ratio_summary": ratio_summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(payload["inflation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
