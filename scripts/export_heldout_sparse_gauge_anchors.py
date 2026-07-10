#!/usr/bin/env python3
"""Batch sparse gauge-anchor export on all or held-out benchmark sequences."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prob4d.sintel_uncertainty import discover_inputs, held_out_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gauge-calibration", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    all_inputs = discover_inputs(args.dataset_dir, args.results_dir)
    if args.split == "test":
        _, inputs = held_out_split(all_inputs)
    else:
        inputs = all_inputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exporter = Path(__file__).with_name("export_sparse_gauge_anchors.py")

    def run(index: int) -> tuple[str, Path]:
        item = inputs[index]
        stem = item.prediction_manifest.parent.name
        output = args.output_dir / item.sequence / f"{stem}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(exporter),
                "--manifest",
                str(item.prediction_manifest),
                "--ground-truth",
                str(item.ground_truth_hdf5),
                "--output",
                str(output),
                "--gauge-calibration",
                str(args.gauge_calibration),
                "--max-depth",
                str(args.max_depth),
                "--initialization-points",
                "16",
                "--anchors-per-window",
                "16",
                "--measurement-std",
                "0.01",
                "--seed",
                str(args.seed + index),
            ],
            check=True,
        )
        return item.sequence, output

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, index): index for index in range(len(inputs))}
        for future in as_completed(futures):
            sequence, output = future.result()
            print(sequence, output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
