"""Profile bounded-memory Prob4D evaluation on deterministic synthetic fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation_modes import evaluate_sequence_modes
from .fusion import FusedSequence
from .metrics import DEFAULT_EVALUATION_CHUNK_SIZE, TruthSequence


def _git_revision(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def _metrics_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_inputs(
    *,
    frames: int,
    height: int,
    width: int,
    seed: int,
    include_flow: bool,
) -> tuple[FusedSequence, TruthSequence]:
    generator = np.random.default_rng(seed)
    shape = (frames, height, width)
    truth_points = generator.normal(size=shape + (3,))
    truth_points[..., 2] += 4.0
    valid = generator.random(shape) > 0.03

    prediction_points = (truth_points - np.asarray([0.2, -0.1, 0.05])) / 1.04
    prediction_points += generator.normal(scale=0.01, size=truth_points.shape)
    point_variance = 0.01 + 0.002 * np.square(
        np.linalg.norm(prediction_points, axis=-1)
    )
    point_covariance = np.zeros(shape + (3, 3), dtype=np.float64)
    diagonal = np.arange(3)
    point_covariance[..., diagonal, diagonal] = point_variance[..., None]

    scene_flow: np.ndarray | None = None
    truth_flow: np.ndarray | None = None
    deform_mask: np.ndarray | None = None
    flow_covariance: np.ndarray | None = None
    if include_flow:
        truth_flow = generator.normal(scale=0.03, size=truth_points.shape)
        scene_flow = truth_flow / 1.04
        scene_flow += generator.normal(scale=0.002, size=truth_points.shape)
        deform_mask = valid & (generator.random(shape) > 0.1)
        flow_covariance = np.zeros_like(point_covariance)
        flow_covariance[..., diagonal, diagonal] = 0.0025

    prediction = FusedSequence(
        frame_indices=np.arange(frames),
        point_map=prediction_points,
        valid_mask=valid,
        point_covariance=point_covariance,
        contributors=np.ones(shape, dtype=np.uint16),
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        flow_covariance=flow_covariance,
    )
    truth = TruthSequence(
        frame_indices=np.arange(frames),
        point_map=truth_points,
        valid_mask=valid,
        scene_flow=truth_flow,
        deform_mask=deform_mask,
    )
    return prediction, truth


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.frames < 2:
        raise ValueError("frames must be at least two")
    if arguments.height < 1 or arguments.width < 1:
        raise ValueError("height and width must be positive")
    if arguments.evaluation_chunk_size < 1:
        raise ValueError("evaluation chunk size must be positive")

    construction_started = time.perf_counter()
    prediction, truth = _build_inputs(
        frames=arguments.frames,
        height=arguments.height,
        width=arguments.width,
        seed=arguments.seed,
        include_flow=arguments.include_flow,
    )
    construction_seconds = time.perf_counter() - construction_started

    evaluation_started = time.perf_counter()
    modes = evaluate_sequence_modes(
        prediction,
        truth,
        boundary_frames=[arguments.frames // 2],
        prefix_frame_stop_exclusive=max(1, arguments.frames // 2),
        evaluation_chunk_size=arguments.evaluation_chunk_size,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    result = modes.to_dict()

    prediction_arrays = [
        prediction.frame_indices,
        prediction.point_map,
        prediction.valid_mask,
        prediction.point_covariance,
        prediction.contributors,
    ]
    truth_arrays = [truth.frame_indices, truth.point_map, truth.valid_mask]
    if prediction.scene_flow is not None:
        assert prediction.deform_mask is not None
        assert prediction.flow_covariance is not None
        assert truth.scene_flow is not None
        assert truth.deform_mask is not None
        prediction_arrays.extend(
            [
                prediction.scene_flow,
                prediction.deform_mask,
                prediction.flow_covariance,
            ]
        )
        truth_arrays.extend([truth.scene_flow, truth.deform_mask])

    repository_root = Path(__file__).resolve().parents[2]
    evaluated_points = modes.metric.metrics.evaluated_points
    legacy_aligned_materialization = (
        prediction.point_map.nbytes + prediction.point_covariance.nbytes
    )
    if prediction.scene_flow is not None:
        assert prediction.flow_covariance is not None
        legacy_aligned_materialization += (
            prediction.scene_flow.nbytes + prediction.flow_covariance.nbytes
        )
    return {
        "schema": "prob4d.evaluation-memory-benchmark",
        "version": 1,
        "repository_revision": _git_revision(repository_root),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "configuration": {
            "frames": arguments.frames,
            "height": arguments.height,
            "width": arguments.width,
            "seed": arguments.seed,
            "include_flow": arguments.include_flow,
            "evaluation_chunk_size": arguments.evaluation_chunk_size,
            "modes": ["metric", "prefix_aligned", "oracle_aligned"],
        },
        "timing_seconds": {
            "input_construction": construction_seconds,
            "evaluation": evaluation_seconds,
        },
        "memory_bytes": {
            "peak_process_rss": _peak_rss_bytes(),
            "retained_prediction": sum(value.nbytes for value in prediction_arrays),
            "retained_truth": sum(value.nbytes for value in truth_arrays),
            "legacy_evaluate_sequence_point_covariance_copies": (
                prediction.point_map.nbytes + prediction.point_covariance.nbytes
            ),
            "legacy_per_aligned_mode_materialization": legacy_aligned_materialization,
            "retained_scalar_diagnostics_upper_bound": evaluated_points * 3 * 8,
        },
        "output": {
            "metrics_digest": _metrics_digest(result),
            "metric_point_rmse": modes.metric.metrics.point_rmse,
            "prefix_point_rmse": (
                None
                if modes.prefix_aligned is None
                else modes.prefix_aligned.metrics.point_rmse
            ),
            "oracle_point_rmse": modes.oracle_aligned.metrics.point_rmse,
            "evaluated_points": evaluated_points,
        },
        "claim_boundary": (
            "Synthetic process-level engineering profile only. Peak RSS includes input "
            "construction and does not establish real-data accuracy, calibration, "
            "Bayesian-PhysTwin benefit, or Causal4D benefit."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--include-flow", action="store_true")
    parser.add_argument(
        "--evaluation-chunk-size",
        type=int,
        default=DEFAULT_EVALUATION_CHUNK_SIZE,
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_benchmark(arguments)
    except ValueError as error:
        parser.error(str(error))
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
