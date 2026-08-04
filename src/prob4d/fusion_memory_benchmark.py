"""Profile dense Prob4D fusion with deterministic synthetic overlapping windows."""

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

from prob4d.data import PredictionWindow
from prob4d.fusion import DEFAULT_FUSION_TILE_SIZE, fuse_windows
from prob4d.sim3 import Sim3
from prob4d.uncertainty import StructuredCovariance


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


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _build_inputs(
    *,
    frames: int,
    height: int,
    width: int,
    contributor_count: int,
    seed: int,
) -> tuple[
    list[PredictionWindow],
    dict[str, Sim3],
    dict[str, StructuredCovariance],
    dict[str, np.ndarray],
]:
    generator = np.random.default_rng(seed)
    base = generator.normal(size=(frames, height, width, 3)).astype(np.float32)
    base[..., 2] += np.float32(4.0)
    mask = np.ones((frames, height, width), dtype=bool)
    frame_indices = np.arange(frames, dtype=np.int64)
    windows: list[PredictionWindow] = []
    gauges: dict[str, Sim3] = {}
    uncertainties: dict[str, StructuredCovariance] = {}
    gauge_covariances: dict[str, np.ndarray] = {}

    for index in range(contributor_count):
        values = base.copy()
        values[..., 0] += np.float32(0.02 * index)
        window_id = f"window-{index:02d}"
        window = PredictionWindow(
            window_id=window_id,
            frame_indices=frame_indices,
            point_map=values,
            valid_mask=mask,
            dense_storage_dtype="float32",
        )
        rays = values.astype(np.float64)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        parallel = np.full(window.shape, 0.01 * (index + 1), dtype=np.float64)
        lateral = np.full(window.shape, 0.02 * (index + 1), dtype=np.float64)
        windows.append(window)
        gauges[window_id] = Sim3(
            scale=1.0 + 0.01 * index,
            translation=np.asarray([0.01 * index, -0.005 * index, 0.0]),
        )
        uncertainties[window_id] = StructuredCovariance(rays, parallel, lateral)
        gauge_covariances[window_id] = np.diag(
            [1e-4, 2e-4, 2e-4, 2e-4, 1e-5, 1e-5, 1e-5]
        )
    return windows, gauges, uncertainties, gauge_covariances


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.frames < 1:
        raise ValueError("frames must be positive")
    if arguments.height < 1 or arguments.width < 1:
        raise ValueError("height and width must be positive")
    if arguments.contributors < 1:
        raise ValueError("contributors must be positive")
    if arguments.fusion_tile_size < 1:
        raise ValueError("fusion tile size must be positive")

    construction_started = time.perf_counter()
    windows, gauges, uncertainties, gauge_covariances = _build_inputs(
        frames=arguments.frames,
        height=arguments.height,
        width=arguments.width,
        contributor_count=arguments.contributors,
        seed=arguments.seed,
    )
    construction_seconds = time.perf_counter() - construction_started

    fusion_started = time.perf_counter()
    fused = fuse_windows(
        windows,
        gauges,
        uncertainties,
        method=arguments.method,
        gauge_covariances=gauge_covariances,
        fusion_tile_size=arguments.fusion_tile_size,
    )
    fusion_seconds = time.perf_counter() - fusion_started

    retained_prediction_bytes = sum(window.dense_vector_storage_bytes for window in windows)
    retained_structured_bytes = sum(
        uncertainty.ray_directions.nbytes
        + uncertainty.parallel_variance.nbytes
        + uncertainty.lateral_variance.nbytes
        for uncertainty in uncertainties.values()
    )
    output_bytes = sum(
        value.nbytes
        for value in (
            fused.frame_indices,
            fused.point_map,
            fused.valid_mask,
            fused.point_covariance,
            fused.contributors,
        )
    )
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "schema": "prob4d.dense-fusion-memory-benchmark",
        "version": 1,
        "repository_revision": _git_revision(repository_root),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "configuration": {
            "height": arguments.height,
            "width": arguments.width,
            "frames": arguments.frames,
            "contributors": arguments.contributors,
            "seed": arguments.seed,
            "method": arguments.method,
            "fusion_tile_size": arguments.fusion_tile_size,
            "gauge_covariance": True,
            "dense_storage_dtype": "float32",
        },
        "timing_seconds": {
            "input_construction": construction_seconds,
            "fusion": fusion_seconds,
        },
        "memory_bytes": {
            "peak_process_rss": _peak_rss_bytes(),
            "retained_prediction_vectors": retained_prediction_bytes,
            "retained_structured_covariance": retained_structured_bytes,
            "fused_output": output_bytes,
        },
        "output": {
            "artifact_digest": _digest_arrays(
                fused.frame_indices,
                fused.point_map,
                fused.valid_mask,
                fused.point_covariance,
                fused.contributors,
            ),
            "point_sum": float(np.sum(fused.point_map)),
            "covariance_trace_sum": float(
                np.trace(fused.point_covariance, axis1=-2, axis2=-1).sum()
            ),
            "contributors_sum": int(fused.contributors.sum()),
        },
        "claim_boundary": (
            "Synthetic process-level engineering profile only; peak RSS includes input "
            "construction and does not establish real-data accuracy or calibration."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--contributors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--method",
        choices=("uniform", "precision", "covariance_intersection"),
        default="covariance_intersection",
    )
    parser.add_argument(
        "--fusion-tile-size",
        type=int,
        default=DEFAULT_FUSION_TILE_SIZE,
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
