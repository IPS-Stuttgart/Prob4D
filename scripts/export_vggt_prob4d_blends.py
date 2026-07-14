#!/usr/bin/env python3
"""Align VGGT to Prob4D without ground truth and export point-map blends."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from prob4d.alignment import estimate_sim3_robust
from prob4d.sintel_uncertainty import _resize_bilinear


def parse_alphas(text: str) -> list[float]:
    """Parse unique Prob4D blend weights in ascending order."""

    values = sorted({float(value) for value in text.split(",")})
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("alphas must be comma-separated values strictly between zero and one")
    return values


def alpha_name(alpha: float) -> str:
    """Return a stable directory name for a Prob4D blend weight."""

    return f"prob4d_{alpha:.2f}".replace(".", "p")


def sampled_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample same-pixel correspondences for model-to-model registration."""

    active = np.flatnonzero(mask.reshape(-1))
    if active.size < 4:
        raise ValueError("fewer than four valid model correspondences")
    if active.size > maximum:
        generator = np.random.default_rng(seed)
        active = np.sort(generator.choice(active, size=maximum, replace=False))
    return source.reshape(-1, 3)[active], target.reshape(-1, 3)[active]


def blend_point_maps(
    prob4d_points: np.ndarray,
    external_points: np.ndarray,
    external_from_local,
    alpha: float,
) -> np.ndarray:
    """Blend in the Prob4D gauge after mapping external points into it."""

    aligned_external = external_from_local.transform_points(external_points)
    return alpha * prob4d_points + (1.0 - alpha) * aligned_external


def git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob4d-results-dir", type=Path, required=True)
    parser.add_argument("--vggt-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0.25,0.50,0.75")
    parser.add_argument("--maximum-registration-points", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    if args.maximum_registration_points < 4:
        raise ValueError("maximum-registration-points must be at least four")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    predictions = sorted(args.prob4d_results_dir.glob("part*/prob4d_uniform/*/*.npz"))
    if not predictions:
        raise FileNotFoundError("no part*/prob4d_uniform prediction files found")
    repository = Path(__file__).resolve().parents[1]

    def run(index: int, prob4d_path: Path) -> tuple[str, dict[str, object]]:
        sequence = prob4d_path.parent.name
        vggt_path = args.vggt_prediction_dir / sequence / prob4d_path.name
        if not vggt_path.exists():
            raise FileNotFoundError(vggt_path)
        output_paths = {
            alpha: args.output_dir / alpha_name(alpha) / sequence / prob4d_path.name
            for alpha in alphas
        }
        report_path = args.output_dir / "reports" / sequence / f"{prob4d_path.stem}.json"
        if (
            args.skip_existing
            and report_path.exists()
            and all(path.exists() for path in output_paths.values())
        ):
            return sequence, json.loads(report_path.read_text(encoding="utf-8"))

        with np.load(prob4d_path, allow_pickle=False) as payload:
            prob4d_points = payload["point_map"].astype(np.float32)
            valid_mask = payload["valid_mask"].astype(bool)
        with np.load(vggt_path, allow_pickle=False) as payload:
            vggt_points = payload["point_map"].astype(np.float32)
        frame_count = min(prob4d_points.shape[0], vggt_points.shape[0])
        prob4d_points = prob4d_points[:frame_count]
        valid_mask = valid_mask[:frame_count]
        vggt_points = _resize_bilinear(vggt_points[:frame_count], prob4d_points.shape[1:3])
        finite = np.isfinite(prob4d_points).all(axis=-1) & np.isfinite(vggt_points).all(axis=-1)
        source, target = sampled_correspondences(
            vggt_points,
            prob4d_points,
            valid_mask & finite,
            maximum=args.maximum_registration_points,
            seed=args.seed + index,
        )
        fit = estimate_sim3_robust(source, target)
        aligned_vggt = fit.transform.transform_points(vggt_points)
        for alpha, output_path in output_paths.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            blended = alpha * prob4d_points + (1.0 - alpha) * aligned_vggt
            np.savez_compressed(output_path, point_map=blended.astype(np.float16))
        report = {
            "sequence": sequence,
            "prob4d_prediction": str(prob4d_path.resolve()),
            "vggt_prediction": str(vggt_path.resolve()),
            "prob4d_commit": git_commit(repository),
            "alphas": alphas,
            "registration": {
                "global_from_vggt": fit.transform.as_vector().tolist(),
                "residual_rms": fit.residual_rms,
                "inlier_fraction": fit.inlier_fraction,
                "correspondences": fit.num_correspondences,
            },
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return sequence, report

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, path): path for index, path in enumerate(predictions)
        }
        for future in as_completed(futures):
            sequence, report = future.result()
            print(sequence, report["registration"]["residual_rms"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
