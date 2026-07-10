#!/usr/bin/env python3
"""Export one fusion variant from cached MotionCrafter prediction bundles."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

from prob4d.benchmark import (
    FUSION_METHOD_NAMES,
    _write_fused_prediction,
    fuse_prediction_bundle_methods,
)
from prob4d.io import load_prediction_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=FUSION_METHOD_NAMES, required=True)
    parser.add_argument("--include-covariance", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


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
    artifact_root = args.artifact_root.resolve()
    manifests = sorted(artifact_root.rglob("predictions.json"))
    if not manifests:
        raise FileNotFoundError(f"no predictions.json files below {artifact_root}")

    samples: list[dict[str, object]] = []
    for index, manifest in enumerate(manifests):
        relative = manifest.parent.relative_to(artifact_root).with_suffix(".npz")
        destination = args.output_dir / relative
        if args.skip_existing and destination.exists():
            samples.append({"prediction": relative.as_posix(), "status": "existing"})
            continue

        started = time.perf_counter()
        bundle = load_prediction_bundle(manifest)
        fused = fuse_prediction_bundle_methods(bundle, method_names={args.method})[args.method]
        _write_fused_prediction(
            destination,
            fused,
            include_covariance=args.include_covariance,
        )
        samples.append(
            {
                "prediction": relative.as_posix(),
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - started,
                "frames": int(fused.frame_indices.size),
                "index": index,
            }
        )
        print(destination, flush=True)
        del bundle, fused
        gc.collect()

    repository = Path(__file__).resolve().parents[1]
    report = {
        "format_version": 1,
        "artifact_root": str(artifact_root),
        "output_directory": str(args.output_dir.resolve()),
        "method": args.method,
        "include_covariance": args.include_covariance,
        "prob4d_commit": git_commit(repository),
        "samples": samples,
    }
    report_path = args.output_dir / "cached_fusion_export.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
