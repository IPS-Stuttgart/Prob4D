#!/usr/bin/env python3
"""Dataset-free native CUT3R check: export, streaming equivalence and prefix closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prob4d.cut3r_native_provider import content_id, run_cut3r_native
from prob4d.cut3r_native_runtime import (
    OFFICIAL_512_CHECKPOINT_SHA256,
    Cut3RNativeRuntime,
)
from prob4d.prediction_provider_manifest import verify_prediction_provider_manifest


def make_frames(root: Path) -> list[Path]:
    from PIL import Image

    root.mkdir()
    y, x = np.indices((384, 512))
    paths = []
    for i in range(3):
        rgb = np.stack(
            [
                (x + i * 8) % 256,
                (y * 2) % 256,
                (((x + i * 8) // 24 + y // 24) % 2) * 180 + 40,
            ],
            axis=-1,
        ).astype(np.uint8)
        path = root / f"{i:06d}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=OFFICIAL_512_CHECKPOINT_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--allow-native-build-compatibility", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    report = {
        "schema": "prob4d.cut3r-native-synthetic-smoke-v1",
        "dataset_accessed": False,
        "scientific_accuracy_claimed": False,
        "status": "failed",
    }
    try:
        paths = make_frames(args.output / "synthetic")
        runtime = Cut3RNativeRuntime(
            args.cut3r_checkout,
            args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            device=args.device,
            image_size=args.image_size,
            allow_native_build_compatibility=args.allow_native_build_compatibility,
        )
        torch = runtime.torch
        torch.cuda.reset_peak_memory_stats(runtime.device)
        run = run_cut3r_native(
            paths[0].parent,
            args.output / "production",
            runtime_factory=lambda: runtime,
            sequence_id="synthetic-runtime-check",
            frame_start=0,
            frame_stop=3,
        )
        verify_prediction_provider_manifest(args.output / "production/prediction/provider.json")
        direct = args.output / "production/direct"
        expected = [np.load(direct / "points" / f"{i:06d}.npy") for i in range(3)]
        runtime.reset()
        prefix_error = max(
            float(np.max(np.abs(runtime.step(path)["points"] - expected[i])))
            for i, path in enumerate(paths[:2])
        )
        runtime.reset()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            views = [runtime.prepare_view(path) for path in paths]
            for view in views:
                height, width = view["img"].shape[-2:]
                view["ray_map"] = torch.full(
                    (1, height, width, 6), torch.nan, device=runtime.device
                )
            predictions, _ = runtime.model.forward_recurrent(views, runtime.device)
            reference_error = max(
                float(np.max(np.abs(runtime.decode_prediction(pred)["points"] - expected[i])))
                for i, pred in enumerate(predictions)
            )
        if prefix_error > 1e-6 or reference_error > 1e-5:
            raise ValueError("synthetic streaming/prefix equivalence tolerance exceeded")
        report.update(
            {
                "status": "success",
                "production_artifact_id": run["artifact_id"],
                "manifest_sha256": run["manifest_sha256"],
                "runtime": runtime.identity,
                "frames_exported": 3,
                "point_shape": list(expected[0].shape),
                "prefix_point_max_abs_error": prefix_error,
                "official_recurrent_point_max_abs_error": reference_error,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
            }
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["artifact_id"] = content_id(report)
        with (args.output / "smoke.json").open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
