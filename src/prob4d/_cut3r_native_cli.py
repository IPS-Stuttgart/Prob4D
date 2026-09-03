"""Lazy CLI for optional native CUT3R inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .cut3r_native_provider import run_cut3r_native
from .cut3r_native_runtime import SUPPORTED_CUT3R_REVISION, Cut3RNativeRuntime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prob4d prediction run-cut3r")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frames", type=Path, help="directory of six-digit numbered images")
    inputs.add_argument("--video", type=Path, help="video from which to emit a bounded prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, required=True, help="exclusive causal bound")
    parser.add_argument("--frame-extension", choices=(".png", ".jpg", ".jpeg"), default=".png")
    parser.add_argument("--cut3r-checkout", type=Path, required=True)
    parser.add_argument("--cut3r-revision", default=SUPPORTED_CUT3R_REVISION)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True, help="digest of trusted weights")
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-native-build-compatibility",
        action="store_true",
        help="accept only the hash-pinned modern-ATen/SM89 build variant",
    )
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    def runtime() -> Cut3RNativeRuntime:
        return Cut3RNativeRuntime(
            args.cut3r_checkout,
            args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            revision=args.cut3r_revision,
            image_size=args.image_size,
            device=args.device,
            seed=args.seed,
            allow_native_build_compatibility=args.allow_native_build_compatibility,
        )

    receipt = run_cut3r_native(
        args.video if args.video is not None else args.frames,
        args.output,
        runtime_factory=runtime,
        sequence_id=args.sequence_id,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        video=args.video is not None,
        extension=args.frame_extension,
        confidence_threshold=args.confidence_threshold,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "artifact_id": receipt["artifact_id"],
                "frames_completed": receipt["frames_completed"],
                "provider_manifest": str(args.output / "prediction/provider.json"),
            },
            sort_keys=True,
        )
    )
    return 0
