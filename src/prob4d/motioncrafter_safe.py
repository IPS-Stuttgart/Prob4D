"""Safe command-line producer and verifier for MotionCrafter bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .motioncrafter import (
    MOTIONCRAFTER_SEED_POLICIES,
    MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    MotionCrafterRunConfig,
)
from .motioncrafter_integrity import (
    MOTIONCRAFTER_MANIFEST_FILENAME,
    verify_motioncrafter_prediction_manifest,
)
from .motioncrafter_runner import SafeMotionCrafterRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path, nargs="?")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["determ", "diff"], default="determ")
    parser.add_argument("--unet-path", default="TencentARC/MotionCrafter")
    parser.add_argument("--vae-path", default="TencentARC/MotionCrafter")
    parser.add_argument("--cache-dir", default="workspace/pretrained_models")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--decode-chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-policy",
        choices=MOTIONCRAFTER_SEED_POLICIES,
        default=MOTIONCRAFTER_SEED_POLICY_LEGACY_COMMON,
    )
    parser.add_argument("--low-memory-usage", action="store_true")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only when the recorded run spec and completed hashes match",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify predictions.json without loading MotionCrafter",
    )
    parser.add_argument(
        "--allow-dirty-upstream",
        action="store_true",
        help="allow a dirty upstream checkout while binding its status digest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    manifest_path = arguments.output_dir / MOTIONCRAFTER_MANIFEST_FILENAME
    if arguments.verify_only:
        verification = verify_motioncrafter_prediction_manifest(manifest_path)
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0
    if arguments.video_path is None:
        parser.error("video_path is required unless --verify-only is used")
    if arguments.upstream_root is None:
        parser.error("--upstream-root is required unless --verify-only is used")

    config = MotionCrafterRunConfig(
        upstream_root=arguments.upstream_root,
        video_path=arguments.video_path,
        output_directory=arguments.output_dir,
        model_type=arguments.model_type,
        unet_path=arguments.unet_path,
        vae_path=arguments.vae_path,
        cache_directory=arguments.cache_dir,
        height=arguments.height,
        width=arguments.width,
        window_size=arguments.window_size,
        overlap=arguments.overlap,
        num_inference_steps=arguments.num_inference_steps,
        guidance_scale=arguments.guidance_scale,
        decode_chunk_size=arguments.decode_chunk_size,
        seed=arguments.seed,
        seed_policy=arguments.seed_policy,
        low_memory_usage=arguments.low_memory_usage,
        frame_start=arguments.frame_start,
        frame_stop=arguments.frame_stop,
        frame_stride=arguments.frame_stride,
    )
    manifest = SafeMotionCrafterRunner(
        config,
        allow_dirty_upstream=arguments.allow_dirty_upstream,
    ).run(resume=arguments.resume)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
