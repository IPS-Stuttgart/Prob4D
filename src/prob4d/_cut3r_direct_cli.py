"""Command-line interface for direct recurrent-online CUT3R point maps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any, cast

from ._cut3r_limits import DEFAULT_CUT3R_IMPORT_LIMITS, Cut3RImportLimits
from .data import DENSE_STORAGE_DTYPES, DenseStorageDType
from .prediction_provider_manifest import verify_prediction_provider_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction import-cut3r-direct",
        description=(
            "convert recurrent-online CUT3R direct XYZ point maps, confidence, "
            "and cameras into a provider-neutral prediction manifest"
        ),
    )
    parser.add_argument("source_root")
    parser.add_argument("output")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--cut3r-revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--input-video-sha256", required=True)
    parser.add_argument("--input-video-byte-count", type=int, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--view-id", default="camera-0")
    parser.add_argument("--window-id", default="cut3r-direct-online")
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument(
        "--storage-dtype",
        choices=DENSE_STORAGE_DTYPES,
        default="float32",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_CUT3R_IMPORT_LIMITS.max_frames,
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_CUT3R_IMPORT_LIMITS.max_height,
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_CUT3R_IMPORT_LIMITS.max_width,
    )
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        default=DEFAULT_CUT3R_IMPORT_LIMITS.max_source_bytes,
    )
    parser.add_argument(
        "--max-dense-bytes",
        type=int,
        default=DEFAULT_CUT3R_IMPORT_LIMITS.max_dense_bytes,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .cut3r_direct_provider_adapter import import_cut3r_direct_prediction_manifest

    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    limits = Cut3RImportLimits(
        max_frames=arguments.max_frames,
        max_height=arguments.max_height,
        max_width=arguments.max_width,
        max_source_bytes=arguments.max_source_bytes,
        max_dense_bytes=arguments.max_dense_bytes,
    )
    manifest = import_cut3r_direct_prediction_manifest(
        arguments.source_root,
        arguments.output,
        sequence_id=arguments.sequence_id,
        cut3r_revision=arguments.cut3r_revision,
        checkpoint_sha256=arguments.checkpoint_sha256,
        input_video_sha256=arguments.input_video_sha256,
        input_video_byte_count=arguments.input_video_byte_count,
        frame_start=arguments.frame_start,
        view_id=arguments.view_id,
        window_id=arguments.window_id,
        confidence_threshold=arguments.confidence_threshold,
        storage_dtype=cast(DenseStorageDType, arguments.storage_dtype),
        limits=limits,
    )
    _, report = verify_prediction_provider_manifest(arguments.output)
    output: dict[str, Any] = {
        **manifest.summary(),
        "verified_payload_count": report["verified_payload_count"],
        "geometry_source": "pts3d-in-self-view-direct-v1",
        "direct_pointmap_preserved": True,
        "depth_reprojection_used": False,
        "execution_mode": "recurrent-online",
        "online_prefix_only": True,
        "global_alignment": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0
