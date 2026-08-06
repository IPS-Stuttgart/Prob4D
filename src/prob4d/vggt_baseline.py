"""Export official VGGT predictions in the Prob4D evaluation format."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .vggt_integrity import (
    VGGT_REPRESENTATIONS,
    build_run_record,
    build_sample_record,
    checkpoint_identity,
    describe_prediction_archive,
    file_sha256,
    relative_member,
    save_vggt_run_metadata,
)


@dataclass(frozen=True)
class Sample:
    """One video and ground-truth pair from a MotionCrafter dataset list."""

    video_path: Path
    data_path: Path


def read_samples(dataset_root: Path) -> list[Sample]:
    """Read the two-column ``filename_list.txt`` dataset convention."""

    filename_list = dataset_root / "filename_list.txt"
    samples: list[Sample] = []
    for line_number, raw_line in enumerate(
        filename_list.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(
                f"{filename_list}:{line_number}: expected two columns, got {len(columns)}"
            )
        samples.append(Sample(Path(columns[0]), Path(columns[1])))
    if not samples:
        raise ValueError(f"no samples found in {filename_list}")
    return samples


def select_partition(samples: list[Sample], index: int, count: int) -> list[Sample]:
    """Select a deterministic strided partition for multi-GPU export."""

    if count < 1:
        raise ValueError("partition count must be positive")
    if not 0 <= index < count:
        raise ValueError("partition index must satisfy 0 <= index < count")
    return samples[index::count]


def canonicalize_to_first_camera(
    world_points: np.ndarray, extrinsics_world_to_camera: np.ndarray
) -> np.ndarray:
    """Express predicted world points in the first predicted camera frame."""

    points = np.asarray(world_points)
    extrinsics = np.asarray(extrinsics_world_to_camera)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("world_points must have shape (T, H, W, 3)")
    if extrinsics.shape != (points.shape[0], 3, 4):
        raise ValueError("extrinsics must have shape (T, 3, 4)")
    rotation = extrinsics[0, :, :3]
    translation = extrinsics[0, :, 3]
    return np.einsum("ij,...j->...i", rotation, points) + translation


def extract_video_frames(video_path: Path, output_directory: Path) -> list[Path]:
    """Decode every video frame to lossless PNG files for VGGT's official loader."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frame_paths: list[Path] = []
    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_path = output_directory / f"{frame_index:05d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"could not write decoded frame: {frame_path}")
            frame_paths.append(frame_path)
            frame_index += 1
    finally:
        capture.release()
    if not frame_paths:
        raise ValueError(f"video contains no decodable frames: {video_path}")
    return frame_paths


def git_commit(repository: Path) -> str:
    """Return the exact baseline implementation revision."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_vggt(
    vggt_root: Path,
    checkpoint: str,
    checkpoint_revision: str | None,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    """Load official VGGT model and conversion utilities lazily."""

    import torch

    sys.path.insert(0, str(vggt_root.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        if checkpoint_revision is not None:
            raise ValueError("--checkpoint-revision is not permitted for a local checkpoint file")
        model = VGGT()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        if checkpoint_revision is None:
            model = VGGT.from_pretrained(checkpoint)
        else:
            model = VGGT.from_pretrained(
                checkpoint,
                revision=checkpoint_revision,
            )
    model.track_head = None
    model.eval().to(device)
    return (
        model,
        load_and_preprocess_images,
        pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map,
    )


def infer_sample(
    *,
    model: Any,
    load_and_preprocess_images: Any,
    pose_encoding_to_extri_intri: Any,
    unproject_depth_map_to_point_map: Any,
    frame_paths: list[Path],
    device: str,
    preprocess_mode: str,
) -> dict[str, np.ndarray]:
    """Run VGGT once and return its two official world-point constructions."""

    import torch

    images = load_and_preprocess_images(
        [str(path) for path in frame_paths], mode=preprocess_mode
    ).to(device)
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=device.startswith("cuda"),
        ),
    ):
        predictions = model(images)

    image_shape = tuple(images.shape[-2:])
    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], image_shape)
    extrinsics_array = extrinsics[0].float().cpu().numpy()
    intrinsics_array = intrinsics[0].float().cpu().numpy()
    direct_points = predictions["world_points"][0].float().cpu().numpy()
    depth = predictions["depth"][0].float().cpu().numpy()
    unprojected_points = unproject_depth_map_to_point_map(depth, extrinsics_array, intrinsics_array)
    return {
        "world_points": canonicalize_to_first_camera(direct_points, extrinsics_array),
        "depth_unprojected": canonicalize_to_first_camera(unprojected_points, extrinsics_array),
        "camera_extrinsics": extrinsics_array,
        "camera_intrinsics": intrinsics_array,
    }


def prediction_path(output_root: Path, sample: Sample, representation: str) -> Path:
    """Map a dataset-relative MP4 path to the evaluator's NPZ convention."""

    return output_root / representation / sample.video_path.with_suffix(".npz")


def _archives_equal(first: Path, second: Path) -> bool:
    expected_fields = {"point_map", "camera_extrinsics", "camera_intrinsics"}
    try:
        with np.load(first, allow_pickle=False) as first_archive:
            with np.load(second, allow_pickle=False) as second_archive:
                if set(first_archive.files) != expected_fields:
                    return False
                if set(second_archive.files) != expected_fields:
                    return False
                return all(
                    np.array_equal(first_archive[name], second_archive[name])
                    for name in sorted(expected_fields)
                )
    except (OSError, ValueError):
        return False


def write_prediction_archive(
    path: Path,
    *,
    point_map: np.ndarray,
    camera_extrinsics: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> None:
    """Write one prediction archive atomically without replacing different data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            point_map=np.asarray(point_map, dtype=np.float16),
            camera_extrinsics=np.asarray(camera_extrinsics, dtype=np.float32),
            camera_intrinsics=np.asarray(camera_intrinsics, dtype=np.float32),
        )
        if path.exists():
            if not _archives_equal(path, temporary):
                raise ValueError(f"refusing to replace different VGGT prediction {path}")
            return
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B")
    parser.add_argument(
        "--checkpoint-revision",
        help="exact remote checkpoint revision; required unless --checkpoint is a file",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = select_partition(
        read_samples(args.dataset_root), args.partition_index, args.partition_count
    )
    if not samples:
        raise ValueError("selected VGGT partition contains no samples")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint_path) if checkpoint_path.is_file() else None
    if checkpoint_sha256 is not None and args.checkpoint_revision is not None:
        raise ValueError("--checkpoint-revision is not permitted for a local checkpoint file")
    integrity_bound = checkpoint_sha256 is not None or args.checkpoint_revision is not None
    if integrity_bound:
        checkpoint_identity(
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_revision=args.checkpoint_revision,
        )
    else:
        print(
            "warning: remote checkpoint is unpinned; outputs remain a legacy "
            "exploratory baseline and cannot be imported provider-neutrally",
            file=sys.stderr,
            flush=True,
        )
    model, image_loader, pose_converter, unprojector = load_vggt(
        args.vggt_root,
        args.checkpoint,
        args.checkpoint_revision,
        args.device,
    )
    vggt_revision = git_commit(args.vggt_root)
    loader_sha256 = file_sha256(Path(__file__).resolve())
    started = time.time()
    sample_records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        output_paths = {
            name: prediction_path(args.output_root, sample, name) for name in VGGT_REPRESENTATIONS
        }
        if args.resume and all(path.exists() for path in output_paths.values()):
            print(f"[{sample_index}/{len(samples)}] verify {sample.video_path}", flush=True)
        else:
            print(f"[{sample_index}/{len(samples)}] infer {sample.video_path}", flush=True)
            with tempfile.TemporaryDirectory(prefix="prob4d-vggt-") as temporary:
                frame_paths = extract_video_frames(
                    args.dataset_root / sample.video_path, Path(temporary)
                )
                arrays = infer_sample(
                    model=model,
                    load_and_preprocess_images=image_loader,
                    pose_encoding_to_extri_intri=pose_converter,
                    unproject_depth_map_to_point_map=unprojector,
                    frame_paths=frame_paths,
                    device=args.device,
                    preprocess_mode=args.preprocess_mode,
                )
            for representation, output_path in output_paths.items():
                write_prediction_archive(
                    output_path,
                    point_map=arrays[representation],
                    camera_extrinsics=arrays["camera_extrinsics"],
                    camera_intrinsics=arrays["camera_intrinsics"],
                )
        members = [
            describe_prediction_archive(
                output_paths[representation],
                representation=representation,
                relative_path=relative_member(
                    output_paths[representation],
                    root=args.output_root,
                    name=f"VGGT {representation} prediction path",
                ),
            )
            for representation in VGGT_REPRESENTATIONS
        ]
        sample_records.append(
            build_sample_record(
                sample_id=sample.video_path.as_posix(),
                input_video_path=args.dataset_root / sample.video_path,
                representations=members,
            )
        )

    elapsed = time.time() - started
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / f"run-part-{args.partition_index:02d}.json"
    if integrity_bound:
        metadata = build_run_record(
            vggt_commit=vggt_revision,
            loader_module_sha256=loader_sha256,
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_revision=args.checkpoint_revision,
            preprocess_mode=args.preprocess_mode,
            partition_index=args.partition_index,
            partition_count=args.partition_count,
            samples=sample_records,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            elapsed_seconds=elapsed,
        )
        save_vggt_run_metadata(metadata_path, metadata)
    else:
        legacy_metadata = {
            "method": "VGGT-1B",
            "official_repository": "https://github.com/facebookresearch/vggt",
            "vggt_commit": vggt_revision,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": None,
            "dataset_root": str(args.dataset_root.resolve()),
            "preprocess_mode": args.preprocess_mode,
            "partition_index": args.partition_index,
            "partition_count": args.partition_count,
            "completed": [sample.video_path.as_posix() for sample in samples],
            "elapsed_seconds": elapsed,
            "integrity_bound": False,
        }
        metadata_path.write_text(
            json.dumps(legacy_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
