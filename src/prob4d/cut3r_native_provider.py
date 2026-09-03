"""Transactional native CUT3R production using only an explicitly bounded prefix."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._strict_json import require_exact_integer, require_json_number
from .cut3r_direct_provider_adapter import import_cut3r_direct_prediction_manifest
from .cut3r_native_runtime import file_sha256, ordinary_path
from .prediction_provider_manifest import verify_prediction_provider_manifest


class NativeRuntime(Protocol):
    identity: dict[str, Any]

    def reset(self) -> None: ...
    def step(self, path: Path) -> dict[str, np.ndarray]: ...


def content_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _stage_inputs(
    source: Path,
    destination: Path,
    *,
    frame_start: int,
    frame_stop: int,
    video: bool,
    extension: str,
) -> tuple[list[Path], dict[str, Any]]:
    source = ordinary_path(source)
    destination.mkdir()
    count = frame_stop - frame_start
    paths = [destination / f"{i:06d}{extension}" for i in range(count)]
    records: list[dict[str, Any]] = []
    if video:
        if not source.is_file():
            raise ValueError("video input must be a file")
        digest = file_sha256(source)
        size = source.stat().st_size
        # Stop emission at the exclusive bound. A video codec can internally
        # decode lookahead/reference frames, but none are supplied to the model.
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=between(n\\,{frame_start}\\,{frame_stop - 1})",
            "-vsync",
            "0",
            "-frames:v",
            str(count),
            "-start_number",
            "0",
            str(destination / "%06d.png"),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"FFmpeg prefix extraction failed: {detail}") from error
        if file_sha256(source) != digest or source.stat().st_size != size:
            raise ValueError("video changed during prefix extraction")
        paths = [destination / f"{i:06d}.png" for i in range(count)]
        input_identity = {"kind": "video", "sha256": digest, "byte_count": size}
    else:
        if not source.is_dir():
            raise ValueError("frame input must be a directory")
        # Do not glob, read, or inspect the suffix; names are exact frame IDs.
        for index, destination_path in enumerate(paths):
            original = ordinary_path(source / f"{index + frame_start:06d}{extension}")
            if not original.is_file():
                raise ValueError("frame input must be an ordinary file")
            digest = file_sha256(original)
            shutil.copyfile(original, destination_path)
            if file_sha256(destination_path) != digest or file_sha256(original) != digest:
                raise ValueError("frame changed during prefix staging")
            records.append(
                {
                    "frame_id": index + frame_start,
                    "sha256": digest,
                    "byte_count": destination_path.stat().st_size,
                }
            )
        input_identity = {
            "kind": "image-prefix",
            "sha256": content_id(records),
            "byte_count": sum(row["byte_count"] for row in records),
        }
    if any(not path.is_file() for path in paths):
        raise ValueError("input does not contain the complete requested prefix")
    return paths, {
        **input_identity,
        "frame_start": frame_start,
        "frame_stop_exclusive": frame_stop,
        "frames": records
        or [
            {
                "frame_id": i + frame_start,
                "sha256": file_sha256(path),
                "byte_count": path.stat().st_size,
            }
            for i, path in enumerate(paths)
        ],
    }


def _save_frame(root: Path, index: int, frame: dict[str, np.ndarray]) -> tuple[int, int]:
    if set(frame) != {"points", "confidence", "pose", "intrinsics"}:
        raise ValueError("native CUT3R frame fields changed")
    points, confidence = frame["points"], frame["confidence"]
    pose, intrinsics = frame["pose"], frame["intrinsics"]
    if points.ndim != 3 or points.shape[-1] != 3 or confidence.shape != points.shape[:2]:
        raise ValueError("native CUT3R point/confidence shape mismatch")
    if pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("native CUT3R camera shape mismatch")
    if not all(np.isfinite(value).all() for value in frame.values()):
        raise ValueError("native CUT3R frame contains non-finite values")
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(
            pose[:3, :3].T @ pose[:3, :3],
            np.eye(3),
            atol=1e-4,
        )
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("native CUT3R camera pose is not rigid")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("native CUT3R focal length is invalid")
    stem = f"{index:06d}"
    np.save(root / "points" / f"{stem}.npy", points, allow_pickle=False)
    np.save(root / "depth" / f"{stem}.npy", points[..., 2], allow_pickle=False)
    np.save(root / "conf" / f"{stem}.npy", confidence, allow_pickle=False)
    np.savez(root / "camera" / f"{stem}.npz", pose=pose, intrinsics=intrinsics)
    return int(points.shape[0]), int(points.shape[1])


def run_cut3r_native(
    source: Path,
    output: Path,
    *,
    runtime_factory: Callable[[], NativeRuntime],
    sequence_id: str,
    frame_start: int,
    frame_stop: int,
    video: bool = False,
    extension: str = ".png",
    confidence_threshold: float = 1.5,
) -> dict[str, Any]:
    """Publish a verified direct provider only after the entire prefix succeeds.

    The output root is write-once. A failure retains a small receipt and partial
    generated outputs, never a successfully published prediction directory.
    """
    start = require_exact_integer(frame_start, name="frame_start", minimum=0)
    stop = require_exact_integer(frame_stop, name="frame_stop", minimum=start + 1)
    if stop - start > 4096:
        raise ValueError("native CUT3R prefix exceeds the 4096-frame limit")
    if extension not in (".png", ".jpg", ".jpeg"):
        raise ValueError("frame extension must be .png, .jpg or .jpeg")
    threshold = require_json_number(confidence_threshold, name="confidence_threshold")
    if threshold < 0 or type(sequence_id) is not str or not sequence_id:
        raise ValueError("invalid confidence threshold or sequence ID")
    parent = ordinary_path(output.parent)
    output = parent / output.name
    output.mkdir(exist_ok=False)
    staging = output / "staging"
    staging.mkdir()
    receipt: dict[str, Any] = {
        "schema": "prob4d.cut3r-native-production-v1",
        "status": "failed",
        "stage": "runtime-initialization",
        "frames_completed": 0,
        "prediction_published": False,
        "uses_truth": False,
        "future_frames_supplied_to_model": False,
        "metric_scale_claimed": False,
        "calibrated_covariance_claimed": False,
        "scene_flow_claimed": False,
        "implementation_sha256s": {
            name: file_sha256(Path(__file__).parent / name)
            for name in (
                "cut3r_native_provider.py",
                "cut3r_native_runtime.py",
                "_cut3r_native_cli.py",
            )
        },
    }
    decoded = staging / "decoded"
    try:
        runtime = runtime_factory()
        receipt["runtime"] = runtime.identity
        receipt["stage"] = "prefix-staging"
        paths, inputs = _stage_inputs(
            source,
            decoded,
            frame_start=start,
            frame_stop=stop,
            video=video,
            extension=extension,
        )
        receipt["input"] = inputs
        direct = output / "direct"
        direct.mkdir()
        for name in ("points", "depth", "conf", "camera"):
            (direct / name).mkdir()
        runtime.reset()
        shape = None
        receipt["stage"] = "provider-inference"
        for index, path in enumerate(paths):
            frame = runtime.step(path)
            current_shape = _save_frame(direct, index, frame)
            if shape is not None and current_shape != shape:
                raise ValueError("input aspect ratio changed within one dense CUT3R window")
            shape = current_shape
            receipt["frames_completed"] = index + 1
        receipt["stage"] = "provider-publication"
        bundle = staging / "prediction"
        bundle.mkdir()
        manifest = import_cut3r_direct_prediction_manifest(
            direct,
            bundle / "provider.json",
            sequence_id=sequence_id,
            cut3r_revision=runtime.identity["cut3r_revision"],
            checkpoint_sha256=runtime.identity["checkpoint_sha256"],
            input_video_sha256=inputs["sha256"],
            input_video_byte_count=inputs["byte_count"],
            frame_start=start,
            confidence_threshold=threshold,
        )
        verify_prediction_provider_manifest(bundle / "provider.json")
        receipt["provider_run_id"] = manifest.provider_run_id
        receipt["manifest_sha256"] = file_sha256(bundle / "provider.json")
        receipt["direct_members"] = [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": file_sha256(path),
                "byte_count": path.stat().st_size,
            }
            for path in sorted(direct.rglob("*"))
            if path.is_file()
        ]
        bundle.rename(output / "prediction")
        receipt.update(status="success", stage="complete", prediction_published=True)
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)[:2000]
        raise
    finally:
        # This is our newly created staging tree, never an input or historical attempt.
        if decoded.exists():
            shutil.rmtree(decoded)
        receipt["artifact_id"] = content_id(receipt)
        _write_json(output / "run.json", receipt)
    return receipt
