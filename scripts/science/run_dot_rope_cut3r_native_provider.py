#!/usr/bin/env python3
"""Run the registered marker-free CUT3R experiment on DOT rope source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import traceback
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from prob4d.cut3r_runtime_contract import (
    require_compiled_cut3r_rope,
    validate_cut3r_runtime_receipt,
)
from prob4d.dot_rope_cut3r_study import (
    Sim3,
    clustered_bootstrap_sim3,
    content_id,
    covariance_closures,
    make_off_axis_probes,
    normalized_gaussian_score,
    parse_coordinate_text,
    robust_fit_sim3,
    sim3_to_vector,
)

PROTOCOL_SCHEMA = "prob4d.dot-rope-cut3r-native-provider-protocol"
REQUEST_SCHEMA = "prob4d.dot-rope-cut3r-native-provider-request"
PROVIDER_SCHEMA = "prob4d.dot-rope-cut3r-native-provider-bundle"
EVALUATION_SCHEMA = "prob4d.dot-rope-cut3r-native-provider-evaluation"
SCHEMA_VERSION = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--protocol-git-blob-sha", required=True)

    smoke = subparsers.add_parser("runtime-smoke")
    _add_common_execution_arguments(smoke)
    smoke.add_argument("--cut3r-checkout", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, required=True)
    smoke.add_argument("--runtime-receipt", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)

    predict = subparsers.add_parser("predict")
    _add_common_execution_arguments(predict)
    predict.add_argument("--dataset-root", type=Path, required=True)
    predict.add_argument("--cut3r-checkout", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--runtime-receipt", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_common_execution_arguments(evaluate)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--provider-bundle", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _add_common_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--prob4d-revision", required=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_identifier(value: object, *, label: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported DOT CUT3R protocol schema")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported DOT CUT3R protocol version")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _validate_identifier(protocol_id, label="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("DOT CUT3R protocol identity mismatch")
    if protocol.get("source_sequences") != ["R01", "R02", "R03"]:
        raise ValueError("source sequence boundary changed")
    if protocol.get("reserved_sequences") != "R04-R70":
        raise ValueError("reserved sequence boundary changed")
    if protocol.get("archive") != "R01-10.zip":
        raise ValueError("source archive changed")
    if protocol.get("camera") != "cam001":
        raise ValueError("camera selection changed")
    if protocol.get("frames") != list(range(1, 8)):
        raise ValueError("frozen frame roster changed")
    windows = protocol.get("windows")
    if windows != {
        "continuous": list(range(1, 8)),
        "window_a": list(range(1, 6)),
        "window_b": list(range(3, 8)),
    }:
        raise ValueError("frozen provider windows changed")
    return protocol


def validate_request(
    request_path: Path,
    protocol_path: Path,
    protocol_git_blob_sha: str,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    request = _load_json(request_path)
    expected_fields = {
        "schema",
        "schema_version",
        "request_id",
        "protocol_path",
        "protocol_git_blob_sha",
        "runtime_smoke_authorized",
        "normal_view_prediction_authorized",
        "marker_2d_evaluation_authorized",
        "marker_3d_evaluation_authorized",
        "source_sequences",
        "reserved_sequences",
        "target_payloads_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
        "claim_boundary",
    }
    if set(request) != expected_fields:
        raise ValueError("execution request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported execution request schema")
    if request["protocol_path"] != protocol_path.as_posix():
        raise ValueError("execution request protocol path changed")
    _validate_identifier(protocol_git_blob_sha, label="protocol Git blob", length=40)
    if request["protocol_git_blob_sha"] != protocol_git_blob_sha:
        raise ValueError("execution request does not bind the reviewed protocol blob")
    if request["source_sequences"] != protocol["source_sequences"]:
        raise ValueError("execution request source roster changed")
    if request["reserved_sequences"] != protocol["reserved_sequences"]:
        raise ValueError("execution request reserved boundary changed")
    for name in (
        "runtime_smoke_authorized",
        "normal_view_prediction_authorized",
        "marker_2d_evaluation_authorized",
        "marker_3d_evaluation_authorized",
    ):
        if request[name] is not True:
            raise ValueError(f"{name} must be explicitly authorized")
    for name in (
        "target_payloads_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if request[name] is not False:
            raise ValueError(f"{name} exceeds the source-only boundary")
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _validate_identifier(request_id, label="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("execution request identity mismatch")
    return {
        "request_id": request_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_git_blob_sha": protocol_git_blob_sha,
    }


def _require_execution_identity(request_id: str, revision: str) -> None:
    _validate_identifier(request_id, label="request_id", length=64)
    _validate_identifier(revision, label="Prob4D revision", length=40)


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\x00" in name:
        raise ValueError(f"unsafe ZIP member path: {name}")


def _prepend_cut3r_import_paths(checkout: Path) -> None:
    for candidate in (checkout, checkout / "src"):
        value = os.fspath(candidate)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


class NativeRopeCut3RRuntime:
    """Minimal CUT3R runtime guarded by the native-RoPE receipt."""

    def __init__(
        self,
        checkout: Path,
        checkpoint: Path,
        runtime_receipt: Path,
        *,
        device: str,
    ) -> None:
        self.checkout = checkout.expanduser().resolve(strict=True)
        self.checkpoint = checkpoint.expanduser().resolve(strict=True)
        if not self.checkpoint.is_file():
            raise ValueError("CUT3R checkpoint is not a regular file")
        expected_receipt = _load_json(runtime_receipt)
        validate_cut3r_runtime_receipt(expected_receipt)
        measured_receipt = require_compiled_cut3r_rope(self.checkout)
        if measured_receipt["artifact_id"] != expected_receipt["artifact_id"]:
            raise ValueError("CUT3R runtime differs from the frozen native-RoPE receipt")
        self.runtime_artifact_id = str(expected_receipt["artifact_id"])
        self.device = device
        _prepend_cut3r_import_paths(self.checkout)
        from add_ckpt_path import add_path_to_dust3r

        add_path_to_dust3r(os.fspath(self.checkpoint))
        import demo
        import torch
        from dust3r.inference import inference
        from dust3r.model import ARCroco3DStereo
        from dust3r.post_process import estimate_focal_knowing_depth
        from dust3r.utils.camera import pose_encoding_to_camera

        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("the registered provider requires an available CUDA device")
        self.demo = demo
        self.torch = torch
        self.inference = inference
        self.estimate_focal = estimate_focal_knowing_depth
        self.pose_encoding_to_camera = pose_encoding_to_camera
        self._reset_seed()
        self.model = ARCroco3DStereo.from_pretrained(os.fspath(self.checkpoint)).to(device)
        self.model.eval()

    def _reset_seed(self) -> None:
        random.seed(42)
        np.random.seed(42)
        self.torch.manual_seed(42)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(42)
        self.torch.backends.cudnn.benchmark = False

    def infer(self, frame_paths: Sequence[Path], *, image_size: int) -> dict[str, np.ndarray]:
        if not frame_paths:
            raise ValueError("CUT3R inference requires at least one frame")
        self._reset_seed()
        views = self.demo.prepare_input(
            img_paths=[os.fspath(path) for path in frame_paths],
            img_mask=[True] * len(frame_paths),
            size=image_size,
            revisit=1,
            update=True,
        )
        with self.torch.inference_mode():
            outputs, _ = self.inference(views, self.model, self.device, verbose=False)
        predictions = outputs["pred"]
        if len(predictions) != len(frame_paths):
            raise ValueError("CUT3R returned a noncanonical prediction count")
        points = self.torch.cat(
            [prediction["pts3d_in_self_view"].detach().cpu() for prediction in predictions],
            dim=0,
        )
        confidence = self.torch.cat(
            [prediction["conf_self"].detach().cpu() for prediction in predictions],
            dim=0,
        )
        poses = self.torch.cat(
            [
                self.pose_encoding_to_camera(prediction["camera_pose"].clone()).detach().cpu()
                for prediction in predictions
            ],
            dim=0,
        )
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("CUT3R direct point maps changed shape")
        if confidence.shape != points.shape[:-1] or poses.shape != (len(frame_paths), 4, 4):
            raise ValueError("CUT3R confidence or pose output changed shape")
        _, height, width, _ = points.shape
        principal = self.torch.tensor(
            [width // 2, height // 2],
            dtype=points.dtype,
            device=points.device,
        ).repeat(len(frame_paths), 1)
        focal = self.estimate_focal(points, principal, focal_mode="weiszfeld").detach().cpu()
        intrinsics = self.torch.eye(3).repeat(len(frame_paths), 1, 1)
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, 0, 2] = principal[:, 0].cpu()
        intrinsics[:, 1, 2] = principal[:, 1].cpu()
        result = {
            "points": points.numpy().astype(np.float32, copy=False),
            "confidence": confidence.numpy().astype(np.float32, copy=False),
            "poses": poses.numpy().astype(np.float64, copy=False),
            "intrinsics": intrinsics.numpy().astype(np.float64, copy=False),
        }
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("CUT3R emitted non-finite provider output")
        self.torch.cuda.empty_cache()
        return result


def _make_synthetic_frames(destination: Path, count: int) -> list[Path]:
    from PIL import Image, ImageDraw

    destination.mkdir(parents=True, exist_ok=False)
    width, height = 640, 512
    x = np.linspace(0.0, 1.0, width)[None, :]
    y = np.linspace(0.0, 1.0, height)[:, None]
    paths: list[Path] = []
    for index in range(count):
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.clip(255.0 * (0.25 + 0.65 * x), 0.0, 255.0)
        image[..., 1] = np.clip(255.0 * (0.2 + 0.65 * y), 0.0, 255.0)
        image[..., 2] = np.clip(255.0 * (0.55 + 0.2 * np.sin(8.0 * x + index)), 0.0, 255.0)
        frame = Image.fromarray(image, mode="RGB")
        draw = ImageDraw.Draw(frame)
        shift = 12 * index
        draw.line(
            [(80 + shift, 390), (220 + shift, 230), (390 + shift, 315), (555, 135 + shift)],
            fill=(245, 245, 245),
            width=12,
        )
        draw.ellipse((255 + shift, 190, 315 + shift, 250), fill=(50, 30, 210))
        path = destination / f"synthetic-{index:02d}.png"
        frame.save(path)
        paths.append(path)
    return paths


def runtime_smoke(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema": "prob4d.dot-rope-cut3r-native-runtime-smoke",
        "schema_version": SCHEMA_VERSION,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "prob4d_revision": args.prob4d_revision,
        "decision": "fail",
        "dataset_opened": False,
        "synthetic_prediction_executed": False,
    }
    try:
        runtime = NativeRopeCut3RRuntime(
            args.cut3r_checkout,
            args.checkpoint,
            args.runtime_receipt,
            device="cuda",
        )
        with tempfile.TemporaryDirectory(prefix="dot-cut3r-smoke-") as temporary:
            frame_paths = _make_synthetic_frames(
                Path(temporary) / "frames", count=3
            )
            prediction = runtime.infer(
                frame_paths,
                image_size=int(protocol["provider"]["image_size"]),
            )
        points = prediction["points"]
        confidence = prediction["confidence"]
        poses = prediction["poses"]
        determinants = [float(np.linalg.det(pose[:3, :3])) for pose in poses]
        if points.shape[0] != 3 or confidence.shape != points.shape[:-1]:
            raise ValueError("synthetic provider output has a noncanonical shape")
        if any(abs(value - 1.0) > 1.0e-3 for value in determinants):
            raise ValueError("CUT3R synthetic camera rotations are not proper")
        result.update(
            {
                "decision": "pass",
                "synthetic_prediction_executed": True,
                "runtime_artifact_id": runtime.runtime_artifact_id,
                "checkpoint_sha256": _sha256(args.checkpoint),
                "checkpoint_size_bytes": int(args.checkpoint.stat().st_size),
                "point_map_shape": list(points.shape),
                "confidence_quantiles": [
                    float(value) for value in np.quantile(confidence, [0.0, 0.25, 0.5, 0.75, 1.0])
                ],
                "camera_rotation_determinants": determinants,
                "gpu": str(runtime.torch.cuda.get_device_name(0)),
                "torch": str(runtime.torch.__version__),
                "torch_cuda": str(runtime.torch.version.cuda),
            }
        )
    except Exception as error:  # evidence must retain a bounded technical failure
        message = f"{type(error).__name__}: {' '.join(str(error).split())}"
        trace = traceback.format_exc()
        redactions = {
            os.fspath(args.cut3r_checkout): "<cut3r-checkout>",
            os.fspath(args.checkpoint): "<cut3r-checkpoint>",
            os.fspath(args.output_dir): "<output-dir>",
        }
        for raw, replacement in sorted(redactions.items(), key=lambda item: -len(item[0])):
            if raw:
                message = message.replace(raw, replacement)
                trace = trace.replace(raw, replacement)
        result["failure"] = message[:2000]
        result["traceback_tail"] = trace.splitlines()[-12:]
    unsigned = dict(result)
    result["artifact_id"] = content_id(unsigned)
    _write_json(output / "smoke-result.json", result)
    print(json.dumps({"decision": result["decision"], "artifact_id": result["artifact_id"]}))
    return 0 if result["decision"] == "pass" else 3


def _image_member(sequence: str, frame: int, camera: str) -> str:
    return f"{sequence}/images/normal_view/frame{frame:06d}_{camera}.jpg"


def _coordinate_member(sequence: str, dimension: int, frame: int, camera: str) -> str:
    return f"{sequence}/coordinates/{dimension}d/frame{frame:06d}_{camera}.txt"


def _save_provider_run(
    path: Path,
    prediction: Mapping[str, np.ndarray],
    *,
    frames: Sequence[int],
    original_sizes: Sequence[tuple[int, int]],
) -> None:
    np.savez_compressed(
        path,
        points=np.asarray(prediction["points"], dtype=np.float16),
        confidence=np.asarray(prediction["confidence"], dtype=np.float16),
        poses=np.asarray(prediction["poses"], dtype=np.float32),
        intrinsics=np.asarray(prediction["intrinsics"], dtype=np.float32),
        frames=np.asarray(frames, dtype=np.int32),
        original_sizes=np.asarray(original_sizes, dtype=np.int32),
    )


def predict(args: argparse.Namespace) -> int:
    from PIL import Image

    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    root = args.dataset_root.expanduser().resolve(strict=True)
    archive_path = (root / str(protocol["archive"])).resolve(strict=True)
    archive_path.relative_to(root)
    if not archive_path.is_file():
        raise ValueError("registered DOT archive is unavailable")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    runs_root = output / "runs"
    runs_root.mkdir()
    runtime = NativeRopeCut3RRuntime(
        args.cut3r_checkout,
        args.checkpoint,
        args.runtime_receipt,
        device="cuda",
    )
    input_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        expected_members = [
            _image_member(sequence, frame, str(protocol["camera"]))
            for sequence in protocol["source_sequences"]
            for frame in protocol["frames"]
        ]
        if any(member not in names for member in expected_members):
            missing = sorted(member for member in expected_members if member not in names)
            raise ValueError(f"registered normal-view members are missing: {missing[:3]}")
        with tempfile.TemporaryDirectory(prefix="dot-cut3r-images-") as temporary:
            temporary_root = Path(temporary)
            for sequence in protocol["source_sequences"]:
                sequence_root = temporary_root / sequence
                sequence_root.mkdir()
                paths_by_frame: dict[int, Path] = {}
                sizes_by_frame: dict[int, tuple[int, int]] = {}
                for frame in protocol["frames"]:
                    member = _image_member(sequence, int(frame), str(protocol["camera"]))
                    _safe_member(member)
                    raw = archive.read(member)
                    input_records.append(
                        {
                            "sequence": sequence,
                            "frame": int(frame),
                            "camera": protocol["camera"],
                            "member": member,
                            "byte_count": len(raw),
                            "sha256": _sha256_bytes(raw),
                        }
                    )
                    destination = sequence_root / f"frame{int(frame):06d}.jpg"
                    destination.write_bytes(raw)
                    with Image.open(destination) as image:
                        image.verify()
                    with Image.open(destination) as image:
                        sizes_by_frame[int(frame)] = (int(image.width), int(image.height))
                    paths_by_frame[int(frame)] = destination
                for run_name, frames in protocol["windows"].items():
                    frame_list = [int(frame) for frame in frames]
                    prediction = runtime.infer(
                        [paths_by_frame[frame] for frame in frame_list],
                        image_size=int(protocol["provider"]["image_size"]),
                    )
                    relative = Path("runs") / f"{sequence}-{run_name}.npz"
                    target = output / relative
                    _save_provider_run(
                        target,
                        prediction,
                        frames=frame_list,
                        original_sizes=[sizes_by_frame[frame] for frame in frame_list],
                    )
                    output_records.append(
                        {
                            "sequence": sequence,
                            "run": run_name,
                            "frames": frame_list,
                            "relative_path": relative.as_posix(),
                            "sha256": _sha256(target),
                            "byte_count": int(target.stat().st_size),
                            "point_map_shape": list(prediction["points"].shape),
                        }
                    )
    manifest: dict[str, Any] = {
        "schema": PROVIDER_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "prob4d_revision": args.prob4d_revision,
        "runtime_artifact_id": runtime.runtime_artifact_id,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_size_bytes": int(args.checkpoint.stat().st_size),
        "dataset": {
            "doi": protocol["dataset_doi"],
            "archive": protocol["archive"],
            "source_sequences": protocol["source_sequences"],
            "reserved_sequences": protocol["reserved_sequences"],
        },
        "inputs": input_records,
        "outputs": output_records,
        "information_boundary": {
            "normal_view_images_opened": True,
            "two_dimensional_markers_opened": False,
            "three_dimensional_markers_opened": False,
            "target_payloads_opened": False,
            "provider_residuals_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
        "decision": "sealed-provider-predictions",
    }
    manifest["provider_bundle_id"] = content_id(manifest)
    _write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "decision": manifest["decision"],
                "provider_bundle_id": manifest["provider_bundle_id"],
                "prediction_runs": len(output_records),
            }
        )
    )
    return 0


def _verify_provider_bundle(bundle: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(bundle / "manifest.json")
    if (
        manifest.get("schema") != PROVIDER_SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("provider bundle schema is unsupported")
    unsigned = dict(manifest)
    provider_bundle_id = unsigned.pop("provider_bundle_id", None)
    _validate_identifier(provider_bundle_id, label="provider_bundle_id", length=64)
    if content_id(unsigned) != provider_bundle_id:
        raise ValueError("provider bundle identity mismatch")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("provider bundle protocol identity changed")
    if manifest.get("decision") != "sealed-provider-predictions":
        raise ValueError("provider bundle was not sealed successfully")
    boundary = manifest.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("provider bundle has no information boundary")
    if boundary.get("two_dimensional_markers_opened") is not False:
        raise ValueError("provider stage opened 2-D marker payloads")
    if boundary.get("three_dimensional_markers_opened") is not False:
        raise ValueError("provider stage opened 3-D marker payloads")
    for record in manifest.get("outputs", []):
        if not isinstance(record, dict):
            raise ValueError("provider output record is malformed")
        relative = PurePosixPath(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("provider output path is unsafe")
        path = bundle.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(bundle.resolve(strict=True))
        if _sha256(path) != record["sha256"] or path.stat().st_size != record["byte_count"]:
            raise ValueError("provider output bytes differ from the sealed manifest")
    return manifest


def _load_run(bundle: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    relative = PurePosixPath(str(record["relative_path"]))
    path = bundle.joinpath(*relative.parts)
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    required = {"points", "confidence", "poses", "intrinsics", "frames", "original_sizes"}
    if set(result) != required:
        raise ValueError("provider run fields changed")
    return result


def _sample_markers(
    run: Mapping[str, np.ndarray],
    frame: int,
    coordinates_2d: np.ndarray,
    coordinates_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(run["frames"], dtype=np.int64)
    matches = np.flatnonzero(frames == frame)
    if matches.size != 1:
        raise ValueError("provider run does not contain the requested frame exactly once")
    index = int(matches[0])
    points = np.asarray(run["points"][index], dtype=np.float64)
    confidence = np.asarray(run["confidence"][index], dtype=np.float64)
    width, height = (int(value) for value in run["original_sizes"][index])
    count = min(coordinates_2d.shape[0], coordinates_3d.shape[0])
    image_coordinates = np.asarray(coordinates_2d[:count], dtype=np.float64).copy()
    if width <= 1 or height <= 1:
        raise ValueError("provider input image dimensions are invalid")
    image_coordinates[:, 0] *= (points.shape[1] - 1.0) / (width - 1.0)
    image_coordinates[:, 1] *= (points.shape[0] - 1.0) / (height - 1.0)
    from prob4d.dot_rope_cut3r_study import bilinear_sample

    sampled_points, valid_points = bilinear_sample(points, image_coordinates)
    sampled_confidence, valid_confidence = bilinear_sample(
        confidence[..., None],
        image_coordinates,
    )
    valid = (
        valid_points
        & valid_confidence
        & np.isfinite(coordinates_3d[:count]).all(axis=1)
        & (sampled_confidence[:, 0] > 0.0)
    )
    if np.count_nonzero(valid) < 6:
        raise ValueError("fewer than six valid marker samples remain")
    marker_indices = np.flatnonzero(valid)
    return (
        sampled_points[valid],
        np.asarray(coordinates_3d[:count], dtype=np.float64)[valid],
        marker_indices,
    )


def _collect_pair(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    frame_payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for frame in frames:
        points_2d, points_3d = frame_payloads[int(frame)]
        first_points, _, first_indices = _sample_markers(first, int(frame), points_2d, points_3d)
        second_points, _, second_indices = _sample_markers(second, int(frame), points_2d, points_3d)
        common, first_positions, second_positions = np.intersect1d(
            first_indices,
            second_indices,
            assume_unique=True,
            return_indices=True,
        )
        if common.size < 6:
            raise ValueError("fewer than six common provider markers remain")
        sources.append(second_points[second_positions])
        targets.append(first_points[first_positions])
        groups.append(np.full(common.size, int(frame), dtype=np.int64))
    return np.concatenate(sources), np.concatenate(targets), np.concatenate(groups)


def _collect_provider_truth(
    run: Mapping[str, np.ndarray],
    frame_payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    providers: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for frame in frames:
        points_2d, points_3d = frame_payloads[int(frame)]
        provider, truth, _ = _sample_markers(run, int(frame), points_2d, points_3d)
        providers.append(provider)
        truths.append(truth)
    return np.concatenate(providers), np.concatenate(truths)


def _rmse(transform: Sim3, source: np.ndarray, target: np.ndarray) -> float:
    residual = transform.apply(source) - target
    return float(math.sqrt(float(np.mean(np.sum(residual * residual, axis=1)))))


def _sequence_evaluation(
    sequence: str,
    runs: Mapping[str, Mapping[str, np.ndarray]],
    frame_payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overlap_frames = [int(value) for value in protocol["evaluation"]["overlap_frames"]]
    fit_a_frames = [int(value) for value in protocol["evaluation"]["metric_fit_a_frames"]]
    fit_b_frames = [int(value) for value in protocol["evaluation"]["metric_fit_b_frames"]]
    score_frames = [int(value) for value in protocol["evaluation"]["score_frames"]]

    source, target, groups = _collect_pair(
        runs["window_a"],
        runs["window_b"],
        frame_payloads,
        overlap_frames,
    )
    estimated_relative, overlap_residuals = robust_fit_sim3(source, target)
    covariance, bootstrap = clustered_bootstrap_sim3(
        source,
        target,
        groups,
        replicates=int(protocol["uncertainty"]["bootstrap_replicates"]),
        seed=int(protocol["uncertainty"]["bootstrap_seed"]) + int(sequence[1:]),
    )

    a_source, a_truth = _collect_provider_truth(runs["window_a"], frame_payloads, fit_a_frames)
    b_source, b_truth = _collect_provider_truth(runs["window_b"], frame_payloads, fit_b_frames)
    continuous_source, continuous_truth = _collect_provider_truth(
        runs["continuous"], frame_payloads, fit_a_frames
    )
    a_to_truth, _ = robust_fit_sim3(a_source, a_truth)
    b_to_truth, _ = robust_fit_sim3(b_source, b_truth)
    continuous_to_truth, _ = robust_fit_sim3(continuous_source, continuous_truth)
    true_relative = a_to_truth.inverse().compose(b_to_truth)

    score_b, score_truth = _collect_provider_truth(runs["window_b"], frame_payloads, score_frames)
    score_continuous, score_continuous_truth = _collect_provider_truth(
        runs["continuous"], frame_payloads, score_frames
    )
    identity_stitch = _rmse(a_to_truth, score_b, score_truth)
    estimated_stitch = _rmse(
        a_to_truth.compose(estimated_relative),
        score_b,
        score_truth,
    )
    continuous = _rmse(continuous_to_truth, score_continuous, score_continuous_truth)
    oracle_window = _rmse(b_to_truth, score_b, score_truth)
    truth_cloud = np.concatenate([payload[1] for payload in frame_payloads.values()])
    truth_centered = truth_cloud - np.mean(truth_cloud, axis=0)
    truth_span = float(np.max(np.linalg.norm(truth_centered, axis=1))) * 2.0
    if truth_span <= 0.0:
        raise ValueError("ground-truth rope span is degenerate")

    probes, provider_span = make_off_axis_probes(
        source,
        count=int(protocol["uncertainty"]["probe_count"]),
    )
    center = sim3_to_vector(estimated_relative)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    minimum_steps = np.asarray(
        [
            1.0e-4 * provider_span,
            1.0e-4 * provider_span,
            1.0e-4 * provider_span,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-4,
        ]
    )
    steps = np.maximum(0.1 * standard_deviation, minimum_steps)
    closures = covariance_closures(
        center,
        covariance,
        probes,
        bootstrap,
        scalar_inflation=float(protocol["uncertainty"]["scalar_inflation"]),
        finite_difference_steps=steps,
        orbit_nodes=int(protocol["uncertainty"]["orbit_nodes"]),
        tensor_gh_order=int(protocol["uncertainty"]["tensor_gh_order"]),
    )
    fixed_mean = estimated_relative.apply(probes).reshape(-1)
    truth_query = true_relative.apply(probes).reshape(-1)
    method_rows: list[dict[str, Any]] = []
    for method, method_covariance in closures.items():
        score = normalized_gaussian_score(
            truth_query,
            fixed_mean,
            method_covariance,
            span=provider_span,
            observation_noise_fraction=float(protocol["uncertainty"]["observation_noise_fraction"]),
        )
        method_rows.append({"sequence": sequence, "method": method, **score})

    sequence_result = {
        "sequence": sequence,
        "overlap_correspondence_count": int(source.shape[0]),
        "overlap_rmse_provider_units": float(
            math.sqrt(float(np.mean(overlap_residuals * overlap_residuals)))
        ),
        "estimated_relative_scale": estimated_relative.scale,
        "true_relative_scale": true_relative.scale,
        "point_metrics": {
            "identity_stitch_rmse": identity_stitch,
            "estimated_stitch_rmse": estimated_stitch,
            "continuous_rmse": continuous,
            "oracle_window_rmse": oracle_window,
            "truth_span": truth_span,
            "identity_stitch_rmse_fraction_of_span": identity_stitch / truth_span,
            "estimated_stitch_rmse_fraction_of_span": estimated_stitch / truth_span,
            "continuous_rmse_fraction_of_span": continuous / truth_span,
            "oracle_window_rmse_fraction_of_span": oracle_window / truth_span,
        },
        "query": {
            "dimension": int(truth_query.size),
            "provider_span": provider_span,
            "fixed_mean_error_fraction_of_span": float(
                np.linalg.norm(truth_query - fixed_mean)
                / (provider_span * math.sqrt(truth_query.size))
            ),
            "bootstrap_transform_count": len(bootstrap),
            "parameter_covariance_eigenvalues": [
                float(value) for value in np.linalg.eigvalsh(covariance)
            ],
        },
    }
    return sequence_result, method_rows


def evaluate(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    _require_execution_identity(args.request_id, args.prob4d_revision)
    bundle = args.provider_bundle.expanduser().resolve(strict=True)
    manifest = _verify_provider_bundle(bundle, protocol)
    if manifest["request_id"] != args.request_id:
        raise ValueError("provider bundle request identity changed")
    if manifest["prob4d_revision"] != args.prob4d_revision:
        raise ValueError("provider bundle revision changed")
    root = args.dataset_root.expanduser().resolve(strict=True)
    archive_path = (root / str(protocol["archive"])).resolve(strict=True)
    archive_path.relative_to(root)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    records_by_sequence: dict[str, dict[str, Mapping[str, Any]]] = {
        sequence: {} for sequence in protocol["source_sequences"]
    }
    for record in manifest["outputs"]:
        records_by_sequence[str(record["sequence"])][str(record["run"])] = record
    sequence_results: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    opened_members: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for sequence in protocol["source_sequences"]:
            frame_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for frame in protocol["frames"]:
                member_2d = _coordinate_member(
                    sequence,
                    2,
                    int(frame),
                    str(protocol["camera"]),
                )
                member_3d = _coordinate_member(
                    sequence,
                    3,
                    int(frame),
                    str(protocol["camera"]),
                )
                if member_2d not in names or member_3d not in names:
                    raise ValueError("registered DOT marker payload is missing")
                raw_2d = archive.read(member_2d)
                raw_3d = archive.read(member_3d)
                opened_members.extend(
                    [
                        {
                            "sequence": sequence,
                            "frame": int(frame),
                            "kind": "2d",
                            "member": member_2d,
                            "byte_count": len(raw_2d),
                            "sha256": _sha256_bytes(raw_2d),
                        },
                        {
                            "sequence": sequence,
                            "frame": int(frame),
                            "kind": "3d",
                            "member": member_3d,
                            "byte_count": len(raw_3d),
                            "sha256": _sha256_bytes(raw_3d),
                        },
                    ]
                )
                points_2d = parse_coordinate_text(raw_2d.decode("utf-8"), 2)
                points_3d = parse_coordinate_text(raw_3d.decode("utf-8"), 3)
                frame_payloads[int(frame)] = (points_2d, points_3d)
            runs = {
                run_name: _load_run(bundle, records_by_sequence[sequence][run_name])
                for run_name in ("continuous", "window_a", "window_b")
            }
            sequence_result, rows = _sequence_evaluation(
                sequence,
                runs,
                frame_payloads,
                protocol,
            )
            sequence_results.append(sequence_result)
            method_rows.extend(rows)

    aggregate_methods: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in method_rows}):
        rows = [row for row in method_rows if row["method"] == method]
        aggregate_methods.append(
            {
                "method": method,
                "sequence_count": len(rows),
                "mean_normalized_nll_per_dimension": float(
                    np.mean([row["normalized_nll_per_dimension"] for row in rows])
                ),
                "mean_mahalanobis": float(np.mean([row["mahalanobis"] for row in rows])),
                "covered_95_count": int(sum(bool(row["covered_95"]) for row in rows)),
                "mean_predictive_sd_fraction_of_span": float(
                    np.mean([row["mean_predictive_sd_fraction_of_span"] for row in rows])
                ),
            }
        )
    aggregate_methods.sort(key=lambda row: row["mean_normalized_nll_per_dimension"])
    result: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": args.request_id,
        "protocol_id": protocol["protocol_id"],
        "prob4d_revision": args.prob4d_revision,
        "provider_bundle_id": manifest["provider_bundle_id"],
        "runtime_artifact_id": manifest["runtime_artifact_id"],
        "decision": "complete-source-evaluation",
        "sequences": sequence_results,
        "method_rows": method_rows,
        "aggregate_methods": aggregate_methods,
        "opened_marker_members": opened_members,
        "information_boundary": {
            "normal_view_images_opened_by_sealed_provider_stage": True,
            "two_dimensional_markers_opened_after_prediction_seal": True,
            "three_dimensional_markers_opened_after_prediction_seal": True,
            "opened_sequences": protocol["source_sequences"],
            "reserved_sequences": protocol["reserved_sequences"],
            "target_payloads_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
            "means_held_fixed_across_uncertainty_methods": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["evaluation_id"] = content_id(result)
    _write_json(output / "result.json", result)
    with (output / "method-summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(aggregate_methods[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_methods)
    with (output / "sequence-methods.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(method_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_rows)
    lines = [
        "# DOT rope marker-free CUT3R source result",
        "",
        f"Evaluation ID: `{result['evaluation_id']}`",
        "",
        "## Uncertainty methods",
        "",
        "| Method | Mean normalized NLL/dim | 95% covered | Mean SD/span |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_methods:
        lines.append(
            "| {method} | {mean_normalized_nll_per_dimension:.6f} | "
            "{covered_95_count}/{sequence_count} | "
            "{mean_predictive_sd_fraction_of_span:.6f} |".format(**row)
        )
    lines.extend(["", "## Reconstruction/stitching", ""])
    for sequence in sequence_results:
        metrics = sequence["point_metrics"]
        lines.append(
            f"- **{sequence['sequence']}**: "
            f"continuous={metrics['continuous_rmse_fraction_of_span']:.6f}, "
            f"identity stitch={metrics['identity_stitch_rmse_fraction_of_span']:.6f}, "
            f"estimated Sim(3) stitch={metrics['estimated_stitch_rmse_fraction_of_span']:.6f}, "
            f"oracle window={metrics['oracle_window_rmse_fraction_of_span']:.6f} (all / span)."
        )
    lines.extend(
        [
            "",
            "Source-development evidence only. R04-R70 remained unopened; no "
            "BayesianPhysTwin or Causal4D outcome was executed.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "evaluation_id": result["evaluation_id"],
                "best_method": aggregate_methods[0]["method"],
            }
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-request":
        result = validate_request(
            args.request,
            args.protocol,
            args.protocol_git_blob_sha,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "runtime-smoke":
        return runtime_smoke(args)
    if args.command == "predict":
        return predict(args)
    if args.command == "evaluate":
        return evaluate(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
