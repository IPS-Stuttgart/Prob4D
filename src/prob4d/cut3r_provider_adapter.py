"""Convert official CUT3R recurrent-online outputs into neutral predictions.

The adapter consumes the deterministic file layout written by CUT3R's recurrent
``demo.py`` path. It never imports or executes CUT3R, Torch, OpenCV, or model
checkpoints. Exact code, checkpoint, input-video, and generated-source identities
are bound into the resulting provider manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypedDict, cast

import numpy as np

from ._atomic_file import publish_temporary_file
from ._strict_json import require_exact_integer, require_revision, require_sha256
from .data import DENSE_STORAGE_DTYPES, DenseStorageDType, PredictionWindow
from .prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)

CUT3R_OFFICIAL_REPOSITORY: Final = "CUT3R/CUT3R"
CUT3R_ONLINE_SOURCE_LAYOUT: Final = "cut3r-demo-recurrent-depth-conf-camera-v1"
_ADAPTER_DOMAIN: Final = "prob4d.cut3r-online-provider-adapter.v1"
_MODEL_SET_DOMAIN: Final = "prob4d.cut3r-online-model-set.v1"
_SOURCE_BUNDLE_DOMAIN: Final = "prob4d.cut3r-online-source-bundle.v1"
_RUN_DOMAIN: Final = "prob4d.cut3r-online-run.v1"
_MEMBER_DOMAIN: Final = "prob4d.cut3r-online-member.v1"


class _SourceMemberDescriptor(TypedDict):
    """Content descriptor for one exact CUT3R output member."""

    path: str
    sha256: str
    byte_count: int


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_id(domain: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read CUT3R source member {path.name!r}") from error
    return digest.hexdigest()


def _file_descriptor(path: Path, *, root: Path) -> _SourceMemberDescriptor:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"CUT3R source member {path.name!r} must be an ordinary file")
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("CUT3R source member escapes the declared output root") from error
    return {
        "path": PurePosixPath(*relative.parts).as_posix(),
        "sha256": _file_sha256(path),
        "byte_count": int(path.stat().st_size),
    }


def _validated_source_directory(root: Path, name: str, suffix: str) -> dict[int, Path]:
    directory = root / name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"CUT3R output requires an ordinary {name!r} directory")
    indexed: dict[int, Path] = {}
    for member in sorted(directory.iterdir()):
        if member.is_symlink() or not member.is_file():
            raise ValueError(f"CUT3R {name!r} directory contains a non-regular member")
        if member.suffix != suffix or len(member.stem) != 6 or not member.stem.isdigit():
            raise ValueError(f"CUT3R {name!r} members must use six-digit {suffix} filenames")
        index = int(member.stem)
        if index in indexed:
            raise ValueError(f"duplicate CUT3R frame index {index} in {name!r}")
        indexed[index] = member
    if not indexed:
        raise ValueError(f"CUT3R {name!r} directory is empty")
    if set(indexed) != set(range(len(indexed))):
        raise ValueError(f"CUT3R {name!r} frame indices must be contiguous from zero")
    return indexed


def _validated_source_members(
    root: Path,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("CUT3R output root must be an ordinary directory")
    depth = _validated_source_directory(root, "depth", ".npy")
    confidence = _validated_source_directory(root, "conf", ".npy")
    camera = _validated_source_directory(root, "camera", ".npz")
    if set(depth) != set(confidence) or set(depth) != set(camera):
        raise ValueError("CUT3R depth, confidence, and camera frame sets disagree")
    return depth, confidence, camera


def _load_npy(
    path: Path,
    *,
    label: str,
) -> tuple[np.ndarray, _SourceMemberDescriptor]:
    root = path.parents[1]
    before = _file_descriptor(path, root=root)
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load CUT3R {label} member {path.name!r}") from error
    after = _file_descriptor(path, root=root)
    if before != after:
        raise ValueError(f"CUT3R {label} member changed while it was being read")
    return np.asarray(value), before


def _load_camera(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, _SourceMemberDescriptor]:
    root = path.parents[1]
    before = _file_descriptor(path, root=root)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"pose", "intrinsics"}:
                raise ValueError("camera archive fields must be exactly pose and intrinsics")
            pose = np.asarray(archive["pose"], dtype=np.float64)
            intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load CUT3R camera member {path.name!r}: {error}") from error
    after = _file_descriptor(path, root=root)
    if before != after:
        raise ValueError("CUT3R camera member changed while it was being read")
    return pose, intrinsics, before


def _validate_camera(pose: np.ndarray, intrinsics: np.ndarray) -> None:
    if pose.shape != (4, 4):
        raise ValueError("CUT3R camera pose must have shape (4, 4)")
    if intrinsics.shape != (3, 3):
        raise ValueError("CUT3R camera intrinsics must have shape (3, 3)")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("CUT3R camera pose and intrinsics must be finite")
    if not np.allclose(pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("CUT3R camera pose must be a homogeneous rigid transform")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError("CUT3R camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("CUT3R camera rotation must be proper")
    if not np.allclose(intrinsics[2], np.asarray([0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("CUT3R camera intrinsics must use the standard final row")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("CUT3R focal lengths must be strictly positive")


def _unproject_world_points(
    depth: np.ndarray,
    confidence: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if depth.ndim != 2:
        raise ValueError("CUT3R depth members must have shape (H, W)")
    if confidence.shape != depth.shape:
        raise ValueError("CUT3R confidence must match the depth grid")
    if depth.dtype.kind not in {"f", "i", "u"} or confidence.dtype.kind not in {
        "f",
        "i",
        "u",
    }:
        raise ValueError("CUT3R depth and confidence must be real numeric arrays")
    _validate_camera(pose, intrinsics)

    depth64 = np.asarray(depth, dtype=np.float64)
    confidence64 = np.asarray(confidence, dtype=np.float64)
    valid = (
        np.isfinite(depth64)
        & np.isfinite(confidence64)
        & (depth64 > 0.0)
        & (confidence64 >= confidence_threshold)
    )

    height, width = depth64.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    homogeneous_pixels = np.stack(
        (columns, rows, np.ones((height, width), dtype=np.float64)),
        axis=-1,
    )
    camera_rays = np.linalg.solve(
        intrinsics,
        homogeneous_pixels.reshape(-1, 3).T,
    ).T.reshape(height, width, 3)
    camera_points = camera_rays * depth64[..., None]
    world = np.einsum("ij,hwj->hwi", pose[:3, :3], camera_points) + pose[:3, 3]
    valid &= np.all(np.isfinite(world), axis=-1)
    world[~valid] = 0.0
    return world, valid


def _verify_source_descriptors(
    root: Path,
    descriptors: Sequence[_SourceMemberDescriptor],
) -> None:
    for descriptor in descriptors:
        relative = PurePosixPath(descriptor["path"])
        candidate = root.joinpath(*relative.parts)
        if _file_descriptor(candidate, root=root) != descriptor:
            raise ValueError("CUT3R source tree changed after canonical loading")


def _canonical_window(
    root: Path,
    *,
    frame_start: int,
    window_id: str,
    confidence_threshold: float,
    storage_dtype: DenseStorageDType,
) -> tuple[PredictionWindow, tuple[_SourceMemberDescriptor, ...]]:
    depth_members, confidence_members, camera_members = _validated_source_members(root)
    points: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    descriptors: list[_SourceMemberDescriptor] = []
    shape: tuple[int, int] | None = None
    for index in range(len(depth_members)):
        depth, depth_descriptor = _load_npy(depth_members[index], label="depth")
        confidence, confidence_descriptor = _load_npy(
            confidence_members[index],
            label="confidence",
        )
        pose, intrinsics, camera_descriptor = _load_camera(camera_members[index])
        world, valid = _unproject_world_points(
            depth,
            confidence,
            pose,
            intrinsics,
            confidence_threshold=confidence_threshold,
        )
        if shape is None:
            shape = valid.shape
        elif valid.shape != shape:
            raise ValueError("CUT3R frames must share one spatial prediction grid")
        points.append(world)
        masks.append(valid)
        descriptors.extend((depth_descriptor, confidence_descriptor, camera_descriptor))

    valid_stack = np.stack(masks, axis=0)
    if not np.any(valid_stack):
        raise ValueError("CUT3R output contains no point above the frozen support threshold")
    dense_dtype = np.float32 if storage_dtype == "float32" else np.float64
    point_stack = np.asarray(np.stack(points, axis=0), dtype=dense_dtype)
    frame_count = point_stack.shape[0]
    return (
        PredictionWindow(
            window_id=window_id,
            frame_indices=np.arange(
                frame_start,
                frame_start + frame_count,
                dtype=np.int64,
            ),
            point_map=point_stack,
            valid_mask=valid_stack,
            dense_storage_dtype=storage_dtype,
        ),
        tuple(descriptors),
    )


def _windows_equal(first: PredictionWindow, second: PredictionWindow) -> bool:
    return (
        first.window_id == second.window_id
        and first.dense_storage_dtype == second.dense_storage_dtype
        and np.array_equal(first.frame_indices, second.frame_indices)
        and np.array_equal(first.point_map, second.point_map)
        and np.array_equal(first.valid_mask, second.valid_mask)
        and first.scene_flow is None
        and second.scene_flow is None
        and first.ray_directions is None
        and second.ray_directions is None
    )


def _write_window_atomically(path: Path, window: PredictionWindow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        window.to_npz(temporary)
        try:
            publish_temporary_file(temporary, path, overwrite=False)
        except FileExistsError:
            existing = PredictionWindow.from_npz(
                path,
                dense_storage_dtype=window.dense_storage_dtype,
            )
            if not _windows_equal(existing, window):
                raise ValueError(
                    f"refusing to replace different canonical CUT3R payload {path.name!r}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _relative_member(path: Path, *, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            "canonical CUT3R payload must lie inside the manifest directory"
        ) from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("canonical CUT3R payload path is not confined")
    return PurePosixPath(*relative.parts).as_posix()


def import_cut3r_online_prediction_manifest(
    source_root: str | Path,
    output_manifest_path: str | Path,
    *,
    sequence_id: str,
    cut3r_revision: str,
    checkpoint_sha256: str,
    input_video_sha256: str,
    input_video_byte_count: int,
    frame_start: int = 0,
    view_id: str = "camera-0",
    window_id: str = "cut3r-online",
    confidence_threshold: float = 1.5,
    storage_dtype: DenseStorageDType = "float32",
) -> PredictionProviderManifestV1:
    """Import one exact recurrent-online CUT3R output tree.

    The caller declares that the source tree was generated with CUT3R's recurrent
    online path, one forward pass, no revisit, and no global alignment. The
    adapter binds that declaration and per-frame prefix lineage but cannot prove
    how an external process was launched.
    """

    revision = require_revision(cut3r_revision, name="CUT3R revision")
    checkpoint = require_sha256(checkpoint_sha256, name="CUT3R checkpoint SHA-256")
    video_sha256 = require_sha256(input_video_sha256, name="input video SHA-256")
    video_bytes = require_exact_integer(
        input_video_byte_count,
        name="input video byte count",
        minimum=1,
    )
    start = require_exact_integer(frame_start, name="frame_start", minimum=0)
    if storage_dtype not in DENSE_STORAGE_DTYPES:
        raise ValueError("storage_dtype must be one of " + ", ".join(DENSE_STORAGE_DTYPES))
    if type(sequence_id) is not str or not sequence_id:
        raise ValueError("sequence_id must be a nonempty string")
    if type(view_id) is not str or not view_id:
        raise ValueError("view_id must be a nonempty string")
    if type(window_id) is not str or not window_id:
        raise ValueError("window_id must be a nonempty string")
    if type(confidence_threshold) not in {int, float}:
        raise TypeError("confidence_threshold must be an int or float")
    threshold = float(confidence_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("confidence_threshold must be finite and non-negative")

    source = Path(source_root)
    window, source_members = _canonical_window(
        source,
        frame_start=start,
        window_id=window_id,
        confidence_threshold=threshold,
        storage_dtype=storage_dtype,
    )
    _verify_source_descriptors(source, source_members)

    loader_id = _file_sha256(Path(__file__).resolve())
    model_set_id = _record_id(
        _MODEL_SET_DOMAIN,
        {
            "official_repository": CUT3R_OFFICIAL_REPOSITORY,
            "cut3r_revision": revision,
            "checkpoint_sha256": checkpoint,
        },
    )
    source_bundle_id = _record_id(
        _SOURCE_BUNDLE_DOMAIN,
        {
            "layout": CUT3R_ONLINE_SOURCE_LAYOUT,
            "members": list(source_members),
        },
    )
    provider_run_id = _record_id(
        _RUN_DOMAIN,
        {
            "model_set_id": model_set_id,
            "loader_id": loader_id,
            "source_bundle_id": source_bundle_id,
            "input_video_sha256": video_sha256,
            "input_video_byte_count": video_bytes,
            "frame_start": start,
            "frame_count": int(window.shape[0]),
            "confidence_threshold": threshold,
            "execution_mode": "recurrent-online",
            "revisit_count": 1,
            "global_alignment": False,
        },
    )
    stochastic_member_id = _record_id(
        _MEMBER_DOMAIN,
        {
            "model_set_id": model_set_id,
            "provider_run_id": provider_run_id,
        },
    )

    output_path = Path(output_manifest_path)
    manifest_root = output_path.parent.resolve()
    sequence_token = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:16]
    payload_path = manifest_root / "payloads" / f"{sequence_token}-cut3r-online.npz"
    _write_window_atomically(payload_path, window)
    payload = PredictionPayloadDescriptorV1(
        product_role="external-sequence",
        window_id=window.window_id,
        path=_relative_member(payload_path, root=manifest_root),
        sha256=_file_sha256(payload_path),
        byte_count=int(payload_path.stat().st_size),
        view_id=view_id,
        stochastic_member_id=stochastic_member_id,
        dependence_group_ids=(
            f"input-video:{video_sha256}",
            f"model-set:{model_set_id}",
            f"provider-run:{provider_run_id}",
        ),
        dense_storage_dtype=storage_dtype,
        has_scene_flow=False,
        has_ray_directions=False,
        frame_lineage=tuple(
            PredictionFrameLineageV1(
                output_frame_id=int(frame_id),
                source_frame_start=start,
                source_frame_stop_exclusive=int(frame_id) + 1,
                contributor_ids=(provider_run_id,),
            )
            for frame_id in window.frame_indices
        ),
    )
    manifest = PredictionProviderManifestV1(
        sequence_id=sequence_id,
        provider_family="CUT3R-online",
        provider_repository=CUT3R_OFFICIAL_REPOSITORY,
        provider_revision=revision,
        provider_run_id=provider_run_id,
        model_set_id=model_set_id,
        loader_id=loader_id,
        coordinate_semantics="sequence-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=(payload,),
        metadata={
            "source_adapter": _ADAPTER_DOMAIN,
            "source_adapter_sha256": loader_id,
            "source_layout": CUT3R_ONLINE_SOURCE_LAYOUT,
            "source_bundle_id": source_bundle_id,
            "source_member_count": len(source_members),
            "source_member_total_bytes": sum(member["byte_count"] for member in source_members),
            "input_video_sha256": video_sha256,
            "input_video_byte_count": video_bytes,
            "checkpoint_sha256": checkpoint,
            "execution_mode": "recurrent-online",
            "online_prefix_only": True,
            "revisit_count": 1,
            "global_alignment": False,
            "confidence_threshold": threshold,
            "confidence_is_support_not_reliability": True,
            "metric_scale_claimed": False,
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    )
    save_prediction_provider_manifest(output_path, manifest)
    verify_prediction_provider_manifest(output_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction import-cut3r-online",
        description=(
            "convert official recurrent-online CUT3R depth/conf/camera outputs "
            "into a provider-neutral prediction manifest"
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
    parser.add_argument("--window-id", default="cut3r-online")
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument(
        "--storage-dtype",
        choices=DENSE_STORAGE_DTYPES,
        default="float32",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    manifest = import_cut3r_online_prediction_manifest(
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
    )
    _, report = verify_prediction_provider_manifest(arguments.output)
    output: dict[str, Any] = {
        **manifest.summary(),
        "verified_payload_count": report["verified_payload_count"],
        "execution_mode": "recurrent-online",
        "online_prefix_only": True,
        "global_alignment": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CUT3R_OFFICIAL_REPOSITORY",
    "CUT3R_ONLINE_SOURCE_LAYOUT",
    "import_cut3r_online_prediction_manifest",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
