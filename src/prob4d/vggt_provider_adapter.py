"""Convert integrity-bound VGGT caches into provider-neutral predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._atomic_file import publish_temporary_file
from .data import DENSE_STORAGE_DTYPES, PredictionWindow
from .prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
    verify_prediction_provider_manifest,
)
from .vggt_integrity import (
    VGGT_OFFICIAL_REPOSITORY,
    VGGT_REPRESENTATIONS,
    checkpoint_identity,
    file_sha256,
    find_sample_record,
    load_vggt_run_metadata,
    record_id,
    relative_member,
    verify_sample_files,
)

_ADAPTER_DOMAIN = "prob4d.vggt-provider-adapter.v1"
_STOCHASTIC_DOMAIN = "prob4d.vggt-deterministic-member.v1"


def _load_source_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["point_map"])
            extrinsics = np.asarray(archive["camera_extrinsics"])
            intrinsics = np.asarray(archive["camera_intrinsics"])
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"cannot reopen verified VGGT archive {path.name!r}") from error
    return points, extrinsics, intrinsics


def _canonical_window(
    *,
    representation: str,
    points: np.ndarray,
    frame_start: int,
    storage_dtype: str,
) -> PredictionWindow:
    if type(frame_start) is not int or frame_start < 0:
        raise ValueError("frame_start must be a nonnegative integer")
    valid = np.all(np.isfinite(points), axis=-1)
    if not np.any(valid):
        raise ValueError(f"VGGT {representation} contains no finite point")
    dense = np.asarray(points, dtype=np.float32 if storage_dtype == "float32" else np.float64)
    dense = dense.copy()
    dense[~valid] = 0.0
    frame_count = int(dense.shape[0])
    return PredictionWindow(
        window_id=f"vggt-{representation}",
        frame_indices=np.arange(
            frame_start,
            frame_start + frame_count,
            dtype=np.int64,
        ),
        point_map=dense,
        valid_mask=valid,
        dense_storage_dtype=storage_dtype,
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
                    f"refusing to replace different canonical VGGT payload {path.name!r}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _adapter_id() -> str:
    return file_sha256(Path(__file__).resolve())


def _selected_representations(values: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(VGGT_REPRESENTATIONS if not values else values)
    if len(set(selected)) != len(selected):
        raise ValueError("VGGT representations must be unique")
    if any(value not in VGGT_REPRESENTATIONS for value in selected):
        raise ValueError("unsupported VGGT representation")
    return selected


def import_vggt_prediction_manifest(
    run_metadata_path: str | Path,
    output_manifest_path: str | Path,
    *,
    sequence_id: str,
    sample_id: str,
    dataset_root: str | Path,
    prediction_root: str | Path,
    representations: Sequence[str] | None = None,
    frame_start: int = 0,
    view_id: str = "camera-0",
    storage_dtype: str = "float32",
) -> PredictionProviderManifestV1:
    """Import one exact VGGT sample without treating constructions as independent."""

    if storage_dtype not in DENSE_STORAGE_DTYPES:
        raise ValueError("storage_dtype must be one of " + ", ".join(DENSE_STORAGE_DTYPES))
    if type(frame_start) is not int or frame_start < 0:
        raise ValueError("frame_start must be a nonnegative integer")
    if not sequence_id:
        raise ValueError("sequence_id must not be empty")
    if not view_id:
        raise ValueError("view_id must not be empty")

    run = load_vggt_run_metadata(run_metadata_path)
    sample = find_sample_record(run, sample_id)
    selected = _selected_representations(representations)
    _, source_paths = verify_sample_files(
        sample=sample,
        dataset_root=dataset_root,
        output_root=prediction_root,
        representations=selected,
    )

    source_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        name: _load_source_arrays(source_paths[name]) for name in selected
    }
    reference_name = selected[0]
    reference_points, reference_extrinsics, reference_intrinsics = source_arrays[reference_name]
    for name in selected[1:]:
        points, extrinsics, intrinsics = source_arrays[name]
        if points.shape != reference_points.shape:
            raise ValueError("VGGT representations disagree on the prediction grid")
        if not np.array_equal(extrinsics, reference_extrinsics):
            raise ValueError("VGGT representations mix different camera extrinsics")
        if not np.array_equal(intrinsics, reference_intrinsics):
            raise ValueError("VGGT representations mix different camera intrinsics")

    output_path = Path(output_manifest_path)
    manifest_root = output_path.parent.resolve()
    sequence_token = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:16]
    frame_count = int(reference_points.shape[0])
    source_stop = frame_start + frame_count
    sample_run_id = str(sample["sample_run_id"])
    model_set_id = str(run["model_set_id"])
    video_sha256 = str(sample["input_video_sha256"])
    dependence_groups = (
        f"model-set:{model_set_id}",
        f"input-video:{video_sha256}",
        f"provider-run:{sample_run_id}",
    )
    deterministic_member = f"{_STOCHASTIC_DOMAIN}:" + record_id(
        _STOCHASTIC_DOMAIN,
        {
            "sample_run_id": sample_run_id,
            "model_set_id": model_set_id,
            "preprocess_mode": run["preprocess_mode"],
        },
    )

    payloads: list[PredictionPayloadDescriptorV1] = []
    source_members = {str(member["representation"]): member for member in sample["representations"]}
    for representation in selected:
        points, _, _ = source_arrays[representation]
        window = _canonical_window(
            representation=representation,
            points=points,
            frame_start=frame_start,
            storage_dtype=storage_dtype,
        )
        payload_path = manifest_root / "payloads" / f"{sequence_token}-vggt-{representation}.npz"
        _write_window_atomically(payload_path, window)
        relative_path = relative_member(
            payload_path,
            root=manifest_root,
            name="canonical VGGT payload path",
        )
        lineage = tuple(
            PredictionFrameLineageV1(
                output_frame_id=int(frame_id),
                source_frame_start=frame_start,
                source_frame_stop_exclusive=source_stop,
                contributor_ids=(sample_run_id,),
            )
            for frame_id in window.frame_indices
        )
        payloads.append(
            PredictionPayloadDescriptorV1(
                product_role="external-sequence",
                window_id=window.window_id,
                path=relative_path,
                sha256=file_sha256(payload_path),
                byte_count=int(payload_path.stat().st_size),
                view_id=view_id,
                stochastic_member_id=deterministic_member,
                dependence_group_ids=dependence_groups,
                dense_storage_dtype=storage_dtype,
                has_scene_flow=False,
                has_ray_directions=False,
                frame_lineage=lineage,
            )
        )

    checkpoint = checkpoint_identity(
        checkpoint=run["checkpoint"],
        checkpoint_sha256=run["checkpoint_sha256"],
        checkpoint_revision=run["checkpoint_revision"],
    )
    adapter_id = _adapter_id()
    manifest = PredictionProviderManifestV1(
        sequence_id=sequence_id,
        provider_family="VGGT",
        provider_repository=VGGT_OFFICIAL_REPOSITORY,
        provider_revision=str(run["vggt_commit"]),
        provider_run_id=sample_run_id,
        model_set_id=model_set_id,
        loader_id=str(run["loader_module_sha256"]),
        coordinate_semantics="sequence-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=tuple(payloads),
        metadata={
            "source_adapter": _ADAPTER_DOMAIN,
            "source_adapter_sha256": adapter_id,
            "source_run_id": run["run_id"],
            "source_sample_run_id": sample_run_id,
            "source_video_sha256": video_sha256,
            "source_video_byte_count": sample["input_video_byte_count"],
            "source_prediction_members": [
                {
                    key: source_members[name][key]
                    for key in (
                        "representation",
                        "sha256",
                        "byte_count",
                        "point_shape",
                        "point_dtype",
                        "valid_point_count",
                        "invalid_point_count",
                    )
                }
                for name in selected
            ],
            "checkpoint_identity": checkpoint,
            "preprocess_mode": run["preprocess_mode"],
            "full_sequence_dependency": True,
            "alternative_constructions_share_evidence": True,
            "uses_truth": False,
            "uses_downstream_physical_innovation": False,
        },
    )
    save_prediction_provider_manifest(output_path, manifest)
    verify_prediction_provider_manifest(output_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction import-vggt",
        description=(
            "convert one integrity-bound VGGT sample into provider-neutral "
            "PredictionWindow payloads"
        ),
    )
    parser.add_argument("run_metadata")
    parser.add_argument("output")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--representation",
        action="append",
        choices=VGGT_REPRESENTATIONS,
        help="repeat to select constructions; defaults to both official products",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--view-id", default="camera-0")
    parser.add_argument(
        "--storage-dtype",
        choices=DENSE_STORAGE_DTYPES,
        default="float32",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    manifest = import_vggt_prediction_manifest(
        arguments.run_metadata,
        arguments.output,
        sequence_id=arguments.sequence_id,
        sample_id=arguments.sample_id,
        dataset_root=arguments.dataset_root,
        prediction_root=arguments.prediction_root,
        representations=arguments.representation,
        frame_start=arguments.frame_start,
        view_id=arguments.view_id,
        storage_dtype=arguments.storage_dtype,
    )
    _, report = verify_prediction_provider_manifest(arguments.output)
    output: dict[str, Any] = {
        **manifest.summary(),
        "verified_payload_count": report["verified_payload_count"],
        "full_sequence_dependency": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


__all__ = ["import_vggt_prediction_manifest", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
