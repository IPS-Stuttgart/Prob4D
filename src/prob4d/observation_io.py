"""Serialization for versioned Prob4D observation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._observation_validation import (
    _validate_sha256,
    pack_symmetric_covariance,
    unpack_symmetric_covariance,
)
from .observation import (
    OBSERVATION_FORMAT_VERSION,
    ObservationArtifact,
    SourceWindowProvenance,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_observation_artifact(
    manifest_path: str | Path,
    artifact: ObservationArtifact,
) -> Path:
    """Write a JSON manifest plus a hash-locked compressed NumPy payload."""

    manifest = Path(manifest_path)
    if manifest.suffix.lower() != ".json":
        raise ValueError("observation manifest path must end in .json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    array_path = manifest.with_suffix(".npz")
    temporary_array = array_path.with_name(f".{array_path.name}.tmp")
    temporary_manifest = manifest.with_name(f".{manifest.name}.tmp")

    arrays: dict[str, np.ndarray] = {
        "frame_indices": artifact.frame_indices,
        "point_mean": artifact.point_mean.astype(np.float32),
        "valid_mask": artifact.valid_mask,
        "point_covariance_packed": pack_symmetric_covariance(
            artifact.point_covariance
        ).astype(np.float32),
        "contributors": artifact.contributors,
        "max_source_frame_used": artifact.max_source_frame_used,
    }
    if artifact.gauge_mean is not None:
        arrays["gauge_mean"] = artifact.gauge_mean.astype(np.float64)
        arrays["gauge_covariance"] = artifact.gauge_covariance.astype(np.float64)
    if artifact.scene_flow is not None:
        arrays["scene_flow"] = artifact.scene_flow.astype(np.float32)
        arrays["deform_mask"] = artifact.deform_mask
        arrays["flow_covariance_packed"] = pack_symmetric_covariance(
            artifact.flow_covariance
        ).astype(np.float32)

    try:
        with temporary_array.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_array, array_path)
        payload = {
            "format_version": OBSERVATION_FORMAT_VERSION,
            "array_file": array_path.name,
            "array_sha256": _sha256(array_path),
            "coordinate_status": artifact.coordinate_status,
            "gauge_status": artifact.gauge_status,
            "covariance_units": artifact.covariance_units,
            "gauge_reference": artifact.gauge_reference,
            "causal_max_frame": artifact.causal_max_frame,
            "source_windows": [asdict(window) for window in artifact.source_windows],
            "frame_contributor_window_ids": [
                list(ids) for ids in artifact.frame_contributor_window_ids
            ],
            "provenance": artifact.provenance,
            "summary": artifact.summary(),
        }
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest)
    finally:
        temporary_array.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def load_observation_artifact(
    manifest_path: str | Path,
    *,
    verify_hash: bool = True,
) -> ObservationArtifact:
    """Load and validate a versioned observation artifact."""

    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format_version") != OBSERVATION_FORMAT_VERSION:
        raise ValueError("unsupported observation artifact format_version")
    array_file = payload.get("array_file")
    if not isinstance(array_file, str) or not array_file:
        raise ValueError("observation manifest has no array_file")
    array_path = (manifest.parent / array_file).resolve()
    if array_path.parent != manifest.parent:
        raise ValueError("array_file must remain in the observation manifest directory")
    if verify_hash:
        expected = _validate_sha256(payload.get("array_sha256"), "array_sha256")
        actual = _sha256(array_path)
        if actual != expected:
            raise ValueError("observation array SHA-256 does not match the manifest")

    with np.load(array_path, allow_pickle=False) as data:
        required = {
            "frame_indices",
            "point_mean",
            "valid_mask",
            "point_covariance_packed",
            "contributors",
            "max_source_frame_used",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"observation array is missing fields: {sorted(missing)}")
        gauge_mean = data["gauge_mean"] if "gauge_mean" in data else None
        gauge_covariance = data["gauge_covariance"] if "gauge_covariance" in data else None
        has_flow = any(
            field in data
            for field in ("scene_flow", "deform_mask", "flow_covariance_packed")
        )
        if has_flow and not all(
            field in data
            for field in ("scene_flow", "deform_mask", "flow_covariance_packed")
        ):
            raise ValueError("observation array contains an incomplete scene-flow product")
        return ObservationArtifact(
            frame_indices=data["frame_indices"],
            point_mean=data["point_mean"],
            valid_mask=data["valid_mask"],
            point_covariance=unpack_symmetric_covariance(data["point_covariance_packed"]),
            contributors=data["contributors"],
            source_windows=tuple(
                SourceWindowProvenance(
                    window_id=item["window_id"],
                    frame_indices=tuple(item["frame_indices"]),
                    correlation_group=item["correlation_group"],
                )
                for item in payload["source_windows"]
            ),
            frame_contributor_window_ids=tuple(
                tuple(ids) for ids in payload["frame_contributor_window_ids"]
            ),
            max_source_frame_used=data["max_source_frame_used"],
            coordinate_status=payload["coordinate_status"],
            gauge_status=payload["gauge_status"],
            covariance_units=payload["covariance_units"],
            gauge_reference=payload.get("gauge_reference"),
            provenance=payload["provenance"],
            causal_max_frame=payload.get("causal_max_frame"),
            gauge_mean=gauge_mean,
            gauge_covariance=gauge_covariance,
            scene_flow=data["scene_flow"] if has_flow else None,
            deform_mask=data["deform_mask"] if has_flow else None,
            flow_covariance=(
                unpack_symmetric_covariance(data["flow_covariance_packed"])
                if has_flow
                else None
            ),
        )
