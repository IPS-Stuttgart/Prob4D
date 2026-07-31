"""Portable manifests for MotionCrafter predictions and evaluation truth."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._immutable_json import frozen_finite_json_mapping, plain_json
from .data import DenseStorageDType, PredictionWindow
from .fusion import FusedSequence
from .metrics import TruthSequence
from .motioncrafter_integrity import (
    resolve_motioncrafter_member,
    verify_motioncrafter_prediction_manifest,
)

FUSED_PREDICTION_SCHEMA = "prob4d.fused-prediction"
FUSED_PREDICTION_VERSION = 1
FusedArtifactFusionMethod = Literal[
    "uniform",
    "precision",
    "covariance_intersection",
    "unspecified",
]

_FUSION_SEMANTICS: dict[str, tuple[str, str]] = {
    "uniform": (
        "gaussian_mixture_second_moment",
        "descriptive_mixture_not_a_conditioned_posterior",
    ),
    "precision": (
        "independent_gaussian_posterior",
        "contributors_treated_as_independent",
    ),
    "covariance_intersection": (
        "unknown_correlation_consistency_bound",
        "cross_contributor_correlation_unknown",
    ),
}


def fusion_covariance_semantics(fusion_method: str) -> tuple[str, str]:
    """Return the covariance meaning and correlation assumption for a fusion rule."""

    try:
        return _FUSION_SEMANTICS[fusion_method]
    except KeyError as error:
        raise ValueError(
            "fusion_method must be one of 'uniform', 'precision', or "
            "'covariance_intersection'"
        ) from error


@dataclass(frozen=True)
class FusedPredictionMetadata:
    """Self-describing semantics stored alongside a dense fused sequence."""

    method_id: str
    fusion_method: FusedArtifactFusionMethod
    covariance_semantics: str
    correlation_assumption: str
    metadata: Mapping[str, Any]
    schema_name: str = FUSED_PREDICTION_SCHEMA
    schema_version: int = FUSED_PREDICTION_VERSION
    legacy_unspecified: bool = False

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip()
        fusion_method = str(self.fusion_method).strip()
        covariance_semantics = str(self.covariance_semantics).strip()
        correlation_assumption = str(self.correlation_assumption).strip()
        if not method_id:
            raise ValueError("fused-prediction method_id must be nonempty")
        if not covariance_semantics or not correlation_assumption:
            raise ValueError("fused-prediction covariance semantics must be nonempty")
        if self.legacy_unspecified:
            if fusion_method != "unspecified":
                raise ValueError("legacy fused-prediction metadata must be unspecified")
        else:
            if self.schema_name != FUSED_PREDICTION_SCHEMA:
                raise ValueError("unsupported fused-prediction schema")
            if self.schema_version != FUSED_PREDICTION_VERSION:
                raise ValueError("unsupported fused-prediction schema version")
            expected_semantics, expected_assumption = fusion_covariance_semantics(
                fusion_method
            )
            if covariance_semantics != expected_semantics:
                raise ValueError("fused-prediction covariance semantics changed")
            if correlation_assumption != expected_assumption:
                raise ValueError("fused-prediction correlation assumption changed")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "fusion_method", fusion_method)
        object.__setattr__(self, "covariance_semantics", covariance_semantics)
        object.__setattr__(self, "correlation_assumption", correlation_assumption)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="fused-prediction metadata",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "fusion_method": self.fusion_method,
            "covariance_semantics": self.covariance_semantics,
            "correlation_assumption": self.correlation_assumption,
            "legacy_unspecified": self.legacy_unspecified,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True)
class FusedPredictionArtifact:
    """A fused sequence together with the exact semantics of its covariance."""

    sequence: FusedSequence
    metadata: FusedPredictionMetadata


def pack_symmetric_covariance(covariance: np.ndarray) -> np.ndarray:
    """Pack the upper triangle of dense 3x3 covariance matrices into six values."""

    covariance = np.asarray(covariance)
    if covariance.shape[-2:] != (3, 3):
        raise ValueError("covariance must end in shape (3, 3)")
    return covariance[..., (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def unpack_symmetric_covariance(packed: np.ndarray) -> np.ndarray:
    """Restore dense 3x3 covariance matrices from six upper-triangle values."""

    packed = np.asarray(packed)
    if packed.shape[-1] != 6:
        raise ValueError("packed covariance must end in six values")
    covariance = np.empty(packed.shape[:-1] + (3, 3), dtype=packed.dtype)
    covariance[..., 0, 0] = packed[..., 0]
    covariance[..., 0, 1] = covariance[..., 1, 0] = packed[..., 1]
    covariance[..., 0, 2] = covariance[..., 2, 0] = packed[..., 2]
    covariance[..., 1, 1] = packed[..., 3]
    covariance[..., 1, 2] = covariance[..., 2, 1] = packed[..., 4]
    covariance[..., 2, 2] = packed[..., 5]
    return covariance


def _text_scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"fused-prediction field {key!r} must be scalar")
    return str(value.item())


def _metadata_from_archive(
    path: Path,
    data: np.lib.npyio.NpzFile,
) -> FusedPredictionMetadata:
    if "artifact_schema" not in data.files:
        return FusedPredictionMetadata(
            method_id="legacy_unspecified",
            fusion_method="unspecified",
            covariance_semantics="unspecified",
            correlation_assumption="unspecified",
            metadata={"source_path": str(path)},
            schema_name="legacy_unspecified",
            schema_version=0,
            legacy_unspecified=True,
        )
    required = {
        "artifact_schema",
        "artifact_version",
        "method_id",
        "fusion_method",
        "covariance_semantics",
        "correlation_assumption",
        "artifact_metadata_json",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(
            f"{path} is missing fused-prediction metadata fields: {sorted(missing)}"
        )
    schema_name = _text_scalar(data, "artifact_schema")
    version = np.asarray(data["artifact_version"])
    if version.shape != ():
        raise ValueError("fused-prediction artifact_version must be scalar")
    try:
        metadata = json.loads(_text_scalar(data, "artifact_metadata_json"))
    except json.JSONDecodeError as error:
        raise ValueError("fused-prediction metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("fused-prediction metadata JSON must be an object")
    return FusedPredictionMetadata(
        schema_name=schema_name,
        schema_version=int(version.item()),
        method_id=_text_scalar(data, "method_id"),
        fusion_method=_text_scalar(data, "fusion_method"),  # type: ignore[arg-type]
        covariance_semantics=_text_scalar(data, "covariance_semantics"),
        correlation_assumption=_text_scalar(data, "correlation_assumption"),
        metadata=metadata,
    )


def save_fused_prediction(
    path: str | Path,
    sequence: FusedSequence,
    *,
    method_id: str,
    fusion_method: str,
    include_covariance: bool = True,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = False,
) -> FusedPredictionMetadata:
    """Write a fused sequence with explicit covariance and dependence semantics."""

    covariance_semantics, correlation_assumption = fusion_covariance_semantics(
        fusion_method
    )
    artifact_metadata = FusedPredictionMetadata(
        method_id=method_id,
        fusion_method=fusion_method,  # type: ignore[arg-type]
        covariance_semantics=covariance_semantics,
        correlation_assumption=correlation_assumption,
        metadata={} if metadata is None else metadata,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "artifact_schema": np.asarray(artifact_metadata.schema_name),
        "artifact_version": np.asarray(artifact_metadata.schema_version, dtype=np.int64),
        "method_id": np.asarray(artifact_metadata.method_id),
        "fusion_method": np.asarray(artifact_metadata.fusion_method),
        "covariance_semantics": np.asarray(artifact_metadata.covariance_semantics),
        "correlation_assumption": np.asarray(artifact_metadata.correlation_assumption),
        "artifact_metadata_json": np.asarray(
            json.dumps(
                plain_json(artifact_metadata.metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "point_map": np.asarray(sequence.point_map, dtype=np.float32),
        "valid_mask": np.asarray(sequence.valid_mask, dtype=bool),
        "frame_indices": np.asarray(sequence.frame_indices, dtype=np.int64),
    }
    if include_covariance:
        payload["point_covariance_packed"] = pack_symmetric_covariance(
            sequence.point_covariance
        ).astype(np.float32)
        payload["contributors"] = np.asarray(sequence.contributors)
    if sequence.scene_flow is not None:
        payload["scene_flow"] = np.asarray(sequence.scene_flow, dtype=np.float32)
        payload["deform_mask"] = np.asarray(sequence.deform_mask, dtype=bool)
        if include_covariance:
            if sequence.flow_covariance is None:
                raise ValueError("scene-flow covariance is missing from fused sequence")
            payload["flow_covariance_packed"] = pack_symmetric_covariance(
                sequence.flow_covariance
            ).astype(np.float32)
    writer = np.savez_compressed if compressed else np.savez
    writer(destination, **payload)
    return artifact_metadata


def load_fused_prediction_artifact(path: str | Path) -> FusedPredictionArtifact:
    """Load a fused prediction and its covariance semantics.

    Historical archives without the versioned metadata remain loadable, but are
    explicitly marked ``legacy_unspecified`` so evidence-bearing evaluators can
    reject them unless the caller opts into that ambiguity.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {
            "frame_indices",
            "point_map",
            "valid_mask",
            "point_covariance_packed",
            "contributors",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"{source} is missing fused uncertainty fields: {sorted(missing)}"
            )
        metadata = _metadata_from_archive(source, data)
        sequence = FusedSequence(
            frame_indices=data["frame_indices"],
            point_map=data["point_map"],
            valid_mask=data["valid_mask"],
            point_covariance=unpack_symmetric_covariance(
                data["point_covariance_packed"]
            ),
            contributors=data["contributors"],
            scene_flow=data["scene_flow"] if "scene_flow" in data else None,
            deform_mask=data["deform_mask"] if "deform_mask" in data else None,
            flow_covariance=(
                unpack_symmetric_covariance(data["flow_covariance_packed"])
                if "flow_covariance_packed" in data
                else None
            ),
        )
    return FusedPredictionArtifact(sequence=sequence, metadata=metadata)


def load_fused_prediction_metadata(path: str | Path) -> FusedPredictionMetadata:
    """Load only the semantics attached to a fused prediction archive."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        return _metadata_from_archive(source, data)


def load_fused_prediction(path: str | Path) -> FusedSequence:
    """Load a fused prediction that was exported with compact covariance."""

    return load_fused_prediction_artifact(path).sequence


@dataclass(frozen=True)
class PredictionBundle:
    manifest_path: Path
    overlap_windows: list[PredictionWindow]
    disjoint_baseline: PredictionWindow
    latent_linear_baseline: PredictionWindow
    metadata: dict

    def dense_storage_summary(self) -> dict[str, object]:
        """Summarize retained dense vector storage without sampling process RSS."""

        windows = [
            *self.overlap_windows,
            self.disjoint_baseline,
            self.latent_linear_baseline,
        ]
        retained_bytes = sum(window.dense_vector_storage_bytes for window in windows)
        float64_equivalent_bytes = 0
        dense_vector_field_count = 0
        for window in windows:
            for array in (
                window.point_map,
                window.scene_flow,
                window.ray_directions,
            ):
                if array is None:
                    continue
                dense_vector_field_count += 1
                float64_equivalent_bytes += array.size * np.dtype(np.float64).itemsize
        return {
            "window_count": len(windows),
            "dense_vector_field_count": dense_vector_field_count,
            "storage_dtypes": sorted(
                {window.dense_storage_dtype for window in windows}
            ),
            "retained_bytes": retained_bytes,
            "float64_equivalent_bytes": float64_equivalent_bytes,
            "retained_fraction_of_float64": (
                0.0
                if float64_equivalent_bytes == 0
                else retained_bytes / float64_equivalent_bytes
            ),
        }


def load_prediction_bundle(
    path: str | Path,
    *,
    dense_storage_dtype: DenseStorageDType = "float64",
) -> PredictionBundle:
    """Load a verified bundle with an explicit dense in-memory storage mode."""

    path = Path(path).resolve()
    verify_motioncrafter_prediction_manifest(path, verify_hashes=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    def member(relative_path: object, *, name: str) -> Path:
        return resolve_motioncrafter_member(root, relative_path, name=name)

    windows = [
        PredictionWindow.from_npz(
            member(item["path"], name=f"overlap window {item['window_id']!r} path"),
            start_frame=item.get("start_frame"),
            window_id=item["window_id"],
            dense_storage_dtype=dense_storage_dtype,
        )
        for item in payload["overlap_windows"]
    ]
    windows.sort(key=lambda window: window.start_frame)
    return PredictionBundle(
        manifest_path=path,
        overlap_windows=windows,
        disjoint_baseline=PredictionWindow.from_npz(
            member(payload["disjoint_baseline"], name="disjoint baseline path"),
            start_frame=0,
            window_id="baseline_disjoint",
            dense_storage_dtype=dense_storage_dtype,
        ),
        latent_linear_baseline=PredictionWindow.from_npz(
            member(payload["latent_linear_baseline"], name="latent-linear baseline path"),
            start_frame=0,
            window_id="baseline_latent_linear",
            dense_storage_dtype=dense_storage_dtype,
        ),
        metadata=payload,
    )


def load_truth(path: str | Path) -> TruthSequence:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "point_map" not in data or "valid_mask" not in data:
            raise ValueError("truth file must contain point_map and valid_mask")
        frames = (
            data["frame_indices"]
            if "frame_indices" in data
            else np.arange(data["point_map"].shape[0])
        )
        return TruthSequence(
            frame_indices=frames,
            point_map=data["point_map"],
            valid_mask=data["valid_mask"],
            scene_flow=data["scene_flow"] if "scene_flow" in data else None,
            deform_mask=data["deform_mask"] if "deform_mask" in data else None,
        )


def save_truth(path: str | Path, truth: TruthSequence) -> None:
    payload = {
        "frame_indices": truth.frame_indices,
        "point_map": truth.point_map.astype(np.float32),
        "valid_mask": truth.valid_mask,
    }
    if truth.scene_flow is not None:
        payload["scene_flow"] = truth.scene_flow.astype(np.float32)
        payload["deform_mask"] = truth.deform_mask
    np.savez_compressed(Path(path), **payload)


__all__ = [
    "FUSED_PREDICTION_SCHEMA",
    "FUSED_PREDICTION_VERSION",
    "FusedPredictionArtifact",
    "FusedPredictionMetadata",
    "PredictionBundle",
    "fusion_covariance_semantics",
    "load_fused_prediction",
    "load_fused_prediction_artifact",
    "load_fused_prediction_metadata",
    "load_prediction_bundle",
    "load_truth",
    "pack_symmetric_covariance",
    "save_fused_prediction",
    "save_truth",
    "unpack_symmetric_covariance",
]
