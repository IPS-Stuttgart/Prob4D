"""Portable manifests for MotionCrafter predictions and evaluation truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import PredictionWindow
from .fusion import FusedSequence
from .metrics import TruthSequence


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


def load_fused_prediction(path: str | Path) -> FusedSequence:
    """Load a fused prediction that was exported with compact covariance."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frame_indices",
            "point_map",
            "valid_mask",
            "point_covariance_packed",
            "contributors",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing fused uncertainty fields: {sorted(missing)}")
        return FusedSequence(
            frame_indices=data["frame_indices"],
            point_map=data["point_map"],
            valid_mask=data["valid_mask"],
            point_covariance=unpack_symmetric_covariance(data["point_covariance_packed"]),
            contributors=data["contributors"],
            scene_flow=data["scene_flow"] if "scene_flow" in data else None,
            deform_mask=data["deform_mask"] if "deform_mask" in data else None,
            flow_covariance=(
                unpack_symmetric_covariance(data["flow_covariance_packed"])
                if "flow_covariance_packed" in data
                else None
            ),
        )


@dataclass(frozen=True)
class PredictionBundle:
    manifest_path: Path
    overlap_windows: list[PredictionWindow]
    disjoint_baseline: PredictionWindow
    latent_linear_baseline: PredictionWindow
    metadata: dict


def load_prediction_bundle(path: str | Path) -> PredictionBundle:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    root = path.parent
    windows = [
        PredictionWindow.from_npz(
            root / item["path"],
            start_frame=item.get("start_frame"),
            window_id=item["window_id"],
        )
        for item in payload["overlap_windows"]
    ]
    if not windows:
        raise ValueError("prediction manifest has no overlap windows")
    windows.sort(key=lambda window: window.start_frame)
    return PredictionBundle(
        manifest_path=path,
        overlap_windows=windows,
        disjoint_baseline=PredictionWindow.from_npz(
            root / payload["disjoint_baseline"],
            start_frame=0,
            window_id="baseline_disjoint",
        ),
        latent_linear_baseline=PredictionWindow.from_npz(
            root / payload["latent_linear_baseline"],
            start_frame=0,
            window_id="baseline_latent_linear",
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
