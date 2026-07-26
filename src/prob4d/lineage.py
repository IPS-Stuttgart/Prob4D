"""Temporal source-lineage contracts for MotionCrafter prediction products."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.integer]

MOTIONCRAFTER_LINEAGE_SCHEMA_VERSION = 1
MOTIONCRAFTER_WINDOWING_MODEL = "motioncrafter_sliding_window_v1"
_PRODUCT_MANIFEST_KEYS = {
    "disjoint": "disjoint_baseline",
    "latent_linear": "latent_linear_baseline",
}


@dataclass(frozen=True)
class MotionCrafterWindowing:
    """Sliding-window parameters used by MotionCrafter's ``_process_windows``."""

    window_size: int
    overlap: int

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError("overlap must lie in [0, window_size)")

    def window_slices(self, frame_count: int) -> tuple[tuple[int, int], ...]:
        """Return the exact half-open input slices used by MotionCrafter."""

        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        window_size = min(self.window_size, frame_count)
        overlap = 0 if frame_count <= self.window_size else self.overlap
        stride = window_size - overlap
        windows: list[tuple[int, int]] = []
        start = 0
        while start < frame_count - overlap:
            windows.append((start, min(start + window_size, frame_count)))
            start += stride
        if not windows or windows[-1][1] != frame_count:
            raise RuntimeError("MotionCrafter window schedule does not cover every frame")
        return tuple(windows)

    def dependency_for_output(
        self,
        frame_indices: IntArray,
        output_frame: int,
    ) -> tuple[int, int, tuple[int, ...], int]:
        """Return inclusive source bounds and contributing internal windows."""

        frames = np.asarray(frame_indices, dtype=np.int64)
        if frames.ndim != 1 or frames.size == 0:
            raise ValueError("frame_indices must be a nonempty vector")
        if np.any(np.diff(frames) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        output_index = int(np.searchsorted(frames, output_frame))
        if output_index == len(frames) or frames[output_index] != output_frame:
            raise ValueError(f"output frame {output_frame} is absent from the prediction")

        windows = self.window_slices(len(frames))
        contributors = tuple(
            index for index, (start, stop) in enumerate(windows) if start <= output_index < stop
        )
        if not contributors:
            raise RuntimeError("output frame has no MotionCrafter source window")
        source_start = min(windows[index][0] for index in contributors)
        source_stop = max(windows[index][1] for index in contributors)
        return (
            int(frames[source_start]),
            int(frames[source_stop - 1]),
            contributors,
            output_index,
        )


@dataclass(frozen=True)
class CausalFrameAudit:
    """Fail-closed source-lineage decision for one prediction frame."""

    product: str
    output_frame: int
    output_index: int
    cutoff_frame: int
    source_frame_min: int
    source_frame_max: int
    source_window_indices: tuple[int, ...]
    window_size: int
    overlap: int
    lineage_source: str
    admissible: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_window_indices"] = list(self.source_window_indices)
        payload["source_window_ids"] = [
            f"internal_window_{index:04d}" for index in self.source_window_indices
        ]
        payload["source_window_count"] = len(self.source_window_indices)
        payload["source_bounds_inclusive"] = True
        payload["admissibility_rule"] = "source_frame_max < cutoff_frame"
        return payload


def motioncrafter_temporal_lineage_manifest(
    *,
    window_size: int,
    overlap: int,
) -> dict[str, object]:
    """Build the versioned temporal-lineage section of ``predictions.json``."""

    disjoint = MotionCrafterWindowing(window_size=window_size, overlap=0)
    latent = MotionCrafterWindowing(window_size=window_size, overlap=overlap)
    return {
        "schema_version": MOTIONCRAFTER_LINEAGE_SCHEMA_VERSION,
        "model": MOTIONCRAFTER_WINDOWING_MODEL,
        "frame_index_source": "prediction archive frame_indices",
        "source_bounds": "inclusive source-video frame identifiers",
        "products": {
            "disjoint_baseline": asdict(disjoint),
            "latent_linear_baseline": asdict(latent),
            "overlap_windows": {
                "window_size_source": "prediction archive frame count",
                "overlap": 0,
            },
        },
    }


def _windowing_from_manifest(
    manifest: Mapping[str, Any],
    product: str,
) -> tuple[MotionCrafterWindowing, str]:
    if product not in _PRODUCT_MANIFEST_KEYS:
        raise ValueError("prediction product must be disjoint or latent_linear")
    product_key = _PRODUCT_MANIFEST_KEYS[product]
    lineage = manifest.get("temporal_lineage")
    if lineage is not None:
        if not isinstance(lineage, Mapping):
            raise ValueError("temporal_lineage must be a mapping")
        version = int(lineage.get("schema_version", -1))
        if version != MOTIONCRAFTER_LINEAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported temporal-lineage schema_version {version}")
        if lineage.get("model") != MOTIONCRAFTER_WINDOWING_MODEL:
            raise ValueError("unsupported temporal-lineage model")
        products = lineage.get("products")
        if not isinstance(products, Mapping) or product_key not in products:
            raise ValueError(f"temporal lineage is missing {product_key}")
        settings = products[product_key]
        if not isinstance(settings, Mapping):
            raise ValueError(f"temporal lineage for {product_key} must be a mapping")
        return (
            MotionCrafterWindowing(
                window_size=int(settings["window_size"]),
                overlap=int(settings["overlap"]),
            ),
            "manifest_temporal_lineage_v1",
        )

    config = manifest.get("config")
    if not isinstance(config, Mapping) or "window_size" not in config:
        raise ValueError(
            "prediction manifest lacks temporal lineage and legacy window configuration; "
            "regenerate the prediction bundle"
        )
    overlap = 0 if product == "disjoint" else int(config.get("overlap", -1))
    return (
        MotionCrafterWindowing(
            window_size=int(config["window_size"]),
            overlap=overlap,
        ),
        "reconstructed_from_legacy_manifest_config",
    )


def audit_motioncrafter_product_frame(
    manifest: Mapping[str, Any],
    frame_indices: IntArray,
    *,
    product: str,
    output_frame: int,
    cutoff_frame: int,
) -> CausalFrameAudit:
    """Audit whether one output was computed exclusively from pre-cutoff RGB."""

    if cutoff_frame < 1:
        raise ValueError("cutoff_frame must be positive")
    if output_frame >= cutoff_frame:
        raise ValueError("output_frame must precede cutoff_frame")
    windowing, lineage_source = _windowing_from_manifest(manifest, product)
    source_min, source_max, contributors, output_index = windowing.dependency_for_output(
        frame_indices,
        output_frame,
    )
    return CausalFrameAudit(
        product=product,
        output_frame=output_frame,
        output_index=output_index,
        cutoff_frame=cutoff_frame,
        source_frame_min=source_min,
        source_frame_max=source_max,
        source_window_indices=contributors,
        window_size=windowing.window_size,
        overlap=windowing.overlap,
        lineage_source=lineage_source,
        admissible=source_max < cutoff_frame,
    )
