"""Fail-closed source selection for causal Prob4D observation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import PredictionWindow
from .lineage import (
    MOTIONCRAFTER_LINEAGE_SCHEMA_VERSION,
    MOTIONCRAFTER_WINDOWING_MODEL,
)
from .metric_anchor import MetricGaugeAnchorV1
from .observation_contract import (
    array_sha256,
    canonical_json_sha256,
    file_sha256,
)

CAUSAL_SOURCE_LINEAGE_SCHEMA_VERSION = 1

_PRODUCER_CONFIG_KEYS = (
    "model_type",
    "unet_path",
    "vae_path",
    "height",
    "width",
    "window_size",
    "overlap",
    "num_inference_steps",
    "guidance_scale",
    "decode_chunk_size",
    "seed",
    "low_memory_usage",
    "frame_start",
    "frame_stride",
)


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SelectedOverlapWindow:
    """One independently decoded window admitted before future payloads open."""

    manifest_index: int
    window_id: str
    source_frame_start: int
    source_frame_stop: int
    payload_sha256: str
    prediction: PredictionWindow

    def __post_init__(self) -> None:
        if self.manifest_index < 0 or not self.window_id:
            raise ValueError("selected overlap-window identity is invalid")
        if (
            self.source_frame_start < 0
            or self.source_frame_stop <= self.source_frame_start
        ):
            raise ValueError("selected overlap-window source bounds are invalid")
        _require_sha256(
            self.payload_sha256,
            name="overlap-window payload_sha256",
        )
        if self.prediction.window_id != self.window_id:
            raise ValueError("selected overlap-window ID changed while loading")

    def lineage_record(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "source_frame_start": self.source_frame_start,
            "source_frame_stop_exclusive": self.source_frame_stop,
            "source_frame_max": int(self.prediction.frame_indices[-1]),
            "frame_indices_sha256": array_sha256(self.prediction.frame_indices),
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class CausalOverlapSelection:
    """Causally complete windows and their append-invariant source digest."""

    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    windows: tuple[SelectedOverlapWindow, ...]
    skipped_window_count: int
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError(
                "causal overlap selection must contain at least one window"
            )
        if self.skipped_window_count < 0:
            raise ValueError("skipped_window_count must be nonnegative")
        _require_sha256(
            self.manifest_sha256,
            name="prediction manifest sha256",
        )
        _require_sha256(
            self.source_artifact_sha256,
            name="causal source sha256",
        )

    @property
    def predictions(self) -> list[PredictionWindow]:
        return [item.prediction for item in self.windows]

    def artifact_lineage_metadata(
        self,
        *,
        causal_frame_stop: int,
    ) -> dict[str, Any]:
        """Return only prefix-stable information suitable for artifact hashing."""

        return {
            "schema_version": CAUSAL_SOURCE_LINEAGE_SCHEMA_VERSION,
            "producer": "Prob4D",
            "motioncrafter_lineage_schema_version": (
                MOTIONCRAFTER_LINEAGE_SCHEMA_VERSION
            ),
            "motioncrafter_windowing_model": MOTIONCRAFTER_WINDOWING_MODEL,
            "source_product": "independently_decoded_overlap_windows",
            "causal_frame_stop_exclusive": causal_frame_stop,
            "admissibility_rule": (
                "source_frame_max < causal_frame_stop_exclusive"
            ),
            "future_prediction_payloads_opened": 0,
            "selected_windows": [
                item.lineage_record() for item in self.windows
            ],
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_digest_scope": (
                "selected payload hashes, selected source bounds, MotionCrafter "
                "revision, lineage declaration, prediction-affecting settings, "
                "and metric anchor"
            ),
        }

    def run_summary(self, *, causal_frame_stop: int) -> dict[str, Any]:
        """Return an operational report that may mention unconsumed entries."""

        return {
            "prediction_manifest": str(self.manifest_path),
            "prediction_manifest_sha256": self.manifest_sha256,
            "causal_frame_stop_exclusive": causal_frame_stop,
            "selected_window_ids": [
                item.window_id for item in self.windows
            ],
            "selected_window_count": len(self.windows),
            "skipped_future_or_crossing_window_count": (
                self.skipped_window_count
            ),
            "future_prediction_payloads_opened": 0,
            "causal_source_artifact_sha256": self.source_artifact_sha256,
        }


def _validate_overlap_lineage(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prediction-manifest format_version")
    lineage = manifest.get("temporal_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(
            "causal observation export requires an explicit temporal_lineage "
            "declaration"
        )
    if (
        int(lineage.get("schema_version", -1))
        != MOTIONCRAFTER_LINEAGE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported MotionCrafter temporal-lineage schema")
    if lineage.get("model") != MOTIONCRAFTER_WINDOWING_MODEL:
        raise ValueError("unsupported MotionCrafter temporal-lineage model")
    products = lineage.get("products")
    if not isinstance(products, Mapping):
        raise ValueError("temporal lineage has no product declarations")
    overlap = products.get("overlap_windows")
    if not isinstance(overlap, Mapping):
        raise ValueError("temporal lineage has no overlap-window declaration")
    if overlap.get("window_size_source") != "prediction archive frame count":
        raise ValueError(
            "overlap-window source extent is not explicitly declared"
        )
    if int(overlap.get("overlap", -1)) != 0:
        raise ValueError(
            "independently decoded overlap windows must have internal overlap zero"
        )
    if not str(manifest.get("motioncrafter_commit", "")):
        raise ValueError("prediction manifest has no MotionCrafter revision")
    if not isinstance(manifest.get("config"), Mapping):
        raise ValueError("prediction manifest has no producer configuration")


def _safe_payload_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("overlap-window path must not be empty")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            "overlap-window path escapes the prediction directory"
        ) from error
    return candidate


def _causal_source_digest(
    manifest: Mapping[str, Any],
    windows: Sequence[SelectedOverlapWindow],
    *,
    metric_anchor: MetricGaugeAnchorV1,
) -> str:
    config = manifest["config"]
    stable_config = {
        key: config[key]
        for key in _PRODUCER_CONFIG_KEYS
        if key in config
    }
    descriptor = {
        "schema_name": "prob4d.causal-observation-source",
        "schema_version": CAUSAL_SOURCE_LINEAGE_SCHEMA_VERSION,
        "prediction_manifest_format_version": manifest["format_version"],
        "motioncrafter_commit": manifest["motioncrafter_commit"],
        "temporal_lineage": manifest["temporal_lineage"],
        "producer_config": stable_config,
        "selected_windows": [
            window.lineage_record() for window in windows
        ],
        "metric_gauge_anchor_id": metric_anchor.artifact_id,
    }
    return canonical_json_sha256(descriptor)


def select_causal_overlap_windows(
    manifest_path: str | Path,
    *,
    causal_frame_stop: int,
    metric_anchor: MetricGaugeAnchorV1,
) -> CausalOverlapSelection:
    """Open only independently decoded windows wholly before the cutoff."""

    if causal_frame_stop < 1:
        raise ValueError("causal_frame_stop must be positive")
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _validate_overlap_lineage(manifest)
    entries = manifest.get("overlap_windows")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prediction manifest has no overlap windows")
    frame_stride = int(manifest["config"].get("frame_stride", 1))
    window_size = int(manifest["config"].get("window_size", 0))
    if frame_stride < 1 or window_size < 1:
        raise ValueError(
            "prediction manifest has invalid frame_stride or window_size"
        )

    selected: list[SelectedOverlapWindow] = []
    skipped = 0
    skipping_started = False
    seen_ids: set[str] = set()
    previous_start = -1
    for manifest_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(
                "overlap-window manifest entries must be mappings"
            )
        window_id = str(entry.get("window_id", ""))
        if not window_id or window_id in seen_ids:
            raise ValueError(
                "overlap-window IDs must be nonempty and unique"
            )
        seen_ids.add(window_id)
        start = int(entry.get("start_frame", -1))
        stop = int(entry.get("stop_frame", -1))
        if start < 0 or stop <= start:
            raise ValueError(
                f"overlap window {window_id!r} has invalid source bounds"
            )
        if start < previous_start:
            raise ValueError(
                "overlap-window manifest entries are not ordered by source frame"
            )
        previous_start = start
        if stop > causal_frame_stop:
            skipped += 1
            skipping_started = True
            continue
        if skipping_started:
            raise ValueError(
                "causally complete overlap windows are not a manifest prefix"
            )

        path = _safe_payload_path(
            manifest_file.parent,
            str(entry.get("path", "")),
        )
        prediction = PredictionWindow.from_npz(
            path,
            start_frame=start,
            window_id=window_id,
        )
        expected_frames = np.arange(
            start,
            stop,
            frame_stride,
            dtype=np.int64,
        )
        if not np.array_equal(prediction.frame_indices, expected_frames):
            raise ValueError(
                f"overlap window {window_id!r} frame IDs disagree with its "
                "manifest bounds"
            )
        if len(prediction.frame_indices) > window_size:
            raise ValueError(
                f"overlap window {window_id!r} exceeds the declared independent "
                "window size"
            )
        if int(prediction.frame_indices[-1]) >= causal_frame_stop:
            raise ValueError(
                f"overlap window {window_id!r} crosses the causal frame boundary"
            )
        selected.append(
            SelectedOverlapWindow(
                manifest_index=manifest_index,
                window_id=window_id,
                source_frame_start=start,
                source_frame_stop=stop,
                payload_sha256=file_sha256(path),
                prediction=prediction,
            )
        )

    if not selected:
        raise ValueError(
            "no independently decoded overlap window is wholly before "
            "causal_frame_stop"
        )
    metric_anchor.require_fixed()
    if selected[0].window_id != metric_anchor.reference_window_id:
        raise ValueError(
            "metric gauge anchor must identify the first retained overlap window"
        )
    if selected[0].payload_sha256 != metric_anchor.source_artifact_sha256:
        raise ValueError(
            "metric gauge anchor is bound to a different reference-window payload"
        )
    source_digest = _causal_source_digest(
        manifest,
        selected,
        metric_anchor=metric_anchor,
    )
    return CausalOverlapSelection(
        manifest_path=manifest_file,
        manifest_sha256=file_sha256(manifest_file),
        manifest=manifest,
        windows=tuple(selected),
        skipped_window_count=skipped,
        source_artifact_sha256=source_digest,
    )


__all__ = [
    "CAUSAL_SOURCE_LINEAGE_SCHEMA_VERSION",
    "CausalOverlapSelection",
    "SelectedOverlapWindow",
    "select_causal_overlap_windows",
]
