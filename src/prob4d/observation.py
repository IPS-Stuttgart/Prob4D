"""Validated, versioned observations for downstream Bayesian fusion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._observation_validation import (
    _REQUIRED_PROVENANCE,
    _integer_array,
    _integer_scalar,
    _json_value,
    _validate_covariances,
    _validate_sha256,
)

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]
CoordinateStatus = Literal["gauge_relative", "metric"]
GaugeStatus = Literal["unresolved", "anchored"]

OBSERVATION_FORMAT_VERSION = 2
FUSION_METHOD_NAMES = (
    "prob4d_uniform",
    "prob4d_uniform_smoothed",
    "prob4d_precision",
    "prob4d_ci",
    "prob4d_ci_smoothed_uncalibrated",
)


@dataclass(frozen=True)
class SourceWindowProvenance:
    """Source frames and dependence group for one decoded prediction window."""

    window_id: str
    frame_indices: tuple[int, ...]
    correlation_group: str

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("source window_id must not be empty")
        if not self.correlation_group:
            raise ValueError("source correlation_group must not be empty")
        frame_array = _integer_array(self.frame_indices, name="source frame_indices", ndim=1)
        frames = tuple(int(frame) for frame in frame_array)
        if not frames or any(second <= first for first, second in pairwise(frames)):
            raise ValueError("source frame_indices must be non-empty and strictly increasing")
        if frames[0] < 0:
            raise ValueError("source frame_indices must be non-negative")
        object.__setattr__(self, "frame_indices", frames)

    @property
    def maximum_source_frame(self) -> int:
        return self.frame_indices[-1]

    def contains(self, frame: int) -> bool:
        return int(frame) in self.frame_indices

    @classmethod
    def from_prediction_window(
        cls,
        window: Any,
        *,
        correlation_group: str,
    ) -> SourceWindowProvenance:
        return cls(
            window_id=str(window.window_id),
            frame_indices=tuple(int(frame) for frame in np.asarray(window.frame_indices)),
            correlation_group=correlation_group,
        )


@dataclass(frozen=True)
class ObservationArtifact:
    """Dense observations plus uncertainty, gauge status, and causal provenance."""

    frame_indices: IntArray
    point_mean: FloatArray
    valid_mask: BoolArray
    point_covariance: FloatArray
    contributors: IntArray
    source_windows: tuple[SourceWindowProvenance, ...]
    frame_contributor_window_ids: tuple[tuple[str, ...], ...]
    max_source_frame_used: IntArray
    coordinate_status: CoordinateStatus
    gauge_status: GaugeStatus
    covariance_units: str
    gauge_reference: str | None
    provenance: dict[str, Any]
    causal_max_frame: int | None = None
    gauge_mean: FloatArray | None = None
    gauge_covariance: FloatArray | None = None
    scene_flow: FloatArray | None = None
    deform_mask: BoolArray | None = None
    flow_covariance: FloatArray | None = None

    def __post_init__(self) -> None:
        frames = _integer_array(self.frame_indices, name="frame_indices", ndim=1)
        if frames.size == 0 or np.any(np.diff(frames) <= 0):
            raise ValueError("frame_indices must be non-empty and strictly increasing")
        if frames[0] < 0:
            raise ValueError("frame_indices must be non-negative")

        points = np.asarray(self.point_mean)
        mask = np.asarray(self.valid_mask, dtype=bool)
        covariance = np.asarray(self.point_covariance)
        if not np.issubdtype(points.dtype, np.floating):
            raise ValueError("point_mean must use a floating-point dtype")
        if not np.issubdtype(covariance.dtype, np.floating):
            raise ValueError("point_covariance must use a floating-point dtype")
        contributor_raw = np.asarray(self.contributors)
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("point_mean must have shape (T, H, W, 3)")
        if points.shape[1] == 0 or points.shape[2] == 0:
            raise ValueError("point_mean spatial dimensions must be non-empty")
        if mask.shape != points.shape[:-1]:
            raise ValueError("valid_mask must have shape (T, H, W)")
        if covariance.shape != points.shape + (3,):
            raise ValueError("point_covariance must have shape (T, H, W, 3, 3)")
        if contributor_raw.shape != mask.shape:
            raise ValueError("contributors must have shape (T, H, W)")
        if frames.shape != (points.shape[0],):
            raise ValueError("frame_indices must match the point_mean time dimension")
        if not np.issubdtype(contributor_raw.dtype, np.integer):
            raise ValueError("contributors must contain integers")
        if np.any(contributor_raw < 0):
            raise ValueError("contributors must be non-negative")
        if np.any(contributor_raw > np.iinfo(np.uint16).max):
            raise ValueError("contributors exceed the uint16 artifact representation")
        contributors = np.asarray(contributor_raw, dtype=np.uint16)
        if not np.array_equal(contributors > 0, mask):
            raise ValueError("contributors must be positive exactly where valid_mask is true")
        if not np.all(np.isfinite(points)):
            raise ValueError("point_mean must be finite")
        _validate_covariances(covariance, name="point_covariance", active=mask)

        source_windows = tuple(self.source_windows)
        if not source_windows:
            raise ValueError("source_windows must not be empty")
        source_ids = [window.window_id for window in source_windows]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source window IDs must be unique")
        source_map = {window.window_id: window for window in source_windows}

        frame_contributors = tuple(tuple(ids) for ids in self.frame_contributor_window_ids)
        if len(frame_contributors) != frames.size:
            raise ValueError("frame_contributor_window_ids must have one entry per frame")
        maximum_source = _integer_array(
            self.max_source_frame_used,
            name="max_source_frame_used",
            ndim=1,
        )
        if maximum_source.shape != frames.shape:
            raise ValueError("max_source_frame_used must have one entry per frame")

        for index, (frame, ids) in enumerate(zip(frames, frame_contributors, strict=True)):
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("each frame must have unique, non-empty source window IDs")
            unknown = set(ids).difference(source_map)
            if unknown:
                raise ValueError(f"frame provenance references unknown source windows: {unknown}")
            if any(not source_map[window_id].contains(int(frame)) for window_id in ids):
                raise ValueError("frame provenance must reference windows containing that frame")
            direct_limit = max(source_map[window_id].maximum_source_frame for window_id in ids)
            if maximum_source[index] < direct_limit:
                raise ValueError("max_source_frame_used cannot precede a direct contributor")
            if int(np.max(contributors[index])) > len(ids):
                raise ValueError("pixel contributor count exceeds frame source-window count")

        if self.coordinate_status not in {"gauge_relative", "metric"}:
            raise ValueError("coordinate_status must be gauge_relative or metric")
        if self.gauge_status not in {"unresolved", "anchored"}:
            raise ValueError("gauge_status must be unresolved or anchored")
        if not self.covariance_units:
            raise ValueError("covariance_units must not be empty")
        if self.coordinate_status == "metric":
            if self.gauge_status != "anchored":
                raise ValueError("metric coordinates require an anchored gauge")
            if self.covariance_units != "m^2":
                raise ValueError("metric observation covariance must use m^2")
        elif self.covariance_units == "m^2":
            raise ValueError("gauge-relative covariance must not be labelled m^2")

        gauge_mean = (
            None
            if self.gauge_mean is None
            else np.asarray(self.gauge_mean, dtype=np.float64)
        )
        gauge_covariance = (
            None
            if self.gauge_covariance is None
            else np.asarray(self.gauge_covariance, dtype=np.float64)
        )
        if self.gauge_status == "anchored":
            if gauge_mean is None or gauge_mean.shape != (7,):
                raise ValueError("anchored gauge_mean must have shape (7,)")
            if gauge_covariance is None or gauge_covariance.shape != (7, 7):
                raise ValueError("anchored gauge_covariance must have shape (7, 7)")
            if not np.all(np.isfinite(gauge_mean)):
                raise ValueError("gauge_mean must be finite")
            _validate_covariances(gauge_covariance, name="gauge_covariance")
        elif gauge_mean is not None or gauge_covariance is not None:
            raise ValueError("unresolved gauge must not claim Gaussian gauge moments")
        if self.gauge_reference is not None and not self.gauge_reference:
            raise ValueError("gauge_reference must be non-empty when supplied")

        flow = None if self.scene_flow is None else np.asarray(self.scene_flow)
        flow_mask = None if self.deform_mask is None else np.asarray(self.deform_mask, dtype=bool)
        flow_covariance = (
            None
            if self.flow_covariance is None
            else np.asarray(self.flow_covariance)
        )
        presence = (flow is not None, flow_mask is not None, flow_covariance is not None)
        if len(set(presence)) != 1:
            raise ValueError("scene_flow, deform_mask, and flow_covariance must appear together")
        if flow is not None:
            if not np.issubdtype(flow.dtype, np.floating):
                raise ValueError("scene_flow must use a floating-point dtype")
            if not np.issubdtype(flow_covariance.dtype, np.floating):
                raise ValueError("flow_covariance must use a floating-point dtype")
            if flow.shape != points.shape or flow_mask.shape != mask.shape:
                raise ValueError("scene-flow arrays must match point observation shapes")
            if flow_covariance.shape != covariance.shape:
                raise ValueError("flow_covariance must match point_covariance shape")
            if np.any(flow_mask & ~mask):
                raise ValueError("deform_mask must be a subset of valid_mask")
            if not np.all(np.isfinite(flow)):
                raise ValueError("scene_flow must be finite")
            _validate_covariances(flow_covariance, name="flow_covariance", active=flow_mask)

        provenance = _json_value(dict(self.provenance))
        missing_provenance = _REQUIRED_PROVENANCE.difference(provenance)
        if missing_provenance:
            raise ValueError(f"provenance is missing required fields: {sorted(missing_provenance)}")
        for key in _REQUIRED_PROVENANCE - {"source_manifest_sha256"}:
            if not isinstance(provenance[key], str) or not provenance[key]:
                raise ValueError(f"provenance field {key!r} must be a non-empty string")
        provenance["source_manifest_sha256"] = _validate_sha256(
            provenance["source_manifest_sha256"],
            "provenance.source_manifest_sha256",
        )

        causal_limit = (
            None
            if self.causal_max_frame is None
            else _integer_scalar(self.causal_max_frame, name="causal_max_frame")
        )
        if causal_limit is not None:
            if causal_limit < 0:
                raise ValueError("causal_max_frame must be non-negative")
            if np.any(frames > causal_limit) or np.any(maximum_source > causal_limit):
                raise ValueError(
                    "causal artifact contains an output or source dependency after causal_max_frame"
                )

        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "point_mean", points)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "point_covariance", covariance)
        object.__setattr__(self, "contributors", contributors)
        object.__setattr__(self, "source_windows", source_windows)
        object.__setattr__(self, "frame_contributor_window_ids", frame_contributors)
        object.__setattr__(self, "max_source_frame_used", maximum_source)
        object.__setattr__(self, "gauge_mean", gauge_mean)
        object.__setattr__(self, "gauge_covariance", gauge_covariance)
        object.__setattr__(self, "scene_flow", flow)
        object.__setattr__(self, "deform_mask", flow_mask)
        object.__setattr__(self, "flow_covariance", flow_covariance)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "causal_max_frame", causal_limit)

    @classmethod
    def from_fused_sequence(
        cls,
        sequence: Any,
        source_windows: tuple[SourceWindowProvenance, ...],
        *,
        coordinate_status: CoordinateStatus,
        gauge_status: GaugeStatus,
        covariance_units: str,
        gauge_reference: str | None,
        provenance: dict[str, Any],
        causal_max_frame: int | None = None,
        global_estimator_source_frame_limit: int | None = None,
        gauge_mean: FloatArray | None = None,
        gauge_covariance: FloatArray | None = None,
    ) -> ObservationArtifact:
        """Build an artifact while conservatively tracking source-frame dependencies."""

        if not source_windows:
            raise ValueError("source_windows must not be empty")
        frames = np.asarray(sequence.frame_indices, dtype=np.int64)
        contributors_by_frame: list[tuple[str, ...]] = []
        maximum_source: list[int] = []
        global_limit = (
            max(window.maximum_source_frame for window in source_windows)
            if global_estimator_source_frame_limit is None
            else _integer_scalar(
                global_estimator_source_frame_limit,
                name="global_estimator_source_frame_limit",
            )
        )
        for frame in frames:
            ids = tuple(
                window.window_id
                for window in source_windows
                if window.contains(int(frame))
            )
            if not ids:
                raise ValueError(f"output frame {int(frame)} has no declared source window")
            contributors_by_frame.append(ids)
            direct_limit = max(
                window.maximum_source_frame
                for window in source_windows
                if window.window_id in ids
            )
            maximum_source.append(max(global_limit, direct_limit))
        return cls(
            frame_indices=frames,
            point_mean=sequence.point_map,
            valid_mask=sequence.valid_mask,
            point_covariance=sequence.point_covariance,
            contributors=sequence.contributors,
            source_windows=source_windows,
            frame_contributor_window_ids=tuple(contributors_by_frame),
            max_source_frame_used=np.asarray(maximum_source, dtype=np.int64),
            coordinate_status=coordinate_status,
            gauge_status=gauge_status,
            covariance_units=covariance_units,
            gauge_reference=gauge_reference,
            provenance=provenance,
            causal_max_frame=causal_max_frame,
            gauge_mean=gauge_mean,
            gauge_covariance=gauge_covariance,
            scene_flow=sequence.scene_flow,
            deform_mask=sequence.deform_mask,
            flow_covariance=sequence.flow_covariance,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "format_version": OBSERVATION_FORMAT_VERSION,
            "frames": int(self.frame_indices.size),
            "first_frame": int(self.frame_indices[0]),
            "last_frame": int(self.frame_indices[-1]),
            "height": int(self.point_mean.shape[1]),
            "width": int(self.point_mean.shape[2]),
            "coordinate_status": self.coordinate_status,
            "gauge_status": self.gauge_status,
            "covariance_units": self.covariance_units,
            "causal_max_frame": self.causal_max_frame,
            "maximum_source_frame_used": int(np.max(self.max_source_frame_used)),
            "source_window_count": len(self.source_windows),
            "has_scene_flow": self.scene_flow is not None,
        }
