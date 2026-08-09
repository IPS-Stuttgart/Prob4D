"""Source-only multi-view spatial support diagnostics for causal tracklets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from .causal_tracklets import CausalTrackletSet
from .spatial_tracklets import (
    SPATIAL_TRACKLET_CLAIM_BOUNDARY,
    seed_cell_ids_by_track,
)


def _strict_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_real(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} and not isinstance(value, (np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CameraPanelSupportPolicyV1:
    """Frozen thresholds for a source-only multi-view spatial support audit."""

    minimum_view_count: int = 2
    minimum_seed_cell_count: int = 8
    minimum_views_per_cell: int = 1
    minimum_supported_frame_fraction: float = 1.0
    require_all_declared_views: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_view_count",
            _strict_integer(
                self.minimum_view_count,
                name="minimum_view_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "minimum_seed_cell_count",
            _strict_integer(
                self.minimum_seed_cell_count,
                name="minimum_seed_cell_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "minimum_views_per_cell",
            _strict_integer(
                self.minimum_views_per_cell,
                name="minimum_views_per_cell",
                minimum=1,
            ),
        )
        fraction = _strict_real(
            self.minimum_supported_frame_fraction,
            name="minimum_supported_frame_fraction",
            maximum=1.0,
        )
        object.__setattr__(self, "minimum_supported_frame_fraction", fraction)
        object.__setattr__(
            self,
            "require_all_declared_views",
            _strict_bool(
                self.require_all_declared_views,
                name="require_all_declared_views",
            ),
        )


@dataclass(frozen=True, slots=True)
class CameraPanelFrameSupportV1:
    """Spatial support decision for one required causal-prefix frame."""

    frame_index: int
    contributing_view_ids: tuple[str, ...]
    union_seed_cell_ids: tuple[int, ...]
    corroborated_seed_cell_ids: tuple[int, ...]
    supported: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_index",
            _strict_integer(self.frame_index, name="frame_index"),
        )
        for name in (
            "contributing_view_ids",
            "union_seed_cell_ids",
            "corroborated_seed_cell_ids",
            "reason_codes",
        ):
            if type(getattr(self, name)) is not tuple:
                raise ValueError(f"{name} must be a tuple")
        views = tuple(
            _strict_string(value, name="contributing_view_id")
            for value in self.contributing_view_ids
        )
        if views != tuple(sorted(set(views))):
            raise ValueError("contributing_view_ids must be sorted and unique")
        union = tuple(
            _strict_integer(value, name="union_seed_cell_id")
            for value in self.union_seed_cell_ids
        )
        corroborated = tuple(
            _strict_integer(value, name="corroborated_seed_cell_id")
            for value in self.corroborated_seed_cell_ids
        )
        if union != tuple(sorted(set(union))):
            raise ValueError("union_seed_cell_ids must be sorted and unique")
        if corroborated != tuple(sorted(set(corroborated))):
            raise ValueError("corroborated_seed_cell_ids must be sorted and unique")
        if not set(corroborated).issubset(set(union)):
            raise ValueError("corroborated seed cells must be in the panel union")
        supported = _strict_bool(self.supported, name="supported")
        reasons = tuple(
            _strict_string(value, name="reason_code") for value in self.reason_codes
        )
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("reason_codes must be sorted and unique")
        if supported == bool(reasons):
            raise ValueError("supported frames must have no reasons and failures must have reasons")
        object.__setattr__(self, "contributing_view_ids", views)
        object.__setattr__(self, "union_seed_cell_ids", union)
        object.__setattr__(self, "corroborated_seed_cell_ids", corroborated)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "contributing_view_ids": list(self.contributing_view_ids),
            "contributing_view_count": len(self.contributing_view_ids),
            "union_seed_cell_ids": list(self.union_seed_cell_ids),
            "union_seed_cell_count": len(self.union_seed_cell_ids),
            "corroborated_seed_cell_ids": list(self.corroborated_seed_cell_ids),
            "corroborated_seed_cell_count": len(self.corroborated_seed_cell_ids),
            "supported": self.supported,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class CameraPanelSupportReportV1:
    """Content-addressed target-free support result for one camera panel."""

    panel_id: str
    causal_frame_stop: int
    required_frame_indices: tuple[int, ...]
    declared_view_ids: tuple[str, ...]
    seed_cell_grid_shape: tuple[int, int]
    policy: CameraPanelSupportPolicyV1
    frame_results: tuple[CameraPanelFrameSupportV1, ...]
    support_feasible: bool
    decision_reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    claim_boundary: str = SPATIAL_TRACKLET_CLAIM_BOUNDARY
    camera_panel_support_id: str | None = None

    def __post_init__(self) -> None:
        panel = _strict_string(self.panel_id, name="panel_id")
        cutoff = _strict_integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if type(self.required_frame_indices) is not tuple or not self.required_frame_indices:
            raise ValueError("required_frame_indices must be a non-empty tuple")
        required = tuple(
            _strict_integer(value, name="required_frame_index")
            for value in self.required_frame_indices
        )
        if required != tuple(sorted(set(required))):
            raise ValueError("required_frame_indices must be sorted and unique")
        if any(frame >= cutoff for frame in required):
            raise ValueError("required frames cross the causal frame stop")
        if type(self.declared_view_ids) is not tuple or not self.declared_view_ids:
            raise ValueError("declared_view_ids must be a non-empty tuple")
        views = tuple(
            _strict_string(value, name="declared_view_id")
            for value in self.declared_view_ids
        )
        if views != tuple(sorted(set(views))):
            raise ValueError("declared_view_ids must be sorted and unique")
        if type(self.seed_cell_grid_shape) is not tuple or len(self.seed_cell_grid_shape) != 2:
            raise ValueError("seed_cell_grid_shape must be a two-element tuple")
        grid = tuple(
            _strict_integer(value, name="seed_cell_grid_shape", minimum=1)
            for value in self.seed_cell_grid_shape
        )
        if not isinstance(self.policy, CameraPanelSupportPolicyV1):
            raise TypeError("policy must be a CameraPanelSupportPolicyV1")
        if type(self.frame_results) is not tuple:
            raise ValueError("frame_results must be a tuple")
        results = self.frame_results
        if tuple(result.frame_index for result in results) != required:
            raise ValueError("frame_results must exactly cover required_frame_indices")
        support = _strict_bool(self.support_feasible, name="support_feasible")
        supported_count = sum(result.supported for result in results)
        fraction = supported_count / len(results)
        expected_support = fraction >= self.policy.minimum_supported_frame_fraction
        if support is not expected_support:
            raise ValueError("support_feasible contradicts frame results and policy")
        reason = _strict_string(self.decision_reason, name="decision_reason")
        expected_reason = (
            "camera-panel-spatial-support-feasible"
            if support
            else "camera-panel-spatial-support-negative"
        )
        if reason != expected_reason:
            raise ValueError("decision_reason contradicts support_feasible")
        if self.claim_boundary != SPATIAL_TRACKLET_CLAIM_BOUNDARY:
            raise ValueError("camera-panel support claim boundary changed")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("camera-panel support metadata must be finite JSON") from error

        object.__setattr__(self, "panel_id", panel)
        object.__setattr__(self, "causal_frame_stop", cutoff)
        object.__setattr__(self, "required_frame_indices", required)
        object.__setattr__(self, "declared_view_ids", views)
        object.__setattr__(self, "seed_cell_grid_shape", grid)
        object.__setattr__(self, "support_feasible", support)
        object.__setattr__(self, "decision_reason", reason)
        object.__setattr__(self, "metadata", metadata)
        expected_id = _sha256_json(self.identity_record())
        if self.camera_panel_support_id is not None and self.camera_panel_support_id != expected_id:
            raise ValueError("camera_panel_support_id mismatch")
        object.__setattr__(self, "camera_panel_support_id", expected_id)

    @property
    def supported_frame_count(self) -> int:
        return sum(result.supported for result in self.frame_results)

    @property
    def supported_frame_fraction(self) -> float:
        return self.supported_frame_count / len(self.frame_results)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_name": "prob4d.camera-panel-spatial-support",
            "schema_version": 1,
            "panel_id": self.panel_id,
            "causal_frame_stop": self.causal_frame_stop,
            "required_frame_indices": list(self.required_frame_indices),
            "declared_view_ids": list(self.declared_view_ids),
            "seed_cell_grid_shape": list(self.seed_cell_grid_shape),
            "policy": asdict(self.policy),
            "frame_results": [result.to_dict() for result in self.frame_results],
            "supported_frame_count": self.supported_frame_count,
            "supported_frame_fraction": self.supported_frame_fraction,
            "support_feasible": self.support_feasible,
            "decision_reason": self.decision_reason,
            "metadata": dict(self.metadata),
            "claim_boundary": self.claim_boundary,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "camera_panel_support_id": self.camera_panel_support_id,
        }


def _tracklet_grid_shape(tracklets: CausalTrackletSet) -> tuple[int, int]:
    raw = tracklets.metadata.get("seed_cell_grid_shape")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
        raise ValueError("tracklets do not contain a valid seed_cell_grid_shape")
    return (
        _strict_integer(raw[0], name="seed_cell_grid_shape[0]", minimum=1),
        _strict_integer(raw[1], name="seed_cell_grid_shape[1]", minimum=1),
    )


def evaluate_camera_panel_tracklet_support(
    tracklets_by_view: Mapping[str, CausalTrackletSet],
    *,
    panel_id: str,
    required_frame_indices: tuple[int, ...],
    policy: CameraPanelSupportPolicyV1 | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CameraPanelSupportReportV1:
    """Audit distributed and cross-view causal tracklet support per frame."""

    if not isinstance(tracklets_by_view, Mapping) or not tracklets_by_view:
        raise ValueError("tracklets_by_view must be a non-empty mapping")
    normalized: dict[str, CausalTrackletSet] = {}
    for raw_view, tracklets in tracklets_by_view.items():
        view = _strict_string(raw_view, name="view_id")
        if not isinstance(tracklets, CausalTrackletSet):
            raise TypeError("tracklets_by_view values must be CausalTrackletSet instances")
        normalized[view] = tracklets
    views = tuple(sorted(normalized))
    actual_policy = CameraPanelSupportPolicyV1() if policy is None else policy
    if not isinstance(actual_policy, CameraPanelSupportPolicyV1):
        raise TypeError("policy must be a CameraPanelSupportPolicyV1")
    if actual_policy.minimum_view_count > len(views):
        raise ValueError("minimum_view_count exceeds the declared panel size")
    if actual_policy.minimum_views_per_cell > len(views):
        raise ValueError("minimum_views_per_cell exceeds the declared panel size")

    cutoffs = {tracklets.causal_frame_stop for tracklets in normalized.values()}
    if len(cutoffs) != 1:
        raise ValueError("camera-panel tracklets must share one causal frame stop")
    cutoff = next(iter(cutoffs))
    grids = {_tracklet_grid_shape(tracklets) for tracklets in normalized.values()}
    if len(grids) != 1:
        raise ValueError("camera-panel tracklets must share one seed-cell grid")
    grid = next(iter(grids))
    required = tuple(
        _strict_integer(value, name="required_frame_index")
        for value in required_frame_indices
    )
    if not required or required != tuple(sorted(set(required))):
        raise ValueError("required_frame_indices must be non-empty, sorted, and unique")
    if any(frame >= cutoff for frame in required):
        raise ValueError("required frames cross the panel causal frame stop")

    cells_by_view_and_frame: dict[str, dict[int, set[int]]] = {}
    for view, tracklets in normalized.items():
        track_cells = seed_cell_ids_by_track(tracklets)
        row_cells = track_cells[np.asarray(tracklets.track_ids, dtype=np.int64)]
        per_frame: dict[int, set[int]] = {}
        for frame in np.unique(tracklets.frame_indices):
            selected = np.flatnonzero(tracklets.frame_indices == frame)
            per_frame[int(frame)] = set(int(value) for value in row_cells[selected])
        cells_by_view_and_frame[view] = per_frame

    frame_results: list[CameraPanelFrameSupportV1] = []
    for frame in required:
        per_view = {
            view: cells_by_view_and_frame[view].get(frame, set()) for view in views
        }
        contributing = tuple(sorted(view for view, cells in per_view.items() if cells))
        counts: dict[int, int] = {}
        for cells in per_view.values():
            for cell_id in cells:
                counts[cell_id] = counts.get(cell_id, 0) + 1
        union = tuple(sorted(counts))
        corroborated = tuple(
            sorted(
                cell_id
                for cell_id, count in counts.items()
                if count >= actual_policy.minimum_views_per_cell
            )
        )
        reasons: list[str] = []
        if len(contributing) < actual_policy.minimum_view_count:
            reasons.append("insufficient-contributing-views")
        if actual_policy.require_all_declared_views and len(contributing) != len(views):
            reasons.append("missing-declared-view")
        if len(corroborated) < actual_policy.minimum_seed_cell_count:
            reasons.append("insufficient-spatial-seed-cells")
        frame_results.append(
            CameraPanelFrameSupportV1(
                frame_index=frame,
                contributing_view_ids=contributing,
                union_seed_cell_ids=union,
                corroborated_seed_cell_ids=corroborated,
                supported=not reasons,
                reason_codes=tuple(sorted(reasons)),
            )
        )

    supported_count = sum(result.supported for result in frame_results)
    supported_fraction = supported_count / len(frame_results)
    support_feasible = (
        supported_fraction >= actual_policy.minimum_supported_frame_fraction
    )
    decision = (
        "camera-panel-spatial-support-feasible"
        if support_feasible
        else "camera-panel-spatial-support-negative"
    )
    return CameraPanelSupportReportV1(
        panel_id=panel_id,
        causal_frame_stop=cutoff,
        required_frame_indices=required,
        declared_view_ids=views,
        seed_cell_grid_shape=grid,
        policy=actual_policy,
        frame_results=tuple(frame_results),
        support_feasible=support_feasible,
        decision_reason=decision,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "CameraPanelFrameSupportV1",
    "CameraPanelSupportPolicyV1",
    "CameraPanelSupportReportV1",
    "evaluate_camera_panel_tracklet_support",
]
