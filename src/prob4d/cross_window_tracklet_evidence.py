"""Content-addressed, joint-gauge evidence for cross-window tracklet association.

The existing :mod:`prob4d.cross_window_tracklets` diagnostic deliberately treats
cross-window covariance as unknown.  This module adds an opt-in evidence layer
that binds the complete tracklet inputs and evaluates candidate residuals with
the full joint cross-window ``Sim(3)`` gauge covariance.  It remains experimental
and does not alter provider-v2 observation identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_json import frozen_finite_json_mapping, plain_json
from .causal_tracklets import CausalTrackletSet
from .cross_window_tracklets import (
    CrossWindowAssociationCandidate,
    CrossWindowAssociationConfig,
    CrossWindowAssociationLink,
    CrossWindowAssociationResult,
    associate_cross_window_tracklets,
)
from .observation_factors import sim3_point_jacobian
from .sim3 import Sim3

FloatArray: TypeAlias = NDArray[np.floating[Any]]

TRACKLET_CONTENT_SCHEMA = "prob4d.causal-tracklet-content"
TRACKLET_ARTIFACT_SCHEMA = "prob4d.causal-tracklet-artifact"
JOINT_GAUGE_ASSOCIATION_SCHEMA = "prob4d.joint-gauge-tracklet-association"
SCHEMA_VERSION = 1
GAUGE_PARAMETERIZATION = "log-scale-rotvec-translation-v1"
DEPENDENCE_SEMANTICS: Literal["joint-cross-window-gauge-v1"] = "joint-cross-window-gauge-v1"
RANKING_SEMANTICS: Literal["isotropic-geometric-mutual-best-joint-gauge-diagnostic-v1"] = (
    "isotropic-geometric-mutual-best-joint-gauge-diagnostic-v1"
)
CONDITIONAL_POINT_CROSS_COVARIANCE_SEMANTICS: Literal["assumed-zero-unavailable-v1"] = (
    "assumed-zero-unavailable-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_RESIDUAL_DIMENSION = 3
_GAUGE_DIMENSION = 7


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _strict_digest(value: Any, *, name: str, pattern: re.Pattern[str]) -> str:
    text = _strict_string(value, name=name)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{name} has a noncanonical digest format")
    return text


def _strict_string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a canonical tuple of strings")
    result = tuple(
        _strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _strict_index(value: Any, *, name: str, upper_bound: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if result < 0 or result >= upper_bound:
        raise ValueError(f"{name} lies outside the gauge prior")
    return result


def _array_descriptor(
    value: np.ndarray,
    *,
    dtype: np.dtype[Any],
    name: str,
) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    canonical = np.ascontiguousarray(np.asarray(array, dtype=canonical_dtype))
    if canonical.dtype.kind == "f" and not np.all(np.isfinite(canonical)):
        raise ValueError(f"{name} must be finite")
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _readonly_covariance_stack(
    value: FloatArray,
    *,
    count: int,
    name: str,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    covariance = np.asarray(raw, dtype=np.float64).copy()
    if covariance.shape != (count, 3, 3):
        raise ValueError(f"{name} must have shape ({count}, 3, 3)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (covariance + covariance.swapaxes(1, 2))
    if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
    if np.any(np.min(eigenvalues, axis=1) < -1e-10 * scale):
        raise ValueError(f"{name} must be positive semidefinite")
    symmetric.setflags(write=False)
    return symmetric


def _readonly_joint_gauge_covariance(
    value: FloatArray,
    *,
    gauge_count: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("joint_gauge_covariance must contain real numeric values")
    covariance = np.asarray(raw, dtype=np.float64).copy()
    dimension = _GAUGE_DIMENSION * gauge_count
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"joint_gauge_covariance must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("joint_gauge_covariance must be finite")
    symmetric = 0.5 * (covariance + covariance.T)
    if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError("joint_gauge_covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("joint_gauge_covariance must be positive semidefinite")
    symmetric.setflags(write=False)
    return symmetric


def _point_matrix(value: FloatArray, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError(f"{name} must have nonempty shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def tracklet_content_descriptor(tracklets: CausalTrackletSet) -> dict[str, object]:
    """Return the canonical content descriptor for one validated tracklet set."""

    if not isinstance(tracklets, CausalTrackletSet):
        raise TypeError("tracklets must be a CausalTrackletSet")
    if not isinstance(tracklets.window_id, str) or not tracklets.window_id:
        raise ValueError("tracklet window_id must be a nonempty string")
    for name, value in (
        ("causal_frame_stop", tracklets.causal_frame_stop),
        ("seed_frame_index", tracklets.seed_frame_index),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"tracklet {name} must be a genuine integer")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in tracklets.source_shape
    ):
        raise TypeError("tracklet source_shape must contain genuine integers")
    return {
        "schema_name": TRACKLET_CONTENT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "window_id": tracklets.window_id,
        "causal_frame_stop": tracklets.causal_frame_stop,
        "source_shape": list(tracklets.source_shape),
        "seed_frame_index": tracklets.seed_frame_index,
        "track_count": tracklets.track_count,
        "observation_count": tracklets.observation_count,
        "arrays": {
            "track_ids": _array_descriptor(
                tracklets.track_ids,
                dtype=np.dtype(np.int64),
                name="track_ids",
            ),
            "frame_indices": _array_descriptor(
                tracklets.frame_indices,
                dtype=np.dtype(np.int64),
                name="frame_indices",
            ),
            "local_frame_indices": _array_descriptor(
                tracklets.local_frame_indices,
                dtype=np.dtype(np.int64),
                name="local_frame_indices",
            ),
            "rows": _array_descriptor(
                tracklets.rows,
                dtype=np.dtype(np.int64),
                name="rows",
            ),
            "columns": _array_descriptor(
                tracklets.columns,
                dtype=np.dtype(np.int64),
                name="columns",
            ),
            "points_local": _array_descriptor(
                tracklets.points_local,
                dtype=np.dtype(np.float64),
                name="points_local",
            ),
            "link_probability": _array_descriptor(
                tracklets.link_probability,
                dtype=np.dtype(np.float64),
                name="link_probability",
            ),
            "association_probability": _array_descriptor(
                tracklets.association_probability,
                dtype=np.dtype(np.float64),
                name="association_probability",
            ),
        },
        "metadata": plain_json(tracklets.metadata),
    }


def tracklet_content_id(tracklets: CausalTrackletSet) -> str:
    """Return a portable SHA-256 identity for a complete tracklet set."""

    return _content_id(tracklet_content_descriptor(tracklets))


def _sim3_id(transform: Sim3) -> str:
    if not isinstance(transform, Sim3):
        raise TypeError("gauge transforms must be Sim3 instances")
    descriptor = {
        "parameterization": GAUGE_PARAMETERIZATION,
        "vector": _array_descriptor(
            transform.as_vector(),
            dtype=np.dtype(np.float64),
            name="Sim3 vector",
        ),
    }
    return _content_id(descriptor)


def _covariance_stack_id(value: FloatArray, *, name: str) -> str:
    return _content_id(
        {
            "semantics": "tracklet-local-conditional-point-covariance-m2-v1",
            "array": _array_descriptor(
                value,
                dtype=np.dtype(np.float64),
                name=name,
            ),
        }
    )


def _joint_gauge_prior_id(gauge_ids: tuple[str, ...], covariance: FloatArray) -> str:
    return _content_id(
        {
            "parameterization": GAUGE_PARAMETERIZATION,
            "semantics": "joint-cross-window",
            "gauge_ids": list(gauge_ids),
            "covariance": _array_descriptor(
                covariance,
                dtype=np.dtype(np.float64),
                name="joint_gauge_covariance",
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class CausalTrackletArtifactV1:
    """A complete tracklet set bound to producer configuration and lineage."""

    tracklets: CausalTrackletSet
    prediction_manifest_id: str
    source_revision: str
    builder_configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tracklets, CausalTrackletSet):
            raise TypeError("tracklets must be a CausalTrackletSet")
        manifest_id = _strict_digest(
            self.prediction_manifest_id,
            name="prediction_manifest_id",
            pattern=_SHA256,
        )
        source_revision = _strict_digest(
            self.source_revision,
            name="source_revision",
            pattern=_GIT_SHA,
        )
        configuration = frozen_finite_json_mapping(
            self.builder_configuration,
            name="builder_configuration",
        )
        object.__setattr__(self, "prediction_manifest_id", manifest_id)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "builder_configuration", configuration)

    @property
    def tracklet_set_id(self) -> str:
        return tracklet_content_id(self.tracklets)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": TRACKLET_ARTIFACT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "tracklet_set_id": self.tracklet_set_id,
            "tracklet_set": tracklet_content_descriptor(self.tracklets),
            "prediction_manifest_id": self.prediction_manifest_id,
            "source_revision": self.source_revision,
            "builder_configuration": plain_json(self.builder_configuration),
        }

    @property
    def artifact_id(self) -> str:
        return _content_id(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["artifact_id"] = self.artifact_id
        return result


@dataclass(frozen=True, slots=True)
class JointGaugeCrossWindowAssociationEvidenceV1:
    """Association result plus complete immutable provenance identities."""

    association: CrossWindowAssociationResult
    left_tracklet_artifact_id: str
    right_tracklet_artifact_id: str
    left_gauge_id: str
    right_gauge_id: str
    gauge_ids: tuple[str, ...]
    left_gauge_transform_id: str
    right_gauge_transform_id: str
    joint_gauge_prior_id: str
    left_conditional_covariance_id: str
    right_conditional_covariance_id: str
    tracklet_producer_revision: str
    association_revision: str
    dependence_semantics: Literal["joint-cross-window-gauge-v1"] = DEPENDENCE_SEMANTICS
    ranking_semantics: Literal["isotropic-geometric-mutual-best-joint-gauge-diagnostic-v1"] = (
        RANKING_SEMANTICS
    )
    conditional_point_cross_covariance_semantics: Literal["assumed-zero-unavailable-v1"] = (
        CONDITIONAL_POINT_CROSS_COVARIANCE_SEMANTICS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.association, CrossWindowAssociationResult):
            raise TypeError("association must be a CrossWindowAssociationResult")
        for name in (
            "left_tracklet_artifact_id",
            "right_tracklet_artifact_id",
            "left_gauge_transform_id",
            "right_gauge_transform_id",
            "joint_gauge_prior_id",
            "left_conditional_covariance_id",
            "right_conditional_covariance_id",
        ):
            object.__setattr__(
                self,
                name,
                _strict_digest(getattr(self, name), name=name, pattern=_SHA256),
            )
        tracklet_producer_revision = _strict_digest(
            self.tracklet_producer_revision,
            name="tracklet_producer_revision",
            pattern=_GIT_SHA,
        )
        association_revision = _strict_digest(
            self.association_revision,
            name="association_revision",
            pattern=_GIT_SHA,
        )
        gauge_ids = _strict_string_tuple(self.gauge_ids, name="gauge_ids")
        left_gauge_id = _strict_string(self.left_gauge_id, name="left_gauge_id")
        right_gauge_id = _strict_string(self.right_gauge_id, name="right_gauge_id")
        if left_gauge_id == right_gauge_id:
            raise ValueError("left and right gauge IDs must differ")
        if left_gauge_id not in gauge_ids or right_gauge_id not in gauge_ids:
            raise ValueError("association gauge IDs must be present in gauge_ids")
        if self.left_tracklet_artifact_id == self.right_tracklet_artifact_id:
            raise ValueError("cross-window evidence requires distinct tracklet artifacts")
        if self.dependence_semantics != DEPENDENCE_SEMANTICS:
            raise ValueError("unsupported dependence_semantics")
        if self.ranking_semantics != RANKING_SEMANTICS:
            raise ValueError("unsupported ranking_semantics")
        if (
            self.conditional_point_cross_covariance_semantics
            != CONDITIONAL_POINT_CROSS_COVARIANCE_SEMANTICS
        ):
            raise ValueError("unsupported conditional_point_cross_covariance_semantics")
        object.__setattr__(
            self,
            "tracklet_producer_revision",
            tracklet_producer_revision,
        )
        object.__setattr__(self, "association_revision", association_revision)
        object.__setattr__(self, "gauge_ids", gauge_ids)
        object.__setattr__(self, "left_gauge_id", left_gauge_id)
        object.__setattr__(self, "right_gauge_id", right_gauge_id)

    @property
    def accepted_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.association.accepted_pairs

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": JOINT_GAUGE_ASSOCIATION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dependence_semantics": self.dependence_semantics,
            "left_tracklet_artifact_id": self.left_tracklet_artifact_id,
            "right_tracklet_artifact_id": self.right_tracklet_artifact_id,
            "left_gauge_id": self.left_gauge_id,
            "right_gauge_id": self.right_gauge_id,
            "gauge_ids": list(self.gauge_ids),
            "left_gauge_transform_id": self.left_gauge_transform_id,
            "right_gauge_transform_id": self.right_gauge_transform_id,
            "joint_gauge_prior_id": self.joint_gauge_prior_id,
            "left_conditional_covariance_id": self.left_conditional_covariance_id,
            "right_conditional_covariance_id": self.right_conditional_covariance_id,
            "tracklet_producer_revision": self.tracklet_producer_revision,
            "association_revision": self.association_revision,
            "ranking_semantics": self.ranking_semantics,
            "conditional_point_cross_covariance_semantics": (
                self.conditional_point_cross_covariance_semantics
            ),
            "association": self.association.to_dict(),
            "claim_boundary": (
                "Source-only cross-window identity evidence. Joint gauge covariance "
                "is retained as a normalized-residual diagnostic but cannot improve "
                "mutual-best rank merely by becoming wider. Conditional point "
                "cross-window covariance is unavailable and assumed zero. This result "
                "does not rewrite provider-v2 point IDs or establish downstream "
                "physical benefit."
            ),
        }

    @property
    def result_id(self) -> str:
        return _content_id(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["result_id"] = self.result_id
        return result


def _joint_gauge_residual_covariance_validated(
    left_points_local_m: FloatArray,
    right_points_local_m: FloatArray,
    *,
    left_global_from_local: Sim3,
    right_global_from_local: Sim3,
    left_conditional_local_covariance_m2: FloatArray,
    right_conditional_local_covariance_m2: FloatArray,
    joint_gauge_covariance: FloatArray,
    left_gauge_index: int,
    right_gauge_index: int,
) -> FloatArray:
    left_jacobian = sim3_point_jacobian(left_global_from_local, left_points_local_m)
    right_jacobian = sim3_point_jacobian(right_global_from_local, right_points_local_m)
    left_start = _GAUGE_DIMENSION * left_gauge_index
    right_start = _GAUGE_DIMENSION * right_gauge_index
    left_slice = slice(left_start, left_start + _GAUGE_DIMENSION)
    right_slice = slice(right_start, right_start + _GAUGE_DIMENSION)
    p_left_left = joint_gauge_covariance[left_slice, left_slice]
    p_right_right = joint_gauge_covariance[right_slice, right_slice]
    p_left_right = joint_gauge_covariance[left_slice, right_slice]
    p_right_left = joint_gauge_covariance[right_slice, left_slice]

    left_global_covariance = left_global_from_local.transform_covariances(
        left_conditional_local_covariance_m2
    )
    right_global_covariance = right_global_from_local.transform_covariances(
        right_conditional_local_covariance_m2
    )
    left_marginal = np.einsum(
        "nia,ab,njb->nij",
        left_jacobian,
        p_left_left,
        left_jacobian,
        optimize=True,
    )
    right_marginal = np.einsum(
        "nia,ab,njb->nij",
        right_jacobian,
        p_right_right,
        right_jacobian,
        optimize=True,
    )
    left_right = np.einsum(
        "nia,ab,njb->nij",
        left_jacobian,
        p_left_right,
        right_jacobian,
        optimize=True,
    )
    right_left = np.einsum(
        "nia,ab,njb->nij",
        right_jacobian,
        p_right_left,
        left_jacobian,
        optimize=True,
    )
    covariance = (
        left_global_covariance
        + right_global_covariance
        + left_marginal
        + right_marginal
        - left_right
        - right_left
    )
    symmetric = 0.5 * (covariance + covariance.swapaxes(1, 2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
    if np.any(np.min(eigenvalues, axis=1) < -1e-9 * scale):
        raise ValueError("joint gauge residual covariance is materially indefinite")
    symmetric.setflags(write=False)
    return symmetric


def joint_gauge_residual_covariance_m2(
    left_points_local_m: FloatArray,
    right_points_local_m: FloatArray,
    *,
    left_global_from_local: Sim3,
    right_global_from_local: Sim3,
    left_conditional_local_covariance_m2: FloatArray,
    right_conditional_local_covariance_m2: FloatArray,
    joint_gauge_covariance: FloatArray,
    left_gauge_index: int,
    right_gauge_index: int,
) -> FloatArray:
    """Return covariance of ``left_world - right_world`` with gauge cross terms."""

    if not isinstance(left_global_from_local, Sim3) or not isinstance(
        right_global_from_local, Sim3
    ):
        raise TypeError("gauge transforms must be Sim3 instances")
    left_points = _point_matrix(left_points_local_m, name="left_points_local_m")
    right_points = _point_matrix(right_points_local_m, name="right_points_local_m")
    if right_points.shape != left_points.shape:
        raise ValueError("left and right point arrays must have the same shape")
    count = len(left_points)
    left_covariance = _readonly_covariance_stack(
        left_conditional_local_covariance_m2,
        count=count,
        name="left_conditional_local_covariance_m2",
    )
    right_covariance = _readonly_covariance_stack(
        right_conditional_local_covariance_m2,
        count=count,
        name="right_conditional_local_covariance_m2",
    )
    raw_joint = np.asarray(joint_gauge_covariance)
    if raw_joint.ndim != 2 or raw_joint.shape[0] != raw_joint.shape[1]:
        raise ValueError("joint_gauge_covariance must be square")
    if raw_joint.shape[0] % _GAUGE_DIMENSION:
        raise ValueError("joint_gauge_covariance dimension must be divisible by seven")
    gauge_count = raw_joint.shape[0] // _GAUGE_DIMENSION
    joint_covariance = _readonly_joint_gauge_covariance(
        joint_gauge_covariance,
        gauge_count=gauge_count,
    )
    left_index = _strict_index(
        left_gauge_index,
        name="left_gauge_index",
        upper_bound=gauge_count,
    )
    right_index = _strict_index(
        right_gauge_index,
        name="right_gauge_index",
        upper_bound=gauge_count,
    )
    if left_index == right_index:
        raise ValueError("left and right gauge indices must differ")
    return _joint_gauge_residual_covariance_validated(
        left_points,
        right_points,
        left_global_from_local=left_global_from_local,
        right_global_from_local=right_global_from_local,
        left_conditional_local_covariance_m2=left_covariance,
        right_conditional_local_covariance_m2=right_covariance,
        joint_gauge_covariance=joint_covariance,
        left_gauge_index=left_index,
        right_gauge_index=right_index,
    )


def _track_frame_rows(tracklets: CausalTrackletSet) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for row, (track_id, frame_index) in enumerate(
        zip(tracklets.track_ids, tracklets.frame_indices, strict=True)
    ):
        result.setdefault(int(track_id), {})[int(frame_index)] = row
    return result


def _normalized_square(
    residual: FloatArray,
    covariance: FloatArray,
    *,
    covariance_floor_m2: float,
) -> float:
    stabilized = covariance + covariance_floor_m2 * np.eye(_RESIDUAL_DIMENSION)
    eigenvalues, eigenvectors = np.linalg.eigh(stabilized)
    eigenvalues = np.maximum(eigenvalues, covariance_floor_m2)
    coordinates = eigenvectors.T @ residual
    mahalanobis_square = float(np.sum(coordinates**2 / eigenvalues))
    return mahalanobis_square / _RESIDUAL_DIMENSION


def _candidate_rank(
    candidate: CrossWindowAssociationCandidate,
    *,
    side: Literal["left", "right"],
) -> tuple[float, float, float, int]:
    other_id = candidate.right_track_id if side == "left" else candidate.left_track_id
    return (
        -candidate.compatibility_score,
        candidate.weighted_rms_m,
        -candidate.effective_support,
        other_id,
    )


def _best_by_side(
    candidates: tuple[CrossWindowAssociationCandidate, ...],
    *,
    side: Literal["left", "right"],
) -> tuple[
    dict[int, CrossWindowAssociationCandidate],
    dict[tuple[int, int], float],
]:
    grouped: dict[int, list[CrossWindowAssociationCandidate]] = {}
    for candidate in candidates:
        track_id = candidate.left_track_id if side == "left" else candidate.right_track_id
        grouped.setdefault(track_id, []).append(candidate)
    best: dict[int, CrossWindowAssociationCandidate] = {}
    margins: dict[tuple[int, int], float] = {}
    for track_id, group in grouped.items():
        ordered = sorted(group, key=lambda item: _candidate_rank(item, side=side))
        selected = ordered[0]
        second = ordered[1].compatibility_score if len(ordered) > 1 else 0.0
        best[track_id] = selected
        margins[(selected.left_track_id, selected.right_track_id)] = max(
            0.0,
            selected.compatibility_score - second,
        )
    return best, margins


def _admit_candidates(
    base: CrossWindowAssociationResult,
    candidates: tuple[CrossWindowAssociationCandidate, ...],
    *,
    left_track_count: int,
    right_track_count: int,
) -> CrossWindowAssociationResult:
    config = base.configuration
    left_best, left_margins = _best_by_side(candidates, side="left")
    right_best, right_margins = _best_by_side(candidates, side="right")
    links: list[CrossWindowAssociationLink] = []
    non_mutual = 0
    ambiguous = 0
    threshold_rejected = 0
    for left_track_id, candidate in sorted(left_best.items()):
        right_candidate = right_best.get(candidate.right_track_id)
        if right_candidate is None or right_candidate.left_track_id != left_track_id:
            non_mutual += 1
            continue
        pair = (candidate.left_track_id, candidate.right_track_id)
        left_margin = left_margins[pair]
        right_margin = right_margins[pair]
        if left_margin < config.minimum_score_margin or right_margin < config.minimum_score_margin:
            ambiguous += 1
            continue
        if (
            candidate.effective_support < config.minimum_effective_support
            or candidate.weighted_rms_m > config.maximum_weighted_rms_m
            or candidate.compatibility_score < config.minimum_compatibility_score
        ):
            threshold_rejected += 1
            continue
        links.append(
            CrossWindowAssociationLink(
                left_track_id=candidate.left_track_id,
                right_track_id=candidate.right_track_id,
                shared_frame_indices=candidate.shared_frame_indices,
                compatibility_score=candidate.compatibility_score,
                left_score_margin=left_margin,
                right_score_margin=right_margin,
            )
        )
    link_tuple = tuple(links)
    linked_left = {link.left_track_id for link in link_tuple}
    linked_right = {link.right_track_id for link in link_tuple}
    return CrossWindowAssociationResult(
        left_window_id=base.left_window_id,
        right_window_id=base.right_window_id,
        causal_frame_stop=base.causal_frame_stop,
        configuration=config,
        candidates=candidates,
        links=link_tuple,
        unmatched_left_track_ids=tuple(
            track_id for track_id in range(left_track_count) if track_id not in linked_left
        ),
        unmatched_right_track_ids=tuple(
            track_id for track_id in range(right_track_count) if track_id not in linked_right
        ),
        possible_track_pair_count=base.possible_track_pair_count,
        spatial_candidate_pair_count=base.spatial_candidate_pair_count,
        spatially_rejected_pair_count=base.spatially_rejected_pair_count,
        evaluated_track_pair_count=base.evaluated_track_pair_count,
        shared_gate_frame_count=base.shared_gate_frame_count,
        insufficient_shared_frame_pair_count=base.insufficient_shared_frame_pair_count,
        zero_support_pair_count=base.zero_support_pair_count,
        low_support_pair_count=base.low_support_pair_count,
        non_mutual_best_count=non_mutual,
        ambiguous_mutual_best_count=ambiguous,
        threshold_rejected_mutual_best_count=threshold_rejected,
    )


def associate_cross_window_tracklets_joint_gauge(
    left: CausalTrackletArtifactV1,
    right: CausalTrackletArtifactV1,
    *,
    left_global_from_local: Sim3,
    right_global_from_local: Sim3,
    left_conditional_local_covariance_m2: FloatArray,
    right_conditional_local_covariance_m2: FloatArray,
    gauge_ids: tuple[str, ...],
    joint_gauge_covariance: FloatArray,
    left_gauge_id: str,
    right_gauge_id: str,
    association_revision: str,
    configuration: CrossWindowAssociationConfig | None = None,
    candidate_chunk_size: int = 256,
) -> JointGaugeCrossWindowAssociationEvidenceV1:
    """Associate two tracklet artifacts with full joint gauge dependence."""

    if not isinstance(left, CausalTrackletArtifactV1) or not isinstance(
        right, CausalTrackletArtifactV1
    ):
        raise TypeError("left and right must be CausalTrackletArtifactV1 instances")
    if left.source_revision != right.source_revision:
        raise ValueError("tracklet artifacts must share the exact source revision")
    association_revision = _strict_digest(
        association_revision,
        name="association_revision",
        pattern=_GIT_SHA,
    )
    left_gauge_id = _strict_string(left_gauge_id, name="left_gauge_id")
    right_gauge_id = _strict_string(right_gauge_id, name="right_gauge_id")
    if left_gauge_id != left.tracklets.window_id:
        raise ValueError("left_gauge_id must equal the left tracklet window_id")
    if right_gauge_id != right.tracklets.window_id:
        raise ValueError("right_gauge_id must equal the right tracklet window_id")
    ordered_gauge_ids = _strict_string_tuple(gauge_ids, name="gauge_ids")
    if left_gauge_id == right_gauge_id:
        raise ValueError("left and right gauge IDs must differ")
    if left_gauge_id not in ordered_gauge_ids or right_gauge_id not in ordered_gauge_ids:
        raise ValueError("left and right gauge IDs must be present in gauge_ids")
    joint_covariance = _readonly_joint_gauge_covariance(
        joint_gauge_covariance,
        gauge_count=len(ordered_gauge_ids),
    )
    left_covariance = _readonly_covariance_stack(
        left_conditional_local_covariance_m2,
        count=left.tracklets.observation_count,
        name="left_conditional_local_covariance_m2",
    )
    right_covariance = _readonly_covariance_stack(
        right_conditional_local_covariance_m2,
        count=right.tracklets.observation_count,
        name="right_conditional_local_covariance_m2",
    )
    config = configuration or CrossWindowAssociationConfig()
    if not isinstance(config, CrossWindowAssociationConfig):
        raise TypeError("configuration must be CrossWindowAssociationConfig")
    base = associate_cross_window_tracklets(
        left.tracklets,
        right.tracklets,
        left_global_from_local=left_global_from_local,
        right_global_from_local=right_global_from_local,
        configuration=config,
        candidate_chunk_size=candidate_chunk_size,
    )
    left_rows = _track_frame_rows(left.tracklets)
    right_rows = _track_frame_rows(right.tracklets)
    left_index = ordered_gauge_ids.index(left_gauge_id)
    right_index = ordered_gauge_ids.index(right_gauge_id)
    candidates: list[CrossWindowAssociationCandidate] = []
    for candidate in base.candidates:
        shared_frames = candidate.shared_frame_indices
        left_indices = np.asarray(
            [left_rows[candidate.left_track_id][frame] for frame in shared_frames],
            dtype=np.int64,
        )
        right_indices = np.asarray(
            [right_rows[candidate.right_track_id][frame] for frame in shared_frames],
            dtype=np.int64,
        )
        left_points = left.tracklets.points_local[left_indices]
        right_points = right.tracklets.points_local[right_indices]
        residuals = left_global_from_local.transform_points(left_points) - (
            right_global_from_local.transform_points(right_points)
        )
        residual_covariance = _joint_gauge_residual_covariance_validated(
            left_points,
            right_points,
            left_global_from_local=left_global_from_local,
            right_global_from_local=right_global_from_local,
            left_conditional_local_covariance_m2=left_covariance[left_indices],
            right_conditional_local_covariance_m2=right_covariance[right_indices],
            joint_gauge_covariance=joint_covariance,
            left_gauge_index=left_index,
            right_gauge_index=right_index,
        )
        weights = (
            left.tracklets.association_probability[left_indices]
            * right.tracklets.association_probability[right_indices]
        )
        support = float(np.sum(weights))
        normalized_squares = np.asarray(
            [
                _normalized_square(
                    residual,
                    covariance,
                    covariance_floor_m2=config.covariance_floor_m2,
                )
                for residual, covariance in zip(
                    residuals,
                    residual_covariance,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        normalized_rms = float(np.sqrt(np.sum(weights * normalized_squares) / support))
        # Joint covariance is an auditable residual diagnostic, not a way to
        # improve mutual-best rank by becoming less informative. Until a separate
        # source-calibrated gate is registered, preserve the covariance-independent
        # geometric compatibility score from the bounded base candidate generator.
        score = candidate.compatibility_score
        candidates.append(
            CrossWindowAssociationCandidate(
                left_track_id=candidate.left_track_id,
                right_track_id=candidate.right_track_id,
                shared_frame_indices=shared_frames,
                effective_support=support,
                weighted_rms_m=candidate.weighted_rms_m,
                maximum_distance_m=candidate.maximum_distance_m,
                normalized_rms=normalized_rms,
                compatibility_score=score,
                used_covariance=True,
            )
        )
    candidate_tuple = tuple(
        sorted(candidates, key=lambda item: (item.left_track_id, item.right_track_id))
    )
    association = _admit_candidates(
        base,
        candidate_tuple,
        left_track_count=left.tracklets.track_count,
        right_track_count=right.tracklets.track_count,
    )
    return JointGaugeCrossWindowAssociationEvidenceV1(
        association=association,
        left_tracklet_artifact_id=left.artifact_id,
        right_tracklet_artifact_id=right.artifact_id,
        left_gauge_id=left_gauge_id,
        right_gauge_id=right_gauge_id,
        gauge_ids=ordered_gauge_ids,
        left_gauge_transform_id=_sim3_id(left_global_from_local),
        right_gauge_transform_id=_sim3_id(right_global_from_local),
        joint_gauge_prior_id=_joint_gauge_prior_id(
            ordered_gauge_ids,
            joint_covariance,
        ),
        left_conditional_covariance_id=_covariance_stack_id(
            left_covariance,
            name="left_conditional_local_covariance_m2",
        ),
        right_conditional_covariance_id=_covariance_stack_id(
            right_covariance,
            name="right_conditional_local_covariance_m2",
        ),
        tracklet_producer_revision=left.source_revision,
        association_revision=association_revision,
    )


__all__ = [
    "CONDITIONAL_POINT_CROSS_COVARIANCE_SEMANTICS",
    "CausalTrackletArtifactV1",
    "DEPENDENCE_SEMANTICS",
    "JOINT_GAUGE_ASSOCIATION_SCHEMA",
    "JointGaugeCrossWindowAssociationEvidenceV1",
    "RANKING_SEMANTICS",
    "SCHEMA_VERSION",
    "TRACKLET_ARTIFACT_SCHEMA",
    "TRACKLET_CONTENT_SCHEMA",
    "associate_cross_window_tracklets_joint_gauge",
    "joint_gauge_residual_covariance_m2",
    "tracklet_content_descriptor",
    "tracklet_content_id",
]
