"""Unfused observation-factor contracts for downstream Bayesian estimators.

The ordinary Prob4D fusion products intentionally collapse overlapping windows
into one trajectory.  This module provides an additional lossless interface for
consumers that need to keep window gauges, view provenance, and correlation
structure explicit.  Gauge covariance is propagated with a first-order Sim(3)
linearization, but the original local observations remain available in the
versioned bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .gauge import GaugeEstimate
from .sim3 import Sim3, skew, so3_right_jacobian

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]

OBSERVATION_FACTOR_SCHEMA = "prob4d.observation-factor-bundle"
OBSERVATION_FACTOR_SCHEMA_VERSION = 2
GAUGE_PARAMETERIZATION = "log-scale-rotvec-translation-v1"


def _readonly(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _require_psd(value: np.ndarray, name: str, *, tolerance: float = 1e-12) -> None:
    symmetric = 0.5 * (value + value.swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    try:
        json.dumps(copied, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("bundle metadata must be finite JSON data") from error
    return copied


@dataclass(frozen=True)
class ObservationFactor:
    """One unfused set of associated 3-D observations in a local window gauge."""

    factor_id: str
    frame_index: int
    view_id: str
    window_id: str
    gauge_id: str
    point_ids: IntArray
    points_local_m: FloatArray
    valid_mask: BoolArray
    local_covariance_m2: FloatArray
    association_probability: FloatArray
    correlation_group_id: str
    causal_frame_limit: int
    ray_directions_local: FloatArray | None = None

    def __post_init__(self) -> None:
        identifiers = {
            "factor_id": self.factor_id,
            "view_id": self.view_id,
            "window_id": self.window_id,
            "gauge_id": self.gauge_id,
            "correlation_group_id": self.correlation_group_id,
        }
        for name, value in identifiers.items():
            if not str(value):
                raise ValueError(f"{name} must not be empty")
        if int(self.frame_index) < 0:
            raise ValueError("frame_index must be non-negative")
        if int(self.causal_frame_limit) < int(self.frame_index):
            raise ValueError("observation factor crosses its causal frame limit")

        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        points = np.asarray(self.points_local_m, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        covariance = np.asarray(self.local_covariance_m2, dtype=np.float64)
        association = np.asarray(self.association_probability, dtype=np.float64)
        if point_ids.ndim != 1 or len(point_ids) == 0:
            raise ValueError("point_ids must be a non-empty vector")
        if len(np.unique(point_ids)) != len(point_ids):
            raise ValueError("point_ids must be unique within a factor")
        if points.shape != (len(point_ids), 3):
            raise ValueError("points_local_m must have shape (N, 3)")
        if valid.shape != (len(point_ids),):
            raise ValueError("valid_mask must have shape (N,)")
        if covariance.shape != (len(point_ids), 3, 3):
            raise ValueError("local_covariance_m2 must have shape (N, 3, 3)")
        if association.shape != (len(point_ids),):
            raise ValueError("association_probability must have shape (N,)")
        if not np.all(np.isfinite(association)) or np.any(
            (association < 0.0) | (association > 1.0)
        ):
            raise ValueError("association_probability must lie in [0, 1]")
        active = valid & (association > 0.0)
        if not np.all(np.isfinite(points[active])):
            raise ValueError("active local observations must be finite")
        if not np.all(np.isfinite(covariance[active])):
            raise ValueError("active local covariances must be finite")
        if np.any(active):
            if not np.allclose(
                covariance[active], covariance[active].swapaxes(1, 2), atol=1e-12
            ):
                raise ValueError("local covariances must be symmetric")
            _require_psd(covariance[active], "local covariances")

        rays: np.ndarray | None = None
        if self.ray_directions_local is not None:
            rays = np.asarray(self.ray_directions_local, dtype=np.float64).copy()
            if rays.shape != points.shape:
                raise ValueError("ray_directions_local must have shape (N, 3)")
            if not np.all(np.isfinite(rays[active])):
                raise ValueError("active ray directions must be finite")
            norms = np.linalg.norm(rays, axis=1)
            if np.any(active & (norms <= np.finfo(np.float64).eps)):
                raise ValueError("active ray directions must be nonzero")
            normalize = norms > np.finfo(np.float64).eps
            rays[normalize] /= norms[normalize, None]
            rays.setflags(write=False)

        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "causal_frame_limit", int(self.causal_frame_limit))
        object.__setattr__(self, "point_ids", _readonly(point_ids))
        object.__setattr__(self, "points_local_m", _readonly(points))
        object.__setattr__(self, "valid_mask", _readonly(valid))
        object.__setattr__(self, "local_covariance_m2", _readonly(covariance))
        object.__setattr__(self, "association_probability", _readonly(association))
        object.__setattr__(self, "ray_directions_local", rays)


@dataclass(frozen=True)
class LinearizedObservationFactor:
    """World-frame moments and gauge Jacobians for one observation factor."""

    factor_id: str
    frame_index: int
    view_id: str
    window_id: str
    gauge_id: str
    correlation_group_id: str
    point_ids: IntArray
    world_mean_m: FloatArray
    conditional_world_covariance_m2: FloatArray
    marginal_world_covariance_m2: FloatArray
    gauge_jacobian: FloatArray
    valid_mask: BoolArray
    association_probability: FloatArray
    ray_directions_world: FloatArray | None = None

    def __post_init__(self) -> None:
        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        mean = np.asarray(self.world_mean_m, dtype=np.float64)
        conditional_covariance = np.asarray(
            self.conditional_world_covariance_m2, dtype=np.float64
        )
        marginal_covariance = np.asarray(
            self.marginal_world_covariance_m2, dtype=np.float64
        )
        jacobian = np.asarray(self.gauge_jacobian, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        probability = np.asarray(self.association_probability, dtype=np.float64)
        count = len(point_ids)
        if mean.shape != (count, 3):
            raise ValueError("world_mean_m must have shape (N, 3)")
        if conditional_covariance.shape != (count, 3, 3):
            raise ValueError(
                "conditional_world_covariance_m2 must have shape (N, 3, 3)"
            )
        if marginal_covariance.shape != (count, 3, 3):
            raise ValueError(
                "marginal_world_covariance_m2 must have shape (N, 3, 3)"
            )
        if jacobian.shape != (count, 3, 7):
            raise ValueError("gauge_jacobian must have shape (N, 3, 7)")
        if valid.shape != (count,) or probability.shape != (count,):
            raise ValueError("linearized factor masks have changed shape")
        rays = None
        if self.ray_directions_world is not None:
            rays = np.asarray(self.ray_directions_world, dtype=np.float64)
            if rays.shape != (count, 3):
                raise ValueError("ray_directions_world must have shape (N, 3)")
            rays = _readonly(rays)
        for name, value in (
            ("point_ids", point_ids),
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional_covariance),
            ("marginal_world_covariance_m2", marginal_covariance),
            ("gauge_jacobian", jacobian),
            ("valid_mask", valid),
            ("association_probability", probability),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "ray_directions_world", rays)


@dataclass(frozen=True)
class StackedObservationFactors:
    """Flattened factor rows with block-structured gauge nuisance parameters."""

    world_mean_m: FloatArray
    conditional_world_covariance_m2: FloatArray
    marginal_world_covariance_m2: FloatArray
    gauge_jacobian: FloatArray
    gauge_prior_covariance: FloatArray
    association_probability: FloatArray
    point_ids: IntArray
    frame_indices: IntArray
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    gauge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = np.asarray(self.world_mean_m, dtype=np.float64)
        conditional_covariance = np.asarray(
            self.conditional_world_covariance_m2, dtype=np.float64
        )
        marginal_covariance = np.asarray(
            self.marginal_world_covariance_m2, dtype=np.float64
        )
        jacobian = np.asarray(self.gauge_jacobian, dtype=np.float64)
        gauge_prior = np.asarray(self.gauge_prior_covariance, dtype=np.float64)
        probability = np.asarray(self.association_probability, dtype=np.float64)
        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        count = len(mean)
        gauge_dimension = 7 * len(self.gauge_ids)
        if mean.shape != (count, 3):
            raise ValueError("stacked world means must have shape (M, 3)")
        if conditional_covariance.shape != (count, 3, 3):
            raise ValueError(
                "stacked conditional covariance must have shape (M, 3, 3)"
            )
        if marginal_covariance.shape != (count, 3, 3):
            raise ValueError(
                "stacked marginal covariance must have shape (M, 3, 3)"
            )
        if jacobian.shape != (count, 3, gauge_dimension):
            raise ValueError("stacked gauge Jacobian has changed shape")
        if gauge_prior.shape != (gauge_dimension, gauge_dimension):
            raise ValueError("gauge prior covariance has changed shape")
        if probability.shape != (count,):
            raise ValueError("stacked association_probability has changed shape")
        if point_ids.shape != (count,) or frame_indices.shape != (count,):
            raise ValueError("stacked integer metadata has changed shape")
        for values in (self.view_ids, self.factor_ids, self.correlation_group_ids):
            if len(values) != count:
                raise ValueError("stacked string metadata has changed length")
        for name, value in (
            ("world_mean_m", mean),
            ("conditional_world_covariance_m2", conditional_covariance),
            ("marginal_world_covariance_m2", marginal_covariance),
            ("gauge_jacobian", jacobian),
            ("gauge_prior_covariance", gauge_prior),
            ("association_probability", probability),
            ("point_ids", point_ids),
            ("frame_indices", frame_indices),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "view_ids", tuple(map(str, self.view_ids)))
        object.__setattr__(self, "factor_ids", tuple(map(str, self.factor_ids)))
        object.__setattr__(
            self, "correlation_group_ids", tuple(map(str, self.correlation_group_ids))
        )
        object.__setattr__(self, "gauge_ids", tuple(map(str, self.gauge_ids)))


@dataclass(frozen=True)
class ObservationFactorBundle:
    """Versioned collection of unfused factors and uncertain Sim(3) gauges."""

    sequence_id: str
    factors: tuple[ObservationFactor, ...]
    gauges: tuple[GaugeEstimate, ...]
    source_revision: str
    causal_frame_limit: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = OBSERVATION_FACTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sequence_id or not self.source_revision:
            raise ValueError("sequence_id and source_revision must not be empty")
        if self.schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION:
            raise ValueError("unsupported observation-factor schema version")
        if int(self.causal_frame_limit) < 0:
            raise ValueError("causal_frame_limit must be non-negative")
        factors = tuple(self.factors)
        gauges = tuple(self.gauges)
        if not factors:
            raise ValueError("an observation-factor bundle must contain factors")
        factor_ids = [factor.factor_id for factor in factors]
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor IDs must be unique")
        gauge_ids = [gauge.window_id for gauge in gauges]
        if not gauges or len(set(gauge_ids)) != len(gauge_ids):
            raise ValueError("gauge IDs must be non-empty and unique")
        gauge_id_set = set(gauge_ids)
        for gauge in gauges:
            covariance = np.asarray(gauge.covariance, dtype=np.float64)
            if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
                raise ValueError("gauge covariance must be a finite 7 by 7 matrix")
            if not np.allclose(covariance, covariance.T, atol=1e-12):
                raise ValueError("gauge covariance must be symmetric")
            _require_psd(covariance, "gauge covariance")
        for factor in factors:
            if factor.gauge_id not in gauge_id_set:
                raise ValueError(f"factor {factor.factor_id!r} references an unknown gauge")
            if factor.causal_frame_limit != int(self.causal_frame_limit):
                raise ValueError("factor and bundle causal frame limits differ")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "gauges", gauges)
        object.__setattr__(self, "causal_frame_limit", int(self.causal_frame_limit))
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

    @property
    def gauge_map(self) -> dict[str, GaugeEstimate]:
        return {gauge.window_id: gauge for gauge in self.gauges}

    @property
    def correlation_group_counts(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for factor in self.factors:
            groups[factor.correlation_group_id] = (
                groups.get(factor.correlation_group_id, 0) + 1
            )
        return groups

    def linearize(self, factor: ObservationFactor | str) -> LinearizedObservationFactor:
        selected = factor
        if isinstance(factor, str):
            matches = [value for value in self.factors if value.factor_id == factor]
            if len(matches) != 1:
                raise KeyError(f"unknown observation factor {factor!r}")
            selected = matches[0]
        gauge = self.gauge_map[selected.gauge_id]
        transform = gauge.global_from_local
        mean = transform.transform_points(selected.points_local_m)
        conditional_covariance = transform.transform_covariances(
            selected.local_covariance_m2
        )
        jacobian = sim3_point_jacobian(transform, selected.points_local_m)
        gauge_covariance = np.einsum(
            "nia,ab,njb->nij",
            jacobian,
            np.asarray(gauge.covariance, dtype=np.float64),
            jacobian,
            optimize=True,
        )
        marginal_covariance = conditional_covariance + gauge_covariance
        marginal_covariance = 0.5 * (
            marginal_covariance + marginal_covariance.swapaxes(1, 2)
        )
        rays = None
        if selected.ray_directions_local is not None:
            rays = transform.rotate_directions(selected.ray_directions_local)
        return LinearizedObservationFactor(
            factor_id=selected.factor_id,
            frame_index=selected.frame_index,
            view_id=selected.view_id,
            window_id=selected.window_id,
            gauge_id=selected.gauge_id,
            correlation_group_id=selected.correlation_group_id,
            point_ids=selected.point_ids,
            world_mean_m=mean,
            conditional_world_covariance_m2=conditional_covariance,
            marginal_world_covariance_m2=marginal_covariance,
            gauge_jacobian=jacobian,
            valid_mask=selected.valid_mask,
            association_probability=selected.association_probability,
            ray_directions_world=rays,
        )

    def stack(self, *, include_invalid: bool = False) -> StackedObservationFactors:
        return stack_observation_factors(self, include_invalid=include_invalid)


def sim3_point_jacobian(transform: Sim3, points_local_m: FloatArray) -> FloatArray:
    """Linearize transformed points against ``Sim3.as_vector()`` parameters."""

    points = np.asarray(points_local_m, dtype=np.float64)
    if points.ndim == 1:
        points = points[None]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_local_m must have shape (N, 3) or (3,)")
    rotation_vector = transform.as_vector()[1:4]
    right_jacobian = so3_right_jacobian(rotation_vector)
    scaled_rotation = transform.scale * transform.rotation
    transformed_vectors = np.einsum("ij,nj->ni", scaled_rotation, points)
    jacobian = np.zeros((len(points), 3, 7), dtype=np.float64)
    jacobian[:, :, 0] = transformed_vectors
    for index, point in enumerate(points):
        jacobian[index, :, 1:4] = (
            -scaled_rotation @ skew(point) @ right_jacobian
        )
    jacobian[:, :, 4:7] = np.eye(3)[None]
    return jacobian


def _block_diagonal(values: list[np.ndarray]) -> np.ndarray:
    dimension = sum(value.shape[0] for value in values)
    result = np.zeros((dimension, dimension), dtype=np.float64)
    offset = 0
    for value in values:
        width = value.shape[0]
        result[offset : offset + width, offset : offset + width] = value
        offset += width
    return result


def stack_observation_factors(
    bundle: ObservationFactorBundle, *, include_invalid: bool = False
) -> StackedObservationFactors:
    """Stack factor rows while retaining separate seven-dimensional gauge blocks."""

    gauge_ids = tuple(gauge.window_id for gauge in bundle.gauges)
    gauge_offsets = {gauge_id: 7 * index for index, gauge_id in enumerate(gauge_ids)}
    gauge_dimension = 7 * len(gauge_ids)
    means: list[np.ndarray] = []
    conditional_covariances: list[np.ndarray] = []
    marginal_covariances: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    probabilities: list[float] = []
    point_ids: list[int] = []
    frame_indices: list[int] = []
    view_ids: list[str] = []
    factor_ids: list[str] = []
    correlation_groups: list[str] = []
    for factor in bundle.factors:
        linearized = bundle.linearize(factor)
        selected = (
            np.ones(len(factor.point_ids), dtype=bool)
            if include_invalid
            else factor.valid_mask & (factor.association_probability > 0.0)
        )
        offset = gauge_offsets[factor.gauge_id]
        for local_index in np.flatnonzero(selected):
            expanded = np.zeros((3, gauge_dimension), dtype=np.float64)
            expanded[:, offset : offset + 7] = linearized.gauge_jacobian[local_index]
            means.append(linearized.world_mean_m[local_index])
            conditional_covariances.append(
                linearized.conditional_world_covariance_m2[local_index]
            )
            marginal_covariances.append(
                linearized.marginal_world_covariance_m2[local_index]
            )
            jacobians.append(expanded)
            probabilities.append(float(factor.association_probability[local_index]))
            point_ids.append(int(factor.point_ids[local_index]))
            frame_indices.append(factor.frame_index)
            view_ids.append(factor.view_id)
            factor_ids.append(factor.factor_id)
            correlation_groups.append(factor.correlation_group_id)
    if not means:
        raise ValueError("observation-factor stack has no selected rows")
    gauge_prior = _block_diagonal(
        [np.asarray(gauge.covariance, dtype=np.float64) for gauge in bundle.gauges]
    )
    return StackedObservationFactors(
        world_mean_m=np.stack(means),
        conditional_world_covariance_m2=np.stack(conditional_covariances),
        marginal_world_covariance_m2=np.stack(marginal_covariances),
        gauge_jacobian=np.stack(jacobians),
        gauge_prior_covariance=gauge_prior,
        association_probability=np.asarray(probabilities),
        point_ids=np.asarray(point_ids),
        frame_indices=np.asarray(frame_indices),
        view_ids=tuple(view_ids),
        factor_ids=tuple(factor_ids),
        correlation_group_ids=tuple(correlation_groups),
        gauge_ids=gauge_ids,
    )


def write_observation_factor_bundle(
    bundle: ObservationFactorBundle,
    manifest_path: str | Path,
    *,
    payload_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write one checksum-bound JSON manifest and non-pickled NPZ payload."""

    manifest = Path(manifest_path)
    payload = (
        Path(payload_path)
        if payload_path is not None
        else manifest.with_suffix(".npz")
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    gauges: list[dict[str, Any]] = []
    for index, gauge in enumerate(bundle.gauges):
        prefix = f"gauge_{index:04d}"
        arrays[f"{prefix}__mean"] = gauge.global_from_local.as_vector()
        arrays[f"{prefix}__covariance"] = np.asarray(gauge.covariance)
        gauges.append(
            {
                "gauge_id": gauge.window_id,
                "mean_key": f"{prefix}__mean",
                "covariance_key": f"{prefix}__covariance",
            }
        )
    factors: list[dict[str, Any]] = []
    for index, factor in enumerate(bundle.factors):
        prefix = f"factor_{index:04d}"
        array_names = {
            "point_ids": f"{prefix}__point_ids",
            "points_local_m": f"{prefix}__points_local_m",
            "valid_mask": f"{prefix}__valid_mask",
            "local_covariance_m2": f"{prefix}__local_covariance_m2",
            "association_probability": f"{prefix}__association_probability",
        }
        arrays[array_names["point_ids"]] = factor.point_ids
        arrays[array_names["points_local_m"]] = factor.points_local_m
        arrays[array_names["valid_mask"]] = factor.valid_mask
        arrays[array_names["local_covariance_m2"]] = factor.local_covariance_m2
        arrays[array_names["association_probability"]] = factor.association_probability
        ray_key = None
        if factor.ray_directions_local is not None:
            ray_key = f"{prefix}__ray_directions_local"
            arrays[ray_key] = factor.ray_directions_local
        factors.append(
            {
                "factor_id": factor.factor_id,
                "frame_index": factor.frame_index,
                "view_id": factor.view_id,
                "window_id": factor.window_id,
                "gauge_id": factor.gauge_id,
                "correlation_group_id": factor.correlation_group_id,
                "causal_frame_limit": factor.causal_frame_limit,
                "arrays": array_names,
                "ray_directions_local_key": ray_key,
            }
        )
    np.savez_compressed(payload, **arrays)
    record = {
        "schema": OBSERVATION_FACTOR_SCHEMA,
        "schema_version": bundle.schema_version,
        "gauge_parameterization": GAUGE_PARAMETERIZATION,
        "sequence_id": bundle.sequence_id,
        "source_revision": bundle.source_revision,
        "causal_frame_limit": bundle.causal_frame_limit,
        "metadata": dict(bundle.metadata),
        "payload": {
            "path": os.path.relpath(payload, manifest.parent),
            "sha256": _sha256(payload),
            "allow_pickle": False,
        },
        "gauges": gauges,
        "factors": factors,
    }
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest, payload


def load_observation_factor_bundle(
    manifest_path: str | Path,
) -> ObservationFactorBundle:
    """Load and validate a checksum-bound observation-factor bundle."""

    manifest = Path(manifest_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("manifest is not a Prob4D observation-factor bundle")
    if record.get("schema_version") != OBSERVATION_FACTOR_SCHEMA_VERSION:
        raise ValueError("unsupported observation-factor schema version")
    if record.get("gauge_parameterization") != GAUGE_PARAMETERIZATION:
        raise ValueError("unsupported gauge parameterization")
    payload = manifest.parent / record["payload"]["path"]
    if _sha256(payload) != record["payload"]["sha256"]:
        raise ValueError("observation-factor payload checksum mismatch")
    gauges: list[GaugeEstimate] = []
    factors: list[ObservationFactor] = []
    with np.load(payload, allow_pickle=False) as arrays:
        for gauge_record in record["gauges"]:
            gauges.append(
                GaugeEstimate(
                    window_id=str(gauge_record["gauge_id"]),
                    global_from_local=Sim3.from_vector(
                        arrays[gauge_record["mean_key"]]
                    ),
                    covariance=arrays[gauge_record["covariance_key"]],
                )
            )
        for factor_record in record["factors"]:
            keys = factor_record["arrays"]
            ray_key = factor_record.get("ray_directions_local_key")
            factors.append(
                ObservationFactor(
                    factor_id=str(factor_record["factor_id"]),
                    frame_index=int(factor_record["frame_index"]),
                    view_id=str(factor_record["view_id"]),
                    window_id=str(factor_record["window_id"]),
                    gauge_id=str(factor_record["gauge_id"]),
                    point_ids=arrays[keys["point_ids"]],
                    points_local_m=arrays[keys["points_local_m"]],
                    valid_mask=arrays[keys["valid_mask"]],
                    local_covariance_m2=arrays[keys["local_covariance_m2"]],
                    association_probability=arrays[
                        keys["association_probability"]
                    ],
                    correlation_group_id=str(
                        factor_record["correlation_group_id"]
                    ),
                    causal_frame_limit=int(factor_record["causal_frame_limit"]),
                    ray_directions_local=(
                        arrays[ray_key] if ray_key is not None else None
                    ),
                )
            )
    return ObservationFactorBundle(
        sequence_id=str(record["sequence_id"]),
        factors=tuple(factors),
        gauges=tuple(gauges),
        source_revision=str(record["source_revision"]),
        causal_frame_limit=int(record["causal_frame_limit"]),
        metadata=record.get("metadata", {}),
        schema_version=int(record["schema_version"]),
    )


__all__ = [
    "GAUGE_PARAMETERIZATION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "LinearizedObservationFactor",
    "ObservationFactor",
    "ObservationFactorBundle",
    "StackedObservationFactors",
    "load_observation_factor_bundle",
    "sim3_point_jacobian",
    "stack_observation_factors",
    "write_observation_factor_bundle",
]
