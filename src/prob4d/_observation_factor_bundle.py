"""Bundles, gauge linearization, and stacking for unfused observations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._observation_factor_types import (
    FloatArray,
    LinearizedObservationFactor,
    ObservationFactor,
    StackedObservationFactors,
    _require_psd,
)
from .gauge import GaugeEstimate
from .sim3 import Sim3, skew, so3_right_jacobian

OBSERVATION_FACTOR_SCHEMA = "prob4d.observation-factor-bundle"
OBSERVATION_FACTOR_SCHEMA_VERSION = 3
LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION = 2
GAUGE_PARAMETERIZATION = "log-scale-rotvec-translation-v1"


def _json_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    try:
        json.dumps(copied, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("bundle metadata must be finite JSON data") from error
    return copied


@dataclass(frozen=True)
class ObservationFactorBundle:
    """Versioned collection of unfused factors and uncertain Sim(3) gauges."""

    sequence_id: str
    factors: tuple[ObservationFactor, ...]
    gauges: tuple[GaugeEstimate, ...]
    source_revision: str
    causal_frame_stop: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = OBSERVATION_FACTOR_SCHEMA_VERSION

    @property
    def causal_frame_limit(self) -> int:
        """Legacy inclusive alias for schema-v2 readers."""

        return self.causal_frame_stop - 1

    def __post_init__(self) -> None:
        if not self.sequence_id or not self.source_revision:
            raise ValueError("sequence_id and source_revision must not be empty")
        if self.schema_version != OBSERVATION_FACTOR_SCHEMA_VERSION:
            raise ValueError("unsupported observation-factor schema version")
        causal_frame_stop = int(self.causal_frame_stop)
        if causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be positive")
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
        group_settings: dict[str, tuple[float, float]] = {}
        for gauge in gauges:
            covariance = np.asarray(gauge.covariance, dtype=np.float64)
            if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
                raise ValueError("gauge covariance must be a finite 7 by 7 matrix")
            if not np.allclose(covariance, covariance.T, atol=1e-12):
                raise ValueError("gauge covariance must be symmetric")
            _require_psd(covariance, "gauge covariance")
        for factor in factors:
            if factor.gauge_id not in gauge_id_set:
                raise ValueError(
                    f"factor {factor.factor_id!r} references an unknown gauge"
                )
            if factor.causal_frame_stop != causal_frame_stop:
                raise ValueError("factor and bundle causal frame stops differ")
            setting = (
                factor.prior_nominal_probability,
                factor.composite_weight,
            )
            previous = group_settings.setdefault(factor.correlation_group_id, setting)
            if previous != setting:
                raise ValueError(
                    "factors in one correlation group must share nominal "
                    "probability and composite weight"
                )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "gauges", gauges)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
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

    @property
    def correlation_group_parameters(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for factor in self.factors:
            result[factor.correlation_group_id] = {
                "prior_nominal_probability": factor.prior_nominal_probability,
                "composite_weight": factor.composite_weight,
            }
        return result

    def linearize(
        self, factor: ObservationFactor | str
    ) -> LinearizedObservationFactor:
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
            prior_reliability=selected.prior_reliability,
            prior_nominal_probability=selected.prior_nominal_probability,
            composite_weight=selected.composite_weight,
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
    """Stack rows while preserving all residual-independent evidence fields."""

    gauge_ids = tuple(gauge.window_id for gauge in bundle.gauges)
    gauge_offsets = {gauge_id: 7 * index for index, gauge_id in enumerate(gauge_ids)}
    gauge_dimension = 7 * len(gauge_ids)
    means: list[np.ndarray] = []
    conditional_covariances: list[np.ndarray] = []
    marginal_covariances: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    association_probabilities: list[float] = []
    prior_reliabilities: list[float] = []
    prior_nominal_probabilities: list[float] = []
    composite_weights: list[float] = []
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
            else factor.valid_mask
            & (factor.association_probability > 0.0)
            & (factor.prior_reliability > 0.0)
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
            association_probabilities.append(
                float(factor.association_probability[local_index])
            )
            prior_reliabilities.append(float(factor.prior_reliability[local_index]))
            prior_nominal_probabilities.append(factor.prior_nominal_probability)
            composite_weights.append(factor.composite_weight)
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
        association_probability=np.asarray(association_probabilities),
        prior_reliability=np.asarray(prior_reliabilities),
        prior_nominal_probability=np.asarray(prior_nominal_probabilities),
        composite_weight=np.asarray(composite_weights),
        point_ids=np.asarray(point_ids),
        frame_indices=np.asarray(frame_indices),
        view_ids=tuple(view_ids),
        factor_ids=tuple(factor_ids),
        correlation_group_ids=tuple(correlation_groups),
        gauge_ids=gauge_ids,
        causal_frame_stop=bundle.causal_frame_stop,
    )
