"""Per-group diagnostics for calibration-transport target support."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._calibration_transport_common import (
    FloatArray,
    _strict_identifier_tuple,
    _strict_nonnegative_real,
    _strict_probability,
)
from ._calibration_transport_model import CalibrationTransportModelV1
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
)


@dataclass(frozen=True, slots=True)
class CalibrationTransportGroupResultV1:
    """One target-prefix support decision and localization diagnostic."""

    unit_id: str
    row_count: int
    nonconformity_score: float
    source_support_p_value: float
    supported: bool
    support_margin: float
    nearest_source_unit_ids: tuple[str, ...]
    nearest_source_distances: tuple[float, ...]
    feature_distance_rms: Mapping[str, Any]
    feature_outside_range_max: Mapping[str, Any]
    worst_distance_feature: str
    worst_outside_range_feature: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        unit_id = require_exact_string(self.unit_id, name="unit_id")
        row_count = require_exact_integer(self.row_count, name="row_count", minimum=1)
        score = _strict_nonnegative_real(
            self.nonconformity_score,
            name="nonconformity_score",
        )
        p_value = _strict_probability(
            self.source_support_p_value,
            name="source_support_p_value",
            lower_open=True,
            upper_open=False,
        )
        if type(self.supported) is not bool:
            raise ValueError("supported must be a Boolean")
        margin = require_json_number(self.support_margin, name="support_margin")
        if type(self.nearest_source_unit_ids) is not tuple:
            raise TypeError("nearest_source_unit_ids must be a canonical tuple")
        nearest_ids = _strict_identifier_tuple(
            self.nearest_source_unit_ids,
            name="nearest_source_unit_ids",
        )
        if type(self.nearest_source_distances) is not tuple:
            raise TypeError("nearest_source_distances must be a canonical tuple")
        distances = tuple(
            _strict_nonnegative_real(item, name=f"nearest_source_distances[{index}]")
            for index, item in enumerate(self.nearest_source_distances)
        )
        if len(distances) != len(nearest_ids):
            raise ValueError("nearest source IDs and distances differ in length")
        distance_mapping = _strict_feature_mapping(
            self.feature_distance_rms,
            name="feature_distance_rms",
        )
        outside_mapping = _strict_feature_mapping(
            self.feature_outside_range_max,
            name="feature_outside_range_max",
        )
        if tuple(distance_mapping) != tuple(outside_mapping):
            raise ValueError("feature diagnostic names changed")
        worst_distance = require_exact_string(
            self.worst_distance_feature,
            name="worst_distance_feature",
        )
        worst_outside = require_exact_string(
            self.worst_outside_range_feature,
            name="worst_outside_range_feature",
        )
        if worst_distance not in distance_mapping or worst_outside not in outside_mapping:
            raise ValueError("worst feature does not occur in feature diagnostics")
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "nonconformity_score", score)
        object.__setattr__(self, "source_support_p_value", p_value)
        object.__setattr__(self, "support_margin", margin)
        object.__setattr__(self, "nearest_source_unit_ids", nearest_ids)
        object.__setattr__(self, "nearest_source_distances", distances)
        object.__setattr__(self, "feature_distance_rms", distance_mapping)
        object.__setattr__(self, "feature_outside_range_max", outside_mapping)
        object.__setattr__(self, "worst_distance_feature", worst_distance)
        object.__setattr__(self, "worst_outside_range_feature", worst_outside)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "row_count": self.row_count,
            "nonconformity_score": self.nonconformity_score,
            "source_support_p_value": self.source_support_p_value,
            "supported": self.supported,
            "support_margin": self.support_margin,
            "nearest_source_unit_ids": list(self.nearest_source_unit_ids),
            "nearest_source_distances": list(self.nearest_source_distances),
            "feature_distance_rms": plain_json(self.feature_distance_rms),
            "feature_outside_range_max": plain_json(
                self.feature_outside_range_max
            ),
            "worst_distance_feature": self.worst_distance_feature,
            "worst_outside_range_feature": self.worst_outside_range_feature,
            "metadata": plain_json(self.metadata),
        }


def _strict_feature_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    mapping = require_mapping(value, name=name)
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    normalized: dict[str, float] = {}
    for key, item in mapping.items():
        feature_name = require_exact_string(key, name=f"{name} key")
        normalized[feature_name] = _strict_nonnegative_real(
            item,
            name=f"{name}[{feature_name!r}]",
        )
    return frozen_finite_json_mapping(normalized, name=name)


def _target_group_result(
    model: CalibrationTransportModelV1,
    *,
    unit_id: str,
    row_count: int,
    embedding: FloatArray,
    metadata: Mapping[str, Any],
) -> CalibrationTransportGroupResultV1:
    standardized = (model.source_embeddings - embedding[None, :]) / model.embedding_scale
    distances = np.sqrt(np.mean(standardized**2, axis=1))
    ordered_indices = sorted(
        range(model.source_unit_count),
        key=lambda index: (float(distances[index]), model.source_unit_ids[index]),
    )
    selected = ordered_indices[: model.policy.neighbor_count]
    selected_distances = tuple(float(distances[index]) for index in selected)
    score = float(np.mean(selected_distances))
    p_value = float(
        (1 + int(np.count_nonzero(model.source_nonconformity_scores >= score)))
        / (model.source_unit_count + 1)
    )
    supported = score <= model.support_threshold

    feature_count = len(model.feature_names)
    quantile_count = len(model.policy.quantile_levels)
    selected_delta = standardized[selected].reshape(
        len(selected),
        feature_count,
        quantile_count,
    )
    feature_distance = np.sqrt(np.mean(selected_delta**2, axis=(0, 2)))

    source_minimum = np.min(model.source_embeddings, axis=0)
    source_maximum = np.max(model.source_embeddings, axis=0)
    lower_excess = np.maximum(source_minimum - embedding, 0.0)
    upper_excess = np.maximum(embedding - source_maximum, 0.0)
    outside = np.maximum(lower_excess, upper_excess) / model.embedding_scale
    feature_outside = np.max(outside.reshape(feature_count, quantile_count), axis=1)

    distance_mapping = {
        name: float(feature_distance[index])
        for index, name in enumerate(model.feature_names)
    }
    outside_mapping = {
        name: float(feature_outside[index])
        for index, name in enumerate(model.feature_names)
    }
    worst_distance = max(
        model.feature_names,
        key=lambda name: (distance_mapping[name], name),
    )
    worst_outside = max(
        model.feature_names,
        key=lambda name: (outside_mapping[name], name),
    )
    nearest_pairs = sorted(
        (
            (model.source_unit_ids[index], float(distances[index]))
            for index in selected
        ),
        key=lambda item: item[0],
    )
    return CalibrationTransportGroupResultV1(
        unit_id=unit_id,
        row_count=row_count,
        nonconformity_score=score,
        source_support_p_value=p_value,
        supported=supported,
        support_margin=float(model.support_threshold - score),
        nearest_source_unit_ids=tuple(item[0] for item in nearest_pairs),
        nearest_source_distances=tuple(item[1] for item in nearest_pairs),
        feature_distance_rms=distance_mapping,
        feature_outside_range_max=outside_mapping,
        worst_distance_feature=worst_distance,
        worst_outside_range_feature=worst_outside,
        metadata=metadata,
    )
