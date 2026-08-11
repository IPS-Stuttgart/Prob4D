"""Source-only fitting for calibration-transport support models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._calibration_transport_common import (
    CALIBRATION_TRANSPORT_CLAIM_BOUNDARY,
    CALIBRATION_TRANSPORT_MODEL_SCHEMA,
    CALIBRATION_TRANSPORT_VERSION,
    _IQR_NORMAL_SCALE,
    _MODEL_FIELDS,
    _MAD_NORMAL_SCALE,
    CalibrationTransportPolicyV1,
    CalibrationTransportUnitV1,
    FloatArray,
    _readonly_float,
    _sha256_json,
    _strict_feature_names,
    _strict_identifier_tuple,
    _strict_integer_tuple,
    _strict_json_matrix,
    _strict_metadata_tuple,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)


def _quantile_embedding(
    values: FloatArray,
    quantile_levels: tuple[float, ...],
) -> FloatArray:
    quantiles = np.quantile(
        values,
        np.asarray(quantile_levels, dtype=np.float64),
        axis=0,
        method="linear",
    )
    embedding = np.asarray(quantiles.T.reshape(-1), dtype=np.float64)
    if not np.all(np.isfinite(embedding)):
        raise ValueError("quantile embedding is non-finite")
    return embedding


def _robust_center_scale(
    embeddings: FloatArray,
    policy: CalibrationTransportPolicyV1,
) -> tuple[FloatArray, FloatArray]:
    center = np.median(embeddings, axis=0)
    mad_scale = _MAD_NORMAL_SCALE * np.median(
        np.abs(embeddings - center[None, :]),
        axis=0,
    )
    quartiles = np.quantile(
        embeddings,
        np.asarray([0.25, 0.75]),
        axis=0,
        method="linear",
    )
    iqr_scale = (quartiles[1] - quartiles[0]) / _IQR_NORMAL_SCALE
    robust_scale = np.maximum(mad_scale, iqr_scale)
    reference = np.maximum(
        np.maximum(np.abs(center), np.max(np.abs(embeddings), axis=0)),
        1.0,
    )
    floor = policy.absolute_scale_floor + policy.relative_scale_floor * reference
    scale = np.maximum(robust_scale, floor)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("calibration transport scaling is non-finite")
    if np.any(scale <= 0.0):
        raise ValueError("calibration transport scale must be positive")
    return np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def _pairwise_distances(embeddings: FloatArray, scale: FloatArray) -> FloatArray:
    standardized = (embeddings[:, None, :] - embeddings[None, :, :]) / scale
    distances = np.sqrt(np.mean(standardized**2, axis=2))
    if not np.all(np.isfinite(distances)):
        raise ValueError("source pairwise support distance is non-finite")
    return np.asarray(distances, dtype=np.float64)


def _source_nonconformity_scores(
    pairwise: FloatArray,
    *,
    neighbor_count: int,
) -> FloatArray:
    source_count = int(pairwise.shape[0])
    if pairwise.shape != (source_count, source_count):
        raise ValueError("source pairwise distance matrix must be square")
    scores = np.empty(source_count, dtype=np.float64)
    for index in range(source_count):
        candidates = np.delete(pairwise[index], index)
        selected = np.partition(candidates, neighbor_count - 1)[:neighbor_count]
        scores[index] = float(np.mean(selected))
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("source nonconformity scores must be finite and non-negative")
    return scores


def _threshold_rank(source_count: int, miscoverage_rate: float) -> int:
    raw_rank = math.ceil((source_count + 1) * (1.0 - miscoverage_rate))
    return min(source_count, max(1, raw_rank))


@dataclass(frozen=True, slots=True)
class CalibrationTransportModelV1:
    """Content-addressed source-only support model."""

    feature_contract_id: str
    feature_names: tuple[str, ...]
    policy: CalibrationTransportPolicyV1
    source_unit_ids: tuple[str, ...]
    source_row_counts: tuple[int, ...]
    source_embeddings: FloatArray
    source_metadata: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    embedding_center: FloatArray = field(init=False)
    embedding_scale: FloatArray = field(init=False)
    source_nonconformity_scores: FloatArray = field(init=False)
    support_threshold: float = field(init=False)
    threshold_rank: int = field(init=False)

    def __post_init__(self) -> None:
        contract_id = require_sha256(
            self.feature_contract_id,
            name="feature_contract_id",
        )
        names = _strict_feature_names(self.feature_names)
        if not isinstance(self.policy, CalibrationTransportPolicyV1):
            raise TypeError("policy must be a CalibrationTransportPolicyV1")
        source_ids = _strict_identifier_tuple(
            self.source_unit_ids,
            name="source_unit_ids",
        )
        source_count = len(source_ids)
        if source_count < self.policy.minimum_source_units:
            raise ValueError(
                "source unit count is below policy.minimum_source_units"
            )
        if self.policy.neighbor_count >= source_count:
            raise ValueError("neighbor_count must be smaller than the source unit count")
        row_counts = _strict_integer_tuple(
            self.source_row_counts,
            name="source_row_counts",
            minimum=1,
        )
        if len(row_counts) != source_count:
            raise ValueError("source_row_counts length differs from source_unit_ids")
        embeddings = _readonly_float(
            self.source_embeddings,
            ndim=2,
            name="source_embeddings",
        )
        expected_dimension = len(names) * len(self.policy.quantile_levels)
        if embeddings.shape != (source_count, expected_dimension):
            raise ValueError(
                "source_embeddings shape differs from source units and feature policy"
            )
        source_metadata = _strict_metadata_tuple(
            self.source_metadata,
            name="source_metadata",
            expected_length=source_count,
        )
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )

        center, scale = _robust_center_scale(embeddings, self.policy)
        pairwise = _pairwise_distances(embeddings, scale)
        scores = _source_nonconformity_scores(
            pairwise,
            neighbor_count=self.policy.neighbor_count,
        )
        rank = _threshold_rank(source_count, self.policy.miscoverage_rate)
        threshold = float(np.sort(scores, kind="stable")[rank - 1])

        object.__setattr__(self, "feature_contract_id", contract_id)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "source_unit_ids", source_ids)
        object.__setattr__(self, "source_row_counts", row_counts)
        object.__setattr__(self, "source_embeddings", embeddings)
        object.__setattr__(self, "source_metadata", source_metadata)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "embedding_center",
            _readonly_float(center, ndim=1, name="embedding_center"),
        )
        object.__setattr__(
            self,
            "embedding_scale",
            _readonly_float(scale, ndim=1, name="embedding_scale"),
        )
        object.__setattr__(
            self,
            "source_nonconformity_scores",
            _readonly_float(
                scores,
                ndim=1,
                name="source_nonconformity_scores",
            ),
        )
        object.__setattr__(self, "support_threshold", threshold)
        object.__setattr__(self, "threshold_rank", rank)

        computed_id = _sha256_json(self._identity_payload())
        if self.model_id is not None:
            supplied = require_sha256(self.model_id, name="model_id")
            if supplied != computed_id:
                raise ValueError("model_id does not match calibration transport content")
        object.__setattr__(self, "model_id", computed_id)

    @property
    def source_unit_count(self) -> int:
        return len(self.source_unit_ids)

    @property
    def embedding_dimension(self) -> int:
        return int(self.source_embeddings.shape[1])

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_TRANSPORT_MODEL_SCHEMA,
            "schema_version": CALIBRATION_TRANSPORT_VERSION,
            "feature_contract_id": self.feature_contract_id,
            "feature_names": list(self.feature_names),
            "policy": self.policy.to_dict(),
            "source_unit_ids": list(self.source_unit_ids),
            "source_row_counts": list(self.source_row_counts),
            "source_embeddings": self.source_embeddings.tolist(),
            "source_metadata": [plain_json(item) for item in self.source_metadata],
            "embedding_center": self.embedding_center.tolist(),
            "embedding_scale": self.embedding_scale.tolist(),
            "source_nonconformity_scores": self.source_nonconformity_scores.tolist(),
            "support_threshold": self.support_threshold,
            "threshold_rank": self.threshold_rank,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CALIBRATION_TRANSPORT_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationTransportModelV1:
        mapping = require_mapping(value, name="calibration transport model")
        require_exact_fields(mapping, _MODEL_FIELDS, name="calibration transport model")
        if mapping["schema"] != CALIBRATION_TRANSPORT_MODEL_SCHEMA:
            raise ValueError("calibration transport model schema changed")
        if require_exact_integer(
            mapping["schema_version"],
            name="schema_version",
            minimum=1,
        ) != CALIBRATION_TRANSPORT_VERSION:
            raise ValueError("calibration transport model version changed")
        if mapping["claim_boundary"] != CALIBRATION_TRANSPORT_CLAIM_BOUNDARY:
            raise ValueError("calibration transport claim boundary changed")

        feature_names_value = mapping["feature_names"]
        source_ids_value = mapping["source_unit_ids"]
        row_counts_value = mapping["source_row_counts"]
        embeddings_value = mapping["source_embeddings"]
        metadata_value = mapping["source_metadata"]
        if type(feature_names_value) is not list:
            raise ValueError("feature_names must be a JSON array")
        if type(source_ids_value) is not list:
            raise ValueError("source_unit_ids must be a JSON array")
        if type(row_counts_value) is not list:
            raise ValueError("source_row_counts must be a JSON array")
        if type(embeddings_value) is not list:
            raise ValueError("source_embeddings must be a JSON matrix")
        if type(metadata_value) is not list:
            raise ValueError("source_metadata must be a JSON array")

        model = cls(
            feature_contract_id=mapping["feature_contract_id"],
            feature_names=tuple(feature_names_value),
            policy=CalibrationTransportPolicyV1.from_dict(mapping["policy"]),
            source_unit_ids=tuple(source_ids_value),
            source_row_counts=tuple(row_counts_value),
            source_embeddings=_strict_json_matrix(
                embeddings_value,
                name="source_embeddings",
            ),
            source_metadata=tuple(
                require_mapping(item, name=f"source_metadata[{index}]")
                for index, item in enumerate(metadata_value)
            ),
            metadata=require_mapping(mapping["metadata"], name="metadata"),
            model_id=mapping["model_id"],
        )
        if model.to_dict() != plain_json(mapping):
            raise ValueError("calibration transport model derived fields changed")
        return model


def fit_calibration_transport_model(
    source_units: Sequence[CalibrationTransportUnitV1],
    *,
    policy: CalibrationTransportPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationTransportModelV1:
    """Fit a support model from complete independent source units only."""

    if isinstance(source_units, (str, bytes)) or not isinstance(source_units, Sequence):
        raise TypeError("source_units must be a sequence")
    units = tuple(source_units)
    if not units or not all(isinstance(item, CalibrationTransportUnitV1) for item in units):
        raise ValueError("source_units must contain CalibrationTransportUnitV1 values")
    ordered = tuple(sorted(units, key=lambda item: item.unit_id))
    if len({item.unit_id for item in ordered}) != len(ordered):
        raise ValueError("source unit IDs must be unique")
    contract_ids = {item.feature_contract_id for item in ordered}
    if len(contract_ids) != 1:
        raise ValueError("source feature contract IDs changed across units")
    feature_names = ordered[0].feature_names
    if any(item.feature_names != feature_names for item in ordered[1:]):
        raise ValueError("source feature names changed across units")
    embeddings = np.stack(
        [
            _quantile_embedding(item.feature_values, policy.quantile_levels)
            for item in ordered
        ]
    )
    return CalibrationTransportModelV1(
        feature_contract_id=ordered[0].feature_contract_id,
        feature_names=feature_names,
        policy=policy,
        source_unit_ids=tuple(item.unit_id for item in ordered),
        source_row_counts=tuple(item.row_count for item in ordered),
        source_embeddings=embeddings,
        source_metadata=tuple(item.metadata for item in ordered),
        metadata={} if metadata is None else metadata,
    )
