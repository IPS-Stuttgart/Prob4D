"""Target-prefix evaluation for calibration-transport support models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._calibration_transport_common import (
    CALIBRATION_TRANSPORT_CLAIM_BOUNDARY,
    CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA,
    CALIBRATION_TRANSPORT_VERSION,
    _EVIDENCE_FIELDS,
    CalibrationTransportUnitV1,
    FloatArray,
    _readonly_float,
    _sha256_json,
    _strict_identifier_tuple,
    _strict_integer_tuple,
    _strict_json_matrix,
    _strict_metadata_tuple,
)
from ._calibration_transport_group import (
    CalibrationTransportGroupResultV1,
    _target_group_result,
)
from ._calibration_transport_model import (
    CalibrationTransportModelV1,
    _quantile_embedding,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class CalibrationTransportEvidenceV1:
    """Replayable target-prefix support evidence for one frozen source model."""

    model: CalibrationTransportModelV1 = field(repr=False)
    target_unit_ids: tuple[str, ...]
    target_row_counts: tuple[int, ...]
    target_embeddings: FloatArray
    target_metadata: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None
    group_results: tuple[CalibrationTransportGroupResultV1, ...] = field(init=False)
    target_group_count: int = field(init=False)
    target_row_count: int = field(init=False)
    supported_group_count: int = field(init=False)
    supported_row_count: int = field(init=False)
    unsupported_group_fraction: float = field(init=False)
    unsupported_row_fraction: float = field(init=False)
    accepted: bool = field(init=False)
    decision_reasons: tuple[str, ...] = field(init=False)
    worst_target_unit_id: str = field(init=False)
    worst_feature_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, CalibrationTransportModelV1):
            raise TypeError("model must be a CalibrationTransportModelV1")
        target_ids = _strict_identifier_tuple(
            self.target_unit_ids,
            name="target_unit_ids",
        )
        if set(target_ids) & set(self.model.source_unit_ids):
            raise ValueError("target unit IDs overlap source unit IDs")
        target_count = len(target_ids)
        row_counts = _strict_integer_tuple(
            self.target_row_counts,
            name="target_row_counts",
            minimum=1,
        )
        if len(row_counts) != target_count:
            raise ValueError("target_row_counts length differs from target_unit_ids")
        embeddings = _readonly_float(
            self.target_embeddings,
            ndim=2,
            name="target_embeddings",
        )
        if embeddings.shape != (target_count, self.model.embedding_dimension):
            raise ValueError("target_embeddings shape differs from model")
        target_metadata = _strict_metadata_tuple(
            self.target_metadata,
            name="target_metadata",
            expected_length=target_count,
        )
        metadata = frozen_finite_json_mapping(
            require_finite_json_mapping(self.metadata, name="metadata"),
            name="metadata",
        )
        results = tuple(
            _target_group_result(
                self.model,
                unit_id=unit_id,
                row_count=row_count,
                embedding=embedding,
                metadata=group_metadata,
            )
            for unit_id, row_count, embedding, group_metadata in zip(
                target_ids,
                row_counts,
                embeddings,
                target_metadata,
                strict=True,
            )
        )
        total_rows = int(sum(row_counts))
        supported_group_count = sum(result.supported for result in results)
        supported_rows = sum(result.row_count for result in results if result.supported)
        unsupported_group_fraction = 1.0 - supported_group_count / target_count
        unsupported_row_fraction = 1.0 - supported_rows / total_rows
        reasons: list[str] = []
        if (
            unsupported_group_fraction
            > self.model.policy.maximum_unsupported_group_fraction
        ):
            reasons.append("unsupported-group-fraction")
        if unsupported_row_fraction > self.model.policy.maximum_unsupported_row_fraction:
            reasons.append("unsupported-row-fraction")
        accepted = not reasons
        worst_result = max(
            results,
            key=lambda result: (result.nonconformity_score, result.unit_id),
        )
        feature_maxima = {
            name: max(
                max(
                    float(result.feature_distance_rms[name]),
                    float(result.feature_outside_range_max[name]),
                )
                for result in results
            )
            for name in self.model.feature_names
        }
        worst_feature = max(
            self.model.feature_names,
            key=lambda name: (feature_maxima[name], name),
        )

        object.__setattr__(self, "target_unit_ids", target_ids)
        object.__setattr__(self, "target_row_counts", row_counts)
        object.__setattr__(self, "target_embeddings", embeddings)
        object.__setattr__(self, "target_metadata", target_metadata)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "group_results", results)
        object.__setattr__(self, "target_group_count", target_count)
        object.__setattr__(self, "target_row_count", total_rows)
        object.__setattr__(self, "supported_group_count", supported_group_count)
        object.__setattr__(self, "supported_row_count", supported_rows)
        object.__setattr__(
            self,
            "unsupported_group_fraction",
            float(unsupported_group_fraction),
        )
        object.__setattr__(
            self,
            "unsupported_row_fraction",
            float(unsupported_row_fraction),
        )
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "decision_reasons", tuple(reasons))
        object.__setattr__(self, "worst_target_unit_id", worst_result.unit_id)
        object.__setattr__(self, "worst_feature_name", worst_feature)

        computed_id = _sha256_json(self._identity_payload())
        if self.evidence_id is not None:
            supplied = require_sha256(self.evidence_id, name="evidence_id")
            if supplied != computed_id:
                raise ValueError("evidence_id does not match calibration transport content")
        object.__setattr__(self, "evidence_id", computed_id)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA,
            "schema_version": CALIBRATION_TRANSPORT_VERSION,
            "model_id": self.model.model_id,
            "feature_contract_id": self.model.feature_contract_id,
            "feature_names": list(self.model.feature_names),
            "target_unit_ids": list(self.target_unit_ids),
            "target_row_counts": list(self.target_row_counts),
            "target_embeddings": self.target_embeddings.tolist(),
            "target_metadata": [plain_json(item) for item in self.target_metadata],
            "group_results": [item.to_dict() for item in self.group_results],
            "target_group_count": self.target_group_count,
            "target_row_count": self.target_row_count,
            "supported_group_count": self.supported_group_count,
            "supported_row_count": self.supported_row_count,
            "unsupported_group_fraction": self.unsupported_group_fraction,
            "unsupported_row_fraction": self.unsupported_row_fraction,
            "accepted": self.accepted,
            "decision_reasons": list(self.decision_reasons),
            "worst_target_unit_id": self.worst_target_unit_id,
            "worst_feature_name": self.worst_feature_name,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CALIBRATION_TRANSPORT_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        model: CalibrationTransportModelV1,
    ) -> CalibrationTransportEvidenceV1:
        mapping = require_mapping(value, name="calibration transport evidence")
        require_exact_fields(
            mapping,
            _EVIDENCE_FIELDS,
            name="calibration transport evidence",
        )
        if mapping["schema"] != CALIBRATION_TRANSPORT_EVIDENCE_SCHEMA:
            raise ValueError("calibration transport evidence schema changed")
        if require_exact_integer(
            mapping["schema_version"],
            name="schema_version",
            minimum=1,
        ) != CALIBRATION_TRANSPORT_VERSION:
            raise ValueError("calibration transport evidence version changed")
        if mapping["claim_boundary"] != CALIBRATION_TRANSPORT_CLAIM_BOUNDARY:
            raise ValueError("calibration transport claim boundary changed")
        if mapping["model_id"] != model.model_id:
            raise ValueError("calibration transport evidence references another model")
        if mapping["feature_contract_id"] != model.feature_contract_id:
            raise ValueError("calibration transport feature contract changed")
        if mapping["feature_names"] != list(model.feature_names):
            raise ValueError("calibration transport feature names changed")

        target_ids_value = mapping["target_unit_ids"]
        row_counts_value = mapping["target_row_counts"]
        embeddings_value = mapping["target_embeddings"]
        metadata_value = mapping["target_metadata"]
        if type(target_ids_value) is not list:
            raise ValueError("target_unit_ids must be a JSON array")
        if type(row_counts_value) is not list:
            raise ValueError("target_row_counts must be a JSON array")
        if type(embeddings_value) is not list:
            raise ValueError("target_embeddings must be a JSON matrix")
        if type(metadata_value) is not list:
            raise ValueError("target_metadata must be a JSON array")
        evidence = cls(
            model=model,
            target_unit_ids=tuple(target_ids_value),
            target_row_counts=tuple(row_counts_value),
            target_embeddings=_strict_json_matrix(
                embeddings_value,
                name="target_embeddings",
            ),
            target_metadata=tuple(
                require_mapping(item, name=f"target_metadata[{index}]")
                for index, item in enumerate(metadata_value)
            ),
            metadata=require_mapping(mapping["metadata"], name="metadata"),
            evidence_id=mapping["evidence_id"],
        )
        if evidence.to_dict() != plain_json(mapping):
            raise ValueError("calibration transport evidence derived fields changed")
        return evidence


def evaluate_calibration_transport(
    model: CalibrationTransportModelV1,
    target_units: Sequence[CalibrationTransportUnitV1],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationTransportEvidenceV1:
    """Evaluate target-prefix support without target residuals or outcomes."""

    if not isinstance(model, CalibrationTransportModelV1):
        raise TypeError("model must be a CalibrationTransportModelV1")
    if isinstance(target_units, (str, bytes)) or not isinstance(target_units, Sequence):
        raise TypeError("target_units must be a sequence")
    units = tuple(target_units)
    if not units or not all(isinstance(item, CalibrationTransportUnitV1) for item in units):
        raise ValueError("target_units must contain CalibrationTransportUnitV1 values")
    ordered = tuple(sorted(units, key=lambda item: item.unit_id))
    if len({item.unit_id for item in ordered}) != len(ordered):
        raise ValueError("target unit IDs must be unique")
    if any(item.feature_contract_id != model.feature_contract_id for item in ordered):
        raise ValueError("target feature contract differs from source model")
    if any(item.feature_names != model.feature_names for item in ordered):
        raise ValueError("target feature names differ from source model")
    embeddings = np.stack(
        [
            _quantile_embedding(item.feature_values, model.policy.quantile_levels)
            for item in ordered
        ]
    )
    return CalibrationTransportEvidenceV1(
        model=model,
        target_unit_ids=tuple(item.unit_id for item in ordered),
        target_row_counts=tuple(item.row_count for item in ordered),
        target_embeddings=embeddings,
        target_metadata=tuple(item.metadata for item in ordered),
        metadata={} if metadata is None else metadata,
    )
