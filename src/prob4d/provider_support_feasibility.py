"""Outcome-blind support feasibility for prospective provider cohorts.

The contract in this module is deliberately upstream of provider residuals, target
outcomes, and dense prediction payloads. It answers only whether the frozen
causal prefix has the declared metadata, geometry, and metric support needed to
run a later provider-competence experiment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ._heldout_promotion_common import (
    _atomic_write_json,
    _load_json,
    _repository,
    _revision,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_bool,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_real,
    _strict_string,
)

PROVIDER_SUPPORT_FEASIBILITY_REQUEST_SCHEMA = (
    "prob4d.provider-support-feasibility-request"
)
PROVIDER_SUPPORT_FEASIBILITY_REQUEST_VERSION = 1
PROVIDER_SUPPORT_FEASIBILITY_SCHEMA = "prob4d.provider-support-feasibility"
PROVIDER_SUPPORT_FEASIBILITY_VERSION = 1
PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY = (
    "This artifact establishes only outcome-blind support feasibility for the exact "
    "frozen causal-prefix streams and identities. It is computed without opening "
    "prediction payloads, provider residuals, or target outcomes. A passing result "
    "does not establish provider competence, calibrated uncertainty, BayesianPhysTwin "
    "benefit, Causal4D benefit, deployment safety, or state of the art."
)

AdmissionRule = Literal["all-streams", "minimum-stream-fraction"]

_STREAM_FIELDS = {
    "group_id",
    "stream_id",
    "causal_frame_start",
    "causal_frame_stop_exclusive",
    "required_frame_ids",
    "available_frame_ids",
    "geometry_supported_frame_ids",
    "minimum_geometry_support_fraction",
    "intrinsics_required",
    "intrinsics_id",
    "extrinsics_required",
    "extrinsics_id",
    "metric_anchor_required",
    "metric_anchor_id",
    "technical_failure_code",
    "metadata",
}
_REQUEST_FIELDS = {
    "schema_name",
    "schema_version",
    "protocol_id",
    "source_repository",
    "source_revision",
    "provider_family",
    "provider_repository",
    "provider_revision",
    "model_set_id",
    "loader_id",
    "cohort_binding_id",
    "promotion_lock_id",
    "coordinate_semantics",
    "admission_rule",
    "minimum_supported_fraction",
    "permitted_technical_exclusion_codes",
    "maximum_technical_exclusions",
    "prediction_payloads_opened",
    "residuals_used",
    "target_outcomes_used",
    "streams",
    "metadata",
    "claim_boundary",
    "request_id",
}
_STREAM_RESULT_FIELDS = {
    "group_id",
    "stream_id",
    "supported",
    "excluded_from_admission",
    "reason_codes",
    "required_frame_count",
    "available_required_frame_count",
    "geometry_supported_required_frame_count",
    "geometry_support_fraction",
}
_RESULT_FIELDS = {
    "schema_name",
    "schema_version",
    "request",
    "stream_results",
    "stream_count",
    "supported_stream_count",
    "excluded_stream_count",
    "evaluable_stream_count",
    "support_fraction",
    "technical_exclusion_count",
    "support_feasible",
    "decision_reason",
    "claim_boundary",
    "provider_support_feasibility_id",
}


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _strict_digest(value, name=name, pattern=_SHA256)


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _strict_string(value, name=name)


def _canonical_integer_tuple(
    value: object,
    *,
    name: str,
    nonempty: bool,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    result = tuple(
        _strict_integer(item, name=f"{name}[{index}]", minimum=0)
        for index, item in enumerate(value)
    )
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _integer_tuple_from_json(
    value: object,
    *,
    name: str,
    nonempty: bool,
) -> tuple[int, ...]:
    return _canonical_integer_tuple(
        tuple(_strict_list(value, name=name)),
        name=name,
        nonempty=nonempty,
    )


def _canonical_string_tuple(
    value: object,
    *,
    name: str,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    result = tuple(
        _strict_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _string_tuple_from_json(value: object, *, name: str) -> tuple[str, ...]:
    return _canonical_string_tuple(tuple(_strict_list(value, name=name)), name=name)


def _admission_rule(value: object, *, name: str) -> AdmissionRule:
    result = _strict_string(value, name=name)
    if result not in {"all-streams", "minimum-stream-fraction"}:
        raise ValueError(
            f"{name} must be 'all-streams' or 'minimum-stream-fraction'"
        )
    return cast(AdmissionRule, result)


@dataclass(frozen=True, slots=True)
class ProviderSupportStreamV1:
    """Outcome-blind support metadata for one frozen group/stream pair."""

    group_id: str
    stream_id: str
    causal_frame_start: int
    causal_frame_stop_exclusive: int
    required_frame_ids: tuple[int, ...]
    available_frame_ids: tuple[int, ...]
    geometry_supported_frame_ids: tuple[int, ...]
    minimum_geometry_support_fraction: float
    intrinsics_required: bool
    intrinsics_id: str | None
    extrinsics_required: bool
    extrinsics_id: str | None
    metric_anchor_required: bool
    metric_anchor_id: str | None
    technical_failure_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            _strict_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "stream_id",
            _strict_string(self.stream_id, name="stream_id"),
        )
        start = _strict_integer(
            self.causal_frame_start,
            name="causal_frame_start",
            minimum=0,
        )
        stop = _strict_integer(
            self.causal_frame_stop_exclusive,
            name="causal_frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ValueError(
                "causal_frame_stop_exclusive must exceed causal_frame_start"
            )
        required = _canonical_integer_tuple(
            self.required_frame_ids,
            name="required_frame_ids",
            nonempty=True,
        )
        available = _canonical_integer_tuple(
            self.available_frame_ids,
            name="available_frame_ids",
            nonempty=False,
        )
        geometry = _canonical_integer_tuple(
            self.geometry_supported_frame_ids,
            name="geometry_supported_frame_ids",
            nonempty=False,
        )
        for frame_name, frames in (
            ("required_frame_ids", required),
            ("available_frame_ids", available),
            ("geometry_supported_frame_ids", geometry),
        ):
            if any(frame < start or frame >= stop for frame in frames):
                raise ValueError(f"{frame_name} crosses the frozen causal prefix")
        if not set(geometry).issubset(available):
            raise ValueError(
                "geometry_supported_frame_ids must be a subset of available_frame_ids"
            )
        minimum_geometry = _strict_real(
            self.minimum_geometry_support_fraction,
            name="minimum_geometry_support_fraction",
        )
        if not 0.0 <= minimum_geometry <= 1.0:
            raise ValueError(
                "minimum_geometry_support_fraction must lie in [0, 1]"
            )
        for field_name in (
            "intrinsics_required",
            "extrinsics_required",
            "metric_anchor_required",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_bool(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "intrinsics_id",
            _optional_digest(self.intrinsics_id, name="intrinsics_id"),
        )
        object.__setattr__(
            self,
            "extrinsics_id",
            _optional_digest(self.extrinsics_id, name="extrinsics_id"),
        )
        object.__setattr__(
            self,
            "metric_anchor_id",
            _optional_digest(self.metric_anchor_id, name="metric_anchor_id"),
        )
        object.__setattr__(
            self,
            "technical_failure_code",
            _optional_string(
                self.technical_failure_code,
                name="technical_failure_code",
            ),
        )
        object.__setattr__(self, "causal_frame_start", start)
        object.__setattr__(self, "causal_frame_stop_exclusive", stop)
        object.__setattr__(self, "required_frame_ids", required)
        object.__setattr__(self, "available_frame_ids", available)
        object.__setattr__(self, "geometry_supported_frame_ids", geometry)
        object.__setattr__(
            self,
            "minimum_geometry_support_fraction",
            minimum_geometry,
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.group_id, self.stream_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "stream_id": self.stream_id,
            "causal_frame_start": self.causal_frame_start,
            "causal_frame_stop_exclusive": self.causal_frame_stop_exclusive,
            "required_frame_ids": list(self.required_frame_ids),
            "available_frame_ids": list(self.available_frame_ids),
            "geometry_supported_frame_ids": list(
                self.geometry_supported_frame_ids
            ),
            "minimum_geometry_support_fraction": (
                self.minimum_geometry_support_fraction
            ),
            "intrinsics_required": self.intrinsics_required,
            "intrinsics_id": self.intrinsics_id,
            "extrinsics_required": self.extrinsics_required,
            "extrinsics_id": self.extrinsics_id,
            "metric_anchor_required": self.metric_anchor_required,
            "metric_anchor_id": self.metric_anchor_id,
            "technical_failure_code": self.technical_failure_code,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportStreamV1:
        mapping = _strict_mapping(value, name="provider support stream")
        _exact_keys(mapping, _STREAM_FIELDS, name="provider support stream")
        return cls(
            group_id=mapping["group_id"],
            stream_id=mapping["stream_id"],
            causal_frame_start=mapping["causal_frame_start"],
            causal_frame_stop_exclusive=mapping[
                "causal_frame_stop_exclusive"
            ],
            required_frame_ids=_integer_tuple_from_json(
                mapping["required_frame_ids"],
                name="required_frame_ids",
                nonempty=True,
            ),
            available_frame_ids=_integer_tuple_from_json(
                mapping["available_frame_ids"],
                name="available_frame_ids",
                nonempty=False,
            ),
            geometry_supported_frame_ids=_integer_tuple_from_json(
                mapping["geometry_supported_frame_ids"],
                name="geometry_supported_frame_ids",
                nonempty=False,
            ),
            minimum_geometry_support_fraction=mapping[
                "minimum_geometry_support_fraction"
            ],
            intrinsics_required=mapping["intrinsics_required"],
            intrinsics_id=mapping["intrinsics_id"],
            extrinsics_required=mapping["extrinsics_required"],
            extrinsics_id=mapping["extrinsics_id"],
            metric_anchor_required=mapping["metric_anchor_required"],
            metric_anchor_id=mapping["metric_anchor_id"],
            technical_failure_code=mapping["technical_failure_code"],
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ProviderSupportFeasibilityRequestV1:
    """Frozen support requirements and metadata observations for one cohort."""

    protocol_id: str
    source_repository: str
    source_revision: str
    provider_family: str
    provider_repository: str
    provider_revision: str
    model_set_id: str
    loader_id: str
    cohort_binding_id: str
    promotion_lock_id: str
    coordinate_semantics: str
    admission_rule: AdmissionRule
    minimum_supported_fraction: float
    permitted_technical_exclusion_codes: tuple[str, ...]
    maximum_technical_exclusions: int
    prediction_payloads_opened: bool
    residuals_used: bool
    target_outcomes_used: bool
    streams: tuple[ProviderSupportStreamV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_id",
            "provider_family",
            "coordinate_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_string(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "source_repository",
            _repository(self.source_repository, name="source_repository"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _revision(self.source_revision, name="source_revision"),
        )
        object.__setattr__(
            self,
            "provider_repository",
            _repository(self.provider_repository, name="provider_repository"),
        )
        object.__setattr__(
            self,
            "provider_revision",
            _revision(self.provider_revision, name="provider_revision"),
        )
        for field_name in (
            "model_set_id",
            "loader_id",
            "cohort_binding_id",
            "promotion_lock_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_digest(
                    getattr(self, field_name),
                    name=field_name,
                    pattern=_SHA256,
                ),
            )
        rule = _admission_rule(self.admission_rule, name="admission_rule")
        minimum_fraction = _strict_real(
            self.minimum_supported_fraction,
            name="minimum_supported_fraction",
        )
        if not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_supported_fraction must lie in (0, 1]")
        if rule == "all-streams" and minimum_fraction != 1.0:
            raise ValueError(
                "all-streams admission requires minimum_supported_fraction == 1"
            )
        exclusions = _canonical_string_tuple(
            self.permitted_technical_exclusion_codes,
            name="permitted_technical_exclusion_codes",
        )
        maximum_exclusions = _strict_integer(
            self.maximum_technical_exclusions,
            name="maximum_technical_exclusions",
            minimum=0,
        )
        for field_name in (
            "prediction_payloads_opened",
            "residuals_used",
            "target_outcomes_used",
        ):
            value = _strict_bool(getattr(self, field_name), name=field_name)
            if value:
                raise ValueError(
                    f"{field_name} must be false for support feasibility"
                )
            object.__setattr__(self, field_name, value)
        if type(self.streams) is not tuple or not self.streams:
            raise ValueError(
                "streams must be a nonempty tuple of ProviderSupportStreamV1"
            )
        if not all(isinstance(item, ProviderSupportStreamV1) for item in self.streams):
            raise ValueError(
                "streams must contain only ProviderSupportStreamV1 values"
            )
        streams = tuple(sorted(self.streams, key=lambda item: item.key))
        keys = tuple(item.key for item in streams)
        if len(keys) != len(set(keys)):
            raise ValueError("group_id/stream_id pairs must be unique")
        object.__setattr__(self, "admission_rule", rule)
        object.__setattr__(
            self,
            "minimum_supported_fraction",
            minimum_fraction,
        )
        object.__setattr__(
            self,
            "permitted_technical_exclusion_codes",
            exclusions,
        )
        object.__setattr__(
            self,
            "maximum_technical_exclusions",
            maximum_exclusions,
        )
        object.__setattr__(self, "streams", streams)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    def _identity_dict(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_SUPPORT_FEASIBILITY_REQUEST_SCHEMA,
            "schema_version": PROVIDER_SUPPORT_FEASIBILITY_REQUEST_VERSION,
            "protocol_id": self.protocol_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "provider_family": self.provider_family,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "cohort_binding_id": self.cohort_binding_id,
            "promotion_lock_id": self.promotion_lock_id,
            "coordinate_semantics": self.coordinate_semantics,
            "admission_rule": self.admission_rule,
            "minimum_supported_fraction": self.minimum_supported_fraction,
            "permitted_technical_exclusion_codes": list(
                self.permitted_technical_exclusion_codes
            ),
            "maximum_technical_exclusions": self.maximum_technical_exclusions,
            "prediction_payloads_opened": self.prediction_payloads_opened,
            "residuals_used": self.residuals_used,
            "target_outcomes_used": self.target_outcomes_used,
            "streams": [item.to_dict() for item in self.streams],
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY,
        }

    @property
    def request_id(self) -> str:
        return _sha256_json(self._identity_dict())

    def to_dict(self) -> dict[str, object]:
        payload = self._identity_dict()
        payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportFeasibilityRequestV1:
        mapping = _strict_mapping(
            value,
            name="provider support feasibility request",
        )
        _exact_keys(
            mapping,
            _REQUEST_FIELDS,
            name="provider support feasibility request",
        )
        if mapping["schema_name"] != PROVIDER_SUPPORT_FEASIBILITY_REQUEST_SCHEMA:
            raise ValueError("unsupported provider support request schema")
        if (
            mapping["schema_version"]
            != PROVIDER_SUPPORT_FEASIBILITY_REQUEST_VERSION
        ):
            raise ValueError("unsupported provider support request version")
        if (
            mapping["claim_boundary"]
            != PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY
        ):
            raise ValueError("provider support request claim boundary changed")
        raw_streams = _strict_list(mapping["streams"], name="streams")
        request = cls(
            protocol_id=mapping["protocol_id"],
            source_repository=mapping["source_repository"],
            source_revision=mapping["source_revision"],
            provider_family=mapping["provider_family"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            promotion_lock_id=mapping["promotion_lock_id"],
            coordinate_semantics=mapping["coordinate_semantics"],
            admission_rule=mapping["admission_rule"],
            minimum_supported_fraction=mapping[
                "minimum_supported_fraction"
            ],
            permitted_technical_exclusion_codes=_string_tuple_from_json(
                mapping["permitted_technical_exclusion_codes"],
                name="permitted_technical_exclusion_codes",
            ),
            maximum_technical_exclusions=mapping[
                "maximum_technical_exclusions"
            ],
            prediction_payloads_opened=mapping[
                "prediction_payloads_opened"
            ],
            residuals_used=mapping["residuals_used"],
            target_outcomes_used=mapping["target_outcomes_used"],
            streams=tuple(
                ProviderSupportStreamV1.from_dict(item) for item in raw_streams
            ),
            metadata=_strict_mapping(mapping["metadata"], name="metadata"),
        )
        expected_id = _strict_digest(
            mapping["request_id"],
            name="request_id",
            pattern=_SHA256,
        )
        if request.request_id != expected_id:
            raise ValueError("provider support request identity mismatch")
        return request


@dataclass(frozen=True, slots=True)
class ProviderSupportStreamEvaluationV1:
    """Deterministic support decision for one stream."""

    group_id: str
    stream_id: str
    supported: bool
    excluded_from_admission: bool
    reason_codes: tuple[str, ...]
    required_frame_count: int
    available_required_frame_count: int
    geometry_supported_required_frame_count: int
    geometry_support_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "group_id",
            _strict_string(self.group_id, name="group_id"),
        )
        object.__setattr__(
            self,
            "stream_id",
            _strict_string(self.stream_id, name="stream_id"),
        )
        object.__setattr__(
            self,
            "supported",
            _strict_bool(self.supported, name="supported"),
        )
        excluded = _strict_bool(
            self.excluded_from_admission,
            name="excluded_from_admission",
        )
        if self.supported and excluded:
            raise ValueError("a supported stream cannot be excluded")
        reasons = _canonical_string_tuple(
            self.reason_codes,
            name="reason_codes",
        )
        if self.supported and reasons:
            raise ValueError("supported streams must not have reason codes")
        if not self.supported and not reasons:
            raise ValueError("unsupported streams must have reason codes")
        required = _strict_integer(
            self.required_frame_count,
            name="required_frame_count",
            minimum=1,
        )
        available = _strict_integer(
            self.available_required_frame_count,
            name="available_required_frame_count",
            minimum=0,
        )
        geometry = _strict_integer(
            self.geometry_supported_required_frame_count,
            name="geometry_supported_required_frame_count",
            minimum=0,
        )
        if available > required:
            raise ValueError(
                "available_required_frame_count exceeds required_frame_count"
            )
        if geometry > available:
            raise ValueError(
                "geometry-supported required frames exceed available required frames"
            )
        fraction = _strict_real(
            self.geometry_support_fraction,
            name="geometry_support_fraction",
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("geometry_support_fraction must lie in [0, 1]")
        if abs(fraction - geometry / required) > 1.0e-15:
            raise ValueError("geometry_support_fraction is inconsistent")
        object.__setattr__(self, "excluded_from_admission", excluded)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "required_frame_count", required)
        object.__setattr__(
            self,
            "available_required_frame_count",
            available,
        )
        object.__setattr__(
            self,
            "geometry_supported_required_frame_count",
            geometry,
        )
        object.__setattr__(self, "geometry_support_fraction", fraction)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "stream_id": self.stream_id,
            "supported": self.supported,
            "excluded_from_admission": self.excluded_from_admission,
            "reason_codes": list(self.reason_codes),
            "required_frame_count": self.required_frame_count,
            "available_required_frame_count": (
                self.available_required_frame_count
            ),
            "geometry_supported_required_frame_count": (
                self.geometry_supported_required_frame_count
            ),
            "geometry_support_fraction": self.geometry_support_fraction,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportStreamEvaluationV1:
        mapping = _strict_mapping(value, name="provider support stream result")
        _exact_keys(
            mapping,
            _STREAM_RESULT_FIELDS,
            name="provider support stream result",
        )
        return cls(
            group_id=mapping["group_id"],
            stream_id=mapping["stream_id"],
            supported=mapping["supported"],
            excluded_from_admission=mapping["excluded_from_admission"],
            reason_codes=_string_tuple_from_json(
                mapping["reason_codes"],
                name="reason_codes",
            ),
            required_frame_count=mapping["required_frame_count"],
            available_required_frame_count=mapping[
                "available_required_frame_count"
            ],
            geometry_supported_required_frame_count=mapping[
                "geometry_supported_required_frame_count"
            ],
            geometry_support_fraction=mapping[
                "geometry_support_fraction"
            ],
        )


@dataclass(frozen=True, slots=True)
class ProviderSupportFeasibilityV1:
    """Replayable cohort-level support-feasibility result."""

    request: ProviderSupportFeasibilityRequestV1
    stream_results: tuple[ProviderSupportStreamEvaluationV1, ...]
    stream_count: int
    supported_stream_count: int
    excluded_stream_count: int
    evaluable_stream_count: int
    support_fraction: float
    technical_exclusion_count: int
    support_feasible: bool
    decision_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProviderSupportFeasibilityRequestV1):
            raise ValueError(
                "request must be a ProviderSupportFeasibilityRequestV1"
            )
        if type(self.stream_results) is not tuple or not all(
            isinstance(item, ProviderSupportStreamEvaluationV1)
            for item in self.stream_results
        ):
            raise ValueError(
                "stream_results must contain ProviderSupportStreamEvaluationV1"
            )
        results = tuple(
            sorted(
                self.stream_results,
                key=lambda item: (item.group_id, item.stream_id),
            )
        )
        if tuple((item.group_id, item.stream_id) for item in results) != tuple(
            item.key for item in self.request.streams
        ):
            raise ValueError("stream result keys do not match request streams")
        stream_count = _strict_integer(
            self.stream_count,
            name="stream_count",
            minimum=1,
        )
        supported = _strict_integer(
            self.supported_stream_count,
            name="supported_stream_count",
            minimum=0,
        )
        excluded = _strict_integer(
            self.excluded_stream_count,
            name="excluded_stream_count",
            minimum=0,
        )
        evaluable = _strict_integer(
            self.evaluable_stream_count,
            name="evaluable_stream_count",
            minimum=0,
        )
        technical = _strict_integer(
            self.technical_exclusion_count,
            name="technical_exclusion_count",
            minimum=0,
        )
        if stream_count != len(results):
            raise ValueError("stream_count is inconsistent")
        if supported != sum(item.supported for item in results):
            raise ValueError("supported_stream_count is inconsistent")
        if excluded != sum(item.excluded_from_admission for item in results):
            raise ValueError("excluded_stream_count is inconsistent")
        if evaluable != stream_count - excluded:
            raise ValueError("evaluable_stream_count is inconsistent")
        fraction = _strict_real(
            self.support_fraction,
            name="support_fraction",
        )
        expected_fraction = supported / evaluable if evaluable else 0.0
        if abs(fraction - expected_fraction) > 1.0e-15:
            raise ValueError("support_fraction is inconsistent")
        feasible = _strict_bool(
            self.support_feasible,
            name="support_feasible",
        )
        decision = _strict_string(
            self.decision_reason,
            name="decision_reason",
        )
        object.__setattr__(self, "stream_results", results)
        object.__setattr__(self, "stream_count", stream_count)
        object.__setattr__(self, "supported_stream_count", supported)
        object.__setattr__(self, "excluded_stream_count", excluded)
        object.__setattr__(self, "evaluable_stream_count", evaluable)
        object.__setattr__(self, "support_fraction", fraction)
        object.__setattr__(self, "technical_exclusion_count", technical)
        object.__setattr__(self, "support_feasible", feasible)
        object.__setattr__(self, "decision_reason", decision)

    def _identity_dict(self) -> dict[str, object]:
        return {
            "schema_name": PROVIDER_SUPPORT_FEASIBILITY_SCHEMA,
            "schema_version": PROVIDER_SUPPORT_FEASIBILITY_VERSION,
            "request": self.request.to_dict(),
            "stream_results": [item.to_dict() for item in self.stream_results],
            "stream_count": self.stream_count,
            "supported_stream_count": self.supported_stream_count,
            "excluded_stream_count": self.excluded_stream_count,
            "evaluable_stream_count": self.evaluable_stream_count,
            "support_fraction": self.support_fraction,
            "technical_exclusion_count": self.technical_exclusion_count,
            "support_feasible": self.support_feasible,
            "decision_reason": self.decision_reason,
            "claim_boundary": PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY,
        }

    @property
    def provider_support_feasibility_id(self) -> str:
        return _sha256_json(self._identity_dict())

    def to_dict(self) -> dict[str, object]:
        payload = self._identity_dict()
        payload["provider_support_feasibility_id"] = (
            self.provider_support_feasibility_id
        )
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ProviderSupportFeasibilityV1:
        mapping = _strict_mapping(
            value,
            name="provider support feasibility",
        )
        _exact_keys(
            mapping,
            _RESULT_FIELDS,
            name="provider support feasibility",
        )
        if mapping["schema_name"] != PROVIDER_SUPPORT_FEASIBILITY_SCHEMA:
            raise ValueError("unsupported provider support feasibility schema")
        if mapping["schema_version"] != PROVIDER_SUPPORT_FEASIBILITY_VERSION:
            raise ValueError("unsupported provider support feasibility version")
        if (
            mapping["claim_boundary"]
            != PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY
        ):
            raise ValueError("provider support feasibility claim boundary changed")
        request = ProviderSupportFeasibilityRequestV1.from_dict(
            mapping["request"]
        )
        expected = evaluate_provider_support_feasibility(request)
        identifier = _strict_digest(
            mapping["provider_support_feasibility_id"],
            name="provider_support_feasibility_id",
            pattern=_SHA256,
        )
        if identifier != expected.provider_support_feasibility_id:
            raise ValueError("provider support feasibility identity mismatch")
        if plain_json(mapping) != expected.to_dict():
            raise ValueError(
                "provider support feasibility does not replay from its request"
            )
        return expected


def _evaluate_stream(
    stream: ProviderSupportStreamV1,
    *,
    permitted_exclusions: frozenset[str],
    exclusion_budget_ok: bool,
) -> ProviderSupportStreamEvaluationV1:
    required = set(stream.required_frame_ids)
    available_count = len(required.intersection(stream.available_frame_ids))
    geometry_count = len(
        required.intersection(stream.geometry_supported_frame_ids)
    )
    geometry_fraction = geometry_count / len(required)
    reasons: list[str] = []
    excluded = False
    technical_failure = stream.technical_failure_code
    if technical_failure is not None:
        if technical_failure in permitted_exclusions and exclusion_budget_ok:
            excluded = True
            reasons.append("permitted-technical-exclusion")
        elif technical_failure in permitted_exclusions:
            reasons.append("technical-exclusion-budget-exceeded")
        else:
            reasons.append("unpermitted-technical-failure")
    if available_count != len(required):
        reasons.append("missing-required-frames")
    if geometry_fraction < stream.minimum_geometry_support_fraction:
        reasons.append("insufficient-geometry-support")
    if stream.intrinsics_required and stream.intrinsics_id is None:
        reasons.append("missing-intrinsics")
    if stream.extrinsics_required and stream.extrinsics_id is None:
        reasons.append("missing-extrinsics")
    if stream.metric_anchor_required and stream.metric_anchor_id is None:
        reasons.append("missing-metric-anchor")
    reasons_tuple = tuple(sorted(set(reasons)))
    return ProviderSupportStreamEvaluationV1(
        group_id=stream.group_id,
        stream_id=stream.stream_id,
        supported=not reasons_tuple,
        excluded_from_admission=excluded,
        reason_codes=reasons_tuple,
        required_frame_count=len(required),
        available_required_frame_count=available_count,
        geometry_supported_required_frame_count=geometry_count,
        geometry_support_fraction=geometry_fraction,
    )


def evaluate_provider_support_feasibility(
    request: ProviderSupportFeasibilityRequestV1,
) -> ProviderSupportFeasibilityV1:
    """Evaluate one frozen outcome-blind request deterministically."""

    if not isinstance(request, ProviderSupportFeasibilityRequestV1):
        raise TypeError(
            "request must be a ProviderSupportFeasibilityRequestV1"
        )
    permitted = frozenset(request.permitted_technical_exclusion_codes)
    technical_exclusion_count = sum(
        stream.technical_failure_code in permitted
        for stream in request.streams
    )
    exclusion_budget_ok = (
        technical_exclusion_count <= request.maximum_technical_exclusions
    )
    results = tuple(
        _evaluate_stream(
            stream,
            permitted_exclusions=permitted,
            exclusion_budget_ok=exclusion_budget_ok,
        )
        for stream in request.streams
    )
    stream_count = len(results)
    supported = sum(item.supported for item in results)
    excluded = sum(item.excluded_from_admission for item in results)
    evaluable = stream_count - excluded
    support_fraction = supported / evaluable if evaluable else 0.0
    if not exclusion_budget_ok:
        feasible = False
        decision = "technical-exclusion-budget-exceeded"
    elif evaluable == 0:
        feasible = False
        decision = "no-evaluable-streams"
    elif support_fraction < request.minimum_supported_fraction:
        feasible = False
        decision = "support-threshold-not-met"
    else:
        feasible = True
        decision = "support-feasible"
    return ProviderSupportFeasibilityV1(
        request=request,
        stream_results=results,
        stream_count=stream_count,
        supported_stream_count=supported,
        excluded_stream_count=excluded,
        evaluable_stream_count=evaluable,
        support_fraction=support_fraction,
        technical_exclusion_count=technical_exclusion_count,
        support_feasible=feasible,
        decision_reason=decision,
    )


def write_provider_support_feasibility_request(
    path: Path,
    request: ProviderSupportFeasibilityRequestV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(request, ProviderSupportFeasibilityRequestV1):
        raise TypeError(
            "request must be a ProviderSupportFeasibilityRequestV1"
        )
    _atomic_write_json(path, request.to_dict(), overwrite=overwrite)


def load_provider_support_feasibility_request(
    path: Path,
) -> ProviderSupportFeasibilityRequestV1:
    value, _ = _load_json(
        path,
        name="provider support feasibility request",
    )
    return ProviderSupportFeasibilityRequestV1.from_dict(value)


def write_provider_support_feasibility(
    path: Path,
    result: ProviderSupportFeasibilityV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(result, ProviderSupportFeasibilityV1):
        raise TypeError("result must be a ProviderSupportFeasibilityV1")
    _atomic_write_json(path, result.to_dict(), overwrite=overwrite)


def load_provider_support_feasibility(
    path: Path,
) -> ProviderSupportFeasibilityV1:
    value, _ = _load_json(path, name="provider support feasibility")
    return ProviderSupportFeasibilityV1.from_dict(value)


def _summary(result: ProviderSupportFeasibilityV1) -> dict[str, object]:
    return {
        "provider_support_feasibility_id": (
            result.provider_support_feasibility_id
        ),
        "request_id": result.request.request_id,
        "support_feasible": result.support_feasible,
        "decision_reason": result.decision_reason,
        "stream_count": result.stream_count,
        "supported_stream_count": result.supported_stream_count,
        "excluded_stream_count": result.excluded_stream_count,
        "evaluable_stream_count": result.evaluable_stream_count,
        "support_fraction": result.support_fraction,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a frozen request and write a replayable result",
    )
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument("--compact", action="store_true")
    verify = subparsers.add_parser(
        "verify",
        help="verify and replay a support-feasibility result",
    )
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        request = load_provider_support_feasibility_request(args.request)
        result = evaluate_provider_support_feasibility(request)
        write_provider_support_feasibility(
            args.output,
            result,
            overwrite=args.overwrite,
        )
    else:
        result = load_provider_support_feasibility(args.artifact)
    print(
        json.dumps(
            _summary(result),
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0 if result.support_feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROVIDER_SUPPORT_FEASIBILITY_CLAIM_BOUNDARY",
    "PROVIDER_SUPPORT_FEASIBILITY_REQUEST_SCHEMA",
    "PROVIDER_SUPPORT_FEASIBILITY_REQUEST_VERSION",
    "PROVIDER_SUPPORT_FEASIBILITY_SCHEMA",
    "PROVIDER_SUPPORT_FEASIBILITY_VERSION",
    "ProviderSupportFeasibilityRequestV1",
    "ProviderSupportFeasibilityV1",
    "ProviderSupportStreamEvaluationV1",
    "ProviderSupportStreamV1",
    "evaluate_provider_support_feasibility",
    "load_provider_support_feasibility",
    "load_provider_support_feasibility_request",
    "main",
    "write_provider_support_feasibility",
    "write_provider_support_feasibility_request",
]
