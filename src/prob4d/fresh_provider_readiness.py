"""Prospective fresh-provider readiness and failure localization.

This module composes already-frozen support, source competence, covariance,
query-relevance, and fallback evidence into one terminal decision before a protected
target cohort is opened. It does not replace the existing held-out promotion gate;
it supplies the missing pre-target authorization and explicit stop classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
)
from .source_provider_competence import SourceProviderCompetenceReportV1

FRESH_PROVIDER_COHORT_LOCK_SCHEMA = "prob4d.fresh-provider-cohort-lock"
FRESH_PROVIDER_READINESS_REQUEST_SCHEMA = "prob4d.fresh-provider-readiness-request"
FRESH_PROVIDER_READINESS_DECISION_SCHEMA = "prob4d.fresh-provider-readiness-decision"
FRESH_PROVIDER_TARGET_AUTHORIZATION_SCHEMA = (
    "prob4d.fresh-provider-target-authorization"
)
FRESH_PROVIDER_READINESS_VERSION = 1
FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY = (
    "This artifact composes target-free support and source/calibration evidence into "
    "one terminal readiness or failure-localization decision for an exact frozen "
    "provider route. It does not establish real-provider competence, target benefit, "
    "BayesianPhysTwin benefit, Causal4D intervention benefit, deployment safety, or "
    "state of the art. A target authorization permits exactly one evaluation of the "
    "bound unopened target roster and does not permit target-side retuning."
)

GateName = Literal[
    "support-feasibility",
    "source-mean",
    "identity-reliability",
    "gauge-dependence",
    "point-covariance",
    "query-relevance",
    "exact-fallback",
]
GateStatus = Literal["pass", "fail", "not-evaluated", "technical-failure"]
ReadinessClassification = Literal[
    "support-negative",
    "source-mean-negative",
    "identity-or-association-negative",
    "gauge-or-dependence-negative",
    "point-covariance-localized",
    "query-irrelevant-or-nonidentifiable",
    "ready-for-one-target-evaluation",
    "technical-failure",
]

_GATE_ORDER: tuple[GateName, ...] = (
    "support-feasibility",
    "source-mean",
    "identity-reliability",
    "gauge-dependence",
    "point-covariance",
    "query-relevance",
    "exact-fallback",
)
_GATE_FAILURE_CLASSIFICATION: dict[GateName, ReadinessClassification] = {
    "support-feasibility": "support-negative",
    "source-mean": "source-mean-negative",
    "identity-reliability": "identity-or-association-negative",
    "gauge-dependence": "gauge-or-dependence-negative",
    "point-covariance": "point-covariance-localized",
    "query-relevance": "query-irrelevant-or-nonidentifiable",
    "exact-fallback": "technical-failure",
}
_NEXT_ACTION: dict[ReadinessClassification, str] = {
    "support-negative": "stop-provider-version-before-opening-predictions-or-residuals",
    "source-mean-negative": "stop-provider-version-without-richer-covariance",
    "identity-or-association-negative": "improve-identities-or-association-on-source-only",
    "gauge-or-dependence-negative": "localize-gauge-or-dependence-on-source-only",
    "point-covariance-localized": "authorize-source-only-point-uncertainty-development",
    "query-irrelevant-or-nonidentifiable": "retain-exact-physical-fallback",
    "ready-for-one-target-evaluation": "open-exactly-one-bound-target-evaluation",
    "technical-failure": "retain-evidence-and-repair-only-under-a-new-reviewed-execution",
}

_LOCK_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_id",
        "source_repository",
        "source_revision",
        "provider_repository",
        "provider_revision",
        "model_set_id",
        "loader_id",
        "cohort_binding_id",
        "promotion_lock_id",
        "query_definition_id",
        "fallback_identity_id",
        "development_group_ids",
        "calibration_group_ids",
        "target_group_ids",
        "confirmation_group_ids",
        "target_payloads_opened",
        "target_outcomes_opened",
        "confirmation_payloads_opened",
        "metadata",
        "claim_boundary",
        "fresh_provider_cohort_lock_id",
    }
)
_GATE_FIELDS = frozenset(
    {
        "gate_name",
        "status",
        "evidence_id",
        "reason_codes",
        "metadata",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "cohort_lock",
        "gates",
        "metadata",
        "claim_boundary",
        "fresh_provider_readiness_request_id",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "request",
        "classification",
        "terminal_gate",
        "decision_reasons",
        "next_action",
        "authorize_point_uncertainty_development",
        "authorize_target_evaluation",
        "target_evaluation_budget",
        "claim_boundary",
        "fresh_provider_readiness_decision_id",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "decision",
        "target_group_ids",
        "target_evaluation_budget",
        "target_payloads_opened_at_authorization",
        "target_outcomes_opened_at_authorization",
        "claim_boundary",
        "fresh_provider_target_authorization_id",
    }
)


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _repository(value: object, *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result.count("/") != 1 or result.startswith("/") or result.endswith("/"):
        raise ValueError(f"{name} must use canonical owner/name form")
    return result


def _sorted_unique_strings(
    value: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a canonical tuple")
    result = tuple(
        require_exact_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _strings_from_json(value: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return _sorted_unique_strings(tuple(value), name=name, allow_empty=allow_empty)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    destination = Path(path)
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-"
        f"{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _gate_name(value: object) -> GateName:
    result = require_exact_string(value, name="gate_name")
    if result not in _GATE_ORDER:
        raise ValueError(f"gate_name must be one of {list(_GATE_ORDER)}")
    return cast(GateName, result)


def _gate_status(value: object) -> GateStatus:
    result = require_exact_string(value, name="status")
    if result not in {"pass", "fail", "not-evaluated", "technical-failure"}:
        raise ValueError("status is not a supported readiness gate status")
    return cast(GateStatus, result)


@dataclass(frozen=True, slots=True)
class FreshProviderCohortLockV1:
    """Target-closed binding of the complete prospective provider study."""

    protocol_id: str
    source_repository: str
    source_revision: str
    provider_repository: str
    provider_revision: str
    model_set_id: str
    loader_id: str
    cohort_binding_id: str
    promotion_lock_id: str
    query_definition_id: str
    fallback_identity_id: str
    development_group_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    confirmation_group_ids: tuple[str, ...] = ()
    target_payloads_opened: bool = False
    target_outcomes_opened: bool = False
    confirmation_payloads_opened: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fresh_provider_cohort_lock_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            require_exact_string(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "source_repository",
            _repository(self.source_repository, name="source_repository"),
        )
        object.__setattr__(
            self,
            "source_revision",
            require_revision(self.source_revision, name="source_revision"),
        )
        object.__setattr__(
            self,
            "provider_repository",
            _repository(self.provider_repository, name="provider_repository"),
        )
        object.__setattr__(
            self,
            "provider_revision",
            require_revision(self.provider_revision, name="provider_revision"),
        )
        for name in (
            "model_set_id",
            "loader_id",
            "cohort_binding_id",
            "promotion_lock_id",
            "query_definition_id",
            "fallback_identity_id",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        rosters = {
            "development_group_ids": _sorted_unique_strings(
                self.development_group_ids,
                name="development_group_ids",
                allow_empty=False,
            ),
            "calibration_group_ids": _sorted_unique_strings(
                self.calibration_group_ids,
                name="calibration_group_ids",
                allow_empty=False,
            ),
            "target_group_ids": _sorted_unique_strings(
                self.target_group_ids,
                name="target_group_ids",
                allow_empty=False,
            ),
            "confirmation_group_ids": _sorted_unique_strings(
                self.confirmation_group_ids,
                name="confirmation_group_ids",
                allow_empty=True,
            ),
        }
        roster_names = tuple(rosters)
        for index, first_name in enumerate(roster_names):
            first = set(rosters[first_name])
            for second_name in roster_names[index + 1 :]:
                overlap = first & set(rosters[second_name])
                if overlap:
                    raise ValueError(
                        f"{first_name} and {second_name} overlap: {sorted(overlap)}"
                    )
        for name, roster in rosters.items():
            object.__setattr__(self, name, roster)
        target_payloads_opened = _strict_bool(
            self.target_payloads_opened,
            name="target_payloads_opened",
        )
        target_outcomes_opened = _strict_bool(
            self.target_outcomes_opened,
            name="target_outcomes_opened",
        )
        confirmation_payloads_opened = _strict_bool(
            self.confirmation_payloads_opened,
            name="confirmation_payloads_opened",
        )
        if target_payloads_opened or target_outcomes_opened or confirmation_payloads_opened:
            raise ValueError("fresh-provider readiness requires unopened protected cohorts")
        object.__setattr__(self, "target_payloads_opened", target_payloads_opened)
        object.__setattr__(self, "target_outcomes_opened", target_outcomes_opened)
        object.__setattr__(
            self,
            "confirmation_payloads_opened",
            confirmation_payloads_opened,
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="cohort-lock metadata"),
        )
        object.__setattr__(
            self,
            "fresh_provider_cohort_lock_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": FRESH_PROVIDER_COHORT_LOCK_SCHEMA,
            "schema_version": FRESH_PROVIDER_READINESS_VERSION,
            "protocol_id": self.protocol_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "cohort_binding_id": self.cohort_binding_id,
            "promotion_lock_id": self.promotion_lock_id,
            "query_definition_id": self.query_definition_id,
            "fallback_identity_id": self.fallback_identity_id,
            "development_group_ids": list(self.development_group_ids),
            "calibration_group_ids": list(self.calibration_group_ids),
            "target_group_ids": list(self.target_group_ids),
            "confirmation_group_ids": list(self.confirmation_group_ids),
            "target_payloads_opened": self.target_payloads_opened,
            "target_outcomes_opened": self.target_outcomes_opened,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "metadata": plain_json(self.metadata),
            "claim_boundary": FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["fresh_provider_cohort_lock_id"] = self.fresh_provider_cohort_lock_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> FreshProviderCohortLockV1:
        mapping = require_mapping(value, name="fresh provider cohort lock")
        require_exact_fields(mapping, _LOCK_FIELDS, name="fresh provider cohort lock")
        if mapping["schema"] != FRESH_PROVIDER_COHORT_LOCK_SCHEMA:
            raise ValueError("fresh provider cohort lock schema changed")
        if mapping["schema_version"] != FRESH_PROVIDER_READINESS_VERSION:
            raise ValueError("fresh provider cohort lock version changed")
        if mapping["claim_boundary"] != FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY:
            raise ValueError("fresh provider cohort lock claim boundary changed")
        result = cls(
            protocol_id=mapping["protocol_id"],
            source_repository=mapping["source_repository"],
            source_revision=mapping["source_revision"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            cohort_binding_id=mapping["cohort_binding_id"],
            promotion_lock_id=mapping["promotion_lock_id"],
            query_definition_id=mapping["query_definition_id"],
            fallback_identity_id=mapping["fallback_identity_id"],
            development_group_ids=_strings_from_json(
                mapping["development_group_ids"],
                name="development_group_ids",
                allow_empty=False,
            ),
            calibration_group_ids=_strings_from_json(
                mapping["calibration_group_ids"],
                name="calibration_group_ids",
                allow_empty=False,
            ),
            target_group_ids=_strings_from_json(
                mapping["target_group_ids"],
                name="target_group_ids",
                allow_empty=False,
            ),
            confirmation_group_ids=_strings_from_json(
                mapping["confirmation_group_ids"],
                name="confirmation_group_ids",
                allow_empty=True,
            ),
            target_payloads_opened=mapping["target_payloads_opened"],
            target_outcomes_opened=mapping["target_outcomes_opened"],
            confirmation_payloads_opened=mapping["confirmation_payloads_opened"],
            metadata=require_finite_json_mapping(
                mapping["metadata"], name="cohort-lock metadata"
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("fresh provider cohort lock derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class ReadinessGateV1:
    """One evidence-bound gate in the prospective information order."""

    gate_name: GateName
    status: GateStatus
    evidence_id: str | None
    reason_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _gate_name(self.gate_name)
        status = _gate_status(self.status)
        evidence_id = (
            None
            if self.evidence_id is None
            else require_sha256(self.evidence_id, name="evidence_id")
        )
        reasons = _sorted_unique_strings(
            self.reason_codes,
            name="reason_codes",
            allow_empty=True,
        )
        if status == "not-evaluated":
            if evidence_id is not None or reasons:
                raise ValueError("not-evaluated gates must not carry evidence or reasons")
        else:
            if evidence_id is None:
                raise ValueError("evaluated gates require evidence_id")
            if status == "pass" and reasons:
                raise ValueError("passing gates must not carry failure reasons")
            if status != "pass" and not reasons:
                raise ValueError("failed gates require at least one reason code")
        object.__setattr__(self, "gate_name", name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="gate metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_name": self.gate_name,
            "status": self.status,
            "evidence_id": self.evidence_id,
            "reason_codes": list(self.reason_codes),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> ReadinessGateV1:
        mapping = require_mapping(value, name="readiness gate")
        require_exact_fields(mapping, _GATE_FIELDS, name="readiness gate")
        return cls(
            gate_name=mapping["gate_name"],
            status=mapping["status"],
            evidence_id=mapping["evidence_id"],
            reason_codes=_strings_from_json(
                mapping["reason_codes"],
                name="reason_codes",
                allow_empty=True,
            ),
            metadata=require_finite_json_mapping(
                mapping["metadata"], name="gate metadata"
            ),
        )


@dataclass(frozen=True, slots=True)
class FreshProviderReadinessRequestV1:
    """Complete ordered evidence request for one pre-target decision."""

    cohort_lock: FreshProviderCohortLockV1
    gates: tuple[ReadinessGateV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fresh_provider_readiness_request_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cohort_lock, FreshProviderCohortLockV1):
            raise TypeError("cohort_lock must be FreshProviderCohortLockV1")
        if type(self.gates) is not tuple or len(self.gates) != len(_GATE_ORDER):
            raise ValueError("gates must contain the complete canonical gate order")
        if any(not isinstance(gate, ReadinessGateV1) for gate in self.gates):
            raise TypeError("gates must contain ReadinessGateV1 values")
        if tuple(gate.gate_name for gate in self.gates) != _GATE_ORDER:
            raise ValueError("gates must follow the canonical prospective order")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="readiness metadata"),
        )
        object.__setattr__(
            self,
            "fresh_provider_readiness_request_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": FRESH_PROVIDER_READINESS_REQUEST_SCHEMA,
            "schema_version": FRESH_PROVIDER_READINESS_VERSION,
            "cohort_lock": self.cohort_lock.to_dict(),
            "gates": [gate.to_dict() for gate in self.gates],
            "metadata": plain_json(self.metadata),
            "claim_boundary": FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["fresh_provider_readiness_request_id"] = (
            self.fresh_provider_readiness_request_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> FreshProviderReadinessRequestV1:
        mapping = require_mapping(value, name="fresh provider readiness request")
        require_exact_fields(
            mapping,
            _REQUEST_FIELDS,
            name="fresh provider readiness request",
        )
        if mapping["schema"] != FRESH_PROVIDER_READINESS_REQUEST_SCHEMA:
            raise ValueError("fresh provider readiness request schema changed")
        if mapping["schema_version"] != FRESH_PROVIDER_READINESS_VERSION:
            raise ValueError("fresh provider readiness request version changed")
        if mapping["claim_boundary"] != FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY:
            raise ValueError("fresh provider readiness request claim boundary changed")
        raw_gates = mapping["gates"]
        if not isinstance(raw_gates, list):
            raise ValueError("gates must be a JSON array")
        result = cls(
            cohort_lock=FreshProviderCohortLockV1.from_dict(mapping["cohort_lock"]),
            gates=tuple(ReadinessGateV1.from_dict(item) for item in raw_gates),
            metadata=require_finite_json_mapping(
                mapping["metadata"], name="readiness metadata"
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("fresh provider readiness request derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class FreshProviderReadinessDecisionV1:
    """Terminal pre-target decision derived from one complete request."""

    request: FreshProviderReadinessRequestV1
    classification: ReadinessClassification
    terminal_gate: GateName
    decision_reasons: tuple[str, ...]
    next_action: str
    authorize_point_uncertainty_development: bool
    authorize_target_evaluation: bool
    target_evaluation_budget: int
    fresh_provider_readiness_decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, FreshProviderReadinessRequestV1):
            raise TypeError("request must be FreshProviderReadinessRequestV1")
        classification = require_exact_string(
            self.classification,
            name="classification",
        )
        if classification not in _NEXT_ACTION:
            raise ValueError("classification is not a supported terminal decision")
        terminal_gate = _gate_name(self.terminal_gate)
        reasons = _sorted_unique_strings(
            self.decision_reasons,
            name="decision_reasons",
            allow_empty=False,
        )
        next_action = require_exact_string(self.next_action, name="next_action")
        point_development = _strict_bool(
            self.authorize_point_uncertainty_development,
            name="authorize_point_uncertainty_development",
        )
        target = _strict_bool(
            self.authorize_target_evaluation,
            name="authorize_target_evaluation",
        )
        budget = require_exact_integer(
            self.target_evaluation_budget,
            name="target_evaluation_budget",
            minimum=0,
        )
        expected_point = classification == "point-covariance-localized"
        expected_target = classification == "ready-for-one-target-evaluation"
        if point_development != expected_point:
            raise ValueError("point-uncertainty authorization changed")
        if target != expected_target or budget != (1 if expected_target else 0):
            raise ValueError("target-evaluation authorization changed")
        if next_action != _NEXT_ACTION[classification]:
            raise ValueError("next_action changed")
        object.__setattr__(
            self,
            "classification",
            cast(ReadinessClassification, classification),
        )
        object.__setattr__(self, "terminal_gate", terminal_gate)
        object.__setattr__(self, "decision_reasons", reasons)
        object.__setattr__(self, "next_action", next_action)
        object.__setattr__(
            self,
            "authorize_point_uncertainty_development",
            point_development,
        )
        object.__setattr__(self, "authorize_target_evaluation", target)
        object.__setattr__(self, "target_evaluation_budget", budget)
        object.__setattr__(
            self,
            "fresh_provider_readiness_decision_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": FRESH_PROVIDER_READINESS_DECISION_SCHEMA,
            "schema_version": FRESH_PROVIDER_READINESS_VERSION,
            "request": self.request.to_dict(),
            "classification": self.classification,
            "terminal_gate": self.terminal_gate,
            "decision_reasons": list(self.decision_reasons),
            "next_action": self.next_action,
            "authorize_point_uncertainty_development": (
                self.authorize_point_uncertainty_development
            ),
            "authorize_target_evaluation": self.authorize_target_evaluation,
            "target_evaluation_budget": self.target_evaluation_budget,
            "claim_boundary": FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["fresh_provider_readiness_decision_id"] = (
            self.fresh_provider_readiness_decision_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> FreshProviderReadinessDecisionV1:
        mapping = require_mapping(value, name="fresh provider readiness decision")
        require_exact_fields(
            mapping,
            _DECISION_FIELDS,
            name="fresh provider readiness decision",
        )
        if mapping["schema"] != FRESH_PROVIDER_READINESS_DECISION_SCHEMA:
            raise ValueError("fresh provider readiness decision schema changed")
        if mapping["schema_version"] != FRESH_PROVIDER_READINESS_VERSION:
            raise ValueError("fresh provider readiness decision version changed")
        if mapping["claim_boundary"] != FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY:
            raise ValueError("fresh provider readiness decision claim boundary changed")
        result = cls(
            request=FreshProviderReadinessRequestV1.from_dict(mapping["request"]),
            classification=mapping["classification"],
            terminal_gate=mapping["terminal_gate"],
            decision_reasons=_strings_from_json(
                mapping["decision_reasons"],
                name="decision_reasons",
                allow_empty=False,
            ),
            next_action=mapping["next_action"],
            authorize_point_uncertainty_development=mapping[
                "authorize_point_uncertainty_development"
            ],
            authorize_target_evaluation=mapping["authorize_target_evaluation"],
            target_evaluation_budget=mapping["target_evaluation_budget"],
        )
        expected = evaluate_fresh_provider_readiness(result.request)
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("fresh provider readiness decision derived fields changed")
        if result.fresh_provider_readiness_decision_id != (
            expected.fresh_provider_readiness_decision_id
        ):
            raise ValueError("fresh provider readiness decision replay changed")
        return result


@dataclass(frozen=True, slots=True)
class FreshProviderTargetAuthorizationV1:
    """One-shot authorization for the exact unopened target roster."""

    decision: FreshProviderReadinessDecisionV1
    target_group_ids: tuple[str, ...]
    target_evaluation_budget: int = 1
    target_payloads_opened_at_authorization: bool = False
    target_outcomes_opened_at_authorization: bool = False
    fresh_provider_target_authorization_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, FreshProviderReadinessDecisionV1):
            raise TypeError("decision must be FreshProviderReadinessDecisionV1")
        if not self.decision.authorize_target_evaluation:
            raise ValueError("decision does not authorize target evaluation")
        target_ids = _sorted_unique_strings(
            self.target_group_ids,
            name="target_group_ids",
            allow_empty=False,
        )
        if target_ids != self.decision.request.cohort_lock.target_group_ids:
            raise ValueError("target_group_ids changed from the cohort lock")
        budget = require_exact_integer(
            self.target_evaluation_budget,
            name="target_evaluation_budget",
            minimum=1,
        )
        if budget != 1:
            raise ValueError("fresh provider authorization permits exactly one evaluation")
        payloads = _strict_bool(
            self.target_payloads_opened_at_authorization,
            name="target_payloads_opened_at_authorization",
        )
        outcomes = _strict_bool(
            self.target_outcomes_opened_at_authorization,
            name="target_outcomes_opened_at_authorization",
        )
        if payloads or outcomes:
            raise ValueError("target must still be unopened when authorization is created")
        object.__setattr__(self, "target_group_ids", target_ids)
        object.__setattr__(self, "target_evaluation_budget", budget)
        object.__setattr__(self, "target_payloads_opened_at_authorization", payloads)
        object.__setattr__(self, "target_outcomes_opened_at_authorization", outcomes)
        object.__setattr__(
            self,
            "fresh_provider_target_authorization_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": FRESH_PROVIDER_TARGET_AUTHORIZATION_SCHEMA,
            "schema_version": FRESH_PROVIDER_READINESS_VERSION,
            "decision": self.decision.to_dict(),
            "target_group_ids": list(self.target_group_ids),
            "target_evaluation_budget": self.target_evaluation_budget,
            "target_payloads_opened_at_authorization": (
                self.target_payloads_opened_at_authorization
            ),
            "target_outcomes_opened_at_authorization": (
                self.target_outcomes_opened_at_authorization
            ),
            "claim_boundary": FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["fresh_provider_target_authorization_id"] = (
            self.fresh_provider_target_authorization_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> FreshProviderTargetAuthorizationV1:
        mapping = require_mapping(value, name="fresh provider target authorization")
        require_exact_fields(
            mapping,
            _AUTHORIZATION_FIELDS,
            name="fresh provider target authorization",
        )
        if mapping["schema"] != FRESH_PROVIDER_TARGET_AUTHORIZATION_SCHEMA:
            raise ValueError("fresh provider target authorization schema changed")
        if mapping["schema_version"] != FRESH_PROVIDER_READINESS_VERSION:
            raise ValueError("fresh provider target authorization version changed")
        if mapping["claim_boundary"] != FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY:
            raise ValueError("fresh provider target authorization claim boundary changed")
        result = cls(
            decision=FreshProviderReadinessDecisionV1.from_dict(mapping["decision"]),
            target_group_ids=_strings_from_json(
                mapping["target_group_ids"],
                name="target_group_ids",
                allow_empty=False,
            ),
            target_evaluation_budget=mapping["target_evaluation_budget"],
            target_payloads_opened_at_authorization=mapping[
                "target_payloads_opened_at_authorization"
            ],
            target_outcomes_opened_at_authorization=mapping[
                "target_outcomes_opened_at_authorization"
            ],
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("fresh provider target authorization derived fields changed")
        return result


def evaluate_fresh_provider_readiness(
    request: FreshProviderReadinessRequestV1,
) -> FreshProviderReadinessDecisionV1:
    """Replay the strict information order and return exactly one terminal decision."""

    if not isinstance(request, FreshProviderReadinessRequestV1):
        raise TypeError("request must be FreshProviderReadinessRequestV1")
    classification: ReadinessClassification | None = None
    terminal_gate: GateName | None = None
    reasons: tuple[str, ...] = ()
    for index, gate in enumerate(request.gates):
        if gate.status == "not-evaluated":
            raise ValueError(
                f"gate {gate.gate_name!r} is unevaluated after every earlier gate passed"
            )
        if gate.status == "technical-failure":
            classification = "technical-failure"
            terminal_gate = gate.gate_name
            reasons = tuple(
                sorted(f"{gate.gate_name}:{reason}" for reason in gate.reason_codes)
            )
        elif gate.status == "fail":
            classification = _GATE_FAILURE_CLASSIFICATION[gate.gate_name]
            terminal_gate = gate.gate_name
            reasons = tuple(
                sorted(f"{gate.gate_name}:{reason}" for reason in gate.reason_codes)
            )
        if classification is not None:
            for later in request.gates[index + 1 :]:
                if later.status != "not-evaluated":
                    raise ValueError(
                        "gates after the first terminal result must remain not-evaluated"
                    )
            break
    if classification is None:
        classification = "ready-for-one-target-evaluation"
        terminal_gate = "exact-fallback"
        reasons = ("all-source-and-calibration-gates-passed",)
    assert terminal_gate is not None
    return FreshProviderReadinessDecisionV1(
        request=request,
        classification=classification,
        terminal_gate=terminal_gate,
        decision_reasons=reasons,
        next_action=_NEXT_ACTION[classification],
        authorize_point_uncertainty_development=(
            classification == "point-covariance-localized"
        ),
        authorize_target_evaluation=(
            classification == "ready-for-one-target-evaluation"
        ),
        target_evaluation_budget=(
            1 if classification == "ready-for-one-target-evaluation" else 0
        ),
    )


def authorize_fresh_provider_target(
    decision: FreshProviderReadinessDecisionV1,
) -> FreshProviderTargetAuthorizationV1:
    """Create the one-shot target authorization for a passing decision."""

    return FreshProviderTargetAuthorizationV1(
        decision=decision,
        target_group_ids=decision.request.cohort_lock.target_group_ids,
    )


def support_gate_from_result(result: object) -> ReadinessGateV1:
    """Adapt a validated ProviderSupportFeasibilityV1-like result."""

    evidence_id = require_sha256(
        getattr(result, "provider_support_feasibility_id", None),
        name="provider_support_feasibility_id",
    )
    support_feasible = _strict_bool(
        getattr(result, "support_feasible", None),
        name="support_feasible",
    )
    if support_feasible:
        return ReadinessGateV1(
            gate_name="support-feasibility",
            status="pass",
            evidence_id=evidence_id,
        )
    reason = require_exact_string(
        getattr(result, "decision_reason", None),
        name="decision_reason",
    )
    return ReadinessGateV1(
        gate_name="support-feasibility",
        status="fail",
        evidence_id=evidence_id,
        reason_codes=(reason,),
    )


def source_competence_gates(
    report: SourceProviderCompetenceReportV1,
) -> tuple[ReadinessGateV1, ReadinessGateV1]:
    """Adapt one source report without evaluating identity after a mean failure."""

    if not isinstance(report, SourceProviderCompetenceReportV1):
        raise TypeError("report must be SourceProviderCompetenceReportV1")
    report_id = report.source_provider_competence_id
    mean_gate = ReadinessGateV1(
        gate_name="source-mean",
        status=report.mean_quality_status,
        evidence_id=report_id,
        reason_codes=(
            () if report.mean_quality_status == "pass" else report.mean_quality_reasons
        ),
    )
    if mean_gate.status != "pass":
        return (
            mean_gate,
            ReadinessGateV1(
                gate_name="identity-reliability",
                status="not-evaluated",
                evidence_id=None,
            ),
        )
    identity_gate = ReadinessGateV1(
        gate_name="identity-reliability",
        status=report.identity_reliability_status,
        evidence_id=report_id,
        reason_codes=(
            ()
            if report.identity_reliability_status == "pass"
            else report.identity_reliability_reasons
        ),
    )
    return mean_gate, identity_gate


def unevaluated_gate(name: GateName) -> ReadinessGateV1:
    return ReadinessGateV1(gate_name=name, status="not-evaluated", evidence_id=None)


def write_fresh_provider_readiness_request(
    path: str | Path,
    request: FreshProviderReadinessRequestV1,
    *,
    overwrite: bool = False,
) -> None:
    _atomic_write_json(path, request.to_dict(), overwrite=overwrite)


def load_fresh_provider_readiness_request(
    path: str | Path,
) -> FreshProviderReadinessRequestV1:
    return FreshProviderReadinessRequestV1.from_dict(
        load_json_object(path, name="fresh provider readiness request")
    )


def write_fresh_provider_readiness_decision(
    path: str | Path,
    decision: FreshProviderReadinessDecisionV1,
    *,
    overwrite: bool = False,
) -> None:
    _atomic_write_json(path, decision.to_dict(), overwrite=overwrite)


def load_fresh_provider_readiness_decision(
    path: str | Path,
) -> FreshProviderReadinessDecisionV1:
    return FreshProviderReadinessDecisionV1.from_dict(
        load_json_object(path, name="fresh provider readiness decision")
    )


def write_fresh_provider_target_authorization(
    path: str | Path,
    authorization: FreshProviderTargetAuthorizationV1,
    *,
    overwrite: bool = False,
) -> None:
    _atomic_write_json(path, authorization.to_dict(), overwrite=overwrite)


def load_fresh_provider_target_authorization(
    path: str | Path,
) -> FreshProviderTargetAuthorizationV1:
    return FreshProviderTargetAuthorizationV1.from_dict(
        load_json_object(path, name="fresh provider target authorization")
    )


def _summary(value: Mapping[str, Any]) -> None:
    print(json.dumps(plain_json(value), sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")

    verify_request = subparsers.add_parser("verify-request")
    verify_request.add_argument("--artifact", type=Path, required=True)

    verify_decision = subparsers.add_parser("verify-decision")
    verify_decision.add_argument("--artifact", type=Path, required=True)

    authorize = subparsers.add_parser("authorize-target")
    authorize.add_argument("--decision", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--overwrite", action="store_true")

    verify_authorization = subparsers.add_parser("verify-authorization")
    verify_authorization.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "evaluate":
        request = load_fresh_provider_readiness_request(arguments.request)
        decision = evaluate_fresh_provider_readiness(request)
        write_fresh_provider_readiness_decision(
            arguments.output,
            decision,
            overwrite=arguments.overwrite,
        )
        _summary(
            {
                "classification": decision.classification,
                "decision_id": decision.fresh_provider_readiness_decision_id,
                "target_evaluation_budget": decision.target_evaluation_budget,
            }
        )
        return 0 if decision.authorize_target_evaluation else 2
    if arguments.command == "verify-request":
        request = load_fresh_provider_readiness_request(arguments.artifact)
        _summary({"request_id": request.fresh_provider_readiness_request_id})
        return 0
    if arguments.command == "verify-decision":
        decision = load_fresh_provider_readiness_decision(arguments.artifact)
        _summary(
            {
                "classification": decision.classification,
                "decision_id": decision.fresh_provider_readiness_decision_id,
            }
        )
        return 0
    if arguments.command == "authorize-target":
        decision = load_fresh_provider_readiness_decision(arguments.decision)
        authorization = authorize_fresh_provider_target(decision)
        write_fresh_provider_target_authorization(
            arguments.output,
            authorization,
            overwrite=arguments.overwrite,
        )
        _summary(
            {
                "authorization_id": (
                    authorization.fresh_provider_target_authorization_id
                ),
                "target_evaluation_budget": authorization.target_evaluation_budget,
            }
        )
        return 0
    authorization = load_fresh_provider_target_authorization(arguments.artifact)
    _summary(
        {
            "authorization_id": authorization.fresh_provider_target_authorization_id,
            "target_evaluation_budget": authorization.target_evaluation_budget,
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FRESH_PROVIDER_READINESS_CLAIM_BOUNDARY",
    "FRESH_PROVIDER_READINESS_VERSION",
    "FreshProviderCohortLockV1",
    "FreshProviderReadinessDecisionV1",
    "FreshProviderReadinessRequestV1",
    "FreshProviderTargetAuthorizationV1",
    "ReadinessGateV1",
    "authorize_fresh_provider_target",
    "evaluate_fresh_provider_readiness",
    "load_fresh_provider_readiness_decision",
    "load_fresh_provider_readiness_request",
    "load_fresh_provider_target_authorization",
    "source_competence_gates",
    "support_gate_from_result",
    "unevaluated_gate",
    "write_fresh_provider_readiness_decision",
    "write_fresh_provider_readiness_request",
    "write_fresh_provider_target_authorization",
]
