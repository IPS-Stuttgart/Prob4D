"""Immutable model and selection logic for prospective provider matrices."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_revision,
    require_sha256,
)
from .fresh_provider_readiness import (
    FreshProviderReadinessDecisionV1,
    FreshProviderTargetAuthorizationV1,
    authorize_fresh_provider_target,
)

PROVIDER_READINESS_MATRIX_VERSION: Final = 1
PROVIDER_READINESS_MATRIX_SELECTION_RULE: Final = (
    "first-ready-by-frozen-priority-v1"
)
PROVIDER_READINESS_MATRIX_LOCK_SCHEMA: Final = "prob4d.provider-readiness-matrix-lock"
PROVIDER_READINESS_MATRIX_REQUEST_SCHEMA: Final = (
    "prob4d.provider-readiness-matrix-request"
)
PROVIDER_READINESS_MATRIX_DECISION_SCHEMA: Final = (
    "prob4d.provider-readiness-matrix-decision"
)
PROVIDER_READINESS_MATRIX_AUTHORIZATION_SCHEMA: Final = (
    "prob4d.provider-readiness-matrix-authorization"
)
PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY: Final = (
    "This source-only matrix freezes a finite provider set, adapter qualification, "
    "comparison policy, statistical-unit rosters, and provider priority before "
    "source outcomes. It selects at most one provider for one protected target "
    "evaluation. It does not establish provider competence, calibrated uncertainty, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the art."
)

_PROVIDER_FIELDS: Final = frozenset(
    {
        "provider_id",
        "priority",
        "provider_repository",
        "provider_revision",
        "model_set_id",
        "loader_id",
        "promotion_lock_id",
        "adapter_identity_id",
        "adapter_conformance_id",
        "metadata",
        "provider_identity_id",
    }
)
_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "matrix_id",
        "source_spec_sha256",
        "selection_rule",
        "maximum_target_evaluations",
        "source_repository",
        "source_revision",
        "cohort_binding_id",
        "query_definition_id",
        "fallback_identity_id",
        "development_group_ids",
        "calibration_group_ids",
        "target_group_ids",
        "confirmation_group_ids",
        "comparison_policy",
        "comparison_policy_id",
        "providers",
        "source_payloads_opened",
        "source_outcomes_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "confirmation_payloads_opened",
        "metadata",
        "claim_boundary",
        "provider_readiness_matrix_lock_id",
    }
)
_ENTRY_FIELDS: Final = frozenset(
    {
        "provider_id",
        "priority",
        "decision_file_sha256",
        "decision",
        "metadata",
    }
)
_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "matrix_lock_file_sha256",
        "matrix_lock",
        "source_spec_sha256",
        "entries",
        "metadata",
        "claim_boundary",
        "provider_readiness_matrix_request_id",
    }
)
_DECISION_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "request",
        "provider_results",
        "ready_provider_ids",
        "point_uncertainty_provider_ids",
        "selected_provider_id",
        "unselected_ready_provider_ids",
        "matrix_status",
        "target_evaluation_budget",
        "metadata",
        "claim_boundary",
        "provider_readiness_matrix_decision_id",
    }
)
_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "matrix_decision",
        "selected_provider_id",
        "target_authorization",
        "target_evaluation_budget",
        "metadata",
        "claim_boundary",
        "provider_readiness_matrix_authorization_id",
    }
)


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _repository(value: object, *, name: str) -> str:
    result = require_exact_string(value, name=name)
    if result.count("/") != 1 or result.startswith("/") or result.endswith("/"):
        raise ValueError(f"{name} must use canonical owner/name form")
    return result


def _canonical_ids(
    value: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a canonical tuple")
    result = tuple(require_exact_string(item, name=name) for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _json_ids(value: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return _canonical_ids(tuple(value), name=name, allow_empty=allow_empty)


@dataclass(frozen=True, slots=True)
class ProviderReadinessMatrixProviderV1:
    """One exact provider route frozen before source outcomes."""

    provider_id: str
    priority: int
    provider_repository: str
    provider_revision: str
    model_set_id: str
    loader_id: str
    promotion_lock_id: str
    adapter_identity_id: str
    adapter_conformance_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            require_exact_string(self.provider_id, name="provider_id"),
        )
        object.__setattr__(
            self,
            "priority",
            require_exact_integer(self.priority, name="priority", minimum=0),
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
            "promotion_lock_id",
            "adapter_identity_id",
            "adapter_conformance_id",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                require_finite_json_mapping(
                    self.metadata,
                    name="matrix provider metadata",
                ),
                name="matrix provider metadata",
            ),
        )
        object.__setattr__(self, "provider_identity_id", _sha256_json(self._content()))

    def _content(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "priority": self.priority,
            "provider_repository": self.provider_repository,
            "provider_revision": self.provider_revision,
            "model_set_id": self.model_set_id,
            "loader_id": self.loader_id,
            "promotion_lock_id": self.promotion_lock_id,
            "adapter_identity_id": self.adapter_identity_id,
            "adapter_conformance_id": self.adapter_conformance_id,
            "metadata": plain_json(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content()
        result["provider_identity_id"] = self.provider_identity_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderReadinessMatrixProviderV1:
        mapping = require_mapping(value, name="matrix provider")
        require_exact_fields(mapping, _PROVIDER_FIELDS, name="matrix provider")
        result = cls(
            provider_id=mapping["provider_id"],
            priority=mapping["priority"],
            provider_repository=mapping["provider_repository"],
            provider_revision=mapping["provider_revision"],
            model_set_id=mapping["model_set_id"],
            loader_id=mapping["loader_id"],
            promotion_lock_id=mapping["promotion_lock_id"],
            adapter_identity_id=mapping["adapter_identity_id"],
            adapter_conformance_id=mapping["adapter_conformance_id"],
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="matrix provider metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("matrix provider derived fields changed")
        return result


@dataclass(frozen=True, slots=True)
class ProviderReadinessMatrixLockV1:
    """Source-outcome-blind lock for a finite comparative provider program."""

    matrix_id: str
    source_spec_sha256: str
    source_repository: str
    source_revision: str
    cohort_binding_id: str
    query_definition_id: str
    fallback_identity_id: str
    development_group_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    confirmation_group_ids: tuple[str, ...]
    comparison_policy: Mapping[str, Any]
    providers: tuple[ProviderReadinessMatrixProviderV1, ...]
    selection_rule: str = PROVIDER_READINESS_MATRIX_SELECTION_RULE
    maximum_target_evaluations: int = 1
    source_payloads_opened: bool = False
    source_outcomes_opened: bool = False
    target_payloads_opened: bool = False
    target_outcomes_opened: bool = False
    confirmation_payloads_opened: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    comparison_policy_id: str = field(init=False)
    provider_readiness_matrix_lock_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix_id",
            require_exact_string(self.matrix_id, name="matrix_id"),
        )
        object.__setattr__(
            self,
            "source_spec_sha256",
            require_sha256(self.source_spec_sha256, name="source_spec_sha256"),
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
        for name in ("cohort_binding_id", "query_definition_id", "fallback_identity_id"):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        rosters = {
            "development_group_ids": False,
            "calibration_group_ids": False,
            "target_group_ids": False,
            "confirmation_group_ids": True,
        }
        for name, allow_empty in rosters.items():
            object.__setattr__(
                self,
                name,
                _canonical_ids(getattr(self, name), name=name, allow_empty=allow_empty),
            )
        roster_sets = [set(getattr(self, name)) for name in rosters]
        for index, first in enumerate(roster_sets):
            for second in roster_sets[index + 1 :]:
                if first.intersection(second):
                    raise ValueError("matrix group rosters must be pairwise disjoint")
        if self.selection_rule != PROVIDER_READINESS_MATRIX_SELECTION_RULE:
            raise ValueError("unsupported provider-readiness matrix selection rule")
        object.__setattr__(
            self,
            "maximum_target_evaluations",
            require_exact_integer(
                self.maximum_target_evaluations,
                name="maximum_target_evaluations",
                minimum=1,
                maximum=1,
            ),
        )
        policy = frozen_finite_json_mapping(
            require_finite_json_mapping(
                self.comparison_policy,
                name="matrix comparison policy",
            ),
            name="matrix comparison policy",
        )
        if not policy:
            raise ValueError("matrix comparison policy must not be empty")
        object.__setattr__(self, "comparison_policy", policy)
        object.__setattr__(self, "comparison_policy_id", _sha256_json(policy))
        if type(self.providers) is not tuple or not self.providers:
            raise TypeError("providers must be a nonempty canonical tuple")
        if any(not isinstance(item, ProviderReadinessMatrixProviderV1) for item in self.providers):
            raise TypeError("providers must contain ProviderReadinessMatrixProviderV1")
        providers = tuple(sorted(self.providers, key=lambda item: (item.priority, item.provider_id)))
        if len({item.provider_id for item in providers}) != len(providers):
            raise ValueError("matrix provider IDs must be unique")
        if len({item.priority for item in providers}) != len(providers):
            raise ValueError("matrix provider priorities must be unique")
        object.__setattr__(self, "providers", providers)
        for name in (
            "source_payloads_opened",
            "source_outcomes_opened",
            "target_payloads_opened",
            "target_outcomes_opened",
            "confirmation_payloads_opened",
        ):
            opened = _strict_bool(getattr(self, name), name=name)
            if opened:
                raise ValueError("provider-readiness matrix lock must precede source and target access")
            object.__setattr__(self, name, opened)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                require_finite_json_mapping(self.metadata, name="matrix lock metadata"),
                name="matrix lock metadata",
            ),
        )
        object.__setattr__(
            self,
            "provider_readiness_matrix_lock_id",
            _sha256_json(self._content()),
        )

    def provider(self, provider_id: str) -> ProviderReadinessMatrixProviderV1:
        for provider in self.providers:
            if provider.provider_id == provider_id:
                return provider
        raise ValueError(f"provider {provider_id!r} is not present in the matrix lock")

    def _content(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_READINESS_MATRIX_LOCK_SCHEMA,
            "schema_version": PROVIDER_READINESS_MATRIX_VERSION,
            "matrix_id": self.matrix_id,
            "source_spec_sha256": self.source_spec_sha256,
            "selection_rule": self.selection_rule,
            "maximum_target_evaluations": self.maximum_target_evaluations,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "cohort_binding_id": self.cohort_binding_id,
            "query_definition_id": self.query_definition_id,
            "fallback_identity_id": self.fallback_identity_id,
            "development_group_ids": list(self.development_group_ids),
            "calibration_group_ids": list(self.calibration_group_ids),
            "target_group_ids": list(self.target_group_ids),
            "confirmation_group_ids": list(self.confirmation_group_ids),
            "comparison_policy": plain_json(self.comparison_policy),
            "comparison_policy_id": self.comparison_policy_id,
            "providers": [item.to_dict() for item in self.providers],
            "source_payloads_opened": self.source_payloads_opened,
            "source_outcomes_opened": self.source_outcomes_opened,
            "target_payloads_opened": self.target_payloads_opened,
            "target_outcomes_opened": self.target_outcomes_opened,
            "confirmation_payloads_opened": self.confirmation_payloads_opened,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content()
        result["provider_readiness_matrix_lock_id"] = (
            self.provider_readiness_matrix_lock_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderReadinessMatrixLockV1:
        mapping = require_mapping(value, name="provider-readiness matrix lock")
        require_exact_fields(mapping, _LOCK_FIELDS, name="provider-readiness matrix lock")
        if mapping["schema"] != PROVIDER_READINESS_MATRIX_LOCK_SCHEMA:
            raise ValueError("provider-readiness matrix lock schema changed")
        if mapping["schema_version"] != PROVIDER_READINESS_MATRIX_VERSION:
            raise ValueError("provider-readiness matrix lock version changed")
        if mapping["claim_boundary"] != PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY:
            raise ValueError("provider-readiness matrix lock claim boundary changed")
        raw_providers = mapping["providers"]
        if not isinstance(raw_providers, list):
            raise ValueError("matrix providers must be a JSON array")
        result = cls(
            matrix_id=mapping["matrix_id"],
            source_spec_sha256=mapping["source_spec_sha256"],
            selection_rule=mapping["selection_rule"],
            maximum_target_evaluations=mapping["maximum_target_evaluations"],
            source_repository=mapping["source_repository"],
            source_revision=mapping["source_revision"],
            cohort_binding_id=mapping["cohort_binding_id"],
            query_definition_id=mapping["query_definition_id"],
            fallback_identity_id=mapping["fallback_identity_id"],
            development_group_ids=_json_ids(
                mapping["development_group_ids"],
                name="development_group_ids",
                allow_empty=False,
            ),
            calibration_group_ids=_json_ids(
                mapping["calibration_group_ids"],
                name="calibration_group_ids",
                allow_empty=False,
            ),
            target_group_ids=_json_ids(
                mapping["target_group_ids"],
                name="target_group_ids",
                allow_empty=False,
            ),
            confirmation_group_ids=_json_ids(
                mapping["confirmation_group_ids"],
                name="confirmation_group_ids",
                allow_empty=True,
            ),
            comparison_policy=require_finite_json_mapping(
                mapping["comparison_policy"],
                name="matrix comparison policy",
            ),
            providers=tuple(
                ProviderReadinessMatrixProviderV1.from_dict(item)
                for item in raw_providers
            ),
            source_payloads_opened=mapping["source_payloads_opened"],
            source_outcomes_opened=mapping["source_outcomes_opened"],
            target_payloads_opened=mapping["target_payloads_opened"],
            target_outcomes_opened=mapping["target_outcomes_opened"],
            confirmation_payloads_opened=mapping["confirmation_payloads_opened"],
            metadata=require_finite_json_mapping(
                mapping["metadata"],
                name="matrix lock metadata",
            ),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-readiness matrix lock replay changed")
        return result


def readiness_matrix_provider_metadata(
    lock: ProviderReadinessMatrixLockV1,
    provider_id: str,
) -> dict[str, object]:
    provider = lock.provider(provider_id)
    return {
        "provider_readiness_matrix_lock_id": lock.provider_readiness_matrix_lock_id,
        "provider_readiness_matrix_policy_id": lock.comparison_policy_id,
        "provider_readiness_matrix_provider_id": provider.provider_id,
        "provider_adapter_identity_id": provider.adapter_identity_id,
        "provider_adapter_conformance_id": provider.adapter_conformance_id,
    }


def _binding_errors(
    lock: ProviderReadinessMatrixLockV1,
    provider: ProviderReadinessMatrixProviderV1,
    decision: FreshProviderReadinessDecisionV1,
) -> tuple[str, ...]:
    cohort = decision.cohort_lock
    expected = {
        "source_repository": lock.source_repository,
        "source_revision": lock.source_revision,
        "provider_repository": provider.provider_repository,
        "provider_revision": provider.provider_revision,
        "model_set_id": provider.model_set_id,
        "loader_id": provider.loader_id,
        "cohort_binding_id": lock.cohort_binding_id,
        "promotion_lock_id": provider.promotion_lock_id,
        "query_definition_id": lock.query_definition_id,
        "fallback_identity_id": lock.fallback_identity_id,
        "development_group_ids": lock.development_group_ids,
        "calibration_group_ids": lock.calibration_group_ids,
        "target_group_ids": lock.target_group_ids,
        "confirmation_group_ids": lock.confirmation_group_ids,
    }
    errors = [name for name, value in expected.items() if getattr(cohort, name) != value]
    for name, value in readiness_matrix_provider_metadata(lock, provider.provider_id).items():
        if decision.metadata.get(name) != value:
            errors.append(name)
    return tuple(sorted(errors))


@dataclass(frozen=True, slots=True)
class ProviderReadinessMatrixEntryV1:
    provider_id: str
    priority: int
    decision_file_sha256: str
    decision: FreshProviderReadinessDecisionV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", require_exact_string(self.provider_id, name="provider_id"))
        object.__setattr__(self, "priority", require_exact_integer(self.priority, name="priority", minimum=0))
        object.__setattr__(self, "decision_file_sha256", require_sha256(self.decision_file_sha256, name="decision_file_sha256"))
        if not isinstance(self.decision, FreshProviderReadinessDecisionV1):
            raise TypeError("decision must be FreshProviderReadinessDecisionV1")
        object.__setattr__(self, "metadata", frozen_finite_json_mapping(require_finite_json_mapping(self.metadata, name="matrix entry metadata"), name="matrix entry metadata"))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "priority": self.priority,
            "decision_file_sha256": self.decision_file_sha256,
            "decision": self.decision.to_dict(),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderReadinessMatrixEntryV1:
        mapping = require_mapping(value, name="matrix entry")
        require_exact_fields(mapping, _ENTRY_FIELDS, name="matrix entry")
        return cls(
            provider_id=mapping["provider_id"],
            priority=mapping["priority"],
            decision_file_sha256=mapping["decision_file_sha256"],
            decision=FreshProviderReadinessDecisionV1.from_dict(mapping["decision"]),
            metadata=require_finite_json_mapping(mapping["metadata"], name="matrix entry metadata"),
        )


@dataclass(frozen=True, slots=True)
class ProviderReadinessMatrixRequestV1:
    matrix_lock_file_sha256: str
    matrix_lock: ProviderReadinessMatrixLockV1
    source_spec_sha256: str
    entries: tuple[ProviderReadinessMatrixEntryV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_readiness_matrix_request_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix_lock_file_sha256", require_sha256(self.matrix_lock_file_sha256, name="matrix_lock_file_sha256"))
        object.__setattr__(self, "source_spec_sha256", require_sha256(self.source_spec_sha256, name="source_spec_sha256"))
        if not isinstance(self.matrix_lock, ProviderReadinessMatrixLockV1):
            raise TypeError("matrix_lock must be ProviderReadinessMatrixLockV1")
        if type(self.entries) is not tuple:
            raise TypeError("entries must be a canonical tuple")
        if any(not isinstance(item, ProviderReadinessMatrixEntryV1) for item in self.entries):
            raise TypeError("entries must contain ProviderReadinessMatrixEntryV1")
        entries = tuple(sorted(self.entries, key=lambda item: (item.priority, item.provider_id)))
        if tuple(item.provider_id for item in entries) != tuple(item.provider_id for item in self.matrix_lock.providers):
            raise ValueError("matrix entry roster differs from the matrix lock")
        for entry, provider in zip(entries, self.matrix_lock.providers, strict=True):
            if entry.priority != provider.priority:
                raise ValueError("matrix entry priority differs from the matrix lock")
            errors = _binding_errors(self.matrix_lock, provider, entry.decision)
            if errors:
                raise ValueError(
                    "provider readiness decision differs from the matrix lock: "
                    + ", ".join(errors)
                )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "metadata", frozen_finite_json_mapping(require_finite_json_mapping(self.metadata, name="matrix request metadata"), name="matrix request metadata"))
        object.__setattr__(self, "provider_readiness_matrix_request_id", _sha256_json(self._content()))

    def _content(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_READINESS_MATRIX_REQUEST_SCHEMA,
            "schema_version": PROVIDER_READINESS_MATRIX_VERSION,
            "matrix_lock_file_sha256": self.matrix_lock_file_sha256,
            "matrix_lock": self.matrix_lock.to_dict(),
            "source_spec_sha256": self.source_spec_sha256,
            "entries": [item.to_dict() for item in self.entries],
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content()
        result["provider_readiness_matrix_request_id"] = self.provider_readiness_matrix_request_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderReadinessMatrixRequestV1:
        mapping = require_mapping(value, name="matrix request")
        require_exact_fields(mapping, _REQUEST_FIELDS, name="matrix request")
        if mapping["schema"] != PROVIDER_READINESS_MATRIX_REQUEST_SCHEMA or mapping["schema_version"] != PROVIDER_READINESS_MATRIX_VERSION:
            raise ValueError("provider-readiness matrix request schema changed")
        if mapping["claim_boundary"] != PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY:
            raise ValueError("provider-readiness matrix request claim boundary changed")
        raw_entries = mapping["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("matrix entries must be a JSON array")
        result = cls(
            matrix_lock_file_sha256=mapping["matrix_lock_file_sha256"],
            matrix_lock=ProviderReadinessMatrixLockV1.from_dict(mapping["matrix_lock"]),
            source_spec_sha256=mapping["source_spec_sha256"],
            entries=tuple(ProviderReadinessMatrixEntryV1.from_dict(item) for item in raw_entries),
            metadata=require_finite_json_mapping(mapping["metadata"], name="matrix request metadata"),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-readiness matrix request replay changed")
        return result


@dataclass(frozen=True, slots=True)
class ProviderReadinessMatrixDecisionV1:
    request: ProviderReadinessMatrixRequestV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_results: tuple[Mapping[str, Any], ...] = field(init=False)
    ready_provider_ids: tuple[str, ...] = field(init=False)
    point_uncertainty_provider_ids: tuple[str, ...] = field(init=False)
    selected_provider_id: str | None = field(init=False)
    unselected_ready_provider_ids: tuple[str, ...] = field(init=False)
    matrix_status: str = field(init=False)
    target_evaluation_budget: int = field(init=False)
    provider_readiness_matrix_decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProviderReadinessMatrixRequestV1):
            raise TypeError("request must be ProviderReadinessMatrixRequestV1")
        ready = tuple(item.provider_id for item in self.request.entries if item.decision.authorize_target_evaluation)
        point = tuple(item.provider_id for item in self.request.entries if item.decision.authorize_point_uncertainty_development)
        selected = ready[0] if ready else None
        results = tuple(
            frozen_finite_json_mapping(
                {
                    "provider_id": item.provider_id,
                    "priority": item.priority,
                    "classification": item.decision.classification,
                    "decision_id": item.decision.fresh_provider_readiness_decision_id,
                    "source_ready": item.decision.authorize_target_evaluation,
                    "point_uncertainty_authorized": item.decision.authorize_point_uncertainty_development,
                    "selected_for_target_evaluation": item.provider_id == selected,
                },
                name="matrix provider result",
            )
            for item in self.request.entries
        )
        object.__setattr__(self, "provider_results", results)
        object.__setattr__(self, "ready_provider_ids", ready)
        object.__setattr__(self, "point_uncertainty_provider_ids", point)
        object.__setattr__(self, "selected_provider_id", selected)
        object.__setattr__(self, "unselected_ready_provider_ids", ready[1:] if selected else ())
        object.__setattr__(self, "matrix_status", "provider-selected" if selected else "no-provider-ready")
        object.__setattr__(self, "target_evaluation_budget", 1 if selected else 0)
        object.__setattr__(self, "metadata", frozen_finite_json_mapping(require_finite_json_mapping(self.metadata, name="matrix decision metadata"), name="matrix decision metadata"))
        object.__setattr__(self, "provider_readiness_matrix_decision_id", _sha256_json(self._content()))

    def _content(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_READINESS_MATRIX_DECISION_SCHEMA,
            "schema_version": PROVIDER_READINESS_MATRIX_VERSION,
            "request": self.request.to_dict(),
            "provider_results": [plain_json(item) for item in self.provider_results],
            "ready_provider_ids": list(self.ready_provider_ids),
            "point_uncertainty_provider_ids": list(self.point_uncertainty_provider_ids),
            "selected_provider_id": self.selected_provider_id,
            "unselected_ready_provider_ids": list(self.unselected_ready_provider_ids),
            "matrix_status": self.matrix_status,
            "target_evaluation_budget": self.target_evaluation_budget,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content()
        result["provider_readiness_matrix_decision_id"] = self.provider_readiness_matrix_decision_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderReadinessMatrixDecisionV1:
        mapping = require_mapping(value, name="matrix decision")
        require_exact_fields(mapping, _DECISION_FIELDS, name="matrix decision")
        if mapping["schema"] != PROVIDER_READINESS_MATRIX_DECISION_SCHEMA or mapping["schema_version"] != PROVIDER_READINESS_MATRIX_VERSION:
            raise ValueError("provider-readiness matrix decision schema changed")
        if mapping["claim_boundary"] != PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY:
            raise ValueError("provider-readiness matrix decision claim boundary changed")
        result = cls(
            request=ProviderReadinessMatrixRequestV1.from_dict(mapping["request"]),
            metadata=require_finite_json_mapping(mapping["metadata"], name="matrix decision metadata"),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-readiness matrix decision replay changed")
        return result


@dataclass(frozen=True, slots=True)
class ProviderReadinessMatrixAuthorizationV1:
    matrix_decision: ProviderReadinessMatrixDecisionV1
    selected_provider_id: str
    target_authorization: FreshProviderTargetAuthorizationV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    target_evaluation_budget: int = field(init=False, default=1)
    provider_readiness_matrix_authorization_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.matrix_decision, ProviderReadinessMatrixDecisionV1):
            raise TypeError("matrix_decision must be ProviderReadinessMatrixDecisionV1")
        selected = require_exact_string(self.selected_provider_id, name="selected_provider_id")
        if selected != self.matrix_decision.selected_provider_id:
            raise ValueError("selected provider differs from the matrix decision")
        if not isinstance(self.target_authorization, FreshProviderTargetAuthorizationV1):
            raise TypeError("target_authorization must be FreshProviderTargetAuthorizationV1")
        selected_entry = next(item for item in self.matrix_decision.request.entries if item.provider_id == selected)
        if self.target_authorization.readiness_decision_id != selected_entry.decision.fresh_provider_readiness_decision_id:
            raise ValueError("target authorization differs from the selected readiness decision")
        object.__setattr__(self, "selected_provider_id", selected)
        object.__setattr__(self, "metadata", frozen_finite_json_mapping(require_finite_json_mapping(self.metadata, name="matrix authorization metadata"), name="matrix authorization metadata"))
        object.__setattr__(self, "provider_readiness_matrix_authorization_id", _sha256_json(self._content()))

    def _content(self) -> dict[str, object]:
        return {
            "schema": PROVIDER_READINESS_MATRIX_AUTHORIZATION_SCHEMA,
            "schema_version": PROVIDER_READINESS_MATRIX_VERSION,
            "matrix_decision": self.matrix_decision.to_dict(),
            "selected_provider_id": self.selected_provider_id,
            "target_authorization": self.target_authorization.to_dict(),
            "target_evaluation_budget": self.target_evaluation_budget,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content()
        result["provider_readiness_matrix_authorization_id"] = self.provider_readiness_matrix_authorization_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> ProviderReadinessMatrixAuthorizationV1:
        mapping = require_mapping(value, name="matrix authorization")
        require_exact_fields(mapping, _AUTHORIZATION_FIELDS, name="matrix authorization")
        if mapping["schema"] != PROVIDER_READINESS_MATRIX_AUTHORIZATION_SCHEMA or mapping["schema_version"] != PROVIDER_READINESS_MATRIX_VERSION:
            raise ValueError("provider-readiness matrix authorization schema changed")
        if mapping["claim_boundary"] != PROVIDER_READINESS_MATRIX_CLAIM_BOUNDARY:
            raise ValueError("provider-readiness matrix authorization claim boundary changed")
        result = cls(
            matrix_decision=ProviderReadinessMatrixDecisionV1.from_dict(mapping["matrix_decision"]),
            selected_provider_id=mapping["selected_provider_id"],
            target_authorization=FreshProviderTargetAuthorizationV1.from_dict(mapping["target_authorization"]),
            metadata=require_finite_json_mapping(mapping["metadata"], name="matrix authorization metadata"),
        )
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("provider-readiness matrix authorization replay changed")
        return result


def evaluate_provider_readiness_matrix(
    request: ProviderReadinessMatrixRequestV1,
) -> ProviderReadinessMatrixDecisionV1:
    return ProviderReadinessMatrixDecisionV1(request=request)


def authorize_provider_readiness_matrix_target(
    decision: ProviderReadinessMatrixDecisionV1,
) -> ProviderReadinessMatrixAuthorizationV1:
    if decision.selected_provider_id is None:
        raise ValueError("provider-readiness matrix selected no target provider")
    entry = next(item for item in decision.request.entries if item.provider_id == decision.selected_provider_id)
    return ProviderReadinessMatrixAuthorizationV1(
        matrix_decision=decision,
        selected_provider_id=entry.provider_id,
        target_authorization=authorize_fresh_provider_target(entry.decision),
    )
