"""Append-only target deployment ledgers bound to a pre-target selection lock."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _GIT_SHA,
    _SHA256,
    _exact_keys,
    _sha256_json,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._selection_evidence_records import DeploymentDecisionV1
from .selection_lock import SelectionLockV1

DEPLOYMENT_LEDGER_SCHEMA = "prob4d.deployment-ledger"
DEPLOYMENT_LEDGER_VERSION = 1
DEPLOYMENT_LEDGER_CLAIM_BOUNDARY = (
    "This append-only ledger records target deployment decisions under one immutable "
    "pre-target selection lock. It cannot change candidate selection. Accuracy or physical "
    "benefit must be established by a separately frozen target analysis."
)


@dataclass(frozen=True, slots=True)
class DeploymentLedgerV1:
    """One immutable prefix of target deployment decisions."""

    experiment_id: str
    selection_lock_id: str
    selected_candidate_id: str
    source_repository: str
    source_revision: str
    decisions: tuple[DeploymentDecisionV1, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            _strict_string(self.experiment_id, name="experiment_id"),
        )
        object.__setattr__(
            self,
            "selection_lock_id",
            _strict_digest(
                self.selection_lock_id,
                name="selection_lock_id",
                pattern=_SHA256,
            ),
        )
        selected = _strict_string(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        object.__setattr__(self, "selected_candidate_id", selected)
        repository = _strict_string(self.source_repository, name="source_repository")
        if repository.count("/") != 1 or repository.startswith("/") or repository.endswith("/"):
            raise ValueError("source_repository must have owner/name form")
        object.__setattr__(self, "source_repository", repository)
        object.__setattr__(
            self,
            "source_revision",
            _strict_digest(
                self.source_revision,
                name="source_revision",
                pattern=_GIT_SHA,
            ),
        )
        if type(self.decisions) is not tuple or not all(
            isinstance(decision, DeploymentDecisionV1) for decision in self.decisions
        ):
            raise ValueError("decisions must be a tuple of DeploymentDecisionV1")
        group_ids = tuple(decision.group_id for decision in self.decisions)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("deployment decision group IDs must be unique")
        if any(decision.candidate_id != selected for decision in self.decisions):
            raise ValueError("every deployment decision must use the locked candidate")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    def _base_descriptor(self) -> dict[str, object]:
        return {
            "schema_name": DEPLOYMENT_LEDGER_SCHEMA,
            "schema_version": DEPLOYMENT_LEDGER_VERSION,
            "experiment_id": self.experiment_id,
            "selection_lock_id": self.selection_lock_id,
            "selected_candidate_id": self.selected_candidate_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEPLOYMENT_LEDGER_CLAIM_BOUNDARY,
        }

    def _descriptor_for(
        self,
        decisions: tuple[DeploymentDecisionV1, ...],
    ) -> dict[str, object]:
        base = self._base_descriptor()
        previous_id: str | None = None
        for stop in range(len(decisions)):
            prefix = {
                **base,
                "decisions": [decision.to_dict() for decision in decisions[:stop]],
                "previous_ledger_id": previous_id,
            }
            previous_id = _sha256_json(prefix)
        return {
            **base,
            "decisions": [decision.to_dict() for decision in decisions],
            "previous_ledger_id": previous_id,
        }

    def descriptor(self) -> dict[str, object]:
        return self._descriptor_for(self.decisions)

    @property
    def previous_ledger_id(self) -> str | None:
        value = self.descriptor()["previous_ledger_id"]
        return None if value is None else str(value)

    @property
    def deployment_ledger_id(self) -> str:
        return _sha256_json(self.descriptor())

    @property
    def accepted_update_count(self) -> int:
        return sum(decision.accepted for decision in self.decisions)

    @property
    def fallback_update_count(self) -> int:
        return len(self.decisions) - self.accepted_update_count

    @property
    def exact_fallback_count(self) -> int:
        return sum(
            not decision.accepted and decision.exact_fallback_reproduced
            for decision in self.decisions
        )

    def summary(self) -> dict[str, object]:
        return {
            "deployment_ledger_id": self.deployment_ledger_id,
            "previous_ledger_id": self.previous_ledger_id,
            "selection_lock_id": self.selection_lock_id,
            "selected_candidate_id": self.selected_candidate_id,
            "deployment_group_count": len(self.decisions),
            "accepted_update_count": self.accepted_update_count,
            "fallback_update_count": self.fallback_update_count,
            "exact_fallback_count": self.exact_fallback_count,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["deployment_ledger_id"] = self.deployment_ledger_id
        return result


def build_deployment_ledger(
    selection_lock: SelectionLockV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> DeploymentLedgerV1:
    """Create an empty target ledger bound to one pre-target lock."""

    if not isinstance(selection_lock, SelectionLockV1):
        raise ValueError("selection_lock must be a SelectionLockV1")
    return DeploymentLedgerV1(
        experiment_id=selection_lock.experiment_id,
        selection_lock_id=selection_lock.selection_lock_id,
        selected_candidate_id=selection_lock.selected_candidate_id,
        source_repository=selection_lock.source_repository,
        source_revision=selection_lock.source_revision,
        metadata={} if metadata is None else metadata,
    )


def append_deployment_decision(
    ledger: DeploymentLedgerV1,
    decision: DeploymentDecisionV1,
) -> DeploymentLedgerV1:
    """Return a new ledger prefix without mutating any prior artifact."""

    if not isinstance(ledger, DeploymentLedgerV1):
        raise ValueError("ledger must be a DeploymentLedgerV1")
    if not isinstance(decision, DeploymentDecisionV1):
        raise ValueError("decision must be a DeploymentDecisionV1")
    if any(existing.group_id == decision.group_id for existing in ledger.decisions):
        raise ValueError("deployment group was already appended")
    return replace(ledger, decisions=(*ledger.decisions, decision))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def deployment_ledger_from_dict(value: Any) -> DeploymentLedgerV1:
    """Parse and fully validate one portable deployment-ledger prefix."""

    mapping = _strict_mapping(value, name="deployment ledger")
    _exact_keys(
        mapping,
        {
            "schema_name",
            "schema_version",
            "experiment_id",
            "selection_lock_id",
            "selected_candidate_id",
            "source_repository",
            "source_revision",
            "decisions",
            "previous_ledger_id",
            "metadata",
            "claim_boundary",
            "deployment_ledger_id",
        },
        name="deployment ledger",
    )
    if mapping["schema_name"] != DEPLOYMENT_LEDGER_SCHEMA:
        raise ValueError("unsupported deployment ledger schema")
    version = _strict_integer(mapping["schema_version"], name="schema_version", minimum=1)
    if version != DEPLOYMENT_LEDGER_VERSION:
        raise ValueError("unsupported deployment ledger version")
    if mapping["claim_boundary"] != DEPLOYMENT_LEDGER_CLAIM_BOUNDARY:
        raise ValueError("deployment ledger claim_boundary mismatch")
    decision_values = _strict_list(mapping["decisions"], name="decisions")
    ledger = DeploymentLedgerV1(
        experiment_id=mapping["experiment_id"],
        selection_lock_id=mapping["selection_lock_id"],
        selected_candidate_id=mapping["selected_candidate_id"],
        source_repository=mapping["source_repository"],
        source_revision=mapping["source_revision"],
        decisions=tuple(DeploymentDecisionV1.from_dict(item) for item in decision_values),
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )
    serialized_previous = mapping["previous_ledger_id"]
    if serialized_previous is not None:
        serialized_previous = _strict_digest(
            serialized_previous,
            name="previous_ledger_id",
            pattern=_SHA256,
        )
    ledger_id = _strict_digest(
        mapping["deployment_ledger_id"],
        name="deployment_ledger_id",
        pattern=_SHA256,
    )
    if serialized_previous != ledger.previous_ledger_id:
        raise ValueError("previous_ledger_id mismatch")
    if ledger_id != ledger.deployment_ledger_id:
        raise ValueError("deployment_ledger_id mismatch")
    return ledger


def write_deployment_ledger(
    ledger: DeploymentLedgerV1,
    path: str | os.PathLike[str],
) -> None:
    """Write one immutable deployment prefix atomically."""

    if not isinstance(ledger, DeploymentLedgerV1):
        raise ValueError("ledger must be a DeploymentLedgerV1")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(ledger.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_deployment_ledger(path: str | os.PathLike[str]) -> DeploymentLedgerV1:
    """Load and verify one portable ledger prefix and its previous-prefix identity."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("deployment ledger is unreadable or invalid JSON") from error
    return deployment_ledger_from_dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a deployment ledger and print its compact summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    arguments = parser.parse_args(argv)
    ledger = load_deployment_ledger(arguments.ledger)
    print(json.dumps(ledger.summary(), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEPLOYMENT_LEDGER_CLAIM_BOUNDARY",
    "DEPLOYMENT_LEDGER_SCHEMA",
    "DEPLOYMENT_LEDGER_VERSION",
    "DeploymentDecisionV1",
    "DeploymentLedgerV1",
    "append_deployment_decision",
    "build_deployment_ledger",
    "deployment_ledger_from_dict",
    "load_deployment_ledger",
    "write_deployment_ledger",
]
