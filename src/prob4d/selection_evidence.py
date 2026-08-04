"""Replayable evidence for target-blind calibration and exact fallback decisions.

The artifact retains every object/session-level calibration row, the complete
candidate ordering, the selected candidate, and each deployment guard decision.
Loading an artifact replays selection and verifies that rejected updates reproduce
the declared fallback artifact exactly.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _GIT_SHA,
    _SHA256,
    FINAL_TIE_BREAK,
    SELECTION_EVIDENCE_SCHEMA,
    SELECTION_EVIDENCE_VERSION,
    SELECTION_REPLAY_SCHEMA,
    Aggregation,
    ConstraintRelation,
    Direction,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionRuleV1,
    _exact_keys,
    _sha256_json,
    _strict_digest,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._selection_evidence_records import (
    CalibrationMetricRowV1,
    CandidateSpecV1,
    DeploymentDecisionV1,
)
from ._selection_evidence_replay import (
    SelectionReplayReportV1,
    replay_candidate_order,
)

SELECTION_CLAIM_BOUNDARY = (
    "Selection uses retained calibration groups only. Deployment decisions "
    "record guard outputs and exact fallback identities but are not selection "
    "inputs. Physical benefit requires a separately frozen target analysis."
)


@dataclass(frozen=True, slots=True)
class SelectionEvidenceBundleV2:
    """Complete replayable selection and deployment evidence artifact."""

    experiment_id: str
    source_repository: str
    source_revision: str
    candidates: tuple[CandidateSpecV1, ...]
    calibration_rows: tuple[CalibrationMetricRowV1, ...]
    selection_rule: SelectionRuleV1
    selection_order: tuple[str, ...]
    selected_candidate_id: str
    deployment_decisions: tuple[DeploymentDecisionV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            _strict_string(self.experiment_id, name="experiment_id"),
        )
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
        if type(self.candidates) is not tuple or not self.candidates or not all(
            isinstance(candidate, CandidateSpecV1) for candidate in self.candidates
        ):
            raise ValueError("candidates must be a nonempty tuple of CandidateSpecV1")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(set(candidate_ids)) != len(
            candidate_ids
        ):
            raise ValueError("candidates must be sorted by unique candidate_id")
        if type(self.calibration_rows) is not tuple or not self.calibration_rows or not all(
            isinstance(row, CalibrationMetricRowV1) for row in self.calibration_rows
        ):
            raise ValueError(
                "calibration_rows must be a nonempty tuple of CalibrationMetricRowV1"
            )
        row_keys = tuple(
            (row.group_id, row.candidate_id) for row in self.calibration_rows
        )
        if row_keys != tuple(sorted(row_keys)) or len(set(row_keys)) != len(row_keys):
            raise ValueError("calibration_rows must be sorted by unique group/candidate key")
        if not isinstance(self.selection_rule, SelectionRuleV1):
            raise ValueError("selection_rule must be a SelectionRuleV1")
        replayed_order, _ = replay_candidate_order(
            self.candidates,
            self.calibration_rows,
            self.selection_rule,
        )
        if type(self.selection_order) is not tuple or self.selection_order != replayed_order:
            raise ValueError("selection_order does not match deterministic replay")
        if self.selected_candidate_id != replayed_order[0]:
            raise ValueError("selected_candidate_id does not match deterministic replay")
        if type(self.deployment_decisions) is not tuple or not self.deployment_decisions:
            raise ValueError("deployment_decisions must be a nonempty tuple")
        if not all(
            isinstance(decision, DeploymentDecisionV1)
            for decision in self.deployment_decisions
        ):
            raise ValueError("deployment_decisions must contain DeploymentDecisionV1")
        decision_groups = tuple(
            decision.group_id for decision in self.deployment_decisions
        )
        if decision_groups != tuple(sorted(decision_groups)) or len(
            set(decision_groups)
        ) != len(decision_groups):
            raise ValueError("deployment_decisions must be sorted by unique group_id")
        if any(
            decision.candidate_id != self.selected_candidate_id
            for decision in self.deployment_decisions
        ):
            raise ValueError("every deployment decision must use the selected candidate")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": SELECTION_EVIDENCE_SCHEMA,
            "schema_version": SELECTION_EVIDENCE_VERSION,
            "experiment_id": self.experiment_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "calibration_rows": [row.to_dict() for row in self.calibration_rows],
            "selection_rule": self.selection_rule.to_dict(),
            "selection_order": list(self.selection_order),
            "selected_candidate_id": self.selected_candidate_id,
            "deployment_decisions": [
                decision.to_dict() for decision in self.deployment_decisions
            ],
            "metadata": plain_json(self.metadata),
            "claim_boundary": SELECTION_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.descriptor())

    def replay_report(self) -> SelectionReplayReportV1:
        order, summaries = replay_candidate_order(
            self.candidates,
            self.calibration_rows,
            self.selection_rule,
        )
        accepted = sum(decision.accepted for decision in self.deployment_decisions)
        fallback = len(self.deployment_decisions) - accepted
        exact_fallback = sum(
            decision.exact_fallback_reproduced and not decision.accepted
            for decision in self.deployment_decisions
        )
        return SelectionReplayReportV1(
            evidence_artifact_id=self.artifact_id,
            candidate_order=order,
            selected_candidate_id=order[0],
            candidate_summaries=summaries,
            deployment_group_count=len(self.deployment_decisions),
            accepted_update_count=accepted,
            fallback_update_count=fallback,
            exact_fallback_count=exact_fallback,
        )

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["artifact_id"] = self.artifact_id
        result["replay_digest"] = self.replay_report().replay_digest
        return result


def build_selection_evidence_bundle(
    *,
    experiment_id: str,
    source_repository: str,
    source_revision: str,
    candidates: Sequence[CandidateSpecV1],
    calibration_rows: Sequence[CalibrationMetricRowV1],
    selection_rule: SelectionRuleV1,
    deployment_decisions: Sequence[DeploymentDecisionV1],
    metadata: Mapping[str, Any] | None = None,
) -> SelectionEvidenceBundleV2:
    """Build a canonical bundle and compute its complete replayed selection order."""

    candidate_tuple = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    row_tuple = tuple(
        sorted(calibration_rows, key=lambda item: (item.group_id, item.candidate_id))
    )
    decision_tuple = tuple(
        sorted(deployment_decisions, key=lambda item: item.group_id)
    )
    order, _ = replay_candidate_order(candidate_tuple, row_tuple, selection_rule)
    return SelectionEvidenceBundleV2(
        experiment_id=experiment_id,
        source_repository=source_repository,
        source_revision=source_revision,
        candidates=candidate_tuple,
        calibration_rows=row_tuple,
        selection_rule=selection_rule,
        selection_order=order,
        selected_candidate_id=order[0],
        deployment_decisions=decision_tuple,
        metadata={} if metadata is None else metadata,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def selection_evidence_from_dict(value: Any) -> SelectionEvidenceBundleV2:
    """Parse and replay a strict portable evidence object."""

    mapping = _strict_mapping(value, name="selection evidence")
    _exact_keys(
        mapping,
        {
            "schema_name",
            "schema_version",
            "experiment_id",
            "source_repository",
            "source_revision",
            "candidates",
            "calibration_rows",
            "selection_rule",
            "selection_order",
            "selected_candidate_id",
            "deployment_decisions",
            "metadata",
            "claim_boundary",
            "artifact_id",
            "replay_digest",
        },
        name="selection evidence",
    )
    if mapping["schema_name"] != SELECTION_EVIDENCE_SCHEMA:
        raise ValueError("unsupported selection evidence schema")
    if (
        type(mapping["schema_version"]) is not int
        or mapping["schema_version"] != SELECTION_EVIDENCE_VERSION
    ):
        raise ValueError("unsupported selection evidence version")
    if mapping["claim_boundary"] != SELECTION_CLAIM_BOUNDARY:
        raise ValueError("selection evidence claim_boundary mismatch")
    candidate_values = _strict_list(mapping["candidates"], name="candidates")
    row_values = _strict_list(mapping["calibration_rows"], name="calibration_rows")
    decision_values = _strict_list(
        mapping["deployment_decisions"],
        name="deployment_decisions",
    )
    order_values = _strict_list(mapping["selection_order"], name="selection_order")
    bundle = SelectionEvidenceBundleV2(
        experiment_id=mapping["experiment_id"],
        source_repository=mapping["source_repository"],
        source_revision=mapping["source_revision"],
        candidates=tuple(CandidateSpecV1.from_dict(item) for item in candidate_values),
        calibration_rows=tuple(
            CalibrationMetricRowV1.from_dict(item) for item in row_values
        ),
        selection_rule=SelectionRuleV1.from_dict(mapping["selection_rule"]),
        selection_order=tuple(
            _strict_string(item, name="selection_order item") for item in order_values
        ),
        selected_candidate_id=mapping["selected_candidate_id"],
        deployment_decisions=tuple(
            DeploymentDecisionV1.from_dict(item) for item in decision_values
        ),
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )
    artifact_id = _strict_digest(mapping["artifact_id"], name="artifact_id", pattern=_SHA256)
    replay_digest = _strict_digest(
        mapping["replay_digest"],
        name="replay_digest",
        pattern=_SHA256,
    )
    if artifact_id != bundle.artifact_id:
        raise ValueError("selection evidence artifact_id mismatch")
    if replay_digest != bundle.replay_report().replay_digest:
        raise ValueError("selection evidence replay_digest mismatch")
    return bundle


def write_selection_evidence(
    bundle: SelectionEvidenceBundleV2,
    path: str | os.PathLike[str],
) -> None:
    """Write a canonical evidence artifact atomically."""

    if not isinstance(bundle, SelectionEvidenceBundleV2):
        raise ValueError("bundle must be a SelectionEvidenceBundleV2")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(bundle.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_selection_evidence(
    path: str | os.PathLike[str],
) -> SelectionEvidenceBundleV2:
    """Load, validate, and replay a portable evidence artifact."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("selection evidence is unreadable or invalid JSON") from error
    return selection_evidence_from_dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify an evidence artifact and print its deterministic replay report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args(argv)
    bundle = load_selection_evidence(arguments.evidence)
    print(
        json.dumps(
            bundle.replay_report().to_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Aggregation",
    "CalibrationMetricRowV1",
    "CandidateSpecV1",
    "ConstraintRelation",
    "DeploymentDecisionV1",
    "Direction",
    "FINAL_TIE_BREAK",
    "MetricConstraintV1",
    "MetricOrderV1",
    "SELECTION_CLAIM_BOUNDARY",
    "SELECTION_EVIDENCE_SCHEMA",
    "SELECTION_EVIDENCE_VERSION",
    "SELECTION_REPLAY_SCHEMA",
    "SelectionEvidenceBundleV2",
    "SelectionReplayReportV1",
    "SelectionRuleV1",
    "build_selection_evidence_bundle",
    "load_selection_evidence",
    "replay_candidate_order",
    "selection_evidence_from_dict",
    "write_selection_evidence",
]
