"""Pre-target selection locks for replayable source-calibration decisions.

A selection lock contains only candidate definitions, the complete calibration
matrix, and the frozen deterministic selection rule. Its content identity can be
committed before any target deployment decision or target outcome is opened.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import (
    _GIT_SHA,
    _SHA256,
    MetricConstraintV1,
    MetricOrderV1,
    SelectionRuleV1,
    _exact_keys,
    _sha256_json,
    _strict_digest,
    _strict_integer,
    _strict_list,
    _strict_mapping,
    _strict_string,
)
from ._selection_evidence_records import CalibrationMetricRowV1, CandidateSpecV1
from ._selection_evidence_replay import (
    SelectionReplayReportV1,
    replay_candidate_order,
)

SELECTION_LOCK_SCHEMA = "prob4d.selection-lock"
SELECTION_LOCK_VERSION = 1
SELECTION_LOCK_CLAIM_BOUNDARY = (
    "This artifact authenticates candidate selection from retained source-calibration "
    "groups only. It contains no target deployment decision or target outcome. Physical "
    "benefit requires a separately content-addressed deployment ledger and frozen target "
    "analysis."
)


def _atomic_create(path: Path, content: bytes) -> None:
    """Durably publish one immutable file without replacing a competing writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SelectionLockV1:
    """Complete source-calibration decision sealed before target access."""

    experiment_id: str
    source_repository: str
    source_revision: str
    candidates: tuple[CandidateSpecV1, ...]
    calibration_rows: tuple[CalibrationMetricRowV1, ...]
    selection_rule: SelectionRuleV1
    selection_order: tuple[str, ...]
    selected_candidate_id: str
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
        if (
            type(self.candidates) is not tuple
            or not self.candidates
            or not all(isinstance(candidate, CandidateSpecV1) for candidate in self.candidates)
        ):
            raise ValueError("candidates must be a nonempty tuple of CandidateSpecV1")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(set(candidate_ids)) != len(
            candidate_ids
        ):
            raise ValueError("candidates must be sorted by unique candidate_id")
        if (
            type(self.calibration_rows) is not tuple
            or not self.calibration_rows
            or not all(isinstance(row, CalibrationMetricRowV1) for row in self.calibration_rows)
        ):
            raise ValueError("calibration_rows must be a nonempty tuple of CalibrationMetricRowV1")
        row_keys = tuple((row.group_id, row.candidate_id) for row in self.calibration_rows)
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
        selected = _strict_string(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        if selected != replayed_order[0]:
            raise ValueError("selected_candidate_id does not match deterministic replay")
        object.__setattr__(self, "selected_candidate_id", selected)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": SELECTION_LOCK_SCHEMA,
            "schema_version": SELECTION_LOCK_VERSION,
            "experiment_id": self.experiment_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "calibration_rows": [row.to_dict() for row in self.calibration_rows],
            "selection_rule": self.selection_rule.to_dict(),
            "selection_order": list(self.selection_order),
            "selected_candidate_id": self.selected_candidate_id,
            "metadata": plain_json(self.metadata),
            "claim_boundary": SELECTION_LOCK_CLAIM_BOUNDARY,
        }

    @property
    def selection_lock_id(self) -> str:
        return _sha256_json(self.descriptor())

    def replay_report(self) -> SelectionReplayReportV1:
        order, summaries = replay_candidate_order(
            self.candidates,
            self.calibration_rows,
            self.selection_rule,
        )
        return SelectionReplayReportV1(
            evidence_artifact_id=self.selection_lock_id,
            candidate_order=order,
            selected_candidate_id=order[0],
            candidate_summaries=summaries,
            deployment_group_count=0,
            accepted_update_count=0,
            fallback_update_count=0,
            exact_fallback_count=0,
        )

    def to_dict(self) -> dict[str, object]:
        result = self.descriptor()
        result["selection_lock_id"] = self.selection_lock_id
        result["replay_digest"] = self.replay_report().replay_digest
        return result


def build_selection_lock(
    *,
    experiment_id: str,
    source_repository: str,
    source_revision: str,
    candidates: Sequence[CandidateSpecV1],
    calibration_rows: Sequence[CalibrationMetricRowV1],
    selection_rule: SelectionRuleV1,
    metadata: Mapping[str, Any] | None = None,
) -> SelectionLockV1:
    """Build one canonical lock without requiring any target-side records."""

    candidate_tuple = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    row_tuple = tuple(sorted(calibration_rows, key=lambda item: (item.group_id, item.candidate_id)))
    order, _ = replay_candidate_order(candidate_tuple, row_tuple, selection_rule)
    return SelectionLockV1(
        experiment_id=experiment_id,
        source_repository=source_repository,
        source_revision=source_revision,
        candidates=candidate_tuple,
        calibration_rows=row_tuple,
        selection_rule=selection_rule,
        selection_order=order,
        selected_candidate_id=order[0],
        metadata={} if metadata is None else metadata,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def selection_lock_from_dict(value: Any) -> SelectionLockV1:
    """Parse, validate, and independently replay one portable lock."""

    mapping = _strict_mapping(value, name="selection lock")
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
            "metadata",
            "claim_boundary",
            "selection_lock_id",
            "replay_digest",
        },
        name="selection lock",
    )
    if mapping["schema_name"] != SELECTION_LOCK_SCHEMA:
        raise ValueError("unsupported selection lock schema")
    version = _strict_integer(mapping["schema_version"], name="schema_version", minimum=1)
    if version != SELECTION_LOCK_VERSION:
        raise ValueError("unsupported selection lock version")
    if mapping["claim_boundary"] != SELECTION_LOCK_CLAIM_BOUNDARY:
        raise ValueError("selection lock claim_boundary mismatch")
    candidate_values = _strict_list(mapping["candidates"], name="candidates")
    row_values = _strict_list(mapping["calibration_rows"], name="calibration_rows")
    order_values = _strict_list(mapping["selection_order"], name="selection_order")
    lock = SelectionLockV1(
        experiment_id=mapping["experiment_id"],
        source_repository=mapping["source_repository"],
        source_revision=mapping["source_revision"],
        candidates=tuple(CandidateSpecV1.from_dict(item) for item in candidate_values),
        calibration_rows=tuple(CalibrationMetricRowV1.from_dict(item) for item in row_values),
        selection_rule=SelectionRuleV1.from_dict(mapping["selection_rule"]),
        selection_order=tuple(
            _strict_string(item, name="selection_order item") for item in order_values
        ),
        selected_candidate_id=mapping["selected_candidate_id"],
        metadata=_strict_mapping(mapping["metadata"], name="metadata"),
    )
    lock_id = _strict_digest(
        mapping["selection_lock_id"],
        name="selection_lock_id",
        pattern=_SHA256,
    )
    replay_digest = _strict_digest(
        mapping["replay_digest"],
        name="replay_digest",
        pattern=_SHA256,
    )
    if lock_id != lock.selection_lock_id:
        raise ValueError("selection_lock_id mismatch")
    if replay_digest != lock.replay_report().replay_digest:
        raise ValueError("selection lock replay_digest mismatch")
    return lock


def write_selection_lock(lock: SelectionLockV1, path: str | os.PathLike[str]) -> None:
    """Publish one canonical lock without replacing retained evidence."""

    if not isinstance(lock, SelectionLockV1):
        raise ValueError("lock must be a SelectionLockV1")
    payload = (
        json.dumps(lock.to_dict(), sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _atomic_create(Path(path), payload)


def load_selection_lock(path: str | os.PathLike[str]) -> SelectionLockV1:
    """Load and fully replay one portable lock."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("selection lock is unreadable or invalid JSON") from error
    return selection_lock_from_dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a selection lock and print its target-free replay report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    arguments = parser.parse_args(argv)
    lock = load_selection_lock(arguments.lock)
    print(
        json.dumps(
            lock.replay_report().to_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationMetricRowV1",
    "CandidateSpecV1",
    "MetricConstraintV1",
    "MetricOrderV1",
    "SELECTION_LOCK_CLAIM_BOUNDARY",
    "SELECTION_LOCK_SCHEMA",
    "SELECTION_LOCK_VERSION",
    "SelectionLockV1",
    "SelectionRuleV1",
    "build_selection_lock",
    "load_selection_lock",
    "selection_lock_from_dict",
    "write_selection_lock",
]
