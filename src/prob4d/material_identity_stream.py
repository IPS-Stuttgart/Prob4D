"""Append-only hypotheses for material identity across causal tracklet windows.

The stream preserves every window-local track identifier. It records sparse
source-only cross-window association candidates and pairwise gate decisions,
but it never rewrites those local identifiers into a claim-bearing global point
ID. Updates can only point from already admitted windows to one new window,
which makes the lineage append-only and acyclic by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ._immutable_json import frozen_finite_json_mapping, plain_json
from .cross_window_tracklets import (
    CrossWindowAssociationCandidate,
    CrossWindowAssociationResult,
)

MATERIAL_IDENTITY_STREAM_SCHEMA = "prob4d.material-identity-hypothesis-stream"
MATERIAL_IDENTITY_STREAM_VERSION = 1


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    digest = _strict_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _git_sha(value: Any, *, name: str) -> str:
    revision = _strict_string(value, name=name)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return revision


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(
    value: Any,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if (strictly_positive and result <= minimum) or (not strictly_positive and result < minimum):
        relation = "greater than" if strictly_positive else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _integer_tuple(
    value: Any,
    *,
    name: str,
    nonempty: bool = False,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple of integers")
    result = tuple(_integer(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(next_value <= item for item, next_value in zip(result, result[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise ValueError(f"{name} has invalid fields: " + "; ".join(details))


def _strict_json_record(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("material-identity stream manifest is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("material-identity stream manifest must be a JSON object")
    return value


@dataclass(frozen=True)
class MaterialIdentityHypothesisV1:
    """One sparse source-only candidate between two window-local track IDs."""

    source_window_id: str
    source_track_id: int
    target_window_id: str
    target_track_id: int
    shared_frame_indices: tuple[int, ...]
    compatibility_score: float
    effective_support: float
    weighted_rms_m: float
    maximum_distance_m: float
    normalized_rms: float
    selected_by_pairwise_gate: bool
    association_result_id: str
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        source_window = _strict_string(
            self.source_window_id,
            name="source_window_id",
        )
        target_window = _strict_string(
            self.target_window_id,
            name="target_window_id",
        )
        if source_window == target_window:
            raise ValueError("material-identity hypotheses require distinct windows")
        object.__setattr__(self, "source_window_id", source_window)
        object.__setattr__(self, "target_window_id", target_window)
        object.__setattr__(
            self,
            "source_track_id",
            _integer(self.source_track_id, name="source_track_id"),
        )
        object.__setattr__(
            self,
            "target_track_id",
            _integer(self.target_track_id, name="target_track_id"),
        )
        object.__setattr__(
            self,
            "shared_frame_indices",
            _integer_tuple(
                self.shared_frame_indices,
                name="shared_frame_indices",
                nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "compatibility_score",
            _real(
                self.compatibility_score,
                name="compatibility_score",
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "effective_support",
            _real(
                self.effective_support,
                name="effective_support",
                strictly_positive=True,
            ),
        )
        for name in ("weighted_rms_m", "maximum_distance_m", "normalized_rms"):
            object.__setattr__(self, name, _real(getattr(self, name), name=name))
        tolerance = 1e-12 * max(1.0, self.maximum_distance_m)
        if self.weighted_rms_m > self.maximum_distance_m + tolerance:
            raise ValueError("weighted_rms_m must not exceed maximum_distance_m")
        if type(self.selected_by_pairwise_gate) is not bool:
            raise ValueError("selected_by_pairwise_gate must be a Boolean")
        object.__setattr__(
            self,
            "association_result_id",
            _sha256(self.association_result_id, name="association_result_id"),
        )
        expected = _sha256_json(self.identity_record())
        if self.hypothesis_id is not None and (
            _sha256(self.hypothesis_id, name="hypothesis_id") != expected
        ):
            raise ValueError("material-identity hypothesis ID mismatch")
        object.__setattr__(self, "hypothesis_id", expected)

    @property
    def source_identity(self) -> tuple[str, int]:
        return self.source_window_id, self.source_track_id

    @property
    def target_identity(self) -> tuple[str, int]:
        return self.target_window_id, self.target_track_id

    def identity_record(self) -> dict[str, object]:
        return {
            "source_window_id": self.source_window_id,
            "source_track_id": self.source_track_id,
            "target_window_id": self.target_window_id,
            "target_track_id": self.target_track_id,
            "shared_frame_indices": list(self.shared_frame_indices),
            "compatibility_score": self.compatibility_score,
            "effective_support": self.effective_support,
            "weighted_rms_m": self.weighted_rms_m,
            "maximum_distance_m": self.maximum_distance_m,
            "normalized_rms": self.normalized_rms,
            "selected_by_pairwise_gate": self.selected_by_pairwise_gate,
            "association_result_id": self.association_result_id,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "hypothesis_id": self.hypothesis_id}


@dataclass(frozen=True)
class MaterialIdentityAssociationSummaryV1:
    """Portable candidate and rejection audit for one source-to-target result."""

    source_window_id: str
    target_window_id: str
    causal_frame_stop: int
    association_result_id: str
    source_track_count: int
    target_track_count: int
    possible_track_pair_count: int
    spatial_candidate_pair_count: int
    spatially_rejected_pair_count: int
    evaluated_track_pair_count: int
    shared_gate_frame_count: int
    insufficient_shared_frame_pair_count: int
    zero_support_pair_count: int
    low_support_pair_count: int
    non_mutual_best_count: int
    ambiguous_mutual_best_count: int
    threshold_rejected_mutual_best_count: int
    unmatched_source_track_ids: tuple[int, ...]
    unmatched_target_track_ids: tuple[int, ...]
    hypotheses: tuple[MaterialIdentityHypothesisV1, ...]
    summary_id: str | None = None

    def __post_init__(self) -> None:
        source_window = _strict_string(
            self.source_window_id,
            name="source_window_id",
        )
        target_window = _strict_string(
            self.target_window_id,
            name="target_window_id",
        )
        if source_window == target_window:
            raise ValueError("association summaries require distinct windows")
        object.__setattr__(self, "source_window_id", source_window)
        object.__setattr__(self, "target_window_id", target_window)
        object.__setattr__(
            self,
            "causal_frame_stop",
            _integer(
                self.causal_frame_stop,
                name="causal_frame_stop",
                minimum=1,
            ),
        )
        result_id = _sha256(self.association_result_id, name="association_result_id")
        object.__setattr__(self, "association_result_id", result_id)
        count_names = (
            "source_track_count",
            "target_track_count",
            "possible_track_pair_count",
            "spatial_candidate_pair_count",
            "spatially_rejected_pair_count",
            "evaluated_track_pair_count",
            "shared_gate_frame_count",
            "insufficient_shared_frame_pair_count",
            "zero_support_pair_count",
            "low_support_pair_count",
            "non_mutual_best_count",
            "ambiguous_mutual_best_count",
            "threshold_rejected_mutual_best_count",
        )
        for name in count_names:
            minimum = 1 if name in {"source_track_count", "target_track_count"} else 0
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=minimum),
            )
        if self.possible_track_pair_count != (self.source_track_count * self.target_track_count):
            raise ValueError("possible_track_pair_count differs from track domains")
        if self.spatial_candidate_pair_count > self.possible_track_pair_count:
            raise ValueError("spatial candidate count exceeds possible pair count")
        if self.spatially_rejected_pair_count != (
            self.possible_track_pair_count - self.spatial_candidate_pair_count
        ):
            raise ValueError("spatial rejection accounting is inconsistent")

        unmatched_source = _integer_tuple(
            self.unmatched_source_track_ids,
            name="unmatched_source_track_ids",
        )
        unmatched_target = _integer_tuple(
            self.unmatched_target_track_ids,
            name="unmatched_target_track_ids",
        )
        if any(value >= self.source_track_count for value in unmatched_source):
            raise ValueError("unmatched source track lies outside its domain")
        if any(value >= self.target_track_count for value in unmatched_target):
            raise ValueError("unmatched target track lies outside its domain")
        object.__setattr__(self, "unmatched_source_track_ids", unmatched_source)
        object.__setattr__(self, "unmatched_target_track_ids", unmatched_target)

        if type(self.hypotheses) is not tuple or not all(
            isinstance(value, MaterialIdentityHypothesisV1) for value in self.hypotheses
        ):
            raise TypeError("hypotheses must be a tuple of material-identity hypotheses")
        hypotheses = tuple(self.hypotheses)
        pairs = tuple(
            (hypothesis.source_track_id, hypothesis.target_track_id) for hypothesis in hypotheses
        )
        if pairs != tuple(sorted(pairs)) or len(set(pairs)) != len(pairs):
            raise ValueError("hypotheses must contain sorted unique track pairs")
        for hypothesis in hypotheses:
            if (
                hypothesis.source_window_id != source_window
                or hypothesis.target_window_id != target_window
                or hypothesis.association_result_id != result_id
            ):
                raise ValueError("hypothesis lineage differs from its association summary")
            if hypothesis.source_track_id >= self.source_track_count:
                raise ValueError("hypothesis source track lies outside its domain")
            if hypothesis.target_track_id >= self.target_track_count:
                raise ValueError("hypothesis target track lies outside its domain")
        if self.evaluated_track_pair_count != len(hypotheses):
            raise ValueError("evaluated_track_pair_count must equal hypotheses")
        if self.spatial_candidate_pair_count != (
            len(hypotheses)
            + self.insufficient_shared_frame_pair_count
            + self.zero_support_pair_count
        ):
            raise ValueError("spatial candidate disposition accounting is inconsistent")
        if self.low_support_pair_count > len(hypotheses):
            raise ValueError("low_support_pair_count exceeds hypotheses")
        if self.spatial_candidate_pair_count and not self.shared_gate_frame_count:
            raise ValueError("spatial candidates require shared gate frames")

        selected = tuple(
            hypothesis for hypothesis in hypotheses if hypothesis.selected_by_pairwise_gate
        )
        selected_source = {hypothesis.source_track_id for hypothesis in selected}
        selected_target = {hypothesis.target_track_id for hypothesis in selected}
        if len(selected_source) != len(selected) or len(selected_target) != len(selected):
            raise ValueError("pairwise-selected hypotheses must be one-to-one")
        expected_unmatched_source = tuple(
            track_id
            for track_id in range(self.source_track_count)
            if track_id not in selected_source
        )
        expected_unmatched_target = tuple(
            track_id
            for track_id in range(self.target_track_count)
            if track_id not in selected_target
        )
        if unmatched_source != expected_unmatched_source:
            raise ValueError("unmatched source tracks differ from pairwise-selected links")
        if unmatched_target != expected_unmatched_target:
            raise ValueError("unmatched target tracks differ from pairwise-selected links")
        left_best_count = len({hypothesis.source_track_id for hypothesis in hypotheses})
        if (
            self.non_mutual_best_count
            + self.ambiguous_mutual_best_count
            + self.threshold_rejected_mutual_best_count
            + len(selected)
            != left_best_count
        ):
            raise ValueError("pairwise gate disposition accounting is inconsistent")

        expected = _sha256_json(self.identity_record())
        if self.summary_id is not None and (
            _sha256(self.summary_id, name="summary_id") != expected
        ):
            raise ValueError("material-identity association summary ID mismatch")
        object.__setattr__(self, "summary_id", expected)

    @property
    def selected_hypotheses(self) -> tuple[MaterialIdentityHypothesisV1, ...]:
        return tuple(
            hypothesis for hypothesis in self.hypotheses if hypothesis.selected_by_pairwise_gate
        )

    def identity_record(self) -> dict[str, object]:
        return {
            "source_window_id": self.source_window_id,
            "target_window_id": self.target_window_id,
            "causal_frame_stop": self.causal_frame_stop,
            "association_result_id": self.association_result_id,
            "source_track_count": self.source_track_count,
            "target_track_count": self.target_track_count,
            "possible_track_pair_count": self.possible_track_pair_count,
            "spatial_candidate_pair_count": self.spatial_candidate_pair_count,
            "spatially_rejected_pair_count": self.spatially_rejected_pair_count,
            "evaluated_track_pair_count": self.evaluated_track_pair_count,
            "shared_gate_frame_count": self.shared_gate_frame_count,
            "insufficient_shared_frame_pair_count": (self.insufficient_shared_frame_pair_count),
            "zero_support_pair_count": self.zero_support_pair_count,
            "low_support_pair_count": self.low_support_pair_count,
            "non_mutual_best_count": self.non_mutual_best_count,
            "ambiguous_mutual_best_count": self.ambiguous_mutual_best_count,
            "threshold_rejected_mutual_best_count": (self.threshold_rejected_mutual_best_count),
            "unmatched_source_track_ids": list(self.unmatched_source_track_ids),
            "unmatched_target_track_ids": list(self.unmatched_target_track_ids),
            "hypotheses": [hypothesis.to_record() for hypothesis in self.hypotheses],
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "summary_id": self.summary_id}


@dataclass(frozen=True)
class MaterialIdentityStreamUpdateV1:
    """One append-only update that admits exactly one new target window."""

    update_index: int
    target_window_id: str
    causal_frame_stop: int
    associations: tuple[MaterialIdentityAssociationSummaryV1, ...]
    previous_update_id: str | None = None
    update_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "update_index",
            _integer(self.update_index, name="update_index"),
        )
        target = _strict_string(self.target_window_id, name="target_window_id")
        object.__setattr__(self, "target_window_id", target)
        object.__setattr__(
            self,
            "causal_frame_stop",
            _integer(
                self.causal_frame_stop,
                name="causal_frame_stop",
                minimum=1,
            ),
        )
        if (
            type(self.associations) is not tuple
            or not self.associations
            or not all(
                isinstance(value, MaterialIdentityAssociationSummaryV1)
                for value in self.associations
            )
        ):
            raise TypeError("associations must be a non-empty tuple of summaries")
        associations = tuple(self.associations)
        source_ids = tuple(summary.source_window_id for summary in associations)
        if source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(source_ids):
            raise ValueError("association source windows must be sorted and unique")
        for summary in associations:
            if summary.target_window_id != target:
                raise ValueError("association target differs from update target")
            if summary.causal_frame_stop != self.causal_frame_stop:
                raise ValueError("association causal stop differs from update causal stop")
        previous = self.previous_update_id
        if previous is not None:
            previous = _sha256(previous, name="previous_update_id")
        object.__setattr__(self, "previous_update_id", previous)
        expected = _sha256_json(self.identity_record())
        if self.update_id is not None and (_sha256(self.update_id, name="update_id") != expected):
            raise ValueError("material-identity stream update ID mismatch")
        object.__setattr__(self, "update_id", expected)

    @property
    def source_window_ids(self) -> tuple[str, ...]:
        return tuple(summary.source_window_id for summary in self.associations)

    @property
    def hypothesis_count(self) -> int:
        return sum(len(summary.hypotheses) for summary in self.associations)

    def identity_record(self) -> dict[str, object]:
        return {
            "update_index": self.update_index,
            "target_window_id": self.target_window_id,
            "causal_frame_stop": self.causal_frame_stop,
            "associations": [summary.to_record() for summary in self.associations],
            "previous_update_id": self.previous_update_id,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "update_id": self.update_id}


@dataclass(frozen=True)
class MaterialIdentityHypothesisStreamV1:
    """Acyclic append-only lineage without global point-ID rewriting."""

    sequence_id: str
    case_id: str
    stream_id: str
    source_repository: str
    source_revision: str
    root_window_id: str
    updates: tuple[MaterialIdentityStreamUpdateV1, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        identifiers = {
            "sequence_id": _strict_string(self.sequence_id, name="sequence_id"),
            "case_id": _strict_string(self.case_id, name="case_id"),
            "stream_id": _strict_string(self.stream_id, name="stream_id"),
            "source_repository": _strict_string(
                self.source_repository,
                name="source_repository",
            ),
            "source_revision": _git_sha(
                self.source_revision,
                name="source_revision",
            ),
            "root_window_id": _strict_string(
                self.root_window_id,
                name="root_window_id",
            ),
        }
        repository = identifiers["source_repository"]
        if repository.count("/") != 1 or repository.startswith("/") or repository.endswith("/"):
            raise ValueError("source_repository must have owner/name form")
        if type(self.updates) is not tuple or not all(
            isinstance(value, MaterialIdentityStreamUpdateV1) for value in self.updates
        ):
            raise TypeError("updates must be a tuple of material-identity updates")
        updates = tuple(self.updates)
        admitted = {identifiers["root_window_id"]}
        previous: MaterialIdentityStreamUpdateV1 | None = None
        for index, update in enumerate(updates):
            if update.update_index != index:
                raise ValueError("material-identity update indices must be contiguous")
            expected_previous = None if previous is None else previous.update_id
            if update.previous_update_id != expected_previous:
                raise ValueError("material-identity update hash chain is broken")
            if update.target_window_id in admitted:
                raise ValueError("a material-identity target window was already admitted")
            if not set(update.source_window_ids).issubset(admitted):
                raise ValueError("material-identity update references a future source window")
            if previous is not None and update.causal_frame_stop < previous.causal_frame_stop:
                raise ValueError("material-identity causal stops must be nondecreasing")
            admitted.add(update.target_window_id)
            previous = update
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="material-identity stream metadata",
        )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "updates", updates)
        object.__setattr__(self, "metadata", metadata)
        expected = _sha256_json(self.identity_record())
        if self.artifact_id is not None and (
            _sha256(self.artifact_id, name="artifact_id") != expected
        ):
            raise ValueError("material-identity stream artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def admitted_window_ids(self) -> tuple[str, ...]:
        return (self.root_window_id, *(update.target_window_id for update in self.updates))

    @property
    def causal_frame_stop(self) -> int | None:
        return None if not self.updates else self.updates[-1].causal_frame_stop

    @property
    def hypothesis_count(self) -> int:
        return sum(update.hypothesis_count for update in self.updates)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": MATERIAL_IDENTITY_STREAM_SCHEMA,
            "schema_version": MATERIAL_IDENTITY_STREAM_VERSION,
            "sequence_id": self.sequence_id,
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "root_window_id": self.root_window_id,
            "metadata": plain_json(self.metadata),
            "updates": [update.to_record() for update in self.updates],
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}


def _candidate_hypothesis(
    candidate: CrossWindowAssociationCandidate,
    *,
    result: CrossWindowAssociationResult,
    selected_pairs: set[tuple[int, int]],
) -> MaterialIdentityHypothesisV1:
    pair = (candidate.left_track_id, candidate.right_track_id)
    return MaterialIdentityHypothesisV1(
        source_window_id=result.left_window_id,
        source_track_id=candidate.left_track_id,
        target_window_id=result.right_window_id,
        target_track_id=candidate.right_track_id,
        shared_frame_indices=candidate.shared_frame_indices,
        compatibility_score=candidate.compatibility_score,
        effective_support=candidate.effective_support,
        weighted_rms_m=candidate.weighted_rms_m,
        maximum_distance_m=candidate.maximum_distance_m,
        normalized_rms=candidate.normalized_rms,
        selected_by_pairwise_gate=pair in selected_pairs,
        association_result_id=result.result_id,
    )


def association_summary_from_result(
    result: CrossWindowAssociationResult,
    *,
    target_window_id: str,
) -> MaterialIdentityAssociationSummaryV1:
    """Convert one directed prior-window to new-window association result."""

    if not isinstance(result, CrossWindowAssociationResult):
        raise TypeError("result must be a CrossWindowAssociationResult")
    target = _strict_string(target_window_id, name="target_window_id")
    if target != result.right_window_id:
        raise ValueError("target_window_id must equal the association result's right window")
    selected_pairs = set(result.accepted_pairs)
    hypotheses = tuple(
        sorted(
            (
                _candidate_hypothesis(
                    candidate,
                    result=result,
                    selected_pairs=selected_pairs,
                )
                for candidate in result.candidates
            ),
            key=lambda value: (value.source_track_id, value.target_track_id),
        )
    )
    source_ids = (
        {hypothesis.source_track_id for hypothesis in hypotheses}
        | {link.left_track_id for link in result.links}
        | set(result.unmatched_left_track_ids)
    )
    target_ids = (
        {hypothesis.target_track_id for hypothesis in hypotheses}
        | {link.right_track_id for link in result.links}
        | set(result.unmatched_right_track_ids)
    )
    source_track_count = max(source_ids, default=-1) + 1
    target_track_count = max(target_ids, default=-1) + 1
    if source_track_count < 1 or target_track_count < 1:
        raise ValueError("association result has an empty track domain")
    return MaterialIdentityAssociationSummaryV1(
        source_window_id=result.left_window_id,
        target_window_id=result.right_window_id,
        causal_frame_stop=result.causal_frame_stop,
        association_result_id=result.result_id,
        source_track_count=source_track_count,
        target_track_count=target_track_count,
        possible_track_pair_count=result.possible_track_pair_count,
        spatial_candidate_pair_count=result.spatial_candidate_pair_count,
        spatially_rejected_pair_count=result.spatially_rejected_pair_count,
        evaluated_track_pair_count=len(result.candidates),
        shared_gate_frame_count=result.shared_gate_frame_count,
        insufficient_shared_frame_pair_count=(result.insufficient_shared_frame_pair_count),
        zero_support_pair_count=result.zero_support_pair_count,
        low_support_pair_count=result.low_support_pair_count,
        non_mutual_best_count=result.non_mutual_best_count,
        ambiguous_mutual_best_count=result.ambiguous_mutual_best_count,
        threshold_rejected_mutual_best_count=(result.threshold_rejected_mutual_best_count),
        unmatched_source_track_ids=tuple(result.unmatched_left_track_ids),
        unmatched_target_track_ids=tuple(result.unmatched_right_track_ids),
        hypotheses=hypotheses,
    )


def create_material_identity_stream(
    *,
    sequence_id: str,
    case_id: str,
    stream_id: str,
    source_repository: str,
    source_revision: str,
    root_window_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> MaterialIdentityHypothesisStreamV1:
    """Create a root-only stream before any cross-window identity is asserted."""

    return MaterialIdentityHypothesisStreamV1(
        sequence_id=sequence_id,
        case_id=case_id,
        stream_id=stream_id,
        source_repository=source_repository,
        source_revision=source_revision,
        root_window_id=root_window_id,
        metadata={} if metadata is None else metadata,
    )


def append_material_identity_update(
    stream: MaterialIdentityHypothesisStreamV1,
    results: Sequence[CrossWindowAssociationResult],
    *,
    target_window_id: str,
) -> MaterialIdentityHypothesisStreamV1:
    """Append associations from admitted sources to one previously unseen window."""

    if not isinstance(stream, MaterialIdentityHypothesisStreamV1):
        raise TypeError("stream must be a MaterialIdentityHypothesisStreamV1")
    target = _strict_string(target_window_id, name="target_window_id")
    if target in stream.admitted_window_ids:
        raise ValueError("target_window_id was already admitted")
    result_list = list(results)
    if not result_list:
        raise ValueError("at least one cross-window association result is required")
    summaries = tuple(
        sorted(
            (
                association_summary_from_result(result, target_window_id=target)
                for result in result_list
            ),
            key=lambda value: value.source_window_id,
        )
    )
    source_windows = tuple(summary.source_window_id for summary in summaries)
    if len(set(source_windows)) != len(source_windows):
        raise ValueError("one update cannot repeat a source window")
    if not set(source_windows).issubset(set(stream.admitted_window_ids)):
        raise ValueError("association result references a non-admitted source window")
    causal_stops = {summary.causal_frame_stop for summary in summaries}
    if len(causal_stops) != 1:
        raise ValueError("one update requires one common causal frame stop")
    update = MaterialIdentityStreamUpdateV1(
        update_index=len(stream.updates),
        target_window_id=target,
        causal_frame_stop=causal_stops.pop(),
        associations=summaries,
        previous_update_id=None if not stream.updates else stream.updates[-1].update_id,
    )
    return replace(
        stream,
        updates=(*stream.updates, update),
        artifact_id=None,
    )


def write_material_identity_stream(
    stream: MaterialIdentityHypothesisStreamV1,
    path: str | Path,
) -> Path:
    """Atomically write a content-addressed stream manifest."""

    if not isinstance(stream, MaterialIdentityHypothesisStreamV1):
        raise TypeError("stream must be a MaterialIdentityHypothesisStreamV1")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(stream.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def _hypothesis_from_record(value: Any) -> MaterialIdentityHypothesisV1:
    if not isinstance(value, dict):
        raise ValueError("material-identity hypothesis must be a JSON object")
    expected = {
        "source_window_id",
        "source_track_id",
        "target_window_id",
        "target_track_id",
        "shared_frame_indices",
        "compatibility_score",
        "effective_support",
        "weighted_rms_m",
        "maximum_distance_m",
        "normalized_rms",
        "selected_by_pairwise_gate",
        "association_result_id",
        "hypothesis_id",
    }
    _exact_keys(value, expected, name="material-identity hypothesis")
    frames = value["shared_frame_indices"]
    if not isinstance(frames, list):
        raise ValueError("shared_frame_indices must be a JSON array")
    return MaterialIdentityHypothesisV1(
        source_window_id=value["source_window_id"],
        source_track_id=value["source_track_id"],
        target_window_id=value["target_window_id"],
        target_track_id=value["target_track_id"],
        shared_frame_indices=tuple(frames),
        compatibility_score=value["compatibility_score"],
        effective_support=value["effective_support"],
        weighted_rms_m=value["weighted_rms_m"],
        maximum_distance_m=value["maximum_distance_m"],
        normalized_rms=value["normalized_rms"],
        selected_by_pairwise_gate=value["selected_by_pairwise_gate"],
        association_result_id=value["association_result_id"],
        hypothesis_id=value["hypothesis_id"],
    )


def _summary_from_record(value: Any) -> MaterialIdentityAssociationSummaryV1:
    if not isinstance(value, dict):
        raise ValueError("material-identity association must be a JSON object")
    expected = {
        "source_window_id",
        "target_window_id",
        "causal_frame_stop",
        "association_result_id",
        "source_track_count",
        "target_track_count",
        "possible_track_pair_count",
        "spatial_candidate_pair_count",
        "spatially_rejected_pair_count",
        "evaluated_track_pair_count",
        "shared_gate_frame_count",
        "insufficient_shared_frame_pair_count",
        "zero_support_pair_count",
        "low_support_pair_count",
        "non_mutual_best_count",
        "ambiguous_mutual_best_count",
        "threshold_rejected_mutual_best_count",
        "unmatched_source_track_ids",
        "unmatched_target_track_ids",
        "hypotheses",
        "summary_id",
    }
    _exact_keys(value, expected, name="material-identity association")
    hypotheses = value["hypotheses"]
    unmatched_source = value["unmatched_source_track_ids"]
    unmatched_target = value["unmatched_target_track_ids"]
    if not isinstance(hypotheses, list):
        raise ValueError("association hypotheses must be a JSON array")
    if not isinstance(unmatched_source, list) or not isinstance(unmatched_target, list):
        raise ValueError("unmatched track IDs must be JSON arrays")
    return MaterialIdentityAssociationSummaryV1(
        source_window_id=value["source_window_id"],
        target_window_id=value["target_window_id"],
        causal_frame_stop=value["causal_frame_stop"],
        association_result_id=value["association_result_id"],
        source_track_count=value["source_track_count"],
        target_track_count=value["target_track_count"],
        possible_track_pair_count=value["possible_track_pair_count"],
        spatial_candidate_pair_count=value["spatial_candidate_pair_count"],
        spatially_rejected_pair_count=value["spatially_rejected_pair_count"],
        evaluated_track_pair_count=value["evaluated_track_pair_count"],
        shared_gate_frame_count=value["shared_gate_frame_count"],
        insufficient_shared_frame_pair_count=(value["insufficient_shared_frame_pair_count"]),
        zero_support_pair_count=value["zero_support_pair_count"],
        low_support_pair_count=value["low_support_pair_count"],
        non_mutual_best_count=value["non_mutual_best_count"],
        ambiguous_mutual_best_count=value["ambiguous_mutual_best_count"],
        threshold_rejected_mutual_best_count=(value["threshold_rejected_mutual_best_count"]),
        unmatched_source_track_ids=tuple(unmatched_source),
        unmatched_target_track_ids=tuple(unmatched_target),
        hypotheses=tuple(_hypothesis_from_record(item) for item in hypotheses),
        summary_id=value["summary_id"],
    )


def _update_from_record(value: Any) -> MaterialIdentityStreamUpdateV1:
    if not isinstance(value, dict):
        raise ValueError("material-identity update must be a JSON object")
    expected = {
        "update_index",
        "target_window_id",
        "causal_frame_stop",
        "associations",
        "previous_update_id",
        "update_id",
    }
    _exact_keys(value, expected, name="material-identity update")
    associations = value["associations"]
    if not isinstance(associations, list):
        raise ValueError("material-identity associations must be a JSON array")
    return MaterialIdentityStreamUpdateV1(
        update_index=value["update_index"],
        target_window_id=value["target_window_id"],
        causal_frame_stop=value["causal_frame_stop"],
        associations=tuple(_summary_from_record(item) for item in associations),
        previous_update_id=value["previous_update_id"],
        update_id=value["update_id"],
    )


def load_material_identity_stream(
    path: str | Path,
) -> MaterialIdentityHypothesisStreamV1:
    """Load and fully revalidate a material-identity hypothesis stream."""

    record = _strict_json_record(Path(path))
    expected = {
        "schema",
        "schema_version",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "root_window_id",
        "metadata",
        "updates",
        "artifact_id",
    }
    _exact_keys(record, expected, name="material-identity stream")
    if record["schema"] != MATERIAL_IDENTITY_STREAM_SCHEMA:
        raise ValueError("manifest is not a material-identity hypothesis stream")
    if record["schema_version"] != MATERIAL_IDENTITY_STREAM_VERSION:
        raise ValueError("unsupported material-identity stream version")
    updates = record["updates"]
    if not isinstance(updates, list):
        raise ValueError("material-identity updates must be a JSON array")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("material-identity metadata must be a JSON object")
    return MaterialIdentityHypothesisStreamV1(
        sequence_id=record["sequence_id"],
        case_id=record["case_id"],
        stream_id=record["stream_id"],
        source_repository=record["source_repository"],
        source_revision=record["source_revision"],
        root_window_id=record["root_window_id"],
        updates=tuple(_update_from_record(item) for item in updates),
        metadata=metadata,
        artifact_id=record["artifact_id"],
    )


__all__ = [
    "MATERIAL_IDENTITY_STREAM_SCHEMA",
    "MATERIAL_IDENTITY_STREAM_VERSION",
    "MaterialIdentityAssociationSummaryV1",
    "MaterialIdentityHypothesisStreamV1",
    "MaterialIdentityHypothesisV1",
    "MaterialIdentityStreamUpdateV1",
    "append_material_identity_update",
    "association_summary_from_result",
    "create_material_identity_stream",
    "load_material_identity_stream",
    "write_material_identity_stream",
]
