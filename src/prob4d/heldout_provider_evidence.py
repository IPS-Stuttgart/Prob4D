"""Self-contained replay for held-out Prob4D provider promotion evidence.

This artifact composes the already versioned target-blind selection evidence,
target-free promotion lock, exact provider-report bytes, sealed target query rows,
and deterministic promotion report. Loading replays both candidate selection and
the held-out provider/query decision without importing experiment runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ._heldout_promotion_common import (
    _SHA256,
    _atomic_write_json,
    _exact_keys,
    _load_json,
    _strict_digest,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    load_promotion_lock,
    promotion_lock_from_dict,
)
from ._heldout_promotion_query import (
    HeldoutPromotionQueryResultsV1,
    load_query_results,
    query_results_from_dict,
)
from ._heldout_promotion_report import (
    HeldoutProviderPromotionReportV1,
    evaluate_heldout_promotion,
    load_promotion_report,
    promotion_report_from_dict,
)
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._provider_evaluation_manifest import validate_finite_json
from ._selection_evidence_common import _sha256_json
from .selection_evidence import (
    SelectionEvidenceBundleV2,
    load_selection_evidence,
    selection_evidence_from_dict,
)

HELDOUT_PROVIDER_EVIDENCE_SCHEMA = "prob4d.heldout-provider-evidence"
HELDOUT_PROVIDER_EVIDENCE_VERSION = 2
HELDOUT_PROVIDER_EVIDENCE_REPLAY_SCHEMA = "prob4d.heldout-provider-evidence-replay"
HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY = (
    "This artifact makes target-blind candidate selection and the frozen held-out "
    "Prob4D-to-BayesianPhysTwin promotion decision independently replayable for "
    "the exact retained calibration groups, target groups, provider bytes, and "
    "bootstrap plan. A passing replay establishes only integrity of those frozen "
    "decisions; it does not establish Causal4D intervention benefit, deployment "
    "safety, generalization beyond the declared cohort, or state of the art."
)

MethodRole = Literal["provider", "query"]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _provider_report_text(value: Any) -> str:
    if type(value) is not str or not value:
        raise ValueError("provider_report_json must be a nonempty UTF-8 JSON string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("provider_report_json must be valid UTF-8 text") from error
    return value


def _decode_provider_report(value: str) -> Mapping[str, Any]:
    text = _provider_report_text(value)
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("provider_report_json is invalid JSON") from error
    mapping = _strict_mapping(decoded, name="provider report")
    validate_finite_json(mapping, name="provider report")
    return mapping


def _provider_report_sha256(value: str) -> str:
    return hashlib.sha256(_provider_report_text(value).encode("utf-8")).hexdigest()


def _read_exact_utf8(path: Path, *, name: str) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{name} is unreadable or not UTF-8: {path}") from error


@dataclass(frozen=True, slots=True)
class SelectedCandidateBindingV1:
    """Bind the selected calibration candidate to one frozen promotion arm."""

    candidate_id: str
    arm_id: str
    method_role: MethodRole
    method_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _strict_string(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "arm_id",
            _strict_string(self.arm_id, name="arm_id"),
        )
        role = _strict_string(self.method_role, name="method_role")
        if role not in {"provider", "query"}:
            raise ValueError("method_role must be 'provider' or 'query'")
        object.__setattr__(self, "method_role", cast(MethodRole, role))
        object.__setattr__(
            self,
            "method_id",
            _strict_string(self.method_id, name="method_id"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "arm_id": self.arm_id,
            "method_role": self.method_role,
            "method_id": self.method_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SelectedCandidateBindingV1:
        mapping = _strict_mapping(value, name="selected candidate binding")
        _exact_keys(
            mapping,
            {"candidate_id", "arm_id", "method_role", "method_id"},
            name="selected candidate binding",
        )
        return cls(
            candidate_id=mapping["candidate_id"],
            arm_id=mapping["arm_id"],
            method_role=mapping["method_role"],
            method_id=mapping["method_id"],
        )


@dataclass(frozen=True, slots=True)
class HeldoutProviderEvidenceReplayV2:
    """Compact independent replay receipt for a complete evidence artifact."""

    evidence_id: str
    selection_artifact_id: str
    selection_replay_digest: str
    promotion_lock_id: str
    provider_report_sha256: str
    query_results_id: str
    promotion_report_id: str
    selected_candidate_id: str
    selected_arm_id: str
    candidate_order: tuple[str, ...]
    calibration_group_count: int
    target_group_count: int
    bootstrap_resamples: int
    bootstrap_seed: int
    accepted_update_count: int
    fallback_update_count: int
    exact_fallback_count: int
    provider_passed: bool
    query_passed: bool
    overall_passed: bool

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": HELDOUT_PROVIDER_EVIDENCE_REPLAY_SCHEMA,
            "schema_version": HELDOUT_PROVIDER_EVIDENCE_VERSION,
            "evidence_id": self.evidence_id,
            "selection_artifact_id": self.selection_artifact_id,
            "selection_replay_digest": self.selection_replay_digest,
            "promotion_lock_id": self.promotion_lock_id,
            "provider_report_sha256": self.provider_report_sha256,
            "query_results_id": self.query_results_id,
            "promotion_report_id": self.promotion_report_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_arm_id": self.selected_arm_id,
            "candidate_order": list(self.candidate_order),
            "calibration_group_count": self.calibration_group_count,
            "target_group_count": self.target_group_count,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "accepted_update_count": self.accepted_update_count,
            "fallback_update_count": self.fallback_update_count,
            "exact_fallback_count": self.exact_fallback_count,
            "provider_passed": self.provider_passed,
            "query_passed": self.query_passed,
            "overall_passed": self.overall_passed,
        }

    @property
    def replay_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "replay_id": self.replay_id}


@dataclass(frozen=True, slots=True)
class HeldoutProviderEvidenceV2:
    """Replay-complete selection, target, provider, bootstrap, and report evidence."""

    selection_evidence: SelectionEvidenceBundleV2
    promotion_lock: HeldoutProviderPromotionLockV1
    selected_candidate_binding: SelectedCandidateBindingV1
    provider_report_json: str
    query_results: HeldoutPromotionQueryResultsV1
    promotion_report: HeldoutProviderPromotionReportV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selection_evidence, SelectionEvidenceBundleV2):
            raise ValueError("selection_evidence must be SelectionEvidenceBundleV2")
        if not isinstance(self.promotion_lock, HeldoutProviderPromotionLockV1):
            raise ValueError("promotion_lock must be HeldoutProviderPromotionLockV1")
        if not isinstance(self.selected_candidate_binding, SelectedCandidateBindingV1):
            raise ValueError(
                "selected_candidate_binding must be SelectedCandidateBindingV1"
            )
        if not isinstance(self.query_results, HeldoutPromotionQueryResultsV1):
            raise ValueError("query_results must be HeldoutPromotionQueryResultsV1")
        if not isinstance(self.promotion_report, HeldoutProviderPromotionReportV1):
            raise ValueError(
                "promotion_report must be HeldoutProviderPromotionReportV1"
            )

        selection = self.selection_evidence
        lock = self.promotion_lock
        binding = self.selected_candidate_binding
        query = self.query_results
        report = self.promotion_report

        if selection.experiment_id != lock.experiment_id:
            raise ValueError("selection and promotion experiment_id values differ")
        if selection.source_repository != lock.source_repository:
            raise ValueError("selection and promotion source repositories differ")
        if selection.source_revision != lock.source_revision:
            raise ValueError("selection and promotion source revisions differ")

        calibration_groups = tuple(
            sorted({row.group_id for row in selection.calibration_rows})
        )
        if calibration_groups != lock.calibration_group_ids:
            raise ValueError(
                "selection calibration groups do not match the frozen promotion lock"
            )
        deployment_groups = tuple(
            decision.group_id for decision in selection.deployment_decisions
        )
        if deployment_groups != lock.target_group_ids:
            raise ValueError(
                "selection deployment groups do not match the frozen target groups"
            )

        if binding.candidate_id != selection.selected_candidate_id:
            raise ValueError("selected candidate binding does not match selection replay")
        if binding.arm_id != lock.primary_query_arm_id:
            raise ValueError("selected candidate must bind the primary query arm")
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in selection.candidates
        }
        candidate = candidates_by_id[binding.candidate_id]
        if candidate.method_id != binding.method_id:
            raise ValueError("selected candidate binding method_id changed")
        arm = lock.arms_by_id[binding.arm_id]
        expected_method = (
            arm.provider_method_id
            if binding.method_role == "provider"
            else arm.query_method_id
        )
        if expected_method is None or binding.method_id != expected_method:
            raise ValueError(
                "selected candidate method does not match the bound promotion arm"
            )

        provider_text = _provider_report_text(self.provider_report_json)
        provider_mapping = _decode_provider_report(provider_text)
        provider_sha = _provider_report_sha256(provider_text)
        object.__setattr__(self, "provider_report_json", provider_text)

        if query.promotion_lock_id != lock.promotion_lock_id:
            raise ValueError("query results reference a different promotion lock")
        if report.promotion_lock_id != lock.promotion_lock_id:
            raise ValueError("promotion report references a different promotion lock")
        if report.query_results_id != query.query_results_id:
            raise ValueError("promotion report references different query results")
        if report.provider_report_sha256 != provider_sha:
            raise ValueError("promotion report does not bind exact provider-report bytes")
        if (
            report.provider_evaluation_manifest_sha256
            != lock.provider_evaluation_manifest_sha256
        ):
            raise ValueError(
                "promotion report references a different provider evaluation manifest"
            )

        primary_rows = {
            row.group_id: row
            for row in query.rows
            if row.arm_id == binding.arm_id
        }
        if tuple(sorted(primary_rows)) != lock.target_group_ids:
            raise ValueError(
                "query results do not contain one primary-arm row per target group"
            )
        decisions = {
            decision.group_id: decision
            for decision in selection.deployment_decisions
        }
        for group_id in lock.target_group_ids:
            decision = decisions[group_id]
            row = primary_rows[group_id]
            if row.accepted is None or row.accepted != decision.accepted:
                raise ValueError(
                    f"selection/query acceptance mismatch for target group {group_id!r}"
                )
            if row.deployed_artifact_id != decision.deployed_artifact_id:
                raise ValueError(
                    f"selection/query deployed artifact mismatch for {group_id!r}"
                )
            if row.fallback_artifact_id != decision.fallback_artifact_id:
                raise ValueError(
                    f"selection/query fallback artifact mismatch for {group_id!r}"
                )
            if not decision.accepted and row.exact_fallback_reproduced is not True:
                raise ValueError(
                    f"rejected target group {group_id!r} lacks exact fallback evidence"
                )

        replayed_report = evaluate_heldout_promotion(
            lock,
            query,
            provider_mapping,
            provider_report_sha256=provider_sha,
        )
        if replayed_report.to_dict() != report.to_dict():
            raise ValueError(
                "promotion report does not match deterministic evidence replay"
            )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="evidence metadata"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": HELDOUT_PROVIDER_EVIDENCE_SCHEMA,
            "schema_version": HELDOUT_PROVIDER_EVIDENCE_VERSION,
            "selection_evidence": self.selection_evidence.to_dict(),
            "promotion_lock": self.promotion_lock.to_dict(),
            "selected_candidate_binding": self.selected_candidate_binding.to_dict(),
            "provider_report_json": self.provider_report_json,
            "query_results": self.query_results.to_dict(),
            "promotion_report": self.promotion_report.to_dict(),
            "metadata": plain_json(self.metadata),
            "claim_boundary": HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY,
        }

    @property
    def evidence_id(self) -> str:
        return _sha256_json(self.descriptor())

    def replay_report(self) -> HeldoutProviderEvidenceReplayV2:
        selection_replay = self.selection_evidence.replay_report()
        return HeldoutProviderEvidenceReplayV2(
            evidence_id=self.evidence_id,
            selection_artifact_id=self.selection_evidence.artifact_id,
            selection_replay_digest=selection_replay.replay_digest,
            promotion_lock_id=self.promotion_lock.promotion_lock_id,
            provider_report_sha256=_provider_report_sha256(self.provider_report_json),
            query_results_id=self.query_results.query_results_id,
            promotion_report_id=self.promotion_report.report_id,
            selected_candidate_id=self.selection_evidence.selected_candidate_id,
            selected_arm_id=self.selected_candidate_binding.arm_id,
            candidate_order=self.selection_evidence.selection_order,
            calibration_group_count=len(self.promotion_lock.calibration_group_ids),
            target_group_count=len(self.promotion_lock.target_group_ids),
            bootstrap_resamples=self.promotion_lock.bootstrap_resamples,
            bootstrap_seed=self.promotion_lock.bootstrap_seed,
            accepted_update_count=selection_replay.accepted_update_count,
            fallback_update_count=selection_replay.fallback_update_count,
            exact_fallback_count=selection_replay.exact_fallback_count,
            provider_passed=self.promotion_report.provider_decision.get(
                "overall_passed"
            )
            is True,
            query_passed=self.promotion_report.query_decision.get("overall_passed")
            is True,
            overall_passed=self.promotion_report.overall_passed,
        )

    def to_dict(self) -> dict[str, object]:
        replay = self.replay_report()
        return {
            **self.descriptor(),
            "evidence_id": self.evidence_id,
            "replay_id": replay.replay_id,
        }


_EVIDENCE_FIELDS = {
    "schema_name",
    "schema_version",
    "selection_evidence",
    "promotion_lock",
    "selected_candidate_binding",
    "provider_report_json",
    "query_results",
    "promotion_report",
    "metadata",
    "claim_boundary",
    "evidence_id",
    "replay_id",
}


def build_heldout_provider_evidence(
    *,
    selection_evidence: SelectionEvidenceBundleV2,
    promotion_lock: HeldoutProviderPromotionLockV1,
    selected_candidate_binding: SelectedCandidateBindingV1,
    provider_report_json: str,
    query_results: HeldoutPromotionQueryResultsV1,
    promotion_report: HeldoutProviderPromotionReportV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HeldoutProviderEvidenceV2:
    """Build a canonical artifact and independently replay the promotion report."""

    provider_mapping = _decode_provider_report(provider_report_json)
    provider_sha = _provider_report_sha256(provider_report_json)
    observed_report = (
        evaluate_heldout_promotion(
            promotion_lock,
            query_results,
            provider_mapping,
            provider_report_sha256=provider_sha,
        )
        if promotion_report is None
        else promotion_report
    )
    return HeldoutProviderEvidenceV2(
        selection_evidence=selection_evidence,
        promotion_lock=promotion_lock,
        selected_candidate_binding=selected_candidate_binding,
        provider_report_json=provider_report_json,
        query_results=query_results,
        promotion_report=observed_report,
        metadata={} if metadata is None else metadata,
    )


def heldout_provider_evidence_from_dict(value: Any) -> HeldoutProviderEvidenceV2:
    """Parse, validate, and independently replay one evidence-v2 artifact."""

    mapping = _strict_mapping(value, name="held-out provider evidence")
    _exact_keys(mapping, _EVIDENCE_FIELDS, name="held-out provider evidence")
    if mapping["schema_name"] != HELDOUT_PROVIDER_EVIDENCE_SCHEMA:
        raise ValueError("unsupported held-out provider evidence schema")
    if (
        type(mapping["schema_version"]) is not int
        or mapping["schema_version"] != HELDOUT_PROVIDER_EVIDENCE_VERSION
    ):
        raise ValueError("unsupported held-out provider evidence version")
    if mapping["claim_boundary"] != HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY:
        raise ValueError("held-out provider evidence claim boundary changed")

    evidence = HeldoutProviderEvidenceV2(
        selection_evidence=selection_evidence_from_dict(
            mapping["selection_evidence"]
        ),
        promotion_lock=promotion_lock_from_dict(mapping["promotion_lock"]),
        selected_candidate_binding=SelectedCandidateBindingV1.from_dict(
            mapping["selected_candidate_binding"]
        ),
        provider_report_json=_provider_report_text(mapping["provider_report_json"]),
        query_results=query_results_from_dict(mapping["query_results"]),
        promotion_report=promotion_report_from_dict(mapping["promotion_report"]),
        metadata=_strict_mapping(mapping["metadata"], name="evidence metadata"),
    )
    supplied_evidence_id = _strict_digest(
        mapping["evidence_id"],
        name="evidence_id",
        pattern=_SHA256,
    )
    supplied_replay_id = _strict_digest(
        mapping["replay_id"],
        name="replay_id",
        pattern=_SHA256,
    )
    if supplied_evidence_id != evidence.evidence_id:
        raise ValueError("held-out provider evidence_id mismatch")
    if supplied_replay_id != evidence.replay_report().replay_id:
        raise ValueError("held-out provider replay_id mismatch")
    return evidence


def write_heldout_provider_evidence(
    evidence: HeldoutProviderEvidenceV2,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one canonical evidence artifact through the no-clobber JSON path."""

    if not isinstance(evidence, HeldoutProviderEvidenceV2):
        raise ValueError("evidence must be HeldoutProviderEvidenceV2")
    _atomic_write_json(Path(path), evidence.to_dict(), overwrite=overwrite)


def load_heldout_provider_evidence(path: str | Path) -> HeldoutProviderEvidenceV2:
    """Load and independently replay one complete evidence artifact."""

    mapping, _ = _load_json(Path(path), name="held-out provider evidence")
    return heldout_provider_evidence_from_dict(mapping)


def _pack(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prob4d.heldout_provider_evidence pack",
        description="Compose existing selection and held-out artifacts into evidence v2.",
    )
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--promotion-lock", type=Path, required=True)
    parser.add_argument("--provider-report", type=Path, required=True)
    parser.add_argument("--query-results", type=Path, required=True)
    parser.add_argument("--promotion-report", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument(
        "--method-role",
        choices=("provider", "query"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    selection = load_selection_evidence(parsed.selection_evidence)
    lock = load_promotion_lock(parsed.promotion_lock)
    query = load_query_results(parsed.query_results)
    report = load_promotion_report(parsed.promotion_report)
    provider_text = _read_exact_utf8(parsed.provider_report, name="provider report")
    selected = next(
        candidate
        for candidate in selection.candidates
        if candidate.candidate_id == selection.selected_candidate_id
    )
    binding = SelectedCandidateBindingV1(
        candidate_id=selected.candidate_id,
        arm_id=parsed.arm_id,
        method_role=cast(MethodRole, parsed.method_role),
        method_id=selected.method_id,
    )
    evidence = build_heldout_provider_evidence(
        selection_evidence=selection,
        promotion_lock=lock,
        selected_candidate_binding=binding,
        provider_report_json=provider_text,
        query_results=query,
        promotion_report=report,
    )
    write_heldout_provider_evidence(evidence, parsed.output)
    print(evidence.evidence_id)
    return 0


def _verify(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prob4d.heldout_provider_evidence verify",
        description="Independently replay one complete held-out evidence artifact.",
    )
    parser.add_argument("evidence", type=Path)
    parsed = parser.parse_args(arguments)
    evidence = load_heldout_provider_evidence(parsed.evidence)
    print(
        json.dumps(
            evidence.replay_report().to_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Pack or independently verify a complete held-out provider evidence artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pack", add_help=False)
    subparsers.add_parser("verify", add_help=False)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.command == "pack":
        return _pack(remaining)
    if parsed.command == "verify":
        return _verify(remaining)
    raise AssertionError("unreachable evidence command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HELDOUT_PROVIDER_EVIDENCE_CLAIM_BOUNDARY",
    "HELDOUT_PROVIDER_EVIDENCE_REPLAY_SCHEMA",
    "HELDOUT_PROVIDER_EVIDENCE_SCHEMA",
    "HELDOUT_PROVIDER_EVIDENCE_VERSION",
    "HeldoutProviderEvidenceReplayV2",
    "HeldoutProviderEvidenceV2",
    "MethodRole",
    "SelectedCandidateBindingV1",
    "build_heldout_provider_evidence",
    "heldout_provider_evidence_from_dict",
    "load_heldout_provider_evidence",
    "main",
    "write_heldout_provider_evidence",
]
