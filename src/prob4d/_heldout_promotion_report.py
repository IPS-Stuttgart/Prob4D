"""Retained result and replay surface for held-out provider promotion."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._heldout_promotion_common import (
    _SHA256,
    HELDOUT_PROMOTION_REPORT_SCHEMA,
    HELDOUT_PROMOTION_REPORT_VERSION,
    REPORT_CLAIM_BOUNDARY,
    _atomic_write_json,
    _exact_keys,
    _load_json,
    _strict_bool,
    _strict_digest,
    _strict_mapping,
)
from ._heldout_promotion_evaluation import (
    _provider_report_audit,
    _query_evaluation,
)
from ._heldout_promotion_lock import HeldoutProviderPromotionLockV1
from ._heldout_promotion_query import HeldoutPromotionQueryResultsV1
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import _sha256_json


@dataclass(frozen=True, slots=True)
class HeldoutProviderPromotionReportV1:
    """Replayable composition of provider competence and guarded-query gates."""

    promotion_lock_id: str
    query_results_id: str
    provider_report_sha256: str
    provider_evaluation_manifest_sha256: str
    provider_audit: Mapping[str, Any]
    provider_decision: Mapping[str, Any]
    query_aggregate: Mapping[str, Any]
    query_decision: Mapping[str, Any]
    overall_passed: bool

    def __post_init__(self) -> None:
        for field_name in (
            "promotion_lock_id",
            "query_results_id",
            "provider_report_sha256",
            "provider_evaluation_manifest_sha256",
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
        for field_name in (
            "provider_audit",
            "provider_decision",
            "query_aggregate",
            "query_decision",
        ):
            object.__setattr__(
                self,
                field_name,
                frozen_finite_json_mapping(
                    _strict_mapping(getattr(self, field_name), name=field_name),
                    name=field_name,
                ),
            )
        object.__setattr__(
            self,
            "overall_passed",
            _strict_bool(self.overall_passed, name="overall_passed"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": HELDOUT_PROMOTION_REPORT_SCHEMA,
            "schema_version": HELDOUT_PROMOTION_REPORT_VERSION,
            "promotion_lock_id": self.promotion_lock_id,
            "query_results_id": self.query_results_id,
            "provider_report_sha256": self.provider_report_sha256,
            "provider_evaluation_manifest_sha256": (
                self.provider_evaluation_manifest_sha256
            ),
            "provider_audit": plain_json(self.provider_audit),
            "provider_decision": plain_json(self.provider_decision),
            "query_aggregate": plain_json(self.query_aggregate),
            "query_decision": plain_json(self.query_decision),
            "overall_passed": self.overall_passed,
            "claim_boundary": REPORT_CLAIM_BOUNDARY,
        }

    @property
    def report_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "report_id": self.report_id}


_REPORT_FIELDS = {
    "schema_name",
    "schema_version",
    "promotion_lock_id",
    "query_results_id",
    "provider_report_sha256",
    "provider_evaluation_manifest_sha256",
    "provider_audit",
    "provider_decision",
    "query_aggregate",
    "query_decision",
    "overall_passed",
    "claim_boundary",
    "report_id",
}


def evaluate_heldout_promotion(
    lock: HeldoutProviderPromotionLockV1,
    query_results: HeldoutPromotionQueryResultsV1,
    provider_report: Mapping[str, Any],
    *,
    provider_report_sha256: str,
) -> HeldoutProviderPromotionReportV1:
    """Evaluate both frozen gates without changing any estimator or target split."""

    if not isinstance(lock, HeldoutProviderPromotionLockV1):
        raise ValueError("lock must be HeldoutProviderPromotionLockV1")
    if not isinstance(query_results, HeldoutPromotionQueryResultsV1):
        raise ValueError("query_results must be HeldoutPromotionQueryResultsV1")
    provider_sha = _strict_digest(
        provider_report_sha256,
        name="provider_report_sha256",
        pattern=_SHA256,
    )
    provider_audit, provider_decision = _provider_report_audit(lock, provider_report)
    query_aggregate, query_decision = _query_evaluation(lock, query_results)
    provider_passed = provider_decision.get("overall_passed") is True
    query_passed = query_decision.get("overall_passed") is True
    return HeldoutProviderPromotionReportV1(
        promotion_lock_id=lock.promotion_lock_id,
        query_results_id=query_results.query_results_id,
        provider_report_sha256=provider_sha,
        provider_evaluation_manifest_sha256=(
            lock.provider_evaluation_manifest_sha256
        ),
        provider_audit=provider_audit,
        provider_decision=provider_decision,
        query_aggregate=query_aggregate,
        query_decision=query_decision,
        overall_passed=provider_passed and query_passed,
    )


def promotion_report_from_dict(value: Any) -> HeldoutProviderPromotionReportV1:
    mapping = _strict_mapping(value, name="held-out promotion report")
    _exact_keys(mapping, _REPORT_FIELDS, name="held-out promotion report")
    if mapping["schema_name"] != HELDOUT_PROMOTION_REPORT_SCHEMA:
        raise ValueError("unsupported held-out promotion report schema")
    if mapping["schema_version"] != HELDOUT_PROMOTION_REPORT_VERSION:
        raise ValueError("unsupported held-out promotion report version")
    if mapping["claim_boundary"] != REPORT_CLAIM_BOUNDARY:
        raise ValueError("held-out promotion report claim boundary changed")
    report = HeldoutProviderPromotionReportV1(
        promotion_lock_id=mapping["promotion_lock_id"],
        query_results_id=mapping["query_results_id"],
        provider_report_sha256=mapping["provider_report_sha256"],
        provider_evaluation_manifest_sha256=mapping[
            "provider_evaluation_manifest_sha256"
        ],
        provider_audit=_strict_mapping(mapping["provider_audit"], name="provider_audit"),
        provider_decision=_strict_mapping(
            mapping["provider_decision"],
            name="provider_decision",
        ),
        query_aggregate=_strict_mapping(mapping["query_aggregate"], name="query_aggregate"),
        query_decision=_strict_mapping(mapping["query_decision"], name="query_decision"),
        overall_passed=mapping["overall_passed"],
    )
    supplied = _strict_digest(mapping["report_id"], name="report_id", pattern=_SHA256)
    if supplied != report.report_id:
        raise ValueError("held-out promotion report ID mismatch")
    return report


def write_promotion_report(
    report: HeldoutProviderPromotionReportV1,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(report, HeldoutProviderPromotionReportV1):
        raise ValueError("report must be HeldoutProviderPromotionReportV1")
    _atomic_write_json(Path(path), report.to_dict(), overwrite=overwrite)


def load_promotion_report(
    path: str | os.PathLike[str],
) -> HeldoutProviderPromotionReportV1:
    mapping, _ = _load_json(Path(path), name="held-out promotion report")
    return promotion_report_from_dict(mapping)


def _write_report_markdown(
    lock: HeldoutProviderPromotionLockV1,
    report: HeldoutProviderPromotionReportV1,
    path: Path,
) -> None:
    query = report.query_decision
    provider_status = "PASS" if report.provider_decision.get("overall_passed") else "FAIL"
    query_status = "PASS" if query.get("overall_passed") else "FAIL"
    overall = "PASS" if report.overall_passed else "FAIL"
    lines = [
        "# Held-out Prob4D-to-BayesianPhysTwin promotion gate",
        "",
        f"Promotion lock: `{lock.promotion_lock_id}`.",
        f"Provider report: `{report.provider_report_sha256}`.",
        f"Query results: `{report.query_results_id}`.",
        "",
        f"Provider competence: **{provider_status}**.",
        f"Guarded physical query: **{query_status}**.",
        f"Overall promotion decision: **{overall}**.",
        "",
        "## Guarded-query gates",
        "",
        "| Gate | Observed | Frozen requirement | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    gate_rows = (
        (
            "Paired RMSE difference upper 95% bound",
            query["paired_bootstrap"]["ci95_upper"],
            f"≤ {-lock.query_superiority_margin_mm:.6g} mm",
            query["query_superiority_passed"],
        ),
        (
            "Harmful accepted updates",
            query["observed_harmful_accepted_updates"],
            f"≤ {lock.maximum_harmful_accepted_updates}",
            query["harmful_accepted_updates_passed"],
        ),
        (
            "Worst-group regression",
            query["observed_worst_group_regression_mm"],
            f"≤ {lock.maximum_worst_group_regression_mm:.6g} mm",
            query["worst_group_regression_passed"],
        ),
        (
            "Technical failures",
            query["observed_technical_failures"],
            f"≤ {lock.maximum_technical_failures}",
            query["technical_failures_passed"],
        ),
        (
            "Exact fallback failures",
            query["exact_fallback_failure_count"],
            "= 0",
            query["exact_fallback_passed"],
        ),
    )
    for name, observed, requirement, passed in gate_rows:
        rendered_observed = (
            f"{float(observed):.6g}" if isinstance(observed, float) else str(observed)
        )
        lines.append(
            f"| {name} | {rendered_observed} | {requirement} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    if lock.minimum_mean_accepted_coverage is not None:
        lines.append(
            "| Mean accepted-update coverage | "
            f"{query['observed_mean_accepted_coverage']} | "
            f"≥ {lock.minimum_mean_accepted_coverage:.6g} | "
            f"{'PASS' if query['accepted_coverage_passed'] else 'FAIL'} |"
        )
    lines.extend(["", REPORT_CLAIM_BOUNDARY, ""])
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines), encoding="utf-8")
