"""Deterministic failure attribution for held-out provider promotion reports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from ._heldout_promotion_common import (
    _SHA256,
    _atomic_write_json,
    _exact_keys,
    _load_json,
    _strict_bool,
    _strict_digest,
    _strict_mapping,
    _strict_string,
)
from ._heldout_promotion_report import HeldoutProviderPromotionReportV1
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._selection_evidence_common import _sha256_json

HELDOUT_PROMOTION_DIAGNOSIS_SCHEMA = "prob4d.heldout-provider-promotion-diagnosis"
HELDOUT_PROMOTION_DIAGNOSIS_VERSION = 1


class _BoundarySpec(TypedDict):
    priority: int
    summary: str
    next_action: str


DIAGNOSIS_CLAIM_BOUNDARY = (
    "This diagnosis is a deterministic attribution of failed frozen gates and "
    "metric families. It identifies candidate failure boundaries for follow-up; "
    "it is not causal proof, does not authorize target-set retuning, and does not "
    "change the retained promotion decision."
)

_BOUNDARY_SPECS: dict[str, _BoundarySpec] = {
    "technical_integrity": {
        "priority": 10,
        "summary": "The locked target execution exceeded the allowed technical failures.",
        "next_action": (
            "Resolve only the recorded execution or adapter failure under the frozen "
            "retry policy, then replay the unchanged lock; do not alter estimator settings."
        ),
    },
    "fallback_integrity": {
        "priority": 20,
        "summary": "At least one rejected update failed to reproduce the exact physical fallback.",
        "next_action": (
            "Repair artifact identity and deployment-path parity before interpreting any "
            "scientific result; every rejected row must be byte- and metric-equivalent to fallback."
        ),
    },
    "independent_group_support": {
        "priority": 30,
        "summary": "The frozen independent object/session count is below the registered minimum.",
        "next_action": (
            "Acquire additional independent objects or sessions under a new frozen protocol; "
            "frames, points, tracks, or bootstrap resamples cannot replace independent groups."
        ),
    },
    "observation_quality": {
        "priority": 40,
        "summary": "A registered provider accuracy or reconstruction-error rule failed.",
        "next_action": (
            "Localize the error in the visual predictor or adapter on development/calibration "
            "units, then evaluate the changed method on a new unopened target cohort."
        ),
    },
    "gauge_consistency": {
        "priority": 45,
        "summary": "A registered gauge, alignment, seam, drift, or cycle rule failed.",
        "next_action": (
            "Audit relative-gauge estimation and explicit joint gauge propagation, using the "
            "frozen rowwise, framewise, and persistent comparison arms to isolate the boundary."
        ),
    },
    "identity_persistence": {
        "priority": 50,
        "summary": "A registered tracklet, association, or material-identity rule failed.",
        "next_action": (
            "Inspect termination, association precision, and retained identity hypotheses "
            "without rewriting window-local observation IDs or selecting on target outcomes."
        ),
    },
    "uncertainty_calibration": {
        "priority": 55,
        "summary": "A registered uncertainty or accepted-update coverage gate failed.",
        "next_action": (
            "Audit covariance semantics, NLL/NEES, interval width, and calibration transfer; "
            "fit any replacement calibration only on disjoint source/calibration groups."
        ),
    },
    "support_reliability": {
        "priority": 60,
        "summary": "A registered support-retention, selective-risk, or reliability rule failed.",
        "next_action": (
            "Inspect whether abstention or missing support hides difficult observations and "
            "retain common-support and native-support results separately."
        ),
    },
    "provider_competence_unspecified": {
        "priority": 65,
        "summary": "The provider decision failed without a recognized metric-family boundary.",
        "next_action": (
            "Inspect the exact failed preregistered rules and add an explicit diagnostic only in "
            "a new protocol; do not reinterpret the opened target result post hoc."
        ),
    },
    "guard_calibration": {
        "priority": 70,
        "summary": "The Bayesian guard accepted more harmful updates than the frozen limit.",
        "next_action": (
            "Audit guard features and thresholds on development/calibration evidence only; "
            "prefer additional rejection over accepting a harmful target update."
        ),
    },
    "object_session_transfer": {
        "priority": 80,
        "summary": "The primary arm exceeded the frozen worst-object/session regression limit.",
        "next_action": (
            "Retain the failing groups, characterize the shift without retuning, and test any "
            "revised provider or guard on a newly frozen independent cohort."
        ),
    },
    "query_identifiability_or_physical_model_discrepancy": {
        "priority": 90,
        "summary": (
            "Provider competence passed, but the guarded physical query did not clear the "
            "registered superiority margin."
        ),
        "next_action": (
            "Audit BayesianPhysTwin sensitivity, state/parameter identifiability, and physical "
            "model discrepancy while preserving the successful provider result and target freeze."
        ),
    },
    "downstream_query_superiority": {
        "priority": 95,
        "summary": (
            "The guarded physical query did not clear the superiority margin while provider "
            "competence also failed."
        ),
        "next_action": (
            "Resolve the upstream provider boundary first; the opened result cannot distinguish "
            "query identifiability from observation failure."
        ),
    },
    "promotion_ready": {
        "priority": 1000,
        "summary": "All frozen provider-competence and guarded-query gates passed.",
        "next_action": (
            "Retain and independently replay the sealed evidence before making the bounded "
            "promotion claim; no failure boundary was identified."
        ),
    },
}

_PROVIDER_METRIC_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "technical_integrity",
        ("technical", "execution_failure", "adapter_failure", "load_failure"),
    ),
    (
        "fallback_integrity",
        ("fallback", "deployment_parity"),
    ),
    (
        "gauge_consistency",
        ("gauge", "seam", "drift", "alignment", "cycle"),
    ),
    (
        "identity_persistence",
        ("identity", "association", "tracklet", "track_termination", "termination"),
    ),
    (
        "uncertainty_calibration",
        ("coverage", "nll", "nees", "covariance", "interval", "width", "calibration"),
    ),
    (
        "support_reliability",
        ("support", "retention", "selective", "missingness", "reliability", "abstention"),
    ),
    (
        "observation_quality",
        ("point", "endpoint", "scene_flow", "flow", "rmse", "error", "distance"),
    ),
)

_QUERY_GATE_BOUNDARIES: tuple[tuple[str, str], ...] = (
    ("target_group_count_passed", "independent_group_support"),
    ("harmful_accepted_updates_passed", "guard_calibration"),
    ("worst_group_regression_passed", "object_session_transfer"),
    ("technical_failures_passed", "technical_integrity"),
    ("accepted_coverage_passed", "uncertainty_calibration"),
    ("exact_fallback_passed", "fallback_integrity"),
)


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(value)


def _unique_strings(value: Any, *, name: str, nonempty: bool = False) -> tuple[str, ...]:
    items = _sequence(value, name=name)
    result = tuple(
        _strict_string(item, name=f"{name}[{index}]") for index, item in enumerate(items)
    )
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _required_bool(mapping: Mapping[str, Any], key: str, *, name: str) -> bool:
    if key not in mapping:
        raise ValueError(f"{name} is missing {key!r}")
    return _strict_bool(mapping[key], name=f"{name}.{key}")


def _provider_rule_boundary(rule: Mapping[str, Any]) -> str:
    text = " ".join(
        value.lower().replace("-", "_")
        for key in ("rule_id", "metric", "metric_path")
        if isinstance((value := rule.get(key)), str)
    )
    for boundary_id, tokens in _PROVIDER_METRIC_FAMILIES:
        if any(token in text for token in tokens):
            return boundary_id
    return "provider_competence_unspecified"


def _query_gate_evidence(gate_id: str, decision: Mapping[str, Any]) -> dict[str, object]:
    evidence: dict[str, object] = {
        "evidence_id": f"query-gate:{gate_id}",
        "source": "query_decision",
        "gate_id": gate_id,
        "classification_basis": "direct-gate",
    }
    if gate_id == "target_group_count_passed":
        evidence.update(
            {
                "observed": decision.get("observed_target_group_count"),
                "required": f">= {decision.get('minimum_target_group_count')}",
            }
        )
    elif gate_id == "harmful_accepted_updates_passed":
        evidence.update(
            {
                "observed": decision.get("observed_harmful_accepted_updates"),
                "required": f"<= {decision.get('maximum_harmful_accepted_updates')}",
            }
        )
    elif gate_id == "worst_group_regression_passed":
        evidence.update(
            {
                "observed": decision.get("observed_worst_group_regression_mm"),
                "required": f"<= {decision.get('maximum_worst_group_regression_mm')} mm",
            }
        )
    elif gate_id == "technical_failures_passed":
        evidence.update(
            {
                "observed": decision.get("observed_technical_failures"),
                "required": f"<= {decision.get('maximum_technical_failures')}",
            }
        )
    elif gate_id == "accepted_coverage_passed":
        evidence.update(
            {
                "observed": decision.get("observed_mean_accepted_coverage"),
                "required": f">= {decision.get('minimum_mean_accepted_coverage')}",
            }
        )
    elif gate_id == "exact_fallback_passed":
        evidence.update(
            {
                "observed": decision.get("exact_fallback_failure_count"),
                "required": "= 0 failures",
            }
        )
    elif gate_id == "query_superiority_passed":
        bootstrap = decision.get("paired_bootstrap")
        if isinstance(bootstrap, Mapping):
            evidence["observed"] = bootstrap.get("ci95_upper")
        evidence["required"] = f"<= {-float(decision.get('query_superiority_margin_mm', 0.0))} mm"
        evidence["classification_basis"] = "cross-gate-inference"
    return evidence


def _provider_rule_evidence(
    rule: Mapping[str, Any],
    *,
    rule_id: str,
) -> dict[str, object]:
    return {
        "evidence_id": f"provider-rule:{rule_id}",
        "source": "provider_decision",
        "rule_id": rule_id,
        "metric": rule.get("metric"),
        "metric_path": rule.get("metric_path"),
        "candidate_method": rule.get("candidate_method"),
        "decision_bound": rule.get("decision_bound"),
        "observed": rule.get("decision_bound_value"),
        "required": rule.get("pass_threshold"),
        "classification_basis": "metric-family-inference",
    }


def _add_evidence(
    evidence_by_boundary: dict[str, list[dict[str, object]]],
    boundary_id: str,
    evidence: dict[str, object],
) -> None:
    if boundary_id not in _BOUNDARY_SPECS:
        raise AssertionError(f"unknown promotion diagnosis boundary {boundary_id!r}")
    evidence_by_boundary.setdefault(boundary_id, []).append(evidence)


@dataclass(frozen=True, slots=True)
class HeldoutPromotionDiagnosisV1:
    """Content-addressed candidate failure boundaries for one promotion report."""

    report_id: str
    promotion_lock_id: str
    overall_passed: bool
    failed_provider_rule_ids: tuple[str, ...]
    failed_query_gate_ids: tuple[str, ...]
    boundary_ids: tuple[str, ...]
    findings: tuple[Mapping[str, Any], ...]
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_id",
            _strict_digest(self.report_id, name="report_id", pattern=_SHA256),
        )
        object.__setattr__(
            self,
            "promotion_lock_id",
            _strict_digest(
                self.promotion_lock_id,
                name="promotion_lock_id",
                pattern=_SHA256,
            ),
        )
        object.__setattr__(
            self,
            "overall_passed",
            _strict_bool(self.overall_passed, name="overall_passed"),
        )
        for field_name in (
            "failed_provider_rule_ids",
            "failed_query_gate_ids",
            "boundary_ids",
        ):
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise ValueError(f"{field_name} must be a tuple")
            normalized = tuple(
                _strict_string(value, name=f"{field_name}[{index}]")
                for index, value in enumerate(values)
            )
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{field_name} must contain unique values")
            object.__setattr__(self, field_name, normalized)
        if not self.boundary_ids:
            raise ValueError("boundary_ids must not be empty")
        if type(self.findings) is not tuple:
            raise ValueError("findings must be a tuple")
        normalized_findings: list[Mapping[str, Any]] = []
        finding_boundary_ids: list[str] = []
        for index, finding in enumerate(self.findings):
            normalized = frozen_finite_json_mapping(
                _strict_mapping(finding, name=f"findings[{index}]"),
                name=f"findings[{index}]",
            )
            boundary_id = _strict_string(
                normalized.get("boundary_id"),
                name=f"findings[{index}].boundary_id",
            )
            finding_boundary_ids.append(boundary_id)
            normalized_findings.append(normalized)
        if tuple(finding_boundary_ids) != self.boundary_ids:
            raise ValueError("findings must match boundary_ids in the same order")
        if self.overall_passed != (self.boundary_ids == ("promotion_ready",)):
            raise ValueError("overall_passed is inconsistent with boundary_ids")
        object.__setattr__(self, "findings", tuple(normalized_findings))
        object.__setattr__(self, "summary", _strict_string(self.summary, name="summary"))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": HELDOUT_PROMOTION_DIAGNOSIS_SCHEMA,
            "schema_version": HELDOUT_PROMOTION_DIAGNOSIS_VERSION,
            "report_id": self.report_id,
            "promotion_lock_id": self.promotion_lock_id,
            "overall_passed": self.overall_passed,
            "failed_provider_rule_ids": list(self.failed_provider_rule_ids),
            "failed_query_gate_ids": list(self.failed_query_gate_ids),
            "boundary_ids": list(self.boundary_ids),
            "findings": [plain_json(finding) for finding in self.findings],
            "summary": self.summary,
            "claim_boundary": DIAGNOSIS_CLAIM_BOUNDARY,
        }

    @property
    def diagnosis_id(self) -> str:
        return _sha256_json(self.descriptor())

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor(), "diagnosis_id": self.diagnosis_id}


_DIAGNOSIS_FIELDS = {
    "schema_name",
    "schema_version",
    "report_id",
    "promotion_lock_id",
    "overall_passed",
    "failed_provider_rule_ids",
    "failed_query_gate_ids",
    "boundary_ids",
    "findings",
    "summary",
    "claim_boundary",
    "diagnosis_id",
}


def diagnose_heldout_promotion(
    report: HeldoutProviderPromotionReportV1,
) -> HeldoutPromotionDiagnosisV1:
    """Attribute failed frozen gates to deterministic candidate boundaries."""

    if not isinstance(report, HeldoutProviderPromotionReportV1):
        raise ValueError("report must be HeldoutProviderPromotionReportV1")
    provider = report.provider_decision
    query = report.query_decision
    provider_passed = _required_bool(provider, "overall_passed", name="provider_decision")
    query_passed = _required_bool(query, "overall_passed", name="query_decision")
    if report.overall_passed != (provider_passed and query_passed):
        raise ValueError("promotion report overall decision is internally inconsistent")

    evidence_by_boundary: dict[str, list[dict[str, object]]] = {}
    failed_provider_rule_ids: list[str] = []
    failed_query_gate_ids: list[str] = []

    provider_group_count_passed = _required_bool(
        provider,
        "group_count_passed",
        name="provider_decision",
    )
    if not provider_group_count_passed:
        _add_evidence(
            evidence_by_boundary,
            "independent_group_support",
            {
                "evidence_id": "provider-gate:group_count_passed",
                "source": "provider_decision",
                "gate_id": "group_count_passed",
                "observed": provider.get("observed_group_count"),
                "required": f">= {provider.get('minimum_group_count')}",
                "classification_basis": "direct-gate",
            },
        )

    raw_rules = provider.get("rules")
    rules = _sequence(raw_rules, name="provider_decision.rules")
    provider_rule_states: list[bool] = []
    for index, raw_rule in enumerate(rules):
        rule = _strict_mapping(raw_rule, name=f"provider_decision.rules[{index}]")
        passed = _required_bool(
            rule,
            "passed",
            name=f"provider_decision.rules[{index}]",
        )
        provider_rule_states.append(passed)
        if passed:
            continue
        supplied_rule_id = rule.get("rule_id")
        rule_id = (
            _strict_string(supplied_rule_id, name=f"provider_decision.rules[{index}].rule_id")
            if supplied_rule_id is not None
            else f"provider-rule-{index}"
        )
        if rule_id in failed_provider_rule_ids:
            raise ValueError("provider_decision contains duplicate failed rule IDs")
        failed_provider_rule_ids.append(rule_id)
        boundary_id = _provider_rule_boundary(rule)
        _add_evidence(
            evidence_by_boundary,
            boundary_id,
            _provider_rule_evidence(rule, rule_id=rule_id),
        )

    expected_provider_passed = provider_group_count_passed and all(provider_rule_states)
    if provider_passed != expected_provider_passed:
        raise ValueError("provider_decision overall result is internally inconsistent")

    query_gate_states: list[bool] = []
    for gate_id, boundary_id in _QUERY_GATE_BOUNDARIES:
        passed = _required_bool(query, gate_id, name="query_decision")
        query_gate_states.append(passed)
        if passed:
            continue
        failed_query_gate_ids.append(gate_id)
        _add_evidence(
            evidence_by_boundary,
            boundary_id,
            _query_gate_evidence(gate_id, query),
        )

    superiority_passed = _required_bool(
        query,
        "query_superiority_passed",
        name="query_decision",
    )
    query_gate_states.append(superiority_passed)
    if query_passed != all(query_gate_states):
        raise ValueError("query_decision overall result is internally inconsistent")
    if not superiority_passed:
        gate_id = "query_superiority_passed"
        failed_query_gate_ids.append(gate_id)
        boundary_id = (
            "query_identifiability_or_physical_model_discrepancy"
            if provider_passed
            else "downstream_query_superiority"
        )
        _add_evidence(
            evidence_by_boundary,
            boundary_id,
            _query_gate_evidence(gate_id, query),
        )

    if report.overall_passed:
        _add_evidence(
            evidence_by_boundary,
            "promotion_ready",
            {
                "evidence_id": "promotion-gate:overall_passed",
                "source": "promotion_report",
                "gate_id": "overall_passed",
                "observed": True,
                "required": True,
                "classification_basis": "direct-gate",
            },
        )

    ordered_boundary_ids = tuple(
        sorted(
            evidence_by_boundary,
            key=lambda boundary_id: (
                _BOUNDARY_SPECS[boundary_id]["priority"],
                boundary_id,
            ),
        )
    )
    if not ordered_boundary_ids:
        raise ValueError("promotion diagnosis found no passing or failed gate evidence")

    findings: list[Mapping[str, Any]] = []
    for boundary_id in ordered_boundary_ids:
        specification = _BOUNDARY_SPECS[boundary_id]
        evidence = sorted(
            evidence_by_boundary[boundary_id],
            key=lambda item: str(item["evidence_id"]),
        )
        findings.append(
            {
                "boundary_id": boundary_id,
                "priority": specification["priority"],
                "summary": specification["summary"],
                "next_action": specification["next_action"],
                "evidence": evidence,
            }
        )

    if report.overall_passed:
        summary = "All frozen provider and guarded-query gates passed."
    else:
        primary = ordered_boundary_ids[0]
        summary = (
            f"Identified {len(ordered_boundary_ids)} candidate failure "
            f"{'boundary' if len(ordered_boundary_ids) == 1 else 'boundaries'}; "
            f"inspect {primary!r} first."
        )
    return HeldoutPromotionDiagnosisV1(
        report_id=report.report_id,
        promotion_lock_id=report.promotion_lock_id,
        overall_passed=report.overall_passed,
        failed_provider_rule_ids=tuple(failed_provider_rule_ids),
        failed_query_gate_ids=tuple(failed_query_gate_ids),
        boundary_ids=ordered_boundary_ids,
        findings=tuple(findings),
        summary=summary,
    )


def promotion_diagnosis_from_dict(value: Any) -> HeldoutPromotionDiagnosisV1:
    mapping = _strict_mapping(value, name="held-out promotion diagnosis")
    _exact_keys(mapping, _DIAGNOSIS_FIELDS, name="held-out promotion diagnosis")
    if mapping["schema_name"] != HELDOUT_PROMOTION_DIAGNOSIS_SCHEMA:
        raise ValueError("unsupported held-out promotion diagnosis schema")
    if mapping["schema_version"] != HELDOUT_PROMOTION_DIAGNOSIS_VERSION:
        raise ValueError("unsupported held-out promotion diagnosis version")
    if mapping["claim_boundary"] != DIAGNOSIS_CLAIM_BOUNDARY:
        raise ValueError("held-out promotion diagnosis claim boundary changed")
    raw_findings = _sequence(mapping["findings"], name="findings")
    diagnosis = HeldoutPromotionDiagnosisV1(
        report_id=mapping["report_id"],
        promotion_lock_id=mapping["promotion_lock_id"],
        overall_passed=mapping["overall_passed"],
        failed_provider_rule_ids=_unique_strings(
            mapping["failed_provider_rule_ids"],
            name="failed_provider_rule_ids",
        ),
        failed_query_gate_ids=_unique_strings(
            mapping["failed_query_gate_ids"],
            name="failed_query_gate_ids",
        ),
        boundary_ids=_unique_strings(
            mapping["boundary_ids"],
            name="boundary_ids",
            nonempty=True,
        ),
        findings=tuple(
            _strict_mapping(item, name=f"findings[{index}]")
            for index, item in enumerate(raw_findings)
        ),
        summary=mapping["summary"],
    )
    supplied = _strict_digest(
        mapping["diagnosis_id"],
        name="diagnosis_id",
        pattern=_SHA256,
    )
    if supplied != diagnosis.diagnosis_id:
        raise ValueError("held-out promotion diagnosis ID mismatch")
    return diagnosis


def write_promotion_diagnosis(
    diagnosis: HeldoutPromotionDiagnosisV1,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(diagnosis, HeldoutPromotionDiagnosisV1):
        raise ValueError("diagnosis must be HeldoutPromotionDiagnosisV1")
    _atomic_write_json(Path(path), diagnosis.to_dict(), overwrite=overwrite)


def load_promotion_diagnosis(
    path: str | os.PathLike[str],
) -> HeldoutPromotionDiagnosisV1:
    mapping, _ = _load_json(Path(path), name="held-out promotion diagnosis")
    return promotion_diagnosis_from_dict(mapping)


def write_promotion_diagnosis_markdown(
    diagnosis: HeldoutPromotionDiagnosisV1,
    path: str | os.PathLike[str],
) -> None:
    if not isinstance(diagnosis, HeldoutPromotionDiagnosisV1):
        raise ValueError("diagnosis must be HeldoutPromotionDiagnosisV1")
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    status = "PASS" if diagnosis.overall_passed else "FAIL"
    lines = [
        "# Held-out promotion diagnosis",
        "",
        f"Promotion report: `{diagnosis.report_id}`.",
        f"Diagnosis: `{diagnosis.diagnosis_id}`.",
        f"Overall promotion decision: **{status}**.",
        "",
        diagnosis.summary,
        "",
        "## Candidate boundaries",
        "",
    ]
    for index, finding in enumerate(diagnosis.findings, start=1):
        boundary_id = finding["boundary_id"]
        lines.extend(
            [
                f"### {index}. `{boundary_id}`",
                "",
                str(finding["summary"]),
                "",
                "Evidence:",
            ]
        )
        evidence_items = finding["evidence"]
        if not isinstance(evidence_items, tuple):
            evidence_items = tuple(evidence_items)
        for evidence in evidence_items:
            observed = evidence.get("observed")
            required = evidence.get("required")
            lines.append(
                f"- `{evidence['evidence_id']}`: observed `{observed}`, required `{required}`."
            )
        lines.extend(
            [
                "",
                f"Next action: {finding['next_action']}",
                "",
            ]
        )
    lines.extend(["## Claim boundary", "", DIAGNOSIS_CLAIM_BOUNDARY, ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "DIAGNOSIS_CLAIM_BOUNDARY",
    "HELDOUT_PROMOTION_DIAGNOSIS_SCHEMA",
    "HELDOUT_PROMOTION_DIAGNOSIS_VERSION",
    "HeldoutPromotionDiagnosisV1",
    "diagnose_heldout_promotion",
    "load_promotion_diagnosis",
    "promotion_diagnosis_from_dict",
    "write_promotion_diagnosis",
    "write_promotion_diagnosis_markdown",
]
