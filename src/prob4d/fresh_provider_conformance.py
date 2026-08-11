"""Deterministic conformance corpus for fresh-provider readiness decisions.

The corpus isolates every terminal readiness classification without target data.
It is implementation and interoperability evidence only; it does not establish
provider competence, calibration, downstream benefit, or target authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import require_exact_string
from .fresh_provider_readiness import (
    FreshProviderCohortLockV1,
    FreshProviderReadinessRequestV1,
    GateName,
    GateStatus,
    ReadinessClassification,
    ReadinessGateV1,
    authorize_fresh_provider_target,
    evaluate_fresh_provider_readiness,
)

FRESH_PROVIDER_CONFORMANCE_SCHEMA: Final = (
    "prob4d.fresh-provider-readiness-conformance"
)
FRESH_PROVIDER_CONFORMANCE_VERSION: Final = 1
FRESH_PROVIDER_CONFORMANCE_CLAIM_BOUNDARY: Final = (
    "This deterministic target-free corpus verifies only the software mapping from "
    "ordered readiness gates to terminal decisions, development authorization, "
    "one-shot target budget, and exact target-roster binding. It does not establish "
    "provider competence, uncertainty calibration, BayesianPhysTwin benefit, "
    "Causal4D benefit, deployment safety, or state of the art."
)

_GATE_ORDER: Final[tuple[GateName, ...]] = (
    "support-feasibility",
    "source-mean",
    "identity-reliability",
    "gauge-dependence",
    "point-covariance",
    "query-relevance",
    "exact-fallback",
)

CaseTerminalStatus = Literal["fail", "technical-failure", "all-pass"]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_case_id(value: object) -> str:
    result = require_exact_string(value, name="case_id")
    if result != result.lower() or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in result
    ):
        raise ValueError("case_id must use lowercase ASCII letters, digits, and hyphens")
    return result


@dataclass(frozen=True, slots=True)
class FreshProviderConformanceCaseV1:
    """One isolated expected readiness outcome."""

    case_id: str
    expected_classification: ReadinessClassification
    terminal_gate: GateName
    terminal_status: CaseTerminalStatus

    def __post_init__(self) -> None:
        case_id = _canonical_case_id(self.case_id)
        classification = require_exact_string(
            self.expected_classification,
            name="expected_classification",
        )
        allowed_classifications = {
            "support-negative",
            "source-mean-negative",
            "identity-or-association-negative",
            "gauge-or-dependence-negative",
            "point-covariance-localized",
            "query-irrelevant-or-nonidentifiable",
            "ready-for-one-target-evaluation",
            "technical-failure",
        }
        if classification not in allowed_classifications:
            raise ValueError("expected_classification is unsupported")
        gate = require_exact_string(self.terminal_gate, name="terminal_gate")
        if gate not in _GATE_ORDER:
            raise ValueError("terminal_gate is unsupported")
        status = require_exact_string(self.terminal_status, name="terminal_status")
        if status not in {"fail", "technical-failure", "all-pass"}:
            raise ValueError("terminal_status is unsupported")
        if status == "all-pass":
            if classification != "ready-for-one-target-evaluation":
                raise ValueError("all-pass must expect target readiness")
            if gate != "exact-fallback":
                raise ValueError("all-pass terminates at exact-fallback")
        elif classification == "ready-for-one-target-evaluation":
            raise ValueError("a terminal failure cannot expect target readiness")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(
            self,
            "expected_classification",
            cast(ReadinessClassification, classification),
        )
        object.__setattr__(self, "terminal_gate", cast(GateName, gate))
        object.__setattr__(self, "terminal_status", cast(CaseTerminalStatus, status))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "expected_classification": self.expected_classification,
            "terminal_gate": self.terminal_gate,
            "terminal_status": self.terminal_status,
        }


CONFORMANCE_CASES: Final[tuple[FreshProviderConformanceCaseV1, ...]] = (
    FreshProviderConformanceCaseV1(
        case_id="support-negative",
        expected_classification="support-negative",
        terminal_gate="support-feasibility",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="source-mean-negative",
        expected_classification="source-mean-negative",
        terminal_gate="source-mean",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="identity-negative",
        expected_classification="identity-or-association-negative",
        terminal_gate="identity-reliability",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="gauge-dependence-negative",
        expected_classification="gauge-or-dependence-negative",
        terminal_gate="gauge-dependence",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="point-covariance-localized",
        expected_classification="point-covariance-localized",
        terminal_gate="point-covariance",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="query-irrelevant",
        expected_classification="query-irrelevant-or-nonidentifiable",
        terminal_gate="query-relevance",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="fallback-contract-invalid",
        expected_classification="technical-failure",
        terminal_gate="exact-fallback",
        terminal_status="fail",
    ),
    FreshProviderConformanceCaseV1(
        case_id="source-mean-technical-failure",
        expected_classification="technical-failure",
        terminal_gate="source-mean",
        terminal_status="technical-failure",
    ),
    FreshProviderConformanceCaseV1(
        case_id="ready-for-one-target-evaluation",
        expected_classification="ready-for-one-target-evaluation",
        terminal_gate="exact-fallback",
        terminal_status="all-pass",
    ),
)


@dataclass(frozen=True, slots=True)
class FreshProviderConformanceResultV1:
    """Observed result for one deterministic fixture."""

    case_id: str
    expected_classification: ReadinessClassification
    observed_classification: ReadinessClassification
    expected_terminal_gate: GateName
    observed_terminal_gate: GateName
    request_id: str
    decision_id: str
    authorization_id: str | None
    authorize_point_uncertainty_development: bool
    authorize_target_evaluation: bool
    target_evaluation_budget: int
    passed: bool
    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "expected_classification": self.expected_classification,
            "observed_classification": self.observed_classification,
            "expected_terminal_gate": self.expected_terminal_gate,
            "observed_terminal_gate": self.observed_terminal_gate,
            "request_id": self.request_id,
            "decision_id": self.decision_id,
            "authorization_id": self.authorization_id,
            "authorize_point_uncertainty_development": (
                self.authorize_point_uncertainty_development
            ),
            "authorize_target_evaluation": self.authorize_target_evaluation,
            "target_evaluation_budget": self.target_evaluation_budget,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class FreshProviderConformanceReportV1:
    """Replay-complete aggregate result for the packaged corpus."""

    results: tuple[FreshProviderConformanceResultV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    all_passed: bool = field(init=False)
    fresh_provider_conformance_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.results) is not tuple or not self.results:
            raise ValueError("results must be a nonempty canonical tuple")
        case_ids = tuple(result.case_id for result in self.results)
        if case_ids != tuple(case.case_id for case in CONFORMANCE_CASES):
            raise ValueError("results must follow the packaged conformance case order")
        if any(
            not isinstance(result, FreshProviderConformanceResultV1)
            for result in self.results
        ):
            raise TypeError("results must contain FreshProviderConformanceResultV1 values")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="conformance metadata"),
        )
        object.__setattr__(self, "all_passed", all(result.passed for result in self.results))
        object.__setattr__(
            self,
            "fresh_provider_conformance_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": FRESH_PROVIDER_CONFORMANCE_SCHEMA,
            "schema_version": FRESH_PROVIDER_CONFORMANCE_VERSION,
            "cases": [case.to_dict() for case in CONFORMANCE_CASES],
            "results": [result.to_dict() for result in self.results],
            "all_passed": self.all_passed,
            "metadata": plain_json(self.metadata),
            "claim_boundary": FRESH_PROVIDER_CONFORMANCE_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["fresh_provider_conformance_id"] = self.fresh_provider_conformance_id
        return result


def _cohort_lock() -> FreshProviderCohortLockV1:
    prefix = "prob4d-fresh-provider-conformance-v1"
    return FreshProviderCohortLockV1(
        protocol_id=prefix,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="1" * 40,
        provider_repository="example/conformance-provider",
        provider_revision="2" * 40,
        model_set_id=_sha256_text(f"{prefix}:model-set"),
        loader_id=_sha256_text(f"{prefix}:loader"),
        cohort_binding_id=_sha256_text(f"{prefix}:cohort"),
        promotion_lock_id=_sha256_text(f"{prefix}:promotion"),
        query_definition_id=_sha256_text(f"{prefix}:query"),
        fallback_identity_id=_sha256_text(f"{prefix}:fallback"),
        development_group_ids=("development-object-1", "development-object-2"),
        calibration_group_ids=("calibration-object-1", "calibration-object-2"),
        target_group_ids=("target-object-1", "target-object-2"),
        confirmation_group_ids=("confirmation-object-1",),
        metadata={"corpus": FRESH_PROVIDER_CONFORMANCE_SCHEMA},
    )


def build_fresh_provider_conformance_request(
    case: FreshProviderConformanceCaseV1,
) -> FreshProviderReadinessRequestV1:
    """Materialize one isolated fixture through the public readiness contracts."""

    if not isinstance(case, FreshProviderConformanceCaseV1):
        raise TypeError("case must be FreshProviderConformanceCaseV1")
    terminal_index = _GATE_ORDER.index(case.terminal_gate)
    gates: list[ReadinessGateV1] = []
    for index, gate_name in enumerate(_GATE_ORDER):
        if case.terminal_status == "all-pass" or index < terminal_index:
            gates.append(
                ReadinessGateV1(
                    gate_name=gate_name,
                    status="pass",
                    evidence_id=_sha256_text(f"{case.case_id}:{gate_name}:pass"),
                    metadata={"conformance_case_id": case.case_id},
                )
            )
        elif index == terminal_index:
            status = cast(GateStatus, case.terminal_status)
            gates.append(
                ReadinessGateV1(
                    gate_name=gate_name,
                    status=status,
                    evidence_id=_sha256_text(f"{case.case_id}:{gate_name}:{status}"),
                    reason_codes=(f"injected-{case.case_id}",),
                    metadata={"conformance_case_id": case.case_id},
                )
            )
        else:
            gates.append(
                ReadinessGateV1(
                    gate_name=gate_name,
                    status="not-evaluated",
                    evidence_id=None,
                )
            )
    return FreshProviderReadinessRequestV1(
        cohort_lock=_cohort_lock(),
        gates=tuple(gates),
        metadata={"conformance_case_id": case.case_id},
    )


def _run_case(case: FreshProviderConformanceCaseV1) -> FreshProviderConformanceResultV1:
    request = build_fresh_provider_conformance_request(case)
    decision = evaluate_fresh_provider_readiness(request)
    expected_point = case.expected_classification == "point-covariance-localized"
    expected_target = case.expected_classification == "ready-for-one-target-evaluation"
    expected_budget = 1 if expected_target else 0
    failure_reasons: list[str] = []
    if decision.classification != case.expected_classification:
        failure_reasons.append("classification-mismatch")
    if decision.terminal_gate != case.terminal_gate:
        failure_reasons.append("terminal-gate-mismatch")
    if decision.authorize_point_uncertainty_development != expected_point:
        failure_reasons.append("point-development-authorization-mismatch")
    if decision.authorize_target_evaluation != expected_target:
        failure_reasons.append("target-authorization-mismatch")
    if decision.target_evaluation_budget != expected_budget:
        failure_reasons.append("target-budget-mismatch")

    authorization_id: str | None = None
    if expected_target:
        authorization = authorize_fresh_provider_target(decision)
        authorization_id = authorization.fresh_provider_target_authorization_id
        if authorization.target_group_ids != request.cohort_lock.target_group_ids:
            failure_reasons.append("target-roster-mismatch")
    return FreshProviderConformanceResultV1(
        case_id=case.case_id,
        expected_classification=case.expected_classification,
        observed_classification=decision.classification,
        expected_terminal_gate=case.terminal_gate,
        observed_terminal_gate=decision.terminal_gate,
        request_id=request.fresh_provider_readiness_request_id,
        decision_id=decision.fresh_provider_readiness_decision_id,
        authorization_id=authorization_id,
        authorize_point_uncertainty_development=(
            decision.authorize_point_uncertainty_development
        ),
        authorize_target_evaluation=decision.authorize_target_evaluation,
        target_evaluation_budget=decision.target_evaluation_budget,
        passed=not failure_reasons,
        failure_reasons=tuple(sorted(failure_reasons)),
    )


def run_fresh_provider_conformance() -> FreshProviderConformanceReportV1:
    """Run every packaged fixture in its canonical order."""

    return FreshProviderConformanceReportV1(
        results=tuple(_run_case(case) for case in CONFORMANCE_CASES),
        metadata={
            "case_count": len(CONFORMANCE_CASES),
            "statistical_unit": "none-software-conformance-only",
            "target_payloads_opened": False,
            "target_outcomes_opened": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic fresh-provider readiness conformance corpus."
    )
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_fresh_provider_conformance()
    print(
        json.dumps(
            report.to_dict(),
            sort_keys=True,
            separators=(",", ":") if arguments.compact else None,
            indent=None if arguments.compact else 2,
            allow_nan=False,
        )
    )
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
