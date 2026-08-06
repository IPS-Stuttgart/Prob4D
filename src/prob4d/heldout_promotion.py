"""Frozen held-out Prob4D-to-BayesianPhysTwin promotion gate.

This module coordinates existing Prob4D provider evaluation and BayesianPhysTwin
query outcomes without introducing a new estimator. A target-free lock freezes
object/session splits, exact source/model identities, comparison arms, bootstrap
settings, and decision margins before target outcomes are opened. The run step
seals complete group-by-arm query rows, verifies the independently produced
provider report, and evaluates the guarded physical-query promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ._heldout_promotion_common import (
    HELDOUT_PROMOTION_LOCK_SCHEMA,
    HELDOUT_PROMOTION_LOCK_VERSION,
    HELDOUT_PROMOTION_REPORT_SCHEMA,
    HELDOUT_PROMOTION_REPORT_VERSION,
    HELDOUT_QUERY_RESULTS_SCHEMA,
    HELDOUT_QUERY_RESULTS_VERSION,
    PromotionArmV1,
    _load_json,
)
from ._heldout_promotion_diagnosis import (
    HELDOUT_PROMOTION_DIAGNOSIS_SCHEMA,
    HELDOUT_PROMOTION_DIAGNOSIS_VERSION,
    HeldoutPromotionDiagnosisV1,
    diagnose_heldout_promotion,
    load_promotion_diagnosis,
    promotion_diagnosis_from_dict,
    write_promotion_diagnosis,
    write_promotion_diagnosis_markdown,
)
from ._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    load_promotion_lock,
    promotion_lock_from_config,
    promotion_lock_from_dict,
    write_promotion_lock,
)
from ._heldout_promotion_query import (
    HeldoutPromotionQueryResultsV1,
    PromotionQueryRowV1,
    build_query_results,
    load_query_results,
    query_results_from_dict,
    query_results_from_raw,
    write_query_results,
)
from ._heldout_promotion_report import (
    HeldoutProviderPromotionReportV1,
    _write_report_markdown,
    evaluate_heldout_promotion,
    load_promotion_report,
    promotion_report_from_dict,
    write_promotion_report,
)
from .deform360_cohort_binding import (
    DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY,
    DEFORM360_COHORT_BINDING_SCHEMA,
    DEFORM360_COHORT_BINDING_VERSION,
    Deform360CohortUnitV1,
    Deform360OfficialHubCohortBindingV1,
    bind_cli as _cohort_bind,
    build_deform360_official_hub_cohort_binding,
    deform360_cohort_binding_from_dict,
    load_deform360_cohort_binding,
    validate_deform360_cohort_binding_against_selection,
    validate_deform360_official_hub_selection,
    validate_promotion_config_against_deform360_binding,
    verify_cli as _cohort_verify,
    write_deform360_cohort_binding,
)
from .promotion_evidence import (
    build_promotion_evidence_card,
    load_promotion_evidence_card,
    write_promotion_evidence_card,
)


def _freeze(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider freeze",
        description="Seal one target-free held-out provider promotion lock.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--cohort-binding",
        type=Path,
        help=(
            "optionally require exact agreement with BayesianPhysTwin's "
            "official-Hub Deform360 Stage-0 cohort binding"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    config, _ = _load_json(parsed.config, name="promotion lock configuration")
    if parsed.cohort_binding is not None:
        cohort_binding = load_deform360_cohort_binding(parsed.cohort_binding)
        validate_promotion_config_against_deform360_binding(config, cohort_binding)
    lock = promotion_lock_from_config(config)
    write_promotion_lock(lock, parsed.output)
    print(lock.promotion_lock_id)
    return 0


def _run(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider run",
        description="Seal target query rows and evaluate the frozen promotion gate.",
    )
    parser.add_argument("lock", type=Path)
    parser.add_argument("--provider-report", type=Path, required=True)
    parser.add_argument("--query-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    parsed = parser.parse_args(arguments)
    lock = load_promotion_lock(parsed.lock)
    raw_query, _ = _load_json(parsed.query_results, name="raw promotion query results")
    query_results = query_results_from_raw(raw_query, lock=lock)
    provider_report, provider_bytes = _load_json(
        parsed.provider_report,
        name="provider evaluation report",
    )
    report = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256=hashlib.sha256(provider_bytes).hexdigest(),
    )
    diagnosis = diagnose_heldout_promotion(report)
    evidence_card = build_promotion_evidence_card(lock.to_dict(), report.to_dict())
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    query_output = parsed.output_dir / "query_results.sealed.json"
    report_output = parsed.output_dir / "promotion_report.json"
    markdown_output = parsed.output_dir / "promotion_report.md"
    diagnosis_output = parsed.output_dir / "promotion_diagnosis.json"
    diagnosis_markdown_output = parsed.output_dir / "promotion_diagnosis.md"
    evidence_output = parsed.output_dir / "promotion_evidence_card.json"
    evidence_markdown_output = parsed.output_dir / "promotion_evidence_card.md"
    outputs = (
        query_output,
        report_output,
        markdown_output,
        diagnosis_output,
        diagnosis_markdown_output,
        evidence_output,
        evidence_markdown_output,
    )
    existing_outputs = [path for path in outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "held-out promotion output already exists: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    write_query_results(query_results, query_output)
    write_promotion_report(report, report_output)
    _write_report_markdown(lock, report, markdown_output)
    write_promotion_diagnosis(diagnosis, diagnosis_output)
    write_promotion_diagnosis_markdown(diagnosis, diagnosis_markdown_output)
    write_promotion_evidence_card(
        evidence_card,
        evidence_output,
        evidence_markdown_output,
    )
    print(report_output)
    if parsed.require_pass and not report.overall_passed:
        return 3
    return 0


def _verify(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider verify",
        description="Replay and verify one held-out promotion report.",
    )
    parser.add_argument("lock", type=Path)
    parser.add_argument("--provider-report", type=Path, required=True)
    parser.add_argument("--query-results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--evidence-card",
        type=Path,
        help="optionally verify the derived paper-facing evidence card",
    )
    parsed = parser.parse_args(arguments)
    lock = load_promotion_lock(parsed.lock)
    query_results = load_query_results(parsed.query_results)
    provider_report, provider_bytes = _load_json(
        parsed.provider_report,
        name="provider evaluation report",
    )
    observed = load_promotion_report(parsed.report)
    replayed = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256=hashlib.sha256(provider_bytes).hexdigest(),
    )
    if observed.to_dict() != replayed.to_dict():
        raise ValueError("held-out promotion report does not match deterministic replay")
    replayed_evidence = build_promotion_evidence_card(
        lock.to_dict(),
        observed.to_dict(),
    )
    if parsed.evidence_card is not None:
        observed_evidence = load_promotion_evidence_card(parsed.evidence_card)
        if observed_evidence != replayed_evidence:
            raise ValueError("promotion evidence card does not match deterministic replay")
    print(
        json.dumps(
            {
                "promotion_lock_id": lock.promotion_lock_id,
                "query_results_id": query_results.query_results_id,
                "report_id": observed.report_id,
                "evidence_card_id": replayed_evidence["evidence_card_id"],
                "overall_passed": observed.overall_passed,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _diagnose(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider diagnose",
        description="Attribute a retained promotion report to candidate failure boundaries.",
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parsed = parser.parse_args(arguments)
    report = load_promotion_report(parsed.report)
    diagnosis = diagnose_heldout_promotion(report)
    write_promotion_diagnosis(diagnosis, parsed.output)
    if parsed.markdown is not None:
        write_promotion_diagnosis_markdown(diagnosis, parsed.markdown)
    print(parsed.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run cohort binding, promotion freeze, execution, replay, or diagnosis."""

    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "cohort-bind",
        add_help=False,
        help="bind the authoritative Deform360 Stage-0 selection",
    )
    subparsers.add_parser(
        "cohort-verify",
        add_help=False,
        help="replay the cohort binding and optionally rebind the selection",
    )
    subparsers.add_parser(
        "freeze",
        add_help=False,
        help="seal a target-free promotion lock",
    )
    subparsers.add_parser(
        "run",
        add_help=False,
        help="seal query rows and evaluate the frozen gates",
    )
    subparsers.add_parser(
        "verify",
        add_help=False,
        help="replay a retained promotion report and optional evidence card",
    )
    subparsers.add_parser(
        "diagnose",
        add_help=False,
        help="attribute failed gates to candidate failure boundaries",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.command == "cohort-bind":
        return _cohort_bind(remaining)
    if parsed.command == "cohort-verify":
        return _cohort_verify(remaining)
    if parsed.command == "freeze":
        return _freeze(remaining)
    if parsed.command == "run":
        return _run(remaining)
    if parsed.command == "verify":
        return _verify(remaining)
    if parsed.command == "diagnose":
        return _diagnose(remaining)
    raise AssertionError("unreachable held-out promotion command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFORM360_COHORT_BINDING_CLAIM_BOUNDARY",
    "DEFORM360_COHORT_BINDING_SCHEMA",
    "DEFORM360_COHORT_BINDING_VERSION",
    "HELDOUT_PROMOTION_DIAGNOSIS_SCHEMA",
    "HELDOUT_PROMOTION_DIAGNOSIS_VERSION",
    "HELDOUT_PROMOTION_LOCK_SCHEMA",
    "HELDOUT_PROMOTION_LOCK_VERSION",
    "HELDOUT_PROMOTION_REPORT_SCHEMA",
    "HELDOUT_PROMOTION_REPORT_VERSION",
    "HELDOUT_QUERY_RESULTS_SCHEMA",
    "HELDOUT_QUERY_RESULTS_VERSION",
    "Deform360CohortUnitV1",
    "Deform360OfficialHubCohortBindingV1",
    "HeldoutPromotionDiagnosisV1",
    "HeldoutPromotionQueryResultsV1",
    "HeldoutProviderPromotionLockV1",
    "HeldoutProviderPromotionReportV1",
    "PromotionArmV1",
    "PromotionQueryRowV1",
    "build_deform360_official_hub_cohort_binding",
    "build_promotion_evidence_card",
    "build_query_results",
    "deform360_cohort_binding_from_dict",
    "diagnose_heldout_promotion",
    "evaluate_heldout_promotion",
    "load_deform360_cohort_binding",
    "load_promotion_diagnosis",
    "load_promotion_evidence_card",
    "load_promotion_lock",
    "load_promotion_report",
    "load_query_results",
    "promotion_diagnosis_from_dict",
    "promotion_lock_from_config",
    "promotion_lock_from_dict",
    "promotion_report_from_dict",
    "query_results_from_dict",
    "query_results_from_raw",
    "validate_deform360_cohort_binding_against_selection",
    "validate_deform360_official_hub_selection",
    "validate_promotion_config_against_deform360_binding",
    "write_deform360_cohort_binding",
    "write_promotion_diagnosis",
    "write_promotion_diagnosis_markdown",
    "write_promotion_evidence_card",
    "write_promotion_lock",
    "write_promotion_report",
    "write_query_results",
]
