"""Content-addressed evidence cards for held-out promotion reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROMOTION_EVIDENCE_CARD_SCHEMA = "prob4d.heldout-provider-evidence-card"
PROMOTION_EVIDENCE_CARD_VERSION = 1
_LOCK_SCHEMA = "prob4d.heldout-provider-promotion-lock"
_REPORT_SCHEMA = "prob4d.heldout-provider-promotion-report"
_FIELDS = {
    "schema_name", "schema_version", "evidence_card_id", "experiment_id",
    "promotion_lock_id", "promotion_report_id", "overall_passed", "status",
    "statistical_unit", "repositories", "cohort", "comparison_arms",
    "frozen_inputs", "provider_gate", "guarded_query", "claim_boundary",
    "explicit_non_claims",
}


def _map(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _revision(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) not in {40, 64} or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase Git revision")
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _ids(value: object, name: str) -> list[str]:
    if type(value) is not list or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{name} must be a list of nonempty strings")
    return list(value)


def _arms(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = lock.get("arms")
    if type(values) is not list or not values:
        raise ValueError("promotion lock arms must be a nonempty list")
    result = []
    for index, raw in enumerate(values):
        arm = _map(raw, f"arms[{index}]")
        provider = arm.get("provider_method_id")
        result.append({
            "arm_id": _text(arm.get("arm_id"), f"arms[{index}].arm_id"),
            "role": _text(arm.get("role"), f"arms[{index}].role"),
            "provider_method_id": (
                None if provider is None else _text(provider, "provider_method_id")
            ),
            "query_method_id": _text(arm.get("query_method_id"), "query_method_id"),
            "sensor_assisted": _boolean(arm.get("sensor_assisted"), "sensor_assisted"),
        })
    if [row["arm_id"] for row in result] != sorted(row["arm_id"] for row in result):
        raise ValueError("promotion lock arms must remain sorted by arm_id")
    return result


def build_promotion_evidence_card(
    promotion_lock: Mapping[str, Any],
    promotion_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a compact paper-facing summary from a validated lock and report."""

    lock = _map(promotion_lock, "promotion lock")
    report = _map(promotion_report, "promotion report")
    if (lock.get("schema_name"), lock.get("schema_version")) != (_LOCK_SCHEMA, 1):
        raise ValueError("unsupported held-out promotion lock schema")
    if (report.get("schema_name"), report.get("schema_version")) != (_REPORT_SCHEMA, 1):
        raise ValueError("unsupported held-out promotion report schema")
    lock_id = _digest(lock.get("promotion_lock_id"), "promotion_lock_id")
    if report.get("promotion_lock_id") != lock_id:
        raise ValueError("promotion report references a different promotion lock")
    audit = _map(report.get("provider_audit"), "provider_audit")
    provider = _map(report.get("provider_decision"), "provider_decision")
    query = _map(report.get("query_decision"), "query_decision")
    aggregates = _map(report.get("query_aggregate"), "query_aggregate")
    primary_id = _text(lock.get("primary_query_arm_id"), "primary_query_arm_id")
    primary = _map(aggregates.get(primary_id), "primary query aggregate")
    paired = _map(query.get("paired_bootstrap"), "paired_bootstrap")
    development = _ids(lock.get("development_group_ids"), "development_group_ids")
    calibration = _ids(lock.get("calibration_group_ids"), "calibration_group_ids")
    target = _ids(lock.get("target_group_ids"), "target_group_ids")
    metadata = _map(lock.get("metadata", {}), "metadata")
    unit = metadata.get("statistical_unit", metadata.get("split_semantics"))
    if not isinstance(unit, str) or not unit:
        unit = "complete physical object or acquisition session"
    passed = _boolean(report.get("overall_passed"), "overall_passed")
    descriptor: dict[str, Any] = {
        "schema_name": PROMOTION_EVIDENCE_CARD_SCHEMA,
        "schema_version": PROMOTION_EVIDENCE_CARD_VERSION,
        "experiment_id": _text(lock.get("experiment_id"), "experiment_id"),
        "promotion_lock_id": lock_id,
        "promotion_report_id": _digest(report.get("report_id"), "report_id"),
        "overall_passed": passed,
        "status": "PASS" if passed else "FAIL",
        "statistical_unit": unit,
        "repositories": {
            "prob4d": {
                "repository": _text(lock.get("source_repository"), "source_repository"),
                "revision": _revision(lock.get("source_revision"), "source_revision"),
            },
            "bayesian_phystwin": {
                "repository": _text(
                    lock.get("bayesian_phystwin_repository"), "bayesian_phystwin_repository"
                ),
                "revision": _revision(
                    lock.get("bayesian_phystwin_revision"), "bayesian_phystwin_revision"
                ),
            },
            "motioncrafter": {
                "revision": _revision(
                    lock.get("motioncrafter_revision"), "motioncrafter_revision"
                ),
                "model_set_id": _digest(lock.get("model_set_id"), "model_set_id"),
            },
        },
        "cohort": {
            "development_group_count": len(development),
            "calibration_group_count": len(calibration),
            "target_group_count": len(target),
            "target_group_ids": target,
        },
        "comparison_arms": _arms(lock),
        "frozen_inputs": {
            "prediction_run_spec_id": _digest(
                lock.get("prediction_run_spec_id"), "prediction_run_spec_id"
            ),
            "provider_evaluation_manifest_sha256": _digest(
                lock.get("provider_evaluation_manifest_sha256"),
                "provider_evaluation_manifest_sha256",
            ),
            "frozen_artifact_ids": dict(_map(lock.get("frozen_artifact_ids"), "artifacts")),
            "provider_report_sha256": _digest(
                report.get("provider_report_sha256"), "provider_report_sha256"
            ),
            "query_results_id": _digest(report.get("query_results_id"), "query_results_id"),
        },
        "provider_gate": {
            "passed": _boolean(provider.get("overall_passed"), "provider passed"),
            "reference_method": _text(audit.get("reference_method"), "reference_method"),
            "case_count": _integer(audit.get("case_count"), "case_count"),
            "group_count": _integer(audit.get("group_count"), "group_count"),
            "decision_policy_id": audit.get("decision_policy_id"),
        },
        "guarded_query": {
            "passed": _boolean(query.get("overall_passed"), "query passed"),
            "primary_arm_id": primary_id,
            "physical_fallback_arm_id": _text(
                query.get("physical_fallback_arm_id"), "physical_fallback_arm_id"
            ),
            "paired_candidate_minus_fallback_mm": {
                "mean": _real(paired.get("mean"), "paired mean"),
                "ci95_lower": _real(paired.get("ci95_lower"), "paired lower"),
                "ci95_upper": _real(paired.get("ci95_upper"), "paired upper"),
                "group_count": _integer(paired.get("group_count"), "paired groups"),
                "semantics": _text(paired.get("semantics"), "paired semantics"),
            },
            "mean_query_rmse_mm": _real(primary.get("mean_query_rmse_mm"), "mean RMSE"),
            "accepted_update_count": _integer(
                primary.get("accepted_update_count"), "accepted updates"
            ),
            "rejected_update_count": _integer(
                primary.get("rejected_update_count"), "rejected updates"
            ),
            "harmful_accepted_update_count": _integer(
                primary.get("harmful_accepted_update_count"), "harmful updates"
            ),
            "technical_failure_count": _integer(
                primary.get("technical_failure_count"), "technical failures"
            ),
            "worst_group_regression_mm": _real(
                primary.get("worst_group_regression_mm"), "worst regression"
            ),
            "mean_accepted_coverage": (
                None if primary.get("mean_accepted_coverage") is None
                else _real(primary.get("mean_accepted_coverage"), "accepted coverage")
            ),
            "mean_accepted_width_mm": (
                None if primary.get("mean_accepted_width_mm") is None
                else _real(primary.get("mean_accepted_width_mm"), "accepted width")
            ),
            "exact_fallback_failure_count": _integer(
                query.get("exact_fallback_failure_count"), "fallback failures"
            ),
        },
        "claim_boundary": _text(report.get("claim_boundary"), "claim_boundary"),
        "explicit_non_claims": [
            "No Causal4D intervention-benefit claim.",
            "No general state-of-the-art claim.",
            "No extrapolation beyond frozen objects, sessions, and source identities.",
        ],
    }
    return {**descriptor, "evidence_card_id": _identity(descriptor)}


def promotion_evidence_card_from_dict(value: object) -> dict[str, Any]:
    card = _map(value, "promotion evidence card")
    if set(card) != _FIELDS:
        raise ValueError("promotion evidence card fields changed")
    if (card.get("schema_name"), card.get("schema_version")) != (
        PROMOTION_EVIDENCE_CARD_SCHEMA, PROMOTION_EVIDENCE_CARD_VERSION
    ):
        raise ValueError("unsupported promotion evidence card schema")
    supplied = _digest(card.get("evidence_card_id"), "evidence_card_id")
    descriptor = {key: item for key, item in card.items() if key != "evidence_card_id"}
    if supplied != _identity(descriptor):
        raise ValueError("promotion evidence card ID mismatch")
    return dict(card)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_promotion_evidence_card(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_constant,
    )
    return promotion_evidence_card_from_dict(value)


def render_promotion_evidence_markdown(card: Mapping[str, Any]) -> str:
    value = promotion_evidence_card_from_dict(card)
    repositories = _map(value["repositories"], "repositories")
    prob4d = _map(repositories["prob4d"], "prob4d")
    bpt = _map(repositories["bayesian_phystwin"], "bayesian_phystwin")
    query = _map(value["guarded_query"], "guarded_query")
    paired = _map(query["paired_candidate_minus_fallback_mm"], "paired effect")
    return "\n".join([
        "# Held-out Prob4D promotion evidence card", "",
        f"Evidence card: `{value['evidence_card_id']}`.",
        f"Promotion report: `{value['promotion_report_id']}`.",
        f"Overall decision: **{value['status']}**.", "",
        f"- Prob4D: `{prob4d['repository']}@{prob4d['revision']}`",
        f"- BayesianPhysTwin: `{bpt['repository']}@{bpt['revision']}`", "",
        "| Guarded-query result | Value |", "| --- | ---: |",
        f"| Primary arm | `{query['primary_arm_id']}` |",
        f"| Mean candidate-minus-fallback RMSE | {float(paired['mean']):.6g} mm |",
        f"| Harmful accepted updates | {query['harmful_accepted_update_count']} |",
        f"| Exact fallback failures | {query['exact_fallback_failure_count']} |", "",
        str(value["claim_boundary"]), "",
    ])


def _write(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_promotion_evidence_card(
    card: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    value = promotion_evidence_card_from_dict(card)
    json_destination, markdown_destination = Path(json_path), Path(markdown_path)
    if json_destination == markdown_destination:
        raise ValueError("JSON and Markdown evidence-card paths must differ")
    if json_destination.exists():
        raise FileExistsError(json_destination)
    if markdown_destination.exists():
        raise FileExistsError(markdown_destination)
    _write(json_destination, json.dumps(value, sort_keys=True, indent=2) + "\n")
    try:
        _write(markdown_destination, render_promotion_evidence_markdown(value))
    except Exception:
        json_destination.unlink(missing_ok=True)
        markdown_destination.unlink(missing_ok=True)
        raise


__all__ = [
    "PROMOTION_EVIDENCE_CARD_SCHEMA", "PROMOTION_EVIDENCE_CARD_VERSION",
    "build_promotion_evidence_card", "load_promotion_evidence_card",
    "promotion_evidence_card_from_dict", "render_promotion_evidence_markdown",
    "write_promotion_evidence_card",
]
