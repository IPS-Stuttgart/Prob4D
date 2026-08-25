#!/usr/bin/env python3
"""Authorize and summarize one exact zero-progress CUT3R smoke replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

AUTHORIZATION_SCHEMA = "prob4d.cut3r-source-comparison-smoke-retry-authorization"
AUTHORIZATION_VERSION = 1
AUTHORIZATION_DECISION = "one-exact-zero-progress-smoke-replacement-authorized"
FIRST_SUMMARY_SCHEMA = "prob4d.cut3r-source-comparison-smoke-result"
FIRST_SUMMARY_VERSION = 1
FIRST_SUMMARY_DECISION = "pre-science-technical-failure-no-retry"
HISTORICAL_CALLABLE = "src.dust3r.inference.inference"
REPAIRED_CALLABLE = "dust3r.inference.inference"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the exact authorization")
    validate.add_argument("--repository", type=Path, required=True)
    validate.add_argument("--authorization", type=Path, required=True)
    validate.add_argument("--first-summary", type=Path, required=True)

    adapt = subparsers.add_parser(
        "adapt-plan",
        help="bind the repaired callable to one newly built outcome-blind plan",
    )
    adapt.add_argument("--repository", type=Path, required=True)
    adapt.add_argument("--authorization", type=Path, required=True)
    adapt.add_argument("--first-summary", type=Path, required=True)
    adapt.add_argument("--historical-plan", type=Path, required=True)
    adapt.add_argument("--generated-plan", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser(
        "summarize",
        help="validate one retained replacement smoke and publish a bounded summary",
    )
    summarize.add_argument("--authorization", type=Path, required=True)
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.add_argument("--shard-report", type=Path, required=True)
    summarize.add_argument("--custody-receipt", type=Path, required=True)
    summarize.add_argument("--case-root", type=Path, required=True)
    summarize.add_argument("--smoke-exit-status", type=int, required=True)
    summarize.add_argument("--workflow-run-id", required=True)
    summarize.add_argument("--workflow-run-attempt", required=True)
    summarize.add_argument("--control-plane-sha", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--github-output", type=Path)
    return parser


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{name} contains non-finite JSON constant {value!r}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {name}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, *, name: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    return value


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", os.fspath(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_ancestor(repository: Path, ancestor: str, descendant: str) -> None:
    subprocess.run(
        (
            "git",
            "-C",
            os.fspath(repository),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        check=True,
        capture_output=True,
    )


def validate_authorization(
    *,
    repository: Path,
    authorization_path: Path,
    first_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one-shot authorization against the retained first smoke."""

    repository = repository.resolve(strict=True)
    authorization = _load_json(
        authorization_path.resolve(strict=True),
        name="smoke replacement authorization",
    )
    first_summary = _load_json(
        first_summary_path.resolve(strict=True),
        name="retained first smoke summary",
    )
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported smoke replacement authorization schema")
    if authorization.get("schema_version") != AUTHORIZATION_VERSION:
        raise ValueError("unsupported smoke replacement authorization version")
    if authorization.get("decision") != AUTHORIZATION_DECISION:
        raise ValueError("smoke replacement is not authorized")
    authorization_id = _require_sha256(
        authorization.get("authorization_id"),
        name="authorization_id",
    )
    unsigned = dict(authorization)
    unsigned.pop("authorization_id")
    if authorization_id != _content_id(unsigned):
        raise ValueError("smoke replacement authorization identity is invalid")

    if first_summary.get("schema") != FIRST_SUMMARY_SCHEMA:
        raise ValueError("unexpected first smoke summary schema")
    if first_summary.get("schema_version") != FIRST_SUMMARY_VERSION:
        raise ValueError("unexpected first smoke summary version")
    if first_summary.get("decision") != FIRST_SUMMARY_DECISION:
        raise ValueError("first smoke was not the registered no-retry technical failure")

    expected = cast(Mapping[str, Any], authorization["first_smoke"])
    root_mapping = {
        "artifact_id": "artifact_id",
        "plan_id": "execution_plan_id",
        "case_id_sha256": "case_id_sha256",
        "implementation_revision": "frozen_implementation_revision",
        "attempt_count": "attempt_count",
        "ordinary_success_count": "ordinary_success_count",
        "retained_technical_failure_count": "retained_technical_failure_count",
        "output_file_count_after_failure": "output_file_count_after_failure",
    }
    for expected_name, summary_name in root_mapping.items():
        if first_summary.get(summary_name) != expected.get(expected_name):
            raise ValueError(f"first smoke field changed: {summary_name}")

    first_case_id = cast(str, expected["case_id"])
    measured_case_hash = hashlib.sha256(first_case_id.encode("utf-8")).hexdigest()
    if measured_case_hash != expected["case_id_sha256"]:
        raise ValueError("authorized smoke case identity is internally inconsistent")
    failure = cast(Mapping[str, Any], expected["failure"])
    if first_summary.get("failure") != failure:
        raise ValueError("first smoke failure diagnosis changed")
    boundary = first_summary.get("information_boundary")
    if type(boundary) is not dict:
        raise ValueError("first smoke information boundary is missing")
    for field, expected_value in cast(
        Mapping[str, bool],
        expected["zero_progress"],
    ).items():
        if boundary.get(field) is not expected_value:
            raise ValueError(f"first smoke exceeded zero-progress boundary: {field}")
    if first_summary.get("retry_authorized") is not False:
        raise ValueError("first smoke summary was mutated to authorize a retry")
    if first_summary.get("retry_performed") is not False:
        raise ValueError("first smoke summary was mutated to claim a retry")

    frozen = cast(Mapping[str, Any], authorization["frozen_scientific_inputs"])
    if first_summary.get("source_freeze_id") != frozen["source_freeze_id"]:
        raise ValueError("first smoke source-freeze identity changed")
    if frozen["replacement_smoke_case_id"] != first_case_id:
        raise ValueError("replacement smoke case differs from the zero-progress case")
    control = cast(Mapping[str, Any], authorization["required_control_plane"])
    if control.get("provider_callable") != REPAIRED_CALLABLE:
        raise ValueError("authorization does not bind the repaired CUT3R callable")
    if control.get("runner_name") != "workstation2":
        raise ValueError("authorization does not bind workstation2")
    if control.get("runner_labels") != ["self-hosted", "host-workstation2"]:
        raise ValueError("authorization runner labels changed")
    limits = cast(Mapping[str, Any], authorization["authorization"])
    if limits.get("maximum_replacement_attempts") != 1:
        raise ValueError("authorization is not limited to one replacement")
    for field in (
        "full_source_shards_authorized",
        "source_truth_or_residuals_authorized",
        "target_access_authorized",
        "bayesian_phystwin_authorized",
        "causal4d_authorized",
    ):
        if limits.get(field) is not False:
            raise ValueError(f"authorization exceeds the smoke boundary: {field}")

    head = _require_revision(_git(repository, "rev-parse", "HEAD"), name="repository HEAD")
    _require_ancestor(
        repository,
        _require_revision(control["runtime_fix_commit"], name="runtime_fix_commit"),
        head,
    )
    _require_ancestor(
        repository,
        _require_revision(control["custody_gate_commit"], name="custody_gate_commit"),
        head,
    )
    return authorization, first_summary


def _verify_content_identity(value: Mapping[str, Any], *, id_field: str, name: str) -> None:
    recorded = _require_sha256(value.get(id_field), name=f"{name} {id_field}")
    unsigned = dict(value)
    unsigned.pop(id_field)
    if recorded != _content_id(unsigned):
        raise ValueError(f"{name} content identity is invalid")


def adapt_plan(
    *,
    repository: Path,
    authorization_path: Path,
    first_summary_path: Path,
    historical_plan_path: Path,
    generated_plan_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create the exact repaired plan before any replacement smoke data are opened."""

    authorization, _ = validate_authorization(
        repository=repository,
        authorization_path=authorization_path,
        first_summary_path=first_summary_path,
    )
    historical = _load_json(
        historical_plan_path.resolve(strict=True),
        name="historical source-comparison plan",
    )
    generated = _load_json(
        generated_plan_path.resolve(strict=True),
        name="newly generated source-comparison plan",
    )
    _verify_content_identity(historical, id_field="plan_id", name="historical plan")
    _verify_content_identity(generated, id_field="plan_id", name="generated plan")

    first = cast(Mapping[str, Any], authorization["first_smoke"])
    frozen = cast(Mapping[str, Any], authorization["frozen_scientific_inputs"])
    if historical["plan_id"] != first["plan_id"]:
        raise ValueError("historical plan differs from the retained first smoke")
    if historical["source_freeze_id"] != frozen["source_freeze_id"]:
        raise ValueError("historical plan source-freeze identity changed")
    if generated["source_freeze_id"] != frozen["source_freeze_id"]:
        raise ValueError("generated plan source-freeze identity changed")
    if historical["preflight_artifact_id"] != frozen["preflight_artifact_id"]:
        raise ValueError("historical preflight identity changed")
    if generated["preflight_artifact_id"] != frozen["preflight_artifact_id"]:
        raise ValueError("generated preflight identity changed")

    stable_fields = (
        "schema",
        "schema_version",
        "decision",
        "comparison_lock_id",
        "method",
        "execution",
        "cases",
        "information_boundary",
        "claim_boundary",
    )
    for field in stable_fields:
        if generated.get(field) != historical.get(field):
            raise ValueError(f"replacement plan changed frozen scientific field: {field}")

    historical_provider = dict(cast(Mapping[str, Any], historical["provider"]))
    generated_provider = dict(cast(Mapping[str, Any], generated["provider"]))
    if historical_provider.pop("callable") != HISTORICAL_CALLABLE:
        raise ValueError("historical plan no longer records its original callable")
    generated_callable = generated_provider.pop("callable")
    if generated_callable not in {HISTORICAL_CALLABLE, REPAIRED_CALLABLE}:
        raise ValueError("generated plan records an unsupported provider callable")
    if generated_provider != historical_provider:
        raise ValueError("replacement plan changed frozen provider inputs")
    generated["provider"]["callable"] = REPAIRED_CALLABLE

    if generated.get("runtime") != historical.get("runtime"):
        raise ValueError("replacement smoke runtime differs from the first smoke runtime")
    historical_implementation = cast(Mapping[str, Any], historical["implementation"])
    generated_implementation = cast(Mapping[str, Any], generated["implementation"])
    if historical_implementation.get("revision") != first["implementation_revision"]:
        raise ValueError("historical implementation revision changed")
    head = _require_revision(
        _git(repository.resolve(strict=True), "rev-parse", "HEAD"),
        name="repository HEAD",
    )
    if generated_implementation.get("revision") != head:
        raise ValueError("generated plan does not bind the exact control-plane revision")
    old_hashes = cast(Mapping[str, str], historical_implementation["source_file_sha256"])
    new_hashes = cast(Mapping[str, str], generated_implementation["source_file_sha256"])
    if set(old_hashes) != set(new_hashes):
        raise ValueError("replacement implementation file roster changed")
    changed = sorted(path for path in old_hashes if old_hashes[path] != new_hashes[path])
    if changed != ["scripts/science/run_cut3r_source_comparison.py"]:
        raise ValueError(f"unexpected replacement implementation changes: {changed!r}")

    authorization_id = cast(str, authorization["authorization_id"])
    generated["replacement_smoke_authorization_id"] = authorization_id
    generated["provenance_repair"] = {
        "first_smoke_artifact_id": first["artifact_id"],
        "first_smoke_plan_id": first["plan_id"],
        "from_provider_callable": HISTORICAL_CALLABLE,
        "to_provider_callable": REPAIRED_CALLABLE,
        "runtime_fix_commit": authorization["required_control_plane"]["runtime_fix_commit"],
        "custody_gate_commit": authorization["required_control_plane"]["custody_gate_commit"],
        "scientific_method_changed": False,
        "provider_or_checkpoint_changed": False,
        "source_or_target_roster_changed": False,
    }
    generated.pop("plan_id")
    generated["plan_id"] = _content_id(generated)
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(generated, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if output.exists() and output.read_bytes() != encoded:
        raise FileExistsError("different repaired execution plan already exists")
    output.write_bytes(encoded)
    return generated


def _write_github_output(path: Path | None, values: Mapping[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            text = str(value)
            if "\n" in text or "\r" in text:
                raise ValueError(f"GitHub output {key!r} is not a scalar")
            stream.write(f"{key}={text}\n")


def summarize(
    *,
    authorization_path: Path,
    plan_path: Path,
    output_root: Path,
    shard_report_path: Path,
    custody_receipt_path: Path,
    case_root: Path,
    smoke_exit_status: int,
    workflow_run_id: str,
    workflow_run_attempt: str,
    control_plane_sha: str,
    output_path: Path,
    github_output: Path | None,
) -> dict[str, Any]:
    """Bind one custody-validated replacement result without interpreting outcomes."""

    authorization = _load_json(
        authorization_path.resolve(strict=True),
        name="smoke replacement authorization",
    )
    plan = _load_json(plan_path.resolve(strict=True), name="repaired execution plan")
    report = _load_json(shard_report_path.resolve(strict=True), name="smoke shard report")
    receipt = _load_json(
        custody_receipt_path.resolve(strict=True),
        name="smoke custody receipt",
    )
    case = _load_json(
        (case_root.resolve(strict=True) / "case_manifest.json"),
        name="smoke case manifest",
    )
    _verify_content_identity(plan, id_field="plan_id", name="repaired plan")
    _verify_content_identity(report, id_field="artifact_id", name="smoke report")
    _verify_content_identity(case, id_field="artifact_id", name="smoke case")

    authorization_id = _require_sha256(
        authorization.get("authorization_id"),
        name="authorization_id",
    )
    if plan.get("replacement_smoke_authorization_id") != authorization_id:
        raise ValueError("repaired plan is not bound to the one-shot authorization")
    frozen = cast(Mapping[str, Any], authorization["frozen_scientific_inputs"])
    case_id = cast(str, frozen["replacement_smoke_case_id"])
    if case.get("case_id") != case_id or case_root.name != case_id:
        raise ValueError("replacement smoke case identity changed")
    plan_id = _require_sha256(plan.get("plan_id"), name="plan_id")
    if report.get("plan_id") != plan_id or case.get("plan_id") != plan_id:
        raise ValueError("replacement smoke artifacts do not share one plan identity")
    if report.get("scope") != "development-smoke":
        raise ValueError("replacement execution is not a development smoke")
    if report.get("case_count") != 1:
        raise ValueError("replacement smoke did not contain exactly one case")
    if receipt.get("decision") != "source-comparison-custody-valid":
        raise ValueError("replacement smoke custody did not validate")
    if receipt.get("plan_id") != plan_id:
        raise ValueError("custody receipt plan identity changed")
    if receipt.get("case_ids") != [case_id]:
        raise ValueError("custody receipt case identity changed")
    if receipt.get("case_artifact_ids") != [case["artifact_id"]]:
        raise ValueError("custody receipt case artifact identity changed")
    for field in (
        "source_residuals_or_truth_opened",
        "target_payloads_opened",
        "target_outcomes_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if receipt.get(field) is not False:
            raise ValueError(f"replacement smoke custody exceeded boundary: {field}")
    if receipt.get("decoded_source_frames_retained") is not False:
        raise ValueError("replacement smoke retained decoded source frames")

    status = case.get("status")
    if status == "ordinary-success":
        expected_exit = 0
        decision = "ordinary-success-development-smoke"
        if report.get("ordinary_success_count") != 1:
            raise ValueError("successful smoke report has the wrong success count")
        if report.get("retained_technical_failure_count") != 0:
            raise ValueError("successful smoke report retains a technical failure")
    elif status == "retained-technical-failure":
        expected_exit = 3
        decision = "retained-technical-failure-no-further-retry"
        if report.get("ordinary_success_count") != 0:
            raise ValueError("failed smoke report has the wrong success count")
        if report.get("retained_technical_failure_count") != 1:
            raise ValueError("failed smoke report has the wrong failure count")
    else:
        raise ValueError("replacement smoke has an unsupported status")
    if smoke_exit_status != expected_exit:
        raise ValueError("replacement smoke process status differs from retained evidence")

    summary: dict[str, Any] = {
        "schema": "prob4d.cut3r-source-comparison-smoke-retry-result",
        "schema_version": 1,
        "decision": decision,
        "authorization_id": authorization_id,
        "control_plane_sha": _require_revision(
            control_plane_sha,
            name="control_plane_sha",
        ),
        "workflow_run_id": str(workflow_run_id),
        "workflow_run_attempt": str(workflow_run_attempt),
        "plan_id": plan_id,
        "source_freeze_id": frozen["source_freeze_id"],
        "case_id": case_id,
        "case_artifact_id": case["artifact_id"],
        "shard_report_artifact_id": report["artifact_id"],
        "custody_receipt_id": receipt["receipt_id"],
        "smoke_exit_status": smoke_exit_status,
        "ordinary_success_count": report["ordinary_success_count"],
        "retained_technical_failure_count": report[
            "retained_technical_failure_count"
        ],
        "source_rgb_frames_decoded": case["source_rgb_frames_decoded"],
        "cut3r_inference_executed": case["cut3r_inference_executed"],
        "source_predictions_written": case["source_predictions_written"],
        "decoded_source_frames_retained": False,
        "source_residuals_or_truth_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "full_source_shards_authorized": status == "ordinary-success",
    }
    summary["artifact_id"] = _content_id(summary)
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if output.exists() and output.read_bytes() != encoded:
        raise FileExistsError("different replacement summary already exists")
    output.write_bytes(encoded)
    _write_github_output(
        github_output,
        {
            "decision": decision,
            "plan_id": plan_id,
            "case_artifact_id": case["artifact_id"],
            "custody_receipt_id": receipt["receipt_id"],
            "summary_artifact_id": summary["artifact_id"],
            "smoke_exit_status": smoke_exit_status,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        authorization, _ = validate_authorization(
            repository=args.repository,
            authorization_path=args.authorization,
            first_summary_path=args.first_summary,
        )
        print(
            json.dumps(
                {
                    "authorization_id": authorization["authorization_id"],
                    "decision": authorization["decision"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "adapt-plan":
        plan = adapt_plan(
            repository=args.repository,
            authorization_path=args.authorization,
            first_summary_path=args.first_summary,
            historical_plan_path=args.historical_plan,
            generated_plan_path=args.generated_plan,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "decision": plan["decision"],
                    "plan_id": plan["plan_id"],
                    "provider_callable": plan["provider"]["callable"],
                },
                sort_keys=True,
            )
        )
        return 0
    summary = summarize(
        authorization_path=args.authorization,
        plan_path=args.plan,
        output_root=args.output_root,
        shard_report_path=args.shard_report,
        custody_receipt_path=args.custody_receipt,
        case_root=args.case_root,
        smoke_exit_status=args.smoke_exit_status,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
        control_plane_sha=args.control_plane_sha,
        output_path=args.output,
        github_output=args.github_output,
    )
    print(
        json.dumps(
            {
                "artifact_id": summary["artifact_id"],
                "decision": summary["decision"],
                "plan_id": summary["plan_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
