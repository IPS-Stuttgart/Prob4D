#!/usr/bin/env python3
"""Freeze shared-dependence strength on the already-open DOT R01--R03 source cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.dependence_tempering import temper_shared_dependence
from prob4d.dot_rope_cut3r_study import content_id

PROTOCOL_SCHEMA = "prob4d.dot-rope-dependence-tempering-source-protocol"
REQUEST_SCHEMA = "prob4d.dot-rope-dependence-tempering-source-request"
RESULT_SCHEMA = "prob4d.dot-rope-dependence-tempering-source-calibration"
SCHEMA_VERSION = 1
SOURCE_SEQUENCES = ["R01", "R02", "R03"]
RESERVED_SEQUENCES = "R04-R70"
PROVIDER_RUN_ID = 33329701704
PROVIDER_ARTIFACT_NAME = "dot-rope-cut3r-sealed-provider-33329701704-1"
PROVIDER_BUNDLE_ID = "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7"
POOLED_REQUEST_ID = "909aefd1f26f16bfff492ae36b24d6812291edeed907971f67672eacb60819cb"
SELECTION_RULE = "minimize-worst-sequence-normalized-nll-then-mean-then-smallest-alpha"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--protocol-git-blob-sha", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--request", type=Path, required=True)
    calibrate.add_argument("--protocol", type=Path, required=True)
    calibrate.add_argument("--protocol-git-blob-sha", required=True)
    calibrate.add_argument("--dataset-root", type=Path, required=True)
    calibrate.add_argument("--provider-bundle", type=Path, required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--repository-revision", required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dependence-tempering source protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _hex(protocol_id, name="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("dependence-tempering protocol identity mismatch")
    if protocol.get("source_sequences") != SOURCE_SEQUENCES:
        raise ValueError("source sequence roster changed")
    if protocol.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("reserved sequence boundary changed")
    if protocol.get("provider_run_id") != PROVIDER_RUN_ID:
        raise ValueError("provider run changed")
    if protocol.get("provider_artifact_name") != PROVIDER_ARTIFACT_NAME:
        raise ValueError("provider artifact changed")
    if protocol.get("provider_bundle_id") != PROVIDER_BUNDLE_ID:
        raise ValueError("provider bundle changed")
    if protocol.get("pooled_evaluation_request_id") != POOLED_REQUEST_ID:
        raise ValueError("pooled evaluation request changed")
    if protocol.get("selection_rule") != SELECTION_RULE:
        raise ValueError("source selection rule changed")
    if protocol.get("means_held_fixed") is not True:
        raise ValueError("provider means must remain fixed")
    if protocol.get("confirmation_payloads_opened") is not False:
        raise ValueError("confirmation payloads must remain unopened")
    grid = protocol.get("alpha_grid")
    if not isinstance(grid, list) or len(grid) < 2:
        raise ValueError("alpha_grid must contain at least two strengths")
    strengths = [float(value) for value in grid]
    if strengths[0] != 0.0 or strengths[-1] != 1.0:
        raise ValueError("alpha_grid must retain both endpoint models")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in strengths):
        raise ValueError("alpha_grid strengths must lie in [0, 1]")
    if any(second <= first for first, second in zip(strengths, strengths[1:], strict=True)):
        raise ValueError("alpha_grid must be strictly increasing")
    return protocol


def validate_request(
    request_path: Path,
    protocol_path: Path,
    protocol_git_blob_sha: str,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    request = _read_json(request_path)
    expected = {
        "claim_boundary",
        "confirmation_payloads_opened",
        "protocol_git_blob_sha",
        "protocol_path",
        "provider_artifact_name",
        "provider_run_id",
        "request_id",
        "reserved_sequences",
        "schema",
        "schema_version",
        "source_calibration_authorized",
        "source_sequences",
    }
    if set(request) != expected:
        raise ValueError("dependence-tempering request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported dependence-tempering request schema")
    _hex(protocol_git_blob_sha, name="protocol Git blob", length=40)
    if request["protocol_path"] != protocol_path.as_posix():
        raise ValueError("request protocol path changed")
    if request["protocol_git_blob_sha"] != protocol_git_blob_sha:
        raise ValueError("request does not bind the reviewed protocol blob")
    if request["source_sequences"] != SOURCE_SEQUENCES:
        raise ValueError("request source roster changed")
    if request["reserved_sequences"] != RESERVED_SEQUENCES:
        raise ValueError("request reserved boundary changed")
    if request["provider_run_id"] != protocol["provider_run_id"]:
        raise ValueError("request provider run changed")
    if request["provider_artifact_name"] != protocol["provider_artifact_name"]:
        raise ValueError("request provider artifact changed")
    if request["source_calibration_authorized"] is not True:
        raise ValueError("source calibration must be explicitly authorized")
    if request["confirmation_payloads_opened"] is not False:
        raise ValueError("confirmation payload access exceeds the source boundary")
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("dependence-tempering request identity mismatch")
    return {
        "request_id": request_id,
        "protocol_id": protocol["protocol_id"],
        "provider_run_id": protocol["provider_run_id"],
        "provider_artifact_name": protocol["provider_artifact_name"],
    }


def _load_pooled_evaluator() -> Any:
    path = Path(__file__).with_name("evaluate_dot_rope_cut3r_pooled.py")
    spec = importlib.util.spec_from_file_location("dot_rope_pooled_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pooled DOT evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alpha_method_name(strength: float) -> str:
    value = int(round(1000.0 * float(strength)))
    if not 0 <= value <= 1000:
        raise ValueError("alpha strength is outside [0, 1]")
    return f"dependence_alpha_{value:04d}"


def select_strength(
    method_rows: list[Mapping[str, Any]],
    strengths: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    for strength in strengths:
        method = alpha_method_name(strength)
        rows = [row for row in method_rows if row.get("method") == method]
        by_sequence = {str(row["sequence"]): float(row["normalized_nll_per_dimension"]) for row in rows}
        if set(by_sequence) != set(SOURCE_SEQUENCES):
            raise ValueError(f"source score roster is incomplete for {method}")
        values = [by_sequence[sequence] for sequence in SOURCE_SEQUENCES]
        table.append(
            {
                "alpha": float(strength),
                "method": method,
                "sequence_nll_per_dimension": by_sequence,
                "worst_sequence_nll_per_dimension": float(max(values)),
                "mean_sequence_nll_per_dimension": float(np.mean(values)),
            }
        )
    winner = min(
        table,
        key=lambda row: (
            row["worst_sequence_nll_per_dimension"],
            row["mean_sequence_nll_per_dimension"],
            row["alpha"],
        ),
    )
    return float(winner["alpha"]), table


def calibrate(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    identity = validate_request(args.request, args.protocol, args.protocol_git_blob_sha)
    revision = _hex(args.repository_revision, name="repository revision", length=40)
    pooled_request = Path(str(protocol["pooled_evaluation_request_path"]))
    pooled = _load_pooled_evaluator()
    pooled_value = pooled.validate_request(pooled_request)
    if pooled_value["request_id"] != protocol["pooled_evaluation_request_id"]:
        raise ValueError("bound pooled evaluation request identity changed")

    import prob4d.dot_rope_cut3r_study as study

    original_closures = study.covariance_closures
    strengths = [float(value) for value in protocol["alpha_grid"]]

    def augmented_closures(*closure_args, **closure_kwargs):
        closures = original_closures(*closure_args, **closure_kwargs)
        marginal = closures["pointwise_quadratic"]
        shared = closures["shared_quadratic_curvature"]
        for strength in strengths:
            closures[alpha_method_name(strength)] = temper_shared_dependence(
                marginal,
                shared,
                strength,
            )
        return closures

    study.covariance_closures = augmented_closures
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "augmented-source-evaluation"
    try:
        status = pooled.evaluate(
            argparse.Namespace(
                request=pooled_request,
                dataset_root=args.dataset_root,
                provider_bundle=args.provider_bundle,
                output_dir=raw,
                repository_revision=revision,
            )
        )
    finally:
        study.covariance_closures = original_closures
    if int(status) != 0:
        raise RuntimeError(f"augmented source evaluator returned status {status}")

    augmented_result = _read_json(raw / "result.json")
    selected, table = select_strength(augmented_result["method_rows"], strengths)

    endpoint_checks: dict[str, float] = {}
    for endpoint, reference in ((0.0, "pointwise_quadratic"), (1.0, "shared_quadratic_curvature")):
        method = alpha_method_name(endpoint)
        endpoint_rows = {
            str(row["sequence"]): float(row["normalized_nll_per_dimension"])
            for row in augmented_result["method_rows"]
            if row["method"] == method
        }
        reference_rows = {
            str(row["sequence"]): float(row["normalized_nll_per_dimension"])
            for row in augmented_result["method_rows"]
            if row["method"] == reference
        }
        difference = max(abs(endpoint_rows[key] - reference_rows[key]) for key in SOURCE_SEQUENCES)
        endpoint_checks[method] = float(difference)
        if difference > 1.0e-10:
            raise RuntimeError("dependence-tempering endpoint did not reproduce its registered model")

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": identity["request_id"],
        "protocol_id": identity["protocol_id"],
        "repository_revision": revision,
        "provider_run_id": protocol["provider_run_id"],
        "provider_bundle_id": protocol["provider_bundle_id"],
        "pooled_evaluation_request_id": protocol["pooled_evaluation_request_id"],
        "source_sequences": SOURCE_SEQUENCES,
        "reserved_sequences": RESERVED_SEQUENCES,
        "selection_rule": SELECTION_RULE,
        "selected_alpha": selected,
        "selected_method": alpha_method_name(selected),
        "alpha_table": table,
        "endpoint_nll_max_abs_difference": endpoint_checks,
        "means_held_fixed": True,
        "marginal_variances_preserved_by_construction": True,
        "confirmation_candidate": protocol["confirmation_candidate"],
        "confirmation_reserved_after_candidate": protocol[
            "confirmation_reserved_after_candidate"
        ],
        "confirmation_payloads_opened": False,
        "decision": "source-dependence-strength-frozen",
        "claim_boundary": protocol["claim_boundary"],
    }
    result["calibration_id"] = content_id(result)
    _write_json(output / "calibration.json", result)

    lines = [
        "# DOT rope shared-dependence source calibration",
        "",
        f"Calibration ID: `{result['calibration_id']}`",
        "",
        f"Selected alpha: **{selected:.2f}**",
        "",
        "| alpha | worst source NLL/dim | mean source NLL/dim |",
        "|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['alpha']:.2f} | {row['worst_sequence_nll_per_dimension']:.6f} | "
            f"{row['mean_sequence_nll_per_dimension']:.6f} |"
        )
    lines.extend(
        [
            "",
            "R04-R70 remained unopened. The selected alpha is source-fitted only and must be "
            "frozen before any separately authorized confirmation access.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "calibration_id": result["calibration_id"],
                "selected_alpha": selected,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-request":
        print(
            json.dumps(
                validate_request(args.request, args.protocol, args.protocol_git_blob_sha),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "calibrate":
        return calibrate(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
