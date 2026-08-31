#!/usr/bin/env python3
"""Finalize metadata for one exact DOT pooled-evaluation artifact.

The source evaluation already computed all numerical results. This utility only
repairs the outer request/provenance binding, verifies that the registered
scientific payload is byte-semantically unchanged, and emits a new
content-addressed evidence bundle. It opens no dataset and recomputes no score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prob4d.dot_rope_cut3r_study import content_id

REQUEST_SCHEMA = "prob4d.dot-rope-pooled-result-finalization-request"
REQUEST_SCHEMA_VERSION = 1
SOURCE_RESULT_SCHEMA = "prob4d.dot-rope-cut3r-pooled-evaluation"
FINAL_RESULT_SCHEMA = "prob4d.dot-rope-cut3r-pooled-evaluation-finalized"
FINAL_RESULT_SCHEMA_VERSION = 1
SOURCE_DECISION = "complete-source-evaluation-pooled-marker-support"
SCIENTIFIC_KEYS = (
    "decision",
    "protocol_id",
    "provider_bundle_id",
    "runtime_artifact_id",
    "marker_support_id",
    "information_boundary",
    "marker_sampling",
    "marker_support_audit",
    "aggregate_methods",
    "method_rows",
    "sequences",
    "opened_marker_members",
)
_HEX = re.compile(r"[0-9a-f]+")
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--request", type=Path, required=True)
    finalize.add_argument("--source-root", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--repository-revision", required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_request(path: Path) -> dict[str, Any]:
    request = _read_json(path)
    expected = {
        "claim_boundary",
        "execution_nonce",
        "marker_support_audit_id",
        "no_dataset_access",
        "no_scores_recomputed",
        "pooled_request_id",
        "provider_artifact_name",
        "provider_bundle_id",
        "provider_request_id",
        "provider_run_id",
        "request_id",
        "schema",
        "schema_version",
        "scientific_payload_id",
        "source_artifact_digest",
        "source_artifact_id",
        "source_artifact_name",
        "source_head_sha",
        "source_run_id",
        "source_unfinalized_evaluation_id",
    }
    if set(request) != expected:
        raise ValueError("finalization request fields changed")
    if (
        request["schema"] != REQUEST_SCHEMA
        or request["schema_version"] != REQUEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported finalization request schema")
    if request["no_dataset_access"] is not True:
        raise ValueError("finalization must not access the dataset")
    if request["no_scores_recomputed"] is not True:
        raise ValueError("finalization must not recompute scores")
    if not isinstance(request["execution_nonce"], str) or not request["execution_nonce"]:
        raise ValueError("execution_nonce must be a non-empty string")
    if not isinstance(request["claim_boundary"], str) or not request["claim_boundary"]:
        raise ValueError("claim_boundary must be a non-empty string")
    for name in (
        "provider_run_id",
        "source_artifact_id",
        "source_run_id",
    ):
        if not isinstance(request[name], int) or request[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, length in (
        ("marker_support_audit_id", 64),
        ("pooled_request_id", 64),
        ("provider_bundle_id", 64),
        ("provider_request_id", 64),
        ("scientific_payload_id", 64),
        ("source_head_sha", 40),
        ("source_unfinalized_evaluation_id", 64),
    ):
        _hex(request[name], name=name, length=length)
    if (
        not isinstance(request["source_artifact_digest"], str)
        or _SHA256_DIGEST.fullmatch(request["source_artifact_digest"]) is None
    ):
        raise ValueError("source_artifact_digest must be a sha256 GitHub digest")
    for name in ("provider_artifact_name", "source_artifact_name"):
        if not isinstance(request[name], str) or not request[name]:
            raise ValueError(f"{name} must be a non-empty string")
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("finalization request identity mismatch")
    return request


def scientific_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in SCIENTIFIC_KEYS if key not in result]
    if missing:
        raise ValueError(f"source result lacks scientific fields: {missing}")
    return {key: result[key] for key in SCIENTIFIC_KEYS}


def _validate_source_bundle(
    request: Mapping[str, Any],
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = source_root / "result" / "result.json"
    support_path = source_root / "result" / "marker-support.json"
    result = _read_json(result_path)
    support = _read_json(support_path)
    if result.get("schema") != SOURCE_RESULT_SCHEMA or result.get("schema_version") != 1:
        raise ValueError("source result schema changed")
    if result.get("decision") != SOURCE_DECISION:
        raise ValueError("source result is not a complete pooled evaluation")
    source_evaluation_id = result.get("evaluation_id")
    if source_evaluation_id != request["source_unfinalized_evaluation_id"]:
        raise ValueError("source unfinalized evaluation identity changed")
    unhashed_result = dict(result)
    unhashed_result.pop("evaluation_id", None)
    if content_id(unhashed_result) != source_evaluation_id:
        raise ValueError("source result content identity is invalid")
    if result.get("request_id") != request["provider_request_id"]:
        raise ValueError("source result no longer exhibits the registered request-ID defect")
    if result.get("provider_bundle_id") != request["provider_bundle_id"]:
        raise ValueError("source provider bundle identity changed")
    if result.get("evaluator_prob4d_revision") != request["source_head_sha"]:
        raise ValueError("source evaluator revision changed")
    if support.get("request_id") != request["pooled_request_id"]:
        raise ValueError("marker-support evidence is not bound to the pooled request")
    audit_id = support.get("marker_support_audit", {}).get("audit_id")
    if audit_id != request["marker_support_audit_id"]:
        raise ValueError("marker-support audit identity changed")
    support_id = support.get("support_id")
    unhashed_support = dict(support)
    unhashed_support.pop("support_id", None)
    if content_id(unhashed_support) != support_id:
        raise ValueError("marker-support content identity is invalid")
    payload_id = content_id(scientific_payload(result))
    if payload_id != request["scientific_payload_id"]:
        raise ValueError("scientific payload identity changed")
    required_files = (
        source_root / "official-archive-metadata.json",
        source_root / "result" / "method-summary.csv",
        source_root / "result" / "sequence-methods.csv",
        source_root / "result" / "summary.md",
    )
    for path in required_files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required source evidence is unavailable: {path.name}")
    return result, support


def _write_summary(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# DOT rope marker-free CUT3R source result",
        "",
        f"Finalized evaluation ID: `{result['evaluation_id']}`",
        f"Pooled request ID: `{result['request_id']}`",
        f"Provider request ID: `{result['provider_request_id']}`",
        f"Scientific payload ID: `{result['scientific_payload_id']}`",
        "",
        "## Uncertainty methods",
        "",
        "| Method | Mean normalized NLL/dim | 95% covered | Mean SD/span |",
        "|---|---:|---:|---:|",
    ]
    for row in result["aggregate_methods"]:
        lines.append(
            f"| {row['method']} | {row['mean_normalized_nll_per_dimension']:.6f} | "
            f"{row['covered_95_count']}/{row['sequence_count']} | "
            f"{row['mean_predictive_sd_fraction_of_span']:.6f} |"
        )
    lines.extend(["", "## Reconstruction/stitching", ""])
    for sequence in result["sequences"]:
        metrics = sequence["point_metrics"]
        lines.append(
            f"- **{sequence['sequence']}**: continuous="
            f"{metrics['continuous_rmse_fraction_of_span']:.6f}, identity stitch="
            f"{metrics['identity_stitch_rmse_fraction_of_span']:.6f}, estimated Sim(3) stitch="
            f"{metrics['estimated_stitch_rmse_fraction_of_span']:.6f}, oracle window="
            f"{metrics['oracle_window_rmse_fraction_of_span']:.6f} (all / span)."
        )
    lines.extend(
        [
            "",
            "## Provenance finalization",
            "",
            f"- Source workflow run: `{result['source_evaluation']['run_id']}`.",
            f"- Source artifact ID: `{result['source_evaluation']['artifact_id']}`.",
            f"- Unfinalized evaluation ID: `{result['unfinalized_evaluation_id']}`.",
            (
                "- No provider output, score, covariance, method ranking, or "
                "dataset payload was changed."
            ),
            "",
            (
                "Source-development evidence only. R04-R70 remained unopened; "
                "no BayesianPhysTwin or Causal4D outcome was executed."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def finalize(
    request: Mapping[str, Any],
    source_root: Path,
    output_dir: Path,
    repository_revision: str,
) -> dict[str, Any]:
    revision = _hex(repository_revision, name="repository_revision", length=40)
    source_result, support = _validate_source_bundle(request, source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")

    corrected = dict(source_result)
    source_schema = corrected.pop("schema")
    source_schema_version = corrected.pop("schema_version")
    unfinalized_evaluation_id = corrected.pop("evaluation_id")
    provider_request_id = corrected.pop("request_id")
    corrected["schema"] = FINAL_RESULT_SCHEMA
    corrected["schema_version"] = FINAL_RESULT_SCHEMA_VERSION
    corrected["source_runtime_schema"] = source_schema
    corrected["source_runtime_schema_version"] = source_schema_version
    corrected["unfinalized_evaluation_id"] = unfinalized_evaluation_id
    corrected["request_id"] = request["pooled_request_id"]
    corrected["provider_request_id"] = provider_request_id
    corrected["provider_run_id"] = request["provider_run_id"]
    corrected["provider_artifact_name"] = request["provider_artifact_name"]
    corrected["scientific_payload_id"] = request["scientific_payload_id"]
    corrected["source_evaluation"] = {
        "run_id": request["source_run_id"],
        "head_sha": request["source_head_sha"],
        "artifact_id": request["source_artifact_id"],
        "artifact_name": request["source_artifact_name"],
        "artifact_digest": request["source_artifact_digest"],
    }
    corrected["provenance_finalization"] = {
        "repository_revision": revision,
        "request_id": request["request_id"],
        "no_dataset_access": True,
        "no_scores_recomputed": True,
        "scientific_payload_unchanged": True,
        "reason": (
            "Bind the pooled request as the evaluation request, preserve the sealed "
            "provider request separately, and replace the stale summary identity."
        ),
    }
    corrected["finalization_claim_boundary"] = request["claim_boundary"]
    if content_id(scientific_payload(corrected)) != request["scientific_payload_id"]:
        raise AssertionError("scientific payload changed during metadata finalization")
    corrected["evaluation_id"] = content_id(corrected)

    _write_json(output_dir / "result.json", corrected)
    _write_json(output_dir / "marker-support.json", support)
    for name in ("method-summary.csv", "sequence-methods.csv"):
        shutil.copyfile(source_root / "result" / name, output_dir / name)
    shutil.copyfile(
        source_root / "official-archive-metadata.json",
        output_dir / "official-archive-metadata.json",
    )
    _write_summary(corrected, output_dir / "summary.md")

    file_rows = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name):
        if path.name == "finalization.json":
            continue
        file_rows.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    receipt: dict[str, Any] = {
        "schema": "prob4d.dot-rope-pooled-result-finalization",
        "schema_version": 1,
        "request_id": request["request_id"],
        "source_unfinalized_evaluation_id": unfinalized_evaluation_id,
        "finalized_evaluation_id": corrected["evaluation_id"],
        "scientific_payload_id": request["scientific_payload_id"],
        "scientific_payload_unchanged": True,
        "no_dataset_access": True,
        "no_scores_recomputed": True,
        "files": file_rows,
    }
    receipt["finalization_id"] = content_id(receipt)
    _write_json(output_dir / "finalization.json", receipt)
    return receipt


def main() -> int:
    args = _parser().parse_args()
    request = validate_request(args.request)
    if args.command == "validate-request":
        print(
            json.dumps(
                {
                    "request_id": request["request_id"],
                    "source_run_id": request["source_run_id"],
                    "source_artifact_id": request["source_artifact_id"],
                    "source_artifact_name": request["source_artifact_name"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "finalize":
        receipt = finalize(
            request,
            args.source_root,
            args.output_dir,
            args.repository_revision,
        )
        print(
            json.dumps(
                {
                    "finalization_id": receipt["finalization_id"],
                    "finalized_evaluation_id": receipt["finalized_evaluation_id"],
                    "scientific_payload_id": receipt["scientific_payload_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
