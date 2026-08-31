#!/usr/bin/env python3
"""Materialize the frozen DOT R11--R30 request from verified R04--R10 evidence.

This helper opens no DOT payload. It independently invokes the preregistered
R04--R10 result verifier, binds the exact GitHub artifact metadata and reviewed
R11--R30 protocol blob, and emits the one content-addressed request file expected
by ``dot-rope-query-selective-heldout-v1.yml``. A non-strong prerequisite fails
closed and no output is created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "prob4d.dot-rope-query-selective-heldout-request"
PROTOCOL_SCHEMA = "prob4d.dot-rope-query-selective-heldout-protocol"
SCHEMA_VERSION = 1
REQUIRED_DECISION = "heldout-strong-positive"
PROTOCOL_REPOSITORY_PATH = "protocols/dot-rope-query-selective-heldout-v1.json"
DEFAULT_OUTPUT = "protocols/execution_requests/dot_rope_query_selective_heldout_v1.json"
DEFAULT_CLAIM_BOUNDARY = (
    "Authorize one frozen R11-R30 learned-CUT3R query-selective evaluation only "
    "after independently verified R04-R10 strong-positive evidence. Provider "
    "predictions and query decisions must seal before 3-D marker scoring; R31-R70 "
    "remain unopened. No target-side retuning, BayesianPhysTwin execution, "
    "Causal4D execution, deployment-safety claim, or state-of-the-art claim is "
    "authorized."
)

Verifier = Callable[[Mapping[str, Any], Mapping[str, Any] | None], dict[str, Any]]


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    return value


def _load_verifier(path: Path) -> Verifier:
    spec = importlib.util.spec_from_file_location("dot_r04_r10_verifier", path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load the R04--R10 verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    verifier = getattr(module, "verify", None)
    if not callable(verifier):
        raise ValueError("R04--R10 verifier does not expose verify()")
    return verifier


def _verified_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(protocol)
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported query-selective protocol")
    unsigned = dict(value)
    protocol_id = _hex(unsigned.pop("protocol_id", None), name="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("query-selective protocol identity mismatch")
    prerequisite = value.get("prerequisite")
    if not isinstance(prerequisite, dict):
        raise ValueError("query-selective prerequisite block is missing")
    if prerequisite.get("required_decision") != REQUIRED_DECISION:
        raise ValueError("query-selective protocol no longer requires a strong positive")
    target = value.get("target_sequences")
    if target != [f"R{index:02d}" for index in range(11, 31)]:
        raise ValueError("query-selective target sequence roster changed")
    if value.get("reserved_sequences") != "R31-R70":
        raise ValueError("query-selective reserve changed")
    return value


def _verified_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(value)
    for name in ("id",):
        if not isinstance(artifact.get(name), int) or artifact[name] <= 0:
            raise ValueError(f"artifact {name} must be a positive integer")
    if not isinstance(artifact.get("name"), str) or not artifact["name"]:
        raise ValueError("artifact name must be nonempty")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("artifact digest must be a GitHub SHA-256 digest")
    _hex(digest.removeprefix("sha256:"), name="artifact digest", length=64)
    if artifact.get("expired") is not False:
        raise ValueError("artifact must be unexpired")
    workflow = artifact.get("workflow_run")
    if (
        not isinstance(workflow, dict)
        or not isinstance(workflow.get("id"), int)
        or workflow["id"] <= 0
    ):
        raise ValueError("artifact workflow_run.id must be a positive integer")
    return artifact


def build_request(
    *,
    protocol: Mapping[str, Any],
    protocol_git_blob_sha: str,
    artifact: Mapping[str, Any],
    verification: Mapping[str, Any],
    marker_support_id: str,
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY,
) -> dict[str, Any]:
    frozen = _verified_protocol(protocol)
    bound_artifact = _verified_artifact(artifact)
    _hex(protocol_git_blob_sha, name="protocol Git blob", length=40)
    _hex(marker_support_id, name="marker_support_id", length=64)
    if verification.get("decision") != REQUIRED_DECISION:
        raise ValueError("R04--R10 prerequisite is not strong-positive")
    evaluation_id = _hex(verification.get("result_id"), name="evaluation_id", length=64)
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise ValueError("claim_boundary must be nonempty")
    prerequisite = frozen["prerequisite"]
    unsigned: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol_path": PROTOCOL_REPOSITORY_PATH,
        "protocol_git_blob_sha": protocol_git_blob_sha,
        "target_sequences": list(frozen["target_sequences"]),
        "reserved_sequences": frozen["reserved_sequences"],
        "prerequisite": {
            "protocol_id": prerequisite["protocol_id"],
            "source_calibration_id": prerequisite["source_calibration_id"],
            "run_id": int(bound_artifact["workflow_run"]["id"]),
            "artifact_id": int(bound_artifact["id"]),
            "artifact_name": bound_artifact["name"],
            "artifact_digest": bound_artifact["digest"],
            "evaluation_id": evaluation_id,
            "marker_support_id": marker_support_id,
            "decision": REQUIRED_DECISION,
        },
        "normal_view_prediction_authorized": True,
        "marker_2d_factor_seal_authorized": True,
        "marker_3d_scoring_authorized": True,
        "post_open_tuning_authorized": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "claim_boundary": claim_boundary.strip(),
    }
    return {**unsigned, "request_id": content_id(unsigned)}


def materialize(
    *,
    protocol_path: Path,
    protocol_git_blob_sha: str,
    result_path: Path,
    marker_support_path: Path,
    artifact_metadata_path: Path,
    output_path: Path,
    verifier: Verifier,
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    protocol = _read_json(protocol_path)
    result = _read_json(result_path)
    support = _read_json(marker_support_path)
    artifact = _read_json(artifact_metadata_path)
    verification = verifier(result, support)
    support_id = _hex(support.get("support_id"), name="marker_support_id", length=64)
    request = build_request(
        protocol=protocol,
        protocol_git_blob_sha=protocol_git_blob_sha,
        artifact=artifact,
        verification=verification,
        marker_support_id=support_id,
        claim_boundary=claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(request, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_REPOSITORY_PATH))
    parser.add_argument("--protocol-git-blob-sha", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--marker-support", type=Path, required=True)
    parser.add_argument("--artifact-metadata", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("scripts/science/verify_dot_rope_cut3r_heldout_result.py"),
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--claim-boundary", default=DEFAULT_CLAIM_BOUNDARY)
    return parser


def main() -> int:
    args = _parser().parse_args()
    request = materialize(
        protocol_path=args.protocol,
        protocol_git_blob_sha=args.protocol_git_blob_sha,
        result_path=args.result,
        marker_support_path=args.marker_support,
        artifact_metadata_path=args.artifact_metadata,
        output_path=args.output,
        verifier=_load_verifier(args.verifier),
        claim_boundary=args.claim_boundary,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "request_id": request["request_id"],
                "prerequisite_run_id": request["prerequisite"]["run_id"],
                "prerequisite_evaluation_id": request["prerequisite"]["evaluation_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())