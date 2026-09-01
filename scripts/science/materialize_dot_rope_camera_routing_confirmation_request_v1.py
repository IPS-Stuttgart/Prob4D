#!/usr/bin/env python3
"""Materialize the DOT R21-R30 camera-routing one-shot request from qualified source evidence.

This helper opens no DOT payload. It accepts only the registered R11-R20 provider-rank
positive result plus exact GitHub artifact metadata and the immutable routed-provider
seal. It fails closed for any negative/technical source result and never overwrites an
existing target request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "prob4d.dot-rope-query-selective-camera-routing-confirmation-protocol"
PROTOCOL_ID = "6cfa652c6e5d419dda2b81ccd88ff25f7180a2abc2d3f749d6c8b3e9c1ad1195"
SOURCE_PROTOCOL_ID = "cd57cb81d1aa52707f26bb0f39829e848d15d0a67b064b390b24caf545923690"
RAW_AUDIT_ID = "66724cb78840f4b9ef3becf97e5765924094cf8db6eca2d892c44cfa7edb19b3"
SOURCE_DECISION = "camera-routing-provider-rank-qualified"
SOURCE_RESULT_SCHEMA = "prob4d.dot-r11-r20-camera-routing-provider-rank-result"
PROVIDER_SEAL_SCHEMA = "prob4d.dot-r11-r20-routed-provider-seal"
REQUEST_SCHEMA = "prob4d.dot-rope-camera-routing-confirmation-request"
PROTOCOL_PATH = "protocols/dot-rope-query-selective-camera-routing-confirmation-v1.json"
DEFAULT_OUTPUT = "protocols/execution_requests/dot_rope_camera_routing_confirmation_v1.json"


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def require_hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must contain {length} lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must contain {length} lowercase hexadecimal characters")
    return value


def verify_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(value)
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unsupported confirmation protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    if protocol_id != PROTOCOL_ID or content_id(unsigned) != protocol_id:
        raise ValueError("confirmation protocol identity mismatch")
    prerequisite = protocol.get("prerequisite") or {}
    if prerequisite.get("source_protocol_id") != SOURCE_PROTOCOL_ID:
        raise ValueError("source prerequisite changed")
    if prerequisite.get("required_decision") != SOURCE_DECISION:
        raise ValueError("source prerequisite decision changed")
    if prerequisite.get("minimum_supported_source_sequences") != 9:
        raise ValueError("source support threshold changed")
    if prerequisite.get("raw_routing_audit_id") != RAW_AUDIT_ID:
        raise ValueError("raw routing audit prerequisite changed")
    if protocol.get("target_sequences") != [f"R{i:02d}" for i in range(21, 31)]:
        raise ValueError("target sequence roster changed")
    if protocol.get("reserved_sequences") != "R31-R70":
        raise ValueError("reserve boundary changed")
    order = protocol.get("information_order") or {}
    if order.get("all_five_camera_provider_predictions_sealed_before_target_marker_access") is not True:
        raise ValueError("provider-before-marker order changed")
    if order.get("camera_routing_and_factor_query_predictions_sealed_before_target_3d") is not True:
        raise ValueError("prediction-before-outcome order changed")
    if order.get("r31_r70_payloads_opened") is not False:
        raise ValueError("reserve boundary changed")
    return protocol


def verify_source_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if result.get("schema") != SOURCE_RESULT_SCHEMA or result.get("schema_version") != 1:
        raise ValueError("source provider-rank result schema changed")
    unsigned = dict(result)
    result_id = unsigned.pop("result_id", None)
    require_hex(result_id, name="source result_id", length=64)
    if content_id(unsigned) != result_id:
        raise ValueError("source provider-rank result identity mismatch")
    if result.get("decision") != SOURCE_DECISION:
        raise ValueError("source provider-rank prerequisite is not qualified")
    if result.get("parent_protocol_id") != SOURCE_PROTOCOL_ID:
        raise ValueError("source provider-rank protocol changed")
    if result.get("raw_routing_audit_id") != RAW_AUDIT_ID:
        raise ValueError("source raw routing audit changed")
    supported = result.get("supported_source_sequences")
    if not isinstance(supported, int) or supported < 9 or supported > 10:
        raise ValueError("source provider-rank support is below the registered threshold")
    boundary = result.get("information_boundary") or {}
    for key in (
        "source_reconstruction_error_computed",
        "source_proper_score_computed",
        "r21_r30_payloads_opened",
        "r31_r70_payloads_opened",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"source information boundary changed: {key}")
    return result


def verify_provider_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    seal = dict(value)
    if seal.get("schema") != PROVIDER_SEAL_SCHEMA or seal.get("schema_version") != 1:
        raise ValueError("source routed-provider seal schema changed")
    unsigned = dict(seal)
    seal_id = unsigned.pop("provider_seal_id", None)
    require_hex(seal_id, name="provider_seal_id", length=64)
    if content_id(unsigned) != seal_id:
        raise ValueError("source routed-provider seal identity mismatch")
    if seal.get("parent_protocol_id") != SOURCE_PROTOCOL_ID:
        raise ValueError("source provider seal protocol changed")
    if seal.get("raw_routing_audit_id") != RAW_AUDIT_ID:
        raise ValueError("source provider seal raw audit changed")
    if seal.get("source_marker_payloads_opened") is not False:
        raise ValueError("source provider was not sealed before source marker access")
    if seal.get("confirmation_payloads_opened") is not False:
        raise ValueError("source provider seal indicates confirmation access")
    components = seal.get("components")
    if not isinstance(components, list) or sorted(item.get("camera") for item in components) != [
        "cam001", "cam002", "cam005"
    ]:
        raise ValueError("source routed-provider component roster changed")
    return seal


def verify_artifact(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    artifact = dict(value)
    if not isinstance(artifact.get("id"), int) or artifact["id"] <= 0:
        raise ValueError(f"{label} artifact id must be positive")
    if not isinstance(artifact.get("name"), str) or not artifact["name"]:
        raise ValueError(f"{label} artifact name must be nonempty")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError(f"{label} artifact digest must be SHA-256")
    require_hex(digest.removeprefix("sha256:"), name=f"{label} artifact digest", length=64)
    if artifact.get("expired") is not False:
        raise ValueError(f"{label} artifact must be unexpired")
    workflow = artifact.get("workflow_run")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("id"), int) or workflow["id"] <= 0:
        raise ValueError(f"{label} workflow_run.id must be positive")
    return artifact


def build_request(
    *,
    protocol: Mapping[str, Any],
    protocol_git_blob_sha: str,
    source_result: Mapping[str, Any],
    provider_seal: Mapping[str, Any],
    source_result_artifact: Mapping[str, Any],
    source_provider_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = verify_protocol(protocol)
    result = verify_source_result(source_result)
    seal = verify_provider_seal(provider_seal)
    result_artifact = verify_artifact(source_result_artifact, label="source result")
    provider_artifact = verify_artifact(source_provider_artifact, label="source provider")
    require_hex(protocol_git_blob_sha, name="confirmation protocol Git blob", length=40)
    if result_artifact["workflow_run"]["id"] != provider_artifact["workflow_run"]["id"]:
        raise ValueError("source result/provider artifacts came from different workflow runs")
    if seal["request_id"] != source_result.get("component_results", [{}])[0].get("request_id", seal["request_id"]):
        # Older component-result schema does not duplicate the request id; the seal remains authoritative.
        raise ValueError("source provider request identity changed")
    unsigned: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": 1,
        "protocol_path": PROTOCOL_PATH,
        "protocol_git_blob_sha": protocol_git_blob_sha,
        "protocol_id": frozen["protocol_id"],
        "target_sequences": list(frozen["target_sequences"]),
        "reserved_sequences": frozen["reserved_sequences"],
        "source_prerequisite": {
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "raw_routing_audit_id": RAW_AUDIT_ID,
            "run_id": result_artifact["workflow_run"]["id"],
            "result_artifact_id": result_artifact["id"],
            "result_artifact_name": result_artifact["name"],
            "result_artifact_digest": result_artifact["digest"],
            "provider_artifact_id": provider_artifact["id"],
            "provider_artifact_name": provider_artifact["name"],
            "provider_artifact_digest": provider_artifact["digest"],
            "source_result_id": result["result_id"],
            "source_provider_seal_id": seal["provider_seal_id"],
            "decision": result["decision"],
            "supported_source_sequences": result["supported_source_sequences"],
        },
        "all_five_camera_prediction_authorized": True,
        "target_2d_routing_and_factor_authorized": True,
        "target_3d_one_shot_scoring_authorized": True,
        "post_open_tuning_authorized": False,
        "post_rank_camera_switch_authorized": False,
        "r31_r70_access_authorized": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "claim_boundary": frozen["claim_boundary"],
    }
    return {**unsigned, "request_id": content_id(unsigned)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_PATH))
    parser.add_argument("--protocol-git-blob-sha", required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-provider-seal", type=Path, required=True)
    parser.add_argument("--source-result-artifact-metadata", type=Path, required=True)
    parser.add_argument("--source-provider-artifact-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    request = build_request(
        protocol=read_json(args.protocol),
        protocol_git_blob_sha=args.protocol_git_blob_sha,
        source_result=read_json(args.source_result),
        provider_seal=read_json(args.source_provider_seal),
        source_result_artifact=read_json(args.source_result_artifact_metadata),
        source_provider_artifact=read_json(args.source_provider_artifact_metadata),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "request_id": request["request_id"],
        "source_run_id": request["source_prerequisite"]["run_id"],
        "source_result_id": request["source_prerequisite"]["source_result_id"],
        "supported_source_sequences": request["source_prerequisite"]["supported_source_sequences"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
