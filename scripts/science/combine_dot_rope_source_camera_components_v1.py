#!/usr/bin/env python3
"""Combine fixed routed-camera source-support components into one DOT v3 rank gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_PARENT = "cd57cb81d1aa52707f26bb0f39829e848d15d0a67b064b390b24caf545923690"
EXPECTED_AUDIT = "66724cb78840f4b9ef3becf97e5765924094cf8db6eca2d892c44cfa7edb19b3"
EXPECTED_CANDIDATE = "expanded__overlap-345"
EXPECTED_SEQUENCES = [f"R{index:02d}" for index in range(11, 21)]
MINIMUM_PROMOTION = 9
EXPECTED_RANK = 6


def content_id(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.component_manifest)
    unsigned_manifest = dict(manifest)
    manifest_id = unsigned_manifest.pop("manifest_id", None)
    if content_id(unsigned_manifest) != manifest_id:
        raise ValueError("component manifest identity mismatch")
    if manifest.get("parent_protocol_id") != EXPECTED_PARENT:
        raise ValueError("parent protocol changed")
    if manifest.get("raw_routing_audit_id") != EXPECTED_AUDIT:
        raise ValueError("raw routing audit changed")
    if manifest.get("fixed_candidate") != EXPECTED_CANDIDATE:
        raise ValueError("source support geometry changed")

    rows: list[dict[str, Any]] = []
    component_results = []
    seen: set[str] = set()
    for component in manifest["components"]:
        camera = str(component["camera"])
        result_path = args.results_root / camera / "result.json"
        result = load_json(result_path)
        summaries = [
            item for item in result.get("candidate_summaries", [])
            if item.get("candidate_id") == EXPECTED_CANDIDATE
        ]
        if len(summaries) != 1:
            raise ValueError(f"{camera} lacks the frozen candidate")
        summary = summaries[0]
        expected_component_sequences = list(component["source_sequences"])
        per_sequence = list(summary.get("per_sequence", []))
        if sorted(item.get("sequence") for item in per_sequence) != sorted(expected_component_sequences):
            raise ValueError(f"{camera} result sequence roster changed")
        for item in per_sequence:
            sequence = str(item["sequence"])
            if sequence in seen:
                raise ValueError("source sequence appears in multiple camera components")
            seen.add(sequence)
            rank = int(item.get("factor_rank", 0))
            supported = bool(item.get("supported", False)) and rank == EXPECTED_RANK
            condition = float(item.get("observable_condition_ratio", 0.0))
            if not math.isfinite(condition) or condition < 0.0:
                raise ValueError("invalid observable condition ratio")
            rows.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "supported": supported,
                    "factor_rank": rank,
                    "fit_a_total": int(item.get("fit_a_total", 0)),
                    "fit_b_total": int(item.get("fit_b_total", 0)),
                    "overlap_common_total": int(item.get("overlap_common_total", 0)),
                    "overlap_nonempty_frames": int(item.get("overlap_nonempty_frames", 0)),
                    "normalized_support_margin": float(item.get("normalized_support_margin", 0.0)),
                    "observable_condition_ratio": condition,
                    "factor_error": item.get("factor_error"),
                }
            )
        component_results.append(
            {
                "camera": camera,
                "protocol_id": component["protocol_id"],
                "provider_bundle_id": result.get("provider_bundle_id"),
                "component_result_id": result.get("result_id"),
                "component_decision": result.get("decision"),
            }
        )

    if sorted(seen) != EXPECTED_SEQUENCES:
        raise ValueError("combined components do not cover R11-R20 exactly")
    rows.sort(key=lambda item: item["sequence"])
    supported = [item for item in rows if item["supported"]]
    decision = (
        "camera-routing-provider-rank-qualified"
        if len(supported) >= MINIMUM_PROMOTION
        else "camera-routing-provider-rank-negative"
    )
    result: dict[str, Any] = {
        "schema": "prob4d.dot-r11-r20-camera-routing-provider-rank-result",
        "schema_version": 1,
        "decision": decision,
        "parent_protocol_id": EXPECTED_PARENT,
        "raw_routing_audit_id": EXPECTED_AUDIT,
        "fixed_candidate": EXPECTED_CANDIDATE,
        "minimum_supported_sequences_for_promotion": MINIMUM_PROMOTION,
        "expected_factor_rank": EXPECTED_RANK,
        "supported_source_sequences": len(supported),
        "source_sequence_count": len(rows),
        "minimum_supported_condition_ratio": min(
            (item["observable_condition_ratio"] for item in supported), default=0.0
        ),
        "minimum_supported_margin": min(
            (item["normalized_support_margin"] for item in supported), default=0.0
        ),
        "per_sequence": rows,
        "component_results": component_results,
        "information_boundary": {
            "source_reconstruction_error_computed": False,
            "source_proper_score_computed": False,
            "r21_r30_payloads_opened": False,
            "r31_r70_payloads_opened": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
    }
    result["result_id"] = content_id(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "result_id": result["result_id"],
                "supported_source_sequences": len(supported),
                "minimum_supported_condition_ratio": result["minimum_supported_condition_ratio"],
                "minimum_supported_margin": result["minimum_supported_margin"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
