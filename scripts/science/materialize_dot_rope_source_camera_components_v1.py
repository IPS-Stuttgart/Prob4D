#!/usr/bin/env python3
"""Materialize fixed-camera component protocols from the frozen DOT camera-routing source v3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

V3_SCHEMA = "prob4d.dot-rope-query-selective-camera-routing-source-protocol"
V2_SCHEMA = "prob4d.dot-rope-query-selective-source-support-protocol"
EXPECTED_V3_ID = "cd57cb81d1aa52707f26bb0f39829e848d15d0a67b064b390b24caf545923690"
EXPECTED_RAW_AUDIT_ID = "66724cb78840f4b9ef3becf97e5765924094cf8db6eca2d892c44cfa7edb19b3"
EXPECTED_CANDIDATE = "expanded__overlap-345"
EXPECTED_ROUTING = {
    "R11": "cam005",
    "R12": "cam005",
    "R13": "cam001",
    "R14": "cam001",
    "R15": "cam001",
    "R16": "cam001",
    "R17": "cam002",
    "R18": "cam001",
    "R19": "cam001",
    "R20": "cam005",
}


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


def verify_v3(value: dict[str, Any]) -> None:
    if value.get("schema") != V3_SCHEMA or value.get("schema_version") != 3:
        raise ValueError("camera-routing source protocol schema changed")
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id", None)
    if protocol_id != EXPECTED_V3_ID or content_id(unsigned) != protocol_id:
        raise ValueError("camera-routing source protocol identity changed")
    routing = value.get("routing_development") or {}
    if routing.get("source_audit_id") != EXPECTED_RAW_AUDIT_ID:
        raise ValueError("raw routing audit identity changed")
    if routing.get("selected_candidate") != EXPECTED_CANDIDATE:
        raise ValueError("raw routing support geometry changed")
    if routing.get("selected_camera_routing") != EXPECTED_ROUTING:
        raise ValueError("source camera routing changed")
    if (value.get("information_boundary") or {}).get("r21_r30_payloads_opened") is not False:
        raise ValueError("R21-R30 boundary changed")


def materialize(v3: dict[str, Any], base: dict[str, Any], output: Path) -> dict[str, Any]:
    if base.get("schema") != V2_SCHEMA or base.get("schema_version") != 2:
        raise ValueError("base source-support v2 protocol changed")
    groups: dict[str, list[str]] = defaultdict(list)
    for sequence, camera in EXPECTED_ROUTING.items():
        groups[camera].append(sequence)
    if sorted(groups) != ["cam001", "cam002", "cam005"]:
        raise ValueError("provider camera group roster changed")

    output.mkdir(parents=True, exist_ok=False)
    components = []
    for camera in sorted(groups):
        sequences = sorted(groups[camera])
        component = json.loads(json.dumps(base))
        component.pop("protocol_id", None)
        component["camera"] = camera
        component["source_sequences"] = sequences
        component["support_selection"]["fit_frame_profiles"] = [
            {
                "id": "expanded",
                "metric_fit_a_frames": [1, 2, 3],
                "metric_fit_b_frames": [5, 6, 7],
            }
        ]
        component["support_selection"]["overlap_groups"] = [
            {"id": "overlap-345", "frames": [3, 4, 5]}
        ]
        component["support_selection"]["minimum_source_supported_sequences_for_promotion"] = len(sequences)
        component["support_selection"]["selection_objective"] = [
            "fixed by source raw-support routing audit 66724cb78840f4b9ef3becf97e5765924094cf8db6eca2d892c44cfa7edb19b3"
        ]
        component["claim_boundary"] = (
            f"Source-only routed-camera factor qualification for {camera} on {','.join(sequences)}. "
            "The camera/sequence routing and expanded/overlap-345 geometry were frozen before this provider run. "
            "No reconstruction error or proper score may be used; R21-R70 remain closed."
        )
        component["protocol_id"] = content_id(component)
        filename = f"component-{camera}.json"
        path = output / filename
        path.write_text(json.dumps(component, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        components.append(
            {
                "camera": camera,
                "source_sequences": sequences,
                "protocol_file": filename,
                "protocol_id": component["protocol_id"],
            }
        )

    manifest: dict[str, Any] = {
        "schema": "prob4d.dot-r11-r20-camera-routing-component-manifest",
        "schema_version": 1,
        "parent_protocol_id": v3["protocol_id"],
        "raw_routing_audit_id": EXPECTED_RAW_AUDIT_ID,
        "fixed_candidate": EXPECTED_CANDIDATE,
        "components": components,
        "source_sequence_count": sum(len(item["source_sequences"]) for item in components),
        "confirmation_payloads_opened": False,
        "source_performance_metrics_used": False,
    }
    if manifest["source_sequence_count"] != 10:
        raise ValueError("component roster does not cover R11-R20 exactly")
    manifest["manifest_id"] = content_id(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-v2-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    v3 = load_json(args.protocol)
    verify_v3(v3)
    manifest = materialize(v3, load_json(args.base_v2_protocol), args.output_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
