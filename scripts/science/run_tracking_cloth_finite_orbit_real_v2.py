#!/usr/bin/env python3
"""Run the source-frozen Tracking Cloth finite-orbit replication v2.

The v1 execution terminated before its source seal because OptiTrack Motive
stores marker identity across multiple header rows. A source-only audit then
showed that polyester hand-held recordings use a separate unlabeled-marker
namespace. This separately frozen replication uses the exact parent scientific
construction on the support-feasible A2 cotton/denim/wool cohort, selects one
marker triplet from shake/twist recordings only, seals that selection, and only
then opens collision-family target headers and trajectories.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from prob4d.motive_csv import (
    common_marker_labels,
    read_motive_layout,
    read_motive_markers,
)

BASE_FILENAME = "run_tracking_cloth_finite_orbit_real_v1.py"
BASE_GIT_BLOB_SHA1 = "081da94007493480b1e8efb8169c1c642bed1761"
PROTOCOL_SCHEMA = "prob4d.tracking-cloth-finite-orbit-real.v2"
PROTOCOL_ID = "4652d72f9e8d4c80c69df86b7a48a6f4e307e4131ea3bea04e09deed10db5eb0"
RESULT_SCHEMA = "prob4d.tracking-cloth-finite-orbit-result.v2"
PARENT_PROTOCOL_BLOB = "808d85e29b498ba1e7f92ea4ed67c4f6ec156c6d"
ALLOWED_MATERIALS = ("cotton", "denim", "wool")
REQUIRED_SIZE = "A2"
EXPECTED_SOURCE = 24
EXPECTED_TARGET = 42


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(value)}\0".encode() + value,
        usedforsecurity=False,
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_base() -> ModuleType:
    source = Path(__file__).with_name(BASE_FILENAME)
    if _git_blob_sha1(source.read_bytes()) != BASE_GIT_BLOB_SHA1:
        raise RuntimeError("registered finite-orbit v1 implementation changed")
    module_name = "tracking_cloth_finite_orbit_v1_frozen"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered finite-orbit v1 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module._read_markers = read_motive_markers
    module._common_marker_names = lambda recordings, maximum: common_marker_labels(
        [recording.path for recording in recordings], maximum
    )
    return module


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("v2 protocol must be one JSON object")
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id", None)
    if supplied != PROTOCOL_ID or _sha256(unsigned) != supplied:
        raise ValueError("v2 protocol identity changed")
    if value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("v2 protocol schema changed")
    if value.get("dataset", {}).get("allowed_materials") != list(ALLOWED_MATERIALS):
        raise ValueError("allowed material roster changed")
    if value.get("dataset", {}).get("required_size") != REQUIRED_SIZE:
        raise ValueError("required cloth size changed")
    if value.get("dataset", {}).get("expected_source_files") != EXPECTED_SOURCE:
        raise ValueError("expected source count changed")
    if value.get("dataset", {}).get("expected_target_files") != EXPECTED_TARGET:
        raise ValueError("expected target count changed")
    if value.get("marker_support", {}).get("maximum_common_marker_count") != 20:
        raise ValueError("source marker support changed")
    if value.get("decision", {}).get("expected_target_groups") != EXPECTED_TARGET:
        raise ValueError("decision target count changed")
    order = value.get("information_order", {})
    if order.get("target_header_or_trajectory_opened_before_source_seal") is not False:
        raise ValueError("target information order changed")
    if order.get("target_side_retuning_allowed") is not False:
        raise ValueError("target-side retuning was enabled")
    if order.get("unsupported_target_replacement_allowed") is not False:
        raise ValueError("unsupported target replacement was enabled")
    return value


def _load_parent(protocol: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    parent_spec = protocol["parent_protocol"]
    parent_path = protocol_path.parent.parent / Path(parent_spec["path"])
    payload = parent_path.read_bytes()
    if _git_blob_sha1(payload) != PARENT_PROTOCOL_BLOB:
        raise ValueError("parent finite-orbit protocol changed")
    if parent_spec != {
        "path": "protocols/tracking-cloth-finite-orbit-real-v1.json",
        "git_blob_sha1": PARENT_PROTOCOL_BLOB,
        "protocol_id": "tracking-cloth-finite-orbit-real-v1",
    }:
        raise ValueError("parent protocol binding changed")
    parent = json.loads(payload)
    if parent.get("protocol_id") != parent_spec["protocol_id"]:
        raise ValueError("parent protocol ID changed")
    return parent


def _tokens(path: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", path.casefold()) if token}


def _material_and_size(relative_path: str) -> tuple[str, str]:
    tokens = _tokens(relative_path)
    materials = [value for value in (*ALLOWED_MATERIALS, "polyester") if value in tokens]
    sizes = [value for value in ("a2", "a3") if value in tokens]
    if len(materials) != 1 or len(sizes) != 1:
        raise ValueError(f"material/size metadata is ambiguous: {relative_path}")
    return materials[0], sizes[0].upper()


def _subset(recordings: list[Any], *, source: bool) -> tuple[list[Any], list[Any]]:
    eligible: list[Any] = []
    excluded: list[Any] = []
    for recording in recordings:
        material, size = _material_and_size(recording.relative_path)
        accepted = material in ALLOWED_MATERIALS and size == REQUIRED_SIZE
        (eligible if accepted else excluded).append(recording)
    expected = EXPECTED_SOURCE if source else EXPECTED_TARGET
    if len(eligible) != expected:
        kind = "source" if source else "target"
        raise ValueError(f"expected {expected} eligible {kind} recordings, found {len(eligible)}")
    return eligible, excluded


def _effective_parent(parent: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(parent)
    effective["dataset"]["expected_source_files"] = EXPECTED_SOURCE
    effective["dataset"]["expected_target_files"] = EXPECTED_TARGET
    effective["geometry"]["maximum_common_marker_count"] = 20
    return effective


def _target_support(
    base: ModuleType,
    recordings: list[Any],
    marker_triplet: tuple[str, str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    required = set(marker_triplet)
    for recording in recordings:
        layout = read_motive_layout(recording.path)
        available = set(layout.marker_labels)
        row = {
            "group_id": base._stable_id(recording.relative_path),
            "available_marker_count": len(layout.markers),
            "missing_marker_labels": sorted(required - available),
            "length_units": layout.length_units,
            "parser": "strict-motive-multirow-marker-v1",
        }
        (unsupported if row["missing_marker_labels"] else supported).append(row)
    return supported, unsupported


def _summary(result: dict[str, Any], base: ModuleType, triplet: tuple[str, str, str]) -> str:
    if result["status"] == "target-marker-support-negative":
        return "\n".join(
            [
                "# Tracking Cloth finite-orbit replication v2",
                "",
                "Status: **target-marker-support-negative**",
                "",
                f"Protocol ID: `{result['protocol_id']}`",
                f"Source seal ID: `{result['source_seal_id']}`",
                f"Unsupported target groups: {len(result['target_support']['unsupported'])}",
                "",
                "The source-selected marker triplet was frozen before target header access. "
                "No target trajectory was parsed and no unsupported group was replaced.",
                "",
            ]
        )
    parent_summary = base._make_summary(result, triplet)
    heading = "# Tracking Cloth finite-orbit replication v2"
    if parent_summary.startswith("# Tracking Cloth finite-orbit real-geometry result"):
        parent_summary = parent_summary.replace(
            "# Tracking Cloth finite-orbit real-geometry result",
            heading,
            1,
        )
    prefix = "\n".join(
        [
            parent_summary.rstrip(),
            "",
            "## Support-feasible cohort",
            "",
            f"- source recordings: {EXPECTED_SOURCE} A2 shake/twist recordings",
            f"- held-out recordings: {EXPECTED_TARGET} A2 collision-family recordings",
            f"- materials: {', '.join(ALLOWED_MATERIALS)}",
            (
                "- polyester excluded before target trajectory access because "
                "source marker identity namespaces were inconsistent"
            ),
            f"- protocol ID: `{result['protocol_id']}`",
            "",
        ]
    )
    return prefix


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = _load_protocol(protocol_path)
    parent = _load_parent(protocol, protocol_path)
    effective = _effective_parent(parent)
    base = _load_base()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root does not exist: {dataset_root}")
    csv_paths = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    if len(csv_paths) != protocol["dataset"]["expected_csv_files"]:
        raise ValueError("official CSV roster changed")
    recordings, classification = base._classify_recordings(dataset_root, csv_paths, parent)
    all_source = [recording for recording in recordings if recording.label != "collision"]
    all_target = [recording for recording in recordings if recording.label == "collision"]
    source, excluded_source = _subset(all_source, source=True)
    target, excluded_target = _subset(all_target, source=False)

    marker_names = common_marker_labels(
        [recording.path for recording in source],
        protocol["marker_support"]["maximum_common_marker_count"],
    )
    expected_labels = protocol["marker_support"]["source_header_audit"]["allowed_A2_common_labels"]
    if marker_names != expected_labels:
        raise ValueError("source common marker labels changed")
    source_samples, source_metadata = base._collect_source_samples(
        source,
        marker_names,
        int(effective["geometry"]["source_frames_per_recording"]),
    )
    marker_triplet, selection = base._select_marker_triplet(
        source_samples,
        marker_names,
        float(effective["geometry"]["minimum_anchor_distance_mm"]),
        float(effective["geometry"]["minimum_probe_radius_mm"]),
    )
    source_seal = {
        "schema": "prob4d.tracking-cloth-finite-orbit-source-seal.v2",
        "protocol_id": protocol["protocol_id"],
        "parent_protocol_git_blob_sha1": PARENT_PROTOCOL_BLOB,
        "source_revision": args.source_revision,
        "classification": classification,
        "eligible_source_group_count": len(source),
        "eligible_source_group_ids": [base._stable_id(row.relative_path) for row in source],
        "excluded_source_group_count": len(excluded_source),
        "excluded_source_group_ids": [
            base._stable_id(row.relative_path) for row in excluded_source
        ],
        "eligible_target_group_count_from_filename_metadata": len(target),
        "excluded_target_group_count_from_filename_metadata": len(excluded_target),
        "common_marker_labels": marker_names,
        "selected_anchor_a": marker_triplet[0],
        "selected_anchor_b": marker_triplet[1],
        "selected_probe": marker_triplet[2],
        "selection": selection,
        "source_sample_frames": int(source_samples.shape[0]),
        "source_metadata": source_metadata,
        "target_header_opened": False,
        "target_trajectory_opened": False,
    }
    source_seal["source_seal_id"] = _sha256(source_seal)
    _write_json(output_dir / "source_seal.json", source_seal)

    supported, unsupported = _target_support(base, target, marker_triplet)
    support = {
        "schema": "prob4d.tracking-cloth-target-marker-support.v2",
        "source_seal_id": source_seal["source_seal_id"],
        "required_marker_labels": list(marker_triplet),
        "supported": supported,
        "unsupported": unsupported,
        "target_trajectory_values_parsed": False,
        "unsupported_groups_replaced": False,
    }
    support["support_id"] = _sha256(support)
    _write_json(output_dir / "target_support.json", support)
    if unsupported:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "target-marker-support-negative",
            "protocol_id": protocol["protocol_id"],
            "source_revision": args.source_revision,
            "source_seal_id": source_seal["source_seal_id"],
            "target_support": support,
            "claim_boundary": protocol["claim_boundary"],
        }
        result["result_id"] = _sha256(result)
        _write_json(output_dir / "result.json", result)
        (output_dir / "summary.md").write_text(
            _summary(result, base, marker_triplet), encoding="utf-8"
        )
        return 2

    groups = [
        base._evaluate_recording(recording, marker_triplet, effective) for recording in target
    ]
    for group, recording in zip(groups, target, strict=True):
        material, size = _material_and_size(recording.relative_path)
        group["material"] = material
        group["size"] = size
    aggregate = base._aggregate_groups(groups, effective)
    criteria = base._registered_criteria(aggregate, effective)
    status = (
        "evaluated-real-geometry-passed"
        if all(criteria.values())
        else "evaluated-real-geometry-failed"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "parent_protocol_git_blob_sha1": PARENT_PROTOCOL_BLOB,
        "source_revision": args.source_revision,
        "source_seal_id": source_seal["source_seal_id"],
        "dataset_root_name": dataset_root.name,
        "classification": classification,
        "cohort": {
            "allowed_materials": list(ALLOWED_MATERIALS),
            "required_size": REQUIRED_SIZE,
            "source_groups": len(source),
            "target_groups": len(target),
            "excluded_source_groups": len(excluded_source),
            "excluded_target_groups": len(excluded_target),
        },
        "selected_markers": {
            "anchor_a": marker_triplet[0],
            "anchor_b": marker_triplet[1],
            "probe": marker_triplet[2],
        },
        "selection": selection,
        "target_support": support,
        "aggregate": aggregate,
        "criteria": criteria,
        "groups": groups,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _sha256(result)
    _write_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(_summary(result, base, marker_triplet), encoding="utf-8")
    _write_json(
        output_dir / "inventory.json",
        {
            "schema": "prob4d.tracking-cloth-finite-orbit-inventory.v2",
            "csv_count": len(csv_paths),
            "eligible_source_group_ids": [
                base._stable_id(recording.relative_path) for recording in source
            ],
            "eligible_target_group_ids": [
                base._stable_id(recording.relative_path) for recording in target
            ],
            "excluded_source_group_ids": [
                base._stable_id(recording.relative_path) for recording in excluded_source
            ],
            "excluded_target_group_ids": [
                base._stable_id(recording.relative_path) for recording in excluded_target
            ],
            "raw_payload_uploaded": False,
        },
    )
    return 0 if all(criteria.values()) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
