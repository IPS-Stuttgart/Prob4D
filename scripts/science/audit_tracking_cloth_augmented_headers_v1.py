#!/usr/bin/env python3
"""Header-only audit of trajectory-unopened augmented Tracking Cloth recordings.

The preceding continuous-SO(2) study parsed full trajectories only for the 80
cloth-only layouts. It inspected Motive layouts for the remaining recordings
and returned before calling ``read_motive_markers``. This audit deliberately
stays at the same information boundary: it reads public relative-path metadata
and validated Motive header layouts, but never parses marker trajectory values.

The output is support metadata for a later, separately frozen experiment. It is
not a predictive, calibration, or utility result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.motive_csv import read_motive_layout

SCHEMA = "prob4d.tracking-cloth-augmented-header-audit.v1"
RESULT_SCHEMA = "prob4d.tracking-cloth-augmented-header-audit-result.v1"

_MATERIALS = ("cotton", "denim", "polyester", "wool")
_SIZES = ("a2", "a3")
_CATEGORY_ALIASES = {
    "shake": {"shake"},
    "twist": {"twist"},
    "hitting": {"hitting"},
    "tablecloth": {"tablecloth"},
    "self-collision": {"self-collision", "self-collisions", "selfcollision", "selfcollisions"},
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _tokens(text: str) -> set[str]:
    return {piece for piece in re.split(r"[^a-z0-9]+", text.casefold()) if piece}


def _normalized_parts(relative_path: str) -> set[str]:
    parts: set[str] = set()
    for raw in Path(relative_path).parts:
        normalized = re.sub(r"[^a-z0-9]+", "-", raw.casefold()).strip("-")
        if normalized:
            parts.add(normalized)
            parts.add(normalized.replace("-", ""))
    return parts


def _material_size_category(relative_path: str) -> tuple[str, str, str]:
    tokens = _tokens(relative_path)
    materials = [value for value in _MATERIALS if value in tokens]
    sizes = [value.upper() for value in _SIZES if value in tokens]
    if len(materials) != 1 or len(sizes) != 1:
        raise ValueError(f"ambiguous material/size metadata: {relative_path}")

    parts = _normalized_parts(relative_path)
    categories = [
        category
        for category, aliases in _CATEGORY_ALIASES.items()
        if parts.intersection(aliases)
    ]
    if len(categories) != 1:
        raise ValueError(f"ambiguous motion category metadata: {relative_path}")
    return materials[0], sizes[0], categories[0]


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported header-audit protocol")
    unsigned = dict(value)
    supplied = unsigned.pop("protocol_id", None)
    expected = _sha256_bytes(_canonical_bytes(unsigned))
    if supplied != expected:
        raise ValueError("header-audit protocol identity changed")
    order = value["information_order"]
    if order["marker_trajectory_value_parsing_allowed"] is not False:
        raise ValueError("trajectory parsing was authorized")
    if order["trajectory_hashing_allowed"] is not False:
        raise ValueError("trajectory hashing was authorized")
    if order["raw_data_publication_authorized"] is not False:
        raise ValueError("raw publication was authorized")
    return value


def _natural_label_key(value: str) -> tuple[int, int | str]:
    stripped = value.strip()
    if stripped.isdigit():
        return 0, int(stripped)
    return 1, stripped.casefold()


def _audit_file(path: Path, dataset_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    relative = path.relative_to(dataset_root).as_posix()
    material, size, category = _material_size_category(relative)
    layout = read_motive_layout(path)
    labels = tuple(layout.marker_labels)
    cloth_labels = tuple(protocol["dataset"]["cloth_marker_labels"])
    cloth_set = set(cloth_labels)
    available = set(labels)
    expected_cloth_only = int(protocol["dataset"]["cloth_only_marker_count"])
    augmented = size == protocol["dataset"]["expected_size"] and len(labels) > expected_cloth_only
    missing_cloth = sorted(cloth_set - available, key=_natural_label_key)
    extras = sorted(available - cloth_set, key=_natural_label_key)
    relative_hash = hashlib.sha256(relative.encode()).hexdigest()
    return {
        "group_id": relative_hash[:16],
        "relative_path": relative,
        "relative_path_sha256": relative_hash,
        "material": material,
        "size": size,
        "category": category,
        "marker_count": len(labels),
        "marker_labels": list(labels),
        "marker_unique_ids_sha256": [
            hashlib.sha256(marker.unique_id.encode()).hexdigest()
            for marker in layout.markers
        ],
        "cloth_labels_complete": not missing_cloth,
        "missing_cloth_labels": missing_cloth,
        "extra_marker_labels": extras,
        "augmented_layout_candidate": augmented,
        "header_row_count": layout.header_row_count,
        "data_start_row_zero_based": layout.data_start_row,
        "length_units": layout.length_units,
        "marker_trajectory_values_parsed": False,
    }


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# Tracking Cloth augmented-layout header audit",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- CSV recordings: {result['inventory']['csv_file_count']}",
        f"- augmented A2 layouts: {result['inventory']['augmented_candidate_count']}",
        f"- complete cloth-label support: {result['inventory']['complete_cloth_support_count']}",
        f"- trajectory marker values parsed: {str(result['information_boundary']['marker_trajectory_values_parsed']).lower()}",
        "",
        "| Category | Material | Count | Cloth labels complete | Marker counts |",
        "|---|---|---:|---:|---|",
    ]
    for row in result["strata"]:
        lines.append(
            "| {category} | {material} | {count} | {complete} | {markers} |".format(
                category=row["category"],
                material=row["material"],
                count=row["count"],
                complete=row["complete_cloth_support_count"],
                markers=", ".join(str(value) for value in row["marker_counts"]),
            )
        )
    lines.extend(
        [
            "",
            "This is support metadata only. A later source/confirmation split and every "
            "scientific threshold require a separate frozen protocol before any augmented "
            "trajectory is parsed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    dataset_root: Path,
    protocol_path: Path,
    output_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = _load_protocol(protocol_path)
    csv_paths = sorted(path for path in dataset_root.rglob("*.csv") if path.is_file())
    expected_csv = int(protocol["dataset"]["expected_csv_files"])
    if len(csv_paths) != expected_csv:
        raise ValueError(f"expected {expected_csv} CSV files, found {len(csv_paths)}")

    rows = [_audit_file(path, dataset_root, protocol) for path in csv_paths]
    candidates = [row for row in rows if row["augmented_layout_candidate"]]
    expected_augmented = int(protocol["dataset"]["expected_augmented_layout_recordings"])
    complete = [row for row in candidates if row["cloth_labels_complete"]]

    strata_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        strata_map.setdefault((row["category"], row["material"]), []).append(row)
    strata = [
        {
            "category": category,
            "material": material,
            "count": len(values),
            "complete_cloth_support_count": sum(
                bool(value["cloth_labels_complete"]) for value in values
            ),
            "marker_counts": sorted({int(value["marker_count"]) for value in values}),
            "extra_marker_label_sets": sorted(
                {
                    tuple(value["extra_marker_labels"])
                    for value in values
                }
            ),
        }
        for (category, material), values in sorted(strata_map.items())
    ]

    status = (
        "header-audit-complete"
        if len(candidates) == expected_augmented
        else "header-audit-support-negative"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "source_revision": source_revision,
        "inventory": {
            "csv_file_count": len(rows),
            "augmented_candidate_count": len(candidates),
            "complete_cloth_support_count": len(complete),
            "candidate_by_category": {
                category: sum(row["category"] == category for row in candidates)
                for category in protocol["metadata"]["categories"]
            },
            "candidate_by_material": {
                material: sum(row["material"] == material for row in candidates)
                for material in protocol["metadata"]["materials"]
            },
            "candidate_marker_counts": sorted(
                {int(row["marker_count"]) for row in candidates}
            ),
        },
        "strata": strata,
        "candidates": candidates,
        "information_boundary": {
            "relative_path_metadata_read": True,
            "motive_header_layout_read": True,
            "marker_trajectory_values_parsed": False,
            "trajectory_hashes_computed": False,
            "scientific_threshold_selected": False,
            "scientific_utility_evaluated": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result_bytes = _canonical_bytes(result)
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "prob4d.tracking-cloth-augmented-header-audit-manifest.v1",
        "source_revision": source_revision,
        "protocol_sha256": _sha256_file(protocol_path),
        "result_sha256": _sha256_bytes(result_bytes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "marker_trajectory_values_parsed": False,
        "raw_trajectory_payload_copied": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root is unavailable: {args.dataset_root}")
    result = run(
        args.dataset_root.resolve(),
        args.protocol.resolve(),
        args.output_dir.resolve(),
        args.source_revision,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0 if result["status"] in {
        "header-audit-complete",
        "header-audit-support-negative",
    } else 3


if __name__ == "__main__":
    raise SystemExit(main())
