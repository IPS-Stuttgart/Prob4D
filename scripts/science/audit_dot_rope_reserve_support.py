#!/usr/bin/env python3
"""Freeze a DOT R11-R70 camera map using only 2-D marker visibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "prob4d.dot-rope-reserve-support-audit.v1"
RESULT_SCHEMA = "prob4d.dot-rope-reserve-support-audit-result.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    expected = {
        "schema",
        "evidence_kind",
        "dataset",
        "visibility_rule",
        "camera_selection_rule",
        "next_stage_boundary",
        "information_boundary",
        "registered_outputs",
        "claim_boundary",
    }
    if set(protocol) != expected or protocol["schema"] != SCHEMA:
        raise ValueError("protocol schema or fields changed")
    if protocol["evidence_kind"] != ("outcome-blind-two-dimensional-support-qualification"):
        raise ValueError("evidence kind changed")
    dataset = protocol["dataset"]
    if dataset != {
        "persistent_id": "doi:10.13021/ORC2020/XXLVXM",
        "archive_names": [
            "R11-20.zip",
            "R21-30.zip",
            "R31-40.zip",
            "R41-50.zip",
            "R51-60.zip",
            "R61-70.zip",
        ],
        "sequence_start": 11,
        "sequence_stop": 70,
        "cameras": [f"cam{index:03d}" for index in range(1, 11)],
        "frames": list(range(1, 8)),
        "coordinate_member_template": ("{sequence}/coordinates/2d/frame{frame:06d}_{camera}.txt"),
        "coordinate_columns": [0, 1],
    }:
        raise ValueError("dataset contract changed")
    if protocol["visibility_rule"] != {
        "finite_coordinates_required": True,
        "nonnegative_coordinates_required": True,
        "identity_definition": "stable row index within publisher 2-D coordinate files",
        "common_support_scope": "intersection across all seven registered frames",
    }:
        raise ValueError("visibility rule changed")
    if protocol["camera_selection_rule"] != {
        "order": [
            "maximum common visible marker count across all seven frames",
            "maximum minimum per-frame visible marker count",
            "maximum mean per-frame visible marker count",
            "lowest camera index",
        ],
        "minimum_common_visible_markers": 8,
        "minimum_per_frame_visible_markers": 8,
        "unsupported_sequences_replaced": False,
    }:
        raise ValueError("camera selection rule changed")
    boundary = protocol["information_boundary"]
    if boundary != {
        "zip_member_names_visible": True,
        "two_dimensional_coordinate_values_opened": True,
        "three_dimensional_coordinate_values_opened": False,
        "normal_view_images_opened": False,
        "uv_view_images_opened": False,
        "learned_provider_executed": False,
        "provider_residuals_or_uncertainty_scores_opened": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }:
        raise ValueError("information boundary changed")
    if not isinstance(protocol["claim_boundary"], str) or not protocol["claim_boundary"].strip():
        raise ValueError("claim boundary is empty")
    return protocol


def _archive_for_sequence(sequence_number: int) -> str:
    start = ((sequence_number - 1) // 10) * 10 + 1
    return f"R{start:02d}-{start + 9:02d}.zip"


def _parse_coordinates(raw: bytes, *, member: str) -> list[tuple[float, float]]:
    text = raw.decode("utf-8-sig")
    rows: list[tuple[float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.replace(",", " ").split()
        if len(tokens) < 2:
            raise ValueError(f"{member}:{line_number} has fewer than two columns")
        try:
            x = float(tokens[0])
            y = float(tokens[1])
        except ValueError as error:
            raise ValueError(f"{member}:{line_number} contains a nonnumeric coordinate") from error
        rows.append((x, y))
    if not rows:
        raise ValueError(f"{member} contains no coordinate rows")
    return rows


def _visible_indices(rows: list[tuple[float, float]]) -> set[int]:
    return {
        index
        for index, (x, y) in enumerate(rows)
        if math.isfinite(x) and math.isfinite(y) and x >= 0.0 and y >= 0.0
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archives(
    root: Path,
    metadata_path: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    metadata = _load_json(metadata_path)
    if metadata.get("dataset_persistent_id") != protocol["dataset"]["persistent_id"]:
        raise ValueError("metadata persistent identifier changed")
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("metadata files are missing")
    by_name = {
        row.get("filename"): row
        for row in files
        if isinstance(row, dict) and isinstance(row.get("filename"), str)
    }
    expected = protocol["dataset"]["archive_names"]
    if set(by_name) != set(expected):
        raise ValueError("official archive metadata roster changed")
    paths: dict[str, Path] = {}
    receipts: list[dict[str, Any]] = []
    for name in expected:
        row = by_name[name]
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"archive is missing or symlinked: {name}")
        byte_count = int(row["byte_count"])
        checksum = str(row["md5"]).lower()
        if path.stat().st_size != byte_count or _md5(path) != checksum:
            raise ValueError(f"archive identity changed: {name}")
        paths[name] = path
        receipts.append(
            {
                "filename": name,
                "datafile_id": int(row["datafile_id"]),
                "byte_count": byte_count,
                "md5": checksum,
                "sha256": _sha256(path),
            }
        )
    return paths, receipts


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# DOT R11-R70 outcome-blind support audit",
        "",
        f"Support ID: `{result['support_id']}`",
        "",
        f"Qualified sequences: **{len(result['qualified_sequences'])}/60**",
        "",
        f"Unsupported sequences: **{len(result['unsupported_sequences'])}/60**",
        "",
        "Only publisher 2-D marker-coordinate values were read. No 3-D truth, normal or UV image, provider prediction, residual, covariance score, or downstream outcome was opened.",
        "",
        "| Sequence | selected camera | common markers | minimum/frame | qualified |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["selected_cameras"]:
        lines.append(
            f"| {row['sequence']} | {row['selected_camera']} | "
            f"{row['common_visible_marker_count']} | "
            f"{row['minimum_frame_visible_marker_count']} | "
            f"{str(row['qualified']).lower()} |"
        )
    lines += ["", result["claim_boundary"], ""]
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.protocol)
    output = args.output_dir
    if output.exists():
        raise ValueError("output directory already exists")
    output.mkdir(parents=True)
    try:
        archives, archive_receipts = _verify_archives(
            args.dataset_root,
            args.official_metadata,
            protocol,
        )
        handles = {name: zipfile.ZipFile(path) for name, path in archives.items()}
        try:
            name_sets = {name: set(handle.namelist()) for name, handle in handles.items()}
            camera_rows: list[dict[str, Any]] = []
            selected_rows: list[dict[str, Any]] = []
            accessed_members: list[dict[str, Any]] = []
            minimum_common = int(
                protocol["camera_selection_rule"]["minimum_common_visible_markers"]
            )
            minimum_frame = int(
                protocol["camera_selection_rule"]["minimum_per_frame_visible_markers"]
            )
            template = protocol["dataset"]["coordinate_member_template"]
            for sequence_number in range(
                int(protocol["dataset"]["sequence_start"]),
                int(protocol["dataset"]["sequence_stop"]) + 1,
            ):
                sequence = f"R{sequence_number:02d}"
                archive_name = _archive_for_sequence(sequence_number)
                archive = handles[archive_name]
                names = name_sets[archive_name]
                sequence_camera_rows: list[dict[str, Any]] = []
                for camera in protocol["dataset"]["cameras"]:
                    visible_by_frame: list[set[int]] = []
                    frame_counts: list[int] = []
                    marker_row_counts: list[int] = []
                    for frame in protocol["dataset"]["frames"]:
                        member = template.format(
                            sequence=sequence,
                            frame=int(frame),
                            camera=camera,
                        )
                        if member not in names:
                            raise ValueError(f"registered 2-D member is missing: {member}")
                        raw = archive.read(member)
                        rows = _parse_coordinates(raw, member=member)
                        visible = _visible_indices(rows)
                        visible_by_frame.append(visible)
                        frame_counts.append(len(visible))
                        marker_row_counts.append(len(rows))
                        accessed_members.append(
                            {
                                "archive": archive_name,
                                "sequence": sequence,
                                "camera": camera,
                                "frame": int(frame),
                                "member": member,
                                "byte_count": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "coordinate_row_count": len(rows),
                                "visible_marker_count": len(visible),
                            }
                        )
                    common = set.intersection(*visible_by_frame)
                    row = {
                        "sequence": sequence,
                        "camera": camera,
                        "common_visible_marker_count": len(common),
                        "minimum_frame_visible_marker_count": min(frame_counts),
                        "mean_frame_visible_marker_count": float(
                            sum(frame_counts) / len(frame_counts)
                        ),
                        "maximum_frame_visible_marker_count": max(frame_counts),
                        "minimum_coordinate_row_count": min(marker_row_counts),
                        "maximum_coordinate_row_count": max(marker_row_counts),
                        "common_marker_indices": ";".join(str(index) for index in sorted(common)),
                    }
                    camera_rows.append(row)
                    sequence_camera_rows.append(row)
                selected = max(
                    sequence_camera_rows,
                    key=lambda row: (
                        row["common_visible_marker_count"],
                        row["minimum_frame_visible_marker_count"],
                        row["mean_frame_visible_marker_count"],
                        -int(str(row["camera"])[3:]),
                    ),
                )
                qualified = (
                    int(selected["common_visible_marker_count"]) >= minimum_common
                    and int(selected["minimum_frame_visible_marker_count"]) >= minimum_frame
                )
                selected_rows.append(
                    {
                        "sequence": sequence,
                        "archive": archive_name,
                        "selected_camera": selected["camera"],
                        "common_visible_marker_count": selected["common_visible_marker_count"],
                        "minimum_frame_visible_marker_count": selected[
                            "minimum_frame_visible_marker_count"
                        ],
                        "mean_frame_visible_marker_count": selected[
                            "mean_frame_visible_marker_count"
                        ],
                        "common_marker_indices": selected["common_marker_indices"],
                        "qualified": qualified,
                    }
                )
        finally:
            for handle in handles.values():
                handle.close()

        qualified = [row["sequence"] for row in selected_rows if row["qualified"]]
        unsupported = [row["sequence"] for row in selected_rows if not row["qualified"]]
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "evidence_kind": protocol["evidence_kind"],
            "protocol": protocol,
            "protocol_sha256": _content_id(protocol),
            "repository_revision": args.repository_revision,
            "official_archives": archive_receipts,
            "selected_cameras": selected_rows,
            "qualified_sequences": qualified,
            "unsupported_sequences": unsupported,
            "camera_row_count": len(camera_rows),
            "accessed_member_count": len(accessed_members),
            "information_boundary": protocol["information_boundary"],
            "next_stage_boundary": protocol["next_stage_boundary"],
            "claim_boundary": protocol["claim_boundary"],
        }
        result["support_id"] = _content_id(result)
        _write_json(output / "result.json", result)
        _write_json(
            output / "accessed-members.json",
            {
                "dataset_persistent_id": protocol["dataset"]["persistent_id"],
                "members": accessed_members,
                "three_dimensional_coordinate_values_opened": False,
                "normal_or_uv_images_opened": False,
            },
        )
        _write_csv(output / "camera-support.csv", camera_rows)
        _write_csv(output / "selected-cameras.csv", selected_rows)
        (output / "summary.md").write_text(_summary(result), encoding="utf-8")
        print(
            json.dumps(
                {
                    "support_id": result["support_id"],
                    "qualified_sequence_count": len(qualified),
                    "unsupported_sequence_count": len(unsupported),
                    "accessed_member_count": len(accessed_members),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        failure = {
            "schema": "prob4d.dot-rope-reserve-support-audit-technical-failure.v1",
            "protocol_sha256": _content_id(protocol),
            "repository_revision": args.repository_revision,
            "decision": "technical-failure",
            "failure": (f"{type(error).__name__}: {' '.join(str(error).split())}")[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
            "information_boundary": protocol["information_boundary"],
            "claim_boundary": protocol["claim_boundary"],
        }
        failure["technical_result_id"] = _content_id(failure)
        _write_json(output / "technical-failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-protocol")
    validate.add_argument("--protocol", type=Path, required=True)
    audit = subcommands.add_parser("audit")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--dataset-root", type=Path, required=True)
    audit.add_argument("--official-metadata", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--repository-revision", required=True)
    args = parser.parse_args()
    if args.command == "validate-protocol":
        protocol = _load_protocol(args.protocol)
        print(json.dumps({"protocol_sha256": _content_id(protocol)}))
        return 0
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
