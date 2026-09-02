#!/usr/bin/env python3
"""Inventory a fresh Tracking Cloth cohort without opening trajectory values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile


SCHEMA = "prob4d.tracking-cloth-orbit-tube-roster-audit.v1"


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return member


def _find_archive(root: Path, expected_name: str) -> Path:
    direct = root / expected_name
    candidates = [direct] if direct.is_file() else []
    candidates.extend(path for path in root.rglob(expected_name) if path.is_file())
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one {expected_name!r} below {root}, found {len(unique)}"
        )
    return unique[0]


def _looks_like_data_row(row: list[str]) -> bool:
    if len(row) < 2:
        return False
    try:
        int(row[0].strip())
        float(row[1].strip())
    except ValueError:
        return False
    return True


def _normal(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _label_from_triplet(rows: list[list[str]], coordinate_row: int, column: int) -> str:
    forbidden = {
        "",
        "frame",
        "marker",
        "markers",
        "name",
        "position",
        "quality",
        "time",
        "time (seconds)",
        "type",
        "x",
        "y",
        "z",
    }
    for row_index in range(coordinate_row - 1, -1, -1):
        row = rows[row_index]
        cells = [row[index].strip() if index < len(row) else "" for index in range(column, column + 3)]
        nonempty = [cell for cell in cells if cell]
        if not nonempty:
            continue
        if len(set(nonempty)) == 1:
            candidate = nonempty[0]
        elif cells[0] and not cells[1] and not cells[2]:
            candidate = cells[0]
        else:
            continue
        normalized = _normal(candidate)
        if normalized in forbidden or "unlabeled" in normalized:
            continue
        if ":" in candidate:
            candidate = candidate.rsplit(":", 1)[-1].strip()
        if candidate:
            return candidate
    raise ValueError(f"no marker label found for coordinate triplet at column {column}")


def _parse_header_labels(header: bytes, expected_lines: int) -> tuple[str, ...]:
    text = header.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != expected_lines:
        raise ValueError(f"expected {expected_lines} header rows, found {len(rows)}")
    if any(_looks_like_data_row(row) for row in rows):
        raise ValueError("configured header boundary includes a trajectory data row")

    triplets_by_row: list[tuple[int, list[int]]] = []
    for row_index, row in enumerate(rows):
        starts: list[int] = []
        for column in range(max(0, len(row) - 2)):
            coordinates = tuple(_normal(value) for value in row[column : column + 3])
            if coordinates == ("x", "y", "z"):
                starts.append(column)
        if starts:
            triplets_by_row.append((row_index, starts))
    if not triplets_by_row:
        raise ValueError("no X/Y/Z marker-coordinate row found in the sealed header")

    coordinate_row, starts = max(triplets_by_row, key=lambda item: len(item[1]))
    labels = {_label_from_triplet(rows, coordinate_row, column) for column in starts}
    if len(labels) < 3:
        raise ValueError("fewer than three marker identities found in the header")
    return tuple(sorted(labels, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)))


def _read_fixed_header(
    archive: zipfile.ZipFile,
    member: str,
    *,
    line_count: int,
    maximum_line_bytes: int,
) -> bytes:
    lines: list[bytes] = []
    with archive.open(member, "r") as stream:
        for line_index in range(line_count):
            line = stream.readline(maximum_line_bytes + 1)
            if not line:
                raise ValueError(f"{member}: ended before header line {line_index + 1}")
            if len(line) > maximum_line_bytes:
                raise ValueError(f"{member}: header line exceeds the byte limit")
            if not line.endswith((b"\n", b"\r")) and line_index + 1 < line_count:
                raise ValueError(f"{member}: truncated header line")
            lines.append(line)
    return b"".join(lines)


def _intersection(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    common = set(records[0]["marker_labels"])
    for record in records[1:]:
        common.intersection_update(record["marker_labels"])
    return sorted(
        common,
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )


def run_audit(protocol: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    dataset = protocol["dataset"]
    archive_path = _find_archive(dataset_root, dataset["archive_filename"])
    md5 = _hash_file(archive_path, "md5")
    sha256 = _hash_file(archive_path, "sha256")
    if md5 != dataset["archive_md5"]:
        raise ValueError("Tracking Cloth archive MD5 mismatch")
    if sha256 != dataset["archive_sha256"]:
        raise ValueError("Tracking Cloth archive SHA-256 mismatch")

    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP integrity failure at {bad}")
        csv_members = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".csv")
        )
        for member in csv_members:
            _safe_member(member)
        if len(csv_members) != dataset["expected_csv_files"]:
            raise ValueError("unexpected total Tracking Cloth CSV count")

        token = dataset["fresh_category_token"].casefold()
        fresh_members = [
            member
            for member in csv_members
            if token in PurePosixPath(member).as_posix().casefold()
        ]
        if len(fresh_members) != dataset["expected_fresh_files"]:
            raise ValueError("unexpected fresh self-collision recording count")

        salt = protocol["split"]["sha256_salt"]
        ordered = sorted(
            fresh_members,
            key=lambda member: hashlib.sha256(f"{salt}|{member}".encode()).hexdigest(),
        )
        source_count = int(protocol["split"]["source_count"])
        calibration_count = int(protocol["split"]["calibration_count"])
        target_count = int(protocol["split"]["target_count"])
        if source_count + calibration_count + target_count != len(ordered):
            raise ValueError("split counts do not cover the fresh roster exactly")
        roles = {
            **{member: "source" for member in ordered[:source_count]},
            **{
                member: "calibration"
                for member in ordered[source_count : source_count + calibration_count]
            },
            **{member: "target" for member in ordered[-target_count:]},
        }

        records: list[dict[str, Any]] = []
        header_lines = int(protocol["header_audit"]["line_count"])
        maximum_line_bytes = int(protocol["header_audit"]["maximum_line_bytes"])
        for member in ordered:
            header = _read_fixed_header(
                archive,
                member,
                line_count=header_lines,
                maximum_line_bytes=maximum_line_bytes,
            )
            labels = _parse_header_labels(header, header_lines)
            records.append(
                {
                    "path": member,
                    "path_sha256": hashlib.sha256(member.encode()).hexdigest(),
                    "role": roles[member],
                    "header_sha256": hashlib.sha256(header).hexdigest(),
                    "header_line_count": header_lines,
                    "marker_labels": labels,
                    "trajectory_rows_read": 0,
                }
            )

    by_role = {
        role: [record for record in records if record["role"] == role]
        for role in ("source", "calibration", "target")
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "archive": {
            "filename": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "md5": md5,
            "sha256": sha256,
            "zip_integrity": "pass",
        },
        "counts": {
            "all_csv": len(csv_members),
            "fresh": len(records),
            "source": len(by_role["source"]),
            "calibration": len(by_role["calibration"]),
            "target": len(by_role["target"]),
        },
        "marker_intersections": {
            role: _intersection(role_records)
            for role, role_records in by_role.items()
        }
        | {"all_roles": _intersection(records)},
        "records": records,
        "information_boundary": {
            "zip_directory_opened": True,
            "fixed_header_rows_opened": True,
            "raw_header_text_published": False,
            "trajectory_rows_read": 0,
            "source_trajectory_values_opened": False,
            "calibration_trajectory_values_opened": False,
            "target_trajectory_values_opened": False,
            "split_fixed_before_header_read": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    canonical = json.dumps(
        result,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result["result_id"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/tracking-cloth-orbit-tube-roster-audit-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = run_audit(protocol, args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
