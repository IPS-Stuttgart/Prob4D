#!/usr/bin/env python3
"""Inspect the official DOT rope archives without extracting dataset payloads.

The command is intentionally read-only and standard-library-only. It verifies
the expected archive roster, rejects unsafe ZIP member paths, summarizes layout
and extensions, previews small metadata files, and records NumPy array headers
where possible. Publisher checksums are assumed to have been verified before
this source-only preflight; the command does not redundantly decompress all 9 GB.
It does not score truth, run a provider, or select an evaluation cohort.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import stat
import struct
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_ARCHIVES = (
    "R01-10.zip",
    "R11-20.zip",
    "R21-30.zip",
    "R31-40.zip",
    "R41-50.zip",
    "R51-60.zip",
    "R61-70.zip",
)
TEXT_SUFFIXES = {
    ".txt",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".xml",
    ".ini",
    ".cfg",
    ".conf",
    ".obj",
    ".ply",
}
CANDIDATE_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:3d|gt|ground|truth|joint|marker|model|mesh|point|cloud|"
    r"correspond|template|track|calib|camera|uv|visible|rgb)(?:$|[/_.-])",
    re.IGNORECASE,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\x00" not in name


def _extension(name: str) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    return suffix if suffix else "<none>"


def _top_prefix(name: str, depth: int = 3) -> str:
    parts = PurePosixPath(name).parts
    return "/".join(parts[:depth]) if parts else "<empty>"


def _preview_text(raw: bytes, limit: int = 8192) -> dict[str, Any]:
    sample = raw[:limit]
    encoding = "latin-1"
    text = sample.decode(encoding)
    for candidate in ("utf-8", "utf-8-sig"):
        try:
            text = sample.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    return {
        "encoding": encoding,
        "truncated": len(raw) > limit,
        "preview": text.replace("\x00", "\\0"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _npy_header(raw: bytes) -> dict[str, Any]:
    """Parse an NPY header without importing NumPy or loading an array."""
    stream = io.BytesIO(raw)
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    version = tuple(stream.read(2))
    if version == (1, 0):
        length_raw = stream.read(2)
        if len(length_raw) != 2:
            raise ValueError("truncated NPY v1 header length")
        header_length = struct.unpack("<H", length_raw)[0]
        encoding = "latin-1"
    elif version in {(2, 0), (3, 0)}:
        length_raw = stream.read(4)
        if len(length_raw) != 4:
            raise ValueError("truncated NPY v2/v3 header length")
        header_length = struct.unpack("<I", length_raw)[0]
        encoding = "utf-8" if version == (3, 0) else "latin-1"
    else:
        raise ValueError(f"unsupported NPY version {version}")
    header_raw = stream.read(header_length)
    if len(header_raw) != header_length:
        raise ValueError("truncated NPY header")
    header = ast.literal_eval(header_raw.decode(encoding).strip())
    if not isinstance(header, dict):
        raise ValueError("NPY header is not a dictionary")
    if set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError(f"unexpected NPY header fields: {sorted(header)}")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(type(value) is not int or value < 0 for value in shape):
        raise ValueError("invalid NPY shape")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("invalid NPY fortran_order")
    return {
        "version": list(version),
        "shape": list(shape),
        "fortran_order": header["fortran_order"],
        "dtype_descriptor": header["descr"],
        "header_bytes": stream.tell(),
    }


def inspect_archives(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    logical_root = dataset_root
    resolved_root = dataset_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("dataset root must resolve to a directory")

    actual = tuple(sorted(path.name for path in resolved_root.glob("*.zip")))
    if actual != EXPECTED_ARCHIVES:
        raise ValueError(
            f"archive roster mismatch: expected {EXPECTED_ARCHIVES!r}, found {actual!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    archive_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    text_previews: list[dict[str, Any]] = []
    array_headers: list[dict[str, Any]] = []
    global_extensions: Counter[str] = Counter()
    global_prefixes: Counter[str] = Counter()
    total_uncompressed = 0
    total_compressed = 0

    for archive_name in EXPECTED_ARCHIVES:
        archive_path = resolved_root / archive_name
        archive_stat = archive_path.stat()
        if not stat.S_ISREG(archive_stat.st_mode):
            raise ValueError(f"{archive_name} is not a regular file")
        extension_counts: Counter[str] = Counter()
        prefix_counts: Counter[str] = Counter()
        archive_uncompressed = 0
        archive_compressed = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            for info in members:
                if not _safe_member(info.filename):
                    raise ValueError(f"unsafe ZIP member path: {archive_name}:{info.filename}")
                is_directory = info.is_dir()
                suffix = _extension(info.filename)
                prefix = _top_prefix(info.filename)
                if not is_directory:
                    extension_counts[suffix] += 1
                    prefix_counts[prefix] += 1
                    global_extensions[suffix] += 1
                    global_prefixes[prefix] += 1
                archive_uncompressed += info.file_size
                archive_compressed += info.compress_size
                member_rows.append(
                    {
                        "archive": archive_name,
                        "name": info.filename,
                        "is_directory": is_directory,
                        "uncompressed_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "extension": suffix,
                    }
                )
                if is_directory:
                    continue
                if CANDIDATE_PATTERN.search(info.filename):
                    candidates.append(
                        {
                            "archive": archive_name,
                            "name": info.filename,
                            "uncompressed_bytes": info.file_size,
                            "extension": suffix,
                        }
                    )
                if (
                    suffix in TEXT_SUFFIXES
                    and info.file_size <= 2_000_000
                    and len(text_previews) < 200
                ):
                    raw = archive.read(info)
                    text_previews.append(
                        {
                            "archive": archive_name,
                            "name": info.filename,
                            "uncompressed_bytes": info.file_size,
                            **_preview_text(raw),
                        }
                    )
                if suffix == ".npy" and info.file_size <= 512_000_000 and len(array_headers) < 500:
                    with archive.open(info, "r") as stream:
                        raw = stream.read(min(info.file_size, 1_048_576))
                    try:
                        header = _npy_header(raw)
                    except Exception as exc:  # retain malformed/unsupported header evidence
                        header = {"error": f"{type(exc).__name__}: {exc}"}
                    array_headers.append(
                        {
                            "archive": archive_name,
                            "name": info.filename,
                            "uncompressed_bytes": info.file_size,
                            **header,
                        }
                    )

        total_uncompressed += archive_uncompressed
        total_compressed += archive_compressed
        archive_rows.append(
            {
                "archive": archive_name,
                "filesystem_bytes": archive_stat.st_size,
                "mtime_ns": archive_stat.st_mtime_ns,
                "member_count": len(members),
                "file_count": sum(not item.is_dir() for item in members),
                "directory_count": sum(item.is_dir() for item in members),
                "uncompressed_bytes": archive_uncompressed,
                "compressed_member_bytes": archive_compressed,
                "extensions": dict(extension_counts.most_common()),
                "top_prefixes": dict(prefix_counts.most_common(40)),
            }
        )

    with (output_dir / "members.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(member_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(member_rows)

    for filename, payload in (
        ("candidate_members.json", candidates),
        ("text_previews.json", text_previews),
        ("array_headers.json", array_headers),
    ):
        (output_dir / filename).write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    report: dict[str, Any] = {
        "schema": "prob4d.dot-rope-archive-inventory.v1",
        "evidence_kind": "public-dataset-layout-preflight",
        "dataset": {
            "doi": "10.13021/ORC2020/XXLVXM",
            "logical_root": os.fspath(logical_root),
            "resolved_root": os.fspath(resolved_root),
            "expected_archives": list(EXPECTED_ARCHIVES),
            "filesystem_bytes": sum(row["filesystem_bytes"] for row in archive_rows),
            "uncompressed_bytes": total_uncompressed,
            "compressed_member_bytes": total_compressed,
        },
        "archives": archive_rows,
        "global_extensions": dict(global_extensions.most_common()),
        "global_top_prefixes": dict(global_prefixes.most_common(100)),
        "candidate_member_count": len(candidates),
        "text_preview_count": len(text_previews),
        "array_header_count": len(array_headers),
        "information_boundary": {
            "archive_payload_extracted": False,
            "provider_executed": False,
            "ground_truth_scored": False,
            "method_selected_from_outcomes": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
    }
    report["artifact_id"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    (output_dir / "inventory.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_archives(args.dataset_root, args.output_dir)
    print(json.dumps({"artifact_id": report["artifact_id"], "dataset": report["dataset"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
