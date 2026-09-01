#!/usr/bin/env python3
"""Materialize exact official DOT archives without opening their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


_MD5 = re.compile(r"[0-9a-f]{32}")


def _parse_entry(raw: str) -> tuple[str, str]:
    try:
        name, digest = raw.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entry must be NAME=MD5") from exc
    if not name or name != Path(name).name or "/" in name or "\\" in name:
        raise argparse.ArgumentTypeError("archive name must be a plain filename")
    digest = digest.lower()
    if _MD5.fullmatch(digest) is None:
        raise argparse.ArgumentTypeError("archive digest must be a lowercase MD5")
    return name, digest


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Prob4D-verified-DOT-archive-materializer/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if value.get("status") != "OK":
        raise RuntimeError("DOT metadata endpoint did not return status OK")
    return value


def _resolve(
    metadata: dict[str, Any], entries: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    files = metadata["data"]["latestVersion"]["files"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for entry in files:
        data_file = entry["dataFile"]
        by_name.setdefault(str(data_file["filename"]), []).append(data_file)

    resolved: list[dict[str, Any]] = []
    for name, expected_md5 in entries:
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise RuntimeError(f"official DOT archive identity is ambiguous for {name}")
        data_file = matches[0]
        checksum = data_file.get("checksum") or {}
        if checksum.get("type") != "MD5":
            raise RuntimeError(f"official checksum type changed for {name}")
        if str(checksum.get("value", "")).lower() != expected_md5:
            raise RuntimeError(f"official checksum changed for {name}")
        byte_count = int(data_file["filesize"])
        if byte_count <= 0:
            raise RuntimeError(f"official byte count is invalid for {name}")
        resolved.append(
            {
                "name": name,
                "md5": expected_md5,
                "bytes": byte_count,
                "datafile_id": int(data_file["id"]),
            }
        )
    return resolved


def _materialize(
    *, output_dir: Path, datafile_api_root: str, value: dict[str, Any]
) -> dict[str, Any]:
    name = str(value["name"])
    expected_md5 = str(value["md5"])
    expected_bytes = int(value["bytes"])
    destination = output_dir / name

    valid = (
        destination.is_file()
        and destination.stat().st_size == expected_bytes
        and _md5(destination) == expected_md5
    )
    if not valid:
        part = output_dir / f"{name}.part"
        part.touch(exist_ok=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "12",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                os.fspath(part),
                f"{datafile_api_root.rstrip('/')}/{value['datafile_id']}",
            ],
            check=True,
        )
        if part.stat().st_size != expected_bytes:
            raise RuntimeError(f"downloaded byte count mismatch for {name}")
        measured = _md5(part)
        if measured != expected_md5:
            raise RuntimeError(f"downloaded checksum mismatch for {name}")
        os.replace(part, destination)

    measured = _md5(destination)
    if destination.stat().st_size != expected_bytes or measured != expected_md5:
        raise RuntimeError(f"materialized archive verification failed for {name}")
    return {
        **value,
        "path": os.fspath(destination),
        "measured_md5": measured,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-api", required=True)
    parser.add_argument("--datafile-api-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entry", action="append", type=_parse_entry, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    entries = list(args.entry)
    if len({name for name, _ in entries}) != len(entries):
        raise RuntimeError("duplicate DOT archive entry")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.is_symlink():
        raise RuntimeError("archive output directory must not be a symlink")

    resolved = _resolve(_metadata(args.dataset_api), entries)
    materialized = [
        _materialize(
            output_dir=args.output_dir,
            datafile_api_root=args.datafile_api_root,
            value=value,
        )
        for value in resolved
    ]
    receipt = {
        "schema": "prob4d.verified-dot-archive-materialization",
        "schema_version": 1,
        "archives": materialized,
        "information_boundary": {
            "compressed_archive_bytes_read_for_hashing": True,
            "archive_members_enumerated": False,
            "archives_extracted": False,
            "normal_view_images_opened": False,
            "two_dimensional_markers_opened": False,
            "three_dimensional_markers_opened": False,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
