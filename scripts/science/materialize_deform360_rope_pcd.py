#!/usr/bin/env python3
"""Safely materialize the local packed Deform360 rope point-cloud subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


class MaterializationError(RuntimeError):
    """The mounted release does not satisfy the frozen archive contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path(root: Path) -> Path:
    candidates = (
        root / "raw-repository" / "raw" / "001-rope" / "metadata.json",
        root / "raw" / "001-rope" / "metadata.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MaterializationError("001-rope metadata.json is absent")


def archive_path(root: Path, episode: int) -> Path:
    candidates = (
        root
        / "processed-repository"
        / "processed"
        / "001-rope"
        / f"episode_{episode}"
        / "pcd_clean.tar",
        root
        / "processed"
        / "001-rope"
        / f"episode_{episode}"
        / "pcd_clean.tar",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MaterializationError(f"episode {episode}: pcd_clean.tar is absent")


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    selected: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    for member in archive.getmembers():
        if not member.isfile():
            continue
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise MaterializationError(f"unsafe tar member {member.name!r}")
        if path.suffix.lower() != ".npz":
            continue
        name = path.name
        if not name[:-4].isdigit() or len(name[:-4]) != 6:
            continue
        if name in seen:
            raise MaterializationError(f"duplicate point-cloud member {name}")
        seen.add(name)
        selected.append(member)
    selected.sort(key=lambda member: PurePosixPath(member.name).name)
    if len(selected) < 20:
        raise MaterializationError(
            f"archive contains only {len(selected)} six-digit NPZ frames"
        )
    return selected


def materialize(dataset_root: Path, output_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    if output_root.exists():
        raise MaterializationError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    source_metadata = metadata_path(dataset_root)
    target_metadata = (
        output_root / "raw-repository" / "raw" / "001-rope" / "metadata.json"
    )
    target_metadata.parent.mkdir(parents=True)
    shutil.copyfile(source_metadata, target_metadata)

    episode_records: list[dict[str, Any]] = []
    for episode in range(10):
        source = archive_path(dataset_root, episode)
        target = (
            output_root
            / "processed"
            / "001-rope"
            / f"episode_{episode}"
            / "pcd_clean"
        )
        target.mkdir(parents=True)
        extracted: list[dict[str, Any]] = []
        with tarfile.open(source, mode="r:*") as archive:
            members = safe_members(archive)
            for member in members:
                name = PurePosixPath(member.name).name
                destination = target / name
                stream = archive.extractfile(member)
                if stream is None:
                    raise MaterializationError(f"cannot read tar member {member.name}")
                digest = hashlib.sha256()
                size = 0
                with destination.open("xb") as output:
                    while True:
                        block = stream.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        size += len(block)
                        output.write(block)
                if size != member.size:
                    raise MaterializationError(
                        f"member size mismatch for {member.name}: {size} != {member.size}"
                    )
                extracted.append(
                    {"name": name, "bytes": size, "sha256": digest.hexdigest()}
                )
        episode_records.append(
            {
                "episode": episode,
                "archive": str(source.relative_to(dataset_root)),
                "archive_bytes": source.stat().st_size,
                "archive_sha256": sha256_file(source),
                "extracted_members": len(extracted),
                "extracted_bytes": sum(item["bytes"] for item in extracted),
                "first_member": extracted[0],
                "last_member": extracted[-1],
            }
        )

    record: dict[str, Any] = {
        "schema": "prob4d.deform360-rope-pcd-materialization-v1",
        "dataset_root": str(dataset_root),
        "metadata_source": str(source_metadata.relative_to(dataset_root)),
        "metadata_sha256": sha256_file(source_metadata),
        "output_layout": "processed/001-rope/episode_<0..9>/pcd_clean/<six-digit>.npz",
        "episodes": episode_records,
        "source_data_mutated": False,
        "target_outcomes_scored": False,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["materialization_id"] = hashlib.sha256(canonical).hexdigest()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    record = materialize(args.dataset_root, args.output_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
