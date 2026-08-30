#!/usr/bin/env python3
"""Target-closed Deform360 representation audit for the gpuserver4090 mount."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "prob4d.deform360-query-carrier-audit"
SCHEMA_VERSION = 1
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_FILES_PER_UNIT = 25000

ROSTER = (
    ("038-mat-cloth", 3, 9, "sheet"),
    ("038-black-bag-cloth", 8, 5, "sheet"),
    ("087-plastic-bag-blue-cloth", 8, 3, "sheet"),
    ("144-jar-opener-cloth", 6, 2, "sheet"),
    ("072-cotton-clohesline", 2, 0, "volumetric"),
    ("102-stress-ball", 8, 6, "volumetric"),
    ("053-squeezer", 6, 4, "volumetric"),
    ("063-flower", 9, 7, "volumetric"),
)

GEOMETRY_TOKENS = (
    "track", "trajectory", "point", "cloud", "mesh", "vertex", "particle",
    "control", "deform", "world", "clean", "depth", "robot", "tactile",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("invalid NPY magic")
        major, minor = struct.unpack("BB", handle.read(2))
        if major == 1:
            length = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            length = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"unsupported NPY version {major}.{minor}")
        if length > 1_000_000:
            raise ValueError("unreasonable NPY header")
        text = handle.read(length).decode("latin1")
    import ast

    header = ast.literal_eval(text)
    shape = header.get("shape")
    if not isinstance(shape, tuple) or not all(type(v) is int and v >= 0 for v in shape):
        raise ValueError("invalid NPY shape")
    return {
        "container": "npy",
        "version": [major, minor],
        "shape": list(shape),
        "dtype": str(header.get("descr")),
        "fortran_order": bool(header.get("fortran_order")),
        "values_opened": False,
    }


def zip_npz_headers(path: Path) -> list[dict[str, Any]]:
    import zipfile

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if not info.filename.endswith(".npy"):
                records.append({"member": info.filename, "kind": "non-array", "bytes": info.file_size})
                continue
            if info.file_size > 5_000_000_000:
                records.append({"member": info.filename, "kind": "array", "error": "member-too-large"})
                continue
            with archive.open(info) as handle:
                magic = handle.read(6)
                if magic != b"\x93NUMPY":
                    records.append({"member": info.filename, "kind": "array", "error": "invalid-magic"})
                    continue
                major, minor = struct.unpack("BB", handle.read(2))
                length = struct.unpack("<H", handle.read(2))[0] if major == 1 else struct.unpack("<I", handle.read(4))[0]
                if length > 1_000_000:
                    records.append({"member": info.filename, "kind": "array", "error": "header-too-large"})
                    continue
                import ast

                header = ast.literal_eval(handle.read(length).decode("latin1"))
                shape = header.get("shape")
                records.append({
                    "member": info.filename,
                    "kind": "array",
                    "shape": list(shape) if isinstance(shape, tuple) else None,
                    "dtype": str(header.get("descr")),
                    "fortran_order": bool(header.get("fortran_order")),
                    "compressed_bytes": info.compress_size,
                    "bytes": info.file_size,
                    "values_opened": False,
                })
    return records


def inspect_json(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        return {"container": "json", "error": "too-large", "bytes": size}
    raw = path.read_bytes()
    value = json.loads(raw)
    summary: dict[str, Any] = {
        "container": "json",
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "top_level_type": type(value).__name__,
    }
    if isinstance(value, dict):
        summary["keys"] = sorted(map(str, value.keys()))[:200]
    elif isinstance(value, list):
        summary["length"] = len(value)
    return summary


def candidate_score(relative: str) -> int:
    lower = relative.casefold()
    return sum(token in lower for token in GEOMETRY_TOKENS)


def unit_candidates(root: Path, object_id: str, episode: int) -> dict[str, Any]:
    object_hits: list[Path] = []
    needles = (object_id.casefold(), f"/{episode}/", f"episode_{episode}", f"episode-{episode}")
    for base, dirnames, filenames in os.walk(root, followlinks=False):
        base_path = Path(base)
        dirnames[:] = sorted(
            name for name in dirnames
            if not name.startswith(".") and not (base_path / name).is_symlink()
        )
        base_lower = base.casefold().replace("\\", "/")
        if object_id.casefold() not in base_lower:
            continue
        for name in sorted(filenames):
            path = base_path / name
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                continue
            relative = safe_relative(path, root)
            lower = "/" + relative.casefold().replace("\\", "/") + "/"
            episode_match = any(needle in lower for needle in needles[1:])
            if episode_match or path.name == "metadata.json":
                object_hits.append(path)
            if len(object_hits) >= MAX_FILES_PER_UNIT:
                break
        if len(object_hits) >= MAX_FILES_PER_UNIT:
            break

    records: list[dict[str, Any]] = []
    extensions: Counter[str] = Counter()
    for path in sorted(set(object_hits), key=lambda item: safe_relative(item, root)):
        relative = safe_relative(path, root)
        suffix = path.suffix.casefold() or "<none>"
        extensions[suffix] += 1
        record: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "suffix": suffix,
            "candidate_score": candidate_score(relative),
        }
        try:
            if suffix == ".npy":
                record["header"] = npy_header(path)
            elif suffix == ".npz":
                record["members"] = zip_npz_headers(path)[:1000]
            elif suffix == ".json":
                record["summary"] = inspect_json(path)
            elif suffix in {".pkl", ".pickle", ".h5", ".hdf5", ".ply", ".pcd", ".obj", ".mp4", ".avi", ".mov"}:
                record["inspection"] = "listed-only"
        except Exception as exc:  # bounded diagnostic; no recovery by payload values
            record["inspection_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    ranked = sorted(records, key=lambda record: (-record["candidate_score"], record["path"]))
    return {
        "object_id": object_id,
        "episode": episode,
        "files": len(records),
        "truncated": len(object_hits) >= MAX_FILES_PER_UNIT,
        "extensions": dict(sorted(extensions.items())),
        "ranked_candidates": ranked[:400],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root
    if not root.is_dir() or root.is_symlink() or not os.access(root, os.R_OK):
        raise SystemExit(f"invalid readable non-symlink root: {root}")

    units: list[dict[str, Any]] = []
    for object_id, source_episode, target_episode, stratum in ROSTER:
        for role, episode in (("source", source_episode), ("target", target_episode)):
            unit = unit_candidates(root, object_id, episode)
            unit.update({"role": role, "stratum": stratum})
            units.append(unit)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "runner_name": os.environ.get("RUNNER_NAME", "unknown"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "github_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "access_boundary": {
            "directory_and_file_names_opened": True,
            "file_sizes_opened": True,
            "small_json_opened": True,
            "npy_npz_headers_opened": True,
            "array_values_opened": False,
            "pickle_values_opened": False,
            "hdf5_values_opened": False,
            "media_decoded": False,
            "target_scored": False,
            "dataset_mutated": False,
        },
        "units": units,
    }
    record["audit_id"] = sha256_bytes(canonical_json(record))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Deform360 query-carrier audit v1",
        "",
        f"Audit ID: `{record['audit_id']}`",
        "",
        "| Role | Object | Episode | Files | Top candidate |",
        "|---|---|---:|---:|---|",
    ]
    for unit in units:
        top = unit["ranked_candidates"][0]["path"] if unit["ranked_candidates"] else "—"
        lines.append(f"| {unit['role']} | `{unit['object_id']}` | {unit['episode']} | {unit['files']} | `{top}` |")
    lines.extend([
        "",
        "This audit opens small JSON metadata and NPY/NPZ headers only. It does not open numerical arrays, decode media, or score target outcomes.",
    ])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit_id": record["audit_id"], "units": len(units)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
