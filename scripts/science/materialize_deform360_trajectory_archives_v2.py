#!/usr/bin/env python3
"""Safely materialize bounded Deform360 point-cloud trajectories from tar archives.

The protocol and object/episode roster are frozen before this script opens any
numerical point-cloud values. It never executes pickle payloads, never decodes
media, never mutates the mounted dataset, and emits a strict runtime audit for
an already-authorized evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

PROTOCOL_ID = "deform360-finite-orbit-real-data-v2"
MEMBER_RE = re.compile(r"(?:^|/)(\d{6})\.npz$")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", value)
    )


def safe_members(
    archive: tarfile.TarFile, maximum_member_bytes: int
) -> list[tarfile.TarInfo]:
    accepted: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    for member in archive.getmembers():
        name = member.name
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe archive member path: {name}")
        if name in seen:
            raise ValueError(f"duplicate archive member: {name}")
        seen.add(name)
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"unsupported archive link/device member: {name}")
        if not member.isfile():
            continue
        if member.size <= 0 or member.size > maximum_member_bytes:
            continue
        if MEMBER_RE.search(name):
            accepted.append(member)
    return sorted(accepted, key=lambda item: natural_key(item.name))


def evenly_spaced(
    items: list[tarfile.TarInfo], limit: int
) -> list[tarfile.TarInfo]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=int)
    unique = list(dict.fromkeys(int(index) for index in indices))
    return [items[index] for index in unique]


def load_points(raw: bytes, member_name: str, minimum_points: int) -> np.ndarray:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if "pts" in archive.files:
            points = np.asarray(archive["pts"], dtype=np.float32)
        else:
            candidates = []
            for key in sorted(archive.files):
                value = archive[key]
                if (
                    isinstance(value, np.ndarray)
                    and value.ndim == 2
                    and value.shape[1] == 3
                    and np.issubdtype(value.dtype, np.number)
                ):
                    candidates.append(np.asarray(value))
            if not candidates:
                raise ValueError(f"{member_name}: no numeric Nx3 point array")
            points = np.asarray(max(candidates, key=len), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_points:
        raise ValueError(f"{member_name}: invalid point-cloud shape {points.shape}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < minimum_points:
        raise ValueError(f"{member_name}: fewer than {minimum_points} finite points")
    return points


def materialize_unit(
    *,
    dataset_root: Path,
    output_root: Path,
    object_id: str,
    role: str,
    episode: int,
    maximum_frames: int,
    maximum_points: int,
    minimum_frames: int,
    minimum_points: int,
    maximum_member_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_path = (
        dataset_root
        / "processed-repository"
        / "processed"
        / object_id
        / f"episode_{episode}"
        / "pcd_clean.tar"
    )
    if not archive_path.is_file() or archive_path.is_symlink():
        raise FileNotFoundError(f"missing regular archive: {archive_path}")

    arrays: list[np.ndarray] = []
    member_records: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = evenly_spaced(
            safe_members(archive, maximum_member_bytes), maximum_frames
        )
        if len(members) < minimum_frames:
            raise ValueError(
                f"{object_id} episode {episode}: only {len(members)} admissible frames"
            )
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot extract regular member {member.name}")
            raw = extracted.read(maximum_member_bytes + 1)
            if len(raw) != member.size or len(raw) > maximum_member_bytes:
                raise ValueError(f"bounded read mismatch for {member.name}")
            points = load_points(raw, member.name, minimum_points)
            arrays.append(points)
            member_records.append(
                {
                    "name": member.name,
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                    "points": int(len(points)),
                }
            )

    retained_points = min(maximum_points, min(len(points) for points in arrays))
    if retained_points < minimum_points:
        raise ValueError(
            f"{object_id} episode {episode}: insufficient common point count"
        )
    sampled = []
    for points in arrays:
        indices = np.linspace(0, len(points) - 1, retained_points, dtype=int)
        sampled.append(points[indices])
    trajectory = np.stack(sampled).astype(np.float32, copy=False)
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{object_id} episode {episode}: nonfinite trajectory")

    relative = (
        Path("trajectories") / object_id / f"{role}_episode_{episode}.npz"
    )
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, points=trajectory)

    audit_unit = {
        "object_id": object_id,
        "role": role,
        "episode": episode,
        "episode_path": relative.parent.as_posix(),
        "ranked_candidates": [
            {
                "path": relative.as_posix(),
                "suffix": ".npz",
                "size": destination.stat().st_size,
                "source": "bounded-pcd-clean-tar-materialization",
            }
        ],
    }
    manifest_unit = {
        "object_id": object_id,
        "role": role,
        "episode": episode,
        "archive": archive_path.relative_to(dataset_root).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "selected_frames": len(member_records),
        "retained_points_per_frame": retained_points,
        "trajectory_shape": list(trajectory.shape),
        "trajectory_relative_path": relative.as_posix(),
        "trajectory_sha256": sha256_file(destination),
        "members": member_records,
    }
    return audit_unit, manifest_unit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir() or args.dataset_root.is_symlink():
        raise ValueError("dataset root must be a readable non-symlink directory")
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol identity mismatch")
    if protocol.get("registered_before_numerical_target_access") is not True:
        raise ValueError("protocol is not registered before numerical target access")

    materialization = protocol["materialization"]
    maximum_frames = int(materialization["maximum_frames_per_episode"])
    maximum_points = int(materialization["maximum_points_per_frame"])
    minimum_frames = int(materialization["minimum_frames_per_episode"])
    minimum_points = int(materialization["minimum_points_per_frame"])
    maximum_member_bytes = int(materialization["maximum_npz_member_bytes"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_units: list[dict[str, Any]] = []
    manifest_units: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in protocol["object_roster"]:
        object_id = str(item["object_id"])
        for role, field in (
            ("source", "source_episode"),
            ("target", "target_episode"),
        ):
            episode = int(item[field])
            try:
                audit_unit, manifest_unit = materialize_unit(
                    dataset_root=dataset_root,
                    output_root=args.output_root,
                    object_id=object_id,
                    role=role,
                    episode=episode,
                    maximum_frames=maximum_frames,
                    maximum_points=maximum_points,
                    minimum_frames=minimum_frames,
                    minimum_points=minimum_points,
                    maximum_member_bytes=maximum_member_bytes,
                )
                audit_units.append(audit_unit)
                manifest_units.append(manifest_unit)
            except Exception as exc:
                failures.append(
                    {
                        "object_id": object_id,
                        "role": role,
                        "episode": episode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    audit = {
        "schema": "prob4d.deform360-query-carrier-audit",
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "dataset_root": str(args.output_root),
        "units": audit_units,
        "failures": failures,
        "access_boundary": {
            "array_values_opened": True,
            "pickle_values_opened": False,
            "hdf5_values_opened": False,
            "media_decoded": False,
            "target_scored": False,
            "dataset_mutated": False,
        },
    }
    audit["audit_id"] = sha256_bytes(canonical_bytes(audit))

    manifest = {
        "schema": "prob4d.deform360-bounded-trajectory-materialization-v2",
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_bytes(protocol_raw),
        "dataset_root": str(dataset_root),
        "runner_name": os.environ.get("RUNNER_NAME", "unknown"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "github_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "requested_objects": len(protocol["object_roster"]),
        "successful_units": len(audit_units),
        "failed_units": len(failures),
        "units": manifest_units,
        "failures": failures,
        "source_data_mutated": False,
        "target_outcomes_scored": False,
    }
    manifest["materialization_id"] = sha256_bytes(canonical_bytes(manifest))

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "audit_id": audit["audit_id"],
                "materialization_id": manifest["materialization_id"],
                "successful_units": len(audit_units),
                "failed_units": len(failures),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
