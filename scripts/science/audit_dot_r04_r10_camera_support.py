#!/usr/bin/env python3
"""Audit camera/visibility support on already-open DOT R04-R10 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SEQUENCES = [f"R{i:02d}" for i in range(4, 11)]
FRAMES = list(range(1, 8))
ARCHIVE = "R01-10.zip"
ARCHIVE_MD5 = "ca546ff5f22c0279123ccb18509858ee"
CAMERA_RE = re.compile(
    r"^(R(?:0[4-9]|10))/images/normal_view/frame(\d{6})_(cam\d+)\.jpg$"
)
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(match.group(0)) for match in NUMBER.finditer(line)]
        if values:
            rows.append(values)
    return rows


def jpeg_size(blob: bytes) -> tuple[int, int]:
    if len(blob) < 4 or blob[:2] != b"\xff\xd8":
        raise ValueError("normal-view payload is not a JPEG")
    offset = 2
    sof = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 4 <= len(blob):
        if blob[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(blob) and blob[offset] == 0xFF:
            offset += 1
        if offset >= len(blob):
            break
        marker = blob[offset]
        offset += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(blob):
            break
        length = int.from_bytes(blob[offset : offset + 2], "big")
        if length < 2 or offset + length > len(blob):
            raise ValueError("malformed JPEG segment")
        if marker in sof:
            if length < 7:
                raise ValueError("malformed JPEG SOF")
            height = int.from_bytes(blob[offset + 3 : offset + 5], "big")
            width = int.from_bytes(blob[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("invalid JPEG dimensions")
            return width, height
        offset += length
    raise ValueError("JPEG dimensions unavailable")


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member(sequence: str, dimension: int, frame: int, camera: str) -> str:
    return f"{sequence}/coordinates/{dimension}d/frame{frame:06d}_{camera}.txt"


def image_member(sequence: str, frame: int, camera: str) -> str:
    return f"{sequence}/images/normal_view/frame{frame:06d}_{camera}.jpg"


def finite_pair(row: list[float]) -> tuple[float, float] | None:
    if len(row) < 2:
        return None
    x, y = float(row[-2]), float(row[-1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def valid_indices(rows: list[list[float]], width: int, height: int, mode: str) -> set[int]:
    result: set[int] = set()
    for idx, row in enumerate(rows):
        pair = finite_pair(row)
        if pair is None:
            continue
        x, y = pair
        if mode == "pixel-zero-based":
            valid = 0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0
        elif mode == "pixel-one-based":
            valid = 1.0 <= x <= width and 1.0 <= y <= height
        elif mode == "swapped-pixel-zero-based":
            valid = 0.0 <= y <= width - 1.0 and 0.0 <= x <= height - 1.0
        elif mode == "swapped-pixel-one-based":
            valid = 1.0 <= y <= width and 1.0 <= x <= height
        elif mode == "unit-normalized":
            valid = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        elif mode == "percent-normalized":
            valid = 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0
        else:
            raise ValueError(mode)
        if valid:
            result.add(idx)
    return result


def compact_pair(pair: tuple[float, float]) -> str:
    return f"{pair[0]:.6g},{pair[1]:.6g}"


def readme_evidence(dataset_root: Path) -> dict[str, Any]:
    candidates = [dataset_root / "A_Read_Me.txt", dataset_root / "A_Readme.txt"]
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            terms = ("marker", "coordinate", "camera", "visible", "visibility", "2d", "image")
            lines = [
                line.strip()
                for line in text.splitlines()
                if any(term in line.lower() for term in terms)
            ][:80]
            return {
                "present": True,
                "path": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "relevant_lines": lines,
            }
    return {"present": False, "relevant_lines": []}


def pooled(counts: dict[int, int]) -> dict[str, int | bool]:
    overlap = [counts.get(frame, 0) for frame in (3, 4, 5)]
    fit_a = [counts.get(frame, 0) for frame in (1, 2)]
    fit_b = [counts.get(frame, 0) for frame in (6, 7)]
    out: dict[str, int | bool] = {
        "overlap_total": sum(overlap),
        "overlap_nonempty_frames": sum(value > 0 for value in overlap),
        "fit_a_total": sum(fit_a),
        "fit_b_total": sum(fit_b),
    }
    out["registered_support_feasible"] = bool(
        int(out["overlap_total"]) >= 6
        and int(out["overlap_nonempty_frames"]) >= 2
        and int(out["fit_a_total"]) >= 6
        and int(out["fit_b_total"]) >= 6
    )
    return out


def audit(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dataset root must be a real directory")
    archive_path = (root / ARCHIVE).resolve(strict=True)
    archive_path.relative_to(root)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("official R01-10 archive unavailable")
    if md5(archive_path) != ARCHIVE_MD5:
        raise ValueError("R01-10 archive checksum mismatch")

    modes = (
        "pixel-zero-based",
        "pixel-one-based",
        "swapped-pixel-zero-based",
        "swapped-pixel-one-based",
        "unit-normalized",
        "percent-normalized",
    )
    per_frame: dict[str, Any] = {}
    support: dict[str, dict[str, dict[str, dict[int, set[int]]]]] = {}
    cameras_by_sequence: dict[str, set[str]] = {sequence: set() for sequence in SEQUENCES}
    three_d_hashes: dict[str, dict[int, dict[str, str]]] = {
        sequence: {frame: {} for frame in FRAMES} for sequence in SEQUENCES
    }

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for name in names:
            match = CAMERA_RE.match(name)
            if match:
                sequence, frame_text, camera = match.groups()
                if int(frame_text) in FRAMES:
                    cameras_by_sequence[sequence].add(camera)

        common_cameras = set.intersection(*(value for value in cameras_by_sequence.values()))
        if not common_cameras:
            raise ValueError("no camera is present across R04-R10")
        cameras = sorted(common_cameras)

        for sequence in SEQUENCES:
            support[sequence] = {}
            for camera in cameras:
                support[sequence][camera] = {mode: {} for mode in modes}
                for frame in FRAMES:
                    image_name = image_member(sequence, frame, camera)
                    two_name = member(sequence, 2, frame, camera)
                    three_name = member(sequence, 3, frame, camera)
                    for path in (image_name, two_name, three_name):
                        pure = PurePosixPath(path)
                        if pure.is_absolute() or ".." in pure.parts or path not in names:
                            raise ValueError(f"registered payload missing: {path}")
                    width, height = jpeg_size(archive.read(image_name))
                    raw_2d = archive.read(two_name)
                    raw_3d = archive.read(three_name)
                    rows_2d = numeric_rows(raw_2d.decode("utf-8", errors="replace"))
                    rows_3d = numeric_rows(raw_3d.decode("utf-8", errors="replace"))
                    count = min(len(rows_2d), len(rows_3d))
                    rows_2d = rows_2d[:count]
                    rows_3d = rows_3d[:count]
                    three_d_hashes[sequence][frame][camera] = hashlib.sha256(raw_3d).hexdigest()
                    counts: dict[str, int] = {}
                    for mode in modes:
                        indices = valid_indices(rows_2d, width, height, mode)
                        support[sequence][camera][mode][frame] = indices
                        counts[mode] = len(indices)
                    finite_pairs = [pair for row in rows_2d if (pair := finite_pair(row)) is not None]
                    repeated = Counter(compact_pair(pair) for pair in finite_pairs)
                    sentinels = [
                        {"value": value, "count": occurrences}
                        for value, occurrences in repeated.most_common(8)
                        if occurrences >= 2
                    ]
                    per_frame[f"{sequence}:{camera}:{frame:06d}"] = {
                        "image_size": [width, height],
                        "rows_2d": len(rows_2d),
                        "rows_3d": len(rows_3d),
                        "paired_rows": count,
                        "support_by_coordinate_hypothesis": counts,
                        "repeated_coordinate_pairs": sentinels,
                        "x_range": (
                            [min(pair[0] for pair in finite_pairs), max(pair[0] for pair in finite_pairs)]
                            if finite_pairs
                            else None
                        ),
                        "y_range": (
                            [min(pair[1] for pair in finite_pairs), max(pair[1] for pair in finite_pairs)]
                            if finite_pairs
                            else None
                        ),
                    }

    cameras = sorted(set.intersection(*(value for value in cameras_by_sequence.values())))
    camera_summary: dict[str, Any] = {}
    for camera in cameras:
        mode_summary: dict[str, Any] = {}
        for mode in modes:
            sequence_summary = {}
            for sequence in SEQUENCES:
                counts = {frame: len(support[sequence][camera][mode][frame]) for frame in FRAMES}
                sequence_summary[sequence] = {
                    "per_frame": {str(frame): counts[frame] for frame in FRAMES},
                    "pooled": pooled(counts),
                }
            mode_summary[mode] = {
                "all_sequences_feasible": all(
                    value["pooled"]["registered_support_feasible"]
                    for value in sequence_summary.values()
                ),
                "minimum_fit_a_total": min(
                    int(value["pooled"]["fit_a_total"]) for value in sequence_summary.values()
                ),
                "minimum_fit_b_total": min(
                    int(value["pooled"]["fit_b_total"]) for value in sequence_summary.values()
                ),
                "sequences": sequence_summary,
            }
        camera_summary[camera] = mode_summary

    multiview: dict[str, Any] = {}
    for mode in modes:
        sequence_summary = {}
        for sequence in SEQUENCES:
            counts: dict[int, int] = {}
            for frame in FRAMES:
                union: set[int] = set()
                for camera in cameras:
                    union |= support[sequence][camera][mode][frame]
                counts[frame] = len(union)
            sequence_summary[sequence] = {
                "per_frame_union": {str(frame): counts[frame] for frame in FRAMES},
                "pooled": pooled(counts),
            }
        multiview[mode] = {
            "all_sequences_feasible": all(
                value["pooled"]["registered_support_feasible"]
                for value in sequence_summary.values()
            ),
            "sequences": sequence_summary,
        }

    three_d_consistency = {
        sequence: {
            str(frame): len(set(three_d_hashes[sequence][frame].values())) == 1
            for frame in FRAMES
        }
        for sequence in SEQUENCES
    }
    fixed = [
        {"camera": camera, "mode": mode}
        for camera in cameras
        for mode in modes
        if camera_summary[camera][mode]["all_sequences_feasible"]
    ]
    multiview_feasible = [mode for mode in modes if multiview[mode]["all_sequences_feasible"]]
    cam001_pixel0 = (
        camera_summary.get("cam001", {})
        .get("pixel-zero-based", {})
        .get("all_sequences_feasible", False)
    )
    if fixed:
        decision = "fixed-camera-support-available"
    elif multiview_feasible:
        decision = "multiview-support-available"
    elif not cam001_pixel0 and any(
        camera_summary.get("cam001", {}).get(mode, {}).get("all_sequences_feasible", False)
        for mode in modes
        if mode != "pixel-zero-based"
    ):
        decision = "cam001-coordinate-semantics-mismatch"
    else:
        decision = "support-remains-insufficient-or-ambiguous"

    result = {
        "schema": "prob4d.dot-r04-r10-camera-support-audit",
        "schema_version": 1,
        "decision": decision,
        "scientific_boundary": {
            "sequences_opened": SEQUENCES,
            "r11_r70_opened": False,
            "performance_metrics_computed": False,
            "frozen_r04_r10_result_reinterpreted": False,
        },
        "dataset": {
            "archive": ARCHIVE,
            "archive_md5": ARCHIVE_MD5,
            "cameras_present_on_every_sequence": cameras,
        },
        "readme": readme_evidence(root),
        "cross_camera_3d_payload_identical": three_d_consistency,
        "fixed_camera_feasible_candidates": fixed,
        "multiview_feasible_coordinate_modes": multiview_feasible,
        "camera_summary": camera_summary,
        "multiview_union_summary": multiview,
        "per_frame_diagnostics": per_frame,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["audit_id"] = hashlib.sha256(encoded).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "audit_id": result["audit_id"],
                "cameras": result["dataset"]["cameras_present_on_every_sequence"],
                "fixed_camera_feasible_candidates": result["fixed_camera_feasible_candidates"],
                "multiview_feasible_coordinate_modes": result[
                    "multiview_feasible_coordinate_modes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
