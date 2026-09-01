#!/usr/bin/env python3
"""Audit outcome-blind per-sequence camera routing support on DOT R11-R20."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SEQUENCES = [f"R{i:02d}" for i in range(11, 21)]
FRAMES = list(range(1, 8))
ARCHIVE = "R11-20.zip"
ARCHIVE_MD5 = "23ce3e7067465d3edabe20b4c7cfa388"
SHARED_3D_CAMERA = "cam001"
CAMERA_RE = re.compile(
    r"^(R(?:1[1-9]|20))/images/normal_view/frame(\d{6})_(cam\d+)\.jpg$"
)
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
FIT_PROFILES = [
    {"id": "compact", "fit_a": [1, 2], "fit_b": [6, 7]},
    {"id": "expanded", "fit_a": [1, 2, 3], "fit_b": [5, 6, 7]},
    {"id": "full-window", "fit_a": [1, 2, 3, 4, 5], "fit_b": [3, 4, 5, 6, 7]},
]
OVERLAP_GROUPS = [
    {"id": "overlap-34", "frames": [3, 4]},
    {"id": "overlap-45", "frames": [4, 5]},
    {"id": "overlap-345", "frames": [3, 4, 5]},
]
MIN_FIT = 6
MIN_OVERLAP = 8
MIN_NONEMPTY = 2
PROMOTION_MINIMUM = 9


def numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
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
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
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


def image_member(sequence: str, frame: int, camera: str) -> str:
    return f"{sequence}/images/normal_view/frame{frame:06d}_{camera}.jpg"


def coordinate_member(sequence: str, dimension: int, frame: int, camera: str) -> str:
    return f"{sequence}/coordinates/{dimension}d/frame{frame:06d}_{camera}.txt"


def valid_indices(rows: list[list[float]], width: int, height: int) -> set[int]:
    result: set[int] = set()
    for index, row in enumerate(rows):
        if len(row) < 2:
            continue
        x, y = float(row[-2]), float(row[-1])
        if math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0:
            result.add(index)
    return result


def audit(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve(strict=True)
    archive_path = (root / ARCHIVE).resolve(strict=True)
    archive_path.relative_to(root)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("official R11-20 archive unavailable")
    if md5(archive_path) != ARCHIVE_MD5:
        raise ValueError("R11-20 archive checksum mismatch")

    support: dict[str, dict[str, dict[int, set[int]]]] = {}
    cameras_by_sequence = {sequence: set() for sequence in SEQUENCES}
    row_count_pairs = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for name in names:
            match = CAMERA_RE.match(name)
            if match and int(match.group(2)) in FRAMES:
                cameras_by_sequence[match.group(1)].add(match.group(3))
        cameras = sorted(set.intersection(*(value for value in cameras_by_sequence.values())))
        if not cameras:
            raise ValueError("no common normal-view camera across R11-R20")
        for sequence in SEQUENCES:
            support[sequence] = {}
            for camera in cameras:
                support[sequence][camera] = {}
                for frame in FRAMES:
                    image = image_member(sequence, frame, camera)
                    two_d = coordinate_member(sequence, 2, frame, camera)
                    three_d = coordinate_member(sequence, 3, frame, SHARED_3D_CAMERA)
                    for member in (image, two_d, three_d):
                        pure = PurePosixPath(member)
                        if pure.is_absolute() or ".." in pure.parts or member not in names:
                            raise ValueError(f"registered source member missing: {member}")
                    width, height = jpeg_size(archive.read(image))
                    rows_2d = numeric_rows(archive.read(two_d).decode("utf-8", errors="replace"))
                    rows_3d = numeric_rows(archive.read(three_d).decode("utf-8", errors="replace"))
                    if len(rows_2d) != len(rows_3d):
                        raise ValueError(
                            "camera-specific 2-D and shared 3-D row counts differ: "
                            f"{sequence}:{camera}:{frame}:{len(rows_2d)}!={len(rows_3d)}"
                        )
                    row_count_pairs += 1
                    support[sequence][camera][frame] = valid_indices(rows_2d, width, height)

    candidate_summaries: list[dict[str, Any]] = []
    for fit in FIT_PROFILES:
        for overlap in OVERLAP_GROUPS:
            candidate_id = f"{fit['id']}__{overlap['id']}"
            per_sequence = []
            supported_sequences = 0
            worst_margin = float("inf")
            for sequence in SEQUENCES:
                eligible = []
                per_camera = []
                for camera in cameras:
                    counts = {frame: len(support[sequence][camera][frame]) for frame in FRAMES}
                    fit_a = sum(counts[frame] for frame in fit["fit_a"])
                    fit_b = sum(counts[frame] for frame in fit["fit_b"])
                    overlap_values = [counts[frame] for frame in overlap["frames"]]
                    overlap_total = sum(overlap_values)
                    overlap_nonempty = sum(value > 0 for value in overlap_values)
                    margin = min(fit_a / MIN_FIT, fit_b / MIN_FIT, overlap_total / MIN_OVERLAP)
                    feasible = (
                        fit_a >= MIN_FIT
                        and fit_b >= MIN_FIT
                        and overlap_total >= MIN_OVERLAP
                        and overlap_nonempty >= MIN_NONEMPTY
                    )
                    if feasible:
                        eligible.append(camera)
                    per_camera.append(
                        {
                            "camera": camera,
                            "feasible": feasible,
                            "fit_a_total": fit_a,
                            "fit_b_total": fit_b,
                            "overlap_total": overlap_total,
                            "overlap_nonempty_frames": overlap_nonempty,
                            "normalized_support_margin": margin,
                        }
                    )
                selected_camera = min(eligible) if eligible else None
                if selected_camera is not None:
                    supported_sequences += 1
                    selected_row = next(row for row in per_camera if row["camera"] == selected_camera)
                    worst_margin = min(worst_margin, float(selected_row["normalized_support_margin"]))
                per_sequence.append(
                    {
                        "sequence": sequence,
                        "eligible_cameras": eligible,
                        "selected_camera": selected_camera,
                        "per_camera": per_camera,
                    }
                )
            candidate_summaries.append(
                {
                    "candidate_id": candidate_id,
                    "fit_a_frames": fit["fit_a"],
                    "fit_b_frames": fit["fit_b"],
                    "overlap_frames": overlap["frames"],
                    "supported_sequences": supported_sequences,
                    "worst_selected_camera_support_margin": 0.0 if worst_margin == float("inf") else worst_margin,
                    "selected_frame_count": len(fit["fit_a"]) + len(fit["fit_b"]) + len(overlap["frames"]),
                    "per_sequence": per_sequence,
                }
            )

    selected = sorted(
        candidate_summaries,
        key=lambda row: (
            -int(row["supported_sequences"]),
            -float(row["worst_selected_camera_support_margin"]),
            int(row["selected_frame_count"]),
            str(row["candidate_id"]),
        ),
    )[0]
    decision = (
        "camera-routing-source-support-eligible"
        if int(selected["supported_sequences"]) >= PROMOTION_MINIMUM
        else "camera-routing-source-support-negative"
    )
    routing = {
        row["sequence"]: row["selected_camera"]
        for row in selected["per_sequence"]
        if row["selected_camera"] is not None
    }
    result: dict[str, Any] = {
        "schema": "prob4d.dot-r11-r20-camera-routing-support-v1",
        "schema_version": 1,
        "decision": decision,
        "source_archive": {"name": ARCHIVE, "md5": ARCHIVE_MD5},
        "source_sequences": SEQUENCES,
        "confirmation_sequences": [f"R{i:02d}" for i in range(21, 31)],
        "reserved_sequences": "R31-R70",
        "cameras_present_on_every_source_sequence": cameras,
        "coordinate_mode": "pixel-zero-based",
        "shared_3d_camera_label": SHARED_3D_CAMERA,
        "selection": {
            "minimum_metric_fit_markers": MIN_FIT,
            "minimum_overlap_common_markers": MIN_OVERLAP,
            "minimum_overlap_nonempty_frames": MIN_NONEMPTY,
            "minimum_source_supported_sequences_for_promotion": PROMOTION_MINIMUM,
            "camera_tie_break": "lexicographically-smallest-eligible-camera",
            "outcome_metrics_used": False,
        },
        "selected_candidate": selected,
        "selected_camera_routing": routing,
        "candidate_summaries": candidate_summaries,
        "row_count_pairs_verified": row_count_pairs,
        "information_boundary": {
            "r11_r20_source_images_opened": True,
            "r11_r20_source_2d_markers_opened": True,
            "r11_r20_source_3d_carriers_opened_for_layout_consistency_only": True,
            "source_reconstruction_error_computed": False,
            "source_proper_score_computed": False,
            "r21_r30_payloads_opened": False,
            "r31_r70_payloads_opened": False,
            "provider_predictions_computed": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
        },
        "claim_boundary": (
            "Outcome-blind raw-marker camera-routing feasibility on DOT R11-R20 source data only. "
            "This audit does not establish CUT3R/provider overlap, rank-six factors, prediction benefit, "
            "or confirmation performance. R21-R70 remain closed."
        ),
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
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "audit_id": result["audit_id"],
                "selected_candidate": result["selected_candidate"]["candidate_id"],
                "supported_sequences": result["selected_candidate"]["supported_sequences"],
                "routing": result["selected_camera_routing"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
