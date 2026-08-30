#!/usr/bin/env python3
"""Audit DOT rope marker support against an immutable CUT3R provider bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import bilinear_sample, content_id

REQUEST_SCHEMA = "prob4d.dot-rope-marker-support-audit-request"
RESULT_SCHEMA = "prob4d.dot-rope-marker-support-audit-result"
SCHEMA_VERSION = 1
ARCHIVE = "R01-10.zip"
SEQUENCES = ["R01", "R02", "R03"]
RESERVED = "R04-R70"
FRAMES = list(range(1, 8))
CAMERA = "cam001"
MODES = [
    "pixel-zero-based",
    "pixel-one-based",
    "unit-normalized",
    "percent-normalized",
]
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--request", type=Path, required=True)
    audit.add_argument("--dataset-root", type=Path, required=True)
    audit.add_argument("--provider-bundle", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--repository-revision", required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must have {length} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def validate_request(path: Path) -> dict[str, Any]:
    request = _read_json(path)
    expected = {
        "archive",
        "camera",
        "claim_boundary",
        "diagnostic_hypotheses",
        "frames",
        "marker_payloads_opened",
        "normal_view_pixels_opened",
        "provider_artifact_name",
        "provider_bundle_id",
        "provider_run_id",
        "request_id",
        "reserved_sequences",
        "schema",
        "schema_version",
        "source_sequences",
        "target_payloads_opened",
    }
    if set(request) != expected:
        raise ValueError("marker-support request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported marker-support request schema")
    if request["archive"] != ARCHIVE:
        raise ValueError("marker-support archive changed")
    if request["source_sequences"] != SEQUENCES or request["reserved_sequences"] != RESERVED:
        raise ValueError("marker-support sequence boundary changed")
    if request["frames"] != FRAMES or request["camera"] != CAMERA:
        raise ValueError("marker-support frame or camera roster changed")
    if request["diagnostic_hypotheses"] != MODES:
        raise ValueError("marker-support hypotheses changed")
    if request["marker_payloads_opened"] is not True:
        raise ValueError("source marker access must be explicitly authorized")
    if request["normal_view_pixels_opened"] is not False:
        raise ValueError("the marker audit must not decode images")
    if request["target_payloads_opened"] is not False:
        raise ValueError("the marker audit must not open reserved payloads")
    run_id = request["provider_run_id"]
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("provider_run_id must be a positive integer")
    expected_artifact = f"dot-rope-cut3r-sealed-provider-{run_id}-1"
    if request["provider_artifact_name"] != expected_artifact:
        raise ValueError("provider artifact name is not bound to the run")
    _hex(request["provider_bundle_id"], name="provider_bundle_id", length=64)
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("marker-support request identity mismatch")
    return request


def _safe_member(name: str) -> None:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\x00" in name:
        raise ValueError(f"unsafe ZIP member path: {name}")


def _coordinate_member(sequence: str, dimension: int, frame: int) -> str:
    return f"{sequence}/coordinates/{dimension}d/frame{frame:06d}_{CAMERA}.txt"


def _numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(match.group(0)) for match in NUMBER.finditer(line)]
        if values:
            rows.append(values)
    return rows


def _widths(rows: Sequence[Sequence[float]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(len(row))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _transform(
    raw: np.ndarray,
    *,
    mode: str,
    width: int,
    height: int,
) -> np.ndarray:
    coordinates = np.asarray(raw, dtype=np.float64).copy()
    if mode == "pixel-zero-based":
        return coordinates
    if mode == "pixel-one-based":
        return coordinates - 1.0
    if mode == "unit-normalized":
        coordinates[:, 0] *= width - 1.0
        coordinates[:, 1] *= height - 1.0
        return coordinates
    if mode == "percent-normalized":
        coordinates[:, 0] *= (width - 1.0) / 100.0
        coordinates[:, 1] *= (height - 1.0) / 100.0
        return coordinates
    raise ValueError(f"unsupported coordinate mode: {mode}")


def _verify_provider(
    bundle: Path,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    root = bundle.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("sealed provider manifest is unavailable")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != "prob4d.dot-rope-cut3r-native-provider-bundle"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("provider bundle schema changed")
    unsigned = dict(manifest)
    bundle_id = unsigned.pop("provider_bundle_id", None)
    _hex(bundle_id, name="provider_bundle_id", length=64)
    if content_id(unsigned) != bundle_id or bundle_id != request["provider_bundle_id"]:
        raise ValueError("provider bundle identity mismatch")
    if manifest.get("decision") != "sealed-provider-predictions":
        raise ValueError("provider bundle is not sealed")
    dataset = manifest.get("dataset")
    boundary = manifest.get("information_boundary")
    if not isinstance(dataset, dict) or not isinstance(boundary, dict):
        raise ValueError("provider provenance is incomplete")
    if (
        dataset.get("archive") != ARCHIVE
        or dataset.get("source_sequences") != SEQUENCES
        or dataset.get("reserved_sequences") != RESERVED
    ):
        raise ValueError("provider dataset binding changed")
    if (
        boundary.get("two_dimensional_markers_opened") is not False
        or boundary.get("three_dimensional_markers_opened") is not False
        or boundary.get("target_payloads_opened") is not False
    ):
        raise ValueError("provider information boundary changed")

    paths = {sequence: {} for sequence in SEQUENCES}
    for record in manifest.get("outputs", []):
        if not isinstance(record, dict):
            raise ValueError("provider output record is malformed")
        sequence = str(record.get("sequence"))
        run = str(record.get("run"))
        if sequence not in paths or run not in {"continuous", "window_a", "window_b"}:
            raise ValueError("provider output roster changed")
        relative = PurePosixPath(str(record.get("relative_path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe provider output path")
        path = root.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            raise ValueError("provider output is unavailable")
        if _sha256(path) != record.get("sha256"):
            raise ValueError("provider output digest mismatch")
        if path.stat().st_size != int(record.get("byte_count", -1)):
            raise ValueError("provider output byte count mismatch")
        paths[sequence][run] = path
    if any(set(value) != {"continuous", "window_a", "window_b"} for value in paths.values()):
        raise ValueError("provider output roster is incomplete")
    return manifest, paths


def _load_run(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    required = {"points", "confidence", "poses", "intrinsics", "frames", "original_sizes"}
    if set(result) != required:
        raise ValueError("provider run fields changed")
    return result


def _frame_index(run: Mapping[str, np.ndarray], frame: int) -> int | None:
    matches = np.flatnonzero(np.asarray(run["frames"], dtype=np.int64) == frame)
    if matches.size == 0:
        return None
    if matches.size != 1:
        raise ValueError("provider run contains duplicate frames")
    return int(matches[0])


def _valid_indices(
    run: Mapping[str, np.ndarray],
    *,
    frame: int,
    coordinates: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray | None:
    index = _frame_index(run, frame)
    if index is None:
        return None
    points = np.asarray(run["points"][index], dtype=np.float64)
    confidence = np.asarray(run["confidence"][index], dtype=np.float64)
    width, height = (int(value) for value in run["original_sizes"][index])
    count = min(coordinates.shape[0], truth.shape[0])
    query = np.asarray(coordinates[:count], dtype=np.float64).copy()
    original_valid = (
        np.isfinite(query).all(axis=1)
        & np.isfinite(truth[:count]).all(axis=1)
        & (query[:, 0] >= 0.0)
        & (query[:, 0] <= width - 1.0)
        & (query[:, 1] >= 0.0)
        & (query[:, 1] <= height - 1.0)
    )
    query[:, 0] *= (points.shape[1] - 1.0) / (width - 1.0)
    query[:, 1] *= (points.shape[0] - 1.0) / (height - 1.0)
    sampled_points, point_valid = bilinear_sample(points, query)
    sampled_confidence, confidence_valid = bilinear_sample(confidence[..., None], query)
    valid = (
        original_valid
        & point_valid
        & confidence_valid
        & np.isfinite(sampled_points).all(axis=1)
        & np.isfinite(sampled_confidence[:, 0])
        & (sampled_confidence[:, 0] > 0.0)
    )
    return np.flatnonzero(valid)


def _support(
    runs: Mapping[str, Mapping[str, np.ndarray]],
    *,
    sequence: str,
    frame: int,
    coordinates: np.ndarray,
    truth: np.ndarray,
) -> dict[str, int]:
    valid: dict[str, np.ndarray | None] = {}
    for run_name in ("continuous", "window_a", "window_b"):
        valid[run_name] = _valid_indices(
            runs[sequence][run_name],
            frame=frame,
            coordinates=coordinates,
            truth=truth,
        )
    window_a = valid["window_a"]
    window_b = valid["window_b"]
    common = (
        None
        if window_a is None or window_b is None
        else np.intersect1d(window_a, window_b, assume_unique=True)
    )
    result = {
        name: -1 if indices is None else int(indices.size)
        for name, indices in valid.items()
    }
    result["window_common"] = -1 if common is None else int(common.size)
    return result


def _summarize_candidate(
    name: str,
    support: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> dict[str, Any]:
    sequence_results: dict[str, Any] = {}
    all_feasible = True
    aggregate = 0
    minimum = math.inf
    for sequence in SEQUENCES:
        frames = support[sequence]
        overlap = [frames[frame]["window_common"] for frame in (3, 4, 5)]
        fit_a = [frames[frame]["window_a"] for frame in (1, 2)]
        fit_b = [frames[frame]["window_b"] for frame in (6, 7)]
        pooled = {
            "overlap_common_total": int(sum(overlap)),
            "overlap_nonempty_frames": int(sum(value > 0 for value in overlap)),
            "fit_a_total": int(sum(fit_a)),
            "fit_b_total": int(sum(fit_b)),
            "score_total": int(sum(fit_b)),
        }
        feasible = (
            pooled["overlap_common_total"] >= 6
            and pooled["overlap_nonempty_frames"] >= 2
            and pooled["fit_a_total"] >= 6
            and pooled["fit_b_total"] >= 6
            and pooled["score_total"] >= 2
        )
        values = [
            value
            for frame_support in frames.values()
            for value in frame_support.values()
            if value >= 0
        ]
        sequence_minimum = min(values) if values else 0
        aggregate += sum(values)
        minimum = min(minimum, sequence_minimum)
        all_feasible &= feasible
        sequence_results[sequence] = {
            "feasible_for_pooled_evaluation": feasible,
            "minimum_frame_support": int(sequence_minimum),
            "pooled_support": pooled,
        }
    return {
        "candidate": name,
        "all_sequences_feasible_for_pooled_evaluation": bool(all_feasible),
        "aggregate_support": int(aggregate),
        "minimum_frame_support": int(minimum if math.isfinite(minimum) else 0),
        "sequences": sequence_results,
    }


def _empty_support() -> dict[str, dict[int, dict[str, int]]]:
    return {
        sequence: {
            frame: {
                "continuous": -1,
                "window_a": -1,
                "window_b": -1,
                "window_common": -1,
            }
            for frame in FRAMES
        }
        for sequence in SEQUENCES
    }


def audit(args: argparse.Namespace) -> int:
    request = validate_request(args.request)
    _hex(args.repository_revision, name="repository revision", length=40)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("DOT dataset root must be a real directory")
    archive_path = (dataset_root / ARCHIVE).resolve(strict=True)
    archive_path.relative_to(dataset_root)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("DOT source archive is unavailable")

    manifest, provider_paths = _verify_provider(args.provider_bundle, request)
    runs = {
        sequence: {
            run_name: _load_run(path)
            for run_name, path in sequence_paths.items()
        }
        for sequence, sequence_paths in provider_paths.items()
    }
    current_support = _empty_support()
    candidate_support: dict[str, dict[str, dict[int, dict[str, int]]]] = {}
    diagnostics: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for sequence in SEQUENCES:
            for frame in FRAMES:
                member_2d = _coordinate_member(sequence, 2, frame)
                member_3d = _coordinate_member(sequence, 3, frame)
                if member_2d not in names or member_3d not in names:
                    raise ValueError("registered DOT marker payload is missing")
                _safe_member(member_2d)
                _safe_member(member_3d)
                raw_2d = archive.read(member_2d)
                raw_3d = archive.read(member_3d)
                opened.extend(
                    [
                        {
                            "sequence": sequence,
                            "frame": frame,
                            "kind": "2d",
                            "member": member_2d,
                            "byte_count": len(raw_2d),
                            "sha256": _sha256_bytes(raw_2d),
                        },
                        {
                            "sequence": sequence,
                            "frame": frame,
                            "kind": "3d",
                            "member": member_3d,
                            "byte_count": len(raw_3d),
                            "sha256": _sha256_bytes(raw_3d),
                        },
                    ]
                )
                rows_2d = _numeric_rows(raw_2d.decode("utf-8"))
                rows_3d = _numeric_rows(raw_3d.decode("utf-8"))
                count = min(len(rows_2d), len(rows_3d))
                truth = np.asarray(
                    [row[-3:] for row in rows_3d[:count] if len(row) >= 3],
                    dtype=np.float64,
                )
                if truth.shape != (count, 3):
                    truth = np.full((count, 3), np.nan, dtype=np.float64)
                continuous_index = _frame_index(runs[sequence]["continuous"], frame)
                if continuous_index is None:
                    raise ValueError("continuous provider run is missing a frame")
                width, height = (
                    int(value)
                    for value in runs[sequence]["continuous"]["original_sizes"][
                        continuous_index
                    ]
                )
                point_height, point_width = runs[sequence]["continuous"]["points"][
                    continuous_index
                ].shape[:2]

                current: dict[str, int] | None = None
                if count and all(len(row) >= 2 for row in rows_2d[:count]):
                    coordinates = np.asarray(
                        [[row[-2], row[-1]] for row in rows_2d[:count]],
                        dtype=np.float64,
                    )
                    current = _support(
                        runs,
                        sequence=sequence,
                        frame=frame,
                        coordinates=coordinates,
                        truth=truth,
                    )
                    current_support[sequence][frame] = current

                minimum_width = min((len(row) for row in rows_2d[:count]), default=0)
                best = 0
                for left in range(minimum_width):
                    for right in range(minimum_width):
                        if left == right:
                            continue
                        raw = np.asarray(
                            [[row[left], row[right]] for row in rows_2d[:count]],
                            dtype=np.float64,
                        )
                        for mode in MODES:
                            name = f"columns-{left}-{right}:{mode}"
                            transformed = _transform(
                                raw,
                                mode=mode,
                                width=width,
                                height=height,
                            )
                            support = _support(
                                runs,
                                sequence=sequence,
                                frame=frame,
                                coordinates=transformed,
                                truth=truth,
                            )
                            candidate_support.setdefault(name, _empty_support())[
                                sequence
                            ][frame] = support
                            best = max(best, *(value for value in support.values() if value >= 0))
                diagnostics.append(
                    {
                        "sequence": sequence,
                        "frame": frame,
                        "rows_2d": len(rows_2d),
                        "rows_3d": len(rows_3d),
                        "paired_rows": count,
                        "numeric_widths_2d": _widths(rows_2d),
                        "numeric_widths_3d": _widths(rows_3d),
                        "finite_last_three_3d_rows": int(
                            np.count_nonzero(np.isfinite(truth).all(axis=1))
                        ),
                        "original_image_size": [width, height],
                        "provider_point_map_size": [int(point_width), int(point_height)],
                        "current_parser_support": current,
                        "best_registered_support": best,
                    }
                )

    current_summary = _summarize_candidate(
        "current-last-two:pixel-zero-based",
        current_support,
    )
    candidates = [
        _summarize_candidate(name, support)
        for name, support in candidate_support.items()
    ]
    candidates.sort(
        key=lambda value: (
            not value["all_sequences_feasible_for_pooled_evaluation"],
            -value["minimum_frame_support"],
            -value["aggregate_support"],
            value["candidate"],
        )
    )
    feasible = [
        value for value in candidates
        if value["all_sequences_feasible_for_pooled_evaluation"]
    ]
    current_feasible = current_summary[
        "all_sequences_feasible_for_pooled_evaluation"
    ]
    current_below_six = any(
        value < 6
        for sequence in current_support.values()
        for frame in sequence.values()
        for value in frame.values()
        if value >= 0
    )
    if current_feasible and current_below_six:
        decision = "current-convention-pooled-support-feasible"
    elif current_feasible:
        decision = "current-convention-support-sufficient"
    elif len(feasible) == 1:
        decision = "alternative-coordinate-convention-required"
    elif feasible:
        decision = "coordinate-convention-ambiguous"
    else:
        decision = "source-marker-support-negative"

    interpretations = {
        "current-convention-pooled-support-feasible": (
            "The current coordinate interpretation has enough pooled support, but the "
            "six-markers-per-frame evaluator gate is stricter than the pooled fit."
        ),
        "current-convention-support-sufficient": (
            "The current coordinate interpretation satisfies the registered support "
            "requirements; the previous failure requires another implementation audit."
        ),
        "alternative-coordinate-convention-required": (
            "Exactly one registered coordinate interpretation is support-feasible while "
            "the current interpretation is not."
        ),
        "coordinate-convention-ambiguous": (
            "Multiple coordinate interpretations are support-feasible; the public format "
            "contract must disambiguate them before scoring."
        ),
        "source-marker-support-negative": (
            "No registered coordinate interpretation has enough source support for the "
            "frozen evaluation."
        ),
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "repository_revision": args.repository_revision,
        "decision": decision,
        "dataset": {
            "archive": ARCHIVE,
            "archive_size_bytes": archive_path.stat().st_size,
            "source_sequences": SEQUENCES,
            "reserved_sequences": RESERVED,
        },
        "provider": {
            "provider_run_id": request["provider_run_id"],
            "provider_artifact_name": request["provider_artifact_name"],
            "provider_bundle_id": manifest["provider_bundle_id"],
            "provider_revision": manifest["prob4d_revision"],
            "runtime_artifact_id": manifest["runtime_artifact_id"],
        },
        "current_parser": current_summary,
        "current_parser_any_support_below_six": current_below_six,
        "feasible_candidate_count": len(feasible),
        "top_candidate_hypotheses": candidates[:20],
        "frame_diagnostics": diagnostics,
        "opened_marker_members": opened,
        "information_boundary": {
            "normal_view_pixels_opened": False,
            "provider_predictions_opened": True,
            "source_2d_markers_opened": True,
            "source_3d_markers_opened": True,
            "opened_sequences": SEQUENCES,
            "reserved_sequences": RESERVED,
            "target_payloads_opened": False,
            "performance_metrics_computed": False,
            "protocol_or_threshold_changed": False,
        },
        "interpretation": interpretations[decision],
        "claim_boundary": request["claim_boundary"],
    }
    result["audit_id"] = content_id(result)
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "decision": decision,
                "audit_id": result["audit_id"],
                "feasible_candidate_count": len(feasible),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-request":
        request = validate_request(args.request)
        print(
            json.dumps(
                {
                    "request_id": request["request_id"],
                    "provider_run_id": request["provider_run_id"],
                    "provider_artifact_name": request["provider_artifact_name"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit":
        return audit(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
