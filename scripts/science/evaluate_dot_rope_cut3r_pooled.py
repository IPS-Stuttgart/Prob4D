#!/usr/bin/env python3
"""Evaluate a sealed DOT CUT3R bundle with pooled marker support.

This successor evaluator leaves the immutable R01--R03 CUT3R predictions
unchanged. It repairs evaluation-only defects by mapping marker coordinates
through CUT3R's deterministic resize/crop geometry and assessing support over
the registered multi-frame fit rather than demanding six markers in every
individual frame.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import bilinear_sample, content_id

REQUEST_SCHEMA = "prob4d.dot-rope-cut3r-pooled-evaluation-request"
RESULT_SCHEMA = "prob4d.dot-rope-cut3r-pooled-evaluation"
TECHNICAL_SCHEMA = "prob4d.dot-rope-cut3r-pooled-evaluation-technical-result"
SCHEMA_VERSION = 1
ARCHIVE = "R01-10.zip"
SEQUENCES = ["R01", "R02", "R03"]
RESERVED = "R04-R70"
FRAMES = list(range(1, 8))
CAMERA = "cam001"
SOURCE_PROTOCOL_PATH = "protocols/dot-rope-cut3r-native-provider-v1.json"
SOURCE_PROTOCOL_ID = "af6528f54699f3c9fb185764b029acd897a52e242e799d317e42710ee0f21c2c"
SOURCE_PROTOCOL_GIT_BLOB_SHA1 = "eaf84956189015c35e53a521cf1b152ca813e680"
BASE_EVALUATOR_GIT_BLOB_SHA1 = "612c8ae61b0a64d464256a11992b46c486c88012"
PROVIDER_RUN_ID = 33329701704
PROVIDER_ARTIFACT_NAME = "dot-rope-cut3r-sealed-provider-33329701704-1"
PROVIDER_BUNDLE_ID = "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7"
PROVIDER_REQUEST_ID = "83cc26be92364fc7715d692b3bb966cf914fb9f911e0763823f2789480a00cf2"
PROVIDER_REVISION = "7eb4867e36742d819c514fad21436d4f475b4bed"
SUPPORT_RULE = {
    "overlap_minimum_total_common": 6,
    "overlap_minimum_nonempty_frames": 2,
    "provider_truth_minimum_total": 6,
    "score_minimum_total": 2,
}
PREPROCESSING_TRANSFORM = (
    "cut3r-long-edge-resize-512-center-crop-multiple-of-16-pixel-centers"
)
COORDINATE_MODES = {
    "pixel-zero-based",
    "pixel-one-based",
    "unit-normalized",
    "percent-normalized",
}
SUPPORTIVE_AUDIT_DECISIONS = {
    "current-convention-pooled-support-feasible",
    "current-convention-support-sufficient",
    "alternative-coordinate-convention-required",
}
_HEX = re.compile(r"[0-9a-f]+")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

_ACTIVE_COORDINATE_COLUMNS: tuple[int, int] | None = None
_ACTIVE_COORDINATE_MODE: str | None = None
_MARKER_DIAGNOSTICS: dict[tuple[str, str, int], dict[str, Any]] = {}
_COLLECTION_DIAGNOSTICS: list[dict[str, Any]] = []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    execute = commands.add_parser("evaluate")
    execute.add_argument("--request", type=Path, required=True)
    execute.add_argument("--dataset-root", type=Path, required=True)
    execute.add_argument("--provider-bundle", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--repository-revision", required=True)
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


def _hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} must have {length} lowercase hexadecimal characters")
    return value


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def validate_request(path: Path) -> dict[str, Any]:
    request = _read_json(path)
    expected = {
        "archive",
        "base_evaluator_git_blob_sha1",
        "camera",
        "claim_boundary",
        "coordinate_columns",
        "coordinate_mode",
        "execution_nonce",
        "frames",
        "marker_payloads_opened",
        "marker_support_audit_decision",
        "marker_support_audit_id",
        "marker_support_audit_run_id",
        "normal_view_pixels_opened",
        "performance_metrics_authorized",
        "preprocessing_transform",
        "provider_artifact_name",
        "provider_bundle_id",
        "provider_request_id",
        "provider_revision",
        "provider_run_id",
        "request_id",
        "reserved_sequences",
        "schema",
        "schema_version",
        "selected_coordinate_candidate",
        "source_protocol_git_blob_sha1",
        "source_protocol_id",
        "source_protocol_path",
        "source_sequences",
        "support_rule",
        "target_payloads_opened",
    }
    if set(request) != expected:
        raise ValueError("pooled-evaluation request fields changed")
    if request["schema"] != REQUEST_SCHEMA or request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported pooled-evaluation request schema")
    fixed = {
        "archive": ARCHIVE,
        "base_evaluator_git_blob_sha1": BASE_EVALUATOR_GIT_BLOB_SHA1,
        "camera": CAMERA,
        "frames": FRAMES,
        "preprocessing_transform": PREPROCESSING_TRANSFORM,
        "provider_artifact_name": PROVIDER_ARTIFACT_NAME,
        "provider_bundle_id": PROVIDER_BUNDLE_ID,
        "provider_request_id": PROVIDER_REQUEST_ID,
        "provider_revision": PROVIDER_REVISION,
        "provider_run_id": PROVIDER_RUN_ID,
        "reserved_sequences": RESERVED,
        "source_protocol_git_blob_sha1": SOURCE_PROTOCOL_GIT_BLOB_SHA1,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_protocol_path": SOURCE_PROTOCOL_PATH,
        "source_sequences": SEQUENCES,
        "support_rule": SUPPORT_RULE,
    }
    for name, expected_value in fixed.items():
        if request[name] != expected_value:
            raise ValueError(f"{name} changed")
    columns = request["coordinate_columns"]
    if (
        not isinstance(columns, list)
        or len(columns) != 2
        or any(not isinstance(value, int) or value < 0 for value in columns)
        or columns[0] == columns[1]
    ):
        raise ValueError("coordinate_columns must contain two distinct nonnegative integers")
    mode = request["coordinate_mode"]
    if mode not in COORDINATE_MODES:
        raise ValueError("coordinate_mode is unsupported")
    selected = f"columns-{columns[0]}-{columns[1]}:{mode}"
    if request["selected_coordinate_candidate"] != selected:
        raise ValueError("selected coordinate candidate is inconsistent")
    if request["marker_support_audit_decision"] not in SUPPORTIVE_AUDIT_DECISIONS:
        raise ValueError("marker-support audit did not authorize performance scoring")
    run_id = request["marker_support_audit_run_id"]
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("marker_support_audit_run_id must be a positive integer")
    if request["marker_payloads_opened"] is not True:
        raise ValueError("source marker access must be explicitly authorized")
    if request["normal_view_pixels_opened"] is not False:
        raise ValueError("the pooled evaluator must not decode normal-view images")
    if request["performance_metrics_authorized"] is not True:
        raise ValueError("performance metrics must be explicitly authorized")
    if request["target_payloads_opened"] is not False:
        raise ValueError("reserved payload access exceeds the source boundary")
    if not isinstance(request["execution_nonce"], str) or not request["execution_nonce"]:
        raise ValueError("execution_nonce must be a non-empty string")
    for name, length in (
        ("base_evaluator_git_blob_sha1", 40),
        ("marker_support_audit_id", 64),
        ("provider_bundle_id", 64),
        ("provider_request_id", 64),
        ("provider_revision", 40),
        ("source_protocol_git_blob_sha1", 40),
        ("source_protocol_id", 64),
    ):
        _hex(request[name], name=name, length=length)
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("pooled-evaluation request identity mismatch")
    return request


def _load_base_module() -> Any:
    source = Path(__file__).with_name("run_dot_rope_cut3r_native_provider.py")
    source_bytes = source.read_bytes()
    if _git_blob_sha1(source_bytes) != BASE_EVALUATOR_GIT_BLOB_SHA1:
        raise RuntimeError("registered DOT evaluator source bytes changed")
    spec = importlib.util.spec_from_file_location("dot_cut3r_base_evaluator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the registered DOT evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(match.group(0)) for match in _NUMBER.finditer(line)]
        if values:
            rows.append(values)
    return rows


def _parse_coordinate_text(text: str, dimensions: int) -> np.ndarray:
    rows = _numeric_rows(text)
    if dimensions == 2:
        if _ACTIVE_COORDINATE_COLUMNS is None:
            raise RuntimeError("coordinate columns were not activated")
        first, second = _ACTIVE_COORDINATE_COLUMNS
        selected = [
            [row[first], row[second]]
            for row in rows
            if first < len(row) and second < len(row)
        ]
    elif dimensions == 3:
        selected = [row[-3:] for row in rows if len(row) >= 3]
    else:
        raise ValueError("coordinate dimension must be two or three")
    result = np.asarray(selected, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dimensions or result.shape[0] < 3:
        raise ValueError("coordinate payload has no valid point table")
    return result


def _to_original_pixel_coordinates(
    coordinates: np.ndarray,
    *,
    mode: str,
    width: int,
    height: int,
) -> np.ndarray:
    result = np.asarray(coordinates, dtype=np.float64).copy()
    if mode == "pixel-zero-based":
        return result
    if mode == "pixel-one-based":
        return result - 1.0
    if mode == "unit-normalized":
        result[:, 0] *= width - 1.0
        result[:, 1] *= height - 1.0
        return result
    if mode == "percent-normalized":
        result[:, 0] *= (width - 1.0) / 100.0
        result[:, 1] *= (height - 1.0) / 100.0
        return result
    raise ValueError(f"unsupported coordinate mode: {mode}")


def cut3r_output_coordinates(
    coordinates: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    output_width: int,
    output_height: int,
    image_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Map original JPEG pixel centers through CUT3R's deterministic preprocessing."""

    if min(original_width, original_height, output_width, output_height, image_size) <= 1:
        raise ValueError("image and output dimensions must exceed one pixel")
    scale = image_size / max(original_width, original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    center_x = resized_width // 2
    center_y = resized_height // 2
    half_width = ((2 * center_x) // 16) * 8
    half_height = ((2 * center_y) // 16) * 8
    crop_left = center_x - half_width
    crop_top = center_y - half_height
    expected_width = 2 * half_width
    expected_height = 2 * half_height
    if (expected_width, expected_height) != (output_width, output_height):
        raise ValueError(
            "provider point-map dimensions do not match the registered CUT3R preprocessing"
        )

    raw = np.asarray(coordinates, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    original_valid = (
        np.isfinite(raw).all(axis=1)
        & (raw[:, 0] >= 0.0)
        & (raw[:, 0] <= original_width - 1.0)
        & (raw[:, 1] >= 0.0)
        & (raw[:, 1] <= original_height - 1.0)
    )
    mapped = raw.copy()
    mapped[:, 0] = (
        (raw[:, 0] + 0.5) * resized_width / original_width - 0.5 - crop_left
    )
    mapped[:, 1] = (
        (raw[:, 1] + 0.5) * resized_height / original_height - 0.5 - crop_top
    )
    metadata = {
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "output_width": expected_width,
        "output_height": expected_height,
    }
    return mapped, original_valid, metadata


def _load_run_with_metadata(
    base: Any,
    bundle: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    run = dict(base._ORIGINAL_LOAD_RUN(bundle, record))
    run["_sequence"] = str(record["sequence"])
    run["_run_name"] = str(record["run"])
    return run


def _sample_markers(
    run: Mapping[str, Any],
    frame: int,
    coordinates_2d: np.ndarray,
    coordinates_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _ACTIVE_COORDINATE_MODE is None:
        raise RuntimeError("coordinate mode was not activated")
    frames = np.asarray(run["frames"], dtype=np.int64)
    matches = np.flatnonzero(frames == frame)
    if matches.size != 1:
        raise ValueError("provider run does not contain the requested frame exactly once")
    index = int(matches[0])
    points = np.asarray(run["points"][index], dtype=np.float64)
    confidence = np.asarray(run["confidence"][index], dtype=np.float64)
    original_width, original_height = (int(value) for value in run["original_sizes"][index])
    count = min(coordinates_2d.shape[0], coordinates_3d.shape[0])
    raw = np.asarray(coordinates_2d[:count], dtype=np.float64)
    truth = np.asarray(coordinates_3d[:count], dtype=np.float64)
    pixels = _to_original_pixel_coordinates(
        raw,
        mode=_ACTIVE_COORDINATE_MODE,
        width=original_width,
        height=original_height,
    )
    mapped, original_valid, transform = cut3r_output_coordinates(
        pixels,
        original_width=original_width,
        original_height=original_height,
        output_width=int(points.shape[1]),
        output_height=int(points.shape[0]),
    )
    sampled_points, valid_points = bilinear_sample(points, mapped)
    sampled_confidence, valid_confidence = bilinear_sample(confidence[..., None], mapped)
    finite_truth = np.isfinite(truth).all(axis=1)
    positive_confidence = np.isfinite(sampled_confidence[:, 0]) & (
        sampled_confidence[:, 0] > 0.0
    )
    valid = (
        original_valid
        & valid_points
        & valid_confidence
        & finite_truth
        & np.isfinite(sampled_points).all(axis=1)
        & positive_confidence
    )
    marker_indices = np.flatnonzero(valid)
    key = (str(run["_sequence"]), str(run["_run_name"]), int(frame))
    diagnostic = {
        "sequence": key[0],
        "run": key[1],
        "frame": key[2],
        "paired_marker_rows": int(count),
        "finite_2d_rows": int(np.count_nonzero(np.isfinite(raw).all(axis=1))),
        "finite_3d_rows": int(np.count_nonzero(finite_truth)),
        "original_image_in_bounds": int(np.count_nonzero(original_valid)),
        "provider_grid_in_bounds": int(np.count_nonzero(valid_points)),
        "positive_confidence": int(np.count_nonzero(positive_confidence)),
        "valid_marker_count": int(marker_indices.size),
        "original_image_size": [original_width, original_height],
        "provider_point_map_size": [int(points.shape[1]), int(points.shape[0])],
        "transform": transform,
    }
    previous = _MARKER_DIAGNOSTICS.setdefault(key, diagnostic)
    if previous != diagnostic:
        raise RuntimeError("marker support changed between repeated deterministic queries")
    return sampled_points[valid], truth[valid], marker_indices


def _collect_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    frame_payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    per_frame: dict[str, int] = {}
    for frame in frames:
        points_2d, points_3d = frame_payloads[int(frame)]
        first_points, _, first_indices = _sample_markers(
            first,
            int(frame),
            points_2d,
            points_3d,
        )
        second_points, _, second_indices = _sample_markers(
            second,
            int(frame),
            points_2d,
            points_3d,
        )
        common, first_positions, second_positions = np.intersect1d(
            first_indices,
            second_indices,
            assume_unique=True,
            return_indices=True,
        )
        per_frame[str(int(frame))] = int(common.size)
        if common.size:
            sources.append(second_points[second_positions])
            targets.append(first_points[first_positions])
            groups.append(np.full(common.size, int(frame), dtype=np.int64))
    total = int(sum(per_frame.values()))
    nonempty = int(sum(value > 0 for value in per_frame.values()))
    _COLLECTION_DIAGNOSTICS.append(
        {
            "kind": "window-pair",
            "sequence": str(first["_sequence"]),
            "first_run": str(first["_run_name"]),
            "second_run": str(second["_run_name"]),
            "per_frame_common": per_frame,
            "total_common": total,
            "nonempty_frames": nonempty,
        }
    )
    if total < SUPPORT_RULE["overlap_minimum_total_common"]:
        raise ValueError("pooled overlap has fewer than six common marker samples")
    if nonempty < SUPPORT_RULE["overlap_minimum_nonempty_frames"]:
        raise ValueError("pooled overlap spans fewer than two nonempty frames")
    return np.concatenate(sources), np.concatenate(targets), np.concatenate(groups)


def _collect_provider_truth(
    run: Mapping[str, Any],
    frame_payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    providers: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    per_frame: dict[str, int] = {}
    for frame in frames:
        points_2d, points_3d = frame_payloads[int(frame)]
        provider, truth, _ = _sample_markers(run, int(frame), points_2d, points_3d)
        per_frame[str(int(frame))] = int(provider.shape[0])
        if provider.shape[0]:
            providers.append(provider)
            truths.append(truth)
    total = int(sum(per_frame.values()))
    score_only = str(run["_run_name"]) == "continuous" and tuple(frames) == (6, 7)
    minimum = (
        SUPPORT_RULE["score_minimum_total"]
        if score_only
        else SUPPORT_RULE["provider_truth_minimum_total"]
    )
    _COLLECTION_DIAGNOSTICS.append(
        {
            "kind": "provider-truth",
            "sequence": str(run["_sequence"]),
            "run": str(run["_run_name"]),
            "per_frame": per_frame,
            "total": total,
            "minimum_total": minimum,
            "score_only": score_only,
        }
    )
    if total < minimum:
        raise ValueError("pooled provider/truth support is below its registered minimum")
    return np.concatenate(providers), np.concatenate(truths)


def _diagnostic_payload(
    request: Mapping[str, Any],
    repository_revision: str,
) -> dict[str, Any]:
    marker_frames = sorted(
        _MARKER_DIAGNOSTICS.values(),
        key=lambda row: (row["sequence"], row["run"], row["frame"]),
    )
    payload: dict[str, Any] = {
        "schema": "prob4d.dot-rope-cut3r-pooled-marker-support",
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "repository_revision": repository_revision,
        "selected_coordinate_candidate": request["selected_coordinate_candidate"],
        "coordinate_columns": request["coordinate_columns"],
        "coordinate_mode": request["coordinate_mode"],
        "preprocessing_transform": PREPROCESSING_TRANSFORM,
        "support_rule": SUPPORT_RULE,
        "marker_frames": marker_frames,
        "collections": _COLLECTION_DIAGNOSTICS,
        "marker_support_audit": {
            "run_id": request["marker_support_audit_run_id"],
            "audit_id": request["marker_support_audit_id"],
            "decision": request["marker_support_audit_decision"],
        },
        "information_boundary": {
            "normal_view_pixels_opened": False,
            "sealed_provider_predictions_opened": True,
            "source_2d_markers_opened": True,
            "source_3d_markers_opened": True,
            "opened_sequences": SEQUENCES,
            "reserved_sequences": RESERVED,
            "target_payloads_opened": False,
        },
    }
    payload["support_id"] = content_id(payload)
    return payload


def evaluate(args: argparse.Namespace) -> int:
    global _ACTIVE_COORDINATE_COLUMNS, _ACTIVE_COORDINATE_MODE

    request = validate_request(args.request)
    repository_revision = _hex(
        args.repository_revision,
        name="repository_revision",
        length=40,
    )
    output = args.output_dir
    _ACTIVE_COORDINATE_COLUMNS = tuple(request["coordinate_columns"])
    _ACTIVE_COORDINATE_MODE = str(request["coordinate_mode"])
    _MARKER_DIAGNOSTICS.clear()
    _COLLECTION_DIAGNOSTICS.clear()
    try:
        base = _load_base_module()
        source_protocol = Path(request["source_protocol_path"])
        protocol_bytes = source_protocol.read_bytes()
        if _git_blob_sha1(protocol_bytes) != SOURCE_PROTOCOL_GIT_BLOB_SHA1:
            raise ValueError("source protocol Git blob changed")
        protocol = base._load_protocol(source_protocol)
        if protocol["protocol_id"] != request["source_protocol_id"]:
            raise ValueError("source protocol identity changed")
        base._ORIGINAL_LOAD_RUN = base._load_run

        def load_run(bundle: Path, record: Mapping[str, Any]) -> dict[str, Any]:
            return _load_run_with_metadata(base, bundle, record)

        base._load_run = load_run
        base.parse_coordinate_text = _parse_coordinate_text
        base._sample_markers = _sample_markers
        base._collect_pair = _collect_pair
        base._collect_provider_truth = _collect_provider_truth
        base_args = argparse.Namespace(
            protocol=source_protocol,
            request_id=request["provider_request_id"],
            prob4d_revision=request["provider_revision"],
            dataset_root=args.dataset_root,
            provider_bundle=args.provider_bundle,
            output_dir=output,
        )
        status = int(base.evaluate(base_args))
        if status != 0:
            raise RuntimeError(f"registered evaluator returned status {status}")
        result_path = output / "result.json"
        result = _read_json(result_path)
        predecessor_id = result.pop("evaluation_id")
        result["schema"] = RESULT_SCHEMA
        result["schema_version"] = SCHEMA_VERSION
        result["decision"] = "complete-source-evaluation-pooled-marker-support"
        result["provider_prob4d_revision"] = result.pop("prob4d_revision")
        result["evaluator_prob4d_revision"] = repository_revision
        result["predecessor_evaluation_id"] = predecessor_id
        support = _diagnostic_payload(request, repository_revision)
        result["marker_support_id"] = support["support_id"]
        result["marker_support_audit"] = support["marker_support_audit"]
        result["marker_sampling"] = {
            "selected_coordinate_candidate": request["selected_coordinate_candidate"],
            "preprocessing_transform": PREPROCESSING_TRANSFORM,
            "support_rule": SUPPORT_RULE,
            "support_scope": "pooled-over-registered-frame-groups",
        }
        result["claim_boundary"] = request["claim_boundary"]
        result["evaluation_id"] = content_id(result)
        _write_json(output / "marker-support.json", support)
        _write_json(result_path, result)
        summary_path = output / "summary.md"
        existing = summary_path.read_text(encoding="utf-8")
        summary_path.write_text(
            existing
            + "\n## Evaluation repair\n\n"
            + "Marker coordinates were propagated through the deterministic CUT3R "
            + "resize/crop geometry, and support was pooled over the frozen multi-frame "
            + "fit. No provider prediction or method mean changed.\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "evaluation_id": result["evaluation_id"],
                    "best_method": result["aggregate_methods"][0]["method"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        support = _diagnostic_payload(request, repository_revision)
        _write_json(output / "marker-support.json", support)
        message = f"{type(error).__name__}: {' '.join(str(error).split())}"
        failure: dict[str, Any] = {
            "schema": TECHNICAL_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "request_id": request["request_id"],
            "repository_revision": repository_revision,
            "decision": "technical-failure",
            "failure": message[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
            "marker_support_id": support["support_id"],
            "marker_support_audit": support["marker_support_audit"],
            "provider_bundle_id": request["provider_bundle_id"],
            "information_boundary": support["information_boundary"],
            "claim_boundary": request["claim_boundary"],
        }
        failure["technical_result_id"] = content_id(failure)
        _write_json(output / "technical-failure.json", failure)
        print(
            json.dumps(
                {
                    "decision": failure["decision"],
                    "technical_result_id": failure["technical_result_id"],
                    "failure": failure["failure"],
                },
                sort_keys=True,
            )
        )
        return 3


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
    if args.command == "evaluate":
        return evaluate(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
