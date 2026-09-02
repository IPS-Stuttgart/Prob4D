#!/usr/bin/env python3
"""Evaluate prior-anchored query messages on public DOT R11--R20 data.

The evaluator reuses the exact sealed, marker-free routed CUT3R artifact from
workflow run 33552798863.  It opens only the already-authorized R11--R20 marker
payloads from the official R11-20.zip archive.  R21--R70 are never enumerated.

For every sequence, window A and window B are independently aligned to physical
marker coordinates on disjoint temporal fit groups (frames 1--3 and 5--7).
Both windows then estimate the same frame-3 to frame-4 material-point
displacement query.  A leave-one-sequence-out calibration estimates marginal
and joint error moments from the other nine sequences with equal sequence
weight.  The methods compare single-window posteriors, prior-anchored
query-message covariance intersection, naive independent message addition, and
an empirical joint-correlated Gaussian reference.  The physical fallback is
exact zero displacement (persistence).

This is source/development evidence.  It does not authorize or open R21--R70.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import content_id, robust_fit_sim3
from prob4d.query_message import (
    GaussianQueryBelief,
    GaussianQueryMessage,
    apply_gaussian_query_message,
    compress_gaussian_query_posterior,
    fuse_gaussian_query_messages_covariance_intersection,
    select_pairwise_covariance_intersection,
)
from prob4d.query_posterior import GaussianQueryPosterior

PROTOCOL_SCHEMA = "prob4d.dot-r11-r20-query-message-real-source-protocol"
REQUEST_SCHEMA = "prob4d.dot-r11-r20-query-message-real-source-request"
RESULT_SCHEMA = "prob4d.dot-r11-r20-query-message-real-source-result"
SCHEMA_VERSION = 1

SOURCE_SEQUENCES = tuple(f"R{index:02d}" for index in range(11, 21))
CONFIRMATION_SEQUENCES = tuple(f"R{index:02d}" for index in range(21, 31))
RESERVED_SEQUENCES = "R31-R70"
ARCHIVE_NAME = "R11-20.zip"
ARCHIVE_MD5 = "23ce3e7067465d3edabe20b4c7cfa388"
PROVIDER_RUN_ID = 33552798863
PROVIDER_ARTIFACT_ID = 9818146750
PROVIDER_ARTIFACT_DIGEST = (
    "sha256:70c2cec1cf33b65ae6653a3839fb4f74d57023d1d49cced6c61e6948f4d7b8a6"
)
PROVIDER_SEAL_ID = "38ea78e8bf44cbeaedeeadaee862af3cc6369d35d7e3b5a2b5fac0f020c7145b"
TERMINAL_RANK_RESULT_ID = (
    "a1fc018dc7fb504b35f6fbfc422e7a59edaeb71009c6f29c512019d42f949ced"
)
POOLED_EVALUATOR_PATH = Path("scripts/science/evaluate_dot_rope_cut3r_pooled.py")
POOLED_EVALUATOR_BLOB = "6195e70997f0e9582251c08772b1e423a3062ad6"
CHI2_3_90 = 6.251388631170325
NORMAL_90 = 1.6448536269514722


@dataclass(frozen=True)
class SequenceQueries:
    """Normalized real query observations for one complete DOT sequence."""

    sequence: str
    camera: str
    factor_rank: int
    truth: np.ndarray
    window_a: np.ndarray
    window_b: np.ndarray
    continuous: np.ndarray
    marker_indices: np.ndarray
    rope_span: float
    fit_a_count: int
    fit_b_count: int
    fit_continuous_count: int


@dataclass(frozen=True)
class GaussianModel:
    """One source-fitted common-prior and measurement-error model."""

    prior_covariance: np.ndarray
    bias_a: np.ndarray
    bias_b: np.ndarray
    bias_continuous: np.ndarray
    covariance_a: np.ndarray
    covariance_b: np.ndarray
    covariance_continuous: np.ndarray
    cross_covariance_ab: np.ndarray
    joint_covariance_ab: np.ndarray
    whitened_cross_singular_values: np.ndarray


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
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
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must contain {length} lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must contain {length} lowercase hexadecimal characters")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported DOT query-message real-source protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _hex(protocol_id, name="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("protocol identity mismatch")
    dataset = protocol.get("dataset") or {}
    if dataset.get("archive") != {"name": ARCHIVE_NAME, "md5": ARCHIVE_MD5}:
        raise ValueError("source archive binding changed")
    if tuple(dataset.get("source_sequences") or ()) != SOURCE_SEQUENCES:
        raise ValueError("source sequence roster changed")
    if tuple(dataset.get("confirmation_sequences") or ()) != CONFIRMATION_SEQUENCES:
        raise ValueError("confirmation sequence roster changed")
    if dataset.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("reserve changed")
    provider = protocol.get("sealed_provider") or {}
    expected_provider = {
        "workflow_run_id": PROVIDER_RUN_ID,
        "artifact_id": PROVIDER_ARTIFACT_ID,
        "artifact_digest": PROVIDER_ARTIFACT_DIGEST,
        "provider_seal_id": PROVIDER_SEAL_ID,
    }
    for name, expected in expected_provider.items():
        if provider.get(name) != expected:
            raise ValueError(f"sealed provider {name} changed")
    terminal = protocol.get("terminal_rank_diagnostic") or {}
    if terminal.get("result_id") != TERMINAL_RANK_RESULT_ID:
        raise ValueError("terminal rank result changed")
    if terminal.get("factor_rank_by_sequence") != {
        sequence: (6 if sequence == "R18" else 7) for sequence in SOURCE_SEQUENCES
    }:
        raise ValueError("terminal rank stratification changed")
    evaluation = protocol.get("evaluation") or {}
    if evaluation.get("fit_window_a_frames") != [1, 2, 3]:
        raise ValueError("window-A fit frames changed")
    if evaluation.get("fit_window_b_frames") != [5, 6, 7]:
        raise ValueError("window-B fit frames changed")
    if evaluation.get("fit_continuous_frames") != [1, 2, 3]:
        raise ValueError("continuous fit frames changed")
    if evaluation.get("anchor_frame") != 3 or evaluation.get("query_frame") != 4:
        raise ValueError("registered displacement query changed")
    if evaluation.get("coordinate_columns") != [0, 1]:
        raise ValueError("coordinate columns changed")
    if evaluation.get("coordinate_mode") != "pixel-zero-based":
        raise ValueError("coordinate mode changed")
    if evaluation.get("common_prior_mean") != "exact-zero-displacement-fallback":
        raise ValueError("physical fallback changed")
    boundary = protocol.get("information_boundary") or {}
    required_false = (
        "confirmation_access_authorized",
        "r21_r30_payloads_opened",
        "r31_r70_payloads_opened",
        "target_side_retuning_allowed",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    )
    if any(boundary.get(name) is not False for name in required_false):
        raise ValueError("information boundary changed")
    if boundary.get("source_sequences_previously_opened") is not True:
        raise ValueError("source-opening history changed")
    if boundary.get("sealed_provider_reuse_only") is not True:
        raise ValueError("provider rerun is forbidden")
    return protocol


def _load_request(path: Path, protocol: Mapping[str, Any], protocol_blob: str) -> dict[str, Any]:
    request = _read_json(path)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported DOT query-message real-source request")
    unsigned = dict(request)
    request_id = unsigned.pop("request_id", None)
    _hex(request_id, name="request_id", length=64)
    if content_id(unsigned) != request_id:
        raise ValueError("request identity mismatch")
    if request.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("request protocol identity changed")
    if request.get("protocol_path") != "protocols/dot-r11-r20-query-message-real-v1.json":
        raise ValueError("request protocol path changed")
    if request.get("protocol_git_blob_sha1") != protocol_blob:
        raise ValueError("request does not bind the reviewed protocol blob")
    if tuple(request.get("source_sequences") or ()) != SOURCE_SEQUENCES:
        raise ValueError("request source roster changed")
    if tuple(request.get("confirmation_sequences") or ()) != CONFIRMATION_SEQUENCES:
        raise ValueError("request confirmation roster changed")
    if request.get("reserved_sequences") != RESERVED_SEQUENCES:
        raise ValueError("request reserve changed")
    if request.get("source_evaluation_authorized") is not True:
        raise ValueError("source evaluation was not explicitly authorized")
    for name in (
        "provider_rerun_authorized",
        "confirmation_access_authorized",
        "post_source_target_tuning_authorized",
        "bayesian_phystwin_executed",
        "causal4d_executed",
    ):
        if request.get(name) is not False:
            raise ValueError(f"{name} exceeds the source-only request boundary")
    return request


def validate_request(
    *, protocol_path: Path, request_path: Path, protocol_git_blob_sha1: str
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    _hex(protocol_git_blob_sha1, name="protocol Git blob", length=40)
    request = _load_request(request_path, protocol, protocol_git_blob_sha1)
    return {
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "source_sequences": list(SOURCE_SEQUENCES),
        "confirmation_access_authorized": False,
    }


def _load_pooled_evaluator(protocol: Mapping[str, Any]) -> ModuleType:
    path = POOLED_EVALUATOR_PATH
    source = path.read_bytes()
    expected = str(protocol["evaluation"]["pooled_evaluator_git_blob_sha1"])
    if expected != POOLED_EVALUATOR_BLOB or _git_blob_sha1(source) != expected:
        raise ValueError("registered pooled evaluator source bytes changed")
    spec = importlib.util.spec_from_file_location("dot_query_message_pooled", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the registered pooled evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ACTIVE_COORDINATE_COLUMNS = tuple(protocol["evaluation"]["coordinate_columns"])
    module._ACTIVE_COORDINATE_MODE = str(protocol["evaluation"]["coordinate_mode"])
    module._MARKER_DIAGNOSTICS.clear()
    module._COLLECTION_DIAGNOSTICS.clear()
    return module


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe DOT archive member: {name}")


def _coordinate_member(sequence: str, dimensions: int, frame: int, camera: str) -> str:
    if dimensions == 2:
        return f"{sequence}/coordinates/2d/frame{frame:06d}_{camera}.txt"
    if dimensions == 3:
        return f"{sequence}/coordinates/3d/frame{frame:06d}_cam001.txt"
    raise ValueError("coordinate dimensions must be 2 or 3")


def _verified_provider_records(
    provider_root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, dict[str, Path]], list[dict[str, Any]]]:
    providers_root = provider_root / "providers"
    seal_path = providers_root / "provider-seal.json"
    seal = _read_json(seal_path)
    if seal.get("schema") != "prob4d.dot-r11-r20-routed-provider-seal":
        raise ValueError("routed provider seal schema changed")
    unsigned_seal = dict(seal)
    seal_id = unsigned_seal.pop("provider_seal_id", None)
    if seal_id != PROVIDER_SEAL_ID or content_id(unsigned_seal) != seal_id:
        raise ValueError("routed provider seal identity mismatch")
    if seal.get("confirmation_payloads_opened") is not False:
        raise ValueError("provider seal reports confirmation access")
    if seal.get("source_marker_payloads_opened") is not False:
        raise ValueError("provider stage opened source markers")

    expected_components = {
        item["camera"]: item for item in protocol["sealed_provider"]["components"]
    }
    route: dict[str, str] = {}
    records: dict[str, dict[str, Path]] = {sequence: {} for sequence in SOURCE_SEQUENCES}
    verified_outputs: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    for component in seal.get("components", []):
        camera = str(component["camera"])
        if camera not in expected_components:
            raise ValueError("provider seal contains an unregistered camera component")
        expected = expected_components[camera]
        if component.get("provider_bundle_id") != expected["provider_bundle_id"]:
            raise ValueError("component provider identity changed")
        component_sequences = tuple(component.get("source_sequences") or ())
        if component_sequences != tuple(expected["sequences"]):
            raise ValueError("component source roster changed")
        bundle = providers_root / camera / "bundle"
        manifest_path = bundle / "manifest.json"
        if _sha256_file(manifest_path) != component["manifest_sha256"]:
            raise ValueError("component manifest bytes changed")
        manifest = _read_json(manifest_path)
        unsigned = dict(manifest)
        provider_bundle_id = unsigned.pop("provider_bundle_id", None)
        if provider_bundle_id != expected["provider_bundle_id"]:
            raise ValueError("component manifest provider identity changed")
        if content_id(unsigned) != provider_bundle_id:
            raise ValueError("component manifest content identity mismatch")
        for sequence in component_sequences:
            if sequence in route:
                raise ValueError("sequence is routed to more than one camera")
            route[sequence] = camera
        for record in manifest.get("outputs", []):
            sequence = str(record["sequence"])
            run = str(record["run"])
            if sequence not in component_sequences or run not in {
                "continuous",
                "window_a",
                "window_b",
            }:
                raise ValueError("component output roster changed")
            relative = PurePosixPath(str(record["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe provider output path")
            path = bundle.joinpath(*relative.parts).resolve(strict=True)
            path.relative_to(bundle.resolve(strict=True))
            if path.is_symlink() or not path.is_file():
                raise ValueError("provider output is not a regular file")
            if path.stat().st_size != int(record["byte_count"]):
                raise ValueError("provider output byte count changed")
            if _sha256_file(path) != record["sha256"]:
                raise ValueError("provider output digest changed")
            if run in records[sequence]:
                raise ValueError("duplicate provider run")
            records[sequence][run] = path
            verified_outputs.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "run": run,
                    "relative_path": path.relative_to(provider_root).as_posix(),
                    "byte_count": int(record["byte_count"]),
                    "sha256": record["sha256"],
                }
            )
        seen_components.add(camera)
    if seen_components != set(expected_components):
        raise ValueError("provider component roster is incomplete")
    if set(route) != set(SOURCE_SEQUENCES):
        raise ValueError("provider routing does not cover R11--R20 exactly")
    for sequence, runs in records.items():
        if set(runs) != {"continuous", "window_a", "window_b"}:
            raise ValueError(f"provider runs are incomplete for {sequence}")
    return route, records, verified_outputs


def _load_run(path: Path, *, sequence: str, run_name: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    required = {"points", "confidence", "poses", "intrinsics", "frames", "original_sizes"}
    if set(result) != required:
        raise ValueError("provider run fields changed")
    result["_sequence"] = sequence
    result["_run_name"] = run_name
    return result


def _parse_sequence_payloads(
    archive: zipfile.ZipFile,
    *,
    sequence: str,
    camera: str,
    pooled: ModuleType,
    opened_members: list[dict[str, Any]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    names = set(archive.namelist())
    payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in range(1, 8):
        member_2d = _coordinate_member(sequence, 2, frame, camera)
        member_3d = _coordinate_member(sequence, 3, frame, camera)
        for member in (member_2d, member_3d):
            _safe_member(member)
            if member not in names:
                raise ValueError(f"registered DOT marker member is missing: {member}")
        raw_2d = archive.read(member_2d)
        raw_3d = archive.read(member_3d)
        opened_members.extend(
            (
                {
                    "sequence": sequence,
                    "camera": camera,
                    "frame": frame,
                    "kind": "2d",
                    "member": member_2d,
                    "byte_count": len(raw_2d),
                    "sha256": _sha256_bytes(raw_2d),
                },
                {
                    "sequence": sequence,
                    "camera": "cam001-shared-3d-carrier",
                    "frame": frame,
                    "kind": "3d",
                    "member": member_3d,
                    "byte_count": len(raw_3d),
                    "sha256": _sha256_bytes(raw_3d),
                },
            )
        )
        payloads[frame] = (
            pooled._parse_coordinate_text(raw_2d.decode("utf-8"), 2),
            pooled._parse_coordinate_text(raw_3d.decode("utf-8"), 3),
        )
    return payloads


def _collect_fit(
    pooled: ModuleType,
    run: Mapping[str, Any],
    payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    provider_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    for frame in frames:
        coordinates_2d, coordinates_3d = payloads[int(frame)]
        provider, truth, _ = pooled._sample_markers(
            run,
            int(frame),
            coordinates_2d,
            coordinates_3d,
        )
        if provider.size:
            provider_rows.append(provider)
            truth_rows.append(truth)
    if not provider_rows or sum(row.shape[0] for row in provider_rows) < 6:
        raise ValueError("fewer than six pooled markers remain for Sim(3) fitting")
    return np.concatenate(provider_rows), np.concatenate(truth_rows)


def _common_evaluation_markers(
    pooled: ModuleType,
    runs: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    anchor_frame: int,
    query_frame: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    coordinates_2d, coordinates_3d = payloads[query_frame]
    sampled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run_name in ("window_a", "window_b", "continuous"):
        points, _, indices = pooled._sample_markers(
            runs[run_name],
            query_frame,
            coordinates_2d,
            coordinates_3d,
        )
        sampled[run_name] = (points, indices)
    common = sampled["window_a"][1]
    for run_name in ("window_b", "continuous"):
        common = np.intersect1d(common, sampled[run_name][1], assume_unique=True)
    truth_anchor = np.asarray(payloads[anchor_frame][1], dtype=np.float64)
    truth_query = np.asarray(payloads[query_frame][1], dtype=np.float64)
    valid = common[
        (common < truth_anchor.shape[0])
        & (common < truth_query.shape[0])
        & np.isfinite(truth_anchor[common]).all(axis=1)
        & np.isfinite(truth_query[common]).all(axis=1)
    ]
    positions: dict[str, np.ndarray] = {}
    for run_name, (points, indices) in sampled.items():
        lookup = {int(index): position for position, index in enumerate(indices)}
        positions[run_name] = points[[lookup[int(index)] for index in valid]]
    return positions, truth_anchor[valid], truth_query[valid], valid


def _rope_span(truth: np.ndarray) -> float:
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    span = 2.0 * float(np.max(np.linalg.norm(centered, axis=1), initial=0.0))
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("rope span is degenerate")
    return span


def _extract_sequence_queries(
    *,
    sequence: str,
    camera: str,
    run_paths: Mapping[str, Path],
    archive: zipfile.ZipFile,
    pooled: ModuleType,
    protocol: Mapping[str, Any],
    opened_members: list[dict[str, Any]],
) -> SequenceQueries:
    payloads = _parse_sequence_payloads(
        archive,
        sequence=sequence,
        camera=camera,
        pooled=pooled,
        opened_members=opened_members,
    )
    runs = {
        name: _load_run(path, sequence=sequence, run_name=name)
        for name, path in run_paths.items()
    }
    evaluation = protocol["evaluation"]
    fit_a_source, fit_a_truth = _collect_fit(
        pooled, runs["window_a"], payloads, evaluation["fit_window_a_frames"]
    )
    fit_b_source, fit_b_truth = _collect_fit(
        pooled, runs["window_b"], payloads, evaluation["fit_window_b_frames"]
    )
    fit_c_source, fit_c_truth = _collect_fit(
        pooled, runs["continuous"], payloads, evaluation["fit_continuous_frames"]
    )
    transform_a, _ = robust_fit_sim3(fit_a_source, fit_a_truth)
    transform_b, _ = robust_fit_sim3(fit_b_source, fit_b_truth)
    transform_c, _ = robust_fit_sim3(fit_c_source, fit_c_truth)
    positions, truth_anchor, truth_query, indices = _common_evaluation_markers(
        pooled,
        runs,
        payloads,
        anchor_frame=int(evaluation["anchor_frame"]),
        query_frame=int(evaluation["query_frame"]),
    )
    minimum = int(evaluation["minimum_common_query_markers"])
    if indices.size < minimum:
        raise ValueError(
            f"{sequence} has {indices.size} common query markers; {minimum} required"
        )
    span = _rope_span(np.asarray(payloads[int(evaluation["anchor_frame"])][1]))
    truth_displacement = (truth_query - truth_anchor) / span
    estimate_a = (transform_a.apply(positions["window_a"]) - truth_anchor) / span
    estimate_b = (transform_b.apply(positions["window_b"]) - truth_anchor) / span
    estimate_c = (transform_c.apply(positions["continuous"]) - truth_anchor) / span
    for name, value in {
        "truth": truth_displacement,
        "window_a": estimate_a,
        "window_b": estimate_b,
        "continuous": estimate_c,
    }.items():
        if value.shape != truth_displacement.shape or not np.isfinite(value).all():
            raise ValueError(f"{sequence} {name} query array is malformed")
    rank = int(protocol["terminal_rank_diagnostic"]["factor_rank_by_sequence"][sequence])
    return SequenceQueries(
        sequence=sequence,
        camera=camera,
        factor_rank=rank,
        truth=truth_displacement,
        window_a=estimate_a,
        window_b=estimate_b,
        continuous=estimate_c,
        marker_indices=indices,
        rope_span=span,
        fit_a_count=int(fit_a_source.shape[0]),
        fit_b_count=int(fit_b_source.shape[0]),
        fit_continuous_count=int(fit_c_source.shape[0]),
    )


def _equal_sequence_mean(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one sequence array is required")
    dimension = arrays[0].shape[1]
    means = []
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != dimension or value.shape[0] == 0:
            raise ValueError("sequence arrays must be nonempty matrices with equal columns")
        means.append(np.mean(value, axis=0))
    return np.mean(np.stack(means), axis=0)


def _equal_sequence_second_moment(
    arrays: Sequence[np.ndarray], center: np.ndarray
) -> np.ndarray:
    center_value = np.asarray(center, dtype=np.float64)
    moments = []
    for array in arrays:
        residual = np.asarray(array, dtype=np.float64) - center_value[None, :]
        moments.append(residual.T @ residual / residual.shape[0])
    result = np.mean(np.stack(moments), axis=0)
    return 0.5 * (result + result.T)


def _regularize_covariance(
    covariance: np.ndarray,
    *,
    diagonal_shrinkage: float,
    absolute_floor: float,
) -> np.ndarray:
    raw = np.asarray(covariance, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    if not 0.0 <= diagonal_shrinkage <= 1.0:
        raise ValueError("diagonal shrinkage must lie in [0, 1]")
    symmetric = 0.5 * (raw + raw.T)
    diagonal = np.diag(np.maximum(np.diag(symmetric), 0.0))
    result = (1.0 - diagonal_shrinkage) * symmetric + diagonal_shrinkage * diagonal
    values, vectors = np.linalg.eigh(0.5 * (result + result.T))
    floor = max(float(absolute_floor), np.finfo(np.float64).eps)
    values = np.maximum(values, floor)
    regularized = (vectors * values[None, :]) @ vectors.T
    regularized = 0.5 * (regularized + regularized.T)
    np.linalg.cholesky(regularized)
    return regularized


def _joint_covariance_with_bounded_correlation(
    covariance_a: np.ndarray,
    covariance_b: np.ndarray,
    cross: np.ndarray,
    *,
    cross_shrinkage: float,
    maximum_canonical_correlation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= cross_shrinkage <= 1.0:
        raise ValueError("cross shrinkage must lie in [0, 1]")
    if not 0.0 < maximum_canonical_correlation < 1.0:
        raise ValueError("maximum canonical correlation must lie in (0, 1)")
    root_a = np.linalg.cholesky(covariance_a)
    root_b = np.linalg.cholesky(covariance_b)
    shrunk = (1.0 - cross_shrinkage) * np.asarray(cross, dtype=np.float64)
    whitened = np.linalg.solve(root_a, shrunk)
    whitened = np.linalg.solve(root_b, whitened.T).T
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    clipped = np.minimum(singular, maximum_canonical_correlation)
    bounded_cross = root_a @ ((left * clipped[None, :]) @ right) @ root_b.T
    joint = np.block(
        [[covariance_a, bounded_cross], [bounded_cross.T, covariance_b]]
    )
    joint = 0.5 * (joint + joint.T)
    np.linalg.cholesky(joint)
    return joint, bounded_cross, clipped


def _fit_model(
    training: Sequence[SequenceQueries], protocol: Mapping[str, Any]
) -> GaussianModel:
    if len(training) < int(protocol["calibration"]["minimum_training_sequences"]):
        raise ValueError("too few training sequences for leave-one-sequence-out calibration")
    truth_arrays = [row.truth for row in training]
    error_a = [row.window_a - row.truth for row in training]
    error_b = [row.window_b - row.truth for row in training]
    error_c = [row.continuous - row.truth for row in training]
    bias_a = _equal_sequence_mean(error_a)
    bias_b = _equal_sequence_mean(error_b)
    bias_c = _equal_sequence_mean(error_c)
    prior_second = _equal_sequence_second_moment(truth_arrays, np.zeros(3))
    scale = max(float(np.trace(prior_second) / 3.0), 1.0e-12)
    calibration = protocol["calibration"]
    prior = _regularize_covariance(
        prior_second,
        diagonal_shrinkage=float(calibration["prior_diagonal_shrinkage"]),
        absolute_floor=float(calibration["prior_variance_floor_fraction"]) * scale,
    )
    noise_floor = float(calibration["noise_variance_floor_fraction_of_prior"]) * float(
        np.trace(prior) / 3.0
    )
    covariance_a = _regularize_covariance(
        _equal_sequence_second_moment(error_a, bias_a),
        diagonal_shrinkage=float(calibration["noise_diagonal_shrinkage"]),
        absolute_floor=noise_floor,
    )
    covariance_b = _regularize_covariance(
        _equal_sequence_second_moment(error_b, bias_b),
        diagonal_shrinkage=float(calibration["noise_diagonal_shrinkage"]),
        absolute_floor=noise_floor,
    )
    covariance_c = _regularize_covariance(
        _equal_sequence_second_moment(error_c, bias_c),
        diagonal_shrinkage=float(calibration["noise_diagonal_shrinkage"]),
        absolute_floor=noise_floor,
    )
    joint_errors = [np.concatenate((a, b), axis=1) for a, b in zip(error_a, error_b)]
    joint_bias = np.concatenate((bias_a, bias_b))
    raw_joint = _equal_sequence_second_moment(joint_errors, joint_bias)
    raw_cross = raw_joint[:3, 3:]
    joint, bounded_cross, canonical = _joint_covariance_with_bounded_correlation(
        covariance_a,
        covariance_b,
        raw_cross,
        cross_shrinkage=float(calibration["cross_covariance_shrinkage"]),
        maximum_canonical_correlation=float(
            calibration["maximum_canonical_correlation"]
        ),
    )
    return GaussianModel(
        prior_covariance=prior,
        bias_a=bias_a,
        bias_b=bias_b,
        bias_continuous=bias_c,
        covariance_a=covariance_a,
        covariance_b=covariance_b,
        covariance_continuous=covariance_c,
        cross_covariance_ab=bounded_cross,
        joint_covariance_ab=joint,
        whitened_cross_singular_values=canonical,
    )


def _posterior_identity_measurement(
    prior_covariance: np.ndarray,
    observation: np.ndarray,
    observation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prior_precision = np.linalg.inv(prior_covariance)
    noise_precision = np.linalg.inv(observation_covariance)
    posterior_covariance = np.linalg.inv(prior_precision + noise_precision)
    posterior_mean = posterior_covariance @ noise_precision @ observation
    return posterior_mean, 0.5 * (posterior_covariance + posterior_covariance.T)


def _posterior_joint_measurement(
    prior_covariance: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    joint_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.vstack((np.eye(3), np.eye(3)))
    precision = np.linalg.inv(joint_covariance)
    prior_precision = np.linalg.inv(prior_covariance)
    posterior_covariance = np.linalg.inv(
        prior_precision + design.T @ precision @ design
    )
    stacked = np.concatenate((first, second))
    posterior_mean = posterior_covariance @ design.T @ precision @ stacked
    return posterior_mean, 0.5 * (posterior_covariance + posterior_covariance.T)


def _posterior_object(
    prior_covariance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
) -> GaussianQueryPosterior:
    prior_mean = np.zeros(3, dtype=np.float64)
    reduction = 0.5 * (
        prior_covariance
        - posterior_covariance
        + (prior_covariance - posterior_covariance).T
    )
    return GaussianQueryPosterior(
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        mean_shift=posterior_mean,
        covariance_reduction=reduction,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        innovation_precision_quadratic=0.0,
        innovation_log_determinant=0.0,
        innovation_negative_log_likelihood=0.0,
        observation_dimension=3,
    )


def _message(
    *,
    prior_covariance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    sequence: str,
    marker_index: int,
    run: str,
) -> GaussianQueryMessage:
    return compress_gaussian_query_posterior(
        _posterior_object(prior_covariance, posterior_mean, posterior_covariance),
        query_id="dot-frame3-to-frame4-material-point-displacement-normalized-by-rope-span",
        prior_id=f"dot-loso-prior-held-{sequence}",
        evidence_ids=(f"sealed-cut3r:{sequence}:{run}:frame4:marker-{marker_index}",),
        parity_relative_tolerance=1.0e-9,
    )


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return np.asarray(
        [0.5 * (1.0 + math.erf(float(item) / math.sqrt(2.0))) for item in value],
        dtype=np.float64,
    )


def _probability_update_beats_fallback(
    mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    norm_squared = np.sum(mean * mean, axis=1)
    projected = np.einsum("ni,ij,nj->n", mean, covariance, mean, optimize=True)
    denominator = 2.0 * np.sqrt(np.maximum(projected, 0.0))
    score = np.zeros_like(norm_squared)
    regular = denominator > np.finfo(np.float64).eps
    score[regular] = norm_squared[regular] / denominator[regular]
    score[~regular & (norm_squared > 0.0)] = np.inf
    return _normal_cdf(score)


def _method_metrics(
    truth: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    execute_probability: float,
) -> dict[str, Any]:
    errors = truth - mean
    root = np.linalg.cholesky(0.5 * (covariance + covariance.T))
    whitened = np.linalg.solve(root, errors.T)
    squared_mahalanobis = np.sum(whitened * whitened, axis=0)
    logdet = 2.0 * float(np.log(np.diag(root)).sum())
    nll = 0.5 * (
        squared_mahalanobis + logdet + 3.0 * math.log(2.0 * math.pi)
    )
    fallback_squared = np.sum(truth * truth, axis=1)
    update_squared = np.sum(errors * errors, axis=1)
    probability = _probability_update_beats_fallback(mean, covariance)
    execute = probability >= execute_probability
    deployed = np.where(execute[:, None], mean, 0.0)
    deployed_error = truth - deployed
    deployed_squared = np.sum(deployed_error * deployed_error, axis=1)
    harmful_execute = execute & (update_squared > fallback_squared)
    return {
        "sample_count": int(truth.shape[0]),
        "rmse_per_coordinate_fraction_of_span": float(
            math.sqrt(float(np.mean(errors * errors)))
        ),
        "vector_rmse_fraction_of_span": float(
            math.sqrt(float(np.mean(update_squared)))
        ),
        "mean_gaussian_nll_per_dimension": float(np.mean(nll) / 3.0),
        "normalized_nees": float(np.mean(squared_mahalanobis) / 3.0),
        "coverage_90": float(np.mean(squared_mahalanobis <= CHI2_3_90)),
        "mean_marginal_interval_width_fraction_of_span": float(
            2.0 * NORMAL_90 * np.mean(np.sqrt(np.diag(covariance)))
        ),
        "harmful_update_fraction": float(np.mean(update_squared > fallback_squared)),
        "execute_fraction": float(np.mean(execute)),
        "harmful_execute_fraction_all": float(np.mean(harmful_execute)),
        "harmful_execute_fraction_conditional": (
            float(np.mean(harmful_execute[execute])) if np.any(execute) else 0.0
        ),
        "deployed_decision_rmse_per_coordinate_fraction_of_span": float(
            math.sqrt(float(np.mean(deployed_error * deployed_error)))
        ),
        "deployed_decision_loss_per_coordinate": float(
            np.mean(deployed_squared) / 3.0
        ),
        "mean_probability_update_beats_fallback": float(np.mean(probability)),
    }


def _evaluate_held_sequence(
    row: SequenceQueries,
    model: GaussianModel,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    truth = row.truth
    corrected_a = row.window_a - model.bias_a[None, :]
    corrected_b = row.window_b - model.bias_b[None, :]
    corrected_c = row.continuous - model.bias_continuous[None, :]
    count = truth.shape[0]
    means: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "physical_fallback",
            "window_a_only",
            "window_b_only",
            "continuous_only",
            "two_window_query_ci_equal",
            "pairwise_logdet_ci",
            "naive_independent_message_sum",
            "diagonal_joint_covariance",
            "dense_joint_correlated",
        )
    }
    covariances: dict[str, np.ndarray] = {}
    covariance_a_post = _posterior_identity_measurement(
        model.prior_covariance, np.zeros(3), model.covariance_a
    )[1]
    covariance_b_post = _posterior_identity_measurement(
        model.prior_covariance, np.zeros(3), model.covariance_b
    )[1]
    covariance_c_post = _posterior_identity_measurement(
        model.prior_covariance, np.zeros(3), model.covariance_continuous
    )[1]
    covariance_dense = _posterior_joint_measurement(
        model.prior_covariance,
        np.zeros(3),
        np.zeros(3),
        model.joint_covariance_ab,
    )[1]
    covariance_naive = _posterior_joint_measurement(
        model.prior_covariance,
        np.zeros(3),
        np.zeros(3),
        np.block(
            [
                [model.covariance_a, np.zeros((3, 3))],
                [np.zeros((3, 3)), model.covariance_b],
            ]
        ),
    )[1]
    diagonal_joint = np.diag(np.diag(model.joint_covariance_ab))
    covariance_diagonal = _posterior_joint_measurement(
        model.prior_covariance,
        np.zeros(3),
        np.zeros(3),
        diagonal_joint,
    )[1]
    covariances.update(
        {
            "physical_fallback": model.prior_covariance,
            "window_a_only": covariance_a_post,
            "window_b_only": covariance_b_post,
            "continuous_only": covariance_c_post,
            "naive_independent_message_sum": covariance_naive,
            "diagonal_joint_covariance": covariance_diagonal,
            "dense_joint_correlated": covariance_dense,
        }
    )

    maximum_message_mean_error = 0.0
    maximum_message_covariance_error = 0.0
    maximum_duplicate_mean_error = 0.0
    maximum_duplicate_covariance_error = 0.0
    equal_ci_weights = tuple(float(value) for value in protocol["fusion"]["equal_ci_weights"])
    selected_weights: list[tuple[float, float]] = []
    equal_ci_covariance: np.ndarray | None = None
    selected_ci_covariance: np.ndarray | None = None
    message_payload_nbytes: int | None = None
    message_anchored_nbytes: int | None = None

    for sample_index in range(count):
        mean_a, direct_cov_a = _posterior_identity_measurement(
            model.prior_covariance, corrected_a[sample_index], model.covariance_a
        )
        mean_b, direct_cov_b = _posterior_identity_measurement(
            model.prior_covariance, corrected_b[sample_index], model.covariance_b
        )
        mean_c, direct_cov_c = _posterior_identity_measurement(
            model.prior_covariance,
            corrected_c[sample_index],
            model.covariance_continuous,
        )
        marker = int(row.marker_indices[sample_index])
        message_a = _message(
            prior_covariance=model.prior_covariance,
            posterior_mean=mean_a,
            posterior_covariance=direct_cov_a,
            sequence=row.sequence,
            marker_index=marker,
            run="window_a",
        )
        message_b = _message(
            prior_covariance=model.prior_covariance,
            posterior_mean=mean_b,
            posterior_covariance=direct_cov_b,
            sequence=row.sequence,
            marker_index=marker,
            run="window_b",
        )
        belief_a = apply_gaussian_query_message(message_a)
        belief_b = apply_gaussian_query_message(message_b)
        maximum_message_mean_error = max(
            maximum_message_mean_error,
            float(np.max(np.abs(belief_a.mean - mean_a))),
            float(np.max(np.abs(belief_b.mean - mean_b))),
        )
        maximum_message_covariance_error = max(
            maximum_message_covariance_error,
            float(np.max(np.abs(belief_a.covariance - direct_cov_a))),
            float(np.max(np.abs(belief_b.covariance - direct_cov_b))),
        )
        equal_message = fuse_gaussian_query_messages_covariance_intersection(
            (message_a, message_b),
            weights=equal_ci_weights,
            construction_id="dot-source-equal-query-ci-v1",
        )
        equal_belief = apply_gaussian_query_message(equal_message)
        selected_message = select_pairwise_covariance_intersection(
            message_a,
            message_b,
            grid_size=int(protocol["fusion"]["pairwise_grid_size"]),
            objective="logdet",
        )
        selected_belief = apply_gaussian_query_message(selected_message)
        weight_by_id = dict(
            zip(
                selected_message.component_message_ids,
                selected_message.component_weights,
                strict=True,
            )
        )
        selected_weights.append(
            (weight_by_id.get(message_a.message_id, 0.0), weight_by_id.get(message_b.message_id, 0.0))
        )
        if equal_ci_covariance is None:
            equal_ci_covariance = equal_belief.covariance
            selected_ci_covariance = selected_belief.covariance
            message_payload_nbytes = message_a.payload_nbytes
            message_anchored_nbytes = message_a.anchored_storage_nbytes
        else:
            np.testing.assert_allclose(equal_belief.covariance, equal_ci_covariance)
            np.testing.assert_allclose(selected_belief.covariance, selected_ci_covariance)

        duplicate = fuse_gaussian_query_messages_covariance_intersection(
            (message_a, message_a),
            weights=(0.5, 0.5),
            construction_id="dot-source-duplicate-control-v1",
        )
        duplicate_belief = apply_gaussian_query_message(duplicate)
        maximum_duplicate_mean_error = max(
            maximum_duplicate_mean_error,
            float(np.max(np.abs(duplicate_belief.mean - belief_a.mean))),
        )
        maximum_duplicate_covariance_error = max(
            maximum_duplicate_covariance_error,
            float(np.max(np.abs(duplicate_belief.covariance - belief_a.covariance))),
        )

        dense_mean, _ = _posterior_joint_measurement(
            model.prior_covariance,
            corrected_a[sample_index],
            corrected_b[sample_index],
            model.joint_covariance_ab,
        )
        naive_mean, _ = _posterior_joint_measurement(
            model.prior_covariance,
            corrected_a[sample_index],
            corrected_b[sample_index],
            np.block(
                [
                    [model.covariance_a, np.zeros((3, 3))],
                    [np.zeros((3, 3)), model.covariance_b],
                ]
            ),
        )
        diagonal_mean, _ = _posterior_joint_measurement(
            model.prior_covariance,
            corrected_a[sample_index],
            corrected_b[sample_index],
            diagonal_joint,
        )
        sample_means = {
            "physical_fallback": np.zeros(3),
            "window_a_only": mean_a,
            "window_b_only": mean_b,
            "continuous_only": mean_c,
            "two_window_query_ci_equal": equal_belief.mean,
            "pairwise_logdet_ci": selected_belief.mean,
            "naive_independent_message_sum": naive_mean,
            "diagonal_joint_covariance": diagonal_mean,
            "dense_joint_correlated": dense_mean,
        }
        for name, value in sample_means.items():
            means[name].append(np.asarray(value, dtype=np.float64))

    assert equal_ci_covariance is not None
    assert selected_ci_covariance is not None
    assert message_payload_nbytes is not None
    assert message_anchored_nbytes is not None
    covariances["two_window_query_ci_equal"] = equal_ci_covariance
    covariances["pairwise_logdet_ci"] = selected_ci_covariance
    execute_probability = float(protocol["decision"]["minimum_improvement_probability"])
    metrics = {
        name: _method_metrics(
            truth,
            np.stack(method_means),
            covariances[name],
            execute_probability=execute_probability,
        )
        for name, method_means in means.items()
    }
    weights_array = np.asarray(selected_weights, dtype=np.float64)
    sequence_result = {
        "sequence": row.sequence,
        "camera": row.camera,
        "factor_rank": row.factor_rank,
        "query_marker_count": int(count),
        "marker_indices": row.marker_indices.tolist(),
        "rope_span_dataset_units": row.rope_span,
        "fit_support": {
            "window_a": row.fit_a_count,
            "window_b": row.fit_b_count,
            "continuous": row.fit_continuous_count,
        },
        "calibration": {
            "training_sequence_count": len(SOURCE_SEQUENCES) - 1,
            "prior_covariance": model.prior_covariance.tolist(),
            "window_a_bias": model.bias_a.tolist(),
            "window_b_bias": model.bias_b.tolist(),
            "continuous_bias": model.bias_continuous.tolist(),
            "window_a_covariance": model.covariance_a.tolist(),
            "window_b_covariance": model.covariance_b.tolist(),
            "continuous_covariance": model.covariance_continuous.tolist(),
            "window_ab_cross_covariance": model.cross_covariance_ab.tolist(),
            "window_ab_canonical_correlations": model.whitened_cross_singular_values.tolist(),
        },
        "message_contract": {
            "maximum_single_message_mean_error": maximum_message_mean_error,
            "maximum_single_message_covariance_error": maximum_message_covariance_error,
            "maximum_duplicate_mean_error": maximum_duplicate_mean_error,
            "maximum_duplicate_covariance_error": maximum_duplicate_covariance_error,
            "single_message_payload_nbytes": message_payload_nbytes,
            "single_message_anchored_storage_nbytes": message_anchored_nbytes,
            "pairwise_logdet_ci_mean_window_a_weight": float(np.mean(weights_array[:, 0])),
            "pairwise_logdet_ci_mean_window_b_weight": float(np.mean(weights_array[:, 1])),
        },
        "methods": metrics,
    }
    parity = {
        "single_mean": maximum_message_mean_error,
        "single_covariance": maximum_message_covariance_error,
        "duplicate_mean": maximum_duplicate_mean_error,
        "duplicate_covariance": maximum_duplicate_covariance_error,
    }
    return sequence_result, parity


def _aggregate(sequence_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    method_names = tuple(sequence_results[0]["methods"])
    methods: dict[str, Any] = {}
    scalar_fields = tuple(sequence_results[0]["methods"][method_names[0]])
    for method in method_names:
        rows = [result["methods"][method] for result in sequence_results]
        aggregate_row: dict[str, Any] = {"sequence_count": len(rows)}
        for field in scalar_fields:
            values = [float(row[field]) for row in rows]
            aggregate_row[field] = float(np.mean(values))
            aggregate_row[f"{field}_minimum"] = float(np.min(values))
            aggregate_row[f"{field}_maximum"] = float(np.max(values))
        methods[method] = aggregate_row
    ranks: dict[str, Any] = {}
    for rank in sorted({int(row["factor_rank"]) for row in sequence_results}):
        selected = [row for row in sequence_results if int(row["factor_rank"]) == rank]
        ranks[str(rank)] = {
            "sequence_count": len(selected),
            "sequences": [row["sequence"] for row in selected],
            "methods": {
                method: {
                    "mean_gaussian_nll_per_dimension": float(
                        np.mean(
                            [
                                row["methods"][method]["mean_gaussian_nll_per_dimension"]
                                for row in selected
                            ]
                        )
                    ),
                    "coverage_90": float(
                        np.mean([row["methods"][method]["coverage_90"] for row in selected])
                    ),
                }
                for method in method_names
            },
        }
    return {"methods": methods, "factor_rank_strata": ranks}


def _bootstrap_difference(
    sequence_results: Sequence[Mapping[str, Any]],
    *,
    first: str,
    second: str,
    field: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    differences = np.asarray(
        [
            float(row["methods"][first][field])
            - float(row["methods"][second][field])
            for row in sequence_results
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(replicates, differences.size))
    sampled = np.mean(differences[indices], axis=1)
    return {
        "estimate": float(np.mean(differences)),
        "lower_95": float(np.quantile(sampled, 0.025)),
        "upper_95": float(np.quantile(sampled, 0.975)),
        "wins": int(np.count_nonzero(differences < 0.0)),
        "ties": int(np.count_nonzero(differences == 0.0)),
        "losses": int(np.count_nonzero(differences > 0.0)),
    }


def _source_decision(
    aggregate: Mapping[str, Any],
    parity: Mapping[str, float],
    protocol: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    criteria = protocol["source_decision_criteria"]
    methods = aggregate["methods"]
    ci = methods["two_window_query_ci_equal"]
    naive = methods["naive_independent_message_sum"]
    best_single_nll = min(
        methods["window_a_only"]["mean_gaussian_nll_per_dimension"],
        methods["window_b_only"]["mean_gaussian_nll_per_dimension"],
    )
    best_single_decision = min(
        methods["window_a_only"]["deployed_decision_loss_per_coordinate"],
        methods["window_b_only"]["deployed_decision_loss_per_coordinate"],
    )
    checks = {
        "all_sequences_supported": int(aggregate["sequence_count"])
        >= int(criteria["minimum_supported_sequences"]),
        "single_message_parity": max(parity["single_mean"], parity["single_covariance"])
        <= float(criteria["maximum_message_parity_error"]),
        "duplicate_idempotence": max(parity["duplicate_mean"], parity["duplicate_covariance"])
        <= float(criteria["maximum_duplicate_parity_error"]),
        "ci_nll_not_worse_than_naive": ci["mean_gaussian_nll_per_dimension"]
        <= naive["mean_gaussian_nll_per_dimension"]
        + float(criteria["nll_numerical_tolerance"]),
        "ci_coverage_closer_to_nominal_than_naive": abs(ci["coverage_90"] - 0.9)
        <= abs(naive["coverage_90"] - 0.9)
        + float(criteria["coverage_numerical_tolerance"]),
        "ci_harm_not_worse_than_naive": ci["harmful_execute_fraction_all"]
        <= naive["harmful_execute_fraction_all"]
        + float(criteria["harm_numerical_tolerance"]),
        "ci_retains_single_window_utility": ci["mean_gaussian_nll_per_dimension"]
        <= best_single_nll + float(criteria["maximum_nll_regret_vs_best_single"]),
        "ci_decision_loss_near_best_single": ci["deployed_decision_loss_per_coordinate"]
        <= (
            float(criteria["maximum_decision_loss_ratio_vs_best_single"])
            * best_single_decision
            + float(criteria["decision_loss_numerical_tolerance"])
        ),
    }
    if all(checks.values()):
        return "source-real-overlap-positive", checks
    if (
        checks["all_sequences_supported"]
        and checks["single_message_parity"]
        and checks["duplicate_idempotence"]
    ):
        return "source-real-overlap-mixed", checks
    return "source-real-overlap-negative", checks


def _summary_markdown(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    methods = aggregate["methods"]
    lines = [
        "# DOT R11--R20 query-message real-source evaluation",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"Sequences: {aggregate['sequence_count']} (equal sequence weight)",
        f"Material-point query samples: {aggregate['query_sample_count']}",
        "",
        "| Method | RMSE/coord [% span] | NLL/dim | 90% coverage | nNEES | "
        "Execute [%] | Harmful execute [% all] | Deployed RMSE/coord [% span] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "physical_fallback",
        "window_a_only",
        "window_b_only",
        "continuous_only",
        "dense_joint_correlated",
        "two_window_query_ci_equal",
        "pairwise_logdet_ci",
        "naive_independent_message_sum",
        "diagonal_joint_covariance",
    ):
        row = methods[name]
        lines.append(
            "| {name} | {rmse:.3f} | {nll:.4f} | {coverage:.2f} | {nees:.3f} | "
            "{execute:.2f} | {harm:.2f} | {deployed:.3f} |".format(
                name=name,
                rmse=100.0 * row["rmse_per_coordinate_fraction_of_span"],
                nll=row["mean_gaussian_nll_per_dimension"],
                coverage=100.0 * row["coverage_90"],
                nees=row["normalized_nees"],
                execute=100.0 * row["execute_fraction"],
                harm=100.0 * row["harmful_execute_fraction_all"],
                deployed=100.0
                * row["deployed_decision_rmse_per_coordinate_fraction_of_span"],
            )
        )
    lines.extend(
        [
            "",
            "The covariance model is fitted leave-one-sequence-out from the other "
            "nine already-open source sequences. It is not emitted by CUT3R and is "
            "not a provider-calibration claim.",
            "",
            "R21--R70 were not opened. This result is source/development evidence and "
            "cannot by itself support a held-out, deployment, or safety claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    dataset_root: Path,
    provider_root: Path,
    protocol_path: Path,
    request_path: Path,
    output_dir: Path,
    source_revision: str,
    protocol_git_blob_sha1: str,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = _load_protocol(protocol_path)
    request = _load_request(request_path, protocol, protocol_git_blob_sha1)
    _hex(source_revision, name="source revision", length=40)
    archive_path = (dataset_root / ARCHIVE_NAME).resolve(strict=True)
    archive_path.relative_to(dataset_root.resolve(strict=True))
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("official R11-20.zip must be a regular non-symlink file")
    if _md5_file(archive_path) != ARCHIVE_MD5:
        raise ValueError("official R11-20.zip MD5 mismatch")
    route, records, verified_outputs = _verified_provider_records(provider_root, protocol)
    pooled = _load_pooled_evaluator(protocol)
    opened_members: list[dict[str, Any]] = []
    extracted: list[SequenceQueries] = []
    extraction_failures: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for sequence in SOURCE_SEQUENCES:
            try:
                extracted.append(
                    _extract_sequence_queries(
                        sequence=sequence,
                        camera=route[sequence],
                        run_paths=records[sequence],
                        archive=archive,
                        pooled=pooled,
                        protocol=protocol,
                        opened_members=opened_members,
                    )
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                extraction_failures.append(
                    {
                        "sequence": sequence,
                        "error_type": type(error).__name__,
                        "error": " ".join(str(error).split()),
                    }
                )
    minimum_supported = int(protocol["source_decision_criteria"]["minimum_supported_sequences"])
    if len(extracted) < minimum_supported:
        raise ValueError(
            f"only {len(extracted)} sequences support the real query; {minimum_supported} required"
        )

    sequence_results: list[dict[str, Any]] = []
    parity_maxima = {
        "single_mean": 0.0,
        "single_covariance": 0.0,
        "duplicate_mean": 0.0,
        "duplicate_covariance": 0.0,
    }
    for held in extracted:
        training = [row for row in extracted if row.sequence != held.sequence]
        model = _fit_model(training, protocol)
        result, parity = _evaluate_held_sequence(held, model, protocol)
        sequence_results.append(result)
        for name in parity_maxima:
            parity_maxima[name] = max(parity_maxima[name], float(parity[name]))

    aggregate = _aggregate(sequence_results)
    aggregate["sequence_count"] = len(sequence_results)
    aggregate["query_sample_count"] = int(
        sum(int(row["query_marker_count"]) for row in sequence_results)
    )
    decision, checks = _source_decision(aggregate, parity_maxima, protocol)
    bootstrap = {}
    for first, second in (
        ("two_window_query_ci_equal", "naive_independent_message_sum"),
        ("dense_joint_correlated", "naive_independent_message_sum"),
        ("two_window_query_ci_equal", "window_a_only"),
        ("two_window_query_ci_equal", "window_b_only"),
        ("two_window_query_ci_equal", "physical_fallback"),
    ):
        key = f"{first}-minus-{second}"
        bootstrap[key] = {
            field: _bootstrap_difference(
                sequence_results,
                first=first,
                second=second,
                field=field,
                replicates=int(protocol["statistics"]["bootstrap_replicates"]),
                seed=int(protocol["statistics"]["bootstrap_seed"])
                + sum(ord(character) for character in key + field),
            )
            for field in (
                "mean_gaussian_nll_per_dimension",
                "rmse_per_coordinate_fraction_of_span",
                "deployed_decision_loss_per_coordinate",
                "harmful_execute_fraction_all",
            )
        }

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "checks": checks,
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "source_revision": source_revision,
        "provider": {
            "workflow_run_id": PROVIDER_RUN_ID,
            "artifact_id": PROVIDER_ARTIFACT_ID,
            "artifact_digest": PROVIDER_ARTIFACT_DIGEST,
            "provider_seal_id": PROVIDER_SEAL_ID,
            "verified_output_count": len(verified_outputs),
            "provider_covariance_present": False,
            "covariance_source": "leave-one-sequence-out empirical marker errors",
        },
        "dataset": {
            "archive": ARCHIVE_NAME,
            "archive_md5": ARCHIVE_MD5,
            "archive_sha256": _sha256_file(archive_path),
            "source_sequences": list(SOURCE_SEQUENCES),
            "supported_sequences": [row.sequence for row in extracted],
            "extraction_failures": extraction_failures,
            "confirmation_sequences_opened": False,
            "reserved_sequences_opened": False,
        },
        "query": {
            "definition": (
                "material-marker displacement from frame 3 to frame 4, normalized by "
                "the frame-3 rope span"
            ),
            "physical_fallback": "exact zero displacement (persistence)",
            "window_a_alignment_frames": protocol["evaluation"]["fit_window_a_frames"],
            "window_b_alignment_frames": protocol["evaluation"]["fit_window_b_frames"],
            "continuous_alignment_frames": protocol["evaluation"]["fit_continuous_frames"],
            "query_frame": protocol["evaluation"]["query_frame"],
            "anchor_frame": protocol["evaluation"]["anchor_frame"],
        },
        "message_contract_maxima": parity_maxima,
        "sequence_results": sequence_results,
        "aggregate": aggregate,
        "paired_sequence_bootstrap": bootstrap,
        "opened_marker_members": opened_members,
        "marker_sampling_diagnostics": sorted(
            pooled._MARKER_DIAGNOSTICS.values(),
            key=lambda row: (row["sequence"], row["run"], row["frame"]),
        ),
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "information_boundary": {
            "sealed_provider_predictions_reused": True,
            "provider_inference_rerun": False,
            "source_2d_markers_opened": True,
            "source_3d_markers_opened": True,
            "opened_sequences": list(SOURCE_SEQUENCES),
            "r21_r30_payloads_opened": False,
            "r31_r70_payloads_opened": False,
            "target_side_retuning_allowed": False,
            "bayesian_phystwin_executed": False,
            "causal4d_executed": False,
            "raw_provider_or_marker_payloads_written_to_evidence": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "summary.md").write_text(
        _summary_markdown(result), encoding="utf-8", newline="\n"
    )
    (output_dir / "protocol.json").write_bytes(protocol_path.read_bytes())
    (output_dir / "request.json").write_bytes(request_path.read_bytes())
    manifest = {
        "schema": "prob4d.dot-r11-r20-query-message-real-source-manifest",
        "schema_version": 1,
        "result_id": result["result_id"],
        "protocol_id": protocol["protocol_id"],
        "request_id": request["request_id"],
        "source_revision": source_revision,
        "result_sha256": _sha256_bytes(result_bytes),
        "protocol_sha256": _sha256_file(protocol_path),
        "request_sha256": _sha256_file(request_path),
        "opened_marker_manifest_sha256": _sha256_bytes(_canonical(opened_members)),
        "raw_data_copied_to_evidence": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    timing = {
        "schema": "prob4d.dot-r11-r20-query-message-real-source-timing",
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        "non_authoritative": True,
    }
    _write_json(output_dir / "timing.json", timing)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-request")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--protocol-git-blob-sha1", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--provider-root", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--source-revision", required=True)
    evaluate.add_argument("--protocol-git-blob-sha1", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-request":
        value = validate_request(
            protocol_path=args.protocol,
            request_path=args.request,
            protocol_git_blob_sha1=args.protocol_git_blob_sha1,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "evaluate":
        result = run(
            dataset_root=args.dataset_root.resolve(strict=True),
            provider_root=args.provider_root.resolve(strict=True),
            protocol_path=args.protocol,
            request_path=args.request,
            output_dir=args.output_dir,
            source_revision=args.source_revision,
            protocol_git_blob_sha1=args.protocol_git_blob_sha1,
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "result_id": result["result_id"],
                    "supported_sequences": len(result["dataset"]["supported_sequences"]),
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
