#!/usr/bin/env python3
"""Retrospective Deform360 finite-orbit physical-query pilot.

The protocol is frozen before this script first opens numerical target geometry.
It uses complete object identities as statistical units and compares the same
candidate mean under local, independent-orbit and shared-orbit admission rules.
The result is explicitly a public-real-data development pilot, not fresh
confirmation or deployment evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import struct
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

PROTOCOL_ID = "deform360-finite-orbit-real-data-v1"
QUERIES = ("span_change", "centroid_axis_progress", "named_endpoint_axis_progress")
ARMS = ("fallback", "ungated_candidate", "local_canonical", "independent_orbit", "shared_orbit")


@dataclass(frozen=True)
class SequenceUnit:
    object_id: str
    role: str
    episode: int
    carrier: str
    endpoints: np.ndarray
    anisotropy: np.ndarray
    span: np.ndarray


@dataclass(frozen=True)
class Window:
    object_id: str
    role: str
    query: str
    horizon: int
    frame: int
    gauge: int
    actual: float
    candidate: float
    fallback: float
    candidate_error: float
    fallback_error: float
    advantage: float
    orbit_diameter: float


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(token) if token.isdigit() else token.casefold() for token in re.split(r"(\d+)", value))


def canonical_axis(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    axis = vector / norm
    pivot = int(np.argmax(np.abs(axis)))
    return -axis if axis[pivot] < 0 else axis


def trajectory_orientation(array: np.ndarray, min_frames: int, min_points: int) -> np.ndarray | None:
    raw = np.asarray(array)
    coord_axes = [index for index, size in enumerate(raw.shape) if size == 3]
    if raw.ndim != 3 or not coord_axes:
        return None
    candidates: list[tuple[float, np.ndarray]] = []
    for coord_axis in coord_axes:
        xyz = np.moveaxis(raw, coord_axis, -1)
        for swap in (False, True):
            seq = np.swapaxes(xyz, 0, 1) if swap else xyz
            frames, points = seq.shape[:2]
            if frames < min_frames or points < min_points or frames > 20000 or points > 2_000_000:
                continue
            frame_ids = np.linspace(0, frames - 1, min(frames, 24), dtype=int)
            point_ids = np.linspace(0, points - 1, min(points, 800), dtype=int)
            sample = np.asarray(seq[np.ix_(frame_ids, point_ids, np.arange(3))], dtype=float)
            finite = np.isfinite(sample).all(axis=2)
            centers = []
            radii = []
            for frame, mask in zip(sample, finite):
                valid = frame[mask]
                if len(valid) < min_points:
                    centers.append(np.full(3, np.nan)); radii.append(np.nan); continue
                center = np.median(valid, axis=0)
                centers.append(center)
                radii.append(float(np.median(np.linalg.norm(valid - center, axis=1))))
            centers_arr = np.asarray(centers)
            motion = np.linalg.norm(np.diff(centers_arr, axis=0), axis=1)
            scale = max(float(np.nanmedian(radii)), 1e-9)
            smoothness = float(np.nanmedian(motion) / scale)
            shape_penalty = 0.0
            if frames < 5:
                shape_penalty += 10.0
            if points < 8:
                shape_penalty += 10.0
            candidates.append((smoothness + shape_penalty, seq))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def read_ply(path: Path) -> np.ndarray | None:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            return None
        fmt = ""
        count = 0
        in_vertex = False
        properties: list[tuple[str, str]] = []
        while True:
            line = handle.readline()
            if not line:
                return None
            text = line.decode("ascii", errors="replace").strip()
            parts = text.split()
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                count = int(parts[2]); in_vertex = True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif parts[:1] == ["property"] and in_vertex and len(parts) == 3:
                properties.append((parts[2], parts[1]))
            elif text == "end_header":
                break
        names = [name for name, _ in properties]
        if count <= 0 or not {"x", "y", "z"}.issubset(names):
            return None
        xyz_ids = [names.index(axis) for axis in "xyz"]
        if fmt == "ascii":
            rows = []
            for _ in range(count):
                values = handle.readline().split()
                if len(values) < len(properties):
                    break
                rows.append([float(values[index]) for index in xyz_ids])
            return np.asarray(rows, dtype=float)
        endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else None
        if endian is None:
            return None
        kinds = {
            "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
        }
        if any(kind not in kinds for _, kind in properties):
            return None
        dtype = np.dtype([(name, endian + kinds[kind]) for name, kind in properties])
        values = np.fromfile(handle, dtype=dtype, count=count)
        return np.column_stack([values[axis].astype(float) for axis in "xyz"])


def read_obj(path: Path) -> np.ndarray | None:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(rows, dtype=float) if rows else None


def read_pcd(path: Path) -> np.ndarray | None:
    with path.open("rb") as handle:
        header: dict[str, list[str]] = {}
        while True:
            line = handle.readline()
            if not line:
                return None
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                mode = parts[1].lower(); break
        fields = header.get("FIELDS", header.get("FIELD", []))
        if not {"x", "y", "z"}.issubset(fields):
            return None
        count = int((header.get("POINTS") or ["0"])[0])
        if mode != "ascii":
            return None
        data = np.loadtxt(handle, max_rows=count or None)
        if data.ndim == 1:
            data = data[None, :]
        return np.asarray(data[:, [fields.index(axis) for axis in "xyz"]], dtype=float)


def read_cloud(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.casefold() == ".ply":
            return read_ply(path)
        if path.suffix.casefold() == ".obj":
            return read_obj(path)
        if path.suffix.casefold() == ".pcd":
            return read_pcd(path)
    except (OSError, ValueError, KeyError, struct.error, MemoryError):
        return None
    return None


def numeric_arrays(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".npy":
            yield path.stem, np.load(path, mmap_mode="r", allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for key in sorted(archive.files):
                    value = archive[key]
                    if isinstance(value, np.ndarray):
                        yield key, value
        elif suffix in {".h5", ".hdf5"}:
            try:
                import h5py  # type: ignore
            except ImportError:
                return
            with h5py.File(path, "r") as handle:
                datasets: list[tuple[str, Any]] = []
                handle.visititems(lambda name, value: datasets.append((name, value)) if isinstance(value, h5py.Dataset) else None)
                for key, dataset in sorted(datasets):
                    if dataset.ndim == 3 and 3 in dataset.shape and int(np.prod(dataset.shape)) <= 200_000_000:
                        yield key, np.asarray(dataset)
    except (OSError, ValueError, MemoryError):
        return


def endpoint_frame(points: np.ndarray, tail_fraction: float) -> tuple[np.ndarray, float, float] | None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 8:
        return None
    if len(pts) > 20000:
        ids = np.linspace(0, len(pts) - 1, 20000, dtype=int)
        pts = pts[ids]
    center = np.median(pts, axis=0)
    centered = pts - center
    covariance = centered.T @ centered / max(len(pts) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    axis = canonical_axis(vectors[:, order[0]])
    projection = centered @ axis
    low, high = np.quantile(projection, [tail_fraction, 1.0 - tail_fraction])
    left = pts[projection <= low]
    right = pts[projection >= high]
    if not len(left) or not len(right):
        return None
    endpoints = np.stack([np.mean(left, axis=0), np.mean(right, axis=0)])
    span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    anisotropy = float(values[0] / max(values[1], 1e-15))
    if not np.isfinite(span) or span <= 1e-8:
        return None
    return endpoints, anisotropy, span


def endpoint_sequence(frames: Iterable[np.ndarray], tail_fraction: float, min_anisotropy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    endpoints: list[np.ndarray] = []
    anisotropy: list[float] = []
    spans: list[float] = []
    for points in frames:
        result = endpoint_frame(points, tail_fraction)
        if result is None:
            continue
        pair, ratio, span = result
        endpoints.append(pair); anisotropy.append(ratio); spans.append(span)
    if len(endpoints) < 8 or float(np.median(anisotropy)) < min_anisotropy:
        return None
    sequence = np.asarray(endpoints, dtype=float)
    first_axis = canonical_axis(sequence[0, 1] - sequence[0, 0])
    if float((sequence[0, 1] - sequence[0, 0]) @ first_axis) < 0:
        sequence[0] = sequence[0, ::-1]
    for index in range(1, len(sequence)):
        same = float(np.sum((sequence[index] - sequence[index - 1]) ** 2))
        swapped = float(np.sum((sequence[index, ::-1] - sequence[index - 1]) ** 2))
        if swapped < same:
            sequence[index] = sequence[index, ::-1]
    return sequence, np.asarray(anisotropy), np.asarray(spans)


def candidate_paths(audit_unit: dict[str, Any]) -> list[str]:
    candidates = audit_unit.get("ranked_candidates") or []
    allowed = {".npy", ".npz", ".h5", ".hdf5", ".ply", ".pcd", ".obj"}
    return [entry["path"] for entry in candidates if entry.get("suffix") in allowed]


def select_unit(root: Path, audit_unit: dict[str, Any], args: argparse.Namespace) -> tuple[SequenceUnit | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    paths = candidate_paths(audit_unit)
    cloud_groups: dict[str, list[Path]] = defaultdict(list)
    for relative in paths:
        path = root / relative
        if path.suffix.casefold() in {".ply", ".pcd", ".obj"}:
            cloud_groups[str(path.parent)].append(path)
            continue
        if not path.is_file() or path.is_symlink():
            continue
        for member, raw in numeric_arrays(path):
            sequence = trajectory_orientation(raw, args.min_frames, args.min_points)
            if sequence is None:
                attempts.append({"carrier": f"{relative}::{member}", "decision": "shape-unsupported", "shape": list(raw.shape)})
                continue
            result = endpoint_sequence((np.asarray(frame) for frame in sequence[: args.max_frames]), args.tail_fraction, args.min_anisotropy)
            attempts.append({"carrier": f"{relative}::{member}", "decision": "accepted" if result else "geometry-unsupported", "shape": list(sequence.shape)})
            if result is not None:
                endpoints, anisotropy, span = result
                return SequenceUnit(
                    object_id=audit_unit["object_id"], role=audit_unit["role"], episode=int(audit_unit["episode"]),
                    carrier=f"{relative}::{member}", endpoints=endpoints, anisotropy=anisotropy, span=span,
                ), attempts
    for parent, members in sorted(cloud_groups.items()):
        ordered = sorted(set(members), key=lambda path: natural_key(path.name))[: args.max_frames]
        if len(ordered) < args.min_frames:
            continue
        result = endpoint_sequence((cloud for cloud in (read_cloud(path) for path in ordered) if cloud is not None), args.tail_fraction, args.min_anisotropy)
        attempts.append({"carrier": f"{Path(parent).relative_to(root).as_posix()}::cloud-sequence", "decision": "accepted" if result else "geometry-unsupported", "files": len(ordered)})
        if result is not None:
            endpoints, anisotropy, span = result
            return SequenceUnit(
                object_id=audit_unit["object_id"], role=audit_unit["role"], episode=int(audit_unit["episode"]),
                carrier=f"{Path(parent).relative_to(root).as_posix()}::cloud-sequence", endpoints=endpoints,
                anisotropy=anisotropy, span=span,
            ), attempts
    return None, attempts


def fit_alpha(units: Sequence[SequenceUnit], horizons: Sequence[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for horizon in horizons:
        numerator = 0.0; denominator = 0.0
        for unit in units:
            x = unit.endpoints
            for frame in range(1, len(x) - horizon):
                velocity = horizon * (x[frame] - x[frame - 1])
                displacement = x[frame + horizon] - x[frame]
                numerator += float(np.sum(velocity * displacement))
                denominator += float(np.sum(velocity * velocity))
        value = numerator / denominator if denominator > 1e-15 else 0.0
        result[horizon] = float(np.clip(value, 0.0, 1.5))
    return result


def query_value(current: np.ndarray, future: np.ndarray, axis: np.ndarray, query: str, gauge: int) -> float:
    if query == "span_change":
        return float(np.linalg.norm(future[1] - future[0]) - np.linalg.norm(current[1] - current[0]))
    if query == "centroid_axis_progress":
        return float((np.mean(future, axis=0) - np.mean(current, axis=0)) @ axis)
    if query == "named_endpoint_axis_progress":
        return float((future[gauge] - current[gauge]) @ axis)
    raise ValueError(query)


def make_windows(unit: SequenceUnit, alpha: dict[int, float], horizons: Sequence[int]) -> list[Window]:
    rows: list[Window] = []
    x = unit.endpoints
    for horizon in horizons:
        for frame in range(1, len(x) - horizon):
            current = x[frame]
            previous = x[frame - 1]
            truth = x[frame + horizon]
            candidate_points = current + alpha[horizon] * horizon * (current - previous)
            fallback_points = current
            span = max(float(np.linalg.norm(current[1] - current[0])), 1e-9)
            axis = canonical_axis(current[1] - current[0])
            for query in QUERIES:
                candidate_orbit = [query_value(current, candidate_points, axis, query, gauge) / span for gauge in (0, 1)]
                orbit_diameter = abs(candidate_orbit[0] - candidate_orbit[1])
                for gauge in (0, 1):
                    actual = query_value(current, truth, axis, query, gauge) / span
                    candidate = candidate_orbit[gauge]
                    fallback = query_value(current, fallback_points, axis, query, gauge) / span
                    candidate_error = abs(candidate - actual)
                    fallback_error = abs(fallback - actual)
                    rows.append(Window(
                        object_id=unit.object_id, role=unit.role, query=query, horizon=horizon, frame=frame,
                        gauge=gauge, actual=actual, candidate=candidate, fallback=fallback,
                        candidate_error=candidate_error, fallback_error=fallback_error,
                        advantage=fallback_error - candidate_error, orbit_diameter=orbit_diameter,
                    ))
    return rows


def conservative_quantile(values: Sequence[float], probability: float) -> float:
    array = np.sort(np.asarray([value for value in values if np.isfinite(value)], dtype=float))
    if not len(array):
        return float("nan")
    index = int(math.ceil((len(array) + 1) * probability) - 1)
    return float(array[min(max(index, 0), len(array) - 1)])


def object_aggregate(rows: Sequence[Window]) -> dict[tuple[str, int, int], dict[str, list[float]]]:
    per_object: dict[tuple[str, int, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row.query, row.horizon, row.gauge, row.object_id)
        per_object[key]["candidate"].append(row.candidate_error)
        per_object[key]["fallback"].append(row.fallback_error)
        per_object[key]["advantage"].append(row.advantage)
    result: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (query, horizon, gauge, _object_id), values in per_object.items():
        key = (query, horizon, gauge)
        result[key]["candidate"].append(float(np.median(values["candidate"])))
        result[key]["fallback"].append(float(np.median(values["fallback"])))
        result[key]["advantage"].append(conservative_quantile(values["advantage"], 0.10))
    return result


def calibration_model(rows: Sequence[Window]) -> dict[tuple[str, int, int], dict[str, float]]:
    grouped = object_aggregate(rows)
    model: dict[tuple[str, int, int], dict[str, float]] = {}
    for key, values in grouped.items():
        model[key] = {
            "groups": float(len(values["advantage"])),
            "shared_lcb": conservative_quantile(values["advantage"], 0.10),
            "independent_lcb": conservative_quantile(values["fallback"], 0.10) - conservative_quantile(values["candidate"], 0.90),
            "candidate_q90": conservative_quantile(values["candidate"], 0.90),
            "fallback_q10": conservative_quantile(values["fallback"], 0.10),
        }
    return model


def accepts(arm: str, query: str, horizon: int, model: dict[tuple[str, int, int], dict[str, float]], orbit_diameter: float, margin: float, orbit_tolerance: float) -> bool:
    if arm == "fallback":
        return False
    if arm == "ungated_candidate":
        return True
    records = [model.get((query, horizon, gauge), {}) for gauge in (0, 1)]
    if arm == "local_canonical":
        value = records[0].get("shared_lcb", float("nan"))
        return bool(np.isfinite(value) and value > margin)
    if orbit_diameter > orbit_tolerance:
        return False
    field = "independent_lcb" if arm == "independent_orbit" else "shared_lcb"
    values = [record.get(field, float("nan")) for record in records]
    return bool(all(np.isfinite(value) for value in values) and min(values) > margin)


def split_fold(object_ids: Sequence[str], held_out: str) -> tuple[list[str], list[str]]:
    others = [value for value in object_ids if value != held_out]
    ordered = sorted(others, key=lambda value: sha256_bytes(f"{PROTOCOL_ID}:{held_out}:{value}".encode()))
    calibration_count = min(2, max(1, len(ordered) // 3))
    return ordered[calibration_count:], ordered[:calibration_count]


def bootstrap_ci(values: Sequence[float], seed: int, repetitions: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        samples[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate(units: Sequence[SequenceUnit], horizons: Sequence[int], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_object = {unit.object_id: unit for unit in units if unit.role == "source"}
    target_by_object = {unit.object_id: unit for unit in units if unit.role == "target"}
    object_ids = sorted(set(source_by_object) & set(target_by_object))
    decisions: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    for held_out in object_ids:
        fit_ids, calibration_ids = split_fold(object_ids, held_out)
        alpha = fit_alpha([source_by_object[value] for value in fit_ids], horizons)
        calibration_rows = list(itertools.chain.from_iterable(make_windows(source_by_object[value], alpha, horizons) for value in calibration_ids))
        model = calibration_model(calibration_rows)
        for (query, horizon, gauge), values in sorted(model.items()):
            calibration_records.append({"held_out_object": held_out, "query": query, "horizon": horizon, "gauge": gauge, **values})
        target_rows = make_windows(target_by_object[held_out], alpha, horizons)
        paired: dict[tuple[str, int, int], list[Window]] = defaultdict(list)
        for row in target_rows:
            paired[(row.query, row.horizon, row.frame)].append(row)
        for (query, horizon, frame), gauge_rows in sorted(paired.items()):
            if len(gauge_rows) != 2:
                continue
            candidate_error = max(row.candidate_error for row in gauge_rows)
            fallback_error = max(row.fallback_error for row in gauge_rows)
            diameter = gauge_rows[0].orbit_diameter
            for arm in ARMS:
                accepted = accepts(arm, query, horizon, model, diameter, args.advantage_margin, args.orbit_tolerance)
                deployed_error = candidate_error if accepted else fallback_error
                regret = deployed_error - fallback_error
                decisions.append({
                    "object_id": held_out, "query": query, "horizon": horizon, "frame": frame, "arm": arm,
                    "accepted": int(accepted), "deployed_error": deployed_error,
                    "candidate_error": candidate_error, "fallback_error": fallback_error,
                    "regret_vs_fallback": regret,
                    "harmful_accepted": int(accepted and regret > args.harm_margin),
                    "orbit_diameter": diameter,
                    "fit_objects": ";".join(fit_ids), "calibration_objects": ";".join(calibration_ids),
                })
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(row["query"], row["horizon"], row["arm"])].append(row)
    for (query, horizon, arm), rows in sorted(grouped.items()):
        by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_object[row["object_id"]].append(row)
        object_errors = [float(np.mean([row["deployed_error"] for row in values])) for values in by_object.values()]
        object_regrets = [float(np.mean([row["regret_vs_fallback"] for row in values])) for values in by_object.values()]
        object_acceptance = [float(np.mean([row["accepted"] for row in values])) for values in by_object.values()]
        accepted_rows = [row for row in rows if row["accepted"]]
        ci_low, ci_high = bootstrap_ci(object_regrets, args.seed + horizon + sum(map(ord, query + arm)), args.bootstrap_repetitions)
        summaries.append({
            "query": query, "horizon": horizon, "arm": arm, "objects": len(by_object), "windows": len(rows),
            "acceptance_rate": float(np.mean(object_acceptance)),
            "mean_deployed_error": float(np.mean(object_errors)),
            "mean_regret_vs_fallback": float(np.mean(object_regrets)),
            "regret_ci95_low": ci_low, "regret_ci95_high": ci_high,
            "harmful_accepted_rate": float(np.mean([row["harmful_accepted"] for row in accepted_rows])) if accepted_rows else 0.0,
            "worst_object_regret": max(object_regrets),
        })
    return decisions, summaries, calibration_records


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Deform360 finite-orbit real-data pilot v1", "",
        f"Decision: **{summary['decision']}**", "",
        f"Supported source/target objects: **{summary['supported_object_pairs']}**", "",
        "| Query | H | Arm | Acceptance | Error | Regret | 95% CI | Harmful accepted | Worst-object regret |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("target_results", []):
        lines.append(
            f"| {row['query']} | {row['horizon']} | {row['arm']} | {row['acceptance_rate']:.3f} | "
            f"{row['mean_deployed_error']:.6f} | {row['mean_regret_vs_fallback']:.6f} | "
            f"[{row['regret_ci95_low']:.6f}, {row['regret_ci95_high']:.6f}] | "
            f"{row['harmful_accepted_rate']:.3f} | {row['worst_object_regret']:.6f} |"
        )
    lines.extend([
        "", "Errors are normalized by current principal-axis endpoint span. Complete objects are the resampling units; frames are repeated measurements.", "",
        "This is a retrospective public-real-data development pilot. It does not authorize a fresh-confirmation, learned-provider, BayesianPhysTwin, Causal4D, safety, or state-of-the-art claim.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--min-anisotropy", type=float, default=1.10)
    parser.add_argument("--tail-fraction", type=float, default=0.05)
    parser.add_argument("--orbit-tolerance", type=float, default=0.05)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--harm-margin", type=float, default=0.01)
    parser.add_argument("--minimum-object-pairs", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        root = args.dataset_root
        if not root.is_dir() or root.is_symlink():
            raise ValueError("dataset root must be a readable non-symlink directory")
        audit_raw = args.audit.read_bytes()
        protocol_raw = args.protocol.read_bytes()
        audit = json.loads(audit_raw)
        protocol = json.loads(protocol_raw)
        if protocol.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("protocol identity mismatch")
        if audit.get("schema") != "prob4d.deform360-query-carrier-audit":
            raise ValueError("audit schema mismatch")
        if audit["access_boundary"].get("target_scored") is not False:
            raise ValueError("audit is not target closed")

        units: list[SequenceUnit] = []
        selection_rows: list[dict[str, Any]] = []
        discovery: dict[str, Any] = {}
        for audit_unit in audit["units"]:
            unit, attempts = select_unit(root, audit_unit, args)
            key = f"{audit_unit['object_id']}#{audit_unit['role']}#{audit_unit['episode']}"
            discovery[key] = attempts
            if unit is not None:
                units.append(unit)
                selection_rows.append({
                    "object_id": unit.object_id, "role": unit.role, "episode": unit.episode,
                    "carrier": unit.carrier, "frames": len(unit.endpoints),
                    "median_anisotropy": float(np.median(unit.anisotropy)), "median_span": float(np.median(unit.span)),
                })
        write_csv(args.output_dir / "representation_selection.csv", selection_rows)
        (args.output_dir / "representation_discovery.json").write_text(json.dumps(discovery, indent=2, sort_keys=True) + "\n")

        source_ids = {unit.object_id for unit in units if unit.role == "source"}
        target_ids = {unit.object_id for unit in units if unit.role == "target"}
        paired_ids = sorted(source_ids & target_ids)
        if len(paired_ids) < args.minimum_object_pairs:
            decision = "support-negative"
            decisions: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            calibration: list[dict[str, Any]] = []
            support_reasons = [f"only {len(paired_ids)} complete source/target object pairs; require {args.minimum_object_pairs}"]
        else:
            selected = [unit for unit in units if unit.object_id in paired_ids]
            decisions, results, calibration = evaluate(selected, args.horizons, args)
            decision = "completed-retrospective-real-data-pilot"
            support_reasons = []
        write_csv(args.output_dir / "target_window_decisions.csv", decisions)
        write_csv(args.output_dir / "target_summary.csv", results)
        write_csv(args.output_dir / "calibration_models.csv", calibration)

        prediction_projection = [
            {key: row[key] for key in ("object_id", "query", "horizon", "frame", "candidate_error", "fallback_error", "orbit_diameter")}
            for row in decisions
        ]
        prediction_seal = {
            "schema": "prob4d.deform360-retrospective-prediction-seal",
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "seal_scope": "retrospective-computation-order-only",
            "prediction_projection_sha256": sha256_bytes(canonical_bytes(prediction_projection)),
            "fresh_confirmation_authorized": False,
        }
        prediction_seal["seal_id"] = sha256_bytes(canonical_bytes(prediction_seal))
        (args.output_dir / "prediction_seal.json").write_text(json.dumps(prediction_seal, indent=2, sort_keys=True) + "\n")

        summary = {
            "schema": "prob4d.deform360-finite-orbit-real-data-result",
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "decision": decision,
            "paper_claim_authorized": False,
            "fresh_confirmation_authorized": False,
            "dataset_root": str(root),
            "dataset_audit_sha256": sha256_bytes(audit_raw),
            "protocol_sha256": sha256_bytes(protocol_raw),
            "github_sha": os.environ.get("GITHUB_SHA", "unknown"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "runner_name": os.environ.get("RUNNER_NAME", "unknown"),
            "supported_sequences": len(units),
            "supported_object_pairs": len(paired_ids),
            "paired_object_ids": paired_ids,
            "support_reasons": support_reasons,
            "parameters": {
                "horizons": args.horizons, "min_anisotropy": args.min_anisotropy,
                "tail_fraction": args.tail_fraction, "orbit_tolerance": args.orbit_tolerance,
                "advantage_margin": args.advantage_margin, "harm_margin": args.harm_margin,
                "bootstrap_repetitions": args.bootstrap_repetitions, "seed": args.seed,
            },
            "target_results": results,
            "claim_boundary": (
                "Retrospective real measured Deform360 geometry evidence for a principal-axis C2 finite-orbit query-admission mechanism only; "
                "not fresh confirmation, learned-provider competence, BayesianPhysTwin or Causal4D benefit, deployment safety, or state of the art."
            ),
        }
        summary["result_id"] = sha256_bytes(canonical_bytes(summary))
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.output_dir / "REPORT.md").write_text(report(summary), encoding="utf-8")
        print(json.dumps({"decision": decision, "result_id": summary["result_id"], "object_pairs": len(paired_ids)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "prob4d.deform360-finite-orbit-real-data-result", "schema_version": 1,
            "protocol_id": PROTOCOL_ID, "decision": "technical-negative", "paper_claim_authorized": False,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        }
        failure["result_id"] = sha256_bytes(canonical_bytes(failure))
        (args.output_dir / "summary.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        (args.output_dir / "REPORT.md").write_text(f"# Deform360 finite-orbit real-data pilot v1\n\nDecision: **technical-negative**\n\n`{failure['error']}`\n")
        print(json.dumps(failure, indent=2), file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
