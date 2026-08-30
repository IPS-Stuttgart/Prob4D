"""PokeFlex real-geometry diagnostic for posterior-preserving compression.

The local mirror is explicitly known to be incomplete. This script inventories
all ZIP central directories, deterministically selects a bounded subset with a
supported geometry sequence, and evaluates the Prob4D compression theorem on a
local Gaussian model constructed from real material-point trajectories.

It is not a PointWorld/provider evaluation and does not reopen the earlier
Causal4D PokeFlex target. No archive is extracted persistently, and no pickle or
other executable dataset payload is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import socket
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

from prob4d.posterior_preserving_compression import (
    PosteriorPreservingCompression,
    compress_shared_factor_for_posterior,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    kind: str
    key: str
    members: tuple[str, ...]
    score: tuple[int, int, str]


class DiagonalLowRankOperator:
    """Small controlled-study operator for sigma^2 I + Z Z.T."""

    def __init__(self, variance: float, factor: np.ndarray) -> None:
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("variance must be finite and positive")
        z = np.asarray(factor, dtype=np.float64)
        if z.ndim != 3 or z.shape[0] < 1 or z.shape[1] != 3:
            raise ValueError("factor must have shape (N, 3, R)")
        if not np.all(np.isfinite(z)):
            raise ValueError("factor must be finite")
        self.observation_count = int(z.shape[0])
        self.dimension = 3 * self.observation_count
        self._variance = float(variance)
        self._factor = z.reshape(self.dimension, z.shape[2]).copy()
        scaled = self._factor / math.sqrt(self._variance)
        core = np.eye(z.shape[2], dtype=np.float64) + scaled.T @ scaled
        self._core = np.linalg.cholesky(0.5 * (core + core.T))

    def solve(self, value: object) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if raw.ndim not in (2, 3) or raw.shape[:2] != (
            self.observation_count,
            3,
        ):
            raise ValueError("value must have shape (N, 3) or (N, 3, K)")
        if not np.all(np.isfinite(raw)):
            raise ValueError("value must be finite")
        matrix = raw.reshape(self.dimension, -1)
        base = matrix / self._variance
        rhs = self._factor.T @ base
        correction = self._factor @ np.linalg.solve(
            self._core.T,
            np.linalg.solve(self._core, rhs),
        ) / self._variance
        result = (base - correction).reshape(raw.shape)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("structured solve returned nonfinite values")
        return result


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _central_digest(infos: Iterable[zipfile.ZipInfo]) -> str:
    digest = hashlib.sha256()
    for info in sorted(infos, key=lambda item: item.filename):
        digest.update(info.filename.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(info.file_size).encode())
        digest.update(b"\0")
        digest.update(str(info.CRC).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _candidate_keyword_score(value: str) -> int:
    lowered = value.lower()
    terms = (
        "mesh",
        "vert",
        "particle",
        "track",
        "point",
        "cloud",
        "object",
        "deform",
    )
    return sum(term in lowered for term in terms)


def _normalized_stem(path: PurePosixPath) -> str:
    stem = re.sub(r"\d+", "#", path.stem.lower())
    return re.sub(r"(?:frame|time|step|capture|mesh)[_-]*#", "#", stem)


def discover_candidates(
    infos: Iterable[zipfile.ZipInfo],
    *,
    supported_suffixes: set[str],
    minimum_frames: int,
    maximum_member_bytes: int,
) -> list[Candidate]:
    sequence_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    array_members: list[str] = []
    for info in infos:
        if (
            info.is_dir()
            or info.flag_bits & 0x1
            or info.file_size > maximum_member_bytes
        ):
            continue
        path = PurePosixPath(info.filename)
        suffix = path.suffix.lower()
        if suffix not in supported_suffixes:
            continue
        if suffix in {".npy", ".npz"}:
            array_members.append(info.filename)
        else:
            sequence_groups[
                (str(path.parent), suffix, _normalized_stem(path))
            ].append(info.filename)
    candidates: list[Candidate] = []
    for member in array_members:
        candidates.append(
            Candidate(
                kind="array",
                key=member,
                members=(member,),
                score=(-_candidate_keyword_score(member), -1, member),
            )
        )
    for (parent, suffix, stem), members in sequence_groups.items():
        if len(members) < minimum_frames:
            continue
        ordered = tuple(sorted(members, key=_natural_key))
        key = f"{parent}|{suffix}|{stem}"
        candidates.append(
            Candidate(
                kind="files",
                key=key,
                members=ordered,
                score=(-_candidate_keyword_score(key), -len(ordered), key),
            )
        )
    return sorted(candidates, key=lambda item: item.score)


def _finite_points(value: object, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError(f"{name} must contain at least four 3D points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains nonfinite coordinates")
    return points


def parse_obj(data: bytes, *, maximum_vertices: int) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith("v "):
            continue
        fields = raw_line.split()
        if len(fields) < 4:
            continue
        vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        if len(vertices) > maximum_vertices:
            raise ValueError("OBJ exceeds maximum_vertices_per_frame")
    return _finite_points(vertices, name="OBJ")


def _ply_scalar_dtype(name: str, endian: str) -> str:
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    if name not in mapping:
        raise ValueError(f"unsupported PLY scalar type {name!r}")
    code = mapping[name]
    return code if code.endswith("1") else endian + code


def parse_ply(data: bytes, *, maximum_vertices: int) -> np.ndarray:
    stream = io.BytesIO(data)
    if stream.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    format_name: str | None = None
    vertex_count: int | None = None
    in_vertices = False
    vertex_properties: list[tuple[str, str]] = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("truncated PLY header")
        fields = line.decode("ascii", errors="strict").strip().split()
        if not fields:
            continue
        if fields[0] == "format" and len(fields) >= 2:
            format_name = fields[1]
        elif fields[0] == "element" and len(fields) == 3:
            in_vertices = fields[1] == "vertex"
            if in_vertices:
                vertex_count = int(fields[2])
        elif fields[0] == "property" and in_vertices:
            if len(fields) != 3 or fields[1] == "list":
                raise ValueError("unsupported PLY vertex property")
            vertex_properties.append((fields[2], fields[1]))
        elif fields[0] == "end_header":
            break
    if (
        vertex_count is None
        or vertex_count < 4
        or vertex_count > maximum_vertices
    ):
        raise ValueError("invalid PLY vertex count")
    names = [name for name, _ in vertex_properties]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY has no x/y/z vertex properties")
    xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]
    if format_name == "ascii":
        rows: list[tuple[float, float, float]] = []
        for _ in range(vertex_count):
            fields = stream.readline().split()
            if len(fields) < len(vertex_properties):
                raise ValueError("truncated ASCII PLY vertices")
            rows.append(tuple(float(fields[index]) for index in xyz_indices))
        return _finite_points(rows, name="PLY")
    if format_name not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format {format_name!r}")
    endian = "<" if format_name == "binary_little_endian" else ">"
    dtype = np.dtype(
        [
            (name, _ply_scalar_dtype(type_name, endian))
            for name, type_name in vertex_properties
        ]
    )
    required = vertex_count * dtype.itemsize
    payload = stream.read(required)
    if len(payload) != required:
        raise ValueError("truncated binary PLY vertices")
    records = np.frombuffer(payload, dtype=dtype, count=vertex_count)
    points = np.column_stack([records[axis] for axis in ("x", "y", "z")])
    return _finite_points(points, name="PLY")


def parse_xyz_like(data: bytes, *, maximum_vertices: int) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        fields = re.split(r"[\s,;]+", raw_line.strip())
        if len(fields) < 3:
            continue
        try:
            vertex = (float(fields[0]), float(fields[1]), float(fields[2]))
        except ValueError:
            continue
        vertices.append(vertex)
        if len(vertices) > maximum_vertices:
            raise ValueError("point file exceeds maximum_vertices_per_frame")
    return _finite_points(vertices, name="point file")


def _trajectory_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a real numeric array")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have three dimensions")
    if array.shape[-1] == 3:
        trajectory = array
    elif array.shape[1] == 3:
        trajectory = np.transpose(array, (0, 2, 1))
    else:
        raise ValueError(f"{name} must have shape (T,N,3) or (T,3,N)")
    if trajectory.shape[0] < 2 or trajectory.shape[1] < 4:
        raise ValueError(f"{name} contains too few frames or points")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError(f"{name} contains nonfinite coordinates")
    return trajectory


def load_array_trajectory(data: bytes, suffix: str) -> np.ndarray:
    source = io.BytesIO(data)
    if suffix == ".npy":
        return _trajectory_array(np.load(source, allow_pickle=False), name="NPY")
    with np.load(source, allow_pickle=False) as archive:
        candidates: list[tuple[int, str, np.ndarray]] = []
        for name in archive.files:
            try:
                value = _trajectory_array(archive[name], name=f"NPZ[{name}]")
            except (TypeError, ValueError):
                continue
            candidates.append((-int(value.size), name, value))
        if not candidates:
            raise ValueError("NPZ contains no supported trajectory array")
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]


def _read_member(
    archive: zipfile.ZipFile,
    member: str,
    *,
    maximum_bytes: int,
) -> bytes:
    info = archive.getinfo(member)
    if info.flag_bits & 0x1:
        raise ValueError("encrypted geometry member")
    if info.file_size > maximum_bytes:
        raise ValueError("geometry member exceeds maximum_geometry_member_bytes")
    data = archive.read(info)
    if len(data) != info.file_size:
        raise ValueError("geometry member length mismatch")
    return data


def load_candidate_trajectory(
    archive: zipfile.ZipFile,
    candidate: Candidate,
    *,
    maximum_frames: int,
    maximum_member_bytes: int,
    maximum_vertices: int,
) -> np.ndarray:
    if candidate.kind == "array":
        member = candidate.members[0]
        suffix = PurePosixPath(member).suffix.lower()
        trajectory = load_array_trajectory(
            _read_member(
                archive,
                member,
                maximum_bytes=maximum_member_bytes,
            ),
            suffix,
        )
        selected = np.unique(
            np.linspace(
                0,
                trajectory.shape[0] - 1,
                min(maximum_frames, trajectory.shape[0]),
            ).astype(int)
        )
        return trajectory[selected]
    selected_positions = np.unique(
        np.linspace(
            0,
            len(candidate.members) - 1,
            min(maximum_frames, len(candidate.members)),
        ).astype(int)
    )
    frames: list[np.ndarray] = []
    for position in selected_positions:
        member = candidate.members[int(position)]
        suffix = PurePosixPath(member).suffix.lower()
        data = _read_member(
            archive,
            member,
            maximum_bytes=maximum_member_bytes,
        )
        if suffix == ".obj":
            frame = parse_obj(data, maximum_vertices=maximum_vertices)
        elif suffix == ".ply":
            frame = parse_ply(data, maximum_vertices=maximum_vertices)
        elif suffix in {".xyz", ".pts"}:
            frame = parse_xyz_like(data, maximum_vertices=maximum_vertices)
        else:
            raise ValueError(f"unsupported frame suffix {suffix!r}")
        frames.append(frame)
    counts = {frame.shape[0] for frame in frames}
    if len(counts) != 1:
        raise ValueError(
            "frame vertex counts differ; material identity is not assumed"
        )
    return np.stack(frames)


def _skew(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(
        np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    angle = math.acos(cosine)
    skew_vector = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-10:
        return 0.5 * skew_vector
    if abs(math.pi - angle) < 1e-6:
        values, vectors = np.linalg.eigh(0.5 * (rotation + np.eye(3)))
        axis = vectors[:, int(np.argmax(values))]
        axis /= np.linalg.norm(axis)
        return angle * axis
    return angle * skew_vector / (2.0 * math.sin(angle))


def _sim3_increment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(left @ right_t)) or 1.0
    rotation = left @ np.diag(sign) @ right_t
    variance = float(np.sum(source_centered**2) / source.shape[0])
    if variance <= np.finfo(float).eps:
        raise ValueError("degenerate point geometry")
    scale = float(np.sum(singular * sign) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid fitted Sim(3) scale")
    translation = target_mean - scale * rotation @ source_mean
    return np.concatenate(
        ([math.log(scale)], _rotation_vector(rotation), translation)
    )


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference, ord="fro"))
    numerator = float(np.linalg.norm(actual - reference, ord="fro"))
    return numerator / denominator if denominator else numerator


def _posterior(
    *,
    prior: np.ndarray,
    cross: np.ndarray,
    innovation: np.ndarray,
    conditional_variance: float,
    physical_factor: np.ndarray,
    shared_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DiagonalLowRankOperator]:
    combined = np.concatenate((physical_factor, shared_factor), axis=2)
    operator = DiagonalLowRankOperator(conditional_variance, combined)
    solved_cross = operator.solve(
        cross.T.reshape(operator.observation_count, 3, prior.shape[0])
    ).reshape(operator.dimension, prior.shape[0])
    gain = solved_cross.T
    raw_covariance = prior - cross @ solved_cross
    covariance = 0.5 * (raw_covariance + raw_covariance.T)
    np.linalg.cholesky(covariance)
    mean = gain @ innovation.reshape(-1)
    return gain, covariance, mean, operator


def _query_score(
    truth: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    root = np.linalg.cholesky(covariance)
    error = truth - mean
    white = np.linalg.solve(root, error)
    nees = float(white @ white)
    logdet = 2.0 * float(np.sum(np.log(np.diag(root))))
    nll = 0.5 * (
        nees + logdet + truth.size * math.log(2.0 * math.pi)
    )
    return {
        "euclidean_error_normalized_geometry": float(np.linalg.norm(error)),
        "normalized_nees": nees / truth.size,
        "gaussian_nll_nats": nll,
    }


def evaluate_trajectory(
    trajectory: np.ndarray,
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape (T,N,3)")
    minimum_frames = int(protocol["minimum_sequence_frames"])
    if trajectory.shape[0] < minimum_frames:
        raise ValueError(
            "trajectory contains fewer than minimum_sequence_frames"
        )
    point_count = min(
        int(protocol["sampled_material_points"]),
        trajectory.shape[1],
    )
    indices = np.unique(
        np.linspace(0, trajectory.shape[1] - 1, point_count).astype(int)
    )
    points = np.asarray(trajectory[:, indices], dtype=np.float64)
    extent = np.ptp(points[0], axis=0)
    scale = float(np.linalg.norm(extent))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("first-frame geometry has zero extent")
    normalized = (points - points[0].mean(axis=0)) / scale
    available_prefix = normalized.shape[0] - 1
    window_count = min(int(protocol["maximum_windows"]), available_prefix)
    observation_positions = np.unique(
        np.linspace(0, available_prefix - 1, window_count).astype(int)
    )
    observations = normalized[observation_positions]
    future = normalized[-1]
    window_count = observations.shape[0]
    material_points = observations.shape[1]
    if window_count < 2:
        raise ValueError("at least two observation windows are required")

    increments = np.stack(
        [
            _sim3_increment(observations[index], observations[index + 1])
            for index in range(window_count - 1)
        ]
    )
    if increments.shape[0] > 1:
        spread = np.std(increments, axis=0, ddof=1)
    else:
        spread = np.abs(increments[0])
    typical = np.median(np.abs(increments), axis=0)
    gauge_sigma = np.maximum(spread, typical)
    gauge_sigma = np.clip(
        gauge_sigma,
        float(protocol["gauge_standard_deviation_floor"]),
        float(protocol["gauge_standard_deviation_ceiling"]),
    )
    gauge_root = np.kron(
        np.tril(np.ones((window_count, window_count))),
        np.diag(gauge_sigma),
    )
    gauge_jacobian = np.zeros(
        (3 * material_points * window_count, 7 * window_count)
    )
    for window, frame in enumerate(observations):
        for point_index, point in enumerate(frame):
            row = 3 * (window * material_points + point_index)
            column = 7 * window
            gauge_jacobian[
                row : row + 3,
                column : column + 7,
            ] = np.column_stack((point, -_skew(point), np.eye(3)))
    shared = gauge_jacobian @ gauge_root

    displacement = observations - observations[0]
    displacement_matrix = displacement.reshape(
        window_count,
        3 * material_points,
    )
    temporal, singular, spatial = np.linalg.svd(
        displacement_matrix,
        full_matrices=False,
    )
    if singular.size and singular[0]:
        nonzero = int(np.count_nonzero(singular > 1e-12 * singular[0]))
    else:
        nonzero = 0
    mode_count = min(int(protocol["maximum_spatial_modes"]), nonzero)
    columns: list[np.ndarray] = []
    query_columns: list[np.ndarray] = []
    time = np.linspace(0.0, 1.0, window_count)
    patch_size = max(4, material_points // 8)
    principal = observations[0] - observations[0].mean(axis=0)
    axis = np.linalg.svd(principal, full_matrices=False)[2][0]
    patch = np.argsort(principal @ axis)[-patch_size:]
    for mode in range(mode_count):
        coefficients = temporal[:, mode] * singular[mode]
        mode_xyz = spatial[mode].reshape(material_points, 3)
        columns.append(
            np.concatenate(
                [coefficient * mode_xyz for coefficient in coefficients]
            )
        )
        degree = min(2, window_count - 1)
        future_coefficient = float(
            np.polyval(
                np.polyfit(time, coefficients, degree),
                1.0 + 1.0 / window_count,
            )
        )
        query_columns.append(
            future_coefficient * mode_xyz[patch].mean(axis=0)
        )
    observed_step = np.linalg.norm(
        np.diff(observations, axis=0),
        axis=2,
    )
    prior_scale = max(float(np.median(observed_step)), 1e-3)
    for axis_index in range(3):
        axis_vector = np.eye(3)[axis_index]
        columns.append(
            np.concatenate(
                [
                    prior_scale
                    * time_value
                    * np.tile(axis_vector, (material_points, 1)).reshape(-1)
                    for time_value in time
                ]
            )
        )
        query_columns.append(prior_scale * axis_vector)
    physical = np.column_stack(columns)
    query_factor = np.column_stack(query_columns)
    if mode_count:
        reconstructed = physical[:, :mode_count] @ np.ones(mode_count)
    else:
        reconstructed = np.zeros(physical.shape[0])
    residual = displacement_matrix.reshape(-1) - reconstructed
    conditional_std = float(np.sqrt(np.mean(residual**2)))
    conditional_std = float(
        np.clip(
            conditional_std,
            protocol["conditional_noise_floor_fraction"],
            protocol["conditional_noise_ceiling_fraction"],
        )
    )
    conditional_variance = conditional_std**2
    nugget = (
        float(protocol["query_nugget_fraction"]) * prior_scale
    ) ** 2
    prior = query_factor @ query_factor.T + nugget * np.eye(3)
    cross = query_factor @ physical.T
    innovation = displacement_matrix.reshape(-1)
    truth = (
        future[patch] - observations[0, patch]
    ).mean(axis=0)
    observation_count = window_count * material_points
    physical_m = physical.reshape(observation_count, 3, -1)
    shared_m = shared.reshape(observation_count, 3, -1)

    full_gain, full_covariance, full_mean, full_operator = _posterior(
        prior=prior,
        cross=cross,
        innovation=innovation,
        conditional_variance=conditional_variance,
        physical_factor=physical_m,
        shared_factor=shared_m,
    )
    compression: PosteriorPreservingCompression = (
        compress_shared_factor_for_posterior(
            shared_m,
            prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=full_operator,
            maximum_rank=3,
            rank_relative_tolerance=float(
                protocol["rank_relative_tolerance"]
            ),
            parity_relative_tolerance=float(
                protocol["parity_relative_tolerance"]
            ),
        )
    )
    reduced_gain, reduced_covariance, reduced_mean, _ = _posterior(
        prior=prior,
        cross=cross,
        innovation=innovation,
        conditional_variance=conditional_variance,
        physical_factor=physical_m,
        shared_factor=compression.compressed_factor_m,
    )
    retained = compression.retained_rank
    if retained:
        right = np.linalg.svd(shared, full_matrices=False)[2].T[:, :retained]
        pca_shared = (shared @ right).reshape(
            observation_count,
            3,
            retained,
        )
    else:
        pca_shared = np.empty((observation_count, 3, 0))
    pca_gain, pca_covariance, pca_mean, _ = _posterior(
        prior=prior,
        cross=cross,
        innovation=innovation,
        conditional_variance=conditional_variance,
        physical_factor=physical_m,
        shared_factor=pca_shared,
    )
    empty_shared = np.empty((observation_count, 3, 0))
    conditional_gain, conditional_covariance, conditional_mean, _ = _posterior(
        prior=prior,
        cross=cross,
        innovation=innovation,
        conditional_variance=conditional_variance,
        physical_factor=physical_m,
        shared_factor=empty_shared,
    )
    full_score = _query_score(truth, full_mean, full_covariance)
    reduced_score = _query_score(
        truth,
        reduced_mean,
        reduced_covariance,
    )
    pca_score = _query_score(truth, pca_mean, pca_covariance)
    conditional_score = _query_score(
        truth,
        conditional_mean,
        conditional_covariance,
    )
    compressed_bytes = compression.compressed_factor_m.nbytes
    return {
        "input_frames": int(trajectory.shape[0]),
        "input_vertices": int(trajectory.shape[1]),
        "used_windows": int(window_count),
        "used_material_points": int(material_points),
        "first_frame_extent_input_units": scale,
        "gauge_standard_deviations_normalized_geometry": gauge_sigma.tolist(),
        "conditional_standard_deviation_normalized_geometry": conditional_std,
        "physical_factor_rank": int(physical.shape[1]),
        "query_dimension": 3,
        "full_shared_rank": int(shared.shape[1]),
        "retained_shared_rank": int(retained),
        "numerical_required_rank": int(compression.numerical_required_rank),
        "exact_factor_fallback": bool(compression.exact_fallback),
        "compression_reason": compression.reason,
        "full_vs_compressed_relative_gain_error": _relative_error(
            reduced_gain,
            full_gain,
        ),
        "full_vs_compressed_relative_covariance_error": _relative_error(
            reduced_covariance,
            full_covariance,
        ),
        "full_vs_compressed_mean_difference": float(
            np.linalg.norm(reduced_mean - full_mean)
        ),
        "full_vs_equal_rank_pca_relative_gain_error": _relative_error(
            pca_gain,
            full_gain,
        ),
        "full_vs_equal_rank_pca_relative_covariance_error": _relative_error(
            pca_covariance,
            full_covariance,
        ),
        "full_vs_conditional_only_relative_gain_error": _relative_error(
            conditional_gain,
            full_gain,
        ),
        "full_vs_conditional_only_relative_covariance_error": _relative_error(
            conditional_covariance,
            full_covariance,
        ),
        "shared_factor_payload_bytes_full": int(shared_m.nbytes),
        "shared_factor_payload_bytes_compressed": int(compressed_bytes),
        "shared_factor_payload_reduction_ratio": (
            float(shared_m.nbytes / compressed_bytes)
            if compressed_bytes
            else math.inf
        ),
        "cached_query_message_payload_bytes": int(
            full_gain.nbytes + full_covariance.nbytes
        ),
        "truth_query_normalized_geometry": truth.tolist(),
        "posterior_mean_full_normalized_geometry": full_mean.tolist(),
        "posterior_mean_compressed_normalized_geometry": reduced_mean.tolist(),
        "scores": {
            "full": full_score,
            "posterior_preserving": reduced_score,
            "equal_rank_covariance_pca": pca_score,
            "conditional_only": conditional_score,
            "cached_full_query_message": full_score,
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run(
    dataset_root: Path,
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            "output directory already exists; runs are never overwritten"
        )
    output_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema": protocol["schema"],
        "evidence_class": protocol["evidence_class"],
        "dataset_name": protocol["dataset_name"],
        "dataset_state": protocol["dataset_state"],
        "dataset_root": str(dataset_root),
        "host": socket.gethostname(),
        "claim_boundary": protocol["claim_boundary"],
        "prior_causal4d_pokeflex_target_opened_or_repaired": False,
    }
    if not dataset_root.is_dir():
        result.update(status="dataset-root-missing", archive_inventory=[])
        return result
    archives = sorted(dataset_root.glob(str(protocol["archive_glob"])))
    suffix_histogram: Counter[str] = Counter()
    inventory: list[dict[str, Any]] = []
    selectable: list[tuple[str, Path, Candidate]] = []
    supported = set(protocol["supported_geometry_suffixes"])
    for archive_path in archives:
        relative = archive_path.relative_to(dataset_root).as_posix()
        record: dict[str, Any] = {
            "relative_path": relative,
            "size_bytes": archive_path.stat().st_size,
        }
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                for info in infos:
                    if not info.is_dir():
                        suffix = PurePosixPath(info.filename).suffix.lower()
                        suffix_histogram[suffix or "<none>"] += 1
                candidates = discover_candidates(
                    infos,
                    supported_suffixes=supported,
                    minimum_frames=int(
                        protocol["minimum_sequence_frames"]
                    ),
                    maximum_member_bytes=int(
                        protocol["maximum_geometry_member_bytes"]
                    ),
                )
                record.update(
                    central_directory_status="readable",
                    member_count=len(infos),
                    uncompressed_bytes=sum(info.file_size for info in infos),
                    central_directory_sha256=_central_digest(infos),
                    supported_candidate_count=len(candidates),
                    selected_candidate_key=(
                        candidates[0].key if candidates else None
                    ),
                )
                if candidates:
                    token = hashlib.sha256(
                        f"{relative}\0{candidates[0].key}".encode()
                    ).hexdigest()
                    selectable.append((token, archive_path, candidates[0]))
        except (
            OSError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
            UnicodeError,
            ValueError,
        ) as error:
            record.update(
                central_directory_status="unreadable",
                error_type=type(error).__name__,
                error_message=str(error),
                supported_candidate_count=0,
            )
        inventory.append(record)
    result["archive_inventory"] = inventory
    result["inventory_summary"] = {
        "observed_zip_count": len(archives),
        "observed_zip_bytes": sum(item["size_bytes"] for item in inventory),
        "readable_central_directories": sum(
            item.get("central_directory_status") == "readable"
            for item in inventory
        ),
        "unreadable_central_directories": sum(
            item.get("central_directory_status") == "unreadable"
            for item in inventory
        ),
        "archives_with_supported_candidate": len(selectable),
        "member_suffix_histogram": dict(
            sorted(
                suffix_histogram.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "release_completeness_asserted": False,
    }
    selected = sorted(selectable, key=lambda item: item[0])[
        : int(protocol["maximum_deep_archives"])
    ]
    evaluations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for selection_token, archive_path, candidate in selected:
        relative = archive_path.relative_to(dataset_root).as_posix()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                trajectory = load_candidate_trajectory(
                    archive,
                    candidate,
                    maximum_frames=int(
                        protocol["maximum_sequence_frames"]
                    ),
                    maximum_member_bytes=int(
                        protocol["maximum_geometry_member_bytes"]
                    ),
                    maximum_vertices=int(
                        protocol["maximum_vertices_per_frame"]
                    ),
                )
            evaluation = evaluate_trajectory(
                trajectory,
                protocol=protocol,
            )
            evaluation.update(
                archive_relative_path=relative,
                selection_sha256=selection_token,
                candidate_kind=candidate.kind,
                candidate_key=candidate.key,
                candidate_member_count=len(candidate.members),
            )
            evaluations.append(evaluation)
        except Exception as error:
            failures.append(
                {
                    "archive_relative_path": relative,
                    "selection_sha256": selection_token,
                    "candidate_kind": candidate.kind,
                    "candidate_key": candidate.key,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    result["evaluations"] = evaluations
    result["deep_read_failures"] = failures
    if not archives:
        result["status"] = "no-zip-archives"
    elif not selectable:
        result["status"] = "no-supported-geometry-candidates"
    elif not evaluations:
        result["status"] = (
            "supported-candidates-but-no-evaluable-trajectory"
        )
    else:
        result["status"] = "evaluated-real-geometry"
        result["summary"] = {
            "evaluated_archive_count": len(evaluations),
            "selected_deep_archive_count": len(selected),
            "deep_read_failure_count": len(failures),
            "maximum_full_vs_compressed_relative_gain_error": max(
                item["full_vs_compressed_relative_gain_error"]
                for item in evaluations
            ),
            "maximum_full_vs_compressed_relative_covariance_error": max(
                item["full_vs_compressed_relative_covariance_error"]
                for item in evaluations
            ),
            "maximum_full_vs_compressed_mean_difference": max(
                item["full_vs_compressed_mean_difference"]
                for item in evaluations
            ),
            "retained_ranks": [
                item["retained_shared_rank"] for item in evaluations
            ],
            "full_ranks": [item["full_shared_rank"] for item in evaluations],
            "median_payload_reduction_ratio": float(
                np.median(
                    [
                        item["shared_factor_payload_reduction_ratio"]
                        for item in evaluations
                    ]
                )
            ),
            "archive_independence_asserted": False,
            "provider_competence_asserted": False,
        }
    return _json_safe(result)


def _write_outputs(
    *,
    result: dict[str, Any],
    protocol_bytes: bytes,
    output_dir: Path,
    source_revision: str,
) -> None:
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    summary = result.get("summary", {})
    lines = [
        "# PokeFlex posterior-compression real-geometry diagnostic",
        "",
        f"- Decision: `{result['status']}`",
        (
            "- ZIP archives observed: "
            f"`{result.get('inventory_summary', {}).get('observed_zip_count', 0)}`"
        ),
        f"- Archives evaluated: `{summary.get('evaluated_archive_count', 0)}`",
        (
            "- Deep-read failures: "
            f"`{summary.get('deep_read_failure_count', len(result.get('deep_read_failures', [])))}`"
        ),
    ]
    if summary:
        lines.extend(
            [
                (
                    "- Maximum gain parity error: "
                    f"`{summary['maximum_full_vs_compressed_relative_gain_error']:.6e}`"
                ),
                (
                    "- Maximum covariance parity error: "
                    f"`{summary['maximum_full_vs_compressed_relative_covariance_error']:.6e}`"
                ),
                (
                    "- Median shared-factor payload ratio: "
                    f"`{summary['median_payload_reduction_ratio']:.3f}`"
                ),
            ]
        )
    lines.extend(["", result["claim_boundary"], ""])
    (output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    manifest = {
        "source_revision": source_revision,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "host": result["host"],
        "dataset_root": result["dataset_root"],
        "raw_dataset_files_copied": False,
        "pickle_loaded": False,
        "prior_causal4d_target_opened_or_repaired": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("GITHUB_SHA", "unknown"),
    )
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    expected_schema = (
        "prob4d.pokeflex-posterior-compression-real-geometry.v1"
    )
    if protocol.get("schema") != expected_schema:
        raise ValueError("unsupported protocol schema")
    result = run(args.dataset_root, protocol, args.output_dir)
    _write_outputs(
        result=result,
        protocol_bytes=protocol_bytes,
        output_dir=args.output_dir,
        source_revision=args.source_revision,
    )
    print(
        json.dumps(
            {"status": result["status"], **result.get("summary", {})},
            indent=2,
        )
    )
    return 0 if result["status"] == "evaluated-real-geometry" else 2


if __name__ == "__main__":
    raise SystemExit(main())
